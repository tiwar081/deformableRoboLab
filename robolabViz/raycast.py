"""Warp raycast preview renderer for the RoboLab-styled scene.

Machines without RT cores (H100/H200) cannot run Isaac Sim's RTX renderer, so
this module renders the *same cameras* (intrinsics + poses, including the
hand-mounted wrist camera) by casting rays against the actual scene geometry
with ``wp.Mesh`` BVH queries:

- robot links use the visual meshes of the converted robot USD,
- objects use the Newton shape geometry (cable capsules, boxes, meshes),
- the soft body deforms per frame from ``particle_q``,
- fixtures are approximated by their composed bounding boxes.

Output is flat-shaded but geometrically exact, which is what is needed to
frame cameras, tune the wrist mount against self-occlusion, and verify the
demo headlessly. It also reports per-category pixel coverage for the wrist
camera so occlusion is a number, not a vibe.
"""

from __future__ import annotations

import json
import os

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from newton import Axis
from newton import Mesh as NewtonMesh

from .config import CameraConfig, RoboLabSceneConfig, WristCameraConfig

# Required for cv2 to decode the .exr dome backgrounds (e.g. home_office.exr).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

CATEGORIES = ("background", "robot", "objects", "soft", "fixture")

# Square tile every fixture texture is resized to before stacking into the GPU
# atlas, and the equirectangular size the (tone-mapped) dome HDR is sampled at.
_TEX_TILE = 1024
_BG_H, _BG_W = 1024, 2048


def _load_texture(path: Path | str) -> np.ndarray:
    """Load an sRGB base-color image as an ``(H, W, 3)`` uint8 RGB array."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read texture {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (_TEX_TILE, _TEX_TILE), interpolation=cv2.INTER_AREA)


def _load_background(path: Path | str) -> np.ndarray:
    """Tone-map an HDR/EXR equirectangular environment map to an ``(H, W, 3)``
    uint8 sRGB image for use as the raycast preview backdrop.

    The dome HDR is linear and high-dynamic-range (bright windows/lamps reach
    thousands); we auto-expose so mean scene luminance lands at mid-gray, then
    apply Reinhard tonemapping + an sRGB gamma so the backdrop reads naturally
    without a single light source blowing out the frame."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read background {path}")
    lin = cv2.cvtColor(raw[:, :, :3].astype(np.float32), cv2.COLOR_BGR2RGB)
    lin = np.clip(lin, 0.0, None)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    mean_lum = float(np.mean(lum))
    gain = 0.45 / mean_lum if mean_lum > 1e-6 else 1.0
    mapped = lin * gain
    ldr = mapped / (1.0 + mapped)  # Reinhard
    srgb = np.power(np.clip(ldr, 0.0, 1.0), 1.0 / 2.2)
    u8 = (srgb * 255.0).astype(np.uint8)
    return cv2.resize(u8, (_BG_W, _BG_H), interpolation=cv2.INTER_AREA)


@dataclass
class _Instance:
    name: str
    category: int  # index into CATEGORIES
    kind: str  # "robot_link" | "object_body" | "static" | "soft"
    key: str | int | None  # link name or body index
    verts: np.ndarray  # (n,3) local-frame vertices (soft: placeholder)
    tris: np.ndarray  # (m,3)
    color: tuple[float, float, float]
    # Optional base-color texture (H,W,3 uint8 sRGB), planar-mapped onto this
    # instance in the kernel. ``uv_scale`` is world meters per texture tile.
    texture: np.ndarray | None = None
    uv_scale: float = 0.5


def _quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    if np.all(counts == 3):
        return indices.reshape(-1, 3)
    # vectorized fan triangulation: face f with count c yields c-2 triangles
    # (i0, i_k, i_{k+1}) for k in 1..c-2
    tris_per_face = counts - 2
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])  # first index slot per face
    tri_face = np.repeat(np.arange(len(counts)), tris_per_face)  # face id per output triangle
    group_starts = np.concatenate([[0], np.cumsum(tris_per_face)[:-1]])
    k = np.arange(int(tris_per_face.sum())) - np.repeat(group_starts, tris_per_face) + 1
    base = starts[tri_face]
    return np.stack([indices[base], indices[base + k], indices[base + k + 1]], axis=1).astype(np.int32)


def _box_mesh(extent_min, extent_max) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = np.asarray(extent_min, dtype=np.float32), np.asarray(extent_max, dtype=np.float32)
    verts = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], dtype=np.float32
    )
    tris = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # x faces
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # y faces
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # z faces
        ],
        dtype=np.int32,
    )
    return verts, tris


def _robot_link_meshes(robot_usd: Path, link_names: list[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Extract per-link visual meshes (in link-local frame) from the converted robot USD."""
    stage = Usd.Stage.Open(str(robot_usd), Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for link in link_names:
        link_prim = stage.GetPrimAtPath(root.GetPath().AppendChild(link))
        if not link_prim:
            continue
        link_world = np.array(cache.GetLocalToWorldTransform(link_prim), dtype=np.float64)
        link_world_inv = np.linalg.inv(link_world)
        meshes = []
        rng = Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
        for prim in rng:
            if prim.GetTypeName() != "Mesh":
                continue
            if "collisions" in str(prim.GetPath()):
                continue
            geom = UsdGeom.Mesh(prim)
            pts = np.asarray(geom.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(geom.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
            idx = np.asarray(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            if pts.size == 0 or counts.size == 0:
                continue
            m = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ link_world_inv
            pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
            local_pts = (pts_h @ m)[:, :3].astype(np.float32)  # row-vector convention
            meshes.append((local_pts, _triangulate(counts, idx)))
        if meshes:
            out[link] = meshes
    return out


@wp.kernel
def _transform_verts_kernel(
    rest: wp.array(dtype=wp.vec3),
    vert_inst: wp.array(dtype=wp.int32),
    inst_tf: wp.array(dtype=wp.transform),
    soft_start: int,
    soft_count: int,
    particle_q: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if soft_count > 0 and i >= soft_start and i < soft_start + soft_count:
        out[i] = particle_q[i - soft_start]
    else:
        out[i] = wp.transform_point(inst_tf[vert_inst[i]], rest[i])


@wp.func
def _sample(tex: wp.array(dtype=wp.uint8, ndim=4), ti: int, u: float, v: float, h: int, w: int) -> wp.vec3:
    # tile-wrap then nearest-sample the (sRGB uint8) texture at atlas index ``ti``.
    uu = u - wp.floor(u)
    vv = v - wp.floor(v)
    px = wp.clamp(int(uu * float(w)), 0, w - 1)
    py = wp.clamp(int(vv * float(h)), 0, h - 1)
    return wp.vec3(
        float(tex[ti, py, px, 0]) / 255.0,
        float(tex[ti, py, px, 1]) / 255.0,
        float(tex[ti, py, px, 2]) / 255.0,
    )


@wp.kernel
def _render_kernel(
    mesh_id: wp.uint64,
    cam_pos: wp.vec3,
    cam_quat: wp.quat,
    tan_half_w: float,
    tan_half_h: float,
    width: int,
    height: int,
    face_category: wp.array(dtype=wp.int32),
    face_color: wp.array(dtype=wp.vec3),
    face_tex: wp.array(dtype=wp.int32),
    face_uv: wp.array(dtype=wp.float32),
    tex_atlas: wp.array(dtype=wp.uint8, ndim=4),
    tex_h: int,
    tex_w: int,
    bg: wp.array(dtype=wp.uint8, ndim=3),
    bg_h: int,
    bg_w: int,
    has_bg: int,
    img: wp.array(dtype=wp.uint8, ndim=3),
    category_buf: wp.array(dtype=wp.int32, ndim=2),
):
    px, py = wp.tid()
    u = (float(px) + 0.5) / float(width) * 2.0 - 1.0
    v = 1.0 - (float(py) + 0.5) / float(height) * 2.0
    dir_cam = wp.normalize(wp.vec3(u * tan_half_w, v * tan_half_h, -1.0))
    d = wp.quat_rotate(cam_quat, dir_cam)

    query = wp.mesh_query_ray(mesh_id, cam_pos, d, 100.0)
    if query.result:
        n = query.normal
        if wp.dot(n, d) > 0.0:
            n = -n
        key_light = wp.normalize(wp.vec3(0.35, -0.45, -0.82))
        shade = 0.42 + 0.43 * wp.max(wp.dot(n, -key_light), 0.0) + 0.15 * wp.max(wp.dot(n, -d), 0.0)
        base = face_color[query.face]
        ti = face_tex[query.face]
        if ti >= 0:
            # planar (triplanar) projection by dominant normal axis, in world m.
            hit = cam_pos + d * query.t
            sc = face_uv[query.face]
            ax = wp.abs(n[0])
            ay = wp.abs(n[1])
            az = wp.abs(n[2])
            if az >= ax and az >= ay:
                base = _sample(tex_atlas, ti, hit[0] / sc, hit[1] / sc, tex_h, tex_w)
            elif ax >= ay:
                base = _sample(tex_atlas, ti, hit[1] / sc, hit[2] / sc, tex_h, tex_w)
            else:
                base = _sample(tex_atlas, ti, hit[0] / sc, hit[2] / sc, tex_h, tex_w)
        c = base * shade
        category_buf[py, px] = face_category[query.face]
    else:
        if has_bg == 1:
            # equirectangular latlong lookup (Z up), matching the dome light.
            lon = wp.atan2(d[1], d[0]) * (0.5 / 3.14159265) + 0.5
            lat = wp.acos(wp.clamp(d[2], -1.0, 1.0)) * (1.0 / 3.14159265)
            bx = wp.clamp(int(lon * float(bg_w)), 0, bg_w - 1)
            by = wp.clamp(int(lat * float(bg_h)), 0, bg_h - 1)
            c = wp.vec3(
                float(bg[by, bx, 0]) / 255.0,
                float(bg[by, bx, 1]) / 255.0,
                float(bg[by, bx, 2]) / 255.0,
            )
        else:
            # simple sky/floor gradient background
            g = 0.55 + 0.25 * wp.max(d[2], 0.0)
            c = wp.vec3(g * 0.92, g * 0.94, g)
        category_buf[py, px] = 0

    img[py, px, 0] = wp.uint8(wp.clamp(c[0], 0.0, 1.0) * 255.0)
    img[py, px, 1] = wp.uint8(wp.clamp(c[1], 0.0, 1.0) * 255.0)
    img[py, px, 2] = wp.uint8(wp.clamp(c[2], 0.0, 1.0) * 255.0)


class RaycastPreviewRenderer:
    def __init__(
        self,
        scene: RoboLabSceneConfig,
        object_model: newton.Model | None,
        robot_link_names: list[str] | None,
        output_dir: Path | str,
        device,
        camera_names: list[str] | None = None,
        video_fps: float | None = None,
        frames_per_image: int = 0,
        instances: list[_Instance] | None = None,
    ):
        self.scene = scene
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_per_image = frames_per_image
        self.video_fps = video_fps if video_fps is not None else scene.fps

        self._soft_start = 0
        self._soft_count = 0
        if instances is not None:
            self.instances = instances
        else:
            self.instances = []
            self._build_robot_instances(robot_link_names)
            self._build_object_instances(object_model)
            self._build_fixture_instances()
        self._upload()

        wanted = camera_names or [c.name for c in scene.cameras()]
        self.cameras: dict[str, CameraConfig | WristCameraConfig] = {
            c.name: c for c in scene.cameras() if c.name in wanted
        }
        self._writers: dict[str, "object"] = {}
        self._frame_idx = 0
        self.wrist_coverage: list[dict[str, float]] = []

    # ------------------------------------------------------------- instances

    def _build_robot_instances(self, link_names: list[str]) -> None:
        robot_usd = self.scene.robot_usd
        if robot_usd is None:
            return
        link_meshes = _robot_link_meshes(Path(robot_usd), link_names)
        for link, meshes in link_meshes.items():
            for j, (verts, tris) in enumerate(meshes):
                self.instances.append(
                    _Instance(
                        name=f"{link}_{j}",
                        category=CATEGORIES.index("robot"),
                        kind="robot_link",
                        key=link,
                        verts=verts,
                        tris=tris,
                        color=(0.79, 0.79, 0.81) if "finger" not in link else (0.25, 0.25, 0.27),
                    )
                )

    def _build_object_instances(self, model: newton.Model) -> None:
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_color = np.asarray(model.shape_color.numpy() if hasattr(model.shape_color, "numpy") else model.shape_color)
        labels = list(model.shape_label)

        for s in range(model.shape_count):
            label = labels[s] or f"shape_{s}"
            if not (int(shape_flags[s]) & int(newton.ShapeFlags.VISIBLE)):
                continue
            if any(label.startswith(p) for p in self.scene.skip_shape_label_prefixes):
                continue

            stype, scale = int(shape_type[s]), shape_scale[s]
            if stype == int(newton.GeoType.CAPSULE):
                m = NewtonMesh.create_capsule(
                    float(scale[0]), float(scale[1]), up_axis=Axis.Z, segments=12,
                    compute_normals=False, compute_uvs=False, compute_inertia=False,
                )
                verts = np.asarray(m.vertices, dtype=np.float32)
                tris = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)
            elif stype == int(newton.GeoType.BOX):
                verts, tris = _box_mesh(-np.asarray(scale, dtype=np.float32), np.asarray(scale, dtype=np.float32))
            elif stype == int(newton.GeoType.SPHERE):
                m = NewtonMesh.create_sphere(float(scale[0]), num_latitudes=12, num_longitudes=12,
                                             compute_normals=False, compute_uvs=False, compute_inertia=False)
                verts = np.asarray(m.vertices, dtype=np.float32)
                tris = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)
            elif stype == int(newton.GeoType.MESH) and model.shape_source[s] is not None:
                src = model.shape_source[s]
                verts = np.asarray(src.vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[None, :]
                tris = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
            else:
                continue

            # bake the shape's local transform into the vertices
            tf = shape_transform[s]
            verts = (_quat_rotate_np(tf[3:7], verts.astype(np.float64)) + tf[:3]).astype(np.float32)

            style = None
            for prefix, st in self.scene.object_styles.items():
                if label.startswith(prefix):
                    style = st
                    break
            color = style.color if style is not None else tuple(float(c) for c in shape_color[s])

            body = int(shape_body[s])
            self.instances.append(
                _Instance(
                    name=label,
                    category=CATEGORIES.index("objects"),
                    kind="object_body" if body >= 0 else "static",
                    key=body if body >= 0 else None,
                    verts=verts,
                    tris=tris,
                    color=color,
                )
            )

        if model.particle_count and model.tri_count:
            tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32)
            self.instances.append(
                _Instance(
                    name="soft_body",
                    category=CATEGORIES.index("soft"),
                    kind="soft",
                    key=None,
                    verts=np.zeros((model.particle_count, 3), dtype=np.float32),
                    tris=tris,
                    color=self.scene.soft_body_style.color,
                )
            )

    def _build_fixture_instances(self) -> None:
        from pxr import Gf, Sdf

        off = np.asarray(self.scene.sim_to_viz_translation, dtype=np.float64)
        for fix in self.scene.fixtures:
            if not fix.preview:
                continue
            # Compose the fixture exactly as the stage writer authors it
            # (payload + TRS + child overrides) and take the world bbox, so the
            # preview box matches the rendered USD fixture.
            stage = Usd.Stage.CreateInMemory()
            prim = stage.DefinePrim("/fixture")
            prim.GetPayloads().AddPayload(str(fix.usd_path))
            xf = UsdGeom.Xformable(prim)
            t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "robolab")
            t_op.Set(Gf.Vec3d(*[float(v) for v in fix.translate]))
            r_op = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "robolab")
            r_op.Set(float(fix.rotate_z_deg))
            s_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "robolab")
            s_op.Set(Gf.Vec3f(*[float(v) for v in fix.scale]))
            xf.SetXformOpOrder([t_op, r_op, s_op])
            for child, translate in fix.child_translate_overrides.items():
                over = stage.OverridePrim(prim.GetPath().AppendChild(child))
                over.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*[float(v) for v in translate])
                )
            stage.Load()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
            verts, tris = _box_mesh(mn, mx)
            verts = (verts.astype(np.float64) - off).astype(np.float32)  # viz -> sim frame
            texture = None
            if fix.texture_file is not None and Path(fix.texture_file).exists():
                texture = _load_texture(fix.texture_file)
            self.instances.append(
                _Instance(
                    name=fix.name,
                    category=CATEGORIES.index("fixture"),
                    kind="static",
                    key=None,
                    verts=verts,
                    tris=tris,
                    color=fix.preview_color,
                    texture=texture,
                    uv_scale=fix.texture_uv_scale,
                )
            )

    def _upload(self) -> None:
        all_verts, all_tris, vert_inst = [], [], []
        face_cat, face_color, face_tex, face_uv = [], [], [], []
        atlas: list[np.ndarray] = []  # distinct textures, index -> face_tex value
        offset = 0
        self._inst_index: dict[int, int] = {}
        for i, inst in enumerate(self.instances):
            if inst.kind == "soft":
                self._soft_start = offset
                self._soft_count = len(inst.verts)
            all_verts.append(inst.verts)
            all_tris.append(inst.tris + offset)
            vert_inst.append(np.full(len(inst.verts), i, dtype=np.int32))
            face_cat.append(np.full(len(inst.tris), inst.category, dtype=np.int32))
            face_color.append(np.tile(np.asarray(inst.color, dtype=np.float32), (len(inst.tris), 1)))
            if inst.texture is not None:
                tex_idx = len(atlas)
                atlas.append(inst.texture)
            else:
                tex_idx = -1
            face_tex.append(np.full(len(inst.tris), tex_idx, dtype=np.int32))
            face_uv.append(np.full(len(inst.tris), inst.uv_scale, dtype=np.float32))
            offset += len(inst.verts)

        verts = np.concatenate(all_verts).astype(np.float32)
        tris = np.concatenate(all_tris).astype(np.int32)
        # Texture atlas: (N, tile, tile, 3) uint8; a 1x1x1 dummy when untextured
        # (warp needs a concrete array even if every face_tex is -1).
        atlas_np = np.stack(atlas).astype(np.uint8) if atlas else np.zeros((1, 1, 1, 3), dtype=np.uint8)
        self._tex_h, self._tex_w = atlas_np.shape[1], atlas_np.shape[2]

        # Dome background: tone-mapped equirectangular image sampled on ray miss.
        bg_np = None
        dome = self.scene.dome_light
        if dome is not None and dome.texture_file is not None and Path(dome.texture_file).exists():
            bg_np = _load_background(dome.texture_file)
        self._has_bg = bg_np is not None
        if bg_np is None:
            bg_np = np.zeros((1, 1, 3), dtype=np.uint8)
        self._bg_h, self._bg_w = bg_np.shape[0], bg_np.shape[1]

        with wp.ScopedDevice(self.device):
            self._rest = wp.array(verts, dtype=wp.vec3)
            self._vert_inst = wp.array(np.concatenate(vert_inst), dtype=wp.int32)
            self._inst_tf = wp.zeros(len(self.instances), dtype=wp.transform)
            self._points = wp.zeros(len(verts), dtype=wp.vec3)
            self._face_cat = wp.array(np.concatenate(face_cat), dtype=wp.int32)
            self._face_color = wp.array(np.concatenate(face_color), dtype=wp.vec3)
            self._face_tex = wp.array(np.concatenate(face_tex), dtype=wp.int32)
            self._face_uv = wp.array(np.concatenate(face_uv), dtype=wp.float32)
            self._tex_atlas = wp.array(atlas_np, dtype=wp.uint8)
            self._bg = wp.array(bg_np, dtype=wp.uint8)
            self._tri_array = wp.array(tris.flatten(), dtype=wp.int32)
            # Built lazily on the first frame: building the BVH on the rest
            # positions (zeros for the soft body) gives a degenerate tree that
            # refit() can never repair, making every ray traverse all faces.
            self._mesh: wp.Mesh | None = None
            self._dummy_particles = wp.zeros(1, dtype=wp.vec3)
            self._imgs = {}
            self._cats = {}

    # ----------------------------------------------------------------- frame

    def _camera_pose_sim(self, cfg, robot_link_tfs) -> tuple[np.ndarray, np.ndarray]:
        """Camera world pose in the sim frame -> (pos(3), quat_xyzw(4))."""
        if isinstance(cfg, WristCameraConfig):
            hand = robot_link_tfs[cfg.parent_link]
            hand_p, hand_q = hand[:3].astype(np.float64), hand[3:7].astype(np.float64)
            pos = hand_p + _quat_rotate_np(hand_q, np.asarray(cfg.eye, dtype=np.float64))
            w, x, y, z = cfg.local_orientation_wxyz()
            local_q = np.array([x, y, z, w])
            # quaternion product hand_q * local_q (xyzw)
            q = np.array(
                [
                    hand_q[3] * local_q[0] + hand_q[0] * local_q[3] + hand_q[1] * local_q[2] - hand_q[2] * local_q[1],
                    hand_q[3] * local_q[1] - hand_q[0] * local_q[2] + hand_q[1] * local_q[3] + hand_q[2] * local_q[0],
                    hand_q[3] * local_q[2] + hand_q[0] * local_q[1] - hand_q[1] * local_q[0] + hand_q[2] * local_q[3],
                    hand_q[3] * local_q[3] - hand_q[0] * local_q[0] - hand_q[1] * local_q[1] - hand_q[2] * local_q[2],
                ]
            )
            return pos, q
        off = np.asarray(self.scene.sim_to_viz_translation, dtype=np.float64)
        pos = np.asarray(cfg.position, dtype=np.float64) - off
        w, x, y, z = cfg.orientation_wxyz
        return pos, np.array([x, y, z, w], dtype=np.float64)

    def render_frame(self, robot_link_tfs: dict[str, np.ndarray], object_body_q: np.ndarray, particle_q) -> None:
        import cv2

        from .video import VideoWriter

        tfs = np.zeros((len(self.instances), 7), dtype=np.float32)
        tfs[:, 6] = 1.0
        for i, inst in enumerate(self.instances):
            if inst.kind == "robot_link":
                tfs[i] = robot_link_tfs[inst.key]
            elif inst.kind == "object_body":
                tfs[i] = object_body_q[inst.key]
        self._inst_tf.assign(tfs)

        frame_rgbs: dict[str, np.ndarray] = {}
        particles = particle_q if (particle_q is not None and self._soft_count) else self._dummy_particles
        wp.launch(
            _transform_verts_kernel,
            dim=len(self._rest),
            inputs=[self._rest, self._vert_inst, self._inst_tf, self._soft_start, self._soft_count, particles],
            outputs=[self._points],
            device=self.device,
        )
        if self._mesh is None:
            with wp.ScopedDevice(self.device):
                self._mesh = wp.Mesh(points=self._points, indices=self._tri_array)
        else:
            self._mesh.refit()

        for name, cfg in self.cameras.items():
            pos, q = self._camera_pose_sim(cfg, robot_link_tfs)
            tan_w = cfg.horizontal_aperture / (2.0 * cfg.focal_length)
            tan_h = cfg.vertical_aperture / (2.0 * cfg.focal_length)
            if name not in self._imgs:
                with wp.ScopedDevice(self.device):
                    self._imgs[name] = wp.zeros((cfg.height, cfg.width, 3), dtype=wp.uint8)
                    self._cats[name] = wp.zeros((cfg.height, cfg.width), dtype=wp.int32)
            wp.launch(
                _render_kernel,
                dim=(cfg.width, cfg.height),
                inputs=[
                    self._mesh.id,
                    wp.vec3(*pos.astype(np.float32)),
                    wp.quat(*q.astype(np.float32)),
                    tan_w,
                    tan_h,
                    cfg.width,
                    cfg.height,
                    self._face_cat,
                    self._face_color,
                    self._face_tex,
                    self._face_uv,
                    self._tex_atlas,
                    self._tex_h,
                    self._tex_w,
                    self._bg,
                    self._bg_h,
                    self._bg_w,
                    int(self._has_bg),
                ],
                outputs=[self._imgs[name], self._cats[name]],
                device=self.device,
            )
            rgb = self._imgs[name].numpy()
            frame_rgbs[name] = rgb
            # Per-camera PNG frames (full resolution) into a frames/ subdir.
            if self.frames_per_image and self._frame_idx % self.frames_per_image == 0:
                frames_dir = self.output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                cv2.imwrite(
                    str(frames_dir / f"{name}_{self._frame_idx:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
            if isinstance(cfg, WristCameraConfig):
                cats = self._cats[name].numpy()
                total = cats.size
                cov = {CATEGORIES[k]: float((cats == k).sum()) / total for k in range(len(CATEGORIES))}
                cov["frame"] = self._frame_idx
                self.wrist_coverage.append(cov)

        # The simulation video (RoboLab-style combined observation video):
        # cameras concatenated horizontally at half scale, or the single camera
        # full-frame when only one is previewed. Cameras flagged
        # in_combined_video=False (e.g. an object-inspection still view) are
        # rendered + dumped as PNGs above but kept out of this video.
        imgs = [
            rgb for name, rgb in frame_rgbs.items() if getattr(self.cameras[name], "in_combined_video", True)
        ]
        if imgs:
            if len(imgs) == 1:
                combined = imgs[0]
            else:
                halves = [
                    cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
                    for img in imgs
                ]
                target_h = min(h.shape[0] for h in halves)
                combined = np.concatenate([h[:target_h] for h in halves], axis=1)
            if "simulation" not in self._writers:
                self._writers["simulation"] = VideoWriter(str(self.output_dir / "simulation.mp4"), self.video_fps)
            self._writers["simulation"].write(combined)

        self._frame_idx += 1

    def export_instances(self, path: Path | str) -> None:
        """Serialize the preview geometry so ``robolabViz.rerender`` can re-render
        cameras from a state cache without rebuilding the Newton model."""
        import pickle

        dome = self.scene.dome_light
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "instances": self.instances,
                    "sim_to_viz_translation": self.scene.sim_to_viz_translation,
                    "dome_texture": str(dome.texture_file) if dome and dome.texture_file else None,
                },
                f,
            )

    def close(self) -> dict:
        for w in self._writers.values():
            w.release()
        summary = {}
        if self.wrist_coverage:
            arr = {k: np.array([c[k] for c in self.wrist_coverage]) for k in CATEGORIES}
            summary = {
                "wrist_mean_coverage": {k: float(v.mean()) for k, v in arr.items()},
                "wrist_max_robot_coverage": float(arr["robot"].max()),
                "frames": len(self.wrist_coverage),
            }
            with open(self.output_dir / "wrist_coverage.json", "w") as f:
                json.dump({"per_frame": self.wrist_coverage, "summary": summary}, f, indent=2)
        print(f"[robolabViz] Preview renders written to {self.output_dir}")
        return summary
