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

from .config import CameraConfig, RenderQuality, RoboLabSceneConfig, WristCameraConfig

# Required for cv2 to decode the .exr dome backgrounds (e.g. home_office.exr).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

CATEGORIES = ("background", "robot", "objects", "soft", "fixture")

# Square tile every fixture texture is resized to before stacking into the GPU
# atlas, and the equirectangular size the (tone-mapped) dome HDR is sampled at.
_TEX_TILE = 1024
_BG_H, _BG_W = 1024, 2048


def _load_texture(path: Path | str, tile: int = _TEX_TILE) -> np.ndarray:
    """Load an sRGB base-color image as a square ``(tile, tile, 3)`` uint8 RGB array."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read texture {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)


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
    # Advanced-tier attributes (dataclass defaults do NOT apply to instances
    # unpickled from an older geometry.pkl — _normalize_instances backfills).
    vert_uv: np.ndarray | None = None       # (n,2) per-vertex UVs -> texture is UV-mapped, not planar
    normal_image: np.ndarray | None = None  # tangent-space normal map (planar-mapped fixtures)
    orm_image: np.ndarray | None = None     # occlusion/roughness/metallic map (G = roughness)
    roughness: float = 0.65
    metallic: float = 0.0
    smooth: bool = True                     # interpolate vertex normals (False = faceted, e.g. boxes)


# Backfill for instances unpickled from a geometry.pkl written before a field existed.
_INSTANCE_DEFAULTS = {
    "texture": None, "uv_scale": 0.5, "vert_uv": None, "normal_image": None,
    "orm_image": None, "roughness": 0.65, "metallic": 0.0, "smooth": True,
}


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


def _robot_link_meshes_from_model(model, object_body_min: int | None = None) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per-link ``(verts_local, tris)`` from the PHYSICS model's VISIBLE mesh shapes — i.e. the geometry
    that is ACTUALLY simulated (the convex hulls ``add_usd`` loads, or the URDF colliders), rather than the
    USD's detailed visual meshes. Same return format as :func:`_robot_link_meshes` (keyed by bare body
    name, verts baked into the body frame), so the render pipeline is identical downstream. Used when a
    robot opts into ``render_from_physics`` so the scenic render shows exactly the simulated geometry.

    On the rigid-only path the scene's objects are MERGED into this robot model (``add_builder``), so
    ``object_body_min`` gives the first merged-object body index: those bodies are NOT robot links (they
    are rendered separately as ``object_body`` instances, posed by the body state, NOT by FK), so skip
    them here — otherwise they'd be emitted as ``robot_link`` instances whose FK transform doesn't exist
    (e.g. a ``KeyError: 'bowl'`` at render time)."""
    st = model.shape_type.numpy()
    sf = model.shape_flags.numpy()
    sb = model.shape_body.numpy()
    s_tf = model.shape_transform.numpy()
    s_scale = model.shape_scale.numpy()
    labels = list(model.body_label)
    VIS = int(newton.ShapeFlags.VISIBLE)
    mesh_types = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for s in range(model.shape_count):
        if not (int(sf[s]) & VIS) or int(st[s]) not in mesh_types:
            continue
        src = model.shape_source[s]
        body = int(sb[s])
        if src is None or body < 0:
            continue
        if object_body_min is not None and body >= object_body_min:
            continue                                   # merged scene object, not a robot link (rigid-only)
        name = labels[body].split("/")[-1]            # bare link name -> matches the FK link_tfs keys
        v = np.asarray(src.vertices, dtype=np.float64) * np.asarray(s_scale[s], dtype=np.float64)[None, :]
        tf = np.asarray(s_tf[s], dtype=np.float64)    # shape -> body local transform (pos + quat xyzw)
        v = _quat_rotate_np(tf[3:7], v) + tf[:3]
        tris = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
        out.setdefault(name, []).append((v.astype(np.float32), tris))
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
        # Both lights are FIXED world directions (key + opposite-side fill). A camera-facing
        # fill term here reads as a headlamp that moves with the wrist camera — never that.
        key_light = wp.normalize(wp.vec3(0.35, -0.45, -0.82))
        fill_light = wp.normalize(wp.vec3(-0.45, 0.5, -0.6))
        shade = 0.42 + 0.43 * wp.max(wp.dot(n, -key_light), 0.0) + 0.15 * wp.max(wp.dot(n, -fill_light), 0.0)
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
        elif face_category[query.face] == 4:  # CATEGORIES.index("fixture")
            # Untextured fixture (e.g. the matte black table): a subtle world-space
            # speckle (~3 mm cells, pure ALU) so a uniform surface still gives motion
            # cues in the wrist camera instead of reading as a featureless plane.
            # Partly ADDITIVE so it stays visible on near-black albedos too.
            spos = (cam_pos + d * query.t) * 200.0
            hsh = wp.sin(
                wp.floor(spos[0]) * 12.9898 + wp.floor(spos[1]) * 78.233 + wp.floor(spos[2]) * 37.719
            ) * 43758.5453
            noise = hsh - wp.floor(hsh)
            base = base * (0.94 + 0.12 * noise) + wp.vec3(1.0, 1.0, 1.0) * (0.04 * (noise - 0.5))
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


# ---------------------------------------------------------------- advanced (PBR) tier
#
# A SEPARATE kernel so the flat tier above stays byte-identical. All shading is in
# linear space; the dome environment arrays are linear + auto-exposure-normalized
# (robolabViz.ibl), and the result is ACES-tone-mapped + gamma'd at the end.


@wp.func
def _srgb_to_lin(c: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.pow(c[0], 2.2), wp.pow(c[1], 2.2), wp.pow(c[2], 2.2))


@wp.func
def _aces1(x: float) -> float:
    x = wp.max(x, 0.0)
    return wp.clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


@wp.func
def _aces(c: wp.vec3) -> wp.vec3:
    # Narkowicz ACES fit, per channel.
    return wp.vec3(_aces1(c[0]), _aces1(c[1]), _aces1(c[2]))


@wp.func
def _sample_bilinear(
    tex: wp.array(dtype=wp.uint8, ndim=4), ti: int, u: float, v: float, h: int, w: int
) -> wp.vec3:
    # tile-wrap, bilinear; returns 0..1 sRGB-encoded values (caller linearizes albedo).
    x = (u - wp.floor(u)) * float(w) - 0.5
    y = (v - wp.floor(v)) * float(h) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)
    x0 = ((x0 % w) + w) % w
    x1 = (x0 + 1) % w
    y0c = wp.clamp(y0, 0, h - 1)
    y1c = wp.clamp(y0 + 1, 0, h - 1)
    c00 = wp.vec3(float(tex[ti, y0c, x0, 0]), float(tex[ti, y0c, x0, 1]), float(tex[ti, y0c, x0, 2]))
    c10 = wp.vec3(float(tex[ti, y0c, x1, 0]), float(tex[ti, y0c, x1, 1]), float(tex[ti, y0c, x1, 2]))
    c01 = wp.vec3(float(tex[ti, y1c, x0, 0]), float(tex[ti, y1c, x0, 1]), float(tex[ti, y1c, x0, 2]))
    c11 = wp.vec3(float(tex[ti, y1c, x1, 0]), float(tex[ti, y1c, x1, 1]), float(tex[ti, y1c, x1, 2]))
    return ((c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy) / 255.0


@wp.func
def _sample_env2(env: wp.array(dtype=wp.vec3, ndim=2), d: wp.vec3) -> wp.vec3:
    # bilinear equirect lookup (Z up), same latlong mapping as the flat backdrop.
    h = env.shape[0]
    w = env.shape[1]
    lon = wp.atan2(d[1], d[0]) * (0.5 / 3.14159265) + 0.5
    lat = wp.acos(wp.clamp(d[2], -1.0, 1.0)) * (1.0 / 3.14159265)
    x = lon * float(w) - 0.5
    y = lat * float(h) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)
    x0 = ((x0 % w) + w) % w
    x1 = (x0 + 1) % w
    y0c = wp.clamp(y0, 0, h - 1)
    y1c = wp.clamp(y0 + 1, 0, h - 1)
    return (env[y0c, x0] * (1.0 - fx) + env[y0c, x1] * fx) * (1.0 - fy) + (
        env[y1c, x0] * (1.0 - fx) + env[y1c, x1] * fx
    ) * fy


@wp.func
def _sample_env3(env: wp.array(dtype=wp.vec3, ndim=3), level: int, d: wp.vec3) -> wp.vec3:
    h = env.shape[1]
    w = env.shape[2]
    lon = wp.atan2(d[1], d[0]) * (0.5 / 3.14159265) + 0.5
    lat = wp.acos(wp.clamp(d[2], -1.0, 1.0)) * (1.0 / 3.14159265)
    x = lon * float(w) - 0.5
    y = lat * float(h) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)
    x0 = ((x0 % w) + w) % w
    x1 = (x0 + 1) % w
    y0c = wp.clamp(y0, 0, h - 1)
    y1c = wp.clamp(y0 + 1, 0, h - 1)
    return (env[level, y0c, x0] * (1.0 - fx) + env[level, y0c, x1] * fx) * (1.0 - fy) + (
        env[level, y1c, x0] * (1.0 - fx) + env[level, y1c, x1] * fx
    ) * fy


@wp.func
def _brdf(kd: wp.vec3, f0: wp.vec3, rough: float, n: wp.vec3, v: wp.vec3, l: wp.vec3) -> wp.vec3:
    # Energy-conserving Lambert + GGX (Smith-Schlick G, Schlick F) for an analytic light.
    # v = toward viewer, l = toward light; multiply by L_i * (n.l) * visibility outside.
    # The specular lobe is attenuated by (1-rough)^2: at grazing incidence the
    # 1/(4 n.v) term turns a bright near light into a view-tracking streak on matte
    # surfaces (a "flashlight" that follows the camera) — matte materials here are
    # diffuse-dominated like RoboLab's MDLs, while glossy ones keep their highlight.
    h = wp.normalize(v + l)
    ndv = wp.max(wp.dot(n, v), 1.0e-4)
    ndl = wp.max(wp.dot(n, l), 1.0e-4)
    ndh = wp.max(wp.dot(n, h), 0.0)
    hdv = wp.max(wp.dot(h, v), 0.0)
    a = wp.max(rough * rough, 1.0e-3)
    a2 = a * a
    dd = ndh * ndh * (a2 - 1.0) + 1.0
    dist = a2 / (3.14159265 * dd * dd)
    k = (rough + 1.0) * (rough + 1.0) / 8.0
    g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k))
    fr = f0 + (wp.vec3(1.0, 1.0, 1.0) - f0) * wp.pow(1.0 - hdv, 5.0)
    spec_atten = (1.0 - rough) * (1.0 - rough)
    return kd * (1.0 / 3.14159265) + fr * (dist * g * spec_atten / (4.0 * ndv * ndl + 1.0e-4))


@wp.kernel
def _accum_vert_normals_kernel(
    points: wp.array(dtype=wp.vec3),
    tris: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
):
    f = wp.tid()
    i0 = tris[3 * f + 0]
    i1 = tris[3 * f + 1]
    i2 = tris[3 * f + 2]
    n = wp.cross(points[i1] - points[i0], points[i2] - points[i0])  # area-weighted
    wp.atomic_add(normals, i0, n)
    wp.atomic_add(normals, i1, n)
    wp.atomic_add(normals, i2, n)


@wp.kernel
def _normalize_vert_normals_kernel(normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    n = normals[i]
    l = wp.length(n)
    if l > 1.0e-12:
        normals[i] = n / l


@wp.kernel
def _render_kernel_advanced(
    mesh_id: wp.uint64,
    cam_pos: wp.vec3,
    cam_quat: wp.quat,
    tan_half_w: float,
    tan_half_h: float,
    width: int,
    height: int,
    cam_index: int,
    tris: wp.array(dtype=wp.int32),
    vert_uv: wp.array(dtype=wp.vec2),
    vert_normal: wp.array(dtype=wp.vec3),
    face_category: wp.array(dtype=wp.int32),
    face_color: wp.array(dtype=wp.vec3),
    face_mode: wp.array(dtype=wp.int32),
    face_smooth: wp.array(dtype=wp.int32),
    face_rough: wp.array(dtype=wp.float32),
    face_metal: wp.array(dtype=wp.float32),
    face_tex: wp.array(dtype=wp.int32),
    face_tex_n: wp.array(dtype=wp.int32),
    face_tex_orm: wp.array(dtype=wp.int32),
    face_uv: wp.array(dtype=wp.float32),
    tex_atlas: wp.array(dtype=wp.uint8, ndim=4),
    tex_h: int,
    tex_w: int,
    env_bg: wp.array(dtype=wp.vec3, ndim=2),
    env_irr: wp.array(dtype=wp.vec3, ndim=2),
    env_spec: wp.array(dtype=wp.vec3, ndim=3),
    has_env: int,
    key_dir: wp.array(dtype=wp.vec3),
    key_rad: wp.array(dtype=wp.vec3),
    key_count: int,
    sphere_pos: wp.array(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    sphere_rad: wp.array(dtype=wp.vec3),
    sphere_count: int,
    aa_samples: int,
    shadow_samples: int,
    seed: int,
    img: wp.array(dtype=wp.uint8, ndim=3),
    category_buf: wp.array(dtype=wp.int32, ndim=2),
):
    px, py = wp.tid()
    # Frame-INDEPENDENT per-pixel RNG: the residual sampling noise is a static
    # grain (no temporal flicker) and the whole video is deterministic.
    state = wp.rand_init(seed, (cam_index * height + py) * width + px)
    accum = wp.vec3(0.0, 0.0, 0.0)
    for s in range(aa_samples):
        jx = 0.5  # sample 0 is always the pixel center (categories/coverage match the flat tier)
        jy = 0.5
        if s > 0:
            jx = wp.randf(state)
            jy = wp.randf(state)
        u = (float(px) + jx) / float(width) * 2.0 - 1.0
        v = 1.0 - (float(py) + jy) / float(height) * 2.0
        d = wp.quat_rotate(cam_quat, wp.normalize(wp.vec3(u * tan_half_w, v * tan_half_h, -1.0)))
        query = wp.mesh_query_ray(mesh_id, cam_pos, d, 100.0)
        if query.result:
            f = query.face
            if s == 0:
                category_buf[py, px] = face_category[f]
            i0 = tris[3 * f + 0]
            i1 = tris[3 * f + 1]
            i2 = tris[3 * f + 2]
            w2 = 1.0 - query.u - query.v
            ng = query.normal
            if wp.dot(ng, d) > 0.0:
                ng = -ng
            n = ng
            if face_smooth[f] == 1:
                ns = query.u * vert_normal[i0] + query.v * vert_normal[i1] + w2 * vert_normal[i2]
                if wp.length_sq(ns) > 1.0e-8:
                    ns = wp.normalize(ns)
                    if wp.dot(ns, d) > 0.0:  # two-sided/backface: shade the side we see
                        ns = -ns
                    n = ns
            hit = cam_pos + d * query.t
            base = face_color[f]
            rough = face_rough[f]
            metal = face_metal[f]
            mode = face_mode[f]
            ti = face_tex[f]
            if ti >= 0 and mode == 2:
                # per-vertex UVs (catalog object scans); USD st origin is bottom-left -> V flip.
                uv = query.u * vert_uv[i0] + query.v * vert_uv[i1] + w2 * vert_uv[i2]
                base = _sample_bilinear(tex_atlas, ti, uv[0], 1.0 - uv[1], tex_h, tex_w)
            elif ti >= 0 and mode == 1:
                # planar (triplanar) projection by dominant GEOMETRIC normal axis, world meters;
                # the projection axes double as the tangent frame for the normal map.
                sc = face_uv[f]
                ax = wp.abs(ng[0])
                ay = wp.abs(ng[1])
                az = wp.abs(ng[2])
                tu = 0.0
                tv = 0.0
                t_axis = wp.vec3(0.0, 0.0, 0.0)
                b_axis = wp.vec3(0.0, 0.0, 0.0)
                if az >= ax and az >= ay:
                    tu = hit[0] / sc
                    tv = hit[1] / sc
                    t_axis = wp.vec3(1.0, 0.0, 0.0)
                    b_axis = wp.vec3(0.0, 1.0, 0.0)
                elif ax >= ay:
                    tu = hit[1] / sc
                    tv = hit[2] / sc
                    t_axis = wp.vec3(0.0, 1.0, 0.0)
                    b_axis = wp.vec3(0.0, 0.0, 1.0)
                else:
                    tu = hit[0] / sc
                    tv = hit[2] / sc
                    t_axis = wp.vec3(1.0, 0.0, 0.0)
                    b_axis = wp.vec3(0.0, 0.0, 1.0)
                base = _sample_bilinear(tex_atlas, ti, tu, tv, tex_h, tex_w)
                ni = face_tex_n[f]
                if ni >= 0:
                    tn = _sample_bilinear(tex_atlas, ni, tu, tv, tex_h, tex_w) * 2.0 - wp.vec3(1.0, 1.0, 1.0)
                    n = wp.normalize(t_axis * tn[0] + b_axis * tn[1] + n * wp.max(tn[2], 0.2))
                oi = face_tex_orm[f]
                if oi >= 0:
                    rough = _sample_bilinear(tex_atlas, oi, tu, tv, tex_h, tex_w)[1]  # ORM.g
            albedo = _srgb_to_lin(base)
            f0 = wp.vec3(0.04, 0.04, 0.04) * (1.0 - metal) + albedo * metal
            kd = albedo * (1.0 - metal)
            # diffuse IBL (irr already includes the 1/pi)
            col = wp.cw_mul(kd, _sample_env2(env_irr, n))
            # environment specular along the reflection, roughness-prefiltered (no shadow ray).
            refl = d - n * (2.0 * wp.dot(d, n))
            lvl = wp.clamp(rough * 3.0 - 0.5, 0.0, 2.0)
            l0 = int(wp.floor(lvl))
            l1 = wp.min(l0 + 1, 2)
            lf = lvl - float(l0)
            spec_env = _sample_env3(env_spec, l0, refl) * (1.0 - lf) + _sample_env3(env_spec, l1, refl) * lf
            # The WHOLE env specular (including the F0 floor) is attenuated by
            # (1-rough)^2: on a matte surface even 0.04 * a bright env region renders a
            # smeared view-tracking reflection streak (a "flashlight" following the
            # camera), and grazing fresnel sheen makes the same table read gray from a
            # low camera and black from a high one. Glossy surfaces keep reflections.
            fenv = f0 + (wp.vec3(1.0, 1.0, 1.0) - f0) * wp.pow(1.0 - wp.max(wp.dot(n, -d), 0.0), 5.0)
            col += wp.cw_mul(fenv, spec_env) * ((1.0 - rough) * (1.0 - rough))
            # dome key "suns": shadow-tested directional lights, jittered ~3 deg (soft window edges).
            for k in range(key_count):
                l = key_dir[k]
                l = wp.normalize(
                    l + wp.vec3(
                        (wp.randf(state) - 0.5) * 0.1,
                        (wp.randf(state) - 0.5) * 0.1,
                        (wp.randf(state) - 0.5) * 0.1,
                    )
                )
                ndl = wp.dot(n, l)
                if ndl > 0.0:
                    sq = wp.mesh_query_ray(mesh_id, hit + ng * 3.0e-4, l, 10.0)
                    if not sq.result:
                        col += wp.cw_mul(_brdf(kd, f0, rough, n, -d, l), key_rad[k]) * ndl
            # sphere lights: stratified shadow rays toward the light disc (soft shadows).
            for si in range(sphere_count):
                to_l = sphere_pos[si] - hit
                dist = wp.length(to_l)
                if dist > 1.0e-4:
                    ldir = to_l / dist
                    r = sphere_radius[si]
                    up = wp.vec3(0.0, 0.0, 1.0)
                    if wp.abs(ldir[2]) > 0.9:
                        up = wp.vec3(1.0, 0.0, 0.0)
                    t1 = wp.normalize(wp.cross(ldir, up))
                    t2 = wp.cross(ldir, t1)
                    li = sphere_rad[si] * (r * r / (r * r + dist * dist))
                    # Area-light specular correction: point-sampling a BIG close light
                    # through GGX makes a tight view-dependent streak (a "flashlight"
                    # that follows the camera). Widen the lobe by the light's angular
                    # radius so it renders as the broad soft sheen an RTX area light gives.
                    light_rough = wp.clamp(rough + 0.5 * r / wp.max(dist, r), rough, 1.0)
                    for j in range(shadow_samples):
                        ang = 2.0 * 3.14159265 * (float(j) + wp.randf(state)) / float(shadow_samples)
                        rad = r * wp.sqrt(wp.randf(state))
                        p = sphere_pos[si] + (t1 * wp.cos(ang) + t2 * wp.sin(ang)) * rad
                        sd = p - (hit + ng * 3.0e-4)
                        sl = wp.length(sd)
                        sdir = sd / sl
                        ndl = wp.dot(n, sdir)
                        if ndl > 0.0:
                            sq = wp.mesh_query_ray(mesh_id, hit + ng * 3.0e-4, sdir, sl - r * 0.05)
                            if not sq.result:
                                col += wp.cw_mul(_brdf(kd, f0, light_rough, n, -d, sdir), li) * (
                                    ndl / float(shadow_samples)
                                )
            accum += col
        else:
            if s == 0:
                category_buf[py, px] = 0
            if has_env == 1:
                accum += _sample_env2(env_bg, d)
            else:
                g = wp.pow(0.55 + 0.25 * wp.max(d[2], 0.0), 2.2)  # linear-ized sky/floor gradient
                accum += wp.vec3(g * 0.9, g * 0.94, g) * 1.1
    c = _aces(accum / float(aa_samples))
    img[py, px, 0] = wp.uint8(wp.clamp(wp.pow(c[0], 1.0 / 2.2), 0.0, 1.0) * 255.0)
    img[py, px, 1] = wp.uint8(wp.clamp(wp.pow(c[1], 1.0 / 2.2), 0.0, 1.0) * 255.0)
    img[py, px, 2] = wp.uint8(wp.clamp(wp.pow(c[2], 1.0 / 2.2), 0.0, 1.0) * 255.0)


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
        object_body_min: int | None = None,
        gripper_color: tuple[float, float, float] | None = None,
        robot_model: newton.Model | None = None,
        robot_from_physics: bool = False,
        quality: RenderQuality | None = None,
        write_coverage_json: bool = True,
        video_name: str = "simulation.mp4",
    ):
        self.scene = scene
        self.device = device
        # Render tier. The default RenderQuality() reproduces the historical preview exactly
        # (HDRI backdrop + fixture textures, flat shading), so pre-tier call sites (rerender)
        # behave unchanged; the runner passes RenderQuality.for_mode(--output-style).
        self.quality = quality if quality is not None else RenderQuality()
        self._write_coverage_json = write_coverage_json
        self._gripper_color = gripper_color   # optional tint for hand+finger links (tell robots apart)
        # When True, render the robot from the simulated (physics) model's collision/visual meshes
        # instead of the USD's detailed visual meshes — so the render matches the simulated geometry.
        self._robot_model = robot_model
        self._robot_from_physics = robot_from_physics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_per_image = frames_per_image
        # Every style shares outputs/<robot>/<name>/; the video FILENAME carries the
        # tier (simulation.mp4 vs simulation_advanced.mp4) so the tiers never clobber.
        self.video_name = video_name
        if self.quality.fps is not None:
            self.video_fps = self.quality.fps
        else:
            self.video_fps = video_fps if video_fps is not None else scene.fps

        # On the rigid-only path the "object model" is the robot model (objects merged into it); only
        # bodies at/after this index are the objects (earlier ones are the robot's own links, rendered
        # via FK). None = render every body (the VBD path's separate object model).
        self._object_body_min = object_body_min
        self._soft_start = 0
        self._soft_count = 0
        if instances is not None:
            self.instances = instances
        else:
            self.instances = []
            self._build_robot_instances(robot_link_names)
            self._build_object_instances(object_model)
            self._build_fixture_instances()
        self._normalize_instances()
        self._upload()
        if self.quality.use_pbr:
            self._upload_lighting()

        wanted = camera_names or [c.name for c in scene.cameras()]
        self.cameras: dict[str, CameraConfig | WristCameraConfig] = {
            c.name: c for c in scene.cameras() if c.name in wanted
        }
        self._writers: dict[str, "object"] = {}
        self._frame_idx = 0
        self.wrist_coverage: list[dict[str, float]] = []
        self._render_seconds = 0.0
        self._render_frames = 0

    # ------------------------------------------------------------- instances

    def _normalize_instances(self) -> None:
        """Backfill attributes on instances unpickled from an older geometry.pkl
        (pickle restores __dict__, so newer dataclass fields are simply absent)."""
        for inst in self.instances:
            for k, v in _INSTANCE_DEFAULTS.items():
                if not hasattr(inst, k):
                    setattr(inst, k, v)

    def _build_robot_instances(self, link_names: list[str]) -> None:
        # Two mesh sources: the simulated physics model (true collision geometry) when the robot opts in,
        # else the USD's detailed visual meshes. Both return {link_name: [(verts_local, tris)]}.
        if self._robot_from_physics and self._robot_model is not None:
            link_meshes = _robot_link_meshes_from_model(self._robot_model, self._object_body_min)
        else:
            robot_usd = self.scene.robot_usd
            if robot_usd is None:
                return
            link_meshes = _robot_link_meshes(Path(robot_usd), link_names)
        is_gripper = lambda name: ("finger" in name) or ("hand" in name)
        for link, meshes in link_meshes.items():
            # Default robot look: light-gray links, dark-gray fingers. If a gripper tint is set (used to
            # distinguish robots at a glance), it overrides the hand + finger links.
            if self._gripper_color is not None and is_gripper(link):
                link_color = self._gripper_color
            elif "finger" in link:
                link_color = (0.25, 0.25, 0.27)
            else:
                link_color = (0.79, 0.79, 0.81)
            for j, (verts, tris) in enumerate(meshes):
                self.instances.append(
                    _Instance(
                        name=f"{link}_{j}",
                        category=CATEGORIES.index("robot"),
                        kind="robot_link",
                        key=link,
                        verts=verts,
                        tris=tris,
                        color=link_color,
                        roughness=0.45,  # Franka's plastic shell
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

        def _shape_visible(s: int) -> bool:
            label = labels[s] or f"shape_{s}"
            if not (int(shape_flags[s]) & int(newton.ShapeFlags.VISIBLE)):
                return False
            if any(label.startswith(p) for p in self.scene.skip_shape_label_prefixes):
                return False
            body = int(shape_body[s])
            if self._object_body_min is not None and body < self._object_body_min:
                return False
            return True

        # Advanced tier: bodies labeled with a vendored-catalog asset name render as the
        # asset's textured scan mesh (visual-only re-read; the physics mesh is untouched).
        # Covers both mesh-visual bodies (banana/bowl -> textured instead of flat-colored)
        # and box-modeled ones (the rubik's 55-box model -> the textured scan).
        catalog_visuals: dict[int, object] = {}
        if self.quality.use_pbr and self.quality.use_textures:
            from .viz_assets import catalog_visual

            body_labels = list(model.body_label)
            for b in sorted({int(shape_body[s]) for s in range(model.shape_count) if _shape_visible(s)}):
                if b < 0:
                    continue
                viz = catalog_visual(body_labels[b], self.quality.tex_tile)
                if viz is not None:
                    catalog_visuals[b] = viz
        catalog_replaced: set[int] = set()

        for s in range(model.shape_count):
            label = labels[s] or f"shape_{s}"
            if not _shape_visible(s):
                continue

            stype, scale = int(shape_type[s]), shape_scale[s]
            body = int(shape_body[s])
            if body in catalog_visuals:
                if body in catalog_replaced:
                    continue  # every other visible shape of this body is replaced by the scan
                catalog_replaced.add(body)
                viz = catalog_visuals[body]
                verts = viz.verts.astype(np.float64)
                if stype == int(newton.GeoType.MESH) and model.shape_source[s] is not None:
                    # mesh-visual body: bake exactly like the physics visual (scale + local tf)
                    verts = verts * np.asarray(scale, dtype=np.float64)[None, :]
                else:
                    # box-modeled body (rubik's): center the scan on the body origin
                    verts = verts - 0.5 * (verts.min(axis=0) + verts.max(axis=0))
                tf = shape_transform[s]
                verts = (_quat_rotate_np(tf[3:7], verts) + tf[:3]).astype(np.float32)
                self.instances.append(
                    _Instance(
                        name=label,
                        category=CATEGORIES.index("objects"),
                        kind="object_body",
                        key=body,
                        verts=verts,
                        tris=viz.tris,
                        color=(1.0, 1.0, 1.0),
                        vert_uv=viz.uvs,
                        texture=viz.texture,
                        roughness=viz.roughness,
                        metallic=viz.metallic,
                    )
                )
                continue
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
            mat = style if style is not None else self.scene.default_object_style

            self.instances.append(
                _Instance(
                    name=label,
                    category=CATEGORIES.index("objects"),
                    kind="object_body" if body >= 0 else "static",
                    key=body if body >= 0 else None,
                    verts=verts,
                    tris=tris,
                    color=color,
                    roughness=float(mat.roughness),
                    metallic=float(mat.metallic),
                    smooth=stype != int(newton.GeoType.BOX),  # boxes keep hard-edged shading
                )
            )

        if model.particle_count and model.tri_count:
            tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32)
            soft_style = self.scene.soft_body_style
            self.instances.append(
                _Instance(
                    name="soft_body",
                    category=CATEGORIES.index("soft"),
                    kind="soft",
                    key=None,
                    verts=np.zeros((model.particle_count, 3), dtype=np.float32),
                    tris=tris,
                    color=soft_style.color,
                    roughness=float(soft_style.roughness),
                    metallic=float(soft_style.metallic),
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
            texture = normal_image = orm_image = None
            if (self.quality.use_textures
                    and fix.texture_file is not None and Path(fix.texture_file).exists()):
                texture = _load_texture(fix.texture_file, self.quality.tex_tile)
                if self.quality.use_pbr:  # normal/ORM relief is an advanced-tier feature
                    nf = getattr(fix, "normal_file", None)
                    if nf is not None and Path(nf).exists():
                        normal_image = _load_texture(nf, self.quality.tex_tile)
                    of = getattr(fix, "orm_file", None)
                    if of is not None and Path(of).exists():
                        orm_image = _load_texture(of, self.quality.tex_tile)
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
                    normal_image=normal_image,
                    orm_image=orm_image,
                    smooth=False,  # bbox proxy: keep hard edges
                )
            )

    def _upload(self) -> None:
        all_verts, all_tris, vert_inst, vert_uvs = [], [], [], []
        face_cat, face_color, face_tex, face_uv = [], [], [], []
        face_mode, face_smooth, face_rough, face_metal = [], [], [], []
        face_tex_n, face_tex_orm = [], []
        atlas: list[np.ndarray] = []  # distinct textures, index -> face_tex* value

        def _atlas_add(img: np.ndarray | None) -> int:
            if img is None:
                return -1
            atlas.append(img)
            return len(atlas) - 1

        offset = 0
        self._inst_index: dict[int, int] = {}
        for i, inst in enumerate(self.instances):
            if inst.kind == "soft":
                self._soft_start = offset
                self._soft_count = len(inst.verts)
            all_verts.append(inst.verts)
            all_tris.append(inst.tris + offset)
            vert_inst.append(np.full(len(inst.verts), i, dtype=np.int32))
            nf = len(inst.tris)
            face_cat.append(np.full(nf, inst.category, dtype=np.int32))
            face_color.append(np.tile(np.asarray(inst.color, dtype=np.float32), (nf, 1)))
            tex_idx = _atlas_add(inst.texture)
            uv = inst.vert_uv
            if tex_idx >= 0 and uv is not None and len(uv) == len(inst.verts):
                mode = 2  # UV-mapped
                vert_uvs.append(np.asarray(uv, dtype=np.float32))
            else:
                mode = 1 if tex_idx >= 0 else 0  # planar / flat
                vert_uvs.append(np.zeros((len(inst.verts), 2), dtype=np.float32))
            face_tex.append(np.full(nf, tex_idx, dtype=np.int32))
            face_uv.append(np.full(nf, inst.uv_scale, dtype=np.float32))
            face_mode.append(np.full(nf, mode, dtype=np.int32))
            face_smooth.append(np.full(nf, 1 if inst.smooth else 0, dtype=np.int32))
            face_rough.append(np.full(nf, float(inst.roughness), dtype=np.float32))
            face_metal.append(np.full(nf, float(inst.metallic), dtype=np.float32))
            face_tex_n.append(np.full(nf, _atlas_add(inst.normal_image), dtype=np.int32))
            face_tex_orm.append(np.full(nf, _atlas_add(inst.orm_image), dtype=np.int32))
            offset += len(inst.verts)

        verts = np.concatenate(all_verts).astype(np.float32)
        tris = np.concatenate(all_tris).astype(np.int32)
        # Texture atlas: (N, tile, tile, 3) uint8; a 1x1x1 dummy when untextured
        # (warp needs a concrete array even if every face_tex is -1).
        atlas_np = np.stack(atlas).astype(np.uint8) if atlas else np.zeros((1, 1, 1, 3), dtype=np.uint8)
        self._tex_h, self._tex_w = atlas_np.shape[1], atlas_np.shape[2]

        # Dome background: tone-mapped equirectangular image sampled on ray miss.
        # The lightweight tier skips the (slow) HDR decode (the kernel's built-in
        # sky/floor gradient takes over); the advanced tier decodes the dome ONCE,
        # linearly, in _upload_lighting instead (its kernel never reads this one).
        bg_np = None
        dome = self.scene.dome_light
        if (self.quality.use_hdri and not self.quality.use_pbr
                and dome is not None and dome.texture_file is not None and Path(dome.texture_file).exists()):
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
            # advanced-tier attributes (allocated in every tier; only the advanced kernel reads them)
            self._vert_uv = wp.array(np.concatenate(vert_uvs), dtype=wp.vec2)
            self._vert_normal = wp.zeros(len(verts), dtype=wp.vec3)
            self._face_mode = wp.array(np.concatenate(face_mode), dtype=wp.int32)
            self._face_smooth = wp.array(np.concatenate(face_smooth), dtype=wp.int32)
            self._face_rough = wp.array(np.concatenate(face_rough), dtype=wp.float32)
            self._face_metal = wp.array(np.concatenate(face_metal), dtype=wp.float32)
            self._face_tex_n = wp.array(np.concatenate(face_tex_n), dtype=wp.int32)
            self._face_tex_orm = wp.array(np.concatenate(face_tex_orm), dtype=wp.int32)
            self._tri_count = len(tris)
            # Built lazily on the first frame: building the BVH on the rest
            # positions (zeros for the soft body) gives a degenerate tree that
            # refit() can never repair, making every ray traverse all faces.
            self._mesh: wp.Mesh | None = None
            self._dummy_particles = wp.zeros(1, dtype=wp.vec3)
            self._imgs = {}
            self._cats = {}

    def _upload_lighting(self) -> None:
        """Advanced-tier light precompute (once, at init): linear dome environment
        (backdrop + IBL), diffuse irradiance, prefiltered specular levels, the K
        brightest dome directions as shadow-tested suns, and the scene's sphere
        lights converted to the sim frame."""
        from . import ibl

        q = self.quality
        dome = self.scene.dome_light
        env = None
        if (q.use_hdri and dome is not None
                and dome.texture_file is not None and Path(dome.texture_file).exists()):
            env = ibl.load_linear_env(dome.texture_file)
            # RenderSpec.dome_intensity scales relative to the RoboLab default (500).
            env *= float(dome.intensity) / 500.0
        if env is not None:
            key_dirs, key_rad, residual = ibl.extract_key_lights(env, k=q.ibl_key_lights)
            irr = ibl.build_irradiance(residual, q.irradiance_size)
            spec = ibl.prefilter_env(env, q.env_spec_size)
            self._has_env = True
        else:
            # No dome: constant ambient + the kernel's gradient backdrop.
            key_dirs = np.zeros((0, 3), dtype=np.float32)
            key_rad = np.zeros((0, 3), dtype=np.float32)
            env = np.zeros((2, 4, 3), dtype=np.float32)
            irr = np.full((q.irradiance_size[1], q.irradiance_size[0], 3), 0.35, dtype=np.float32)
            spec = np.full((3, q.env_spec_size[1], q.env_spec_size[0], 3), 0.35, dtype=np.float32)
            self._has_env = False
        self._key_count = len(key_dirs)
        if self._key_count == 0:  # warp wants concrete arrays even when unused
            key_dirs = np.zeros((1, 3), dtype=np.float32)
            key_rad = np.zeros((1, 3), dtype=np.float32)

        off = np.asarray(self.scene.sim_to_viz_translation, dtype=np.float64)
        sp, sr, srad = [], [], []
        for light in self.scene.sphere_lights:
            sp.append(np.asarray(light.position, dtype=np.float64) - off)  # viz -> sim frame
            sr.append(float(light.radius))
            # Radiance in the auto-exposure-normalized frame; intensity scales relative
            # to the RoboLab default SphereLight (5000).
            srad.append(np.asarray(light.color, dtype=np.float32)
                        * (q.sphere_light_scale * float(light.intensity) / 5000.0))
        self._sphere_count = len(sp)
        if self._sphere_count == 0:
            sp, sr, srad = [np.zeros(3)], [0.1], [np.zeros(3, dtype=np.float32)]

        with wp.ScopedDevice(self.device):
            self._env_bg = wp.array(env.astype(np.float32), dtype=wp.vec3)
            self._env_irr = wp.array(irr, dtype=wp.vec3)
            self._env_spec = wp.array(spec, dtype=wp.vec3)
            self._key_dir = wp.array(key_dirs, dtype=wp.vec3)
            self._key_rad = wp.array(key_rad, dtype=wp.vec3)
            self._sphere_pos = wp.array(np.asarray(sp, dtype=np.float32), dtype=wp.vec3)
            self._sphere_radius = wp.array(np.asarray(sr, dtype=np.float32), dtype=wp.float32)
            self._sphere_rad = wp.array(np.asarray(srad, dtype=np.float32), dtype=wp.vec3)

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
        import time

        import cv2

        from .video import VideoWriter

        t_start = time.perf_counter()
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

        advanced = self.quality.use_pbr
        if advanced and self.quality.smooth_normals:
            # Per-vertex normals over the CURRENT (posed/deformed) geometry — smooth
            # robot links, cloth, and mesh objects; face_smooth gates hard-edged boxes.
            self._vert_normal.zero_()
            wp.launch(
                _accum_vert_normals_kernel,
                dim=self._tri_count,
                inputs=[self._points, self._tri_array, self._vert_normal],
                device=self.device,
            )
            wp.launch(
                _normalize_vert_normals_kernel,
                dim=len(self._rest),
                inputs=[self._vert_normal],
                device=self.device,
            )

        for cam_index, (name, cfg) in enumerate(self.cameras.items()):
            pos, q = self._camera_pose_sim(cfg, robot_link_tfs)
            tan_w = cfg.horizontal_aperture / (2.0 * cfg.focal_length)
            tan_h = cfg.vertical_aperture / (2.0 * cfg.focal_length)
            if name not in self._imgs:
                with wp.ScopedDevice(self.device):
                    self._imgs[name] = wp.zeros((cfg.height, cfg.width, 3), dtype=wp.uint8)
                    self._cats[name] = wp.zeros((cfg.height, cfg.width), dtype=wp.int32)
            if advanced:
                wp.launch(
                    _render_kernel_advanced,
                    dim=(cfg.width, cfg.height),
                    inputs=[
                        self._mesh.id,
                        wp.vec3(*pos.astype(np.float32)),
                        wp.quat(*q.astype(np.float32)),
                        tan_w,
                        tan_h,
                        cfg.width,
                        cfg.height,
                        cam_index,
                        self._tri_array,
                        self._vert_uv,
                        self._vert_normal,
                        self._face_cat,
                        self._face_color,
                        self._face_mode,
                        self._face_smooth,
                        self._face_rough,
                        self._face_metal,
                        self._face_tex,
                        self._face_tex_n,
                        self._face_tex_orm,
                        self._face_uv,
                        self._tex_atlas,
                        self._tex_h,
                        self._tex_w,
                        self._env_bg,
                        self._env_irr,
                        self._env_spec,
                        int(self._has_env),
                        self._key_dir,
                        self._key_rad,
                        self._key_count,
                        self._sphere_pos,
                        self._sphere_radius,
                        self._sphere_rad,
                        self._sphere_count,
                        max(1, int(self.quality.aa_samples)),
                        max(1, int(self.quality.shadow_samples)),
                        int(self.quality.noise_seed),
                    ],
                    outputs=[self._imgs[name], self._cats[name]],
                    device=self.device,
                )
            else:
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
        # cameras concatenated horizontally at ``quality.video_scale`` (0.5 in the
        # lightweight tier, full-res in mp4_advanced), or the single camera
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
                scale = float(self.quality.video_scale)
                if scale < 0.999:
                    # keep even dims for the yuv420p encode
                    imgs = [
                        cv2.resize(
                            img,
                            (max(2, int(img.shape[1] * scale)) // 2 * 2, max(2, int(img.shape[0] * scale)) // 2 * 2),
                            interpolation=cv2.INTER_AREA,
                        )
                        for img in imgs
                    ]
                target_h = min(h.shape[0] for h in imgs)
                combined = np.concatenate([h[:target_h] for h in imgs], axis=1)
            if "simulation" not in self._writers:
                self._writers["simulation"] = VideoWriter(str(self.output_dir / self.video_name), self.video_fps)
            self._writers["simulation"].write(combined)

        self._frame_idx += 1
        self._render_seconds += time.perf_counter() - t_start
        self._render_frames += 1

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
            # The JSON artifact is an mp4_advanced output; the summary itself is always
            # computed (the occlusion check + console print consume it in every tier).
            if self._write_coverage_json:
                with open(self.output_dir / "wrist_coverage.json", "w") as f:
                    json.dump({"per_frame": self.wrist_coverage, "summary": summary}, f, indent=2)
        if self._render_frames:
            avg_ms = 1000.0 * self._render_seconds / self._render_frames
            print(f"[robolabViz] render avg {avg_ms:.1f} ms/frame over {self._render_frames} frames")
        print(f"[robolabViz] Preview renders written to {self.output_dir}")
        return summary
