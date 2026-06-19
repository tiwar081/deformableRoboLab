"""Author the RoboLab-styled scene as a time-sampled USD using pure ``pxr``.

No kit/Isaac process is required: the stage is plain OpenUSD referencing the
vendored RoboLab fixture assets, the converted robot USD, auto-generated prims
for the Newton object model (cable capsules, rigid shapes), and a deforming
mesh for the soft body. Each rendered physics frame becomes one USD time
sample, so the result plays back in usdview / Omniverse / Isaac Sim and can be
RTX-rendered offline on an RTX-capable machine (``robolabViz.isaac_render``).

Cameras are authored as USD Camera prims with RoboLab's intrinsics; the wrist
camera is a child of the robot's hand link prim, so it follows the puppeteered
hand automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import newton

from .config import CameraConfig, ObjectStyle, RoboLabSceneConfig, WristCameraConfig


def _sanitize(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def _matrix_from_pq(p, q_xyzw) -> Gf.Matrix4d:
    m = Gf.Matrix4d()
    m.SetRotate(Gf.Quatd(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])))
    m.SetTranslateOnly(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    return m


def _vertex_normals(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.zeros_like(points)
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    for k in range(3):
        np.add.at(n, tris[:, k], face_n)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens[lens < 1.0e-12] = 1.0
    return n / lens


class RoboLabStageWriter:
    def __init__(
        self,
        output_path: Path | str,
        scene: RoboLabSceneConfig,
        object_model: newton.Model,
        num_frames: int,
    ):
        self.scene = scene
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        self.stage = Usd.Stage.CreateNew(str(self.output_path))
        self.stage.SetMetadata("metersPerUnit", 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.stage.SetTimeCodesPerSecond(scene.fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(max(num_frames - 1, 0))

        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())

        self._materials: dict[tuple, UsdShade.Material] = {}
        self._frames_written = 0

        self._author_lights()
        self._author_fixtures()
        for cam in scene.exterior_cameras:
            self._author_world_camera(cam)

        sim = UsdGeom.Xform.Define(self.stage, "/World/sim")
        off = scene.sim_to_viz_translation
        sim.AddTranslateOp().Set(Gf.Vec3d(float(off[0]), float(off[1]), float(off[2])))

        self.robot_link_ops: dict[str, UsdGeom.XformOp] = {}
        if scene.robot_usd is not None:
            self._author_robot()

        self.body_ops: dict[int, UsdGeom.XformOp] = {}
        self.soft_mesh: UsdGeom.Mesh | None = None
        self._soft_tris: np.ndarray | None = None
        self._author_objects(object_model)

    # ------------------------------------------------------------------ scene

    def _rel(self, asset_path: Path | str) -> str:
        rel = os.path.relpath(str(asset_path), start=str(self.output_path.parent))
        return rel if rel.startswith((".", "/")) else "./" + rel

    def _author_lights(self) -> None:
        dome_cfg = self.scene.dome_light
        if dome_cfg is not None:
            dome = UsdLux.DomeLight.Define(self.stage, "/World/background")
            dome.CreateIntensityAttr(float(dome_cfg.intensity))
            if dome_cfg.texture_file is not None:
                dome.CreateTextureFileAttr(self._rel(dome_cfg.texture_file))
                dome.CreateTextureFormatAttr(dome_cfg.texture_format)
            # Kit-specific: keeps the HDR visible as the image background.
            dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
                bool(dome_cfg.visible_in_primary_ray)
            )

        for cfg in self.scene.sphere_lights:
            light = UsdLux.SphereLight.Define(self.stage, f"/World/{_sanitize(cfg.name)}")
            light.CreateIntensityAttr(float(cfg.intensity))
            light.CreateRadiusAttr(float(cfg.radius))
            light.CreateColorAttr(Gf.Vec3f(*[float(c) for c in cfg.color]))
            light.CreateExposureAttr(float(cfg.exposure))
            UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*[float(v) for v in cfg.position]))

    def _author_fixtures(self) -> None:
        for fix in self.scene.fixtures:
            prim = self.stage.DefinePrim(f"/World/{_sanitize(fix.name)}")
            prim.GetPayloads().AddPayload(self._rel(fix.usd_path))
            # Suffixed ops + a local xformOpOrder override: fixture assets ship
            # their own (sometimes non-TRS) op stacks, which we fully replace.
            xf = UsdGeom.Xformable(prim)
            t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "robolab")
            t_op.Set(Gf.Vec3d(*[float(v) for v in fix.translate]))
            r_op = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "robolab")
            r_op.Set(float(fix.rotate_z_deg))
            s_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "robolab")
            s_op.Set(Gf.Vec3f(*[float(v) for v in fix.scale]))
            xf.SetXformOpOrder([t_op, r_op, s_op])
            for child, translate in fix.child_translate_overrides.items():
                over = self.stage.OverridePrim(prim.GetPath().AppendChild(child))
                over.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*[float(v) for v in translate])
                )

    def _author_camera_prim(self, path: str, cfg: CameraConfig | WristCameraConfig) -> UsdGeom.Camera:
        cam = UsdGeom.Camera.Define(self.stage, path)
        cam.CreateFocalLengthAttr(float(cfg.focal_length))
        cam.CreateFocusDistanceAttr(float(cfg.focus_distance))
        cam.CreateHorizontalApertureAttr(float(cfg.horizontal_aperture))
        cam.CreateVerticalApertureAttr(float(cfg.vertical_aperture))
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(cfg.clipping[0]), float(cfg.clipping[1])))
        return cam

    def _author_world_camera(self, cfg: CameraConfig) -> None:
        cam = self._author_camera_prim(f"/World/{_sanitize(cfg.name)}", cfg)
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in cfg.position]))
        w, x, y, z = cfg.orientation_wxyz
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(w), float(x), float(y), float(z)))

    # ------------------------------------------------------------------ robot

    def _author_robot(self) -> None:
        robot = self.stage.DefinePrim("/World/sim/robot")
        robot.GetReferences().AddReference(self._rel(self.scene.robot_usd))
        # Rendering-only: the "None" physics variant references the visual base
        # layer and deactivates the imported joints.
        vsets = robot.GetVariantSets()
        for vset_name in ("Physics", "Sensor", "Robot"):
            if vsets.HasVariantSet(vset_name):
                vset = vsets.GetVariantSet(vset_name)
                if "None" in vset.GetVariantNames():
                    vset.SetVariantSelection("None")

        wrist = self.scene.wrist_camera
        if wrist is not None:
            cam_path = f"/World/sim/robot/{wrist.parent_link}/{_sanitize(wrist.name)}"
            cam = self._author_camera_prim(cam_path, wrist)
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in wrist.eye]))
            w, x, y, z = wrist.local_orientation_wxyz()
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))

    def _link_op(self, link_name: str) -> UsdGeom.XformOp:
        op = self.robot_link_ops.get(link_name)
        if op is None:
            prim = self.stage.OverridePrim(f"/World/sim/robot/{link_name}")
            op = UsdGeom.Xformable(prim).MakeMatrixXform()
            self.robot_link_ops[link_name] = op
        return op

    # ---------------------------------------------------------------- objects

    def _style_for_label(self, label: str, shape_color) -> ObjectStyle:
        for prefix, style in self.scene.object_styles.items():
            if label.startswith(prefix):
                return style
        base = self.scene.default_object_style
        return ObjectStyle(color=tuple(float(c) for c in shape_color), roughness=base.roughness, metallic=base.metallic)

    def _material(self, style: ObjectStyle) -> UsdShade.Material:
        key = (tuple(round(float(c), 4) for c in style.color), round(style.roughness, 4), round(style.metallic, 4))
        if key in self._materials:
            return self._materials[key]
        idx = len(self._materials)
        mat = UsdShade.Material.Define(self.stage, f"/World/Looks/object_style_{idx}")
        shader = UsdShade.Shader.Define(self.stage, f"/World/Looks/object_style_{idx}/preview")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in style.color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(style.roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(style.metallic))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._materials[key] = mat
        return mat

    def _author_shape_geom(self, parent_path: str, name: str, shape_type: int, scale, source) -> UsdGeom.Gprim | None:
        path = f"{parent_path}/{name}"
        if shape_type == int(newton.GeoType.CAPSULE):
            cap = UsdGeom.Capsule.Define(self.stage, path)
            cap.CreateAxisAttr(UsdGeom.Tokens.z)
            cap.CreateRadiusAttr(float(scale[0]))
            cap.CreateHeightAttr(2.0 * float(scale[1]))
            return cap
        if shape_type == int(newton.GeoType.SPHERE):
            sph = UsdGeom.Sphere.Define(self.stage, path)
            sph.CreateRadiusAttr(float(scale[0]))
            return sph
        if shape_type == int(newton.GeoType.BOX):
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(2.0)  # unit cube; half-extents go into the prim matrix
            return cube
        if shape_type == int(newton.GeoType.CYLINDER):
            cyl = UsdGeom.Cylinder.Define(self.stage, path)
            cyl.CreateAxisAttr(UsdGeom.Tokens.z)
            cyl.CreateRadiusAttr(float(scale[0]))
            cyl.CreateHeightAttr(2.0 * float(scale[1]))
            return cyl
        if shape_type == int(newton.GeoType.MESH) and source is not None:
            mesh = UsdGeom.Mesh.Define(self.stage, path)
            verts = np.asarray(source.vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[None, :]
            tris = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.flatten()))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            return mesh
        print(f"[robolabViz] Skipping shape {name}: unsupported geo type {shape_type}.")
        return None

    def _author_objects(self, model: newton.Model) -> None:
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_color = np.asarray(model.shape_color.numpy() if hasattr(model.shape_color, "numpy") else model.shape_color)
        labels = list(model.shape_label)
        body_labels = list(model.body_label)

        UsdGeom.Xform.Define(self.stage, "/World/sim/objects")
        body_prims: dict[int, str] = {}

        for s in range(model.shape_count):
            label = labels[s] or f"shape_{s}"
            if not (int(shape_flags[s]) & int(newton.ShapeFlags.VISIBLE)):
                continue
            if any(label.startswith(p) for p in self.scene.skip_shape_label_prefixes):
                continue

            body = int(shape_body[s])
            if body >= 0:
                parent_path = body_prims.get(body)
                if parent_path is None:
                    body_name = _sanitize(body_labels[body] or f"body_{body}")
                    parent_path = f"/World/sim/objects/{body_name}"
                    xform = UsdGeom.Xform.Define(self.stage, parent_path)
                    self.body_ops[body] = xform.MakeMatrixXform()
                    body_prims[body] = parent_path
            else:
                parent_path = "/World/sim/objects"

            geom = self._author_shape_geom(parent_path, _sanitize(label), int(shape_type[s]), shape_scale[s], model.shape_source[s])
            if geom is None:
                continue
            tf = shape_transform[s]
            local = _matrix_from_pq(tf[:3], tf[3:7])
            if int(shape_type[s]) == int(newton.GeoType.BOX):
                scale_m = Gf.Matrix4d()
                scale_m.SetScale(Gf.Vec3d(float(shape_scale[s][0]), float(shape_scale[s][1]), float(shape_scale[s][2])))
                local = scale_m * local
            UsdGeom.Xformable(geom.GetPrim()).MakeMatrixXform().Set(local)
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                self._material(self._style_for_label(label, shape_color[s]))
            )

        if model.particle_count and model.tri_count:
            tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32)
            mesh = UsdGeom.Mesh.Define(self.stage, "/World/sim/soft_body")
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.flatten()))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self._material(self.scene.soft_body_style))
            self.soft_mesh = mesh
            self._soft_tris = tris

    # ----------------------------------------------------------------- frames

    def write_frame(
        self,
        frame: int,
        robot_link_tfs: dict[str, np.ndarray] | None,
        object_body_q: np.ndarray,
        particle_q: np.ndarray | None,
    ) -> None:
        t = Usd.TimeCode(frame)

        if robot_link_tfs is not None:
            for link, tf in robot_link_tfs.items():
                self._link_op(link).Set(_matrix_from_pq(tf[:3], tf[3:7]), t)

        for body, op in self.body_ops.items():
            tf = object_body_q[body]
            op.Set(_matrix_from_pq(tf[:3], tf[3:7]), t)

        if self.soft_mesh is not None and particle_q is not None:
            pts = np.ascontiguousarray(particle_q, dtype=np.float32)
            self.soft_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts), t)
            normals = _vertex_normals(pts, self._soft_tris)
            self.soft_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)), t)

        self._frames_written += 1

    def save(self) -> None:
        if self._frames_written:
            self.stage.SetEndTimeCode(self._frames_written - 1)
        self.stage.GetRootLayer().Save()
        print(f"[robolabViz] Saved {self._frames_written} frames to {self.output_path}")
