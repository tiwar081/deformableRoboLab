"""Scene configuration for the RoboLab-style visualization layer.

``droid_scene_config()`` reproduces the exact scene used by RoboLab's
``examples/run_recorded.py`` (DROID rig): same fixtures, lighting, background,
and camera intrinsics/poses. Everything is a plain dataclass field so demos
can swap tables, lights, cameras, or object styling without touching the
stage/render code.

Conventions:

- The *viz frame* is RoboLab's world frame: robot base at the origin, +X in
  front of the robot, Z up, meters. Cameras/lights/fixtures are configured in
  this frame, verbatim from RoboLab.
- The *sim frame* is whatever the physics example uses. The example provides
  ``sim_to_viz_translation`` (viz = sim + offset); the demos keep the robot
  axis-aligned so a translation is sufficient. All physics state stays in sim
  coordinates and is re-rooted under one Xform in the USD stage.
- Quaternions are (w, x, y, z) like USD's ``Gf.Quat*``. Camera orientation
  uses the USD/OpenGL convention: -Z is the view direction, +Y is up in the
  image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def look_at_quat_wxyz(eye, target, up) -> tuple[float, float, float, float]:
    """Orientation (w,x,y,z) for a -Z-forward/+Y-up camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1.0e-6:
        raise ValueError("Camera up vector is parallel to the view direction.")
    right /= norm
    true_up = np.cross(right, forward)
    # Columns are the camera basis vectors in the parent frame: X=right, Y=up, Z=-forward.
    m = np.stack([right, true_up, -forward], axis=1)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k], 0.0) + 1.0) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        w = (m[k, j] - m[j, k]) / s
        x, y, z = q
    return (float(w), float(x), float(y), float(z))


@dataclass
class CameraConfig:
    """A camera fixed in the viz (world) frame.

    Default intrinsics are RoboLab's DROID exterior cameras: focal 2.1 with a
    5.376 x 3.024 aperture (~104 deg horizontal FOV) at 1280x720.
    """

    name: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    width: int = 1280
    height: int = 720
    focal_length: float = 2.1
    horizontal_aperture: float = 5.376
    vertical_aperture: float = 3.024
    focus_distance: float = 28.0
    clipping: tuple[float, float] = (0.01, 1.0e4)


@dataclass
class WristCameraConfig:
    """A camera rigidly mounted on a robot link, expressed as look-at in the link frame.

    ``eye`` is the camera position in the parent link's frame; the camera looks
    at ``target`` with ``up`` as the image-up hint.

    Provenance. The *intrinsics* are RoboLab's wrist camera verbatim: focal 2.8,
    aperture 5.376 x 3.024 (~88 deg horizontal FOV) — from ``_WRIST_CAM`` in
    ``_external/RoboLab/robolab/robots/droid.py``. The *pose*, however, is NOT
    RoboLab's: RoboLab mounts that camera on its Robotiq 2F-85 gripper's
    ``base_link`` with offset ``pos=(0.011, -0.031, -0.074),
    rot=(-0.420, 0.570, 0.576, -0.409)`` (opengl). RoboLab's robot is a Franka +
    Robotiq 2F-85 (``franka_robotiq_2f_85_flattened.usd``); this repo simulates
    the Franka fr3 two-finger hand, a different gripper (Robotiq base->fingertip
    is 162.8 mm vs the fr3 hand TCP at ~103 mm), so that offset frame doesn't
    exist here and its numbers don't transfer. We keep RoboLab's mount *concept*
    (rigidly fixed near the gripper, looking down over the fingers toward the
    grasp) and solve the concrete fr3_hand-frame pose ourselves via raycast
    occlusion sweeps (see ``robolab_viz.rerender``): in the fr3_hand frame
    (+z toward the fingertips, +x the hand's front face), the eye sits 8 cm in
    front of the housing just above its top edge — high enough to see over the
    housing, low enough that the fat fr3_link7/link8 wrist cylinders stay out of
    frame — looking down past the fingertips. Fingers + grasp center the frame;
    the housing reads as a strip along the bottom (~29% mean robot coverage over
    the cable demo, vs 45% for a flush side mount and full blockage for mounts
    above the wrist cylinders).
    """

    name: str = "wrist_camera"
    parent_link: str = "fr3_hand"
    eye: tuple[float, float, float] = (0.08, 0.0, -0.025)
    target: tuple[float, float, float] = (0.0, 0.0, 0.18)
    up: tuple[float, float, float] = (1.0, 0.0, 0.0)
    width: int = 1280
    height: int = 720
    focal_length: float = 2.8
    horizontal_aperture: float = 5.376
    vertical_aperture: float = 3.024
    focus_distance: float = 28.0
    clipping: tuple[float, float] = (0.005, 1.0e4)

    def local_orientation_wxyz(self) -> tuple[float, float, float, float]:
        return look_at_quat_wxyz(self.eye, self.target, self.up)


@dataclass
class FixtureConfig:
    """A static USD asset placed in the viz frame (table, pedestal, ...)."""

    name: str
    usd_path: Path
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_z_deg: float = 0.0
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Child-prim translate overrides, replicating RoboLab's scene-level `over`s.
    # WARNING: overriding a child re-positions the geometry relative to the
    # asset's authored layout; pair it with a compensating ``translate`` or the
    # fixture will shift off its footprint. Prefer ``table_fixture_from_footprint``
    # which solves the translate from the asset bbox instead.
    child_translate_overrides: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Approximate the fixture with its bounding box in the raycast preview.
    preview: bool = True
    preview_color: tuple[float, float, float] = (0.55, 0.42, 0.30)


def _asset_world_extents(
    usd_path: Path, rotate_z_deg: float, scale: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bbox of an asset's default prim after scale + Z-rotation (pre-translate)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetDefaultPrim()
    ).ComputeAlignedRange()
    mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    corners = corners * np.asarray(scale, dtype=np.float64)
    a = math.radians(rotate_z_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    corners = corners @ rot.T
    return corners.min(axis=0), corners.max(axis=0)


def fixture_world_bbox(fixture: FixtureConfig) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bbox of a placed fixture: the asset bbox after
    scale + Z-rotation, then translate. (Ignores ``child_translate_overrides``.)
    Used to check a visual fixture against a physics footprint without needing
    the full USD stage."""
    mn, mx = _asset_world_extents(fixture.usd_path, fixture.rotate_z_deg, fixture.scale)
    t = np.asarray(fixture.translate, dtype=np.float64)
    return mn + t, mx + t


def table_fixture_from_footprint(
    name: str,
    usd_path: Path,
    *,
    top_z: float,
    center_xy: tuple[float, float],
    rotate_z_deg: float = 0.0,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    preview_color: tuple[float, float, float] = (0.55, 0.42, 0.30),
) -> FixtureConfig:
    """Place a tabletop asset so its *visible top surface coincides with a
    physics contact plane* and its footprint is centered on the workspace.

    The translate is solved from the asset's composed bounding box (after
    ``scale`` and ``rotate_z_deg``) so that the bbox top lands exactly at
    ``top_z`` and the bbox is centered at ``center_xy`` — both in the viz frame.
    This is how the example keeps the rendered table consistent with the
    invisible physics table it replaces: pass the physics table's top height
    and footprint center, and the visual table lines up with where objects
    actually rest. The asset must be a tabletop whose top face is its bbox max.
    """
    mn, mx = _asset_world_extents(usd_path, rotate_z_deg, scale)
    center = 0.5 * (mn + mx)
    return FixtureConfig(
        name=name,
        usd_path=usd_path,
        translate=(float(center_xy[0] - center[0]), float(center_xy[1] - center[1]), float(top_z - mx[2])),
        rotate_z_deg=rotate_z_deg,
        scale=scale,
        preview_color=preview_color,
    )


@dataclass
class DomeLightConfig:
    texture_file: Path | None = None
    intensity: float = 500.0
    texture_format: str = "latlong"
    visible_in_primary_ray: bool = True


@dataclass
class SphereLightConfig:
    name: str = "sphere_light"
    position: tuple[float, float, float] = (0.0, -0.6, 0.7)
    intensity: float = 5000.0
    radius: float = 0.5
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    exposure: float = 0.0


@dataclass
class ObjectStyle:
    """UsdPreviewSurface material applied to auto-generated object prims."""

    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = 0.6
    metallic: float = 0.0


@dataclass
class RoboLabSceneConfig:
    robot_usd: Path | None = None
    robot_root_prim: str = "fr3"
    # viz = sim + sim_to_viz_translation; the example fills this in so the
    # robot base lands at the viz-frame origin like RoboLab's DROID rig.
    sim_to_viz_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixtures: list[FixtureConfig] = field(default_factory=list)
    dome_light: DomeLightConfig = field(default_factory=DomeLightConfig)
    sphere_lights: list[SphereLightConfig] = field(default_factory=list)
    exterior_cameras: list[CameraConfig] = field(default_factory=list)
    wrist_camera: WristCameraConfig | None = None
    # Physics shapes whose labels start with any of these are not drawn (the
    # physics table is replaced by the visual work-table fixture; proxies are
    # contact bridges, never rendered).
    skip_shape_label_prefixes: tuple[str, ...] = ("table", "robot_contact_table", "left_gripper", "right_gripper")
    # Styling for auto-generated object prims, keyed by shape-label prefix.
    object_styles: dict[str, ObjectStyle] = field(default_factory=dict)
    soft_body_style: ObjectStyle = field(default_factory=lambda: ObjectStyle(color=(0.83, 0.72, 0.55), roughness=0.9))
    default_object_style: ObjectStyle = field(default_factory=ObjectStyle)
    fps: float = 60.0

    def cameras(self) -> list[CameraConfig | WristCameraConfig]:
        cams: list[CameraConfig | WristCameraConfig] = list(self.exterior_cameras)
        if self.wrist_camera is not None:
            cams.append(self.wrist_camera)
        return cams


def droid_scene_config(
    table_top_z: float = 0.0,
    table_center_xy: tuple[float, float] = (0.57, 0.0),
) -> RoboLabSceneConfig:
    """RoboLab ``run_recorded.py`` scene: DROID cameras, home-office dome, maple table.

    Values are copied from RoboLab's ``robolab/variations/{camera,lighting,
    backgrounds}.py`` and ``assets/scenes/*.usda``.

    The work table is placed by ``table_fixture_from_footprint`` so its visible
    top surface sits at ``table_top_z`` and its footprint is centered on
    ``table_center_xy`` — both in the viz frame. The example passes its physics
    table's top height and footprint center here, which is what keeps the
    rendered table consistent with the invisible contact table it replaces
    (objects rest exactly on the visible surface, and the table spans the whole
    contact region). The maple top (0.7 x 1.0 m) is yawed 90 deg so its 1.0 m
    edge runs along +X, covering this repo's 0.9 x 0.7 m physics footprint.
    """
    return RoboLabSceneConfig(
        fixtures=[
            FixtureConfig(
                name="franka_stand",
                usd_path=ASSETS_DIR / "fixtures" / "franka_table.usd",
                translate=(-0.087, 0.0, 0.0),
                rotate_z_deg=180.0,
                preview_color=(0.35, 0.35, 0.38),
            ),
            table_fixture_from_footprint(
                name="work_table",
                usd_path=ASSETS_DIR / "fixtures" / "table_maple.usda",
                top_z=table_top_z,
                center_xy=table_center_xy,
                rotate_z_deg=90.0,
                preview_color=(0.55, 0.42, 0.30),
            ),
        ],
        dome_light=DomeLightConfig(
            texture_file=ASSETS_DIR / "backgrounds" / "home_office.exr",
            intensity=500.0,
            texture_format="latlong",
            visible_in_primary_ray=True,
        ),
        sphere_lights=[
            SphereLightConfig(name="sphere_light", position=(0.0, -0.6, 0.7), intensity=5000.0, radius=0.5),
        ],
        exterior_cameras=[
            # DROID exterior_image_1_left placement (RoboLab OverShoulderLeftCameraCfg).
            CameraConfig(
                name="over_shoulder_left_camera",
                position=(0.05, 0.57, 0.66),
                orientation_wxyz=(-0.393, -0.195, 0.399, 0.805),
            ),
            # Viewport/video camera (RoboLab EgocentricMirroredCameraCfg).
            CameraConfig(
                name="egocentric_mirrored_camera",
                position=(1.5, 0.0, 1.0),
                orientation_wxyz=(0.653, 0.271, 0.271, 0.653),
                width=864,
                height=480,
                focal_length=24.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.29,
                focus_distance=400.0,
            ),
        ],
        wrist_camera=WristCameraConfig(),
        object_styles={
            # Jacketed cable: near-black rubber.
            "vbd_cable": ObjectStyle(color=(0.07, 0.07, 0.08), roughness=0.55),
        },
    )
