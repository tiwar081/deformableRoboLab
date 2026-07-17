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
from dataclasses import dataclass, field, fields
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
    # When False the camera is still rendered (and dumped as PNG frames) but is
    # left out of the side-by-side simulation.mp4 — e.g. an extra object-inspection
    # still view that should not change the kept observation video.
    in_combined_video: bool = True


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
    occlusion sweeps (see ``robolabViz.rerender``): in the fr3_hand frame
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
    # Optional base-color image (sRGB PNG/JPG) the raycast preview planar-maps
    # onto the fixture's bounding box (top + sides), reproducing the wood-grain
    # look of the RTX MDL material. When None the flat ``preview_color`` is used.
    # ``texture_uv_scale`` is world meters per texture tile (smaller = more tiles).
    texture_file: Path | None = None
    texture_uv_scale: float = 0.5
    # Optional tangent-space normal map + ORM (occlusion/roughness/metallic) map,
    # planar-mapped alongside ``texture_file``. Only the advanced (PBR) render
    # tier samples these; the flat preview ignores them.
    normal_file: Path | None = None
    orm_file: Path | None = None


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
    bottom_z: float | None = None,
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

    ``bottom_z`` (optional) additionally solves the z-scale so the asset's
    UNDERSIDE lands at ``bottom_z``: the workspace footprint overlaps the
    franka_stand in x/y, so a full-height table asset (native ~0.7 m tall)
    pierces the stand's body — its underside must sit ABOVE the stand's top
    (the robot-base plane, viz z=0) or the two fixtures visibly interpenetrate.
    Objects/physics are untouched: the visible top stays pinned at ``top_z``.
    """
    if bottom_z is not None:
        mn0, mx0 = _asset_world_extents(usd_path, rotate_z_deg, scale)
        native_h = float(mx0[2] - mn0[2])
        if top_z - bottom_z <= 0 or native_h <= 0:
            raise ValueError(f"table fixture: need top_z ({top_z}) > bottom_z ({bottom_z}).")
        scale = (scale[0], scale[1], scale[2] * (top_z - bottom_z) / native_h)
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


# ---------------------------------------------------------------- asset registries
#
# Every asset below resolves only from the vendored ``assets/`` tree — nothing
# in this package reads ``_external/`` (assume that checkout can be deleted).

# Vendored work tables (RoboLab fixtures): name -> (fixture usd, top-surface
# material directory relative to ``assets/`` holding ``<stem>_BaseColor.png``
# (+ ``_N.png`` normal and ``_ORM.png`` roughness maps, sampled by the advanced
# PBR tier only), flat fallback color). The four tables share identical geometry
# and differ only in the top-slab material; the raycast preview planar-maps the
# base-color image onto the table block while the RTX/USD path uses the table's
# own MDL material. ``black`` is a flat matte paint (no wood texture).
TABLE_TEXTURES: dict[str, tuple[str, str | None, tuple[float, float, float]]] = {
    "maple": ("table_maple.usda", "materials/Base/Wood/Walnut_Planks", (0.45, 0.31, 0.19)),
    "oak": ("table_oak.usda", "materials/Base/Wood/Oak", (0.62, 0.46, 0.29)),
    "bamboo": ("table_bamboo.usda", "materials/Base/Wood/Bamboo", (0.74, 0.60, 0.36)),
    "black": ("table_black.usda", None, (0.05, 0.05, 0.06)),
}


def available_tables() -> list[str]:
    """Names of the vendored work-table textures (``--table`` choices)."""
    return [name for name in TABLE_TEXTURES if (ASSETS_DIR / "fixtures" / TABLE_TEXTURES[name][0]).exists()]


def work_table_fixture(
    table: str,
    *,
    top_z: float,
    center_xy: tuple[float, float],
    rotate_z_deg: float = 90.0,
    bottom_z: float | None = 0.002,
) -> FixtureConfig:
    """Build the ``work_table`` fixture for a named vendored table, placed from
    the physics footprint (see ``table_fixture_from_footprint``). The default
    ``bottom_z`` (2 mm above the robot-base plane) lifts the table's underside
    clear of the franka_stand so the two fixtures never interpenetrate — the
    visible table becomes the tabletop slab itself, matching the physics slab."""
    if table not in TABLE_TEXTURES:
        raise ValueError(f"Unknown table {table!r}; available: {available_tables()}")
    usd_name, mat_rel, color = TABLE_TEXTURES[table]
    fix = table_fixture_from_footprint(
        name="work_table",
        usd_path=ASSETS_DIR / "fixtures" / usd_name,
        top_z=top_z,
        center_xy=center_xy,
        rotate_z_deg=rotate_z_deg,
        bottom_z=bottom_z,
        preview_color=color,
    )
    if mat_rel is not None:
        mat_dir = ASSETS_DIR / mat_rel
        stem = mat_dir.name
        fix.texture_file = mat_dir / f"{stem}_BaseColor.png"
        fix.normal_file = mat_dir / f"{stem}_N.png"
        fix.orm_file = mat_dir / f"{stem}_ORM.png"
    return fix


def available_backgrounds() -> dict[str, Path]:
    """Map background name (filename stem) -> vendored ``.hdr``/``.exr`` path,
    searched recursively under ``assets/backgrounds`` (``--background`` choices)."""
    out: dict[str, Path] = {}
    bg_dir = ASSETS_DIR / "backgrounds"
    if bg_dir.exists():
        for p in sorted(bg_dir.rglob("*")):
            if p.suffix.lower() in (".hdr", ".exr"):
                out.setdefault(p.stem, p)
    return out


def resolve_background(name_or_path: str | Path) -> Path:
    """Resolve a background by stem (``home_office``, ``garage_2k``, or the
    ``_2k``-less ``garage``) or by direct path. Vendored assets only."""
    p = Path(name_or_path)
    if p.suffix.lower() in (".hdr", ".exr") and p.exists():
        return p
    name = str(name_or_path)
    bgs = available_backgrounds()
    for key in (name, f"{name}_2k"):
        if key in bgs:
            return bgs[key]
    matches = [v for k, v in bgs.items() if name in k]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Unknown background {name_or_path!r}; available: {sorted(bgs)}"
        + (f" (ambiguous: {[m.stem for m in matches]})" if matches else "")
    )


def background_dome(name_or_path: str | Path = "home_office", intensity: float = 500.0) -> DomeLightConfig:
    """RoboLab-style dome light from a vendored background (the equirectangular
    HDR is both the image-based light and the visible backdrop)."""
    return DomeLightConfig(
        texture_file=resolve_background(name_or_path),
        intensity=intensity,
        texture_format="latlong",
        visible_in_primary_ray=True,
    )


@dataclass
class ObjectAsset:
    """A vendored RoboLab object USD (textured mesh) with catalog metadata.

    These are scene props (apple, banana, rubiks_cube, bowl, mug) brought into
    the repo for future demos; ``usd_path`` is self-contained (mesh + local
    ``textures/``). ``dims`` is the asset's axis-aligned size in meters.
    """

    name: str
    usd_path: Path
    dims: tuple[float, float, float]
    object_class: str = ""
    description: str = ""


def _object_catalog() -> dict[str, ObjectAsset]:
    import json

    catalog_path = ASSETS_DIR / "objects" / "object_catalog.json"
    if not catalog_path.exists():
        return {}
    repo_root = ASSETS_DIR.parent
    out: dict[str, ObjectAsset] = {}
    for entry in json.loads(catalog_path.read_text()):
        usd = repo_root / entry["usd_path"]
        if not usd.exists():
            continue
        dims = tuple(float(d) for d in entry.get("dims") or (0.0, 0.0, 0.0))
        out[entry["name"]] = ObjectAsset(
            name=entry["name"],
            usd_path=usd,
            dims=dims,  # type: ignore[arg-type]
            object_class=entry.get("class", ""),
            description=entry.get("description", ""),
        )
    return out


def available_objects() -> list[str]:
    """Names of the vendored RoboLab object assets (see ``object_asset``)."""
    return sorted(_object_catalog())


def object_asset(name: str) -> ObjectAsset:
    """Look up a vendored RoboLab object by catalog name."""
    catalog = _object_catalog()
    if name not in catalog:
        raise ValueError(f"Unknown object {name!r}; available: {sorted(catalog)}")
    return catalog[name]


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


@dataclass
class RenderQuality:
    """Raycast-renderer tier knobs: which features are on and how many samples they get.

    ``RenderQuality()`` (all defaults) reproduces the historical preview exactly —
    HDRI backdrop + table texture with the flat shade — so call sites that predate
    the render tiers (``robolabViz.rerender``) behave unchanged. The runner builds
    a tier with :meth:`for_mode` from ``--output-style``; a demo overrides single
    knobs via ``RenderSpec.quality``.
    """

    use_hdri: bool = True          # decode the dome EXR/HDR (backdrop; + IBL when use_pbr)
    use_textures: bool = True      # fixture planar maps + catalog object UV textures
    use_pbr: bool = False          # advanced kernel: GGX + shadows + IBL + ACES tone map
    smooth_normals: bool = False   # per-vertex normals recomputed each frame (advanced)
    aa_samples: int = 1            # primary rays per pixel (sample 0 is always the pixel center)
    shadow_samples: int = 0        # sphere-light shadow rays per primary sample
    ibl_key_lights: int = 0        # importance-sampled dome "suns", shadow-tested (1 ray each/sample)
    video_scale: float = 0.5       # per-camera scale when concatenating into simulation.mp4
    fps: float | None = None       # video fps override (None = scene fps)
    tex_tile: int = 1024           # square tile size textures are resized to in the GPU atlas
    irradiance_size: tuple[int, int] = (64, 32)   # (W, H) cosine-convolved dome (diffuse IBL)
    env_spec_size: tuple[int, int] = (256, 128)   # (W, H) of each prefiltered specular dome level
    sphere_light_scale: float = 3.0  # sphere-light radiance in auto-exposure-normalized units
    noise_seed: int = 7919         # fixed, frame-independent RNG seed -> deterministic video

    @staticmethod
    def for_mode(mode: str) -> "RenderQuality":
        """The tier for an ``--output-style``: ``mp4`` = the flat preview without the
        (slow) HDRI decode — fixture textures stay ON so the wrist camera keeps motion
        cues over the table; ``mp4_advanced`` = the full PBR path."""
        if mode == "mp4_advanced":
            return RenderQuality(
                use_pbr=True, smooth_normals=True, aa_samples=4, shadow_samples=2,
                ibl_key_lights=2, video_scale=1.0, tex_tile=2048,
            )
        if mode == "mp4":
            return RenderQuality(use_hdri=False)
        raise ValueError(f"No render tier for output style {mode!r} (expected mp4 or mp4_advanced).")


@dataclass
class RenderSpec:
    """Per-demo visual customization, declared in a DEMO data file via ``DemoSpec.render``.

    Every field defaults to "keep the centralized DROID look" (``droid_scene_config``),
    so a demo only states its deltas. Camera fields apply to BOTH mp4 render modes;
    appearance fields (background, textures, lights, styles) are fully visible only in
    ``mp4_advanced`` — the lightweight ``mp4`` mode strips backgrounds/textures and
    keeps flat base colors. Precedence for fields that also exist as CLI flags
    (``--table``, ``--background``, ``--wrist-eye/-target``): explicit CLI flag >
    this spec > ``settings.yaml``.
    """

    # -- look (droid_scene_config inputs, consumed by the caller BEFORE the factory) --
    background: str | None = None                  # dome HDRI stem/path (resolve_background)
    table: str | None = None                       # TABLE_TEXTURES key (maple/oak/bamboo/black)
    # -- look (scene mutations, applied by :meth:`apply` AFTER the factory) --
    dome_intensity: float | None = None
    sphere_lights: list[SphereLightConfig] | None = None   # REPLACES the default ([] = none)
    object_styles: dict[str, ObjectStyle] = field(default_factory=dict)  # prefix->style, MERGED over defaults
    soft_body_style: ObjectStyle | None = None
    default_object_style: ObjectStyle | None = None
    extra_fixtures: list[FixtureConfig] = field(default_factory=list)    # visual-only props (appended)
    # -- cameras (both mp4 modes) --
    exterior_cameras: list[CameraConfig] | None = None     # REPLACES the DROID exterior pair
    extra_cameras: list[CameraConfig] = field(default_factory=list)      # appended
    preview_cameras: list[str] | None = None       # names rendered per frame (default: the runner's
                                                   # SCENIC_CAMERAS); each joins simulation.mp4 iff its
                                                   # in_combined_video flag is True
    wrist_camera: WristCameraConfig | None = None  # full replace (parent_link still robot-retargeted)
    wrist_eye: tuple[float, float, float] | None = None    # pose-only tweaks of the default wrist cam
    wrist_target: tuple[float, float, float] | None = None
    # -- advanced-tier knobs --
    quality: "RenderQuality | dict | None" = None  # dict = per-field overrides of the mode tier;
                                                   # a RenderQuality instance replaces the tier wholesale

    def apply(self, scene: RoboLabSceneConfig) -> None:
        """Mutate a freshly built scene config with this demo's deltas.

        ``background``/``table`` are NOT applied here — they parameterize
        ``droid_scene_config`` itself, so the caller resolves them first.
        """
        if self.dome_intensity is not None:
            scene.dome_light.intensity = float(self.dome_intensity)
        if self.sphere_lights is not None:
            scene.sphere_lights = list(self.sphere_lights)
        scene.object_styles.update(self.object_styles)
        if self.soft_body_style is not None:
            scene.soft_body_style = self.soft_body_style
        if self.default_object_style is not None:
            scene.default_object_style = self.default_object_style
        scene.fixtures.extend(self.extra_fixtures)
        if self.exterior_cameras is not None:
            scene.exterior_cameras = list(self.exterior_cameras)
        scene.exterior_cameras.extend(self.extra_cameras)
        if self.wrist_camera is not None:
            scene.wrist_camera = self.wrist_camera
        if scene.wrist_camera is not None:
            if self.wrist_eye is not None:
                scene.wrist_camera.eye = tuple(self.wrist_eye)
            if self.wrist_target is not None:
                scene.wrist_camera.target = tuple(self.wrist_target)

    def resolve_quality(self, base: RenderQuality) -> RenderQuality:
        """The mode tier ``base`` with this spec's ``quality`` overrides applied."""
        if self.quality is None:
            return base
        if isinstance(self.quality, RenderQuality):
            return self.quality
        from dataclasses import replace

        unknown = set(self.quality) - {f.name for f in fields(RenderQuality)}
        if unknown:
            raise ValueError(f"RenderSpec.quality has unknown knobs {sorted(unknown)}.")
        return replace(base, **self.quality)


def droid_scene_config(
    table_top_z: float = 0.0,
    table_center_xy: tuple[float, float] = (0.57, 0.0),
    table: str = "maple",
    background: str = "home_office",
    robot_yaw_deg: float = 0.0,
) -> RoboLabSceneConfig:
    """RoboLab ``run_recorded.py`` scene: DROID cameras, dome background, work table.

    Values are copied from RoboLab's ``robolab/variations/{camera,lighting,
    backgrounds}.py`` and ``assets/scenes/*.usda``.

    ``table`` selects a vendored work-table texture (``maple``/``oak``/``bamboo``/
    ``black`` — see ``available_tables``); ``background`` selects a vendored dome
    HDR/EXR (``home_office``, ``garage_2k``, ... — see ``available_backgrounds``).
    Both also drive the raycast preview (wood-grain table, equirectangular
    backdrop) so the choice is visible without an RTX renderer.

    The work table is placed by ``table_fixture_from_footprint`` so its visible
    top surface sits at ``table_top_z`` and its footprint is centered on
    ``table_center_xy`` — both in the viz frame. The example passes its physics
    table's top height and footprint center here, which is what keeps the
    rendered table consistent with the invisible contact table it replaces
    (objects rest exactly on the visible surface, and the table spans the whole
    contact region). The top (0.7 x 1.0 m) is yawed 90 deg so its 1.0 m edge
    runs along +X, covering this repo's 0.9 x 0.7 m physics footprint.
    """
    # The franka_stand is ROBOT-relative (the viz frame is the base frame, translation-only): when
    # the physics base is yawed (agentic_pipeline robot placement), the stand's offset and rotation
    # must follow the base yaw or the pedestal renders detached from the rotated robot.
    yaw = math.radians(robot_yaw_deg)
    stand_dx, stand_dy = (-0.087 * math.cos(yaw), -0.087 * math.sin(yaw))
    return RoboLabSceneConfig(
        fixtures=[
            FixtureConfig(
                name="franka_stand",
                usd_path=ASSETS_DIR / "fixtures" / "franka_table.usd",
                translate=(stand_dx, stand_dy, 0.0),
                rotate_z_deg=180.0 + robot_yaw_deg,
                preview_color=(0.35, 0.35, 0.38),
            ),
            work_table_fixture(
                table,
                top_z=table_top_z,
                center_xy=table_center_xy,
                rotate_z_deg=90.0,
            ),
        ],
        dome_light=background_dome(background, intensity=500.0),
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
