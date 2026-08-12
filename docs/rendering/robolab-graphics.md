# RoboLab Graphics (`robolabViz/`)

An alternative renderer reproducing NVIDIA RoboLab's look (the DROID rig in
`_external/RoboLab/examples/run_recorded.py`: home-office dome, sphere key light, maple work
table, Franka pedestal, over-shoulder + wrist cameras) on top of the *unchanged* Newton
physics. **Self-contained** — never imports the `robolab` package or `isaaclab`; RoboLab's
fixture/material/background assets are vendored into `assets/` (provenance in
`assets/ATTRIBUTION.md`). **Hard rule: nothing in the repo reads `_external/` at runtime —
assume that checkout can be deleted.**

## Usage

- Rendering is built into **every** demo via `--output-style` — there are no separate `_robolab`
  files. Three styles, all writing into `outputs/<robot>/<name>/` (`<robot>` = the active robot's
  `FRANKA.short_name`, so the two robots' outputs never collide); default from `settings.yaml`
  `render.style` (ships as `mp4`); deprecated aliases `basic`→`usd`, `scenic`→`mp4_advanced`:

  | style | renderer | artifacts |
  |---|---|---|
  | `usd` | Newton `ViewerUSD` (lightest) | `<name>.usd` (time-sampled) |
  | `mp4` (default) | warp raycast, **lightweight tier**: flat shade with fixed world lights, no HDRI decode; fixture textures kept + a world-space speckle on untextured fixtures so the wrist camera always has motion cues | `simulation.mp4` (two cams half-scale, ~1280×360) |
  | `mp4_advanced` | warp raycast, **PBR tier**: HDRI image-based lighting + dome "suns", soft sphere-light shadows, GGX materials, textured catalog objects, wood normal/ORM maps, 4× jittered AA, ACES | `simulation_advanced.mp4` (full-res, 2560×720) + `frames/` stills + `wrist_coverage.json` |

  `--output graphs|both` (gripper physics PNG) works in every style. Both mp4 tiers force
  `--viewer null` internally; the two tiers are two kernels behind one `RenderQuality` config
  (`config.RenderQuality.for_mode`), so the flat tier is byte-identical to the historical preview.
- The shared glue is `robolabViz.scenic.ScenicGraspExample` (a `GraspExample` subclass each demo
  inherits). It reads the robot base pose (`robot_base_xform`), table (`table_pos/table_half`),
  and optional `soft_start_pos` straight off the physics example, so a new demo needs no
  per-demo viz config. This is the one place `robolabViz` imports `deformableManipulationTools`
  (kept out of `robolabViz/__init__.py` so plain `import robolabViz` stays Newton-free).
- **Per-demo customization: `DemoSpec.render = robolabViz.RenderSpec(...)`** (the activation
  mechanism is centralized in `scenic.py`; the spec itself lives in the demo data file). Every
  field defaults to the central DROID look; precedence for fields that also exist as CLI flags is
  **CLI flag > RenderSpec > settings.yaml**. Camera fields apply to both mp4 tiers; appearance is
  fully visible in `mp4_advanced`. The full surface:

  ```python
  from robolabViz import CameraConfig, ObjectStyle, RenderQuality, RenderSpec, SphereLightConfig
  DEMO = DemoSpec(..., render=RenderSpec(
      background="garage",                 # dome HDRI (resolve_background stem)
      table="black",                       # work-table material (TABLE_TEXTURES key)
      dome_intensity=800.0,                # dome brightness (default 500)
      sphere_lights=[SphereLightConfig(position=(0.4, -0.5, 0.9), intensity=8000.0)],  # replaces
      object_styles={"cube": ObjectStyle(color=(0.16, 0.42, 0.85), roughness=0.35)},   # merges
      soft_body_style=ObjectStyle(color=(0.83, 0.35, 0.20), roughness=0.85),
      extra_fixtures=[...],                # visual-only FixtureConfig props
      exterior_cameras=[CameraConfig(...)],# replaces the DROID pair
      extra_cameras=[CameraConfig(...)],   # appends
      preview_cameras=["over_shoulder_left_camera", "wrist_camera"],  # which render per frame
      wrist_eye=(0.08, 0.0, -0.025), wrist_target=(0.0, 0.0, 0.18),   # hand-frame pose tweaks
      quality={"aa_samples": 8, "shadow_samples": 4},  # advanced-tier knob overrides (RenderQuality)
  ))
  ```
  `rigidCube_soft_franka` is the live exemplar (black table + garage + object styles).
  **Lighting is scene-level, never per-camera** (RoboLab's pattern: one env-level lighting cfg,
  every camera renders the same lit scene): lights live only on the scene config
  (`dome_light` + `sphere_lights`), cameras carry zero light state, and every light direction in
  both kernels is a fixed world quantity — so a `RenderSpec`/CLI lighting tweak always moves all
  camera views together, and nothing can produce a light that follows the wrist camera. (The two
  physically view-dependent terms — GGX speculars and grazing fresnel — are area-light-widened
  and roughness-attenuated respectively so they read as sheen, not as a camera flashlight.)
- `--table {maple,oak,bamboo,black}`, `--background <name>` (e.g. `home_office`, `garage_2k`,
  `machine_shop_01_2k`). `available_tables()/_backgrounds()/_objects()` enumerate the vendored
  sets; `resolve_background` accepts the `_2k`-less stem. Choices render in the raycast
  preview too, not just RTX.
- mp4-style opt-ins: `--usd` (time-sampled RoboLab scene USD at `outputs/<robot>/<name>/<name>.usd`
  — the same slot the `usd` style's plain Newton USD uses), `--npz` (state cache `<name>.state.npz`
  + `geometry.pkl` that `robolabViz.rerender` replays without re-simulating; captured from an
  `mp4`-tier run the pickled geometry is untextured — capture with `mp4_advanced` for full-look
  replays), `--frames-per-image N` (PNG still cadence; default 30 in `mp4_advanced`, off otherwise).
- `--objectview True` adds a fixed `object_view_camera` on the soft body (soft demos only),
  dumped as PNG stills only (`CameraConfig.in_combined_video=False`, so `simulation.mp4` is
  unchanged).

## Architecture

- **Robot (mirrors whichever robot `settings.yaml` selects — reads the active `FRANKA`).** The
  renderer branches on `FRANKA.loader` / `FRANKA.render_from_physics`, so switching `robot:` in
  `settings.yaml` switches the render path with no code change (and re-homes the output under a
  different `short_name`):
  - **`franka_panda_isaacsim` (the DEFAULT, native USD, `render_from_physics=True`)** — rendered/FK'd
    **directly** from `FRANKA.usd_path` (`assets/robots/franka_panda_isaacsim/franka.usd`); the robot
    links are drawn from the simulated meshes (see the rigid-only / `render_from_physics` gotcha below).
  - **`fr3_franka_hand` (URDF, `render_from_physics=False`)** — the `fr3_franka_hand.urdf` is converted
    to USD once via Isaac Sim's importer (`robot_usd.py`, subprocess, cached in `assets/robots/`) and
    puppeteered per frame by an *uncollapsed* FK mirror (`robot_fk.py`, `collapse_fixed_joints=False`) —
    the physics model collapses fixed joints, so the hand link the wrist camera mounts to has no
    simulated body. `test_final` asserts FK/sim parity on shared links.
- **Scene + objects:** time-sampled USD with pure `pxr` (`stage.py`, no kit): fixtures via
  payload; cable/rigid from Newton shapes; soft body as a per-frame deforming mesh. Two render
  paths consume the same USD: `raycast.py` (warp `wp.Mesh` raycaster, flat-shaded but exact —
  default, any CUDA GPU) and `isaac_render.py` (offline RTX). **This box is an H200 (Hopper,
  no RT cores): RTX cannot run here** (crashes at GPU init) — use it only on an RT-capable GPU.
- **Visual table** placed by `config.table_fixture_from_footprint(top_z, center_xy)` from the
  physics `table_pos/table_half`, so the visible surface coincides with the contact plane.
- **Preview materials (flat tier):** each `FixtureConfig.texture_file` (sRGB base-color PNG) is
  planar/triplanar-projected onto the fixture box in world meters (`texture_uv_scale` = m/tile);
  on a ray miss the dome HDR/EXR is tone-mapped (auto-exposed Reinhard + sRGB) and sampled as
  an equirectangular (latlong, Z-up) backdrop. `black` table = matte paint (no texture).
  `geometry.pkl` carries the table texture + dome path so `rerender` reproduces the look.
- **Advanced (PBR) tier** (`mp4_advanced`; a separate kernel in `raycast.py`, no RT cores needed):
  all shading in linear space, ACES-tone-mapped. Lighting = the dome decoded LINEARLY once
  (`ibl.py`): a cosine-convolved irradiance map (diffuse IBL), 3 Gaussian-prefiltered specular
  levels (reflection lookups), and the K brightest dome regions extracted as shadow-tested
  directional "suns"; plus the scene's `SphereLightConfig`s with stratified soft-shadow rays.
  Materials = GGX with `ObjectStyle.roughness/metallic` finally honored; per-frame per-vertex
  smooth normals (hard-edged boxes opt out via `face_smooth`). Textures = the wood tables gain
  `_N`/`_ORM` maps (relief + roughness), and bodies labeled with a vendored-catalog asset name
  (banana, bowl, mug, apple_01, rubiks_cube) render as the asset's textured scan via a
  **visual-only** re-read (`viz_assets.py`, `newton.usd.get_mesh(load_uvs=True)`) — NEVER loaded
  through `mesh_collision.load_usd_mesh`, whose UV-less read must stay untouched (UV loading
  vertex-splits faceVarying assets and would change the coacd input/cache = a physics change).
  Sampling is deterministic: a frame-independent per-pixel RNG (`RenderQuality.noise_seed`) makes
  the residual noise a static grain and the mp4 byte-stable (x264 runs single-threaded).
- **Vendored asset library** (`assets/`): `backgrounds/{default,indoors}/*.hdr|exr` (no outdoor
  env maps exist in RoboLab's set — indoor/default only), `fixtures/table_*.usda` (offline MDL
  refs), `objects/{objaverse,ycb,hot3d}/*.usd` + `textures/` (apple, banana, rubiks_cube, bowl,
  mug) with a trimmed `objects/object_catalog.json` exposed via `config.object_asset(name)`.

## Gotchas (fixed, would recur)

- **DAE up-axis double-correction.** Isaac's importer Y-up→Z-up-rotates every Collada visual,
  but the Franka DAEs already declare `Z_UP`, mis-rotating each visual 90° (closed fingers end
  up sideways). `robot_usd.fix_dae_visual_orientation()` resets the spurious mesh orient to
  identity (FR3 visual origins are identity); runs once post-conversion, idempotent.
- **Wrist camera pose is repo-tuned, not RoboLab's** — RoboLab's offset is for a Robotiq
  gripper this repo doesn't use. Intrinsics are RoboLab's (`_WRIST_CAM`, focal 2.8); pose solved
  in the `fr3_hand` frame by raycast occlusion sweeps (`eye=(0.08,0,-0.025)`, `target=(0,0,0.18)`).
- **Don't use `child_translate_overrides` to re-center a fixture** — it shifts geometry off the
  authored layout; use `table_fixture_from_footprint` instead.
- **Build the raycast `wp.Mesh` on the first frame's real positions**, not the rest buffer
  (soft-body verts = 0 give a degenerate BVH `refit()` can't repair → every ray scans all faces,
  ~3.6 s/frame).
- **Shared `Mesh` BVH segfault:** reusing one `Mesh` across the object AND combined viz model
  crashes (finalizing the viz model rebuilds the shared BVH and frees GPU memory the object
  pipeline points at). The viz model must finalize BEFORE the object model. This is now enforced
  **automatically** by `deformableManipulationTools.framework.GraspExample` (it detects mesh shapes
  and orders the build so the object model owns the live BVH) — no example need handle it.
- **`render_from_physics` on a rigid-only scene emitted scene objects as robot links** (`KeyError`
  at render frame 0, e.g. on `pickplace_ycb_franka`'s bowl). On the rigid-only path the scene's
  objects are MERGED into the robot model, so `raycast._robot_link_meshes_from_model` (which draws the
  robot from its simulated meshes) saw them as links — but they have no FK transform. Fix: it takes
  `object_body_min` (= `object_body_start`) and skips bodies `>= object_body_min`; those render as
  `object_body` instances (posed by body state), the exact complement of `_build_object_instances`'s
  cutoff. Only `render_from_physics=True` robots (the panda) hit it; the USD-mesh path (fr3) never did.
