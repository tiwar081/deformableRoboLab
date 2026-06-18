# RoboLab Graphics (`robolab_viz/`)

An alternative renderer reproducing NVIDIA RoboLab's look (the DROID rig in
`_external/RoboLab/examples/run_recorded.py`: home-office dome, sphere key light, maple work
table, Franka pedestal, over-shoulder + wrist cameras) on top of the *unchanged* Newton
physics. **Self-contained** — never imports the `robolab` package or `isaaclab`; RoboLab's
fixture/material/background assets are vendored into `assets/` (provenance in
`assets/ATTRIBUTION.md`). **Hard rule: nothing in the repo reads `_external/` at runtime —
assume that checkout can be deleted.**

## Usage

- `examples/example_<name>_robolab.py` subclasses the physics example, reuses its physics
  verbatim, swaps only the viz (forces `--viewer null`):
  `python -m examples cable_soft_franka_robolab --device cuda:0`.
- Customization surface: `robolab_viz/config.py`. `droid_scene_config(table=, background=)`
  returns RoboLab's DROID values as overridable dataclasses (fixtures, lights, cameras,
  object styles) — where future demos swap table / lights / cameras / styling.
- `--table {maple,oak,bamboo,black}`, `--background <name>` (e.g. `home_office`, `garage_2k`,
  `machine_shop_01_2k`). `available_tables()/_backgrounds()/_objects()` enumerate the vendored
  sets; `resolve_background` accepts the `_2k`-less stem. Choices render in the raycast
  preview too, not just RTX.
- Outputs in `outputs/robolab_preview/`: `combined.mp4` (the only kept video) + per-camera PNG
  frames (`--png-every`) + `wrist_coverage.json`. Opt-in: `--usd` (time-sampled scene USD),
  `--npz` (state cache + `geometry.pkl` that `robolab_viz.rerender` replays without re-simulating).
- `--objectview True` adds a fixed `object_view_camera` on the soft body, dumped as PNG stills
  only (`CameraConfig.in_combined_video=False`, so the kept video is unchanged).

## Architecture

- **Robot:** the *same* `fr3_franka_hand.urdf` converted to USD once via Isaac Sim's importer
  (`robot_usd.py`, subprocess, cached in `assets/robots/`). Puppeteered per frame by an
  *uncollapsed* FK mirror (`robot_fk.py`, `collapse_fixed_joints=False`) — the physics model
  collapses fixed joints, so the hand link the wrist camera mounts to has no simulated body.
  `test_final` asserts FK/sim parity on shared links.
- **Scene + objects:** time-sampled USD with pure `pxr` (`stage.py`, no kit): fixtures via
  payload; cable/rigid from Newton shapes; soft body as a per-frame deforming mesh. Two render
  paths consume the same USD: `raycast.py` (warp `wp.Mesh` raycaster, flat-shaded but exact —
  default, any CUDA GPU) and `isaac_render.py` (offline RTX). **This box is an H200 (Hopper,
  no RT cores): RTX cannot run here** (crashes at GPU init) — use it only on an RT-capable GPU.
- **Visual table** placed by `config.table_fixture_from_footprint(top_z, center_xy)` from the
  physics `table_pos/table_half`, so the visible surface coincides with the contact plane.
- **Preview materials:** each `FixtureConfig.texture_file` (sRGB base-color PNG) is
  planar/triplanar-projected onto the fixture box in world meters (`texture_uv_scale` = m/tile);
  on a ray miss the dome HDR/EXR is tone-mapped (auto-exposed Reinhard + sRGB) and sampled as
  an equirectangular (latlong, Z-up) backdrop. `black` table = matte paint (no texture).
  `geometry.pkl` carries the table texture + dome path so `rerender` reproduces the look.
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
  pipeline points at). Finalize the viz model BEFORE the object model.
