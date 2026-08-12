# Examples

These demos exist to exercise and test the PHYSICS ENGINE — that is why this doc lives in
`docs/physicsEngine/`. Tests for other pipeline stages (agentic generation, trajectory) use their
own examples, documented with those stages.

Each demo is a **DATA FILE** `examples/<name>.py` declaring one `DEMO = DemoSpec(...)` (scene +
policy + optional render look; schema in `deformableManipulationTools/demo_runner.py`). The single
runner `example.py` plays it — to add a demo, write one data file and change nothing else. The
`--output-style` flag picks how a run is rendered (default from `settings.yaml` `render.style`,
which ships as `mp4`); every style writes into `outputs/<robot>/<name>/` (`<robot>` = the active
robot's short_name, so the two robots' outputs never collide):

```bash
# mp4 (default): lightweight simulation.mp4 (over-shoulder-left + wrist cameras, side by
#   side, flat shading — no HDRI decode; table texture kept for motion cues), on any CUDA GPU.
python example.py --demo examples/<name>.py --device cuda:0
python -m examples <name> --device cuda:0          # equivalent shim

# mp4_advanced: the RoboLab look — HDRI image-based lighting, soft shadows, PBR materials,
#   textured catalog objects, anti-aliasing; full-res simulation_advanced.mp4 + frames/ PNG
#   stills (every --frames-per-image frames, default 30) + wrist_coverage.json.
python -m examples <name> --output-style mp4_advanced --device cuda:0

# usd: a plain time-sampled Newton USD at outputs/<robot>/<name>/<name>.usd (lightest).
python -m examples <name> --output-style usd --device cuda:0
# CPU 1-frame smoke:
python -m examples cable_rigidCube_franka --output-style usd --device cpu --num-frames 1 \
  --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

Deprecated aliases (accepted with a warning): `basic`→`usd`, `scenic`→`mp4_advanced`.

mp4-style opt-in extras: `--usd` (also write the full time-sampled RoboLab USD scene to
`outputs/<robot>/<name>/<name>.usd`), `--npz` (state cache `<name>.state.npz` + `geometry.pkl` for
`robolabViz.rerender`; captured from an `mp4` run the pickled geometry is untextured),
`--objectview` (extra object-inspection still camera, soft demos only; PNG-only, so pair with
`--frames-per-image`/`mp4_advanced`), `--table` / `--background` (vendored work-table / dome by
name; precedence CLI > the demo's `RenderSpec` > `settings.yaml`), `--wrist-eye` / `--wrist-target`.
A demo customizes its own look (background, table, lights, cameras, object styles, quality knobs)
by declaring `render=robolabViz.RenderSpec(...)` in its `DemoSpec` — `rigidCube_soft_franka` is the
exemplar. See [docs/rendering/robolab-graphics.md](../rendering/robolab-graphics.md).

Terminal output is tee'd to `outputs/terminal`.

## Physics demos

All examples share the **centralized** physics: the generic `DataDrivenExample`
(`deformableManipulationTools.demo_runner`) turns each demo's `DemoSpec` into the build + policy
the framework expects; objects are added via the `deformableManipulationTools.assets` builders and
all params come from `deformableManipulationTools.params` (realistic, per-object authored — see
solver-architecture.md "Contact materials"). Physical, bounded grip force, no cap; all pass
`--test` in all three output styles.

Every non-YCB demo draws from **one** shared object set: the same `CABLE` (rubber-jacketed), the
same `SOFT_BLOCK` (a raspberry-like 5 cm fruit: E ≈ 5.7 kPa, 81 g, delicate), the same 1 kg
`RIGID_CUBE` (solid steel), and the centralized `PLATE` (~2 kg metal presser). The demos are
therefore directly cross-comparable.

- **`cable_rigidCube_franka`** — descends to a cable on the table, grasps, lifts, sweeps
  side to side; the shared 1 kg rigid cube sits on the table as an obstacle. 8 substeps.
- **`cable_soft_franka`** — same cable demo + the shared soft FEM block the swept cable dents/nudges.
  16 substeps.
- **`rigidCube_soft_franka`** — grasps the shared 1 kg `RIGID_CUBE` via a pre-grasp waypoint,
  carries it, drops it half-offset onto the shared `SOFT_BLOCK`. 16 substeps. Since the block
  became a raspberry-like fruit (2026-07-11) the dropped steel cube squashes it and rolls off —
  honest physics; the old foam block used to catch and cradle the cube.
- **`soft_compression_franka`** — grasps the centralized `PLATE` (sheet + handle, ~2 kg) by its
  handle, presses/drops it onto the shared `SOFT_BLOCK` to compress it. 16 substeps. Since the
  block became a raspberry-like fruit, the 2 kg plate flattens it nearly to the table (its
  ~900 Pa of press stress vs 5.7 kPa tissue plus extrusion) — expected physics. Known
  imperfection (root-caused 2026-07-09, deferred): the held plate slowly pivots about the jaw
  axis during the carry (~18° since the 2026-07-10 arm-damping fix; was ~25°) — the current
  contact config has no static friction to arrest rotational creep; the measured fix trades it
  for a jerkier grasp and is reserved for a future slippage effort (SOLVERS.md §6).
- **`soft_pickplace_franka`** — the delicate-fruit pick-and-place: slow ~4.4 s squeeze dwell,
  slow lift, grasp 1 cm below the equator, `force_target=3.5 N` (peak ~3.2 N on the fruit vs the
  5–10 N raspberry crush range; every sub-2 N target measurably sheds — see deformables.md).
  16 substeps. The proxies carry particle collision and the coupling harvests the proxy↔particle
  reaction (soft grip).

- **`pickplace_ycb_franka`** — picks a rubik's cube and a banana and drops them into a bowl; a
  rigid-mesh friction/impact demo. **Rigid-only → routes to a single `SolverMuJoCo`** (no
  deformable present), the faster true-two-way path. The bowl and banana collide as **coacd
  convex-hull pieces** while their full meshes render — a raw concave bowl mesh ejected the solve
  (the old fly-away); the banana is held by genuine friction (slips physically if the grip is
  marginal). 16 substeps.
- **`pickplace_ycb_vbd_franka`** — the SAME ycb scene **plus a token soft cube** in a table
  corner, whose particles auto-route the whole workspace to the **split MuJoCo+VBD** dynamic-proxy
  path. The A/B twin of `pickplace_ycb_franka` demonstrating the centralized solver routing (≈2.2×
  slower than the rigid-only MuJoCo path). Two `GraspWindow`s — rubik's cube (`force_target=30`)
  then banana (`force_target=80`). 16 substeps. Known imperfection (root-caused 2026-07-09,
  deferred): the banana's EDGE grasp is intermittent and NO force target fixes it — the wedge
  converts squeeze into a self-ejection load in the same proportion at ANY force (measured at
  10/30/45/80 N across both contact modes), and the current config has no static friction to pin
  it (SOLVERS.md §6). Real fix = compliant fingertips or a flatter grasp point. The rubik's cube
  releases cleanly in BOTH ycb demos; its drop point is INTENTIONALLY offset from the bowl centre
  to observe the bowl–cube collision at an odd angle, so the cube bouncing on/off the rim is
  expected behaviour, not a defect.

## Cloth — the shirt fold (recipe + history in [cloths.md](cloths.md))

The shirt (the SI-converted Newton cloth, `ClothConfig` defaults) drops inflated from clear of the
table and arm, settles flat and centred on the ONE shared table, and the Franka folds it following
Newton's grasp recipe: command the fingertip 5 mm BELOW the table top (the stopper converts the
excess into pressing normal force — the μ·N anchor), pinch to a finite jaw gap (never 0), lift,
drag, release. Fully per-object physical friction (pads eff ≈ 0.45, table eff ≈ 0.35). Runs on the
standard split
MuJoCo-robot + VBD-cloth path at 10 substeps / 5 VBD iterations (Newton's dt — part of the contact
parity).

- **`cloth_franka`** — ONE hot-dog fold: pinch the middle of the shirt's +x edge region and fold it
  in half (the +x edge lands on the −x edge). Grip = **force windows** (`GraspWindow`,
  `force_target=2 N`, inside the shell's achievable squeeze) through the standard dynamic proxies —
  the target-relative admittance law converges to a stable ~8–9 mm pinch
  ([gripper.md](gripper.md)). 1380 frames (23 s).
- **`cloth_franka_full`** — Newton's FULL four-fold sequence **in spirit** (top-left, bottom-left,
  top-right, bottom-right folds, then the bottom hem carried all the way to the shirt's TOP edge —
  a full half fold): Newton's pose offsets mapped about the shirt centre at SCALE 0.8 onto the
  shared table, every grasp TOP-DOWN (no tilted wrist), grip = Newton's explicit finger schedule
  (finite 8 mm jaw). Each fold is the same clean cycle — glide empty at cruise height (15 cm) to
  above the grasp point, vertical descent to the table top (the two weakest grasps press 5 mm
  below it, the μ·N anchor), close, vertical lift, carry at height, release, rise clear — so the
  empty gripper never plows the cloth and the held cloth is carried, not scraped. 4590 frames
  (76.0 s).

## Rendering

There are no separate `_robolab` files anymore: rendering is built into every demo via
`--output-style` (`usd` / `mp4` default / `mp4_advanced`). The shared wiring lives in
`robolabViz.scenic.ScenicGraspExample` — it reads the robot base pose, table, and (optional)
soft-object position straight off the physics example, so a new demo gets the render for free.
A demo may declare its own look with `DemoSpec.render = robolabViz.RenderSpec(...)`
(only `rigidCube_soft_franka` does today). See [docs/rendering/robolab-graphics.md](../rendering/robolab-graphics.md).
