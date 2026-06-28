# Examples

Each demo is a **DATA FILE** `examples/<name>.py` declaring one `DEMO = DemoSpec(...)` (scene +
policy only; schema in `deformableManipulationTools/demo_runner.py`). The single runner `example.py`
plays it — to add a demo, write one data file and change nothing else. The `--output-style` flag picks
how a run is rendered (default `scenic`):

```bash
# scenic (default): robolabViz renders outputs/<robot>/<name>/{frames/, simulation.mp4}
#   (<robot> = the active robot's short_name, so the two robots' renders never collide),
#   (over-shoulder-left + wrist cameras, side by side), on any CUDA GPU.
python example.py --demo examples/<name>.py --device cuda:0
python -m examples <name> --device cuda:0          # equivalent shim
python -m examples <name> --device cuda:0 --frames-per-image 60   # PNG still cadence

# basic: a plain Newton USD at outputs/<name>.usd (the old `--viewer usd` behaviour).
python -m examples <name> --output-style basic --device cuda:0
# CPU 1-frame smoke:
python -m examples cable_rigidCube_franka --output-style basic --device cpu --num-frames 1 \
  --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

Scenic opt-in extras: `--usd` (also write the full time-sampled RoboLab USD scene to
`outputs/<robot>/<name>/<name>.usd`), `--npz` (state cache `<name>.state.npz` + `geometry.pkl` for
`robolabViz.rerender`),
`--objectview` (extra object-inspection still camera, soft demos only), `--table` / `--background`
(vendored work-table / dome by name), `--wrist-eye` / `--wrist-target`. See
[docs/robolab-graphics.md](robolab-graphics.md).

Terminal output is tee'd to `outputs/terminal`.

## Physics demos

All examples share the **centralized** physics: `class Example(GraspExample)` from
`deformableManipulationTools` owns the whole build + the dynamic-proxy `TwoWayProxyCoupling`;
objects are added via the `deformableManipulationTools.assets` builders; all params come from
`deformableManipulationTools.params`. Physical, bounded grip force (~10–90 N), no cap. Each example
subclasses `ScenicGraspExample` (`robolabViz.scenic`, itself a `GraspExample`) and defines ONLY its
**scene** (`configure` + `build_scene`) and **policy** (`plan` + `set_robot_targets`) + the per-demo
`check_physics` asserts; all pass `--test` (in either output style).

Every non-YCB demo draws from **one** shared object set: the same `CABLE`, the same `SOFT_BLOCK`
(medium-stiffness 5 cm FEM block), the same 1 kg `RIGID_CUBE`, and the centralized `PLATE`. The
demos are therefore directly cross-comparable.

- **`cable_rigidCube_franka`** — descends to a cable on the table, grasps, lifts, sweeps
  side to side; the shared 1 kg rigid cube sits on the table as an obstacle. 8 substeps.
- **`cable_soft_franka`** — same cable demo + the shared soft FEM block the swept cable dents/nudges.
  16 substeps.
- **`rigidCube_soft_franka`** — grasps the shared 1 kg `RIGID_CUBE` via a pre-grasp waypoint,
  carries it, drops it half-offset onto the shared `SOFT_BLOCK` to squash it. 16 substeps.
- **`soft_compression_franka`** — grasps the centralized `PLATE` (sheet + handle, ~2 kg) by its
  handle, presses/drops it onto the shared `SOFT_BLOCK` to compress it. 16 substeps.
- **`soft_pickplace_franka`** — picks up the shared `SOFT_BLOCK` (5 cm) and places it at a
  target. 16 substeps. The proxies carry particle collision and the coupling harvests the
  proxy↔particle reaction (soft grip).

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
  then banana (`force_target=80`, the firmest stable value for the slip-prone curved mesh). 16 substeps.

## Cloth (in flight — see [ONGOING.md](ONGOING.md))

- **`cloth_franka`** — a Franka attempts a per-corner SCOOP pickup of a flat T-shirt (the first
  cloth-manipulation demo). Replicates Newton's `example_cloth_franka` motion (~45° tilt, all 9 DOFs
  from keyframes, not the force `GripController`). The sim is **stable** (≈critically-damped cloth
  contact + self-contact), but the shirt is **NOT lifted/dragged/moved** — an architectural limit of
  the gripper-proxy bridge (friction-independent; left visible, not faked). See
  [cloths.md](cloths.md) and [ONGOING.md](ONGOING.md). 600 frames, 16 substeps.

## Scenic rendering

There are no separate `_robolab` files anymore: scenic rendering is built into every demo via
`--output-style scenic` (the default). The shared wiring lives in `robolabViz.scenic.ScenicGraspExample`
— it reads the robot base pose, table, and (optional) soft-object position straight off the physics
example, so a new demo gets the RoboLab look for free. See [docs/robolab-graphics.md](robolab-graphics.md).
