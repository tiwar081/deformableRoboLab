# Examples

All in `examples/`, registered in `examples/__init__.py`. Each demo is a **single file**
`<name>.py`; the `--output-style` flag picks how a run is rendered (default `scenic`):

```bash
# scenic (default): robolabViz renders outputs/<name>/{frames/, simulation.mp4}
#   (over-shoulder-left + wrist cameras, side by side), on any CUDA GPU.
python -m examples <name> --device cuda:0
python -m examples <name> --device cuda:0 --frames-per-image 60   # PNG still cadence

# basic: a plain Newton USD at outputs/<name>.usd (the old `--viewer usd` behaviour).
python -m examples <name> --output-style basic --device cuda:0
# CPU 1-frame smoke:
python -m examples cable_rigidCube_franka --output-style basic --device cpu --num-frames 1 \
  --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

Scenic opt-in extras: `--usd` (also write the full time-sampled RoboLab USD scene to
`outputs/<name>/<name>.usd`), `--npz` (state cache + `geometry.pkl` for `robolabViz.rerender`),
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

- **`cable_rigidCube_franka`** — descends to a cable on the table, grasps, lifts, sweeps
  side to side; a rigid cube sits on the table. 8 substeps.
- **`cable_soft_franka`** — same cable demo + a soft FEM block the swept cable dents/nudges.
  16 substeps.
- **`rigidCube_soft_franka`** — grasps a heavy steel cube (~1 kg) via a pre-grasp waypoint,
  carries it, drops it half-offset onto a pillow-soft block (`SOFT_BLOCK_PILLOW`). 16 substeps.
- **`soft_compression_franka`** — grasps a metal sheet (plate + handle) by its handle, presses/
  drops it onto the soft block to compress it. 16 substeps.
- **`soft_pickplace_franka`** — picks up a small soft FEM block (~33 mm) and places it at a
  target. 16 substeps. The proxies carry particle collision and the coupling harvests the
  proxy↔particle reaction (soft grip).

- **`pickplace_ycb_franka`** — picks a rubik's cube and a banana and drops them into a bowl; a
  rigid-mesh friction/impact demo, on the same centralized dynamic-proxy path as the rest (proof
  VBD hosts rigid **meshes**). The bowl and banana collide as **coacd convex-hull pieces** while
  their full meshes render — a raw concave bowl mesh ejected the solve (the old fly-away); the
  banana is held by genuine friction (slips physically if the grip is marginal). 16 substeps.

## Scenic rendering

There are no separate `_robolab` files anymore: scenic rendering is built into every demo via
`--output-style scenic` (the default). The shared wiring lives in `robolabViz.scenic.ScenicGraspExample`
— it reads the robot base pose, table, and (optional) soft-object position straight off the physics
example, so a new demo gets the RoboLab look for free. See [docs/robolab-graphics.md](robolab-graphics.md).
