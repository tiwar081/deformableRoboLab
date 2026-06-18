# Examples

All in `examples/`, registered in `examples/__init__.py` (default:
`cable_rigidCube_franka`). Run pattern:

```bash
python -m examples <name> --viewer usd --device cuda:0      # 720 frames / 12 s default
python -m examples <name> --viewer null --device cuda:0     # headless verification
# CPU 1-frame smoke:
python -m examples cable_rigidCube_franka --viewer usd --device cpu --num-frames 1 \
  --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

Terminal output is tee'd to `outputs/terminal`.

## Physics demos

All of these (except `pickplace_ycb`) share the **centralized** grip: `class Example(GraspExample)`
(`examples/franka_common.py`) + the dynamic-proxy `TwoWayProxyCoupling` (`examples/grip_coupling.py`)
+ params from `assets/params.py`. Physical, bounded grip force (~10–90 N), no cap. Each only defines
its object + keyframe motion + `test_final`; all pass `--test`.

- **`cable_rigidCube_franka`** — descends to a cable on the table, grasps, lifts, sweeps
  side to side; a rigid cube sits on the table. 8 substeps.
- **`cable_soft_franka`** — same cable demo + a soft FEM block the swept cable dents/nudges.
  16 substeps. (`CABLE_DIAG=1` prints a per-frame grip/lift health line.)
- **`rigidCube_soft_franka`** — grasps a heavy steel cube (~1 kg) via a pre-grasp waypoint,
  carries it, drops it half-offset onto a pillow-soft block (`SOFT_BLOCK_PILLOW`). 16 substeps.
- **`soft_compression_franka`** — grasps a metal sheet (plate + handle) by its handle, presses/
  drops it onto the soft block to compress it. 16 substeps.
- **`soft_pickplace_franka`** — picks up a small soft FEM block (~33 mm) and places it at a
  target. 16 substeps. The proxies carry particle collision and the coupling harvests the
  proxy↔particle reaction (soft grip).

`pickplace_ycb_franka` — rubik's-cube + banana + bowl friction/impact demo, kept on the VBD
object framework as the "VBD can host rigid meshes" proof. **The one example still on the legacy
`grip_force.py` clamp** (not yet on the dynamic proxy): it has a pre-existing object fly-away that
is out of scope — migrate + retire `grip_force.py` once that is fixed.

## RoboLab-graphics variants

`example_<name>_robolab.py` subclass the physics example and swap only the renderer (force
`--viewer null`). See [docs/robolab-graphics.md](robolab-graphics.md). E.g.:
`python -m examples cable_soft_franka_robolab --device cuda:0`.
