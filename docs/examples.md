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

- **`cable_rigidCube_franka`** — descends to a cable on the table, grasps, lifts, sweeps
  side to side; a rigid cube sits on the table. 8 substeps. Two-way cable coupling
  (`cable_coupling.py`, M1 kinematic).
- **`cable_soft_franka`** — same cable demo + a soft FEM block the swept cable dents/nudges.
  16 substeps. Two-way cable coupling (M1 default; M2 force-limited WIP behind
  `force_limited_grip` — see ONGOING.md).
- **`rigidCube_soft_franka`** — grasps a heavy rigid cube (steel, ~1 kg) via a pre-grasp
  waypoint, carries it, drops it half-offset onto a pillow-soft block
  (`k_mu=1.25e2,k_lambda=6.25e2`); cube squashes the block edge and rolls off. 16 substeps.
  Force-limited grip via `grip_force.py` (rigid clamp).
- **`soft_compression_franka`** — grasps a heavy metal sheet (~2× the cube; 18×12×0.8 cm
  plate + grasp handle) by its handle, drops it half-offset onto the soft block; settles
  tilted holding ~1 cm compression. 16 substeps. Force-limited grip (rigid clamp).
- **`soft_pickplace_franka`** — picks up a small graspable soft FEM block (~33 mm), carries
  it across the table, places it at a target. 16 substeps. Force-limited grip via
  `SoftGripWidth` (squeeze-to-force). Caveat: the small block is too soft to hold a steady
  15 N — a lower target gives a cleaner hold.

`pickplace_ycb_franka` — rubik's-cube + banana + bowl friction/impact demo, kept on the VBD
object framework as the "VBD can host rigid meshes" proof. Force-limited grip via
`grip_force.py`. Has a known pre-existing passive resting-object ejection (out of scope).

## RoboLab-graphics variants

`example_<name>_robolab.py` subclass the physics example and swap only the renderer (force
`--viewer null`). See [docs/robolab-graphics.md](robolab-graphics.md). E.g.:
`python -m examples cable_soft_franka_robolab --device cuda:0`.
