# RoboLab VBD

Franka manipulation demos on Newton physics (`_external/newton`), built toward a realistic
deformable-object simulation environment. Detailed knowledge is split into `docs/` (indexed
below); this file holds the project vision, the standards/rules, and the recurring gotchas an
agent must know before editing.

## Project vision & priorities

Building a simulation environment on Newton that handles **deformable bodies and
deformable↔rigid interaction** realistically. Primary deformable types: **cables/wires,
zip-ties, clothing, towels** (the current rod + FEM block are the first two).

Final vision: render **custom scenes** (table + background + any combination of deformable
and rigid assets) and **simulate a robot policy** in them, with high graphical realism
(NVIDIA Isaac Sim / RTX is the likely high-fidelity render path).

Priorities, in order: **1) physics fidelity → 2) render quality → 3) speed.**

## Solver framework rule

Two-way contact only happens inside one solver; the MuJoCo↔VBD bridge is one-way.

- **Any deformable/soft object present** → split `SolverMuJoCo` (robot) + `SolverVBD` (all
  objects) + kinematic gripper-proxy bridge. VBD is the only Newton solver hosting
  rigid+cable+soft+mutual two-way contact in one world.
- **Rigid-only** → a single `SolverMuJoCo` for robot + objects (true two-way grasp, mature
  mesh contact). Preferred for new rigid-only demos.
- `pickplace_ycb_franka` is rigid-only but kept on VBD on purpose (proof VBD hosts rigid meshes).

Details: [docs/solver-architecture.md](docs/solver-architecture.md).

## Physics rules (favor faithful simulation over visual shortcuts)

- No object self-attachment, auto-grasping, or teleporting into a grasp.
- No guided/scripted/kinematically-driven motion for passive scene objects.
- No collision-free bypasses between interacting objects.
- Robot motion may be commanded (actuators/targets), but object pickup, dragging, settling,
  and contact response must come from modeled contacts, constraints, gravity, solver dynamics.
- If a demo can't yet do a task physically, **leave the failure visible** and improve the
  model/contacts/solver/controller — don't hide it.
- The kinematic gripper proxies are only a contact bridge (mirror finger poses); they must not
  directly move, attach, or constrain objects.
- **No velocity clamps on objects** (robot/table excepted).
- **Never read or depend on `_external/` at runtime.** Import or copy what you need; assume
  `_external/` (newton, RoboLab) can be deleted and the codebase must still run.

## Newton version (environment gotcha)

`newton` is editable-installed from `_external/newton`, **drifted off the README-pinned commit
`2a1d4215`** to `2c242002`. Newton `c1af91d2` "Use absolute VBD damping" reinterprets VBD
damping from stiffness-relative (`D = kd·ke`) to **absolute** units and reformulates tet
damping into an objective `C=FᵀF` strain-rate metric that no longer damps rigid rotation.
This is why the examples carry inflated object-specific damping and 4×-softened soft blocks —
**if `_external/newton` is re-pinned/updated, re-derive the damping values.** Solver-wide
damping (`soft_contact_kd`, blanket `shape_material_kd`) is intentionally left native (`1e-4`/`1e2`).

## Recurring mistakes to avoid (update as they recur)

- **Viz shows soft bodies frozen / objects penetrating them / contact-before-touching** →
  `_sync_viz_state` must copy `particle_q`/`particle_qd` (not just body transforms) from the
  object sim state. This is the #1 recurring soft-body bug.
- **CUDA-graph capture** needs an **even substep count** and **one uncaptured warm-up frame**
  (lazy allocations raise inside capture). It falls back to the uncaptured loop on CPU/failure.
- **Stiff penalty contact + light element ejects** (`η = ke·dt²/m_reduced > 1`, with VBD
  `alpha=0` over-correction). Never feed the object reaction into the gripper DOF as stiff
  continuous in-loop feedback — it chatters and ejects. Use position control + a ramped setpoint.
- **Don't `shape_material_mu.fill_()` after finalize** — it clobbers per-shape friction that
  the grasp relies on (cable/pads set high on purpose).
- Verify changes with **instrumented headless `--viewer null --device cuda:0`** runs +
  `test_final`, not just by looking at a video.

## docs/ index

This repo's own docs:

- [docs/solver-architecture.md](docs/solver-architecture.md) — solver framework rule, robot &
  VBD object solver config + the `alpha=0`/ALM rationale, CUDA-graph capture rules, the viz
  particle-copy bug, the verification standard.
- [docs/gripper.md](docs/gripper.md) — one-way proxy bridge, the centralized force limit in
  `examples/grip_force.py` (rigid clamp / soft squeeze, 0→15 N ramp), the no-continuous-feedback
  stability invariant, obstacle non-penetration.
- [docs/deformables.md](docs/deformables.md) — cable (rod) and soft-FEM-block tuned parameters
  + reasons; notes on future cloth/zip-tie deformables.
- [docs/examples.md](docs/examples.md) — per-example descriptions and run commands.
- [docs/robolab-graphics.md](docs/robolab-graphics.md) — the `robolab_viz/` RoboLab-look
  renderer (raycast + offline RTX), customization surface, vendored assets, render gotchas.
- [docs/ONGOING.md](docs/ONGOING.md) — the **live log of in-flight work + recent changes**
  (volatile, changes often). Always read it for the current state before editing active areas
  (e.g. the cable coupling / gripper). Its own header lists what's currently unresolved — trust
  the file, not a summary here.

External references (large; consult on demand, don't read up front):

- [docs/NVIDIA_Newton_release.md](docs/NVIDIA_Newton_release.md) — NVIDIA's Newton GA release
  article. **The authoritative reference for the cable two-way MuJoCo↔VBD coupling** (dynamic
  finite-mass proxies, `sync_proxy_state` momentum-consistent undo, staggered one-step-lag
  step), plus the Isaac Lab Franka-cube physics config (`impratio=1000`, `iterations=20`, …),
  Kamino closed-chain, and hydroelastic-SDF contact. Primary source for the coupling work and
  the Isaac Lab / high-fidelity path.
- [docs/SOLVERS.md](docs/SOLVERS.md) — deep solver reference: why `SolverVBD` is fragile for
  *rigid* objects (the basis of the framework rule), how the object↔gripper two-way physics is
  wired, and an annotated catalog of Newton's upstream examples (cable, cloth, mpm, kamino,
  softbody, …) + RoboLab examples. Check here for upstream patterns when adding a new object
  type, solver, or render feature.
