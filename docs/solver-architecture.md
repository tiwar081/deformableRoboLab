# Solver Architecture

## Solver framework selection (the rule)

Full two-way contact only happens *inside one solver* (VBD); the MuJoCo↔VBD bridge crosses
solvers via dynamic finite-mass finger proxies that mirror the fingers in the VBD model and feed
the net object reaction back to the arm/EE (one-step lag) — so the arm DOES feel the object, but
through the proxy bridge, not a shared contact.

- **Any deformable/soft object present** (cable/rod, cloth, FEM block) → split
  **`SolverMuJoCo` (robot) + `SolverVBD` (all objects)** with the **dynamic gripper-proxy
  bridge** (`examples/grip_coupling.py`). VBD is the only Newton solver that hosts
  rigid+cable+soft+their mutual two-way contact in one world, so every object that must touch a
  deformable lives in the VBD model.
- **Rigid-only** → a **single `SolverMuJoCo`** for robot + objects (Newton
  `brick_stacking`/`panda_hydro` pattern): true two-way frictional grasp, MuJoCo's mature
  convex/SDF/hydroelastic mesh contact, none of the VBD-rigid-mesh fragility. Preferred for
  new rigid-only demos.
- Caveat: `pickplace_ycb_franka` is rigid-only but deliberately kept on VBD as the proof
  VBD can host arbitrary rigid mesh shapes (so the scene could later gain a soft object).

## Robot side

- Franka from `franka_emika_panda/urdf/fr3_franka_hand.urdf`, `collapse_fixed_joints=True`,
  `force_show_colliders=False`, `enable_self_collisions=False` (Newton `cloth_franka` convention).
- EE control point: `fr3_link7` + local offset `(0,0,0.22)` → fingertip-pad bottoms.
- `SolverMuJoCo(solver="newton", integrator="implicitfast", cone="elliptic",
  use_mujoco_contacts=False)` + Newton `CollisionPipeline`.
- A hidden robot-side table collider (`robot_contact_table`) stops the gripper at the table
  surface (verified by driving the EE 8 cm below the top — halts exactly). Add any fixed
  obstacle the gripper must not cross as a static collider in the robot model the same way.
- IK (`newton.ik.IKSolver`: position + rotation + joint-limit objectives) solves keyframe
  poses once at startup.

- `SolverVBD`, 12 iterations, `rigid_contact_stick_motion_eps=0.0`, **NVIDIA default-hard
  contacts** (`alpha=0.95`, no cross-step history).
  - *History (now resolved):* the cable path once used `rigid_avbd_contact_alpha=0.0` +
    `rigid_contact_history=True` to hold the lifted cable with a *kinematic* proxy. That
    accumulates the ALM multiplier `λ` (`f_n = ke·pen + λ`, unbounded → 1e4–1e6 N grip): a
    kinematic proxy early-outs so the runaway is computed-but-not-applied (stable, uncontrolled),
    but a **dynamic** proxy applies it → divergence. The fix was to drop alpha=0+history (default
    contacts) AND re-derive the overdamped contact `kd` (a second, independent cause); the cable
    is then held by honest squeeze friction at a bounded force. Full analysis in ONGOING.md.
- Object contacts: `CollisionPipeline(contact_matching="latest", soft_contact_margin=0.01)`.
- The object model contains the visible table, manipulated objects, the soft block (where
  present), and the **dynamic** gripper proxies.
- Bridge is **TWO-WAY** via the proxy: robot motion → proxy → object squeeze; object reaction →
  harvested → net load fed to the arm/EE. See [docs/gripper.md](gripper.md).

## Runtime (CUDA-graph capture)

- The substep loop is device-resident and captured into a CUDA graph (one
  `wp.capture_launch` per frame); keyframe trajectory + proxy sync are Warp kernels reading
  frame time from a device buffer.
- Capture happens **after one uncaptured warm-up frame** (lazy allocations raise inside
  capture) and requires an **even substep count** (state swap must return to its starting
  binding per captured frame). Falls back to the uncaptured loop on CPU / capture failure.
- Measured H200 (null viewer): `cable_rigidCube` 11.6, `cable_soft` 66.8, `rigidCube_soft`
  57.0 ms/frame.

## Visualization (critical recurring bug)

The viewer renders a separate combined viz model (robot builder + object builder).
`_sync_viz_state` **must copy `particle_q`/`particle_qd` from the object sim state** in
addition to body transforms. Copying only bodies renders every soft body frozen at its rest
shape while the sim deforms it underneath — the historical root cause of all "soft body
never deforms / objects penetrate it / contact before touching" reports.

## Verification standard

Verify every demo with instrumented **headless null-viewer `cuda:0`** runs reading sim state
per frame (grasp tracking, finger joints, object heights, soft-particle displacement,
NaN/ejection/pass-through, viz/sim particle parity after `render()`) — not just visually.
`test_final` in each example asserts the outcome, time-gated so 1-frame smoke runs pass.
