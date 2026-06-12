# Ongoing Workflow: Cable Franka Examples

## Current Examples

The current cable/Franka demos are:

- `examples/cable_rigidCube_franka.py`
  - Working rigid-body cable grasp demo.
  - Franka starts open, descends to a cable resting on the table, closes until contacts stop the fingers, pauses at the grasp pose, lifts, then sweeps side to side.
  - Includes a smaller rigid cube on the table.
- `examples/cable_soft_franka.py`
  - Same cable/Franka setup.
  - Replaces the rigid cube with the tetrahedral rubber duck soft body used by Newton's `softbody_franka` example.
- `examples/__init__.py`
  - Registers both examples.
  - Defaults to `cable_rigidCube_franka` when no example name is provided.

Run commands:

```bash
python -m examples cable_rigidCube_franka --viewer usd --device cuda:0
python -m examples cable_soft_franka --viewer usd --device cuda:0
```

CPU smoke-test command:

```bash
python -m examples cable_rigidCube_franka --viewer usd --device cpu --num-frames 1 --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

## Physics Rules

Do not add shortcuts that break physical fidelity:

- No object self-attachment.
- No guided or scripted object motion.
- No collision-free bypasses.
- Cable, cube, and soft body movement should happen through gravity, contacts, friction, and solver dynamics.

The kinematic gripper proxy bodies in the object/VBD model are only a split-solver contact bridge: they mirror the real Franka finger body poses so the VBD cable can collide against the same imported finger collision geometry. They should not directly move, attach, or constrain the cable.

## Solver Architecture

The current examples use a split solver setup.

Robot side:

- Franka is imported from `franka_emika_panda/urdf/fr3_franka_hand.urdf`.
- Import matches Newton's `cloth_franka` convention:
  - `collapse_fixed_joints=True`
  - `force_show_colliders=False`
  - `enable_self_collisions=False`
- End-effector control uses `fr3_link7` with local offset `(0, 0, 0.22)`, matching the 22 cm tool offset in `cloth_franka`.
- Robot dynamics use `newton.solvers.SolverMuJoCo` with Newton contacts:
  - `solver="newton"`
  - `integrator="implicitfast"`
  - `cone="elliptic"`
  - `use_mujoco_contacts=False`
- A hidden robot-side table collider is included in the robot model. This is what prevents the gripper from passing through the table.

Object side:

- Object simulation uses `newton.solvers.SolverVBD`.
- The object model contains:
  - visible table
  - VBD cable rod
  - rigid cube or soft duck, depending on the example
  - kinematic gripper proxy bodies for cable/finger contact
- Object contacts use `newton.CollisionPipeline(..., contact_matching="latest")`.
- VBD is configured with `rigid_contact_history=True`, `rigid_contact_stick_motion_eps=0.0`,
  `rigid_avbd_contact_alpha=0.0`, and 12 iterations. Hard contacts are kept (penalty-only
  contacts lose the lifted cable), but sticky contact-point replay is disabled and
  penetration is corrected fully each step, following Newton's `cable_twist` example, which
  also drives kinematic bodies in persistent cable contact. With the defaults
  (`alpha=0.95`, sticky replay on), penetration accumulating against the moving pads spikes
  the contact force and its friction bound, ejecting the pinched cable with multi-m/s
  phantom velocity kicks at lift/sweep accelerations.
- The split-solver bridge is one-way (robot -> objects), the same limitation as Newton's
  `cloth_franka`/`softbody_franka` (`integrate_with_external_rigid_solver=True`). The cable
  cannot push back on the robot, so the grasp width must be commanded (see below).

Terminal output is automatically tee'd to `outputs/terminal`.

## Cable And Gripper Contact

Cable:

- Starts on the table, not in the gripper.
- Uses `add_rod(...)` with `wrap_in_articulation=True`.
- Radius: `0.008`.
- Segment length: `0.035`.
- Node count: `15`.
- Density: `1200` (realistic jacketed cable). The earlier `80` made segments so light
  (0.6 g) that pinch-contact residuals during arm motion converted into multi-m/s
  ejection kicks.
- Laid with a 2 cm bow (`_cable_layout_positions`). A perfectly straight round rod on a
  flat table has a free rolling mode (VBD has no rolling friction) and slowly rolls off
  the table; the bow locks it geometrically. The grasp/IK target is computed from the
  actual bowed node positions (midpoint of nodes 3 and 4).
- The start position clamp accounts for the full cable extent so the whole cable rests
  on the table (a real-weight cable end draping past the edge drags the cable off).
- Friction is `1.5`; cable friction is restored after the blanket object material fill so
  it is not overwritten.

Gripper:

- The cable exists only in the VBD object model, so no contact force can stop the fingers
  in the robot model. Commanding fully closed (`0.0`) therefore drives the pads straight
  through the cable (the old behavior: cable squeezed out above the pads, or tunneling).
- Instead the close target stops at the cable, like Newton's `softbody_franka`, which
  also closes to a finite width sized for its object: per-finger target =
  `cable_radius + cable_contact_margin + gripper_proxy_margin - 1 mm interference`
  = `0.009`. The 1 mm interference times the contact stiffness sets a bounded grip force.
- `gripper_open` is `0.04`, the URDF prismatic joint upper limit (was 0.045, out of range).
- The VBD gripper proxies copy the imported Franka finger collision shapes from the robot
  builder. Each finger has four imported collision boxes; the fingertip pad is a
  17.5 x 15.2 x 18.5 mm box whose inner face meets the grip center at joint position 0.
- Proxy contact margin is `0.001` (gap stays `0.008` for broad-phase headroom). The old
  8 mm margin inflated the contact surface a full cable radius away from the visible pads,
  which made grasps look wrong; it was compensating for the fully-closed command.
- Proxy friction is restored to `mu=1.0` after finalizing the object model.
- Proxy pose sync follows VBD's documented kinematic protocol (write `body_q` on both
  states each substep; the solver finite-differences against its internal `body_q_prev`
  for contact friction velocity), matching Newton's `cable_twist` kinematic driving.

## Motion Timing

The finalized timing is intentionally slower and easier to inspect:

- Descend/open approach: `0.0s -> 2.8s`.
- Close gripper: `2.8s -> 4.0s`.
- Pause closed at grasp pose: `4.0s -> 4.8s`.
- Lift/return before sweeping: `4.8s -> 6.8s`.
- Side-to-side sweep starts at `6.8s`, with its amplitude smoothstep-ramped over `1.5s`
  so the commanded velocity is continuous at onset (a step in target velocity kicks the
  pinched cable out of the grasp).
- Sweep frequency is `0.18 Hz`.
- Default output length is `720` frames.

The rigid cube half-size is `0.025`.

## Soft Variant

`examples/cable_soft_franka.py` replaces the cube with Newton's soft-body rubber duck:

- Asset: `manipulation_objects/rubber_duck/model.usda`.
- Prim: `/root/Model/TetMesh`.
- Construction: `newton.TetMesh.create_from_usd(...)` plus `builder.add_soft_mesh(...)`.
- Soft-body position is near the old cube location.
- Soft settings copied from Newton's `softbody_franka` pattern:
  - particle radius: `0.005`
  - soft contact margin: `0.01`
  - particle self-contact radius: `0.003`
  - particle self-contact margin: `0.005`
  - `k_mu=1.0e6`
  - `k_lambda=1.0e6`
  - `k_damp=1.0e-6`

## Verification Status

Instrumented headless verification (June 12, 2026): full 720-frame (12 s) runs on
`cuda:0` with per-frame grasp metrics (finger joint positions, distance from the EE
grip point to the nearest cable node, cable/cube heights, pad-local cable position and
per-pad penetration).

- `cable_rigidCube_franka`: GRASP HELD end to end. Fingers track the 0.009 close target
  exactly; both pads hold ~1 mm penetration through hold and lift; the grasped node
  tracks the grip point within ~1.3-3.3 cm through 5+ seconds of full-amplitude sweep.
  No NaNs, no tunneling, cube stays on the table.
- `test_final` now asserts the cable is still grasped and lifted at the final frame
  (gated on `sim_time >= 7.0` so short smoke runs still pass). The old "cube came too
  close to the cable" check was inherited from the `non_touching_object` era and
  contradicted this demo's cable-meets-cube intent; it was removed.
- Known limitations (intentional, documented): coupling is one-way (cable cannot stop
  the fingers; the grasp width is commanded), only the finger bodies have proxies (the
  palm/arm links do not collide with objects), and the soft example's proxies have
  `has_particle_collision=False` (gripper cannot touch the duck; only the cable does).

## Repo Notes

- `README.md` has the updated run commands.
- `CODEX.md` is tracked by git but is currently absent in this checkout; keep the no-hacks physics rule above as the active handoff note unless that file is restored.
