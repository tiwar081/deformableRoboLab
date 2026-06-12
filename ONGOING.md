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
- `examples/rigidCube_soft_franka.py`
  - Same solver framework (SolverMuJoCo robot + SolverVBD objects + kinematic finger proxies).
  - Franka grasps a heavy rigid cube (steel-like density, ~1 kg), carries it above the
    soft duck, opens the gripper, and the cube drops onto the duck. Verifies the
    rigid-cube/soft-duck interaction (the third pairing of the interaction matrix:
    cable-cube, cable-duck, cube-duck).
  - Grasp width formula matches the cable examples, with the cube half-width in place
    of the cable radius: per-finger close target = `cube_half + margins - 1 mm`.
  - Timeline: approach to a pre-grasp waypoint above the cube 0-2.2s, vertical descend
    2.2-3.2s, close 3.2-4.4s, hold 4.4-5.0s, carry to above the duck 5.0-7.0s, settle
    7.0-8.0s, open gripper 8.0-8.6s (cube falls ~10 cm onto the duck), retreat from 8.6s.
  - The pre-grasp waypoint is required: the joint-space path from home arcs sideways,
    and without it the open pads clip the 50 mm cube (only 15 mm lateral clearance)
    and knock it away before the grasp.
  - Cube-duck contact sizing: the flat cube face touches ~600 duck particles at once
    and VBD sums per-contact forces on the body, so this example uses
    `soft_contact_ke=1e3`, `soft_contact_kd=1e-3`, and restores the cube shape to the
    same values (`_restore_cube_materials`) because the body-particle pair material is
    the average of `soft_contact_*` and the shape's material. The averaging with the
    stiff table/pad shapes (5e4) keeps the cube's body-body contacts stiff. The pair
    value is bounded on both sides: it must stay well above the squashy duck's
    structural stiffness (~1.5e3 N/m at this contact size) or the cube digs through
    the particle cloud instead of loading the FEM (pair 1e2 let the cube fall through
    the duck), and well below the ejection regime (pair ~2.5e4 with pair kd ~50
    launched the 1 kg cube at >100 m/s).
    `rigid_body_particle_contact_buffer_size=4096` avoids per-body contact-buffer
    overflow (warnings showed ~650 contacts; dropped contacts destabilize the
    impact into NaN).
  - Verified (June 12, 2026, instrumented 720-frame headless run): cube grasped and
    carried without disturbance, dropped from ~13 cm onto the duck, and comes to rest
    ON the squashed duck at the drop point (final xy offset 4 mm; cube bottom at
    z=0.081 on the compressed duck top at 0.080, duck column squashed from ~5.8 cm to
    ~1.1 cm). No NaN, no ejection, no pass-through; `test_final` passes.
  - Note: with the squashy k_mu=1e4 material the duck's cantilevered head droops
    ~7 cm during initial settling (COM moves only ~5 mm; it is rotation, not sliding).
    If that reads as too floppy in renders, raise k_mu to ~3e4 (one line in both duck
    examples) at the cost of a shallower resting squash under the cube.
- `examples/soft_compression_franka.py`
  - Same solver framework. Franka grasps a heavy metal sheet (2x the old cube's mass)
    by its handle, carries it, and drops it half-offset onto the soft block; the sheet
    settles tilted on the block edge, holding ~1 cm of sustained compression.
- `examples/__init__.py`
  - Registers all four examples.
  - Defaults to `cable_rigidCube_franka` when no example name is provided.

IMPORTANT viz lesson: the viewer renders a separate combined viz model, and
`_sync_viz_state` must copy `particle_q`/`particle_qd` from the object sim state in
addition to body transforms. Until June 12, 2026 it copied only bodies, so every soft
body rendered frozen at its rest shape while the simulation deformed it underneath --
the root cause of ALL "soft body never deforms / objects penetrate it / contact happens
before touching" reports (the contacts were real, against the simulated surface, which
had settled away from the stale rendered one).

Run commands:

```bash
python -m examples cable_rigidCube_franka --viewer usd --device cuda:0
python -m examples cable_soft_franka --viewer usd --device cuda:0
python -m examples rigidCube_soft_franka --viewer usd --device cuda:0
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

Runtime (June 12, 2026): the substep loop is fully device-resident and captured into a
CUDA graph (one `wp.capture_launch` per frame), following Newton's robot examples. The
keyframe trajectory and the gripper-proxy pose sync run as Warp kernels reading frame
time from a device buffer; capture happens after one uncaptured warm-up frame (lazy
solver/pipeline allocations raise inside capture) and requires an even substep count
(the state-swap must return to its starting binding per captured frame). Falls back to
the uncaptured loop on CPU or capture failure. Measured on an H200 (null viewer):
`cable_rigidCube` 74.8 -> 11.6 ms/frame; with the soft block and upstream contact
config (tile solve off), `cable_soft` 66.8 ms/frame and `rigidCube_soft` 57.0 ms/frame
(pre-capture baseline was ~306).

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

## Soft Body (Both Soft Examples)

The soft body in `cable_soft_franka` and `rigidCube_soft_franka` is the FEM block from
Newton's `rigid_soft_contact` example (the only upstream two-way VBD rigid+soft scene),
which replaced the rubber duck. The duck regressed repeatedly: its light particles
(~0.015 g) made the staggered rigid<->particle coupling fragile (eject/NaN/pass-through
trade-offs), and its cantilevered head drooped on squashy materials. The block is the
configuration upstream actually tuned.

- Construction: `builder.add_soft_grid(...)`, 4x4x4 cells of 0.0125 m (5x5x5 cm — the
  same size as the rigid cube, 125 particles, ~12.5 g total), centered at
  `soft_start_pos` on the table.
- Material: `density=100`, `k_damp=1.0` everywhere. `k_mu=1.0e4`, `k_lambda=5.0e4`
  (upstream verbatim) in `cable_soft_franka` and `soft_compression_franka`;
  `rigidCube_soft_franka` uses a much softer pillow-like `k_mu=5.0e2`,
  `k_lambda=2.5e3` with a half-offset drop so the cube visibly sinks in and rolls off
  (verified stable: ~17 cm transient particle displacement at impact recovers cleanly).
- Contact (upstream verbatim): `soft_contact_ke=1.0e5`, `soft_contact_kd=1.0e-4`,
  `soft_contact_kf=1.0e3`, `soft_contact_mu=0.3`, `particle_max_velocity=50`,
  `particle_enable_tile_solve=False`, `rigid_body_particle_contact_buffer_size=512`.
  The cube shape's material is restored to `ke=1e5, kd=1e-4` so the averaged cube-block
  pair is exactly upstream's sphere-grid pairing (1e5); the cable keeps its authored
  `ke=2e4` (pair 6e4).
- Particle radius `0.0035` (scaled from upstream's 0.01 along with the geometry): the
  contact boundary sits one particle radius above the rendered surface, so a large
  radius reads as contact-before-touching; 3.5 mm stays visually tight while leaving
  >2 substeps of contact engagement at the fastest approach (~1.5 mm/substep).
- Deviations from upstream, and why: geometry scaled to the table (theirs is a 2x1x1 m
  beam), 16 substeps instead of 32 (runtime; verified stable), `rigid_contact_hard=True`
  kept for body-body contacts (the grasp needs hard contacts; upstream's
  `rigid_contact_hard=False` only affects body-body and their scene has no grasp),
  12 VBD iterations (grasp recipe).
- Verified (June 12, 2026, instrumented 720-frame runs, after the viz fix): the swept
  cable contacts the block and dents/nudges it, grasp held end to end; the dropped
  cube squashes the pillow block at its edge and rolls off (stable); the dropped sheet
  settles tilted on the block edge with ~1 cm sustained compression (block top
  0.120 -> 0.110). All `test_final` checks pass; viz particle state verified to track
  the simulation exactly.

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
