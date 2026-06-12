# Ongoing Workflow: Minimal Cable Franka

## Current Scope

`examples/example_minimal_cable_franka.py` is currently focused on the smallest working interaction:
the Franka lowers to a cable resting on the table, closes the gripper, lifts, and sweeps the grasped cable
side to side across the table. The block and soft-body objects are intentionally deferred until this contact
and pickup behavior works.

## Solver Choice

- The scene now follows Newton's `robot_panda_hydro` example.
- The robot, table, and cable live in one Newton model.
- Physics stepping uses `newton.solvers.SolverMuJoCo` with `use_mujoco_contacts=False`, `solver="newton"`,
  `integrator="implicitfast"`, `cone="elliptic"`, and Newton's explicit `CollisionPipeline`.
- Contacts use the Panda Hydro-style hydroelastic SDF setup:
  - gripper finger/hand meshes get SDF hydroelastic collision
  - extra finger pad meshes are added from Newton's manipulation pad asset
  - non-finger robot meshes are convexified for collision
  - the table is a static SDF mesh
  - cable capsules use hydroelastic primitive SDFs

## Cable Representation

- The cable starts on the table, not in the robot's grasp.
- The cable is not kinematic and is not manually driven.
- To stay on the `robot_panda_hydro` MuJoCo solver path, the cable is represented as a dynamic capsule chain:
  - one free root joint
  - one dynamic capsule body per cable segment
  - ball joints between neighboring segments
- This deliberately avoids `ModelBuilder.add_rod` for now because Newton rods create CABLE joints, which belong
  to the VBD cable examples rather than the MuJoCo Panda Hydro path requested for this iteration.

## Control

- Only the robot moves autonomously.
- Robot targets are produced with Newton's analytic IK pattern from `robot_panda_hydro`.
- The IK waypoints move from rest, down to the tabletop cable, close the gripper, lift, then loop between left and
  right sweep positions while keeping the gripper closed.
- Cable movement should now occur only through contact and gravity.

## Verification Status

- Ran `python3 -m py_compile examples/example_minimal_cable_franka.py`.
- Did not run the simulation, per request; GPU behavior still needs user-side testing.
