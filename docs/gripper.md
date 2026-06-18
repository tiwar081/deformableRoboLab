# Gripper & contact bridge

## One-way bridge fundamentals

Objects exist only in the VBD model, so nothing in the robot model stops the fingers —
commanding fully closed drives the pads through the object. The contact bridge is the
**kinematic gripper proxies**: invisible bodies in the VBD object model that mirror the real
Franka finger poses each substep, so objects collide against the imported finger collision
geometry. They must not directly move, attach, or constrain objects.

- Proxies copy the imported finger collision shapes (4 boxes/finger; fingertip pad
  17.5×15.2×18.5 mm, inner face at the grip center at q=0). Proxy margin 0.001, gap 0.008
  (broad-phase headroom), friction restored to mu=1.0 after the blanket material fill.
- Proxy sync (kinematic): write `body_q` on both states each substep; VBD finite-differences
  against its internal `body_q_prev` for contact friction velocity (matches `cable_twist`).
- **Pre-grasp waypoint straight above wide objects is required**: the joint-space path from
  home arcs sideways and the open pads (80 mm gap) clip a 50 mm object before the grasp.
- `gripper_open = 0.04` (URDF prismatic upper limit).

## Force limit — centralized in `examples/grip_force.py`

Used by the rigid/soft examples (`pickplace_ycb`, `rigidCube_soft`, `soft_compression`,
`soft_pickplace`). Contact-driven (no baked object dimensions), grasp-window-gated,
**0→15 N linear ramp over 0.5 s, capped at 15 N**, public Newton API only:

- `GripForceClamp` (rigid): on first *detected* contact, applies an explicit **two-point
  capped-Coulomb clamp** to the grasped body's `body_f` — normal ramps 0→15 N + friction
  hold; the penalty squeeze is relieved (pads ease out) so it can't over-squeeze/eject.
- `RigidGripWidth`: contact-driven width control — creep to first contact, then ease out.
- `SoftGripWidth`: squeeze-to-force servo — close until the *measured* per-pad soft reaction
  reaches the ramped setpoint.
- Reaction readback (public API): rigid via `SolverVBD.collect_rigid_contact_forces`; soft
  recomputed from `soft_contact_*` geometry as `ke·penetration` per proxy.

**Stability invariant (do not violate):** the gripper is *position*-controlled; never feed
the object reaction back into the gripper DOF as stiff continuous in-loop feedback — across
one substep of lag against the stiff penalty contact (`k·dt²/m > 1`) it chatters to hundreds
of N and ejects the object. Use position control + a creeping/ramped setpoint (tried-and-
rejected: continuous force feedback).

Prior approach (superseded): a force-triggered *latch* (`_update_grip_target_kernel`) that
creeps closed and latches the width when the reaction crosses a threshold. Same invariant.

## Cable: two-way coupling — `examples/cable_coupling.py`

The cable uses a different bridge (`TwoWayProxyCoupling`): the cable reaction is harvested
and fed back to the arm so the grasp is two-way. M1 (kinematic proxy) is the verified
default; M2 (NVIDIA dynamic finite-mass proxy, for a true actuator-limited grip) is WIP.
**Status, design, and the dynamic-vs-kinematic divergence analysis live in ONGOING.md.**

## Obstacle (table) non-penetration

See the robot-side `robot_contact_table` static collider in
[docs/solver-architecture.md](solver-architecture.md).
