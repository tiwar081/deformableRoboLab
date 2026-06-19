# Gripper & contact bridge

The grip is **centralized**: one implementation (`deformableManipulationTools/grip.py`, built into
the run by `deformableManipulationTools/framework.py`) and one parameter set
(`deformableManipulationTools/params.py`) are shared by every example, so the same robot grips
every object the same way. Physical, bounded force; **no force cap, no post-hoc clamp**.

## Dynamic finite-mass proxy bridge

Objects live only in the VBD object model, so nothing in the robot (MuJoCo) model stops the
fingers. The bridge is two **dynamic finite-mass finger proxies** in the object model
(`grip.build_gripper_proxies`): invisible bodies with the imported finger collision
shapes, mass `GRIP.proxy_mass` (10 kg, ≈ the reflected articulated-chain inertia).

Each substep (`grip.TwoWayProxyCoupling`, NVIDIA's recipe — staggered one-step lag):

1. `apply_to_robot` — feed the previous step's harvested object reaction onto the **arm/EE**.
2. robot solves; swap.
3. `sync_proxies` — re-pin each proxy to its finger pose+velocity, **subtracting** the velocity
   gravity + the lagged wrench will impart over `dt` (momentum-consistent undo), so the proxy
   stays slaved to the finger while still participating as a finite-mass contact body.
4. object (VBD) solves the squeeze; swap.
5. `harvest` — collect the object→proxy reaction for next step (rigid via
   `SolverVBD.collect_rigid_contact_forces`; **soft** FEM blocks via a recomputed
   `n·ke·penetration` over the public `Contacts.soft_contact_*` geometry, enabled by passing
   `soft_contact_ke=` and building the proxies with `has_particle_collision=True`).

**Feedback is net-to-EE, not per-finger.** The two pad wrenches summed cancel the internal
squeeze and leave the external load (weight + motion reaction), which goes to the arm; the
position-held fingers keep their grip. The per-pad reaction is available as tactile data via
`TwoWayProxyCoupling.raw_force_norms()` but is **never** fed to the finger DOFs.

### Stability invariant (do not violate)

Never feed the object reaction into the gripper **DOF / per-finger** — confirmed empirically:
the pad reaction is outward, so routing it to each finger pushes the pads open and the grasp is
lost (grip → 0). Keep the fingers position-controlled and feed the **net** reaction to the arm.

## Grip-force tuning

The grip force is **emergent and physical** — the position-controlled squeeze against bounded
contact (NVIDIA default-hard contacts + re-derived physical damping), not a clamp. Typical
operating point ≈ **30–90 N** for the cable, comparable for the rigid/soft grips; aim for
**~10–30 N** by adjusting the knobs below. **All knobs are centralized in
`deformableManipulationTools/params.py`**, so
changing them changes every demo identically (same robot, same grip).

Knobs (most-used first), all in `deformableManipulationTools.params.GRIP` unless noted:

- **`grasp_interference`** (default 1 mm) — how far past the object surface the close target bites.
  The per-example close target is `gripper_closed = object_half_width + object_margin +
  proxy_margin − grasp_interference`. **More interference → deeper squeeze → more force.** This is
  the primary force knob.
- **`proxy_ke`** (5e4 N/m) — proxy contact stiffness. Higher → more force per unit penetration.
- **`FRANKA.finger_effort`** (20 N) — the finger actuator force limit (physical, *not* a cap on
  the grip). It bounds how hard the actuator can drive the close, so it bounds the *transient*
  squeeze; the steady grip is set by interference × stiffness.
- **`proxy_kd`** (1e2 N·s/m, absolute) — contact damping. Re-derived for the pinned Newton; the
  old 4e5/5e6 values were ~1e4× critical and dominated the grip with a spurious
  velocity-proportional force. Leave near 1e2 unless re-deriving for a Newton bump (see CLAUDE.md).
- **`proxy_mass`** (10 kg) — heavier → the proxy resists being pushed off the object (maintains
  penetration); too light loses the grip. 10 kg is a stable default.

**Contact geometry sets the force scale, not just the knobs.** The emergent force ≈ `ke · penetration`
summed over every contact point, so at the *same* `grasp_interference` a flat box face gripped by
the flat pads makes a large multi-point patch and reads high (the rubik's cube ~155 N), while a
curved or small object (the banana) makes a tiny patch, reads low, and can be marginal/slip. It's
bounded and net-to-EE ≈ 0 either way (not a divergence). If a box grip is uncomfortably hard, give
it a hair of pad clearance (raise `gripper_closed`); don't expect a curved object to hold as firmly
at the same bite.

To target a specific force: tweak `grasp_interference` first (it's roughly linear in the steady
grip), verify with an instrumented headless run logging `GraspExample.grip_force_norms()` (per-pad
`[left, right]` N), then `--test`.

## Legacy clamp — fully retired

The old post-hoc **0→15 N capped-Coulomb clamp / squeeze-to-force** (`GripForceClamp`,
`RigidGripWidth`, the old `examples/grip_force.py`) is **gone**. Every example — including
`pickplace_ycb_franka` — now uses the dynamic finite-mass proxy. ycb's old object fly-away was the
raw-concave-bowl-mesh ejection, fixed by coacd convex-hull collision (see SOLVERS.md §4 /
deformables.md); the grip is genuine contact friction, so an object drops physically when the pads
open (no spring/latch holding it).

## Obstacle (table) non-penetration

The robot-side hidden `robot_contact_table` collider stops the gripper at the table surface; the
object-model table is filtered against the dynamic proxies (a dynamic proxy re-pinned against the
static table resolves explosively). Both come from `deformableManipulationTools.params.TABLE` /
`deformableManipulationTools.robot.build_franka_robot` / `build_gripper_proxies`.
