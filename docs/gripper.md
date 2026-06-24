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

### Palm/EE blocker proxy (`build_gripper_proxies(palm_proxy=True)`)

The Franka hand is collapsed into `link7` and its collider exists only in MuJoCo, so a fast deformable
(the swept cable) passes straight through the gripper **palm and EE**; and each finger's collider is 4
sparse boxes (smaller than the rendered finger), so the cable also clips the **fingers** (~1 mm). The
gripper proxy collision is therefore **CENTRALIZED and identical for every VBD demo** (no per-demo
knobs): (1) each finger proxy is ONE box = the AABB of its 4 boxes (gap-free, max-y grasp face
unchanged so the grip is identical, and cheaper); (2) a THIRD proxy — a single synthetic box
(`GRIP.palm_box_half/offset`, in the `link7`/EE frame, spanning wrist→hand up to just below the finger
origins, 6 cm clear of the fingertip grasp zone) mirrors the EE body and blocks the palm/EE; and (3)
the proxies are particle-colliding (`GRIP.proxy_gap`). All built unconditionally in
`build_gripper_proxies`. It is a
**blocker only**: re-pinned to the EE each substep, **not harvested and not in the grip signal**
(grip = `proxy_bodies[:2]`, the fingers), so it never perturbs the force-stop grip. The coupling
loops over N proxies generically for sync + net-to-EE; the framework pins any proxy beyond the
fingers to the EE. It cut cable-through-palm from 219 frames (37 mm) to 0. **Enabled only on the
extended-deformable (cable) demos** — compact-object pick-place demos have nothing that reaches the
palm, and the extra body perturbs the (separately fragile) soft grip latch, so they leave it off.

## Force-stop close — no preset width (`grip.GripController`)

The grip width is **discovered online from the contact force**, not preset from object geometry
("specify force, get emergent geometry"). `grip.GripController` (built by the framework, one per
demo that declares `grasp_windows`) owns the finger DOFs; the demo's policy kernel writes only the
**arm** DOFs (`set_arm_targets`). This is the only legal place to close a force loop: the fingers
feel nothing in their own (MuJoCo) solver, so we derive a finger **position** command from a force
**reading** — never inject force into the finger DOF (the [stability invariant](#stability-invariant-do-not-violate)
holds unchanged).

Per substep the controller (a `dim=1` Warp kernel, persistent on-device `latch_state`, CUDA-graph
safe) does, per declared `params.GraspWindow`:

1. **Close** — smoothstep the finger target from `gripper_open` toward `GRIP.min_close_width` over
   `[close_start, close_end]`.
2. **Latch** — when the both-pads grip signal `min(|f_left|,|f_right|)` (device-side
   `TwoWayProxyCoupling.grip_force_signal`, recomputed at the end of `harvest()`) crosses
   `force_target` for `latch_debounce` substeps (and the jaws have closed past `latch_arm_margin`),
   **FREEZE the finger target at the MEASURED finger position minus `grip_bite`**.
3. **Hold / release** — hold the frozen width; if the window has a release, smoothstep back open over
   `[release_start, release_end]`.

A demo declares only the policy timing + force:
```python
self.grasp_windows = [GraspWindow(close_start=2.8, close_end=4.6, force_target=10.0)]   # cable: hold to end
self.grasp_windows = [GraspWindow(3.2, 4.4, release_start=8.0, release_end=8.6, force_target=8.0)]  # soft: pick&place
```

### Why these two choices are load-bearing

- **Latch the MEASURED finger position, not the commanded ramp.** The open-loop command races far
  ahead of the effort-limited (`FRANKA.finger_effort` = 20 N) fingers — the fingers are ~11 mm at
  contact while the command has raced to ~3 mm. Freezing the *command* leaves the fingers straining
  inward → **600–720 N** dynamic spikes during the cable sweep. Freezing the *measured*
  `robot_state.joint_q[finger]` holds the actual contact width → bounded force.
- **Bite a small `grip_bite` past the discovered contact.** A cable grip is a loose cage: at the
  working width the static harvested force is ~0 (force only appears once the pads COMPRESS the
  rigid rod, or dynamically on lift), so measured-first-contact alone is too loose and the cable
  slips on the lift. `measured − grip_bite` gives a firm but bounded squeeze ≈ `proxy_ke · grip_bite`
  (5e4 · 2.5 mm ≈ 125 N ceiling), and the bite is relative to the FORCE-discovered contact, so it
  stays geometry-independent.

### Knobs (centralized in `params.GripConfig`)

- **`force_target`** (default 8 N; per-`GraspWindow` overridable) — the both-pads threshold that
  signals "contact made". A compliant object wants it low; it is the contact *trigger*, not the
  held grip.
- **`grip_bite`** (default 2.5 mm; per-`GraspWindow` overridable) — inward squeeze past the
  discovered contact; the firmness knob. **It is GEOMETRY-DEPENDENT** — held grip
  ≈ `proxy_ke · grip_bite · (contact points)`, so a flat object gripped by the flat pads (large
  multi-point patch) needs a much smaller bite than a thin rod: the cable holds at 2.5 mm (≈125 N,
  line contact), but the rigid cube / plate handle would hit **kN** at 2.5 mm, so they use
  `grip_bite=0` (latch at first solid contact — flat/solid objects already have a real static grip
  there) and read ~65–73 N. Set it per object on the `GraspWindow`.
- **`proxy_ke`** (5e4 N/m), **`proxy_kd`** (1e2 N·s/m abs), **`proxy_mass`** (10 kg) — proxy contact
  stiffness/damping/inertia (see CLAUDE.md for the `kd` re-derivation landmine).
- **`min_close_width`** (1 mm), **`latch_debounce`** (2), **`latch_arm_margin`** (3 mm) — the
  fully-closed floor (if the grasp never reaches `force_target`) and the false-latch guards.

**Contact geometry still sets the force scale.** The held grip ≈ `ke · penetration` over every
contact point, so a flat box face reads higher than a curved/small object at the same bite; bounded
and net-to-EE ≈ 0 either way. Verify with an instrumented headless run logging
`GraspExample.grip_force_norms()` (per-pad `[left, right]` N) + `--test`.

**All VBD demos use force-stop:** `cable_rigidCube`, `soft_pickplace`, `cable_soft`, `rigidCube_soft`,
`soft_compression`, and `pickplace_ycb_vbd` (the last declares **two** `GraspWindow`s — rubik's cube
then banana). Only the rigid-only MuJoCo demo `pickplace_ycb_franka` keeps a fixed close target
(`MUJOCO_GRIP.close_target`, true two-way contact stops the pads — already geometry-independent).

### Legacy `grasp_interference` (deprecated)

`grasp_interference` (1 mm) used to set a geometric close target
`gripper_closed = object_half_width + object_margin + proxy_margin − grasp_interference`. The
force-stop controller replaced it on every VBD demo; the param remains only as a documented default
and is no longer read by any example.

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
