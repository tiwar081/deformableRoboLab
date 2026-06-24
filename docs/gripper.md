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
(the swept cable) passes straight through the gripper **palm and EE**. The gripper proxy collision is
**CENTRALIZED and identical for every VBD demo** (no per-demo knobs): (1) each finger proxy carries the
finger's TRUE collision shapes (the sparse URDF boxes, copied one-for-one — the more faithful collider;
an earlier single-AABB box was reverted as a non-physical fattening). The gaps between the sparse boxes
let the swept cable clip ~1 mm into the **fingers** (accepted; the palm blocker below stops the larger
palm/wrist penetration). (2) a THIRD proxy — a single synthetic box
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
   `TwoWayProxyCoupling.grip_force_signal`, recomputed at the end of `harvest()`) **strictly exceeds**
   `force_target` for `latch_debounce` substeps (and the jaws have closed past `latch_arm_margin`),
   **FREEZE the finger target at the MEASURED finger position minus the inward bite**.
3. **Hold / release** — hold the frozen width; if the window has a release, smoothstep back open over
   `[release_start, release_end]`.

### The grasp MODE: compressible vs. incompressible (centralized, the only per-window knob)

`force_target` and the inward bite are NOT tuned per demo — they are derived from one declarative
flag, `GraspWindow.compressible`, so the same robot grips like-for-like objects identically:

- **Incompressible** (rigid bodies + the cable — NO particles): `force_target = 0`, so the strict-`>`
  latch fires the instant the grip signal first becomes positive — i.e. **FIRST contact** (both pads
  pressing; the signal is exactly 0 until the pads touch). It then bites `GRIP.grasp_interference`
  (1 mm) past the measured contact. This is the geometry-FREE equivalent of the old preset close width
  `object_half + margins − grasp_interference`, and it reproduces it: the rigid cube latches ~26 mm
  (= old `cube_half + margins − 1 mm`), the rubik's cube ~29 mm, the plate handle ~13 mm.
- **Compressible** (soft FEM block): `force_target = GRIP.force_target` (8 N — a compliant block needs
  a real squeeze threshold, not first-contact) and bites **ZERO** past it. Any inward bite crushes the
  already-deforming block (the old 2.5 mm bite drove it to a 462 N non-deterministic crush); biting 0
  holds it gently.

```python
self.grasp_windows = [GraspWindow(close_start=2.8, close_end=4.6, grip_bite=CABLE.grip_bite)]  # cable: hold to end
self.grasp_windows = [GraspWindow(3.2, 4.4, release_start=8.0, release_end=8.6)]                # rigid: incompressible default
self.grasp_windows = [GraspWindow(3.2, 4.4, release_start=8.0, release_end=8.6, compressible=True)]  # soft: pick&place
```

### Why these choices are load-bearing

- **Latch the MEASURED finger position, not the commanded ramp.** The open-loop command races far
  ahead of the effort-limited (`FRANKA.finger_effort` = 20 N) fingers — the fingers are ~11 mm at
  contact while the command has raced to ~3 mm. Freezing the *command* leaves the fingers straining
  inward → **600–720 N** dynamic spikes during the cable sweep. Freezing the *measured*
  `robot_state.joint_q[finger]` holds the actual contact width → bounded force.
- **The cable needs a BIGGER bite than the solid bodies** (`GraspWindow.grip_bite` override =
  `CABLE.grip_bite` ≈ 3 mm). The cable grip is a loose LINE-contact cage: its static harvested force
  decays to ~0 at the working width (force only appears as the pads compress the rod, or dynamically
  on lift), and a light rolling rod registers "first contact" ~1.7 mm WIDER than `object_half + margins`
  (vs a solid box, which stops sharply at its face). So first-contact + the default 1 mm leaves the
  cage too open and the cable slips on the sweep; ~3 mm reproduces the old preset's firm cage. A solid
  box doesn't need this — its flat multi-point patch already reads firm at 1 mm (~150 N).

### Knobs (centralized in `params.GripConfig` / per object in params)

- **`compressible`** (`GraspWindow`, default `False`) — selects the mode above. The ONE physics knob a
  demo declares; everything else is derived.
- **`force_target`** (`GRIP`, 8 N) — the both-pads squeeze threshold used for **compressible** grasps.
  Incompressible grasps latch at first contact (`force_target = 0`).
- **`grasp_interference`** (`GRIP`, 1 mm) — the inward bite for **incompressible** grasps, past the
  force-discovered first contact. Held grip ≈ `proxy_ke · bite · (contact points)`, so the same bite
  gives ~150 N on a flat box (multi-point patch) but only a loose cage on a thin rod — hence:
- **`CableConfig.grip_bite`** (≈ 3 mm) — the cable's bite override (`GraspWindow.grip_bite`), the lone
  per-object bite, justified by the cable's loose line-contact regime. Compressible grasps bite 0.
- **`proxy_ke`** (5e4 N/m), **`proxy_kd`** (1e2 N·s/m abs), **`proxy_mass`** (10 kg) — proxy contact
  stiffness/damping/inertia (see CLAUDE.md for the `kd` re-derivation landmine).
- **`min_close_width`** (1 mm), **`latch_debounce`** (2), **`latch_arm_margin`** (3 mm) — the
  fully-closed floor (if the grasp never reaches the threshold) and the false-latch guards.

**Contact geometry still sets the force scale.** The held grip ≈ `ke · penetration` over every
contact point, so a flat box face reads higher than a curved/small object at the same bite; bounded
and net-to-EE ≈ 0 either way. Verify with an instrumented headless run logging
`GraspExample.grip_force_norms()` (per-pad `[left, right]` N) + `--test`.

**All VBD demos use force-stop:** `cable_rigidCube`, `soft_pickplace`, `cable_soft`, `rigidCube_soft`,
`soft_compression`, and `pickplace_ycb_vbd` (the last declares **two** `GraspWindow`s — rubik's cube
then banana). Only `soft_pickplace` is `compressible=True`; the rest are incompressible (the two cable
demos add the `CABLE.grip_bite` override). The rigid-only MuJoCo demo `pickplace_ycb_franka` keeps a
fixed close target (`MUJOCO_GRIP.close_target`, true two-way contact stops the pads).

### Legacy `grasp_interference` (revived as the incompressible bite)

`grasp_interference` (1 mm) used to set the geometric preset close target
`gripper_closed = object_half + object_margin + proxy_margin − grasp_interference`. The force-stop
controller now uses it as the **inward bite for incompressible grasps** past the force-discovered
first contact — so the discovered-online grasp lands at the same firm width the preset hand-computed,
without the demo ever needing the object's size.

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
