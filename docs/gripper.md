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
**CENTRALIZED and identical for every VBD demo**: (1) each finger proxy carries the finger's collision
shapes — the FR3's sparse URDF boxes copied one-for-one, or a CONVEX_MESH finger deep-copied so the
object model owns its BVH. The gaps between sparse boxes let the swept cable clip ~1 mm
into the **fingers** (accepted; the palm blocker below stops the larger palm/wrist penetration). (2) a THIRD proxy — a single synthetic box
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

## Force control — target grasp force (`grip.GripController`)

The grip width is **derived from the contact force**, not preset from object geometry. `grip.GripController`
(built by the framework, one per demo that declares `grasp_windows`) owns the finger DOFs; the demo's policy
kernel writes only the **arm** DOFs (`set_arm_targets`). This is the only legal place to close a force loop:
the fingers feel nothing in their own (MuJoCo) solver, so we derive a finger **position** command from a force
**reading** — never inject force into the finger DOF (the [stability invariant](#stability-invariant-do-not-violate)
holds unchanged). **ONE unified control law for every object** (rigid, cable, AND soft) — the grasp does not
depend on the object type. **The one per-demo knob is the target grasp force** (`GraspWindow.force_target`); the
close **speed** is centralized (`GRIP.grip_rate_max`) — identical for every grasp.

**Cloth on the force grip — RESOLVED by the target-relative law.** A pinched thin shell's
achievable squeeze is ~0.3–3 N. Under the old fixed constants a ≤2 N target was literally dead (the
2 N absolute deadband exceeded the whole error range: the measured 1 N trial latched a loose ~14 mm
jaw and shed the wad) and the workaround — an unreachable 8 N target — crept the jaw shut forever
(there is no width floor; the regulator alone must stop the close). With `window_params` the cloth
demo declares `force_target=2 N` (inside the achievable range): per-window gain 1.5e-3, deadband
0.13 N — the regulator latches at the 0.3 N engage floor, tightens at ~2.4 mm/s, and CONVERGES to a
stable force equilibrium near the proven ~8–9 mm jaw, staying live to re-tighten a shedding pinch.
An abrupt commanded zero-gap pinch still expels a shell — never do that with a schedule.

### The law: bidirectional ASYMMETRIC admittance (a `dim=1` Warp kernel, CUDA-graph safe)

A velocity-form force regulator on the **closing-axis-projected** squeeze that may move the jaw open OR closed:

1. **Approach** — from `close_start`, close at the centralized speed `GRIP.grip_rate_max` until the filtered
   squeeze first reaches the engage threshold `clamp(engage_frac·target, engage_floor, engage_cap)`; seed the
   regulator at the **measured** finger width (the open-loop command races ahead of the effort-limited fingers,
   so seed from what they actually reached). **The engage threshold scales with the target** so a low-target SOFT
   grip engages at a light touch — *before* the fast approach over-compresses the compliant block — while a
   high-target rigid/cable grip engages firmly (a firmer seed is what gives the cable its cage).
2. **Regulate** — `F_filt = EMA(grip_squeeze_signal, force_filter_tau)`; **direction-dependent
   gains** from `GripConfig.window_params(target)`: when UNDER target, `w_dot = k_close·(err + db)`
   (responsive — chase the target / restore a **decaying** grip); when OVER target,
   `w_dot = k_open·(err − db)` with `k_open = k_close/k_open_ratio` (reluctant — the anti-drop
   asymmetry). Both directions are rate-capped by the SAME physical `grip_rate_max`.
3. **Release** (optional window) — smoothstep the held width back to `gripper_open`.

The **gain asymmetry is the stability mechanism**, and it is what lets a force regulator hold a passive object
whose contact force is spiky and **load-dominated**. "Grab firmly, release reluctantly": a lift/sweep LOAD or a
rigid SPIKE transiently dwarfs the target, but the open gain is `k_open_ratio` (20×) smaller than the close gain,
so jaw retreat during a transient of size α·target lasting T is only `(3e-3/k_open_ratio)·α·T` (~0.15 mm for a
2×-target, 0.5 s spike) — the grasp survives — while a genuine sustained over-force or a decay still moves the jaw
the right way. The EMA filter eats the shortest spikes before the gains even see them. The RATE CAP is symmetric
and physical (`grip_rate_max` both ways — a real jaw opens as fast as it closes; the asymmetry belongs in the
software gains, not disguised as a hardware limit, which is what the old 200×-slower open cap was). The signal is
the **closing-axis projection** (`grip_squeeze_signal`, below), not the raw magnitude, so the load (tangential to
the jaw axis) is partly rejected before the asymmetry even sees it.

**The target is the only thing that differs between objects** (and the target-scaled engage threshold falls out of
it automatically). A soft block needs a gentle target (~5 N) — its engage threshold is then small, so it is held
without crush; no "freeze" mode is needed, the regulator simply settles there. A rigid box (~30 N) has a steep
force-width curve, so the target is a true squeeze setpoint. The cable is the degenerate case (below), where the
target instead sets the cage geometry. Tune the per-object target up if it slips, down if it crushes.

```python
self.grasp_windows = [GraspWindow(close_start=2.8, close_end=4.6, force_target=30.0)]                  # cable: held to end
self.grasp_windows = [GraspWindow(3.2, 4.4, release_start=8.0, release_end=8.6)]                       # rigid: GRIP default target
self.grasp_windows = [GraspWindow(3.2, 4.4, release_start=8.0, release_end=8.6, force_target=5.0)]     # soft: gentle target
```

### Why the cable target is what it is

The cable is the pathological case: its static projected squeeze is **flat at ~3–6 N at every width** (the rod
rolls/deforms rather than building force as the pads close), so there is no width where closing raises force to a
higher setpoint. The grasp is therefore **geometric** — the jaw must reach a cage tight enough to trap the 8 mm-radius
rod (~8–9 mm). `force_target = 30 N` drives the asymmetric regulator to that cage (`w` settles ~8.5–9.3 mm, matching
the old design's 8.4 mm) at a physical force (median ~100 N, dynamic peak ~170 N). Lower targets leave the cage too
loose (the rod slips on the lift); much higher targets crush the rod to a non-physical force.

### Knobs (all centralized in `params.GripConfig`; one per-demo target)

- **`force_target`** (`GraspWindow`, default `None`) — **the one per-demo knob**: the target grasp force [N].
  `None` → the centralized default `GRIP.force_target` (30 N). **Projected (closing-axis) units.**
- **Target-relative, derived per window by `GripConfig.window_params(ft)`** — the ONE centralized
  derivation, consumed by the controller packing AND the instrumentation reference bands:
  - `k_adm` → `k_close = min(k_adm·adm_ref_force/ft, k_adm_cap)` and `k_open = k_close/k_open_ratio`
    (ratio 20): relative-error admittance with the anti-drop asymmetry in the GAINS — the
    full-scale-error close speed `k_close·ft = 3e-3 m/s` is target-independent, so a 2 N cloth grasp
    regulates as briskly (in fraction-of-target units) as a 30 N cube grasp, and every grasp opens
    20× more reluctantly than it closes. `k_adm_cap` (2e-3) guards discrete-loop stability
    (`k·ke·dt ≪ 1`) at very low targets.
  - `grip_force_deadband` → `db_w = max(grip_force_deadband·ft/adm_ref_force, 0.1 N)`: a limit-cycle
    killer is a fraction-of-setpoint quantity (1/15 of target); the absolute floor is the
    measurement-noise band. (The old 2 N ABSOLUTE deadband exceeded low targets entirely —
    `w_dot = 0` forever for ft ≤ 2 N.)
  - **Anchor**: the close side of `window_params(adm_ref_force=30)` equals the legacy `(1e-4, 2.0)`
    exactly — the long-tested 30 N CLOSING behavior (cable cage, rigid cube, rubik's) is
    bit-identical; the OPEN side is the redefined gain-asymmetry mechanism (validated behaviorally
    on the cable, the spike-dominated worst case).
- **Fixed — physical robot properties (sim2real: not retunable on real hardware, so not in sim):**
  **`grip_rate_max`** (0.04 m/s, just under the real Franka Hand's ~0.05 m/s max finger speed —
  SYMMETRIC: it caps opening AND closing, and is the blind-approach speed),
  **`engage_floor`** (0.3 N — the contact-DETECTION floor of a real force estimate; scaling it down
  would fire engage on noise), **`force_filter_tau`** (0.05 s — set by the signal's noise spectrum),
  and the finger actuator gains/effort (`RobotConfig`).
- **Fixed — scale-free by function:** **`engage_frac`** (0.15, already relative: engage =
  `clamp(engage_frac·target, floor, cap)`), **`engage_cap`** (2 N — bounds the approach-phase
  ram-in, which is target-independent), **`k_open_ratio`** (20, dimensionless — the close/open gain
  asymmetry; 10 lost the cable cage in cable_soft's long high-spike sweep; raise it if a spike-dominated grasp opens too readily, it is the only anti-drop knob).
- **`proxy_ke`** (5e4 N/m), **`proxy_kd`** (1e2 N·s/m abs), **`proxy_mass`** (10 kg) — proxy contact
  stiffness/damping/inertia (see CLAUDE.md for the `kd` re-derivation landmine).

**The grip signals.** `TwoWayProxyCoupling` recomputes two both-pads readings at the end of `harvest()`:
`grip_squeeze_signal = min` of the per-pad reactions **projected onto the jaw closing axis** (the admittance
signal — it rejects the lift load, which is tangential to that axis) and `grip_force_signal = min(|f_left|,
|f_right|)` (raw magnitude, kept as a diagnostic). Verify with an instrumented headless run logging
`coupling.grip_signal_values()` / `coupling.raw_force_norms()` + `--test`.

**Every VBD demo uses the same controller** (`cable_rigidCube`, `soft_pickplace`, `cable_soft`, `rigidCube_soft`,
`soft_compression`, `pickplace_ycb_vbd` — the last declares **two** `GraspWindow`s, rubik's cube then banana),
differing only in `force_target`. The rigid-only MuJoCo demo `pickplace_ycb_franka` has no coupling
(`force_stop_enabled=0`) → a plain smoothstep close to `MUJOCO_GRIP.close_target`, with true two-way contact
stopping the pads.

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
