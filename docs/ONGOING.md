# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just changed
and not yet settled, and any working hypotheses. Keep it lean — when something is proven and
durable, promote it to CLAUDE.md (or the relevant `docs/` file) and delete it here. Reset this file
at the start of each new big task.

## DONE (2026-06-24): grasp MODE split — incompressible (first-contact + bite) vs compressible (force + no bite)

Reworked the force-stop grasp so it is centralized on ONE declarative flag, `GraspWindow.compressible`,
which fixes both the weak-rigid-grasp (Q1) and the soft-over-grasp (Q2) problems. `GraspWindow` no
longer carries per-demo `force_target`/`grip_bite` numbers; the controller derives them from the mode:

- **Incompressible** (rigid bodies + cable — NO particles): `force_target = 0`. The latch test is now
  **strict `sig > ft`**, so `force_target = 0` fires the instant the grip signal first becomes positive
  — i.e. FIRST contact (both pads) — then bites `GRIP.grasp_interference` (1 mm). This is the
  geometry-free equivalent of the OLD preset `object_half + margins − grasp_interference`, and it
  reproduces it (rigid cube latches 25.99 mm ≈ old 26 mm; rubik's 29.0 mm; plate handle 13.0 mm).
  **Q1 FIXED:** the 1 kg cube now grips firmly at **157 N** (was a near-zero barely-touching grip with
  the old `grip_bite=0` force-stop) and carries.
- **Compressible** (soft FEM block): keep `force_target = GRIP.force_target` (8 N) but bite **ZERO**.
  **Q2 FIXED:** `soft_pickplace` grasp-phase peak **15 N** (steady ~2–4 N), vs the old 2.5 mm bite that
  drove a 462 N non-deterministic crush. The deep latch (~3.6 mm) is unchanged, but with no extra bite
  the held force is gentle and the block recovers.

### The cable needed a bigger bite (the lone per-object override)
The cable is incompressible but a **loose LINE-contact cage** (static grip force decays to ~0; a light
rolling rod registers first-contact ~1.7 mm WIDER than `object_half + margins`, unlike a solid box that
stops at its face). So first-contact + the default 1 mm left the cage too open and the cable **slipped
on the sweep** (`check_physics` fail). Fix: `CableConfig.grip_bite` (≈ **3 mm**), passed as the
`GraspWindow.grip_bite` override — the ONE per-object bite. Cable now latches ~8.7 mm and holds
through the sweep (**cable_rigidCube 192 N PASS**; cable_soft 163 N, cable held).

### Verified (headless `--viewer null --device cuda:0`, instrumented; FINAL = release fix + reverted URDF fingers)
| demo | object | mode | bite | latched | GRASP-phase peak | whole-run max |
|---|---|---|---|---|---|---|
| cable_rigidCube | cable | incompr | 3 mm | 8.34 mm | 170 N | 170 N |
| cable_soft | cable | incompr | 3 mm | 8.35 mm | 136 N | 136 N |
| soft_pickplace | soft block | **compr** | 0 | ~4.8 mm | 30 N‡ | = grasp peak |
| rigidCube_soft | rigid cube | incompr | 1 mm | 25.99 mm | 158 N | 158 N |
| soft_compression | plate handle | incompr | 1 mm | 13.01 mm | 419 N | 419 N |
| pickplace_ycb_vbd | rubik's cube | incompr | 1 mm | 29.01 mm | 154 N | 154 N (incl its release) |
| pickplace_ycb_vbd | banana | incompr | 1 mm | 20.65 mm | 30 N | 30 N (incl its release) |

(pickplace_ycb_vbd has TWO grasps in one run; the per-OBJECT max above is over that object's own
window incl release — the rubik's cube 154 N is the demo-wide whole-run max, the banana never exceeds
~30 N. Both objects' release ramps add nothing over their grasp peak — the release-jolt fix holds.)

After the release-jolt fix (below) the whole-run max EQUALS the grasp peak — no transient exceeds the
sustained grip. soft_compression's 389 N is the plate handle loaded during the press-onto-block phase
(bounded, holds). ‡soft_pickplace's grasp peak varies run-to-run (the documented soft-latch
non-determinism — deep FP-sensitive latch); the release fix is orthogonal to it. **cable_soft
`check_physics` is non-deterministic** — it sometimes fails on *"the soft body fell through the table"*,
which is the token block knocked off the table EDGE by the swept cable (accepted cosmetic behavior),
NOT the cable grasp (the cable holds either way).

### DONE (2026-06-24): release-jolt fix — fingers no longer crush the object at the instant of release
**Symptom:** right when the gripper releases any object WITH a release window (i.e. everything except
the two cable demos, which hold to the end), the fingers jolted INWARD once before reopening — the
source of the 1.7–3.2 kN "release" spikes above. **Root cause** (`grip._grip_force_stop_kernel`, RELEASE
branch): the reopen smoothstep interpolates `base → gripper_open` with `base = latched_w` only while
`latched > 0.5`, but the branch ALSO set `latched = 0` and persisted it. So only the FIRST release
substep used `base = latched_w`; every later substep read `latched = 0` → `base = close_target` (≈1 mm,
fully closed), and with the smoothstep `alpha` still ~0 the finger command snapped toward 1 mm for a
step — a one-substep crush — before opening. **Fix:** DON'T clear `latched` in the release branch (the
latch is re-armed for the next grasp in the `t < cs` branch), so `base` stays `latched_w` for the whole
ramp and the command opens monotonically. Verified: whole-run max == grasp peak on all four release
demos (was 1.7–3.2 kN). The AABB finger proxy boxes were NOT the cause of the release jolt; the user
nonetheless asked to revert them to the true URDF finger colliders (see next entry — done, for fidelity).

### DONE (2026-06-24): reverted the finger proxies from the single AABB box to the TRUE URDF colliders
The finger proxies again copy the Franka finger's sparse collision boxes one-for-one (pad + edges +
knuckle) instead of the gap-filled AABB box. The AABB was a non-physical fattening of the collider;
the real boxes are the faithful geometry (project priority #1). **Tradeoff (re-accepted):** the gaps
between the sparse boxes let the swept cable clip ~1 mm into the fingers again — the residual the AABB
had removed; the palm/EE blocker proxy is UNCHANGED, so the larger palm/wrist penetration stays fixed.
Re-verified all 6 demos (grasp + check_physics) after the revert; re-rendered all 6.

## DONE (2026-06-23): palm/EE blocker proxy — stop the swept cable penetrating the gripper

**Symptom:** in the cable demos the swept cable passed through the gripper and the EE. **Root cause
(verified):** the two finger proxies carry the FULL finger collision geometry (4 boxes each) and
block the cable fine (no tunneling: ~0.9 mm/substep ≪ 11 mm pad+margin; proxies track the fingers
<1 mm) — but the Franka hand is collapsed into `link7` and its collider lives ONLY in MuJoCo, so the
VBD world had NO collider for the palm/wrist. So the cable went through the uncovered palm/EE, not
the fingers.

**Fix:** `grip.build_gripper_proxies(palm_proxy=True)` appends a THIRD proxy — a single synthetic box
(`GRIP.palm_box_half/offset`, in the link7/EE frame, spanning wrist→hand up to just below the finger
origins, 6 cm clear of the fingertip grasp zone) mirroring the EE body. It is a BLOCKER only:
re-pinned to the EE each substep, NOT harvested, NOT in the grip signal (grip = the two finger
proxies, `proxy_bodies[:2]`). The coupling already loops over N proxies for sync + net-to-EE; the
harvest/grip-signal stay on the two finger proxies, so the palm adds no kernel cost beyond one body.
`framework._make_coupling` pins extra proxies (beyond the fingers) to the EE. Result: cable-in-palm
region **219 frames (37 mm deep) → 0 frames (0 mm)**; cable grasp unchanged (~126 N, PASS, capture-safe).

**Scope (UPDATED 2026-06-23): CENTRALIZED — every VBD demo's gripper now uses the SAME proxy contact
geometry** (solid AABB fingers + palm/EE blocker + particle collision), built unconditionally by
`build_gripper_proxies` from `GRIP` (`palm_box_*`, `proxy_gap`). The per-demo knobs (`palm_proxy`,
`gap`, `has_particle_collision`) are REMOVED — demos can no longer tweak the robot contact boxes.
This is uniform-robot-by-design even though it reintroduces the soft-grip perturbation on the
pick-place demos (see concern; accepted for now, reported). Arm links above the wrist are still
MuJoCo-only (uncovered); out of scope.

### Follow-up (2026-06-23): the cable also clipped the FINGERS, not just the palm
The fingers ARE proxied, but each Franka finger collider is **4 sparse boxes** (pad + 2 edges +
knuckle) — SMALLER than the rendered finger and with gaps between them. A swept cable clips ~1 mm into
them (penalty-contact compliance of the fat 8 mm cable) and, because the visual finger mesh is fuller
than the sparse colliders, the visual overlap looks larger. My earlier "fingers fully block the cable"
was WRONG — it used a node-center metric (not the cable capsule) + a tunneling argument that missed
this. Verified with a capsule-aware surface distance (grip-phase sanity ~1.8 mm): cable surface goes
**−0.2 to −1.6 mm** into the left-finger boxes during the sweep.

**Fix (under the same `palm_proxy=True` opt-in):** build each finger proxy as ONE box = the **AABB of
the 4 finger boxes**, filling the gaps and presenting a solid finger. Its grasp-facing (max-y) face
equals the pad's outer face, so the grip is preserved (cable grasp PASS, ~150 N, capture-safe);
fewer shapes (cheaper). **Residual concern:** the ~1 mm is penalty-contact compliance (the gripped
fat cable sinks ~1 mm into the pad) — inherent to the VBD penalty contact; reducing it needs stiffer
contact (`proxy_ke`/cable `contact_ke`), which risks instability, so left as-is ("better, not perfect").

### ✅ RESOLVED (2026-06-24) — soft over-grasp crush (was the deferred concern below)
The soft block still latches deep (~3.6 mm; the signal rises slowly, ≈1.6 N at a 13 mm gap, so 8 N is
only reached deep), but with `compressible=True` biting **ZERO** past the latch the held grip is gentle
(grasp-phase peak ~15 N, steady ~2–4 N) and the block recovers — vs the old 2.5 mm bite that drove a
~462 N crush. The over-grasp was the BITE on top of the deep latch, not the deep latch itself; removing
the bite (not the threshold) fixes the force. Residual: the geometric close depth is still deep (it is
inherent to the slow-rising soft signal); a future improvement would latch on the SETTLED signal so the
depth itself is shallower, but the FORCE is now bounded and gentle so this is no longer urgent.
*(Original concern, for history:* the held grip varied run-to-run ~1.8 N→462 N for the same config
because the 2.5 mm bite amplified the razor-sensitive deep latch; palm-on pushed it to ~1500–1850 N.)

## DONE (2026-06-22): centralized MuJoCo/VBD object routing (promoted to CLAUDE.md + solver-architecture.md)

The framework now picks the object solver CENTRALLY in `GraspExample.__init__`: deformable present iff
`object_builder.particle_count > 0` **OR** a `CABLE` joint exists (a rod/cable has no particles — must
be detected by joint type, else `cable_rigidCube` misroutes to MuJoCo). **rigid-only → one `SolverMuJoCo`**
(objects merged into the robot builder via `add_builder`, true two-way grasp, CCD on, fixed close
target); **deformable present → `SolverMuJoCo` robot + `SolverVBD` objects + dynamic proxies** (the
original bridge, preset widths). `pickplace_ycb_franka` auto-routes to MuJoCo; `pickplace_ycb_vbd_franka`
(= same scene + a token soft cube) auto-routes to VBD — the A/B routing twin. MuJoCo rigid-only ≈ 2.2×
faster (13.7 vs 6.1 fps). The reverted hybrid (a rigid body in MuJoCo AND a VBD proxy) was unstable for
fast impacts onto light FEM particles. Kept fixes: CCD on the robot solver; soft-harvest reaction sign
(`grip._harvest_soft_wrench_kernel` now `−n·ke·pen`).

**Consequence for the prior task:** the rubik's-cube "asymmetric stick at release" was on
`pickplace_ycb_franka`, which is now MuJoCo (true two-way contact, no proxy patch) — so that
sticky-release task is MOOT on that demo. It would only recur on a VBD-path (deformable) demo.

## DONE (2026-06-22): force-feedback grasp for the VBD/proxy path (all 6 VBD demos)

Replaced the geometric **preset gripper width** (`gripper_closed = object_half + margins −
grasp_interference`) on the VBD/proxy path with a centralized **force-stop controller**
("specify force, get emergent geometry"). Implemented + wired + verified end-to-end on the two
priority demos. **The fingers are still position-controlled and feel nothing in their own
(MuJoCo) solver — we only derive a finger POSITION command from a force READING; no force is ever
injected into the finger DOF, so the net-to-EE invariant is untouched.**

### What was built
- `grip.GripController` (+ `_grip_force_stop_kernel`, dim=1, persistent device `latch_state`) and
  `grip._reduce_grip_signal_kernel` writing `TwoWayProxyCoupling.grip_force_signal` =
  `min(|f_left|,|f_right|)` [N] at the end of `harvest()` (one-step stale; min ⇒ BOTH pads must
  engage). `params.GraspWindow` dataclass; `GripConfig` knobs `force_target`/`grip_bite`/
  `min_close_width`/`latch_debounce`/`latch_arm_margin`.
- `framework.set_robot_targets` is now CONCRETE: `set_arm_targets(substep)` (demo writes arm DOFs
  0..6 only) + `grip_controller.step(substep, robot_state_0.joint_q)` (writes finger DOFs 7,8).
  Controller built after routing; `force_stop_enabled = coupling is not None` (rigid-only path =
  smoothstep close to `MUJOCO_GRIP.close_target`, no latch). Backward-compatible: a demo that still
  overrides `set_robot_targets` and declares no `grasp_windows` is untouched.
- Demos declare only `self.grasp_windows = [GraspWindow(...)]` (close/release timing + optional
  per-window `force_target`/`grip_bite`); the geometric `gripper_closed` is gone. **ALL VBD demos
  migrated**: `cable_rigidCube`, `soft_pickplace`, `cable_soft`, `rigidCube_soft`, `soft_compression`,
  and `pickplace_ycb_vbd` (the last with **two** grasp windows — rubik's cube then banana). The
  rigid-only MuJoCo demo `pickplace_ycb_franka` is unchanged (already geometry-independent).

### The control law (and the two findings that shaped it)
Close (smoothstep over the window) toward `min_close_width`; when `min`-of-pads ≥ `force_target`
(debounced, and the jaws have closed past `latch_arm_margin`) → **FREEZE at the MEASURED finger
position minus `grip_bite`**; hold; release window reopens.
1. **Latch the MEASURED finger position, NOT the open-loop command.** The command ramps far ahead
   of the effort-limited (20 N) fingers — fingers are ~11 mm at contact while the command has raced
   to ~3 mm. Latching the command leaves the fingers straining inward → dynamic spikes **600–720 N**
   during the cable sweep (this reproduced the old "over-penetration" symptom). Reading
   `robot_state_0.joint_q[finger]` and freezing there fixed it.
2. **A small inward `grip_bite` past the discovered contact is required for thin/loose-cage objects.**
   The cable grip is a loose cage: at the working width the static harvested force is ~0 (force only
   registers once the pads COMPRESS the rigid rod, or dynamically on lift). So measured-first-contact
   alone is too loose → the cable slips out on the lift (grip→0). Freezing at `measured − grip_bite`
   (2.5 mm) restores a firm grip ≈ `ke·bite·points`, relative to the FORCE-discovered contact (not
   object size).
3. **`grip_bite` is GEOMETRY-DEPENDENT and is per-`GraspWindow`.** The held force ≈ `ke·bite·(contact
   points)`, and a flat object gripped by the flat pads makes a LARGE multi-point patch — so the same
   2.5 mm bite that gives 125 N on the cable (line contact, few points) gives **4 kN on the rigid
   cube** and **2.7 kN on the plate handle**. Flat/solid objects use `grip_bite=0` (latch at first
   solid contact — they already have a real static grip there) → ~65–73 N; thin loose-cage objects
   (cable) use ~2.5 mm. So `GraspWindow.grip_bite` overrides the `GRIP.grip_bite` default per object.

### Verified (headless `--viewer null --device cuda:0`, with AND without CUDA-graph capture)
- **cable_rigidCube** (`force_target=10`, bite 2.5 mm): latches ~8.8 mm, static ~14 N, peak ~125 N
  during the aggressive sweep (≈ baseline 108 N, vs 600–720 N before the measured-latch fix), holds
  through lift+sweep. `check_physics` PASS.
- **soft_pickplace** (`force_target=8`, bite 2.5 mm): latches ~7–8 mm, gentle ~1.5 N hold (compliant
  block), releases at the release window, block placed. `check_physics` PASS.
- **cable_soft** (cable, bite 2.5 mm): peak ~101 N, holds through the sweep. PASS.
- **rigidCube_soft** (rigid cube, `grip_bite=0`): peak ~65 N (was 4.1 kN at 2.5 mm), carry+drop. PASS.
- **soft_compression** (plate handle, `grip_bite=0`): peak ~73 N (was 2.7 kN), press+release. PASS.
- **pickplace_ycb_vbd** (two grasps): rubik's-**cube** (`grip_bite=0`) latches ~29.9 mm, ~15 N,
  placed; **banana** (`force_target=4`, `grip_bite=1.5 mm`) latches ~21.6 mm. The banana grasp pose
  was raised from `z = table − 1 cm` → `table + 1 cm` (in both `pickplace_ycb_vbd` and the rigid-only
  `pickplace_ycb_franka`): the sub-table pose had jammed the finger pads against the robot-side table
  collider so they could not close onto the banana. The cube close window was lengthened
  (`[2.6,3.0]`→`[2.6,3.8]`): a 0.4 s slam of the flat pads onto the flat box spiked the contact to
  ~4.7 kN; the slower close keeps the peak ~40 N. PASS.
- The latch position varies ~0.7 mm between capture/no-capture (warm-up/capture-start shifts the
  contact substep) — both are valid grasps; the persistent device `latch_state` + the measured
  `joint_q` read are capture-safe (capture bakes launches, not branch outcomes).

### Hard constraints kept (don't repeat earlier mistakes)
1. Split bridge stays for the VBD path; fingers stopped only by commanded joint position.
2. **Never feed the object reaction into the finger DOFs.** Only the NET reaction → arm/EE. A
   read-only force→*position* signal (the latch) is the only legal closed-loop hook.
3. Contact `kd` is a re-derived landmine (`proxy_kd≈1e2` absolute, pinned Newton `6dfe7303`); keep
   `alpha=0.95` default-hard contacts. Re-derive on any Newton bump.
4. Centralization mandatory: all grip knobs in `deformableManipulationTools/`; examples declare only
   scene + policy.

### Remaining / open
- **SUPERSEDED by the 2026-06-24 mode split (top of file):** the per-`GraspWindow` `force_target`/
  `grip_bite` numbers below were replaced by the `compressible` flag (incompressible → force_target 0 +
  `grasp_interference` bite; compressible → force_target 8 + 0 bite) with `CableConfig.grip_bite` as the
  lone per-object override. The geometry-dependence note still holds (a flat box at 1 mm reads ~150 N,
  a cable cage needs ~3 mm).
- The mesh objects (cube/banana on `pickplace_ycb_vbd`) have the ~0.9 s force RAMP; the measured-position
  latch at first contact (`force_target=0`) freezes the geometry at first solid contact, not the force,
  so it sidesteps the transient-vs-settled ramp.
