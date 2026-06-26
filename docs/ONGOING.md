# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just changed
and not yet settled, and any working hypotheses. Keep it lean — when something is proven and
durable, promote it to CLAUDE.md (or the relevant `docs/` file) and delete it here. Reset this file
at the start of each new big task.

## DONE (2026-06-26): fix panda scenic render of rigid-only demos (`KeyError: 'bowl'`)

`pickplace_ycb_franka` (the rigid-only MuJoCo demo) crashed at scenic render frame 0 with `KeyError: 'bowl'`
on the **panda** (`render_from_physics=True`). Root cause in `robolabViz/raycast.py`: on the rigid-only path
the scene's objects are MERGED into the robot MuJoCo model (`add_builder`), and `_robot_link_meshes_from_model`
iterated ALL visible mesh shapes, so it emitted the bowl/banana meshes as `robot_link` instances — whose FK
transform doesn't exist (`robot_link_tfs['bowl']`). Fix: pass `object_body_min` (= `object_body_start`) to
`_robot_link_meshes_from_model` and skip bodies `>= object_body_min` — the exact complement of the cutoff
`_build_object_instances` already uses (those bodies render as `object_body` instances, posed by body state).
The fr3 (`render_from_physics=False`, USD link meshes) never hit this; it surfaced once the panda became default.
Routing itself is correct & unchanged: `has_deformable = particle_count>0 or has_cable_joint` → rigid-only YCB
scene → `_build_rigid_only_mujoco` (single MuJoCo, no VBD/proxies).

## DONE (2026-06-26): ONE unified target-force admittance grip for EVERY object (rigid, cable, soft)

**Rule change:** a demo may now tune **one** physics knob — the **target grasp force** (`GraspWindow.force_target`).
Replaced the **close-only post-contact** servo (the `2026-06-25` "TRUE force-target grasp" entry below, now
SUPERSEDED) with a **bidirectional impedance/admittance** controller that may move the jaw open OR closed to
regulate a target force, stable against the spiky/decaying/load-dominated contact force. **Then unified the soft
and incompressible paths into ONE law** — the grasp no longer depends on object type; only the target differs.

### The control law (`grip._grip_force_stop_kernel` + `GripController`) — one law for all objects
Velocity-form admittance on the **closing-axis-projected** squeeze (`grip_squeeze_signal`, new). Approach at the
centralized `GRIP.grip_rate_max` until `engage_force`, then `w_dot = k_adm·(F_filt − target)` (`F_filt = EMA`),
deadbanded and **asymmetrically rate-limited**: close fast (`grip_rate_max`, chase target / restore a DECAYING
grip), open **very slowly** (`grip_rate_open` ≈ 0.2 mm/s). The asymmetry IS the stability mechanism ("grab firmly,
release reluctantly"): a lift/sweep LOAD or rigid SPIKE transiently dwarfs the target but the jaw barely opens
before it passes, so the grasp survives while genuine sustained over-force / decay still moves the jaw. The same
law holds a soft block at a gentle target (~8 N, no crush — a smooth force-width curve means no special freeze is
needed) and a rigid box at a firm target (~30 N). `max_overclose`, the close-only servo, the `compressible` flag,
and the soft-specific latch-then-freeze are all **removed**.

### Why the projection + asymmetry are both needed (probed empirically)
The lift load is tangential to the jaw axis, so projecting the per-pad reaction onto the closing axis rejects most
of it. But the cable is pathological: its **static** projected squeeze is **flat at ~3–6 N at EVERY width** (the
8 mm-radius rod rolls/deforms instead of building force), so there is NO force setpoint that fixes the cage width —
the grasp is geometric. So the target instead *drives* the asymmetric regulator to a cage tight enough to trap the
rod. `force_target = 30 N` → `w` settles ~8.5–9.3 mm (≈ the old 8.4 mm), floor ~30 N, dynamic peak ~150 N (old
~140 N). Lower → cage too loose, rod slips on the lift; much higher → crushes the rod to non-physical force.

### What it touched
- `params.py`: `GRIP` lost `max_overclose`, `grip_servo_rate`, `force_target_rigid`, `latch_debounce`,
  `latch_arm_margin`; gained `k_adm`, `force_filter_tau`, `grip_rate_max` (centralized squeeze speed),
  `grip_rate_open`, `engage_force`; `force_target` is now the single default (30 N). `GraspWindow` gained
  `force_target` (the one knob) and **lost `compressible`** (unified — no object-type mode).
- `grip.py`: `_reduce_grip_signal_kernel` also writes `grip_squeeze_signal` (closing-axis projected min);
  `TwoWayProxyCoupling.grip_squeeze_signal` + `grip_signal_values()` host getter; `_grip_force_stop_kernel`
  collapsed to ONE admittance law (5-float windows `[cs,ce,rs,re,ft]`, 3-float state `[w,f_filt,engaged]`) +
  the unchanged rigid-only smoothstep fallback; `GripController` simplified (`grip_widths()` debug getter).
- `framework.py`: passes `coupling.grip_squeeze_signal` to `GripController`.
- demos: `cable_rigidCube` `force_target=30`; `soft_pickplace` `force_target=8` (was `compressible=True`);
  stale comments updated across the VBD demos.

### Per-object target tuning (2026-06-26) + engage-scales-with-target
The soft block was over-compressed because the absolute `engage_force` (2 N) needed DEEP compression of a
compliant block before engaging. Fix: the engage threshold now SCALES with the target —
`clamp(engage_frac·target, engage_floor, engage_cap)` (0.15 / 0.3 N / 2 N, all centralized & identical across
demos). A low-target soft grip engages at a light touch (gentle, no crush); a high-target rigid/cable grip caps
at 2 N (its original firm seed → no cable regression). The deadband stays ABSOLUTE 2 N for every demo.
Final per-object targets (the only per-demo grasp knob): cable 30, rigid cube 30 (default), plate handle 50,
soft block 5, rubik's cube 30, banana 80. (An earlier RELATIVE deadband attempt regressed the cable demos and
was reverted — only the engage scaling was kept.)

### Verified (headless `--viewer null --device cuda:0 --test`): demos `check_physics` PASS
cable_rigidCube (holds through lift+sweep), soft_pickplace (gentle 8 N, no crush under the unified law),
rigidCube_soft, cable_soft, soft_compression, pickplace_ycb_vbd (two windows + releases), pickplace_ycb_franka
(rigid-only, `force_stop_enabled=0`). Ran WITH CUDA-graph capture (replay parity holds). cloth_franka is a
known-failing experimental probe (regressed by the new controller — deferred).
**Open / could improve:** the cable's ~150 N dynamic peak is inherent to its geometric (rolling line-contact)
grip, not a bug; tune `force_target` if a softer/firmer cage is wanted. The default `force_target_rigid` (30 N)
now also governs the other incompressible demos via the admittance — re-tune per-demo if any needs a different grip.

## DONE (2026-06-25): swappable robot via `settings.yaml` + Isaac Sim Panda (USD) as the default

Added a repo-root **`settings.yaml`** (loaded by `deformableManipulationTools/settings.py`) that
centrally selects the robot; **default is now `franka_panda_isaacsim`** (Isaac Sim Franka Panda, native
USD), with `fr3_franka_hand` (the old URDF robot) still selectable. Durable design promoted to CLAUDE.md
("Central config: settings.yaml"). The two robots are a verified **kinematic drop-in** for each other
(identical link7/finger/TCP world pose at the shared `home_q`), so all physics/policy/IK is shared;
only `RobotConfig.loader`/`usd_path` + the `*_link`/`*finger`/`hand` suffixes differ.

### What it touched
- `params.py`: `RobotConfig` gained `loader`/`usd_path`/`hand_link_suffix`; `ROBOTS` registry +
  `FRANKA = ROBOTS[SETTINGS.robot]`. `robot.py`: `build_franka_robot` branches `add_usd` vs `add_urdf`
  (shared gains/gravcomp/table after). Render: `robolabViz/robot_fk.py` (`RobotVisualFK` USD support),
  `robolabViz/scenic.py` (robot source from `FRANKA`; wrist-camera `parent_link = FRANKA.hand_link_suffix`).
  `examples/__init__.py` reads render-table/background/device defaults from `settings.yaml`. pyyaml pinned.
  Render-only gripper tint: `RobotConfig.viz_gripper_color` (machinery in scenic →
  `RaycastPreviewRenderer(gripper_color=...)`; physics untouched) colours the hand+finger links to tell
  robots apart. Currently toggled OFF (panda `viz_gripper_color=None`); set it to a tuple to re-enable.
  Render mesh source: `RobotConfig.render_from_physics` (panda = True) makes the scenic raycast draw the
  robot from the SIMULATED physics meshes (the per-link convex hulls `add_usd` loads — the same blocky
  gripper the `franka_vbd_proxies` proxy viz shows) instead of the USD's detailed visual meshes, so the
  render matches the simulated collision geometry. `_robot_link_meshes_from_model` in `robolabViz/raycast.py`;
  fr3 stays False (keeps its converted-USD visual). Render-only.
- **Examples were NOT touched** (they're fully config-driven via `FRANKA`/`home_q`/suffixes).

### Two panda-specific physics fixes (the panda's links are each one CONVEX_MESH; fr3's were box primitives)
1. **Box finger proxies** (`grip.build_gripper_proxies`): a CONVEX_MESH finger proxy is contacted late
   then explosively by VBD (rigidCube spiked to **1979 N** at a 2.3 mm latch). Auto-detect a non-box
   finger collider → substitute a per-finger **AABB box** → physical **60.7 N** at 26.8 mm (= fr3's
   64.9 N @ 26.8 mm). Per-finger AABB also fixes mirroring (panda fingers are both identity-posed; the
   right pad is a mirrored MESH, unlike fr3's 180°-flipped body). Palm/EE blocker proxy unchanged.
2. **Viz-first BVH ordering** (`framework._build_split_mujoco_vbd`): the panda's per-link mesh BVHs are
   shared into the viz model; finalizing viz LAST freed them → robot narrow-phase OOB (illegal memory
   access), only surfacing once the OBJECT side (cable capsules / ycb meshes) reused the freed pool
   (so cable/ycb crashed, box/soft passed). Fix = finalize the viz model FIRST (SOLVERS §4); this
   replaces the old object-only `mesh_first` branch (viz-first protects robot + object meshes both).
   Also fixed the proxy viz (`visualizations/franka_vbd_proxies.py`) to render CONVEX_MESH links.

### Verified — all 6 VBD demos PASS headless with the panda (`check_physics`):
| demo | mode | latched | peak grip (min-pads) | fps |
|---|---|---|---|---|
| cable_rigidCube | incompr | 8.7 mm | ~196 N (fr3 177) | 54 |
| cable_soft | incompr | 8.7 mm | ~153 N | 11 |
| rigidCube_soft | incompr | 26.8 mm | 60.7 N (fr3 64.9) | 12 |
| soft_compression | incompr | 14.0 mm | ~59 N | 12 |
| soft_pickplace | **compr** | 17–18 mm | 10–61 N (soft, run-varies) | 13 |
| pickplace_ycb_vbd | incompr ×2 | 29.8/18.9 mm | ~53 N | 6 |

Scenic re-render of all 6 with the panda: DONE — all 6 `simulation.mp4` written, all `test_final` (FK
parity + table-footprint + wrist-coverage) PASS; the rendered gripper meshes load from `franka.usd`
(`panda_leftfinger`/`panda_rightfinger`/`panda_hand`), confirmed. Also re-verified the rigid-only
`pickplace_ycb_franka` (7th demo, single-MuJoCo path) PASS with the panda, and an fr3 regression of the
mesh (ycb) + capsule (cable) paths PASS under the new viz-first ordering. fps include instrumentation syncs.

## DONE (2026-06-25): TRUE force-target grasp — fixes cable-slides-up + plate over-grip/"float"

Replaced the **first-contact + fixed-bite** latch (the `2026-06-24` entry below, now SUPERSEDED) with a
real force-target controller. Root cause it fixes: the first-contact latch never controlled FORCE — it
froze a position the instant the pads touched, so grip strength was an accident of contact stiffness.
Confirmed by before/after (`9a6a5ec` preset vs the broken HEAD) instrumented runs:
- **cable** sweep sustained-grip FLOOR collapsed `20.8 N → 1.9 N`; it slid up `+14.5 mm → +33 mm`
  (slides *through* the pads). The lone per-object `CABLE.grip_bite=3 mm` band-aid was insufficient.
- **plate** held in BOTH versions but at an absurd, NON-physical ~180–440 N squeeze (pads barely
  touching, held by the invisible 8 mm-gap proxy) → reads as "floats". Not a grip failure — the
  OVER-grip face of the same uncontrolled-force bug.

### The control law (`grip._grip_force_stop_kernel` + `GripController`, all centralized in `GRIP`)
Close (smoothstep) → **latch to halt the close** → **close-only hold servo to a force target** → freeze
under load. Per-window mode from `GraspWindow.compressible` (the ONLY knob; no per-object numbers):
- **Incompressible** (rigid + cable): latch at FIRST contact (just stops the racing smoothstep — the
  rigid target is only a brief close-transient, so latching ON it is unreliable), then a **close-only**
  servo tightens to `GRIP.force_target_rigid` (30 N). It NEVER opens, so the lift/sweep LOAD (sig ≫ ft)
  can't loosen the grip — the squeeze rises under load. A stiff object reaches 30 N in ~0.5 mm; a
  SETTLING cable (steady force decays at any width, so the servo would run to the crush floor) is
  stopped by the centralized `GRIP.max_overclose` (3 mm) cap — the force-informed replacement for the
  old per-object bite.
- **Compressible** (soft FEM): latch AT `GRIP.force_target` (8 N, gentle) and FREEZE — no servo (extra
  tightening crushes/ejects the block). The soft block builds force smoothly, so the threshold is
  reliable here.

Also: **`GRIP.proxy_gap` 8 mm → 2 mm** — force builds over `gap` (`f≈ke·(gap−sep)`), so the old 8 mm
held a stiff object firmly only at a non-physical deep squeeze (pad floats far off at any physical
force). 2 mm lets the 30 N target engage the pad near the true surface (no float), still ≫ the swept
cable's ~0.9 mm/substep (no tunneling). And the **palm/EE blocker proxy is now harvested two-way**
(`_harvest_proxy_wrench_kernel` loops ALL proxies; the EE sum + sync already generic) — its reaction
flows to the EE / is undone in sync instead of being a free one-way shove; still OUT of the grip signal.

Removed: `GRIP.grasp_interference`, `GraspWindow.grip_bite`, `CableConfig.grip_bite` (no per-demo knobs).

### Verified (headless `--viewer null --device cuda:0`): all 7 demos `check_physics` PASS
| demo | object | mode | latched | grip force (physical now) |
|---|---|---|---|---|
| cable_soft / cable_rigidCube | cable | incompr | ~8.4 mm (cap-bound) | floor ~10 N, peak ~140 N, slip +16 mm (was +33) |
| soft_compression | plate handle | incompr | ~14 mm | **~28–57 N** (was ~180–440 N), pads engaged, places |
| rigidCube_soft | 1 kg cube | incompr | — | holds + carries (was already fine) |
| soft_pickplace | soft block | **compr** | — | gentle, lifted + placed (mode split fixed an over-compress) |
| pickplace_ycb_vbd | rubik's + banana | incompr | — | both grasps PASS |
| pickplace_ycb_franka | YCB meshes | rigid-only MuJoCo | — | unaffected (force_stop_enabled=0), PASS |

Open: rubik's-cube release-stick (asymmetric stick to the right pad on the VBD-path release; ground
truth = the clean rigid-only `pickplace_ycb_franka`) — user DEFERRED; investigate later (likely a
penalty-contact tangential-stiction release artifact, separate from the grip-strength fix).

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
