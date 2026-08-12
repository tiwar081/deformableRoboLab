# Solver Architecture

## Solver framework selection (the rule — CENTRALIZED & automatic)

The framework chooses the object solver **centrally** in `GraspExample.__init__`: it builds the scene
into a neutral `ModelBuilder`, then routes — a deformable is present iff `object_builder.particle_count
> 0` **OR** a `CABLE` joint exists. FEM/cloth create particles; a rod/cable (`add_rod`) is capsule
bodies coupled by `CABLE` joints and carries **NO** particles, so it must be detected by joint type —
otherwise a cable-only scene (e.g. `cable_rigidCube`) misroutes to MuJoCo, whose forward kinematics
skips `CABLE` joints and fails to build the articulation. Rigid boxes/meshes have neither. An example
only declares its scene + policy; it never selects the solver.

- **Any deformable present** (cable/rod, cloth, FEM block) → split **`SolverMuJoCo` (robot) +
  `SolverVBD` (all objects)** with the **dynamic gripper-proxy bridge** (`grip.py`, wired by
  `framework._build_split_mujoco_vbd`). Full two-way contact only happens inside VBD; the MuJoCo↔VBD
  bridge crosses solvers via dynamic finite-mass finger proxies that mirror the fingers in the VBD
  model and feed the net object reaction back to the arm/EE (one-step lag) — the arm feels the object
  through the proxy bridge, not a shared contact. Every object that must touch a deformable lives in
  the VBD model. **The grip is FORCE-CONTROLLED** here by the centralized `GripController` with ONE unified
  law for every object (rigid, cable, AND soft): a bidirectional asymmetric admittance regulator. A demo
  declares only a `GraspWindow` + its one knob, `force_target` — see [gripper.md](gripper.md).
- **Rigid-only** → robot AND objects in **one `SolverMuJoCo`** (`framework._build_rigid_only_mujoco`
  merges the objects into the robot `ModelBuilder` via `add_builder`; Newton `brick_stacking`/`panda`
  pattern): true two-way frictional grasp, MuJoCo's mature convex/mesh contact, **CCD on**
  (`make_robot_solver` passes `ccd_iterations`/`ccd_tolerance`/`enable_multiccd`), none of the
  VBD-rigid-mesh fragility. The gripper closes to a FIXED target (`MUJOCO_GRIP.close_target`; **no
  object-size preset width** — contact + the finger actuator effort hold the object, cf.
  `_external/RoboLab`); `set_mujoco_grip_controller` stiffens the fingers for the real grasp. No VBD
  model/proxies/coupling; demos read object poses via `self.object_body_q()` (routing-agnostic; objects
  sit after `object_body_start` in the robot model). ≈ **2.2× faster** than the VBD+proxy path (ycb).
  - **There is no `object_model` on this path**, so any tool that inspects objects generically must be
    routing-agnostic: `object_model.body_label` is `None` here and labels must be read from
    `robot_model` offset by `object_body_start` (this silently broke the pipeline's settle harness and
    its settled-pose write-back for rigid-only scenes). Same class of trap as `object_body_q()`.
- `pickplace_ycb_franka` (rigid coacd-mesh bowl/banana + a box cube) auto-routes to MuJoCo;
  `pickplace_ycb_vbd_franka` is the SAME scene **plus a token soft cube** in a table corner, whose
  particles auto-route the whole workspace to VBD — the A/B twin demonstrating the centralized
  decision. Concave meshes (bowl/banana) collide as **coacd convex-hull pieces**
  (`assets.add_ycb_mesh`) in either solver — a raw concave mesh gives contradictory normals (SOLVERS §4).

## Robot side

- Franka from `franka_emika_panda/urdf/fr3_franka_hand.urdf`, `collapse_fixed_joints=True`,
  `force_show_colliders=False`, `enable_self_collisions=False` (Newton `cloth_franka` convention).
- EE control point: `fr3_link7` + local offset `(0,0,0.22)` → fingertip-pad bottoms.
- `SolverMuJoCo(solver="newton", integrator="implicitfast", cone="elliptic",
  use_mujoco_contacts=False)` + Newton `CollisionPipeline`.
- A hidden robot-side table collider (`robot_contact_table`) stops the gripper at the table
  surface (verified by driving the EE 8 cm below the top — halts exactly). Add any fixed
  obstacle the gripper must not cross as a static collider in the robot model the same way.
- The base mounts `TableConfig.base_drop` (3 cm) BELOW the worktop (framework builds it at
  `top_z − base_drop`): keeps objects/waypoints world-fixed while the visible work-table slab
  stands proud of the franka_stand (the viz solves the slab between base plane and top).
- IK (`newton.ik.IKSolver`: position + rotation + joint-limit objectives) solves keyframe
  poses once at startup.

- `SolverVBD`, 12 iterations, `rigid_contact_stick_motion_eps=0.0`, **NVIDIA default-hard
  contacts** (`alpha=0.95`, no cross-step history). Central config; a demo must not override it.
  Known, chosen limitation: NO static friction (sub-cone creep — a heavy grasped object slowly
  pivots about the jaw axis). The measured alternative, soft contacts
  (`rigid_contact_hard=False, friction_epsilon=0.05`), restores stiction and fixes the pivot but
  makes the initial grasp JOLT the robot through the one-substep-lagged EE feedback — it is
  RESERVED for an explicit future re-opening of the object-slippage problem (not routine tuning);
  the full trade study lives in SOLVERS.md §6. Both contact-mode knobs are CENTRAL (one
  `framework.py` solver_kwargs line shared by every demo — never per-demo `object_solver_kwargs`),
  and any switch requires re-running the full demo matrix.
  - *History (resolved):* the cable path once used `rigid_avbd_contact_alpha=0.0` +
    `rigid_contact_history=True` to hold the lifted cable with a *kinematic* proxy. That
    accumulates the ALM multiplier `λ` (`f_n = ke·pen + λ`, unbounded → 1e4–1e6 N grip): a
    kinematic proxy early-outs so the runaway is computed-but-not-applied (stable, uncontrolled),
    but a **dynamic** proxy applies it → divergence. The fix was to drop alpha=0+history AND
    re-derive the overdamped contact `kd` (a second, independent cause). NOTE (measured 2026-07-09):
    re-enabling `rigid_contact_history=True` even at `alpha=0.95` is still wrong — the decay-capped
    λ integrates ~10× on the kinematically-imposed pinch (460–670 N at a 30 N target → ejection);
    SOLVERS.md §6.
- Object contacts: `CollisionPipeline(contact_matching="latest", soft_contact_margin=0.01)`.
- The object model contains the visible table, manipulated objects, the soft block (where
  present), and the **dynamic** gripper proxies.
- Bridge is **TWO-WAY** via the proxy: robot motion → proxy → object squeeze; object reaction →
  harvested → net load fed to the arm/EE. See [docs/physicsEngine/gripper.md](gripper.md).

## Contact materials — per-object authorship + ONE central coupling law

Two standing constraints (they govern every new object added to the sim):

1. **An object authors ONLY its own contact properties** (its `params.py`/asset config). It never
   defines another object's property *for* it. (The old `ClothConfig.shape_contact_ke/kd` — cloth
   fields that set the PADS' and TABLE's material — are gone.)
2. **The coupling of two objects' materials into one contact is CENTRAL and pairing-blind.** The
   framework and demos never branch on soft×soft / soft×rigid / rigid×rigid; one law covers all.

Mechanics: Newton itself mixes every contact pair — rigid↔rigid from the two shapes' stored
materials, body↔particle from the scalar particle side (`model.soft_contact_*`, set from the
scene's particle-deformable config) and the shape's stored material — as **arithmetic-mean ke/kd,
geometric-mean mu** (`_average_contact_material` in the VBD kernels). Geometric mu is scale-free,
so authored mu values enter every contact as-is. Arithmetic ke/kd is a *parallel-spring* law (the
stiff side dominates) — physically wrong across scales (touching surfaces are springs/dashpots in
SERIES: the soft side dominates → harmonic mean) and numerically fatal (a pad ke=5e4 into a cloth
ke=15 contact → expulsion). So `framework._build_split_mujoco_vbd` stores, for every shape
**decisively stiffer than the particle side (`k > 2.5·s` — the band where the arithmetic mean is
dominated by the wrong side)**, the **derived** value `m = 2·harm(s,k) − s` (`k` = the shape's own
authored value, `s` = the particle side): Newton's arithmetic mean then lands exactly on
`harm(s,k)`, the harmonic mean of the two objects' own authored materials. Shapes at comparable or
softer scales pass through authored (harm ≈ arith there, ≤25% apart; a softer-than-particle shape
never dominates). The harvest ke (`coupling_soft_ke`) reads the pads' final stored value, so it
tracks automatically. Consequences to know:

- **A builder MUST register its authored material** (`assets._register_material`): the framework
  blanket-fills all shape materials post-finalize (uniform proxy placeholder) and re-applies only
  REGISTERED ones. `add_table`/`add_ycb_mesh` didn't register until 2026-07-11 — the table silently
  ran at the fill's mu (invisible while the fill happened to equal the authored value), and the ycb
  twins ran DIFFERENT object friction per solver path. If you add a builder, register.
- **The band is measured, not aesthetic**: compensating comparable-scale shapes wrecks the
  scene's rigid↔rigid pairs through the shared stored arrays — pads 5e4 → 3.3e4 softened the
  cable↔pad cage eff 3.5e4 → 2.67e4 and the swept cable slipped out of the hold
  (`cable_soft_franka`, 2026-07-10); band-limited, the cage is bit-preserved and the hold returns.
- **Deformable-type asymmetry falls out naturally**: a soft FEM body authors its contact skin at
  rigid-comparable scale — its compliance lives in the TISSUE (k_mu/k_lambda), the contact barely
  penetrates, and nothing enters the band in FEM scenes (VBD is implicit; the old proven block ran
  contact η≈540). A thin SHELL has no volume compliance, so the cloth genuinely needs the
  band-compensated soft effective contact — and cloth scenes have no live rigid↔rigid pairs
  (proxies are collision-filtered against the table).
- `model.soft_contact_*` is **ONE scalar per scene** (Newton limitation): two particle deformables
  in one scene cannot own distinct contact materials; true per-pair materials need upstream
  Newton support.
- **Friction has NO exception** (since 2026-07-11): every shape's authored mu enters every
  contact geometrically, cloth included. The old cloth-scene mu=1.5 stamp — needed while the
  recipe leaned on Newton's cheat friction — is retired: at the realistic material set the fold
  recipe compensates the physical cloth↔table friction with pressing normal force (fingertips
  commanded below the table top; anchoring is mu·N). Measured: without the press the pinch sheds
  at the drag onset; with it the fold matches the pre-retirement band (see [cloths.md](cloths.md)).

## Runtime (CUDA-graph capture)

- The substep loop is device-resident and captured into a CUDA graph (one
  `wp.capture_launch` per frame); keyframe trajectory + proxy sync are Warp kernels reading
  frame time from a device buffer.
- Capture happens **after one uncaptured warm-up frame** (lazy allocations raise inside
  capture) and requires an **even substep count** (state swap must return to its starting
  binding per captured frame). Falls back to the uncaptured loop on CPU / capture failure.
- Measured H200 (null viewer): `cable_rigidCube` 11.6, `cable_soft` 66.8, `rigidCube_soft`
  57.0 ms/frame.

## Visualization (critical recurring bug)

The viewer renders a separate combined viz model (robot builder + object builder).
`_sync_viz_state` **must copy `particle_q`/`particle_qd` from the object sim state** in
addition to body transforms. Copying only bodies renders every soft body frozen at its rest
shape while the sim deforms it underneath — the historical root cause of all "soft body
never deforms / objects penetrate it / contact before touching" reports.

## Verification standard

Verify every demo with instrumented **headless null-viewer `cuda:0`** runs reading sim state
per frame (grasp tracking, finger joints, object heights, soft-particle displacement,
NaN/ejection/pass-through, viz/sim particle parity after `render()`) — not just visually.
`test_final` in each example asserts the outcome, time-gated so 1-frame smoke runs pass.
Measured run-to-run noise (GPU atomics; identical code): grip metrics ±3%, cloth fold placement
±4 mm — and a RELEASED object's rest position scatters up to ~27 cm between identical runs, so
never use it as a pass/fail signal; use grasp retention/engagement and grasped-object tracking.

## Known limitations (standing — root-caused, deliberately open)

These are decided trade-offs, not undiscovered bugs. Each has a measured trade study; **do not
re-walk them opportunistically** while working on something else.

- **No static friction in the rigid grasp** → [SOLVERS.md](SOLVERS.md) §6. Symptoms: the ~2 kg plate
  slowly pivots about the jaw axis while carried (`soft_compression_franka`, ~18°); the banana's edge
  grasp is intermittent at ANY force target (its wedge converts squeeze into self-ejection in the
  same proportion — measured at 10/30/45/80 N; raising mu does not help, the cone limit isn't what
  binds). The measured fix (`rigid_contact_hard=False, friction_epsilon=0.05`) works but jolts the
  initial grasp through the one-substep-lagged EE feedback; it is RESERVED for an explicit
  re-opening of the slippage problem. Real fix for the banana: compliant fingertips or a flatter
  grasp point. Both knobs stay central in `framework.py`, never per-demo, and a switch requires the
  full demo matrix.
- **Squeeze-signal under-reads on tilted contact normals** → [SOLVERS.md](SOLVERS.md) §6 (last
  paragraph). The regulated signal projects each pad's force onto the jaw axis, so on wedge faces it
  under-reads ~3× and the regulator over-squeezes. The physical fix is per-pad contact-NORMAL force
  as the signal — but the cable cage currently depends on the projected signal's behaviour, so the
  change needs a full-demo revalidation.
- **Residual penalty-contact ring at impact/grasp moments** (pre-existing) →
  [SOLVERS.md](SOLVERS.md) §5.
- **Validation coverage is uneven**: `cloth_franka`, `cable_soft`, `soft_pickplace` are
  metric-verified at the final materials per the standard above; the other five demos are
  render-verified only.
- **Two demos' narratives changed with the raspberry-like block** ([examples.md](examples.md)):
  `soft_compression` (the 2 kg plate now flattens the fruit) and `rigidCube_soft` (the steel cube
  squashes it and rolls off). Both are honest physics — if those *stories* should be preserved, add a
  second, firmer canonical soft object (`params.py` allows distinct named instances) rather than
  de-realizing the berry.
