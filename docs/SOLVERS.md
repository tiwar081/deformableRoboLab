## Troubles using SolverVBD for *rigid* objects

`SolverVBD` is built for deformables (cloth, FEM, rods, particles). In this repo we
also run the *object side* of every demo through it — rigid cube/sheet/cable/bowl,
plus the **dynamic finite-mass** gripper "proxy" bodies — so the robot (MuJoCo) and the objects
share one contact world. That works, but rigid bodies in VBD have a row of sharp
edges. Recording them here so the next person doesn't relearn them.

> **Read this first.** Troubles #1–3 below describe the **superseded kinematic-proxy +
> force-latch era**, kept for the lessons but NOT what the code does today. Both the *kinematic*
> proxy (#1–2) and the *force-triggered position latch* (#2) are gone: the grip is now **dynamic
> finite-mass proxies** closed by a **force-controlled admittance regulator** (no latch), and the
> VBD object side runs **NVIDIA default-hard contacts** (`alpha=0.95`), not the `alpha=0` those
> troubles call for. For what the code does NOW see [gripper.md](gripper.md),
> [solver-architecture.md](solver-architecture.md), and the "current dynamic proxy" section below.
> The traps in #4–5 (mesh fragility, light-body ejection) are still live.

### 1. VBD won't hand you forces on a kinematic body

The split-solver "bridge" is the kinematic gripper pads (proxies) that mirror the
real fingers into the object model. To make the gripper *feel* the object you need
the reaction on those proxies — but VBD's per-body contact accumulation early-exits
for `body_inv_mass <= 0` (kinematic), and it never writes `State.body_f` for *any*
rigid body. The only public readback is `SolverVBD.collect_rigid_contact_forces`
(per-contact force on body1, with the `body_q_prev` snapshot caveat). For **soft**
contacts there is *no* force readback at all — we recompute `ke·penetration` from
the public `soft_contact_*` geometry (the same penalty law VBD uses internally).

### 2. Stiff penalty contacts + lagged cross-solver feedback ⇒ chatter / ejection

The robot is MuJoCo, the objects are VBD — so any "objects push back on the robot"
coupling is *explicit* and *one substep late*. Feeding a VBD contact reaction back
into the robot across that lag is unstable against the stiff penalty contact: the
classic `k·dt²/m > 1` condition. With a light gripper finger it chattered to
hundreds of N and ejected the grasped object; stiffening the finger instead drove
the penalty force into a multi-m/s pinch-ejection. Two things died here:

- **Whole-arm two-way coupling** (arm feels payload) — built, then abandoned. A
  real arm has the payload capacity anyway; only the *gripper DOF* needs to respond.
- **Continuous gripper force feedback** — replaced by a **force-triggered position
  latch**: position-control the gripper, creep it closed, and the moment the
  (read-only) contact reaction crosses a threshold, latch the width and hold. No
  in-loop force feedback ⇒ it cannot eject. Rigid → stops at the surface instantly;
  soft → compresses until the threshold.

### 3. Pinch ejection from the moving kinematic pads

Even without feedback, penetration accumulating against the *moving* proxy pads
spikes the contact force and kicks a pinched object out at multi-m/s. The
kinematic-proxy era fixed this with the non-default VBD setting
`rigid_avbd_contact_alpha=0.0` (full per-step penetration correction).

**This is now reversed — `alpha=0` is the wrong choice for the *dynamic* proxy.** It
accumulates the ALM multiplier into a 1e4–1e6 N phantom grip (a kinematic proxy early-outs
so the runaway is computed-but-not-applied — stable, uncontrolled — but a dynamic proxy
*applies* it and diverges; see the "current dynamic proxy" section below). The object side
runs **NVIDIA default-hard contacts** with the ALM state zeroed every substep (`alpha=0.95`,
no `rigid_contact_history`, `rigid_contact_stick_motion_eps=0.0` — see `framework.py`
`_build_split_mujoco_vbd`). This kills the runaway at a known, measured cost: **no static
friction** (grasped objects can creep below the Coulomb cone — §6 documents the full trade
study, including the soft-contact alternative that restores stick). Pinch stability comes from
physical contact damping (the re-derived `kd`), not `alpha=0`.

### 4. Mesh rigid bodies are fragile in the narrow phase

Putting real mesh shapes (banana, bowl) on rigid bodies surfaced several crashes,
each a hard segfault or an eject-to-infinity, not a clean error:

- **Shared `Mesh` BVH across two finalized models.** Re-using one `Mesh` object in
  both the object model and the combined viz model crashes: finalizing the viz
  model rebuilds the shared mesh's BVH and frees the GPU memory the object
  collision pipeline still points at → segfault in `narrow_phase.launch_custom_write`.
  Fix: finalize the *viz* model first (it never collides, so its stale BVH is
  harmless); the object model, finalized last, owns the live BVH.
- **Full-res mesh BVH crashes the midphase** in a multi-shape scene. Decimate to
  ~1k–1.2k tris. And note: once *any* mesh collides, the midphase processes *every*
  mesh shape in the model — including visual-only ones — so keep them all low-poly.
- **A *dynamic concave* mesh ejects the whole solve.** A box wedging into a raw
  concave/non-watertight mesh (the bowl) gets contradictory contact normals → a
  single-substep penalty spike that throws *every* body to infinity (it looked like
  the banana "straightened and flew", but it was just collateral to the global
  blow-up). More substeps made it *worse* (it's geometric, not a timestep issue).
  Fix: **convex decomposition** (coacd) into convex-hull pieces — consistent
  normals, stable, and the cavity is preserved. Do **not** quadric-decimate the
  hull pieces: decimation breaks convexity and the spike returns.
- **Concave mesh needs a big contact budget.** A box settling into the bowl
  generates thousands of triangle contacts; the default
  `rigid_body_contact_buffer_size=2048` overflows → spike → ejection. Raised to 16384.
  (`max_triangle_pairs` is separately capped at 1,048,576 by a 20-bit contact id.)

### 5. Penalty contacts punish bad masses and ring

- **Light bodies get flung.** A naive `density` gave a ~23 g bowl / ~46 g banana,
  so any contact threw them. Use realistic (YCB) masses — both accurate and far
  more stable. (We scale the bowl body to an exact target mass after finalize so a
  `--bowl-mass` knob is independent of the convex-piece volume.) Root cause: contact
  stability is the dimensionless `η = ke·dt²/m_reduced` of the pair, and the blanket
  `ke=5e4` puts a sub-~0.1 kg body past `η=1`; the over-correction then converts
  penetration into a `∝1/m` velocity and ejects the lighter-coupled member (often the
  *other* body — e.g. at `--bowl-mass 0.05` the 0.2 kg cube flies, not the bowl).
  Stiffness retuning can't fully fix it because `ke` is averaged across the pair
  (`avg_ke ≥ ke_heavy/2`). **The fix here is realistic masses + physical contact
  damping — NOT a velocity clamp.** Clamping object velocity is forbidden by the
  CLAUDE.md physics rule "No velocity clamps on objects" (robot/table excepted): a clamp
  hides the instability instead of curing it. (`particle_max_velocity` is moot anyway —
  inert under VBD, see `assets.py`.)
- **Residual ring.** Even stable, the penalty contacts leave brief high-velocity
  transients at impact/grasp moments (they don't displace anything, but they're not
  pretty). They're chaotic/non-deterministic and *increase* with more substeps —
  i.e. not a dt instability. Honest open item; likely a contact-damping question.
  The absolute-damping `kd` semantics must be re-derived on every Newton bump — see
  CLAUDE.md **Newton version** (current pin: Newton `v0.2.3-665`).

### 6. Static friction trade study (2026-07-09) — the current config has none, by choice

The grasp-slip investigation (instrumented headless A/B, `deformableManipulationTools` + a
scratch metrics harness). **Outcome: the codebase KEEPS the default-hard + per-substep-zeroed
config** (calm arm, no grasp jolt) and accepts the slip symptoms below as known limitations.
The soft-contact alternative that fixes them is documented at the end of this section, but it is
**reserved for whenever the object-slippage problem is explicitly taken on again** — slipping may
well persist in other demos in the meantime; do NOT reach for the contact mode as routine
per-symptom tuning. Everything here is measured — don't re-walk the dead ends.

**Symptoms.** The ~2 kg PLATE held by its handle pivoted about the jaw axis like a frictionless
pin joint (24.5° during the carry, squeeze decaying −19 N/s); the ycb banana squirted out of the
closing jaws (no grip even at the 80 N band-aid target). Ruled out empirically: friction mu (pad
1.5 × banana 2.0, geometric mean ≈ 1.73 — the Coulomb cone was never the binding constraint) and
contact-manifold degeneracy (measured ≈ 8.6 pad↔handle contacts with ≈ 2–5 cm lever arms — plenty
of torque capacity IF friction sticks).

**Root cause.** In hard/ALM mode with the ALM state cold-started every substep (the CURRENT
config: `stick_motion_eps=0`, no history, and the framework collides every substep so the
tangential multiplier `λ_t` is `zero_()`'d 960×/s), friction has no memory: any tangential load
below the cone still creeps every substep. Rotational creep about the grasp line is exactly the
plate pivot and the banana roll-out.

**The dead end — do NOT "fix" this by enabling persistence.** Measured: Newton-default stick eps
alone (config B) leaves the creep (the deadzone freeze needs the body to be near-stationary in
WORLD space, so it is inert during any carry). `rigid_contact_history=True` (configs C/D, with and
without anchor replay) arrests the creep but turns the ALM multiplier into a **force integrator on
the kinematically-imposed pinch**: the teleported pads cannot yield, the violation never resolves,
so `λ` grows to its decay balance (~10× the penalty force) — measured 460–670 N at a 30 N target;
the width regulator is the only relief valve, it overshoots, and the object is ejected. This is
the same mechanism as the old `alpha=0` phantom grip, merely capped by the `α·γ` decay. It is
structural: ALM's job is to grow `λ` until the violation resolves, and an imposed pinch never
resolves. (No upstream Newton example pinches an object between externally-driven bodies, so
upstream never hits it.)

**The OPTION — reserved for an explicit future attack on the object-slippage problem — soft
rigid contacts (`rigid_contact_hard=False, friction_epsilon=0.05`), Newton's other rigid path**
(their rj45 insertion example runs it): penalty normal force — the honest `ke·pen` pinch the
framework already assumes — plus **IPC-regularized Coulomb friction** whose (regularized) static
branch is evaluated fresh each substep. No cross-substep state, no integrator; residual creep is
bounded ∝ `friction_epsilon`. Verified end-to-end (one line in `framework.py`
`_build_split_mujoco_vbd` replacing the `rigid_contact_stick_motion_eps=0.0` kwarg): plate pivot
24.5° → 3.9° with the regulator holding target and a calm hold; every VBD demo passed `--test`
at unchanged runtimes. **WARNING — the reason it is NOT enabled: the initial grasp is measurably
jerkier.** The restored stiction is near-rigid tangentially, and through the bridge's
one-substep-lagged EE feedback the contact onset JOLTS the robot (and transients rattle it —
worse at higher grip force); it was judged a worse artifact than the slow slip it cures. Two
rules if that future effort happens: (1) `rigid_contact_hard` and `friction_epsilon` are
**central physics config — set them ONCE in `framework.py`'s `solver_kwargs`, identically for
every demo** (never per-demo via `object_solver_kwargs`; cross-demo consistency is the point of
the centralized solver build), and (2) re-run the full demo matrix watching hand-speed swing,
not just test_final.

**`friction_epsilon=0.05` is a measured two-sided constraint inside that option.** The stiction's
near-zero-slip tangential response has effective damping `≈ 2μ·f_n/ε` — at Newton's default
`ε=1e-2` and a realistic 50–100 N pinch that is ~10⁴ N·s/m, and fed to the arm/EE through the
**one-substep-lagged** coupling it drives a ~20–60 Hz arm↔contact limit cycle: the robot visibly
shakes while holding ANY rigid object, scaling with grip force and payload (the object-side solve
is implicit and stays calm; only the arm-side injection is explicit+lagged). Measured: at
ε=0.01–0.02 the arm buzzes (hand-speed swing 0.37–0.46 m/s on a 30 N cube hold, worse at higher
force); at ε=0.05 it is calm during holds (0.126 cube, 0.013 plate) while the plate still holds
at 3.9° tilt (BETTER than ε=0.01's 7.5° — less chatter also means less micro-slip). Dead ends
measured so you don't re-walk them:
- **Low-passing the EE wrench fails both ways.** τ=10 ms removes the arm's contact "flinch" — its
  fast force relief (the width regulator is far too slow) — and the unrelieved pinch ran to
  1656 N and launched the object; τ=2 ms changes nothing because the limit cycle sits in the SAME
  ~20–60 Hz band as the flinch. There is no frequency separation; the feedback must stay raw.
- **Heavier proxies don't help** (`proxy_mass` 10→40 kg: cube-hold swing 0.46→0.37) — the loop is
  arm-side, not proxy-noise.

**The banana-class edge grasp is marginal in EVERY config.** A grasp on a steep curved wedge
converts squeeze into a self-ejection load in the SAME proportion at any force (load/cone ≈
`sinθ/μ`, grip-force-independent — measured slide-out at 10/30/45 N under soft ε=0.05, and the
same grasp is "intermittent at 80 N" under the current hard mode). Raising the force target does
NOT help — it only buys over-squeeze rattle (the projected squeeze signal under-reads the tilted
faces, so the true pinch runs ~3× the reading). The real fix is future work: compliant fingertips
(model the pad as a deformable) or a flatter grasp point (a policy choice).

**Honest open item from the same investigation:** the regulated squeeze signal is the min per-pad
force **projected onto the jaw closing axis**; on tilted contact normals (the banana wedge) it
under-reads (~30 N reading vs 90–140 N true pad force), so the regulator over-squeezes. The
physical fix direction is measuring the squeeze as the contact-NORMAL component per pad (what a
real gripper's force sensing reports), but the cable cage currently depends on the projected
signal's behaviour — needs a full-demo revalidation. Tracked in [ONGOING.md](ONGOING.md).

### Bottom line

VBD is fine as a shared rigid+deformable contact world *if* you treat its rigid
contacts as stiff penalty contacts: keep the cross-solver feedback **net-to-EE and
one-step-lagged** (never per-finger), run **NVIDIA default-hard contacts** with physical
contact damping (NOT `alpha=0` — that diverges a dynamic proxy; and know the measured
static-friction trade in §6 before touching the contact mode), convex-decompose any
dynamic concave mesh, keep mesh BVHs small and unshared, and use realistic masses. The
public API is enough (`collect_rigid_contact_forces` — it handles both contact modes —
`soft_contact_*`, `State.body_f` on the MuJoCo side) — we never modified or imported
`newton._src`.

## How object↔gripper two-way physics is implemented (current: dynamic proxy)

The robot (MuJoCo) and the objects (VBD) are *separate* solvers; the bridge is a pair of
**DYNAMIC finite-mass gripper-proxy bodies** in the object model that mirror the real Franka
fingers (`deformableManipulationTools.grip.TwoWayProxyCoupling`, wired in by `framework.py`). The
authoritative description is [gripper.md](gripper.md); the essentials:

**Forward — gripper → object (every substep).** Each proxy is re-pinned to its finger's
pose+velocity with the gravity + lagged-contact-wrench velocity deltas pre-subtracted
(momentum-consistent undo), so it stays slaved to the finger *yet participates as a finite-mass
contact body* in the VBD solve. The squeeze is a real VBD contact (penalty normal + Coulomb
friction) — the friction carries the object through the lift. The grip width is **force-controlled**
by the centralized `grip.GripController` (a bidirectional asymmetric admittance regulator; the one
per-demo knob is `GraspWindow.force_target`) — **not** a preset width and **not** a force-triggered
latch. See [gripper.md](gripper.md).

**Backward — object → arm/EE (every substep, net).** The object→proxy reaction is harvested (rigid
via `SolverVBD.collect_rigid_contact_forces`; soft via recomputed `ke·penetration` over the public
`soft_contact_*` geometry) and the **NET** of the two pads (internal squeeze cancels, external load
remains) is fed onto the **arm/EE** one substep later — so the arm DOES feel the payload. Per-pad
reaction is never fed to the gripper DOFs (that pushes the pads open and loses the grasp).

This **supersedes** the earlier *kinematic*-proxy + force-triggered-latch design (Troubles #1–3
above were its symptoms). A kinematic proxy can't return body forces and, with `alpha=0`+history,
accumulated the ALM multiplier into a 1e4–1e6 N phantom grip; the dynamic proxy with NVIDIA
default-hard contacts and re-derived physical `kd` is stable and gives a bounded ~10–90 N grip
(static-friction trade study: §6). Full analysis in [gripper.md](gripper.md) and
[solver-architecture.md](solver-architecture.md).

# Newton Example Reference

The solvers (from `newton/solvers.py`), so the per-example notes below can stay terse:

- `SolverMuJoCo` — rigid bodies + generalized-coordinate articulations; contacts from
  MuJoCo or from a Newton `CollisionPipeline` (`use_mujoco_contacts=False`).
- `SolverFeatherstone` — rigid/articulations; often a kinematic integrator ahead of a
  deformable solve.
- `SolverVBD` — implicit; rigid bodies, particles, cloth, soft bodies, cable/rod
  joints; `integrate_with_external_rigid_solver=True` gives one-way rigid->deformable
  coupling.
- `SolverXPBD` — native position-based rigid/articulation/particle/cloth solver
  (common default).
- `SolverSemiImplicit` — native semi-implicit, fully differentiable; the diffsim
  backbone.
- `SolverKamino` — maximal-coordinate rigid/articulations (PADMM), good for closed
  loops.
- `SolverStyle3D` — anisotropic garment cloth only. `SolverImplicitMPM` — MPM
  materials only.
- `newton.ik.IKSolver` — joint-target generation only, not a physics integrator.

Every example follows the same three-bullet shape: objects involved; motion
(init -> motion -> final/significance); solver(s) and why.

## basic

`example_basic_shapes`:
- one free rigid body per collision primitive (sphere, ellipsoid, capsule, cylinder,
  box, bunny mesh; cone is visual-only).
- all spawned at z=2 above a ground plane and dropped; each settles to its primitive's
  rest height. Smoke test that every shape type collides and rests correctly.
- `SolverXPBD` default (or `SolverVBD` via `--solver`), 10 iters: pure rigid
  contact/friction settling, no articulation, so the cheap position-based solver fits;
  the VBD path just stiffens contact ke/kd.

`example_basic_joints`:
- three articulations, each a static anchor + a swinging cuboid child demonstrating
  REVOLUTE, PRISMATIC (limited), and BALL joints; a sphere anchors the ball joint.
- children start perturbed off their joint axes and swing/slide under gravity onto
  their constrained DOF, then rest. Shows the four basic joint types and their allowed
  motion.
- `SolverXPBD` default (or `SolverVBD`): constrained rigid articulation with contacts;
  `eval_fk` seeds the maximal-coordinate state from the edited `joint_q`.

`example_basic_urdf`:
- a quadruped URDF (floating base, self-collisions off) replicated across ~100 worlds.
- each starts at standing joint angles with matching PD targets, drops, and settles
  into a stand (no trained policy). Demonstrates URDF import + world replication + PD
  settling.
- `SolverXPBD` default (or `SolverVBD`): rigid articulation with joint PD targets and
  ground contact — no generalized-coordinate solver needed.

`example_basic_conveyor`:
- one kinematic annular belt (revolute), two static rails, a visual island, and 18
  free bags (boxes/capsules/spheres).
- the belt joint is kinematically prescribed to spin; bags spawn on it and are carried
  around by belt friction, kept on by the rails. Demonstrates friction-driven
  transport off a kinematically driven body.
- `SolverXPBD` default (or `SolverVBD`): rigid friction transport; collision filters
  exclude belt-rail/belt-ground so the belt only pushes bags.

`example_basic_heightfield`:
- a sinusoidal heightfield terrain + five dropped rigid spheres.
- spheres fall from z=1 and roll/settle into the terrain troughs without tunneling.
  Contrasts native vs MuJoCo heightfield collision.
- `SolverXPBD` default (native `CollisionPipeline`) or `SolverMuJoCo` via `--solver`
  (its own contacts): both rigid, the point is comparing the two contact backends.

`example_basic_pendulum`:
- a double pendulum (two box links, two revolute joints), first link anchored at z=5.
- released horizontal and swings freely — classic chaotic double-pendulum motion,
  asserted to stay in-plane. The simplest free articulated mechanism.
- `SolverXPBD`: tiny rigid articulation; `eval_fk` seeds body state from `joint_q`.

`example_basic_plotting`:
- a humanoid (MJCF) replicated across ~4 worlds.
- starts at z=1.5 and falls onto the ground while the example streams MuJoCo solver
  diagnostics (iteration count, energy, active constraints) to live plots / a PNG.
  Demonstrates reading solver internals.
- `SolverMuJoCo` is required because the plotted diagnostics (`solver_niter`, energy,
  `nefc`) only exist in the MuJoCo backend.

`example_recording`:
- a humanoid (MJCF) replicated across 100 worlds with randomized initial poses.
- each falls/flails under gravity (no control) while `ViewerFile` auto-records the
  model and every state to a `.bin`. Demonstrates simulation recording for later
  replay.
- `SolverMuJoCo` (newton solver, euler): a MuJoCo articulation integrator drives the
  fall; the solver choice is incidental — recording is the point.

`example_basic_viewer`:
- no physics model — instanced animated shapes (sphere, box, cone, cylinder, capsule,
  bunny) + debug axis lines.
- `step()` is empty; shapes are procedurally animated in `render()`. Demonstrates
  driving the Viewer directly without a Newton model.
- no physics solver — pure visualization.

`example_replay_viewer`:
- no physics model — a ReplayUI that loads a recorded `.bin` and reconstructs a
  Model + State.
- `step()` is a no-op; the user scrubs recorded frames on a timeline. The playback
  counterpart to `example_recording`.
- no physics solver — pure replay.

## sensors

`example_sensor_contact`:
- a hinged flap (revolute), two plates, and a dynamic cube + sphere riding the flap
  (USD scene), on a ground plane.
- the hinge is position-driven open, tipping the cube and sphere onto the plates;
  per-counterpart contact sensors light up the matching plate (cube/plate1,
  sphere/plate2). Demonstrates `SensorContact` total + per-counterpart forces.
- `SolverMuJoCo` (pyramidal cone): MuJoCo exposes the rich per-contact force data the
  sensor consumes, plus the driven hinge.

`example_sensor_imu`:
- three free "axis cube" bodies, each carrying an IMU site, each spawned rotated about
  a different axis.
- they fall and settle face-up, and `SensorIMU` reads the accelerometer (mapped to
  per-cube color); each measures gravity along its expected resting axis. Demonstrates
  IMU readout.
- `SolverMuJoCo`: supplies the body accelerations the IMU samples; rigid settling under
  gravity fits naturally.

`example_sensor_tiled_camera`:
- 24 worlds (6x4), each randomly populated with primitives (+ optional Gaussian splat)
  plus a Franka FR3 arm.
- no dynamics — the Franka joints are animated kinematically while `SensorTiledCamera`
  renders color/depth/normal/semantic/shape-index tiles per world. Demonstrates
  multi-world tiled camera sensing.
- no physics solver — kinematic posing (`eval_fk`) + sensor rendering only.

## cable

All cable examples use a single `SolverVBD` (5 iters), because VBD natively handles
rod/cable joints, bending stiffness, and capsule-capsule + ground contact in one
implicit solve. Differences are in topology and what is kinematically driven.

`example_cable_y_junction`:
- one Y-shaped deformable rod (three 20-segment capsule+cable-joint branches sharing a
  junction node), one tip pinned (zero mass).
- laid out at z~1.25 with one tip pinned; the free branches fall and settle while the
  junction holds the three branches connected. Demonstrates a branched cable topology.
- `SolverVBD`; the pin is a zero-mass body, not solver-driven.

`example_cable_twist`:
- three zigzag rods (64 segments) with increasing bend stiffness (1e2/1e3/1e4), first
  capsule of each made kinematic.
- start flat; a kernel continuously spins each cable's first capsule, propagating twist
  down the chain and across the 90deg bends. Demonstrates twist transport vs bend
  stiffness.
- `SolverVBD` with hard contact (`rigid_avbd_contact_alpha=0`); first capsules
  kinematically driven by writing `body_q` each substep (one-way). (This repo's object side
  once borrowed this `alpha=0` choice but has since reverted to NVIDIA default-hard contacts —
  `alpha=0` diverges the dynamic gripper proxy; see the dynamic-proxy notes above and §6.)

`example_cable_pile`:
- a dense pile of many wavy rods (40 segments) stacked in alternating-orientation
  layers, on an optionally sloped ground.
- start stacked above ground and fall into a settled pile. Stress-tests dense
  cable-to-cable contact, stacking stability, and friction.
- `SolverVBD` (`rigid_contact_history=True`, larger buffer, `contact_matching=latest`);
  nothing driven — pure settling.

`example_cable_bundle_hysteresis`:
- a 7-cable bundle (1 center + 6 ring) + 4 kinematic frictionless capsule obstacles.
- the obstacles cyclically load/hold/release (triangle-wave) the bundle, which springs
  back with plastic memory. Demonstrates Dahl-friction bending hysteresis.
- `SolverVBD` with optional per-joint Dahl plasticity; obstacles kinematically driven
  (one-way).

`example_cable_cross_slide_table`:
- a cable-driven XY cross-slide: fixed base, prismatic X/Y carriages, 7 pulleys (2
  kinematic inputs + 5 passive sheaves), and one closed-loop rod wrapped through them;
  zero gravity.
- only the two input pulleys are driven (rotated to trace a rectangle); the cable
  wrapping + passive pulleys + prismatic joints move the table marker along the
  commanded path. Demonstrates a cable-driven mechanism with groove contact.
- `SolverVBD` (`rigid_contact_hard=False`): couples articulated rigid stages, passive
  pulleys, and the deforming closed loop in one solve; inputs kinematically driven.

## cloth

`example_cloth_bending`:
- one pre-curved triangle-mesh cloth.
- dropped tilted onto the ground; bending stiffness keeps it from flattening so it
  retains the curve at rest. Demonstrates bending-stiffness behavior.
- `SolverVBD` (10 iters, self-contact on): handles membrane + bending + particle/ground
  contact.

`example_cloth_hanging`:
- one planar cloth grid pinned on its left edge.
- starts horizontal and sags from the pinned edge under gravity. A benchmark run across
  four cloth-capable backends.
- user-selectable `SolverVBD` (default) / `SolverStyle3D` / `SolverXPBD` /
  `SolverSemiImplicit` — the example exists to compare them on one scene.

`example_cloth_twist`:
- one square cloth (50x50); left/right edge columns are kinematic handles; zero gravity.
- the two edge columns are kinematically counter-rotated, twisting the cloth into a
  tightening spiral. Demonstrates intersection-free self-contact under extreme twist.
- `SolverVBD` (4 iters, self-contact on): chosen for robust self-contact; edges driven
  via `particle_q` (one-way).

`example_cloth_poker_cards`:
- 52 stiff cloth "cards", a static box platform, a kinematic sphere striker.
- cards drop and stack on the box; the sphere then sweeps in and knocks them off.
  Demonstrates stiff-cloth stacking + a kinematic disturbance.
- `SolverVBD` (10 iters, self-contact on); CUDA graph disabled because the sphere is
  animated via per-substep `body_q` edits (one-way).

`example_cloth_rollers`:
- one spiral-wound cloth + two cylinders (built as cloth meshes with deactivated verts
  = kinematic rollers); zero gravity, cm units.
- both cylinders and the attached cloth edge are kinematically rotated, unrolling the
  cloth between rollers. Demonstrates unrolling with heavy layer self-contact.
- `SolverVBD` (12 iters, shape contacts off): self-contact carries the interaction;
  rollers + edge driven via rotation kernels (one-way).

`example_cloth_franka`:
- a Franka Panda (URDF), a static table, a deformable T-shirt mesh.
- the arm follows a long scripted Jacobian-IK key-pose sequence to pinch, lift, drag,
  and fold the shirt at multiple corners. Demonstrates intersection-free robot-cloth
  folding.
- `SolverFeatherstone` (robot, particles/gravity off during its substep) +
  `SolverVBD` (cloth, `integrate_with_external_rigid_solver=True`): split solvers, one-
  way robot->cloth — the convention this repo's split design follows.

`example_cloth_h1`:
- a Unitree H1 humanoid (fixed base) wearing a Style3D jacket.
- IK drives the hands to raise/wave/drop; per-frame body transforms are interpolated
  across substeps and fed to the cloth so the jacket follows. Demonstrates interpolated
  robot kinematics driving cloth.
- `SolverStyle3D` (garment) + `newton.ik.IKSolver` (joint targets, not an integrator);
  robot bodies kinematically interpolated, one-way into the cloth.

`example_cloth_style3d`:
- a garment cloth draped over a static avatar mesh (alt path: a pinned high-res grid).
- no driving — the garment settles and drapes under gravity, colliding with the static
  avatar. Demonstrates anisotropic garment draping.
- `SolverStyle3D` (4 iters): purpose-built garment solver using anisotropic tri/edge
  stiffness + UV panels; avatar is collision-only.

`example_cloth_stiff_material_hanging` (vbd):
- one square cloth with very stiff membrane (tri_ke=1e8), one edge column pinned.
- hangs from the pinned edge; velocities stay bounded where a non-PSD StVK Hessian
  would blow up. A numerical-robustness test of the stable Neo-Hookean membrane.
- `SolverVBD` (10 iters, self-contact off): the stable membrane model is the thing
  under test.

`example_cloth_stiff_material_stretch` (vbd):
- five 20x20 sheets, each pinned on both +/-X edges with a different Poisson ratio;
  gravity off.
- each right edge is kinematically ramped to 2x stretch; the orthogonal contraction is
  checked against a closed-form area-ratio prediction. Validates the membrane model.
- `SolverVBD` (20 iters): the constitutive model under test; the right edge is a driven
  boundary condition.

## contacts

`example_nut_bolt_hydro`:
- rigid bolt + nut mesh pairs (with SDFs) replicated across ~20 worlds.
- the nut starts seated above the bolt, settles, and threads itself down by rotating
  while the bolt stays fixed. Demonstrates fine threaded contact via hydroelastic
  pressure fields.
- `SolverMuJoCo` (`use_mujoco_contacts=False`) + a Newton `CollisionPipeline` with a
  `HydroelasticSDF` callback: MuJoCo's stiff implicit solver suits high-stiffness fine
  threads; re-collides every 2 substeps to track them.

`example_nut_bolt_sdf`:
- same rigid bolt + nut pairs (higher-res SDF, no hydroelastic), ~100 worlds.
- identical threading scenario. Demonstrates the plain SDF point-contact path as a
  contrast to the hydroelastic version.
- `SolverMuJoCo` (Newton contacts, standard `CollisionPipeline`); collides once per
  frame, not per substep.

`example_pyramid`:
- ~20 stacked pyramids of boxes + (non-test) a wrecking-ball sphere on an inclined ramp.
- the ball rolls down the ramp and smashes the pyramids; in test mode the ball is
  omitted and the stacks should just stand. Stress-tests narrow-phase contact at scale.
- `SolverXPBD` (2 iters, SAP broad phase): the cheap position-based solver is robust
  for large rigid box stacks.

`example_brick_stacking`:
- a Franka FR3, a static table, three SDF-mesh LEGO bricks, a static floor brick.
- an APPROACH->GRASP->LIFT->MOVE->PLACE FSM with IK picks the red brick, stacks it on
  green, then moves the pair onto blue (contact-driven, no attachment). Demonstrates
  SDF-mesh pick-and-place.
- `SolverMuJoCo` (Newton contacts, impratio=50) for robot + bricks in one solver +
  `newton.ik.IKSolver` for arm targets; MuJoCo fits the stiff frictional grasp.

`example_contacts_rj45_plug`:
- a static socket, a dynamic plug, a spring-loaded revolute latch (SDF meshes), and a
  trailing rod/cable with a pinned far end.
- the user drags the plug into the socket; the latch deflects and locks on full
  insertion while the cable sags. Demonstrates interactive SDF insertion with a
  compliant latch + coupled cable.
- single `SolverVBD` (12 iters, `rigid_contact_hard=False`): one solver for the rigid
  meshes, joints, and cable together with soft compliant contacts.

## ik

These use `newton.ik.IKSolver` only — IK generates joint coordinates (rendered via
`eval_fk`); there is no physics integrator or contact solve.

`example_ik_franka`:
- a single Franka FR3 + ground plane.
- the user drags a gizmo to set the TCP pose; IK re-solves every frame so the arm
  tracks it. Kinematic tracking, no dynamics.
- `IKSolver` (position + rotation + joint-limit objectives, analytic Jacobian).

`example_ik_h1`:
- a single Unitree H1 (fixed base).
- four gizmos set both hand and both foot targets; IK satisfies all four end-effector
  poses simultaneously. Demonstrates multi-end-effector full-body IK.
- `IKSolver` (4x position + 4x rotation + joint limits).

`example_ik_custom`:
- a single Franka FR3 + a movable obstacle particle.
- one gizmo sets the TCP target, another moves the obstacle; IK reaches the target
  while a custom softplus objective pushes 7 links off the sphere. Demonstrates a
  user-defined `IKObjective`.
- `IKSolver` with the L-BFGS optimizer + custom `CollisionSphereAvoidObjective`
  residuals — collision avoidance solved inside IK as a penalty.

`example_ik_cube_stacking`:
- a Franka + cubes (MuJoCo + IK), per the inventory — IK plans the stacking poses while
  MuJoCo integrates the grasp dynamics.

## multiphysics

`example_rigid_soft_contact`:
- one rigid sphere + one deformable tetrahedral FEM grid (Neo-Hookean) + ground.
- the sphere falls from z=2.5 onto the grid; the grid compresses and recovers while the
  sphere settles. The reference scene for TWO-WAY rigid<->soft contact in one solver.
- selectable `SolverXPBD` (default) / `SolverSemiImplicit` / `SolverVBD`, all single-
  solver two-way; 32 substeps for the heavy impact. This repo's object side is modeled
  on the VBD path.

`example_softbody_dropping_to_cloth`:
- one tetrahedral soft body + one cloth grid (edges fixed) + ground.
- the soft body falls onto the cloth; the cloth stretches and the body deforms as they
  interact. Demonstrates deformable<->deformable contact.
- `SolverVBD` (10 iters, self-contact on): one implicit solver for coupled soft body +
  cloth + their contacts.

`example_softbody_gift`:
- four stacked soft-body blocks + two cloth straps wrapped around them (one rotated
  90deg) + ground.
- the stack falls and the straps cinch to hold it together like a gift. Demonstrates
  soft-body + cloth wrapping/constraining contact.
- `SolverVBD` (15 iters, topology-filtered self-contact): handles the stacked bodies
  plus wrapping cloth in one solve.

## diffsim

All diffsim examples use `SolverSemiImplicit`, the fully differentiable native
integrator: gradients backprop through the rollout (via `wp.Tape`) to whatever is being
optimized. Contacts, where present, come from a `requires_grad` `CollisionPipeline`.

`example_diffsim_ball`:
- one differentiable particle + a static wall box + ground (restitution 1).
- launched with some velocity, it bounces off wall and floor toward a target. Gradient
  descent optimizes the INITIAL VELOCITY to hit the target; also validates analytic vs
  finite-diff gradients.

`example_diffsim_bear`:
- a tetrahedral soft-body bear on the ground.
- sinusoidal phase signals feed a tanh network producing per-tet muscle activations.
  Adam optimizes the NETWORK WEIGHTS to maximize forward momentum — the bear learns to
  run. Learned soft-body locomotion.

`example_diffsim_cloth`:
- a 16x16 cloth grid with aerodynamic lift/drag, no colliders.
- launched and glides under gravity + aero. Gradient descent optimizes the INITIAL
  VELOCITIES so the cloth's center of mass reaches a target. Differentiable cloth +
  aero.

`example_diffsim_drone`:
- a rigid quadcopter (4 thrust props) + a static capsule obstacle, run in parallel
  rollout copies.
- sampled-MPC: noised control waypoints spawn rollouts, SGD optimizes the CONTROL
  TRAJECTORIES against a cost (reach target, stay upright, avoid obstacle), and the real
  drone steps under the best one toward alternating targets. Differentiable MPC.

`example_diffsim_soft_body`:
- a small FEM soft grid + a static wall + ground (restitution 1).
- launched and bounces toward a target. SGD optimizes the per-tet MATERIAL (Lame)
  PARAMETERS so the center of mass hits the target. Differentiable system ID.

`example_diffsim_spring_cage`:
- one movable particle sprung to 8 fixed cage particles, no contacts.
- the springs pull it to equilibrium; gradient descent optimizes the SPRING REST
  LENGTHS so it settles at a target. Differentiable constraint design.

## mpm

All MPM examples use `SolverImplicitMPM` — the only solver that models elastoplastic /
granular / fluid continua via material points. Per-particle material attributes select
the rheology.

`example_mpm_granular`:
- a box of MPM sand particles + a configurable static collider (cube/wedge/concave
  ramp) + ground.
- particles free-fall and pile/flow around the collider as a frictional granular
  material. Forward sim only. One-way (particles vs static shapes).

`example_mpm_snow_ball`:
- a static sloped heightfield, a conforming snow-pack layer, and a ball of snow
  particles.
- the ball rolls down the slope, compressing and fracturing the pack (per-particle
  compression/damage). Demonstrates snow plasticity on terrain.

`example_mpm_viscous`:
- a static thick-walled funnel filled with viscous-fluid particles + ground.
- particles flow out through the bottom aperture under gravity as a viscoplastic liquid.
  Demonstrates viscous-fluid MPM with mesh collision.

`example_mpm_multi_material`:
- a kinematic base bed + sand, snow, and mud particle blocks in one model.
- all fall onto the bed, each behaving per its own per-particle rheology. Demonstrates
  heterogeneous materials coexisting in one MPM solve.

`example_mpm_beam_twist`:
- one elastic MPM beam (Neo-Hookean-like), both ends kinematic.
- the right end is kinematically rotated 360deg over 1000 frames, twisting the elastic
  beam (colored by deviatoric stress). Large elastic deformation under kinematic twist
  (one-way driving).

`example_mpm_grain_rendering`:
- a tall column of MPM sand on the ground.
- the column collapses and settles; many sub-particle render grains are advected per
  MPM particle for a denser visual. Demonstrates high-fidelity grain rendering.

`example_mpm_anymal`:
- an ANYmal C quadruped walking on a bed of MPM sand.
- a pretrained RL policy walks the robot forward; feet/shanks plow through the sand,
  which is treated as ONE-WAY (robot disturbs sand, sand doesn't push back).
- two solvers: `SolverMuJoCo` integrates the articulation under the policy,
  `SolverImplicitMPM` advances the sand reading the robot bodies as kinematic colliders.

`example_mpm_twoway_coupling`:
- a dozen heavy rigid boxes (MuJoCo model) above a bed of MPM sand.
- boxes fall and sink into the sand; MPM collider impulses are summed into per-body
  forces fed back to the rigid solver (with the prior step's force subtracted for a
  consistent BC) — TWO-WAY. The canonical two-way MPM-coupling pattern.
- `SolverMuJoCo` (`use_mujoco_contacts=False`) + `SolverImplicitMPM` with explicit
  impulse feedback between them.

## kamino

All Kamino examples use `SolverKamino` — a maximal-coordinate (PADMM) constrained-
dynamics solver, well suited to closed loops and batched heterogeneous worlds.

`example_kamino_basic_fourbar`:
- a four-bar linkage (rigid boxes + loop joints) replicated across worlds.
- reset to a valid pose with softened (squishy) PD gains; swings and settles under
  gravity + the loop constraint. Forward sim only. Closed-loop mechanism in maximal
  coordinates — where generalized-coordinate solvers struggle.

`example_kamino_basic_dr_testmech`:
- a Disney Research "TestMech" articulated mechanism, replicated, no collisions.
- warm-started and advanced under gravity + joint constraints with tight PADMM
  tolerances. Pure constrained articulated dynamics (collisions + FK solver off).

`example_kamino_basic_heterogeneous`:
- six different mechanisms (fourbar, nunchaku, hinged boxes, box pendulum, box-on-plane,
  cartpole), one per world, + ground.
- all simulate simultaneously under gravity/contacts, offset for viewing. Demonstrates a
  single Kamino solve over topologically DIFFERENT worlds.

`example_kamino_robot_dr_legs`:
- a Disney Research "DR Legs" legged robot (meshes + box colliders), replicated +
  ground.
- reset to a standing pose; an implicit joint-space PD controller stabilizes the legs
  as they settle. Contacts from Kamino's detector or a Newton pipeline (CLI-selectable).
  Contacting legged robot with PD control in maximal coordinates.

`example_kamino_robot_anymal_d`:
- an ANYmal D quadruped (self-collisions on), replicated + ground.
- reset 1 m up, falls, and settles onto the ground (velocities go small). Contacting
  quadruped settling in maximal coordinates.

## robot

Almost all robot examples use `SolverMuJoCo` (generalized coordinates), since these are
contact-rich articulated robots; the variations are policy vs PD targets, and Newton vs
MuJoCo contacts.

`example_robot_h1`:
- a Unitree H1 humanoid (USD), replicated (~4 worlds).
- all DOFs position-controlled to the default pose; the robot settles and stands
  stably. USD import + stable standing.
- `SolverMuJoCo` (100/50 iters): handles the stiff-jointed articulation; Newton contacts
  by default.

`example_robot_g1`:
- a Unitree G1 29-DOF humanoid with hands (USD), replicated.
- actuated joints position-controlled to default; the G1 balances standing. High-DOF
  stable stand.
- `SolverMuJoCo` (elliptic cone, impratio=100): the cone + high impratio give accurate
  hand/foot friction.

`example_robot_ur10`:
- a UR10 arm on a static pedestal (USD), replicated (~100).
- each DOF follows a precomputed sinusoidal joint-target trajectory written through an
  `ArticulationView` — continuous waving. Trajectory playback + batched target writes.
- `SolverMuJoCo(disable_contacts=True)`: the arm only tracks targets in free space, so
  contacts are unneeded and disabling them speeds the batch.

`example_robot_anymal_c_walk`:
- an ANYmal C quadruped (URDF) on the ground.
- a PhysX-trained RL TorchScript policy outputs position targets each step, walking the
  robot forward (with lab<->mujoco joint reordering). Closed-loop RL walking + follow
  camera.
- `SolverMuJoCo` (newton solver, ls=50): suits the contact-rich locomotion loop;
  Newton contacts by default.

`example_robot_anymal_d`:
- an ANYmal D quadruped (USD) added to ~8 worlds via `add_world`.
- placed at z=0.68, position-controlled to the default pose, drops slightly and settles
  standing (no policy). Batched multi-world quadruped standing.
- `SolverMuJoCo` (elliptic cone, impratio=100): stable foot contacts for standing.

`example_robot_allegro_hand`:
- a Wonik Allegro hand (fixed root) + a free cube (USD), replicated (~100).
- a kernel drives the 20 finger joints with a sinusoidal grasp target AND rotates the
  hand root via its joint parent transform, so the hand continuously rotates while
  holding the cube. In-hand manipulation + animating a kinematic root.
- `SolverMuJoCo` (elliptic cone, impratio=20, `use_mujoco_contacts=False`): dexterous
  many-contact grasping needs the elliptic cone + Newton contact pipeline.

`example_robot_policy`:
- a runtime-selected legged robot (g1/go2/anymal, USD).
- an IsaacLab/PhysX-trained RL TorchScript policy drives it with keyboard velocity
  teleop; the policy runs decimated and writes position targets. Loading/running
  pretrained policies + live control.
- `SolverMuJoCo` (uses MuJoCo's own contacts): generalized-coordinate articulation +
  contacts for legged RL.

`example_robot_panda_hydro`:
- a Franka FR3 + hand with pad meshes, a static table, a manipulated pen/cube, and a
  cup (SDF hydroelastic shapes).
- an analytic-Jacobian IK waypoint trajectory (rest->grasp->close->lift->over-cup->
  release) is broadcast to all worlds as joint targets, doing a pick-and-place into the
  cup. Hydroelastic SDF contacts + IK manipulation.
- `SolverMuJoCo` (`use_mujoco_contacts=False`, elliptic cone, impratio=1000) +
  `HydroelasticSDF` pipeline + a separate `ik.IKSolver`: MuJoCo handles arm dynamics
  while the SDF hydroelastic pipeline produces soft area-based grip contacts.

`example_robot_cartpole`:
- a cartpole articulation (cart + two poles, USD), replicated (~100), no contacts.
- second pole offset 0.3 rad; the poles swing freely (passive) and all worlds stay
  bit-identical. Basic articulation + cross-world determinism.
- `SolverMuJoCo` (`contacts=None`): contact-free articulated dynamics — the natural fit
  (Semi-Implicit / Featherstone also noted as alternatives).

## selection

These exist to demonstrate the `ArticulationView` selection API (batched reads/writes,
masked partial resets, runtime attribute edits), not new physics. All use
`SolverMuJoCo`.

`example_selection_multiple`:
- 3 stacked NV "ant" articulations per world, world replicated (~16) — multiple
  articulations per world.
- resets give randomized jump/spin velocities and joint poses; each step applies random
  per-DOF forces; alternating disjoint halves are reset every 2 s. Demonstrates
  `count_per_world`, masked resets, and batched root/DOF get/set.
- `SolverMuJoCo`: contact-rich ants; MuJoCo's own contacts (no explicit collide).

`example_selection_articulations`:
- two different articulations per world — an ant and an nv_humanoid — replicated (~16).
- resets randomize root jump/spin (ants and humanoids spin opposite); random forces on
  the ants; masked half-resets. Demonstrates managing TWO separate `ArticulationView`s
  over heterogeneous robots.
- `SolverMuJoCo`: mixed contact-rich articulations, MuJoCo contacts.

`example_selection_materials`:
- NV "ant" articulations replicated (~16).
- every 2 s a reset flips root y-velocity AND randomizes each shape's friction, then
  calls `notify_model_changed(SHAPE_PROPERTIES)`; the ants slide with varying grip.
  Demonstrates RUNTIME material edits via the view + the required solver notify.
- `SolverMuJoCo`: friction is exactly what MuJoCo's contact solver models.

`example_selection_cartpole`:
- cartpole articulations (USD), replicated (~16), no ground/contacts.
- initial states randomized via the view; each step reads `joint_q` and applies a
  bang-bang control force (`joint_f`) through the view. Demonstrates the obs->control
  loop via `ArticulationView` + `set_visible_worlds`.
- `SolverMuJoCo(disable_contacts=True)`: force-controlled articulation, no contacts
  needed.

## softbody

`example_softbody_franka`:
- a Franka Panda (FR3 URDF), a static table, a ground, and a deformable tetrahedral
  rubber-duck soft mesh on the table.
- a keyframe sequence (approach->descend->pinch->lift->hold->place->release) solved each
  frame by IK grasps, lifts, and replaces the duck — object motion from contact, not
  scripting. The convention source for this repo's Franka import + EE offset.
- `SolverFeatherstone` (robot as a kinematic integrator, particles/gravity off) +
  `SolverVBD` (duck, `integrate_with_external_rigid_solver=True`) + `IKSolver` for pose
  targets: split solvers, ONE-WAY robot->duck.

`example_softbody_hanging`:
- four tetrahedral soft-body grids (Neo-Hookean), each with a different damping value,
  pinned on the left.
- all start horizontal and sag under gravity into different drooped shapes — showing
  damping's effect. Damped volumetric soft-body elasticity.
- `SolverVBD` (10 iters, self-contact off): the only solver supporting volumetric
  tetrahedral soft bodies (the example explicitly rejects non-VBD).

## Integration patterns worth knowing

- Featherstone + VBD (`cloth_franka`, `softbody_franka`): advance the robot with
  particles disabled, run a `CollisionPipeline`, then VBD with
  `integrate_with_external_rigid_solver=True`. One-way coupling.
- MuJoCo + Newton contacts (`use_mujoco_contacts=False`) is one solver, not two: a
  Newton `CollisionPipeline` feeds contacts into the MuJoCo step. This repo's robot
  side uses exactly this.
- True two-way rigid<->deformable in one solver: `rigid_soft_contact` with VBD — the
  template for this repo's object side.
- `mpm_twoway_coupling`: explicit impulse feedback from deformable solver to rigid
  solver — the upstream pattern to follow if this repo ever adds object->robot force
  feedback.
- VBD examples call `builder.color()` before `finalize()`; kinematic driving writes
  `body_q` directly and refreshes contacts (`set_rigid_history_update`) when not
  colliding every substep.

# RoboLab Example Reference

IMPORTANT — RoboLab does NOT use Newton or any of the solvers above. Upstream RoboLab
(`_external/RoboLab`) is a manipulation benchmark built on **NVIDIA Isaac Lab / Isaac
Sim**, whose physics engine is **PhysX (GPU)**. There is no `SolverMuJoCo` / `SolverVBD`
/ split-solver framework anywhere in it (`grep` for `import newton` / `Solver*` returns
nothing). The "solver" in RoboLab is PhysX's TGS rigid-body solver, configured ONCE and
reused through-and-through. So this repo's split SolverMuJoCo+SolverVBD design and the
upstream RoboLab benchmark share only assets/look-and-feel, not physics — the
`robolabViz/` integration borrows RoboLab's DROID rendering + asset library, never its
simulation.

The single shared physics config (`robolab/core/environments/base.py`, applied to every
task via the base env, so it is genuinely uniform across all 120+ tasks AND every
AI-generated scene/task):

- PhysX `solver_type = 1` (TGS — the temporal-Gauss-Seidel rigid solver),
  `num_position_iterations = 32`, `num_velocity_iterations = 1`.
- `sim.dt = 1/120`, render every 8 steps; CCD on; `contact_offset = 0.02`,
  `rest_offset = 0.01`; `bounce_threshold_velocity = 0.2`,
  `max_depenetration_velocity = 100`; GPU pipeline (`use_fabric=True`).
- Objects are PhysX rigid bodies (`RigidBodyPropertiesCfg`). DeformableObject plumbing
  exists (`core/world`, `core/events/reset_pose`, logging) but NO shipped task/scene
  instantiates a `DeformableObjectCfg`/cloth/particle — every RoboLab task is a rigid
  Franka + rigid objects scene. (Contrast: deformables are the whole point of this
  repo's Newton object side.)
- Robot: Franka Panda articulation with PhysX implicit PD actuators
  (`ImplicitActuatorCfg`, per-joint stiffness/damping), articulation
  `solver_position_iteration_count = 8` (the DROID variant uses 64). Control is either a
  `DifferentialIKController` (DLS) for pose actions or a `JointPositionAction`, with a
  `BinaryJointPositionAction` gripper.

Because the physics config is shared, the per-example differences are in the CONTROL
path and scene, not the solver:

`examples/run_empty.py`:
- a registered task's Franka + its rigid objects (or the empty DROID scene).
- steps the env with random actions for one episode. Smoke test that the env builds and
  the action/observation loop runs.
- PhysX TGS (shared config); no Newton.

`examples/run_recorded.py`:
- a task's Franka + rigid objects, plus a recorded HDF5 trajectory.
- replays previously recorded joint/action trajectories to verify task behavior and
  success detection. Playback, not a fresh policy.
- PhysX TGS (shared config); the robot is driven from the recorded actions through the
  normal action path.

`examples/run_gripper_toggle.py`:
- a task's Franka + rigid objects.
- holds the arm at its current joint positions while toggling the gripper open/closed —
  sanity-checks the `BinaryJointPositionAction` gripper path on a new robot/scene.
- PhysX TGS (shared config); gripper via binary joint-position action.

`examples/run_abs_ik_demo.py`:
- a task's Franka + rigid objects.
- commands a sequence of absolute end-effector pose targets and measures position/
  rotation tracking error against tolerances. Validates the absolute-IK action.
- PhysX TGS (shared config); arm driven by `DifferentialIKController` (DLS,
  `use_relative_mode=False`).

`examples/run_rel_ik_demo.py`:
- a task's Franka + rigid objects.
- drives per-DOF relative end-effector deltas in +/- phases. Validates the relative-IK
  action.
- PhysX TGS (shared config); arm driven by `DifferentialIKController` (DLS,
  `use_relative_mode=True`).

Generated environments (the `/robolab-scenegen` + `/robolab-taskgen` skills, and the
`robolab/registrations/**` auto-registrations — DROID jointpos / abs_ik / lighting /
background variations / the example registration): all inherit the SAME base PhysX
config above. They vary the scene assets, robot action space, cameras, and
lighting/background — never the solver. So "RoboLab uses the same solver through-and-
through" is true WITHIN RoboLab (one PhysX TGS config everywhere); it is just not the
Newton solver stack this repo otherwise documents.
