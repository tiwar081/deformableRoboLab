# RoboLab VBD

Franka manipulation demos on Newton physics (`_external/newton`), built toward a realistic
deformable-object simulation environment. Detailed knowledge is split into `docs/` (indexed
below, read when relevant); this file holds the project vision, the standards/rules, and the
recurring gotchas an agent must know before editing.

## Central config: `settings.yaml` (repo root)

The one place OUTSIDE the package to flip project-wide options, loaded by
`deformableManipulationTools/settings.py` (missing keys fall back to defaults, so it may be partial).
Today it holds the **active robot** plus default render output (`render.style`: `usd` | `mp4` |
`mp4_advanced`, default `mp4`; plus table/background look) + device. Switch robots by editing
`robot:` — no code change. The two robots are **kinematically identical** (same arm, same TCP/finger
world pose at the shared `home_q`); they differ ONLY in the end-effector geometry and the load path,
so ALL physics/policy/IK code is shared:

- `franka_panda_isaacsim` (DEFAULT) — Isaac Sim Franka Panda, **native USD** (`assets/robots/franka_panda_isaacsim/franka.usd`),
  `add_usd`, `panda_*` links. Each link is ONE `CONVEX_MESH` (both visual + collider).
- `fr3_franka_hand` — Franka FR3 + hand, Newton **URDF** asset pack, `add_urdf`, `fr3_*` links, primitive box colliders.

`params.ROBOTS` is the registry; `FRANKA = ROBOTS[SETTINGS.robot]` is the active `RobotConfig` that
the whole codebase imports. A new robot = a new `RobotConfig` (set `loader`/`usd_path` + the link-name
suffixes) added to `ROBOTS`. The render path (`robolabViz/scenic.py`) reads `FRANKA` to mirror the
robot too (native-USD rendered/FK'd directly; URDF converted once via Isaac Sim).

## Project vision & priorities

Building a simulation environment on Newton that handles **deformable bodies and
deformable↔rigid interaction** realistically. Primary deformable types: **cables/wires,
zip-ties, clothing, towels**. The rod (cable) + FEM block are done; the cloth (`ClothConfig` =
Newton's shirt SI-converted, `add_cloth`) is done including the **flat-sheet grasp**: `cloth_franka`
reproduces Newton's folding sequence through the standard proxy bridge at fully per-object
PHYSICAL friction (recipe: fingertip pressed 5 mm below the table top — mu*N anchoring — and a
finite jaw gap never 0; [docs/physicsEngine/cloths.md](docs/physicsEngine/cloths.md)).
`cloth_franka` grips via the force `GripController` (`force_target=2 N`, inside the shell's
achievable squeeze — the target-relative admittance law converges to a stable ~8–9 mm pinch,
[docs/physicsEngine/gripper.md](docs/physicsEngine/gripper.md)); the default pad is the finger's own collider. Live status:
[docs/ONGOING.md](docs/ONGOING.md).

Final vision: render **custom scenes** (table + background + any combination of deformable
and rigid assets) and **simulate a robot policy** in them, with high graphical realism
(NVIDIA Isaac Sim / RTX is the likely high-fidelity render path).

Priorities, in order: **1) physics fidelity → 2) speed. → 3) render quality**

## Solver framework rule

Two-way contact only happens inside one solver; the MuJoCo↔VBD bridge is one-way. The framework
chooses the object solver **CENTRALLY** in `GraspExample.__init__`: a deformable is present iff
`object_builder.particle_count > 0` **OR** a `CABLE` joint exists (FEM/cloth create particles; a
rod/cable is capsule bodies + `CABLE` joints with NO particles, so it is detected by joint type —
else a cable-only scene misroutes to MuJoCo, whose FK skips `CABLE` joints; rigid boxes/meshes have
neither). An example only declares its scene; it never picks the solver.

- **Any deformable present** → split `SolverMuJoCo` (robot) + `SolverVBD` (ALL objects) + **dynamic
  finite-mass gripper-proxy bridge** (NVIDIA recipe: proxies mirror the fingers, the object's contact
  reaction is harvested and the net external load fed to the arm/EE one step later). VBD is the only
  Newton solver hosting rigid+cable+soft+mutual two-way contact in one world. **Here the grip is
  FORCE-CONTROLLED** by the centralized `GripController` with ONE unified law for every object (rigid,
  cable, AND soft): a bidirectional admittance regulator, asymmetric in its GAINS (opens ~20× more reluctantly than it closes; the jaw-speed cap is symmetric and physical). A demo declares only a `GraspWindow`
  and the one allowed knob, its `force_target`. See [docs/physicsEngine/gripper.md](docs/physicsEngine/gripper.md). (The
  controller's gain/deadband are derived PER TARGET by `GripConfig.window_params` — anchored
  bit-exact at 30 N, so low targets like the 2 N cloth pinch regulate briskly and converge instead
  of dying in an absolute deadband. Physical params — max finger speed, engage floor, filter tau —
  stay fixed for sim2real. See gripper.md "Knobs".)
- **Rigid-only** → robot AND objects in ONE `SolverMuJoCo` (objects merged into the robot builder via
  `add_builder`), true two-way grasp, **CCD on** (`make_robot_solver`). The gripper closes to a FIXED
  target (`MUJOCO_GRIP.close_target`; no object-size preset width — contact + actuator effort hold it,
  cf. `_external/RoboLab`). No proxies/coupling; the single MuJoCo model is also the viz model.
- `pickplace_ycb_franka` (rigid meshes) auto-routes to MuJoCo; `pickplace_ycb_vbd_franka` is the same
  scene **plus a token soft cube**, which auto-routes it to VBD — the A/B twin proving the routing.
  (MuJoCo rigid-only ≈ 2.2× faster than the VBD+proxy path on the same ycb scene.)

Details: [docs/physicsEngine/solver-architecture.md](docs/physicsEngine/solver-architecture.md).

## Physics rules (favor faithful simulation over visual shortcuts)

- No object self-attachment, auto-grasping, or teleporting into a grasp.
- No guided/scripted/kinematically-driven motion for passive scene objects.
- No collision-free bypasses between interacting objects.
- Robot motion may be commanded (actuators/targets), but object pickup, dragging, settling,
  and contact response must come from modeled contacts, constraints, gravity, solver dynamics.
- If a demo can't yet do a task physically, **leave the failure visible** and improve the
  model/contacts/solver/controller — don't hide it.
- The gripper proxies are only a contact bridge (dynamic finite-mass bodies slaved to the finger
  pose via the momentum-consistent undo); they must not directly move, attach, or constrain
  objects. Object reaction goes back to the **arm/EE** (net load), never into the gripper DOF.
- **No velocity clamps on objects** (robot/table excepted).
- **Nothing that makes sim2real harder**: the robot must not gain capabilities it lacks in the
  real world (e.g. retuning parameters that are fixed physical hardware properties, sensing
  signals a real gripper couldn't measure, or contact behavior no physical pad could produce).
- **Never read or depend on `_external/` at runtime.** Import or copy what you need; assume
  `_external/` (newton, RoboLab) can be deleted and the codebase must still run.

## Code layout — physics is centralized in `deformableManipulationTools/`

**Requirement (governs the structure of every change): all physics, robot, and asset properties
live in the `deformableManipulationTools/` package, never inline in an example.** An `examples/`
script declares ONLY (1) the **scene** (environment initialization — which objects, where;
table/background) and (2) the **policy** (the robot motion). If a change touches the solver loop,
robot, grip, contact materials, masses, or how an object's collision/viz geometry is built, it goes
in the package so every demo — and every *future* demo — inherits it and cannot reintroduce a
per-object bug (e.g. the ycb raw-concave-mesh ejection). If a new demo needs a knob the package
doesn't expose, add the knob to the package, don't special-case it in the example.

**THE GRASP IS FULLY CENTRALIZED — the ONLY object/demo-specific grasp knob is the target grasp
force** (`GraspWindow.force_target` [N], one per grasped object). Everything else about the grasp (the
control law, close speed, engage threshold, materials, proxies — and the target-DERIVED gain/deadband,
`GripConfig.window_params`) is centralized and identical in form for every object — rigid, cable, AND soft (one unified bidirectional admittance regulator;
see [docs/physicsEngine/gripper.md](docs/physicsEngine/gripper.md)). A demo script must NOT contain any other grasp detail: no
preset widths, no compressible/object-type flags, no per-object gains or biases. The demo specifies
*when* to grasp (the `GraspWindow` times, a policy concern) and *how hard* (`force_target`), nothing
more. If a grasp needs tuning beyond the target force, fix it centrally in the package, not the demo.

**CONTACT-MATERIAL OWNERSHIP (two standing constraints; mechanics + numbers in
[docs/physicsEngine/solver-architecture.md](docs/physicsEngine/solver-architecture.md) "Contact materials").** (1) An object
authors ONLY its own contact properties — it never defines another object's property FOR it, and
its builder must REGISTER them (`assets._register_material`) or the blanket proxy-fill silently
replaces them post-finalize (the 2026-07-11 table bug). (2) The coupling of two objects' materials
into one contact is ONE central pairing-blind law (geometric mu; band-limited harmonic ke/kd) —
never a per-pairing branch. Materials are REALISTIC per object (RoboLab friction anchors, contact
kd from per-pairing critical damping) with NO friction exceptions — the cloth fold compensates
physical friction with pressing normal force, not a mu stamp. Do NOT quietly re-inflate a mu/kd to
fix a grasp — grasp problems are policy (dwell/speed/force_target) or central-model problems.

- **`params.py`** — single source of truth for ALL physics parameters (frozen dataclasses):
  `FRANKA`, `GRIP`, `TABLE`, `CABLE`, `SOFT_BLOCK`, `RIGID_CUBE`, `PLATE`, and the
  DISTINCT YCB objects `RUBIKS_CUBE`/`BOWL_YCB`/`BANANA_YCB`. **ONE table (`TABLE`) is shared by
  EVERY demo** (the former per-demo `TABLE_YCB`/`TABLE_CLOTH` variants were removed; the ycb and
  cloth scenes are mapped onto it). **One** soft block + **one** rigid cube (1 kg) are shared
  identically by every non-YCB demo so the demos are cross-comparable — no per-demo object
  variants, and an example never creates or modifies an object. Asset builders
  register their authored contact material; the framework restores it after the blanket proxy-fill,
  so an example never re-applies a material override by hand.
- **`framework.py`** — `GraspExample`: owns the entire build (robot+solver, object-model assembly,
  finalize ordering, materials, masses, coupling, the **centralized per-deformable particle solver
  config** `_particle_solver_config`) + the substep loop + CUDA-graph capture + viz.
- **`demo_runner.py`** — `DemoSpec` data schema (`Obj`/`WP`/`Sweep`) + the generic `DataDrivenExample`
  + the one policy executor kernel that reproduces every demo motion (waypoint blend, tilt/yaw, force
  grip OR explicit fingers, cable sweep). This is what makes a demo a pure data file; `example.py` plays it.
- **`robot.py`** — Franka builder, MuJoCo solver, yaw-aware gripper IK (`solve_gripper_ik`).
- **`grip.py`** — dynamic finite-mass proxies (`build_gripper_proxies`) + `TwoWayProxyCoupling`
  (the one grip: net-to-EE feedback, rigid + soft-particle harvest, no cap).
- **`assets.py`** — object builders that encapsulate the collision/viz NUANCES: `add_table`,
  `add_cable`, `add_soft_block`, `add_cloth`, `add_rigid_box`, `add_rubiks_cube`, `add_ycb_mesh`
  (+ the centralized per-deformable particle VBD configs `PARTICLE_SOLVER_KWARGS` (FEM) /
  `cloth_particle_kwargs` (cloth self-contact), which `framework._particle_solver_config` auto-applies
  by deformable type — a demo never declares solver physics).
- **`mesh_collision.py` + `coacd_worker.py`** — concave meshes COLLIDE as coacd convex-hull pieces
  while the full mesh RENDERS (a raw concave mesh ejects the VBD solve — SOLVERS.md §4). coacd
  segfaults if co-loaded with Newton, so decomposition runs in a subprocess and is disk-cached.

**Each demo is a DATA FILE, not a script.** `example.py` (repo root) is the ONE runner; every
`examples/<name>.py` declares a single `DEMO = DemoSpec(...)` (in `deformableManipulationTools/demo_runner.py`)
holding ONLY the **scene** (a list of `Obj(kind, config, pos, …)`) and the **policy** — arm `WP`
waypoints (TCP pos + optional `yaw`/`tilt`), the grasp (`grasp_windows` → the force GripController, OR
an explicit `finger_schedule`), and an optional `Sweep` — plus OPTIONALLY the **render look**
(`render=robolabViz.RenderSpec(...)`: background/table/lights/cameras/object styles/quality knobs;
precedence CLI flag > RenderSpec > settings.yaml). To add a demo, write one data file; nothing
in `example.py`/`demo_runner.py` changes. Run with `python example.py --demo examples/<name>.py` or the
shim `python -m examples <name>`; `--output-style` picks the renderer — all artifacts in
`outputs/<robot>/<name>/` (`<robot>`=active robot `short_name`): `usd` (lightest, Newton time-sampled
`<name>.usd`), `mp4` (DEFAULT: lightweight flat-shaded `simulation.mp4`, both cameras), or
`mp4_advanced` (RoboLab look: HDRI-lit PBR ray tracing → `simulation_advanced.mp4` + `frames/` +
`wrist_coverage.json`;
deprecated aliases `basic`→`usd`, `scenic`→`mp4_advanced`). The generic `DataDrivenExample`
(subclass of `robolabViz.scenic.ScenicGraspExample`)
turns the spec into the configure/plan/build_scene/policy the framework expects, so a demo gets the
render + force grip + solver routing for free. A demo file must contain NO physics/solver/grip
detail — only scene + policy + optional render look (the one grasp knob is `GraspWindow.force_target`).
Import the physics API with `from deformableManipulationTools import …` and the render-look classes
with `from robolabViz import RenderSpec, ObjectStyle, …`. Grip-force tuning: [docs/physicsEngine/gripper.md](docs/physicsEngine/gripper.md).
Renderer details: [docs/rendering/robolab-graphics.md](docs/rendering/robolab-graphics.md).

## Newton version (environment gotcha)

`newton` is editable-installed from `_external/newton`, currently a fresh clone at **`6dfe7303`**
(Newton `v0.2.3-665`, README pin was `2a1d4215`). It keeps the **absolute VBD damping** semantics
(`kd` is absolute [N·s/m], not stiffness-relative `D=kd·ke`) and the objective `C=FᵀF` tet-damping
metric (rigid rotations no longer damped). It also adds fix **#3125** (rigid contact no longer
injects energy for yawed finite-radius/small-radius cable contacts).

**Damping must be re-derived per Newton bump — and the carried-over CONTACT `kd` were wrong.**
`add_rod`-internal damping (`stretch_damping`, `bend_damping`) and tet `k_damp` are tuned for this
build. But the per-shape **contact** `kd` values (cable `20·ke=4e5`, proxy `1e2·ke=5e6`) were
~1e4× the contact critical damping; once the alpha=0 force-runaway was removed they dominated the
grip with a spurious velocity-proportional force (~4e4 N). The contact `kd` now lives centrally in
`params.py`, re-derived 2026-07-10 from per-pairing critical damping (`CableConfig.contact_kd` and
`GRIP.proxy_kd` = 30 ≈ 0.9× critical for a cable node; `TABLE.object_kd` = 1e2 ≈ 0.22× critical for
the 1 kg cube), not in any example.
**If you bump Newton again, re-check both the internal damping AND the contact `kd`.**

## Recurring mistakes to avoid (update as they recur)

- **Never copy numbers from Newton's cm-gram examples verbatim** (this broke the cloth grasp for
  weeks and is the same class of error as the carried-over contact `kd` above). Newton's cloth
  example runs in centimetre-gram units: its `ke=1e4 / kd=1e1 / density=0.02` pasted into our
  metre-kg world were ~1000×/100×/0.1× off *relative to particle weight* — each number looked
  plausible alone. Convert the whole set ([M/T²] ×1e-3, [M/T] ×1e-3, bending ×1e-5, area density
  ×10, lengths ×0.01, and match dt), then verify the dimensionless groups η = ke_eff·dt²/m and
  kd_eff/kd_crit against the source. Also: VBD body↔particle contact mixes the SHAPE's stored
  material into the contact (arithmetic mean); the framework re-targets stored shape ke/kd
  centrally onto the harmonic mean of the two objects' own authored values (the contact-material
  rule above) — see [docs/physicsEngine/cloths.md](docs/physicsEngine/cloths.md).
- **A penalty pinch on a thin shell must close to a FINITE gap, never 0.** The MuJoCo fingers feel
  no VBD object, so they really do reach a commanded 0 width, and a zero-gap pinch EXPELS the cloth
  (measured: 17 captured particles → 0). Newton closes to 8 mm; `cloth_franka` does the same.
- **A robot with mesh colliders (e.g. the panda USD: each link is one `CONVEX_MESH`) needs two
  things the URDF robot didn't.** (1) **Mesh finger proxies must be DEEP-COPIED and left↔right
  filtered:** `build_gripper_proxies` copies the finger collider into the VBD object model (the
  DEFAULT pad = the finger's own collider). Sharing the robot's `Mesh` object frees its BVH under
  the object narrow phase (GJK-MPR faults, CUDA error 700), and the two mesh fingers must be
  collision-filtered against each other.
  (2) **Viz-first BVH:** the robot's mesh BVHs are shared into the viz model; finalizing viz LAST frees
  them and corrupts the robot narrow-phase (SOLVERS §4) — only manifests once the OBJECT side (cable
  capsules / ycb meshes) reuses the freed pool, so it looked like a cable/mesh-only crash.
  `_build_split_mujoco_vbd` finalizes the **viz model first** (it never collides; its stale BVH is harmless).
- **Viz shows soft bodies frozen / objects penetrating them / contact-before-touching** →
  `_sync_viz_state` must copy `particle_q`/`particle_qd` (not just body transforms) from the
  object sim state. This is the #1 recurring soft-body bug.
- **CUDA-graph capture** needs an **even substep count** and **one uncaptured warm-up frame**
  (lazy allocations raise inside capture). It falls back to the uncaptured loop on CPU/failure.
- **Never feed the object reaction into the gripper DOF / per-finger** — confirmed empirically: the
  pad reaction is outward, so routing it to each finger pushes the pads open and the grasp is lost
  (grip force → 0). Feed the **net** reaction to the arm/EE (the internal squeeze cancels) and keep
  the fingers position-controlled.
- **Unphysical grip force has MORE than one cause.** When the harvested grip force is absurd, check
  *both* the VBD contact mode (`alpha=0`+`rigid_contact_history` accumulate ALM `λ` → 1e4–1e6 N) AND
  the contact damping `kd` (overdamped absolute `kd` → a velocity-proportional ~1e4 N during motion).
  A dynamic finite-mass proxy diverges if *either* is unfixed; with both fixed it is stable (the old
  "dynamic proxies NaN structurally" claim was the overdamped `kd`, not the proxy).
- **Stiff penalty contact + light element ejects** (`η = ke·dt²/m_reduced > 1`, with VBD
  `alpha=0` over-correction). Prefer default-hard contacts + physical damping over `alpha=0`.
- **The rigid grasp has NO static friction, by measured choice** (SOLVERS.md §6 trade study).
  The current hard/ALM config zeroes the multiplier per substep → sub-cone creep (a heavy held
  object slowly pivots about the jaw axis; a steep wedge can slide out; raising mu does NOT help —
  the cone limit isn't what's binding). The measured dead ends — do NOT re-walk them:
  `rigid_contact_history=True` (any alpha) integrates ~10× on the kinematically-imposed pinch and
  EJECTS the object; low-passing the EE feedback wrench removes the arm's load-bearing contact
  flinch (10 ms → 1656 N pinch, object launched; 2 ms → no effect). The one working alternative,
  soft contacts (`rigid_contact_hard=False, friction_epsilon=0.05`), fixes the slip (plate pivot
  24.5°→3.9°) but makes the initial grasp JOLT/rattle the robot through the one-substep-lagged EE
  feedback — it is RESERVED for an explicit future re-opening of the object-slippage problem,
  NOT for routine tuning when some object slips. If that happens: both knobs are central physics
  config (ONE framework.py solver_kwargs line, identical for every demo — never per-demo), and
  re-run the full demo matrix watching hand-speed swing, not just test_final.
- **Don't `shape_material_mu.fill_()` after finalize** — it clobbers per-shape friction that
  the grasp relies on (per-object AUTHORED values — see the contact-material rule).
- Verify changes with **instrumented headless `--viewer null --device cuda:0`** runs +
  `test_final`, not just by looking at a video.

## docs/ index — read when relevant

Docs live in subfolders of `docs/`, grouped by pipeline stage. The ONE file at the root is
[docs/ONGOING.md](docs/ONGOING.md) — live log of in-flight work; read before editing an area it
names as active. Trust the file, not a summary here.

- [docs/project-overview.md](docs/project-overview.md) — strictly high-level map
  of the end-to-end pipeline and package layout; changes only on major restructuring.

`docs/physicsEngine/` — the simulation core:

- [solver-architecture.md](docs/physicsEngine/solver-architecture.md) — the solver framework rule, robot & VBD
  object solver config + `alpha=0`/ALM rationale, CUDA-graph capture rules, the verification standard.
- [gripper.md](docs/physicsEngine/gripper.md) — the centralized proxy grip + harvest (net-to-EE, no per-finger
  feedback) and the unified admittance force controller; per-demo knob is `GraspWindow.force_target`.
- [deformables.md](docs/physicsEngine/deformables.md) — cable (rod) + soft-FEM-block tuned parameters and reasons.
- [cloths.md](docs/physicsEngine/cloths.md) — adding a cloth deformable (`ClothConfig` + `add_cloth`) + the
  cloth gotchas (≈critical `soft_contact_kd`, particle self-contact, the flat-sheet grasp limit).
- [examples.md](docs/physicsEngine/examples.md) — per-example descriptions and run commands. These
  demos exist to test the physics engine; other pipeline stages (agentic, traj) use their own.
- [SOLVERS.md](docs/physicsEngine/SOLVERS.md) — deep solver reference (large; consult on demand): why
  `SolverVBD` is fragile for *rigid* objects, the object↔gripper wiring, and an annotated catalog
  of Newton's upstream examples.

`docs/agenticPipeline/` — scene/task/env generation + scoring:

- [agentic-pipeline.md](docs/agenticPipeline/agentic-pipeline.md) — the three-stage pipeline
  (`agent_pipeline.py` + `agentic_pipeline/`): scene gen (objects only + settle check) → task gen
  (task + ROBOT PLACEMENT: edge-touching robot table, base-aware reach) → env gen (look + cameras);
  robot-POV direction words, prompt templates in `agentic_pipeline/prompts/`, the SKILL.md
  interactive session, and the --user / userless / --scene_init modes.
- [scene-generator.md](docs/agenticPipeline/scene-generator.md) — the agentic scene generator
  (`agentic_pipeline/scene_generator.py`): prompt → LLM → spatial solver → settle-only DemoSpec → over-the-shoulder
  still; the imported-object catalog (`assets/objects/scene_catalog.json`, incl. deformables +
  every inferred parameter); the RoboLab scene-gen vs environment-gen boundary.
- [task-generator.md](docs/agenticPipeline/task-generator.md) — the agentic task generator
  (`agentic_pipeline/task_generator.py`): scene → LLM → validate → deformable-aware feasibility checks → ideated
  `Task` (instruction variants + goal predicate + subtasks). Mirrors RoboLab's task-gen structure;
  adds affordance gating by object type (fold=cloth, coil=cable, stack=rigid) + folded-volume
  container fit. Ideates the task only — grasp force/trajectory are the downstream pipeline's job.
- [success-evaluators.md](docs/agenticPipeline/success-evaluators.md) — the SCORING layer
  (`agentic_pipeline/success.py`): `SceneState`, the `driver` field + `evaluable: false` semantics,
  and the queue of 11 unscorable predicates — why each AABB heuristic was withdrawn 2026-07-27 and
  what to build instead (two pairs are blocked on per-asset annotation, not geometry code).

`docs/trajPipeline/` — trajectory generation (grasp substrate + the executable stage):

- [README.md](docs/trajPipeline/README.md) — the stage between task gen and simulation: what
  exists, the pre-shaped-approach contract, and the index of the docs below. Live status is in
  ONGOING.md.
- [trajectory-generation.md](docs/trajPipeline/trajectory-generation.md) — the stage itself
  (`deformableManipulationTools/traj_gen/`, landed 2026-08-12): online selection (physics-tiered
  re-rank + score-weighted sampling), Bezier legs with collision-driven control points, the
  measured headless rollout, and the 2-attempt grasp-failure LLM loop. Consumes a pipeline run
  dir; writes `traj.json` (picked up by `build.demo_from_dir`) + `traj_result.json` + the video.
- [grasp-library.md](docs/trajPipeline/grasp-library.md) — the per-asset grasp record store:
  canonical OBB frame (+ ambiguity detectors), the pad-seated v2 pose convention, `pad_seat` and
  the three seat modes with their measured rationale, schema/versioning rules, catalog coverage.
- [grasp-passes.md](docs/trajPipeline/grasp-passes.md) — the parallel-agent pass framework and
  the seven passes (fixture, geometric, obb_face, obb_bucket, vlm_regions, rim_pinch,
  shake_validate): measured findings + dead ends beyond each pass's in-code docs.
- [grasp-selection.md](docs/trajPipeline/grasp-selection.md) — selection for a placed object
  (prune → clearance → projection → score → sample) and the Z-X-Z projection result that closed
  the IK-vocabulary question.

`docs/rendering/`:

- [robolab-graphics.md](docs/rendering/robolab-graphics.md) — the `robolabViz/` RoboLab-look renderer,
  customization surface, vendored assets, render gotchas.

`docs/SKILLS/` — working procedures:

- [update-ongoing.md](docs/SKILLS/update-ongoing.md) — when and how to write ONGOING.md entries.
- [promote-ongoing-to-docs.md](docs/SKILLS/promote-ongoing-to-docs.md) — moving durable ONGOING.md
  content into `docs/` and resetting the file (done per big task).
- [writing-claude-md.md](docs/SKILLS/writing-claude-md.md) — creating/improving this CLAUDE.md.

`docs/external/` — external works (large; consult on demand, don't read up front):

- [NVIDIA_Newton_release.md](docs/external/NVIDIA_Newton_release.md) — authoritative reference for the
  two-way MuJoCo↔VBD proxy coupling + the Isaac Lab Franka-cube physics config / high-fidelity path.
- [NVIDIA_cloth_manip.md](docs/external/NVIDIA_cloth_manip.md) — NVIDIA's Isaac Lab + Newton cloth /
  deformable-manipulation writeup. Read when working the cloth/towel/zip-tie roadmap.
- [robolab.md](docs/external/robolab.md) — summary of NVIDIA's RoboLab benchmark (paper + the
  `_external/RoboLab` repo): our sim2real reference (arm gains, friction anchors) and the
  `robolabViz` look source; its stated deformables gap is this project's vision.
