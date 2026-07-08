# RoboLab VBD

Franka manipulation demos on Newton physics (`_external/newton`), built toward a realistic
deformable-object simulation environment. Detailed knowledge is split into `docs/` (indexed
below, read when relevant); this file holds the project vision, the standards/rules, and the
recurring gotchas an agent must know before editing.

## Central config: `settings.yaml` (repo root)

The one place OUTSIDE the package to flip project-wide options, loaded by
`deformableManipulationTools/settings.py` (missing keys fall back to defaults, so it may be partial).
Today it holds the **active robot** plus default scenic render look + device. Switch robots by editing
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
reproduces Newton's exact folding sequence through the standard proxy bridge at physical friction
(Newton's recipe: fingertip to table, FIXED 8 mm jaw — [docs/cloths.md](docs/cloths.md)). Cloth
demos drive the fingers with an explicit `finger_schedule` (Newton's fixed-width close), NOT the
force `GripController` — its engage/deadband constants are rigid-scale; retuning it for shells is
an open item in [docs/gripper.md](docs/gripper.md), as is the legacy box-slice proxy-pad
limitation (the default pad is now the finger's own collider; [docs/cloths.md](docs/cloths.md)
gotcha 8).

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
  cable, AND soft): a bidirectional asymmetric admittance regulator. A demo declares only a `GraspWindow`
  and the one allowed knob, its `force_target`. See [docs/gripper.md](docs/gripper.md). (Exception:
  CLOTH demos close to Newton's fixed 8 mm jaw via an explicit `finger_schedule` — a shell's ~0.5 N
  reaction is below the controller's rigid-scale engage floor; retune pending, see gripper.md.)
- **Rigid-only** → robot AND objects in ONE `SolverMuJoCo` (objects merged into the robot builder via
  `add_builder`), true two-way grasp, **CCD on** (`make_robot_solver`). The gripper closes to a FIXED
  target (`MUJOCO_GRIP.close_target`; no object-size preset width — contact + actuator effort hold it,
  cf. `_external/RoboLab`). No proxies/coupling; the single MuJoCo model is also the viz model.
- `pickplace_ycb_franka` (rigid meshes) auto-routes to MuJoCo; `pickplace_ycb_vbd_franka` is the same
  scene **plus a token soft cube**, which auto-routes it to VBD — the A/B twin proving the routing.
  (MuJoCo rigid-only ≈ 2.2× faster than the VBD+proxy path on the same ycb scene.)

Details: [docs/solver-architecture.md](docs/solver-architecture.md).

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
control law, close speed, engage threshold, deadband, gains, materials, proxies) is centralized and
identical for every object — rigid, cable, AND soft (one unified bidirectional admittance regulator;
see [docs/gripper.md](docs/gripper.md)). A demo script must NOT contain any other grasp detail: no
preset widths, no compressible/object-type flags, no per-object gains or biases. The demo specifies
*when* to grasp (the `GraspWindow` times, a policy concern) and *how hard* (`force_target`), nothing
more. If a grasp needs tuning beyond the target force, fix it centrally in the package, not the demo.

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
an explicit `finger_schedule`), and an optional `Sweep`. To add a demo, write one data file; nothing
in `example.py`/`demo_runner.py` changes. Run with `python example.py --demo examples/<name>.py` or the
shim `python -m examples <name>`; `--output-style` picks the renderer (`scenic` default →
`outputs/<robot>/<name>/{frames/, simulation.mp4}`, `<robot>`=active robot `short_name`; or `basic` →
`outputs/<name>.usd`). The generic `DataDrivenExample` (subclass of `robolabViz.scenic.ScenicGraspExample`)
turns the spec into the configure/plan/build_scene/policy the framework expects, so a demo gets the
RoboLab look + force grip + solver routing for free. A demo file must contain NO physics/solver/grip
detail — only scene + policy (the one grasp knob is `GraspWindow.force_target`). Import the public API
with `from deformableManipulationTools import …`. Grip-force tuning: [docs/gripper.md](docs/gripper.md).

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
grip with a spurious velocity-proportional force (~4e4 N). The re-derived physical contact `kd≈1e2`
now lives centrally in `params.py` (`CableConfig.contact_kd`, `GRIP.proxy_kd`), not in any example.
**If you bump Newton again, re-check both the internal damping AND the contact `kd`.**

## Recurring mistakes to avoid (update as they recur)

- **Never copy numbers from Newton's cm-gram examples verbatim** (this broke the cloth grasp for
  weeks and is the same class of error as the carried-over contact `kd` above). Newton's cloth
  example runs in centimetre-gram units: its `ke=1e4 / kd=1e1 / density=0.02` pasted into our
  metre-kg world were ~1000×/100×/0.1× off *relative to particle weight* — each number looked
  plausible alone. Convert the whole set ([M/T²] ×1e-3, [M/T] ×1e-3, bending ×1e-5, area density
  ×10, lengths ×0.01, and match dt), then verify the dimensionless groups η = ke_eff·dt²/m and
  kd_eff/kd_crit against the source. Also: VBD body↔particle contact AVERAGES the shape material
  into the contact, so the pad/table shape ke is part of the cloth contact — see
  [docs/cloths.md](docs/cloths.md).
- **A penalty pinch on a thin shell must close to a FINITE gap, never 0.** The MuJoCo fingers feel
  no VBD object, so they really do reach a commanded 0 width, and a zero-gap pinch EXPELS the cloth
  (measured: 17 captured particles → 0). Newton closes to 8 mm; `cloth_franka` does the same.
- **A robot with mesh colliders (e.g. the panda USD: each link is one `CONVEX_MESH`) needs two
  things the URDF robot didn't.** (1) **Mesh finger proxies must be DEEP-COPIED and left↔right
  filtered:** `build_gripper_proxies` copies the finger collider into the VBD object model (the
  DEFAULT pad = the finger's own collider). Sharing the robot's `Mesh` object frees its BVH under
  the object narrow phase (GJK-MPR faults, CUDA error 700), and the two mesh fingers must be
  collision-filtered against each other. The legacy `box_slice_proxy=True` opts a mesh finger into
  thin box SLICES (`_finger_box_slices`, primitive narrow phase) — kept only as the A/B twin
  `cloth_franka_sliceProxies`: the slice stack's stepped inner face sheds a pinched cloth wad
  (see [docs/cloths.md](docs/cloths.md) / `grip.py`).
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
- **Don't `shape_material_mu.fill_()` after finalize** — it clobbers per-shape friction that
  the grasp relies on (cable/pads set high on purpose).
- Verify changes with **instrumented headless `--viewer null --device cuda:0`** runs +
  `test_final`, not just by looking at a video.

## docs/ index — read when relevant

This repo's own docs:

- [docs/solver-architecture.md](docs/solver-architecture.md) — the solver framework rule, robot & VBD
  object solver config + `alpha=0`/ALM rationale, CUDA-graph capture rules, the verification standard.
- [docs/gripper.md](docs/gripper.md) — the centralized proxy grip + harvest (net-to-EE, no per-finger
  feedback) and the unified admittance force controller; per-demo knob is `GraspWindow.force_target`.
- [docs/deformables.md](docs/deformables.md) — cable (rod) + soft-FEM-block tuned parameters and reasons.
- [docs/cloths.md](docs/cloths.md) — adding a cloth deformable (`ClothConfig` + `add_cloth`) + the
  cloth gotchas (≈critical `soft_contact_kd`, particle self-contact, the flat-sheet grasp limit).
- [docs/examples.md](docs/examples.md) — per-example descriptions and run commands.
- [docs/robolab-graphics.md](docs/robolab-graphics.md) — the `robolabViz/` RoboLab-look renderer,
  customization surface, vendored assets, render gotchas.
- [docs/ONGOING.md](docs/ONGOING.md) — live log of in-flight work; read before editing an area it
  names as active. Trust the file, not a summary here.
- Creating/improving this CLAUDE.md → [docs/howToWriteCLAUDE.md](docs/howToWriteCLAUDE.md).

External references (large; consult on demand, don't read up front):

- [docs/NVIDIA_Newton_release.md](docs/NVIDIA_Newton_release.md) — authoritative reference for the
  two-way MuJoCo↔VBD proxy coupling + the Isaac Lab Franka-cube physics config / high-fidelity path.
- [docs/NVIDIA_cloth_manip.md](docs/NVIDIA_cloth_manip.md) — NVIDIA's Isaac Lab + Newton cloth /
  deformable-manipulation writeup. Read when working the cloth/towel/zip-tie roadmap.
- [docs/SOLVERS.md](docs/SOLVERS.md) — deep solver reference: why `SolverVBD` is fragile for *rigid*
  objects, the object↔gripper wiring, and an annotated catalog of Newton's upstream examples.
