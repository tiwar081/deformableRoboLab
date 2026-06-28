# RoboLab VBD

Franka manipulation demos on Newton physics (`_external/newton`), built toward a realistic
deformable-object simulation environment. Detailed knowledge is split into `docs/` (indexed
below); this file holds the project vision, the standards/rules, and the recurring gotchas an
agent must know before editing.

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
zip-ties, clothing, towels**. The rod (cable) + FEM block are done; the cloth **shell**
infrastructure exists (`ClothConfig`, `add_cloth`, [docs/cloths.md](docs/cloths.md)) but
cloth *manipulation* is in flight — the gripper can't yet move a flat sheet (see
[docs/ONGOING.md](docs/ONGOING.md)).

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
  and the one allowed knob, its `force_target`. See [docs/gripper.md](docs/gripper.md).
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
  `FRANKA`, `GRIP`, `TABLE`/`TABLE_YCB`, `CABLE`, `SOFT_BLOCK`, `RIGID_CUBE`, `PLATE`, and the
  DISTINCT YCB objects `RUBIKS_CUBE`/`BOWL_YCB`/`BANANA_YCB`. **One** soft block + **one** rigid
  cube (1 kg) are shared identically by every non-YCB demo so the demos are cross-comparable — no
  per-demo object variants, and an example never creates or modifies an object. Asset builders
  register their authored contact material; the framework restores it after the blanket proxy-fill,
  so an example never re-applies a material override by hand.
- **`framework.py`** — `GraspExample`: owns the entire build (robot+solver, object-model assembly,
  finalize ordering, materials, masses, coupling) + the substep loop + CUDA-graph capture + viz.
  A demo subclass implements `configure`/`plan`/`build_scene`/`set_robot_targets`/`test_final`.
- **`robot.py`** — Franka builder, MuJoCo solver, yaw-aware gripper IK (`solve_gripper_ik`).
- **`grip.py`** — dynamic finite-mass proxies (`build_gripper_proxies`) + `TwoWayProxyCoupling`
  (the one grip: net-to-EE feedback, rigid + soft-particle harvest, no cap).
- **`assets.py`** — object builders that encapsulate the collision/viz NUANCES: `add_table`,
  `add_cable`, `add_soft_block`, `add_cloth`, `add_rigid_box`, `add_rubiks_cube`, `add_ycb_mesh`
  (+ the centralized `PARTICLE_SOLVER_KWARGS` / `cloth_solver_kwargs` VBD configs).
- **`mesh_collision.py` + `coacd_worker.py`** — concave meshes COLLIDE as coacd convex-hull pieces
  while the full mesh RENDERS (a raw concave mesh ejects the VBD solve — SOLVERS.md §4). coacd
  segfaults if co-loaded with Newton, so decomposition runs in a subprocess and is disk-cached.

`examples/` keeps only the thin demo scripts + the run harness (`__init__.py`); the shared terminal
helper lives in `deformableManipulationTools/helper.py`. Each demo is **one file** `<name>.py` (no
separate `_robolab` files); it subclasses `robolabViz.scenic.ScenicGraspExample` and `--output-style`
picks the renderer — `scenic` (default: `outputs/<robot>/<name>/{frames/, simulation.mp4}`, both
policy cameras; `<robot>` = the active robot's `short_name`) or `basic` (`outputs/<name>.usd`). The
scenic glue (`robolabViz/scenic.py`) reads the robot
base pose / table / soft-object position off the physics example, so a new demo gets the RoboLab look
for free. Import the public API with `from deformableManipulationTools import …`.
Grip-force tuning: [docs/gripper.md](docs/gripper.md).

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

- **A robot with mesh colliders (e.g. the panda USD: each link is one `CONVEX_MESH`) needs two
  things the URDF robot didn't.** (1) **Box finger proxies:** `build_gripper_proxies` copies the
  finger collider into the VBD object model; a CONVEX_MESH proxy is contacted LATE then explosively
  by the VBD penalty solver (spiked the grip to ~2 kN at a ~2 mm latch). It auto-detects a non-box
  finger collider and substitutes a per-finger AABB box (well-behaved, ~60 N, pad face matches fr3 to
  <0.5 mm). (2) **Viz-first BVH:** the robot's mesh BVHs are shared into the viz model; finalizing viz
  LAST frees them and corrupts the robot narrow-phase (SOLVERS §4) — only manifests once the OBJECT
  side (cable capsules / ycb meshes) reuses the freed pool, so it looked like a cable/mesh-only crash.
  `_build_split_mujoco_vbd` finalizes the **viz model first** (it never collides; its stale BVH is
  harmless). The panda's per-link finger orientation also differs (both fingers identity-posed; the
  right pad is a MIRRORED mesh, not a flipped body) — the per-finger AABB handles this automatically.
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

## docs/ index

This repo's own docs:

- [docs/solver-architecture.md](docs/solver-architecture.md) — solver framework rule, robot &
  VBD object solver config + the `alpha=0`/ALM rationale, CUDA-graph capture rules, the viz
  particle-copy bug, the verification standard.
- [docs/gripper.md](docs/gripper.md) — the **centralized** dynamic finite-mass proxy grip
  (`deformableManipulationTools/grip.py`, params in `params.py`): net-to-EE feedback, rigid +
  soft-particle harvest, the no-per-finger-feedback stability invariant, and **the force controller**
  — ONE unified bidirectional asymmetric admittance regulator for every object (rigid, cable, soft);
  the one per-demo knob is `GraspWindow.force_target`.
- [docs/deformables.md](docs/deformables.md) — cable (rod) and soft-FEM-block tuned parameters
  + reasons; notes on future zip-tie deformables.
- [docs/cloths.md](docs/cloths.md) — how to add a **cloth-type** deformable (shirt/towel/sheet):
  `ClothConfig` + `add_cloth` + `cloth_solver_kwargs`, the grasp knob, and the cloth-specific gotchas
  (the ≈critical `soft_contact_kd` a thin shell needs, particle self-contact, the flat-sheet grasp limit).
- [docs/examples.md](docs/examples.md) — per-example descriptions and run commands.
- [docs/robolab-graphics.md](docs/robolab-graphics.md) — the `robolabViz/` RoboLab-look
  renderer (raycast + offline RTX), customization surface, vendored assets, render gotchas.
- [docs/ONGOING.md](docs/ONGOING.md) — the **live log of in-flight work + recent changes**
  (volatile, changes often). Always read it for the current state before editing active areas
  (e.g. the cable coupling / gripper). Its own header lists what's currently unresolved — trust
  the file, not a summary here.
- When asked to create/improve this CLAUDE.md → [docs/howToWriteCLAUDE.md](docs/howToWriteCLAUDE.md)
  (the working procedure: keep it short, universal, point-don't-paste, "if X then Y").

External references (large; consult on demand, don't read up front):

- [docs/NVIDIA_Newton_release.md](docs/NVIDIA_Newton_release.md) — NVIDIA's Newton GA release
  article. **The authoritative reference for the cable two-way MuJoCo↔VBD coupling** (dynamic
  finite-mass proxies, `sync_proxy_state` momentum-consistent undo, staggered one-step-lag
  step), plus the Isaac Lab Franka-cube physics config (`impratio=1000`, `iterations=20`, …),
  Kamino closed-chain, and hydroelastic-SDF contact. Primary source for the coupling work and
  the Isaac Lab / high-fidelity path.
- [docs/NVIDIA_cloth_manip.md](docs/NVIDIA_cloth_manip.md) — NVIDIA's Isaac Lab + Newton **cloth /
  deformable manipulation** blog extract (VBD for thin deformables, Franka cloth grasp). Read when
  starting the cloth/towel/zip-tie deformables on the project roadmap.
- [docs/SOLVERS.md](docs/SOLVERS.md) — deep solver reference: why `SolverVBD` is fragile for
  *rigid* objects (the basis of the framework rule), how the object↔gripper two-way physics is
  wired, and an annotated catalog of Newton's upstream examples (cable, cloth, mpm, kamino,
  softbody, …) + RoboLab examples. Check here for upstream patterns when adding a new object
  type, solver, or render feature.
