# RoboLab VBD

Franka manipulation demos built on Newton physics (`_external/newton`). The
solver framework is chosen by scene contents (see **Solver Framework Selection**):
the split `SolverMuJoCo` (robot) + `SolverVBD` (objects) framework whenever any
deformable/soft object is present, and a single `SolverMuJoCo` (robot + objects)
for rigid-only scenes.

## Solver Framework Selection

Two-way contact only happens *inside one solver*; the MuJoCo↔VBD bridge is
unavoidably one-way (objects can't push the arm; grasp width is commanded/latched).
So the rule, driven by scene contents:

- **Any deformable or soft object present** (cable/rod, cloth, FEM block) → use the
  **split `SolverMuJoCo` (robot) + `SolverVBD` (all objects)** framework with the
  kinematic gripper-proxy bridge. VBD is the only Newton solver that hosts
  rigid+cable+soft+their mutual two-way contact in one world, so every object that
  must touch a deformable lives in the VBD object model. This covers
  `cable_rigidCube_franka`, `cable_soft_franka`, `rigidCube_soft_franka`,
  `soft_compression_franka`, `soft_pickplace_franka`, and any future
  deformable+soft+rigid scene.
- **Rigid bodies only** → use a **single `SolverMuJoCo`** for the robot *and* the
  objects (the Newton `brick_stacking` / `panda_hydro` / RoboLab pattern). This gives
  a true two-way frictional grasp (no proxy bridge, no commanded width / force latch),
  MuJoCo's mature convex/SDF/hydroelastic mesh contact, and drops all of SOLVERS.md
  §4's VBD-rigid-mesh fragility. Preferred for any new rigid-only demo.

Caveat: `pickplace_ycb_franka` is rigid-only but is *deliberately kept on the VBD
object framework* as the proof that VBD can host arbitrary rigid mesh shapes (so the
same scene could later gain a soft object without re-architecting). See **Light-body
contact stability** for the instability this exposes and the general fix.

## Project Objectives

- Build physically faithful Franka demos: grasp, carry, drop, and sweep objects with
  all object motion produced by contacts, friction, gravity, and solver dynamics.
- Cover the interaction matrix between a rod cable, rigid bodies (cube, sheet), and a
  soft FEM block: cable-cube, cable-soft, cube-soft, sheet-soft.
- Use the same solver framework in every example so it generalizes (SolverMuJoCo robot
  + SolverVBD objects + kinematic finger proxies as the contact bridge).
- Keep runtime low: device-resident substep loops captured into CUDA graphs.
- Verify every demo with instrumented headless runs (not just visually), including
  viz/sim parity.

## Physics Rules

Favor physically faithful simulation over visual shortcuts:

- No object self-attachment, auto-grasping, or teleporting into a grasp.
- No guided, scripted, or kinematically driven motion for passive scene objects.
- No collision-free bypasses between interacting objects.
- Robot motion may be commanded through actuators or targets, but object pickup,
  dragging, settling, and contact response must come from modeled contacts,
  constraints, gravity, and solver dynamics.
- If a demo cannot yet perform a task physically, leave the failure visible and improve
  the model, contacts, solver integration, or controller instead of hiding it.
- The kinematic gripper proxy bodies in the object model are only a split-solver
  contact bridge: they mirror the real Franka finger poses so objects collide against
  the imported finger collision geometry. They must not directly move, attach, or
  constrain objects.
- Don't touch ANYTHING in _external/. Build the codebase to be independent of it (to use files from _external/, either import them or copy them over: otherwise assume that this folder can be removed at ANY time and the codebase is still expected to run as intended).

## Current Examples

All in `examples/`, registered in `examples/__init__.py`
(default: `cable_rigidCube_franka`).

- `cable_rigidCube_franka.py` — Franka descends to a cable on the table, closes until
  the commanded grasp width, lifts, and sweeps side to side; a rigid cube sits on the
  table. 8 substeps.
- `cable_soft_franka.py` — same cable demo with a soft FEM block on the table; the
  swept cable dents and nudges the block. 16 substeps.
- `rigidCube_soft_franka.py` — Franka grasps a heavy rigid cube (steel density, ~1 kg)
  via a pre-grasp waypoint, carries it, and drops it half-offset onto a pillow-soft
  block (`k_mu=5e2, k_lambda=2.5e3`); the cube squashes the block edge and rolls off.
  16 substeps. Gripper is force-limited (`--grip-force-control`, default on): it
  closes until the contact reaction reaches a threshold, then latches — for the rigid
  cube this halts it at the surface. See **Force-Limited Gripper**.
- `soft_compression_franka.py` — Franka grasps a heavy metal sheet (2x the cube's
  mass; 18x12x0.8 cm plate + grasp handle) by its handle and drops it half-offset onto
  the soft block; the sheet settles tilted on the block edge holding ~1 cm sustained
  compression. 16 substeps.
- `soft_pickplace_franka.py` — Franka picks up a small graspable soft FEM block
  (~33 mm, `k_mu=2e3`), carries it across the table, and places it at a target. The
  force-limited gripper compresses the block *gradually* until the squeeze reaction
  reaches the threshold, then latches — the soft counterpart to the rigid stop-at-
  contact. 16 substeps. See **Force-Limited Gripper**.

Run commands:

```bash
python -m examples cable_rigidCube_franka --viewer usd --device cuda:0
python -m examples cable_soft_franka --viewer usd --device cuda:0
python -m examples rigidCube_soft_franka --viewer usd --device cuda:0
python -m examples soft_compression_franka --viewer usd --device cuda:0
python -m examples soft_pickplace_franka --viewer usd --device cuda:0
```

CPU smoke test:

```bash
python -m examples cable_rigidCube_franka --viewer usd --device cpu --num-frames 1 --output-path /tmp/robolab_vbd_smoke.usd --quiet
```

Terminal output is tee'd to `outputs/terminal`. Default length is 720 frames (12 s).

A RoboLab-graphics variant of `cable_soft_franka` lives at
`examples/example_cable_soft_franka_robolab.py` (same physics, RoboLab DROID
rendering) — see **RoboLab Graphics** below.

## Solver Architecture

Robot side:

- Franka imported from `franka_emika_panda/urdf/fr3_franka_hand.urdf` with
  `collapse_fixed_joints=True`, `force_show_colliders=False`,
  `enable_self_collisions=False` (Newton `cloth_franka` convention).
- End-effector control point: `fr3_link7` + local offset `(0, 0, 0.22)` — this lands
  at the fingertip-pad bottoms.
- Dynamics: `SolverMuJoCo(solver="newton", integrator="implicitfast", cone="elliptic",
  use_mujoco_contacts=False)` with a Newton `CollisionPipeline`.
- A hidden robot-side table collider keeps the gripper from passing through the table.
- IK (`newton.ik.IKSolver`, position+rotation+joint-limit objectives) solves the
  keyframe poses once at startup.

Object side:

- `SolverVBD` with 12 iterations, `rigid_contact_history=True`,
  `rigid_contact_stick_motion_eps=0.0`, `rigid_avbd_contact_alpha=0.0`, hard body-body
  contacts. Rationale: penalty-only contacts lose the lifted cable; with the defaults
  (`alpha=0.95`, sticky replay on) penetration accumulating against the moving pads
  spikes the contact force and ejects pinched objects with multi-m/s phantom kicks.
  Newton's `cable_twist` (also kinematically driven, persistent cable contact) uses the
  same alpha=0 choice.
- Object contacts: `CollisionPipeline(contact_matching="latest",
  soft_contact_margin=0.01)`.
- The object model contains the visible table, the manipulated objects, the soft block
  (where present), and the kinematic gripper proxies.
- The split-solver bridge is ONE-WAY (robot -> objects), like Newton's
  `cloth_franka`/`softbody_franka`. Objects cannot push back on the robot, so grasp
  widths must be commanded (see below).

Runtime:

- The substep loop is fully device-resident and captured into a CUDA graph (one
  `wp.capture_launch` per frame). The keyframe trajectory and proxy pose sync are Warp
  kernels reading frame time from a device buffer.
- Capture happens after one uncaptured warm-up frame (lazy allocations raise inside
  capture) and requires an even substep count (the state swap must return to its
  starting binding per captured frame). Falls back to the uncaptured loop on CPU or
  capture failure.
- Measured on an H200 (null viewer): `cable_rigidCube` 11.6 ms/frame, `cable_soft`
  66.8 ms/frame, `rigidCube_soft` 57.0 ms/frame (pre-capture baselines: 74.8 / ~306).

Visualization (IMPORTANT):

- The viewer renders a separate combined viz model (robot builder + object builder).
- `_sync_viz_state` must copy `particle_q`/`particle_qd` from the object sim state in
  addition to body transforms. Copying only bodies renders every soft body frozen at
  its rest shape while the simulation deforms it underneath — historically the root
  cause of all "soft body never deforms / objects penetrate it / contact happens
  before touching" reports.

## Gripper Grasping (One-Way Bridge)

- Objects exist only in the VBD model, so nothing stops the fingers in the robot
  model. Commanding fully closed drives the pads through the object. The close target
  must itself stop at the object: per-finger target =
  `object_half_width + object_contact_margin + proxy_margin - 1 mm interference`.
  The 1 mm interference times contact stiffness sets a bounded grip force.
  (Cable: 0.009; cube: 0.026; sheet handle: 0.013.)
- `gripper_open = 0.04`, the URDF prismatic upper limit.
- Proxies copy the imported finger collision shapes (4 boxes per finger; the fingertip
  pad is 17.5x15.2x18.5 mm with its inner face at the grip center at q=0).
  Proxy margin 0.001, gap 0.008 (broad-phase headroom), friction restored to mu=1.0
  after the blanket material fill.
- Proxy pose sync follows VBD's kinematic protocol: write `body_q` on both states each
  substep; VBD finite-differences against its internal `body_q_prev` for contact
  friction velocity (matches Newton's `cable_twist` kinematic driving).
- Pre-grasp waypoint straight above wide objects is required: the joint-space path
  from home arcs sideways and the open pads (80 mm gap) clip a 50 mm object before the
  grasp.

## Force-Limited Gripper (`rigidCube_soft_franka`, `soft_pickplace_franka`)

The gripper closes to a FORCE THRESHOLD instead of a hand-tuned width, so it cannot
crush/penetrate the grasped object. The arm is deliberately NOT loaded by the object
(a real arm has the payload capacity to carry task objects); only the gripper DOFs are
governed by the contact.

- Threshold: `grip_force_threshold` (per finger). Grounded in the assets — Newton's
  Franka examples cap the gripper effort at 20 N and RoboLab sets the finger
  `effort_limit=200 N` (a saturation cap, not a grasp force); default here is 15 N
  (rigid) / 8 N (soft), `--grip-threshold` to override.
- Mechanism — a **force-triggered latch**, NOT continuous force feedback. The gripper
  is position-controlled (stable) and creeps closed at `grip_close_rate` (0.02 m/s);
  each substep a device kernel (`_update_grip_target_kernel`) reads the contact
  reaction and, the moment it reaches the threshold, latches the current width and
  holds it. No stiff in-loop force feedback, so the grasp cannot eject.
  - **Why not continuous feedback:** feeding the object reaction back into the gripper
    DOF (or the arm) across one substep of lag is unstable against the stiff penalty
    contact (`k·dt²/m > 1`) — it chatters to hundreds of N and ejects the object.
    Tried and rejected; the latch sidesteps it (position control once latched).
- **Rigid** (`rigidCube_soft_franka`): the reaction is read from the PUBLIC
  `SolverVBD.collect_rigid_contact_forces`, projected onto the closing axis. A rigid
  contact is a near-step (0 → large within one substep of closing), so the latch fires
  at first contact — the gripper halts at the cube surface, undeformed; the held force
  reads high (stiff-contact response) but the cube is not penetrated (carried cleanly).
- **Soft** (`soft_pickplace_franka`): Newton exposes no soft-contact force readback, so
  the reaction is recomputed from the PUBLIC soft-contact geometry
  (`soft_contact_particle/shape/body_pos/normal` + `particle_q` + `particle_radius` +
  `soft_contact_ke`) as `ke·penetration` summed per proxy — VBD's own penalty law. The
  soft reaction ramps gradually with compression, so the latch fires after the block
  has been squeezed to the threshold. The proxies set `has_particle_collision=True` so
  the pads contact the soft block (the drop examples keep it `False`).
- Both stay on PUBLIC Newton API (`collect_rigid_contact_forces`, `Contacts.soft_*`,
  `particle_q`); nothing in `_external` is modified or imported from `newton._src`.
- Verified on cuda:0 (A100): rigid latches at the cube surface, carries and drops near
  the soft block (`test_final` PASS, deterministic); soft compresses gradually
  (reaction 0→~6 N), lifts the block to 0.25 m, carries it across the table, and places
  it within 5 mm of the target (`test_final` PASS, deterministic).

## Obstacle (table) non-penetration

The robot-side hidden table collider (`robot_contact_table`, in the robot MuJoCo model)
keeps the gripper from passing through the table — verified by driving the EE 8 cm
below the table top: it halts exactly at the surface. Any fixed obstacle the gripper
must not pass through should be added as a static collider in the robot model the same
way (it does not need to be in the VBD object model unless objects also collide with it).

## Light-body contact stability (`pickplace_ycb_franka`)

VBD's rigid contact is a penalty/ALM force `≈ ke·penetration` whose magnitude is
**mass-independent**, so a contact's stability is set by the dimensionless stiffness
`η = ke·dt²/m_reduced` of the *pair* (`m_reduced = m0·m1/(m0+m1)`). The object model
blanket-fills one stiffness (`shape_material_ke = 5e4`), which is fine for `η < 1` but
explodes once a light body pushes the pair past `η = 1`: with `alpha=0` (full
per-step penetration correction, required for the grasp — see SOLVERS.md §3) the
over-correction converts penetration into a velocity `∝ 1/m` too large to resolve, and
the lighter-coupled member is ejected at multi-m/s.

Measured (substeps=16 → `dt = 1/(60·16) = 1.04e-3 s`, `ke = 5e4`):

- bowl `0.5 kg`: `η = 0.11` → clean (cube knocks it, all rest at z≈0.078).
- bowl `0.05 kg`: `η = 1.09`; cube↔bowl pair `m_reduced = 0.04 kg` → `η_pair = 1.36` → the
  **cube** (not the bowl) is flung to z≈−11 m or (42, 0, 29) m, the bowl to y≈−21 m.
  Chaotic/non-deterministic: which body ejects varies run-to-run, but it is always the
  ≥1-member that shares the unstable pair. This is why "the cube also flies away."

**General fix — per-substep rigid-body velocity clamp** (`_clamp_body_velocity_kernel`,
`--max-body-speed`, default 3.0 m/s linear / 50 rad/s angular). Run on the object state's
`body_qd` after every VBD substep (and captured in the CUDA graph), it bounds any spike
before it feeds the next substep's inertial prediction, breaking the escalation loop. It
is the rigid analog of Newton's `particle_max_velocity` (which the soft examples already
rely on) and a **pure safety net**: the highest legitimate speed here is a free-fall drop
(~1.1 m/s) ≪ the clamp, so it never touches normal motion. Verified: with the clamp the
`0.05 kg` bowl behaves identically to the clean `0.5 kg` baseline (cube/bowl peak
1.09/0.79 m/s, both settle on the table; `test_final` PASS) — and the peaks stay *below*
the clamp because arresting the runaway early stops it ever escalating.

Why not retune stiffness instead: contact `ke` is *averaged* across the pair, so a light
body touching a stiff (`ke=5e4`) heavy one still sees `avg_ke ≥ 2.5e4` → `η_pair` can't be
brought under ~0.5 by softening the light body alone. More substeps (`η ∝ dt²`) also work
but cost speed and don't help geometric wedging (SOLVERS.md §4). The velocity clamp is the
cheap, mass/geometry/stiffness-agnostic net — `body_qd` is public API; nothing in
`_external` is touched.

## Cable

- `add_rod(..., wrap_in_articulation=True)`: radius 0.008, segment length 0.035,
  15 nodes, friction 1.5, density 1200 (realistic jacketed cable; lighter cables turn
  pinch-contact residuals into ejection kicks).
- Laid with a 2 cm bow (`_cable_layout_positions`): a perfectly straight round rod on
  a flat table has a free rolling mode (VBD has no rolling friction) and rolls off the
  table; the bow locks it geometrically. The grasp/IK target is the midpoint of nodes
  3 and 4 of the actual bowed layout.
- The start-position clamp accounts for the full cable extent so the whole cable rests
  on the table.
- Cable shape material (`ke=2e4, kd=20, mu=1.5`) is restored after the blanket object
  material fill.
- Cable-example timing: descend 0-2.8 s, close 2.8-4.0, hold 4.0-4.8, lift 4.8-6.8,
  sweep from 6.8 s at 0.18 Hz with amplitude smoothstep-ramped over 1.5 s (a step in
  target velocity kicks the pinched cable out of the grasp).

## Soft Body (FEM Block)

The soft body is the FEM grid from Newton's `rigid_soft_contact` example (the only
upstream two-way VBD rigid+soft scene), scaled to the table:

- `add_soft_grid(...)`, 4x4x4 cells of 0.0125 m (5x5x5 cm, 125 particles, ~12.5 g),
  centered at `soft_start_pos` on the table. `density=100`, `k_damp=1.0`.
- Stiffness per example: `k_mu=1e4, k_lambda=5e4` (upstream values) in `cable_soft`
  and `soft_compression`; pillow-soft `k_mu=5e2, k_lambda=2.5e3` in `rigidCube_soft`.
- Contact (upstream values): `soft_contact_ke=1e5`, `kd=1e-4`, `kf=1e3`, `mu=0.3`,
  `particle_max_velocity=50`, `particle_enable_tile_solve=False`, particle radius
  0.0035 (the contact boundary sits one particle radius above the rendered surface —
  large radii read as contact-before-touching).
- The body-particle pair material is the AVERAGE of `soft_contact_*` and the rigid
  shape's material, and VBD sums per-contact forces on the body. The dropping
  cube/sheet shapes are restored to `ke=1e5, kd=1e-4` so their pair matches upstream's
  sphere-grid pairing; averaging against the stiff table/pad shapes keeps body-body
  contacts stiff regardless.
- `rigid_body_particle_contact_buffer_size`: 4096 in the drop examples (a flat face
  contacts hundreds of particles; overflow drops contacts frame-to-frame and
  destabilizes impacts into NaN), 512 in `cable_soft`.
- Two-way coupling stability is governed by `sqrt(pair_ke / m_body) * substep_dt` and
  by particle mass; the 16-substep examples sit comfortably within it. Upstream uses
  32 substeps for a much heavier impactor.

## Verification

- Every demo is verified with instrumented headless runs (null viewer, `cuda:0`)
  reading the simulation state per frame: grasp tracking (distance from the EE grip
  point to the grasped object), finger joint positions, object heights, soft-particle
  displacement/compression, NaN/ejection/pass-through detection, and viz/sim particle
  parity after `render()`.
- `test_final` in each example asserts the demo's outcome (grasp held and lifted;
  object dropped and landed near the soft block; nothing tunneled), time-gated so
  1-frame smoke runs pass.
- Known, intentional limitations: coupling is one-way (objects cannot stop the
  fingers; grasp width is commanded); only the finger bodies have proxies (palm/arm
  links do not collide with objects); proxies have `has_particle_collision=False`
  (the gripper itself cannot touch the soft block).

# RoboLab Graphics (`robolab_viz/`)

An alternative renderer reproducing NVIDIA RoboLab's look (the DROID rig in
`_external/RoboLab/examples/run_recorded.py`: home-office dome, sphere key light,
maple work table, Franka pedestal, over-shoulder + wrist cameras) on top of the
*unchanged* Newton physics. Self-contained — it never imports the `robolab`
package or `isaaclab`, and RoboLab's fixture/material/background assets are
vendored into `assets/` (provenance in `assets/ATTRIBUTION.md`); removing
`_external/RoboLab` does not break it.

- `examples/example_cable_soft_franka_robolab.py` subclasses
  `cable_soft_franka.Example`, reuses its physics verbatim, and swaps only the
  visualization (forces `--viewer null`):
  `python -m examples cable_soft_franka_robolab --device cuda:0`.
- Customization surface: `robolab_viz/config.py`. `droid_scene_config(table=,
  background=)` returns RoboLab's DROID values as overridable dataclasses
  (fixtures, lights, cameras, object styles) — this is where future demos swap
  table / lights / cameras / styling.
- Configurable look: `--table {maple,oak,bamboo,black}` and `--background <name>`
  (e.g. `home_office`, `garage_2k`, `machine_shop_01_2k`). `available_tables()` /
  `available_backgrounds()` / `available_objects()` enumerate the vendored sets;
  `resolve_background` accepts the `_2k`-less stem too. The choices are visible in
  the raycast preview, not just the RTX path (see below).
- Outputs in `outputs/robolab_preview/`: `combined.mp4` (the only kept video) +
  per-camera PNG frames (`--png-every`, default 60) + `wrist_coverage.json`.
  Opt-in: `--usd` writes the time-sampled scene USD; `--npz` writes the state
  cache (+ `geometry.pkl`) that `robolab_viz.rerender` replays without
  re-simulating.

Architecture:

- Robot: the *same* `fr3_franka_hand.urdf` is converted to USD once via Isaac
  Sim's importer (`robot_usd.py`, subprocess, cached in `assets/robots/`) so the
  rendered robot matches the simulated one. It is puppeteered per frame by an
  *uncollapsed* FK mirror (`robot_fk.py`, `collapse_fixed_joints=False`) — the
  physics model collapses fixed joints, so the hand link the wrist camera mounts
  to has no simulated body. `test_final` asserts FK/sim parity on shared links.
- Scene + objects authored as a time-sampled USD with pure `pxr` (`stage.py`, no
  kit): fixtures via payload; cable/rigid objects generated from the Newton
  shapes; soft body as a per-frame deforming mesh. Two render paths consume the
  same USD: `raycast.py` (warp `wp.Mesh` raycaster, flat-shaded but
  geometrically exact — the default, runs on any CUDA GPU) and `isaac_render.py`
  (offline RTX). **This box is an H200 (Hopper, no RT cores): the RTX renderer
  cannot run here** (crashes at GPU init) — use it only on an RT-capable GPU.
- Visual table placed by `config.table_fixture_from_footprint(top_z, center_xy)`
  from the physics `table_pos/table_half`, so the visible surface coincides with
  the contact plane and covers the footprint (asserted in `test_final`).
- The raycast preview reproduces the table material and dome backdrop (not just
  the RTX path): each `FixtureConfig.texture_file` (an sRGB base-color PNG) is
  planar/triplanar-projected onto the fixture box in world meters
  (`texture_uv_scale` = m per tile), and on a ray miss the dome HDR/EXR is
  tone-mapped (auto-exposed Reinhard + sRGB) and sampled as an equirectangular
  (latlong, Z-up) backdrop. The four tables share one geometry and differ only
  in the top-slab material, so the preview maps the wood base-color while the
  USD/RTX path uses each table's MDL; `black` is matte paint (no texture, flat
  color). `geometry.pkl` carries the table texture (in the instances) and the
  dome path so `rerender` reproduces the same look.
- Vendored asset library (`assets/`, provenance in `assets/ATTRIBUTION.md`):
  `backgrounds/{default,indoors}/*.hdr|exr` (curated; **no outdoor env maps exist
  in RoboLab's set** — only PNG previews, so indoor/default only),
  `fixtures/table_{maple,oak,bamboo,black}.usda` (offline MDL refs), and
  `objects/{objaverse,ycb,hot3d}/*.usd` + `textures/` (apple, banana,
  rubiks_cube, bowl, mug) with a trimmed `objects/object_catalog.json`
  (dims/class) exposed via `config.object_asset(name)`. **Hard rule: nothing in
  the repo reads `_external/` at runtime — assume that checkout can be deleted.**

Gotchas (fixed, would recur):

- **DAE up-axis double-correction.** Isaac's importer Y-up->Z-up-rotates every
  Collada visual, but the Franka DAEs already declare `Z_UP`, mis-rotating each
  visual 90 deg (closed fingers end up sideways).
  `robot_usd.fix_dae_visual_orientation()` resets the spurious mesh orient to
  identity (FR3 visual origins are identity); runs once post-conversion,
  idempotent.
- **Wrist camera pose is repo-tuned, not RoboLab's** — RoboLab's offset is
  relative to a Robotiq gripper this repo doesn't use. Intrinsics are RoboLab's
  (`_WRIST_CAM`, focal 2.8); the pose is solved in the `fr3_hand` frame by
  raycast occlusion sweeps (`eye=(0.08,0,-0.025)`, `target=(0,0,0.18)`).
- **Don't use `child_translate_overrides` to re-center a fixture** — it shifts
  geometry off the authored layout (once moved the table +0.35 m in y); use
  `table_fixture_from_footprint` instead.
- **Build the raycast `wp.Mesh` on the first frame's real positions**, not the
  rest buffer (soft-body verts = 0 give a degenerate BVH `refit()` cannot repair
  -> every ray scans all faces, ~3.6 s/frame).

