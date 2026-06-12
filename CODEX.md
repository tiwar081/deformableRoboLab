# SolverVBD Example Inventory

## Project Physics Rules

This codebase should favor physically faithful simulation over visual shortcuts.

- Do not make objects self-attach, auto-grasp, or teleport into a grasp.
- Do not give passive scene objects guided, fully scripted, or kinematically driven motion to fake interaction.
- Do not make interacting objects collision-free to avoid solver/contact problems.
- Robot motion may be commanded through actuators or targets, but object pickup, dragging, settling, and contact response must come from modeled contacts, constraints, gravity, and solver dynamics.
- If a demo cannot yet perform a task physically, leave the failure visible and improve the model, contacts, solver integration, or controller instead of hiding it with scripted object motion.

Repository inspected: `_external/newton`

Scope: source examples under `_external/newton/newton/examples` that directly instantiate
`newton.solvers.SolverVBD` or expose a VBD solver path. I found 21 such examples. The
DeepWiki answer that only names a few test entries is incomplete for this checkout.

## Complete Example List

- `basic.example_basic_conveyor`
- `basic.example_basic_joints`
- `basic.example_basic_shapes`
- `basic.example_basic_urdf`
- `cable.example_cable_bundle_hysteresis`
- `cable.example_cable_cross_slide_table`
- `cable.example_cable_pile`
- `cable.example_cable_twist`
- `cable.example_cable_y_junction`
- `cloth.example_cloth_bending`
- `cloth.example_cloth_franka`
- `cloth.example_cloth_hanging`
- `cloth.example_cloth_poker_cards`
- `cloth.example_cloth_rollers`
- `cloth.example_cloth_twist`
- `contacts.example_contacts_rj45_plug`
- `multiphysics.example_rigid_soft_contact`
- `multiphysics.example_softbody_dropping_to_cloth`
- `multiphysics.example_softbody_gift`
- `softbody.example_softbody_franka`
- `softbody.example_softbody_hanging`

Test harness cross-check:

- `_external/newton/newton/tests/test_examples.py` explicitly runs `basic.example_basic_urdf`
  with `--solver vbd`, `basic.example_basic_joints` with `--solver vbd`,
  `cloth.example_cloth_hanging` with its default VBD solver, and
  `multiphysics.example_rigid_soft_contact` with `--solver vbd`.
- `_external/newton/newton/tests/test_rigid_friction_ramp.py` has VBD visual/debug solver
  cases using `SolverVBD(model, iterations=40, rigid_contact_k_start=1.0e5)`.

## Solver Integration In `examples/example_minimal_cable_franka.py`

This project script uses one unified Newton model and one `SolverVBD` instance for the
robot, rigid object, table, and rod cable. It does not split the robot into Featherstone
or MuJoCo while using VBD only for the cable.

- Model construction: `Example._build_model()` creates one `newton.ModelBuilder`, imports
  the Franka URDF, adds a static table, adds a dynamic box labeled `non_touching_object`,
  and creates the cable with `builder.add_rod(..., wrap_in_articulation=True)`. The rod
  cable is therefore represented as rigid capsule bodies connected by rod/cable joints,
  not as a particle soft body.
- Finalization and state setup: the builder is finalized once, `newton.eval_fk()` initializes
  body poses from joint coordinates, one `model.control()` buffer is created, and
  `control.joint_target_q` is initialized from `model.joint_q`.
- Solver: the script constructs exactly one solver:
  `newton.solvers.SolverVBD(self.model, iterations=args.vbd_iterations,
  particle_enable_self_contact=False, particle_enable_tile_solve=False,
  rigid_contact_hard=True, rigid_body_contact_buffer_size=512,
  rigid_body_particle_contact_buffer_size=512)`.
- Contacts: it uses `self.contacts = self.model.contacts()` and calls
  `self.model.collide(self.state_0, self.contacts)` every substep before the VBD step.
  There is no explicit `CollisionPipeline` and no contact refresh throttling in this script.
- Robot control: `_set_robot_targets()` writes a time-varying target joint configuration into
  `self.control.joint_target_q`. VBD then consumes that control buffer in the same
  `self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)`
  call that advances the cable, object, table contacts, and robot articulation.
- Practical implication: this is the "single VBD owns everything" pattern. It is simpler
  than the Franka cloth/softbody examples, but robot articulation quality and contact
  behavior are whatever `SolverVBD` provides for this mixed rigid/rod scene.

Useful comparison examples:

- `_external/newton/newton/examples/multiphysics/example_rigid_soft_contact.py` is also a
  single-solver mixed scene. It adds both a rigid sphere and a soft grid to one model, then
  selects exactly one solver (`SolverSemiImplicit`, `SolverXPBD`, or `SolverVBD`) and calls
  `model.collide()` followed by that solver's `step()` every substep.
- `_external/newton/newton/examples/softbody/example_softbody_franka.py` is the main
  two-solver Franka pattern. It uses `SolverFeatherstone` for the robot and `SolverVBD`
  for the soft body with `integrate_with_external_rigid_solver=True`. In the simulation
  loop it temporarily disables particles, advances the robot, restores particles/gravity,
  collides with a `CollisionPipeline`, then steps VBD for the soft body.
- `_external/newton/newton/examples/cloth/example_cloth_franka.py` uses the same two-solver
  integration idea for cloth: `SolverFeatherstone` updates the robot, then `SolverVBD`
  with `integrate_with_external_rigid_solver=True` advances cloth and cloth-body contacts.
- `_external/newton/newton/examples/cable/example_cable_cross_slide_table.py` is a useful
  VBD cable/rod reference. It also uses one `SolverVBD` for rigid bodies, pulleys, joints,
  and cable rods, but adds an explicit `CollisionPipeline(broad_phase="explicit",
  contact_matching="latest")`, `rigid_contact_history=True`, and direct kinematic body
  driving for input pulleys.

## Per-Example Notes

### basic.example_basic_shapes

File: `_external/newton/newton/examples/basic/example_basic_shapes.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 37. VBD mode raises default shape
  contact stiffness/damping at lines 41-44 and calls `builder.color()` at lines 97-99.
- Rigid bodies: adds a ground plane, then separate rigid bodies for sphere, ellipsoid,
  capsule, cylinder, box, mesh bunny, and cone using `add_body()` plus shape methods at
  lines 49-95.
- Deformables/cables/cloth: none.
- Solver: `SolverVBD(self.model, iterations=10)` when `--solver vbd` is selected at
  lines 105-109; otherwise XPBD.
- Contacts: creates `self.contacts = self.model.contacts()` and calls
  `self.model.collide(self.state_0, self.contacts)` every substep at lines 117 and 147.
- Controls: creates `self.control = self.model.control()` at line 115. No authored actuator
  targets; viewer forces are applied each substep.

### basic.example_basic_joints

File: `_external/newton/newton/examples/basic/example_basic_joints.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 45. It always calls
  `builder.color()` before finalizing at line 178.
- Rigid bodies: creates three articulations from links and joints: fixed+revolute,
  fixed+prismatic, and fixed+ball. Geometry is boxes/spheres on `add_link()` bodies at
  lines 68-172, plus a ground plane.
- Deformables/cables/cloth: none.
- Solver: `SolverVBD(self.model, iterations=2)` for `--solver vbd` at lines 184-189;
  otherwise XPBD. `eval_fk()` is called before solver construction so VBD sees the edited
  joint pose as the structural rest pose.
- Contacts: creates `model.contacts()` at line 199, then calls `model.collide()` every
  substep at line 220.
- Controls: creates a `model.control()` buffer at line 195. Initial joint positions are
  set in the builder, but no runtime actuator command is authored beyond viewer forces.

### basic.example_basic_urdf

File: `_external/newton/newton/examples/basic/example_basic_urdf.py`

- ModelBuilder: uses two builders. `quadruped = newton.ModelBuilder()` at line 39 loads the
  URDF; `scene = newton.ModelBuilder()` at line 77 replicates the quadruped worlds and adds
  the ground.
- Rigid bodies: `quadruped.add_urdf(...)` creates the articulated quadruped at lines 55-62.
  The scene builder replicates it with `scene.replicate(quadruped, self.world_count)` at
  line 80 and adds a ground plane at line 82.
- Deformables/cables/cloth: none.
- Solver: `SolverVBD(self.model, iterations=2)` when `--solver vbd` is selected at
  lines 89-94; otherwise XPBD. VBD mode also tunes joint targets and shape contact material
  before URDF import.
- Contacts: creates `model.contacts()` at line 103. `model.collide()` is called on the
  configured refresh cadence; VBD calls `set_rigid_history_update(refresh_contacts)` before
  stepping at lines 125-133.
- Controls: `self.control = self.model.control()` at line 101. Builder joint targets are
  initialized to the initial quadruped pose at line 74; viewer forces can also be applied.

### basic.example_basic_conveyor

File: `_external/newton/newton/examples/basic/example_basic_conveyor.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 165. VBD-compatible contact
  stiffness is stored in belt, rail, and bag `ShapeConfig`s at lines 184-202, and
  `builder.color()` is called at line 342.
- Rigid bodies: adds a ground plane, static visual island, a kinematic conveyor belt link
  with mesh shape and revolute joint, static rail mesh shapes, and many dynamic bag links
  with box/capsule/sphere shapes and free joints at lines 167-339.
- Deformables/cables/cloth: none.
- Solver: `SolverVBD(self.model, iterations=5, rigid_body_contact_buffer_size=512)` for
  `--solver vbd` at lines 345-347; otherwise XPBD.
- Contacts: uses `self.contacts = self.model.contacts()` at line 354 and calls
  `model.collide()` every substep at line 404. Collision filters keep belt from colliding
  with rails and ground while bags remain collidable.
- Controls: the belt is kinematically driven. A Warp kernel writes the belt joint position
  and velocity each substep, then `eval_fk(..., body_flag_filter=KINEMATIC)` updates the
  belt body before collision detection at lines 382-405.

### cable.example_cable_y_junction

File: `_external/newton/newton/examples/cable/example_cable_y_junction.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 53, with default shape contact
  material set at lines 54-56.
- Rigid bodies: `builder.add_rod_graph(...)` builds a Y-shaped rod graph as capsule rigid
  bodies and cable joints at lines 78-87. One tip body is pinned by zeroing mass and inertia
  at lines 89-97. Optional ground is added at lines 99-100.
- Deformables/cables/cloth: the cable is represented as rod graph rigid bodies/joints, not
  particle cloth.
- Solver: `SolverVBD(self.model, iterations=self.sim_iterations)` at lines 107-110.
- Contacts: creates `model.contacts()` at line 115, then calls `model.collide()` every
  substep at line 149.
- Controls: no authored actuator controls. Viewer forces can pull on bodies; one cable tip
  is pinned through zero mass/inertia.

### cable.example_cable_twist

File: `_external/newton/newton/examples/cable/example_cable_twist.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 138, with contact material at
  lines 140-143.
- Rigid bodies: creates three rods with `builder.add_rod(...)` at lines 167-175. The first
  body of each rod is made kinematic by zeroing mass/inertia at lines 177-183. Ground is
  added at line 193.
- Deformables/cables/cloth: cables are rod rigid bodies/joints built by `add_rod()`.
- Solver: `SolverVBD(self.model, iterations=self.sim_iterations, rigid_avbd_contact_alpha=0.0)`
  at line 202.
- Contacts: creates `model.contacts()` at line 208. Contacts are refreshed on a cadence;
  when refreshed, `model.collide()` is called and VBD history is synchronized with
  `set_rigid_history_update(refresh_contacts)` at lines 243-249.
- Controls: a Warp kernel spins the kinematic first capsules each substep at lines 232-238.

### cable.example_cable_pile

File: `_external/newton/newton/examples/cable/example_cable_pile.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 58, with `builder.rigid_gap = 0.0`
  and contact material at lines 59-66.
- Rigid bodies: adds either a sloped plane or ground plane, then creates layered cable lanes
  with `builder.add_rod(...)` at lines 78-157.
- Deformables/cables/cloth: cables are rod rigid bodies/joints.
- Solver: `SolverVBD(self.model, iterations=self.sim_iterations,
  rigid_body_contact_buffer_size=256, rigid_contact_history=True)` at lines 163-168.
- Contacts: uses `CollisionPipeline(self.model, contact_matching="latest")` when creating
  contacts at lines 173-174, then calls `model.collide()` every substep at line 200.
- Controls: no authored controls; viewer forces are applied each substep.

### cable.example_cable_bundle_hysteresis

File: `_external/newton/newton/examples/cable/example_cable_bundle_hysteresis.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 175. When Dahl friction is
  requested, VBD custom attributes are registered before adding joints at lines 178-180.
- Rigid bodies: creates a bundle of rods via `builder.add_rod(...)` at lines 199-219, then
  adds zero-mass kinematic obstacle capsule bodies at lines 221-259 and a ground plane at
  lines 262-270.
- Deformables/cables/cloth: cables are rod rigid bodies/joints. Optional Dahl cable
  friction parameters are authored on `model.vbd` after finalization at lines 278-281.
- Solver: `SolverVBD(self.model, iterations=self.sim_iterations)` at lines 283-286.
- Contacts: creates `model.contacts()` at line 293. Contacts refresh on a cadence, with
  `model.collide()` and `set_rigid_history_update(refresh_contacts)` at lines 367-373.
- Controls: a Warp kernel moves the kinematic obstacles through a triangle-wave loading,
  hold, and release cycle at lines 338-365.

### cable.example_cable_cross_slide_table

File: `_external/newton/newton/examples/cable/example_cable_cross_slide_table.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 623. Helper functions in the
  same file add kinematic and passive guided pulleys with primitive cylinder shapes and
  revolute joints.
- Rigid bodies: builds a fixed base, prismatic X/Y slide bodies, passive and kinematic
  pulley bodies, visual markers, a cable rod, and ball-joint cable anchors at lines 633-920.
- Deformables/cables/cloth: the cable loop is a rod created by `builder.add_rod(...)` at
  lines 865-876. Its wrapped pose is assigned after finalization at lines 976-990.
- Solver: `SolverVBD(self.model, iterations=sim_iterations,
  rigid_body_contact_buffer_size=256, rigid_contact_hard=True, rigid_contact_history=True)`
  at lines 936-942.
- Contacts: creates contacts with `CollisionPipeline(self.model, broad_phase="explicit",
  contact_matching="latest")` at lines 947-948 and calls `model.collide()` each substep at
  line 1033. Same-cable body collision pairs are filtered at lines 346-352 and 878.
- Controls: input pulleys are kinematic. The `drive_input_pulleys` kernel writes their body
  transforms every substep before collision and VBD stepping at lines 1018-1034.

### cloth.example_cloth_bending

File: `_external/newton/newton/examples/cloth/example_cloth_bending.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 48; default shape contact material
  is set at lines 50-55.
- Rigid bodies: only a ground plane at line 72.
- Deformables/cables/cloth: loads a USD cloth mesh and adds it with `builder.add_cloth_mesh`
  at lines 56-69. Calls `builder.color(include_bending=True)` at line 71.
- Solver: `SolverVBD(self.model, self.iterations, particle_enable_self_contact=True,
  particle_self_contact_radius=0.2, particle_self_contact_margin=0.35)` at lines 79-85.
- Contacts: creates an explicit `CollisionPipeline(..., broad_phase="nxn",
  soft_contact_margin=0.1)` at lines 87-92 and uses `collision_pipeline.collide()` every
  substep at line 118.
- Controls: no authored controls; viewer forces can be applied.

### cloth.example_cloth_hanging

File: `_external/newton/newton/examples/cloth/example_cloth_hanging.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at lines 47-51, with solver-specific
  cloth parameters selected before construction.
- Rigid bodies: only a ground plane at lines 53-59.
- Deformables/cables/cloth: creates a hanging cloth grid via `builder.add_cloth_grid(...)`
  for VBD/XPBD/semi-implicit at lines 61-109. In VBD mode it uses `tri_ke`, `tri_ka`, and
  `tri_kd`, then calls `builder.color(include_bending=True)` at lines 99-112.
- Solver: VBD default path uses `SolverVBD(model=self.model, iterations=self.iterations,
  particle_enable_self_contact=True, particle_self_contact_radius=0.02,
  particle_self_contact_margin=0.03)` at lines 131-138.
- Contacts: uses `model.contacts()` at line 144 and `model.collide()` every substep at
  line 165.
- Controls: no authored controls beyond viewer forces.

### cloth.example_cloth_twist

File: `_external/newton/newton/examples/cloth/example_cloth_twist.py`

- ModelBuilder: `scene = newton.ModelBuilder(gravity=0)` at line 147.
- Rigid bodies: none.
- Deformables/cables/cloth: adds a USD square cloth mesh with `scene.add_cloth_mesh(...)`
  at lines 148-161. Edge particles on both sides are made inactive/fixed by clearing
  `ParticleFlags.ACTIVE` at lines 168-178. Calls `scene.color()` at line 162.
- Solver: `SolverVBD(self.model, self.iterations, particle_enable_self_contact=True,
  particle_self_contact_radius=0.002, particle_self_contact_margin=0.0035)` at lines 180-186.
- Contacts: creates `model.contacts()` at line 191. Before substeps it calls `model.collide()`
  and `solver.rebuild_bvh(self.state_0)` at lines 232-234, then VBD steps with contacts.
- Controls: a custom `apply_rotation` kernel moves the fixed side particles to twist the
  cloth before each solver step at lines 241-260.

### cloth.example_cloth_poker_cards

File: `_external/newton/newton/examples/cloth/example_cloth_poker_cards.py`

- ModelBuilder: `builder = newton.ModelBuilder(gravity=-9.8)` at line 64.
- Rigid bodies: adds a zero-density cube as a static support and a zero-density sphere that
  is manually animated as a kinematic striker at lines 66-108. A ground plane is added at
  lines 173-178.
- Deformables/cables/cloth: adds 52 card-like cloth grids with high stretch/bending
  stiffness at lines 135-171 and calls `builder.color(include_bending=True)` at line 181.
- Solver: `SolverVBD(..., iterations=self.iterations, particle_enable_self_contact=True,
  particle_self_contact_radius=0.001, particle_self_contact_margin=0.0015,
  particle_topological_contact_filter_threshold=2,
  particle_rest_shape_contact_exclusion_radius=0.0)` at lines 191-200.
- Contacts: uses `CollisionPipeline(..., broad_phase="nxn", soft_contact_margin=0.005)`
  at lines 210-216 and calls `collision_pipeline.collide()` every substep at line 253.
- Controls: directly rewrites the sphere body transform in `state_0.body_q` each substep to
  move it into the card pile at lines 243-250.

### cloth.example_cloth_rollers

File: `_external/newton/newton/examples/cloth/example_cloth_rollers.py`

- ModelBuilder: `builder = newton.ModelBuilder(gravity=0.0)` at line 225.
- Rigid bodies: only a ground plane is added at line 297, but external collision contacts
  are disabled later.
- Deformables/cables/cloth: adds a rolled cloth mesh plus two cylinder surface meshes using
  `builder.add_cloth_mesh(...)` at lines 247-294. The cloth edge and all cylinder vertices
  are made inactive/kinematic through `particle_flags` edits at lines 324-350. Calls
  `builder.color(include_bending=False)` at line 300.
- Solver: `SolverVBD(..., particle_enable_self_contact=True,
  particle_self_contact_radius=0.3, particle_self_contact_margin=0.6,
  particle_vertex_contact_buffer_size=48, particle_edge_contact_buffer_size=64,
  particle_collision_detection_interval=5,
  particle_topological_contact_filter_threshold=2)` at lines 360-371.
- Contacts: sets `self.contacts = None` at line 389 and does not call `model.collide()`;
  it rebuilds the VBD particle BVH before stepping at line 412.
- Controls: custom kernels rotate the fixed cloth edge and cylinder particles to unroll the
  cloth at lines 417-477.

### cloth.example_cloth_franka

File: `_external/newton/newton/examples/cloth/example_cloth_franka.py`

- ModelBuilder: `self.scene = ModelBuilder(gravity=-981.0)` at line 111. A separate
  `franka = ModelBuilder()` is populated by `create_articulation()` and added as a world at
  lines 115-120.
- Rigid bodies: robot rigid bodies come from Franka URDF import in `create_articulation()`
  at lines 363-377. The scene also adds a static table box at lines 124-139 and a ground
  plane at line 169.
- Deformables/cables/cloth: loads a shirt mesh and adds it with `self.scene.add_cloth_mesh`
  at lines 141-165, then calls `self.scene.color()` at line 167.
- Solver: robot uses `SolverFeatherstone` at line 239. Cloth uses
  `SolverVBD(..., integrate_with_external_rigid_solver=True, particle_enable_self_contact=True,
  particle_collision_detection_interval=-1, ...)` at lines 242-257.
- Contacts: uses `CollisionPipeline(self.model, soft_contact_margin=self.cloth_body_contact_margin)`
  at lines 229-234. Each substep runs the external robot solver first, then
  `collision_pipeline.collide()` and `cloth_solver.step(...)` at lines 545-580.
- Controls: computes end-effector velocity targets from key poses and a Jacobian
  pseudoinverse, assigns `state_0.joint_qd` from `target_joint_qd`, and lets Featherstone
  advance the robot before cloth contact at lines 480-566.

### contacts.example_contacts_rj45_plug

File: `_external/newton/newton/examples/contacts/example_contacts_rj45_plug.py`

- ModelBuilder: `builder = newton.ModelBuilder(gravity=-9.81)` at line 241. It registers
  VBD custom attributes at line 242 so joints can be marked soft via `vbd:joint_is_hard`.
- Rigid bodies: adds static socket mesh, dynamic plug and latch mesh bodies, a soft D6 joint
  from world to plug, a soft revolute latch joint, and a rod cable at lines 245-332. The
  first cable bodies and far cable end are made kinematic by zeroing mass/inertia at
  lines 340-345.
- Deformables/cables/cloth: cable is a rod built with `builder.add_rod(...)`, not a
  particle cloth/soft mesh.
- Solver: `SolverVBD(self.model, iterations=12, rigid_contact_hard=False,
  rigid_body_contact_buffer_size=256)` at lines 387-392.
- Contacts: uses `self.contacts = self.model.contacts()` at line 383 and calls
  `model.collide()` every substep at line 449. It filters cable segments that overlap the
  connector at rest at lines 334-338.
- Controls: applies a gizmo/picking spring force with `_apply_gizmo_force`, syncs the
  kinematic cable anchor bodies to the plug before collision, and aligns cable capsule
  orientations after each solve at lines 410-467.

### multiphysics.example_rigid_soft_contact

File: `_external/newton/newton/examples/multiphysics/example_rigid_soft_contact.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 80. VBD-specific contact
  material is selected at lines 69-78 and `builder.color()` is called for VBD at line 115.
- Rigid bodies: adds a ground plane and one rigid sphere body/shape at lines 83 and 101-112.
- Deformables/cables/cloth: adds a volumetric soft grid with `builder.add_soft_grid(...)`
  at lines 85-99.
- Solver: `SolverVBD(model=self.model, iterations=10, particle_enable_self_contact=False,
  particle_enable_tile_solve=False, rigid_contact_hard=False,
  rigid_body_particle_contact_buffer_size=512)` at lines 136-144.
- Contacts: uses `model.contacts()` at line 149 and `model.collide()` every substep at
  line 173.
- Controls: no authored controls; the sphere falls/interacts under gravity and viewer
  forces.

### multiphysics.example_softbody_dropping_to_cloth

File: `_external/newton/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 36.
- Rigid bodies: only a ground plane at line 37.
- Deformables/cables/cloth: adds one tetrahedral soft grid above one cloth grid at
  lines 39-74, then calls `builder.color()` at line 77.
- Solver: `SolverVBD(model=self.model, iterations=self.iterations,
  particle_enable_self_contact=True, particle_self_contact_radius=0.01,
  particle_self_contact_margin=0.02, particle_enable_tile_solve=True)` at lines 86-93.
- Contacts: uses `model.contacts()` at line 99 and calls `model.collide()` every substep at
  line 120.
- Controls: no authored controls beyond viewer forces.

### multiphysics.example_softbody_gift

File: `_external/newton/newton/examples/multiphysics/example_softbody_gift.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 159.
- Rigid bodies: only a ground plane at line 160.
- Deformables/cables/cloth: adds four soft tet blocks with `builder.add_soft_mesh(...)` at
  lines 162-175 and two cloth straps with `builder.add_cloth_mesh(...)` at lines 177-207.
  Calls `builder.color()` at line 210.
- Solver: `SolverVBD(model=self.model, iterations=self.iterations,
  particle_enable_self_contact=True, particle_self_contact_radius=0.04,
  particle_self_contact_margin=0.06,
  particle_topological_contact_filter_threshold=1, particle_enable_tile_solve=False)` at
  lines 219-227.
- Contacts: uses `model.contacts()` at line 233 and calls `model.collide()` every substep at
  line 259.
- Controls: no authored controls beyond viewer forces.

### softbody.example_softbody_hanging

File: `_external/newton/newton/examples/softbody/example_softbody_hanging.py`

- ModelBuilder: `builder = newton.ModelBuilder()` at line 35.
- Rigid bodies: only a ground plane at line 36.
- Deformables/cables/cloth: creates four volumetric soft grids with different damping values
  via `builder.add_soft_grid(...)` at lines 44-65, then calls `builder.color()` at line 68.
- Solver: `SolverVBD(model=self.model, iterations=self.iterations,
  particle_enable_self_contact=False, particle_enable_tile_solve=False)` at lines 75-80.
- Contacts: uses `model.contacts()` at line 86 and `model.collide()` every substep at
  line 107.
- Controls: no authored controls beyond viewer forces.

### softbody.example_softbody_franka

File: `_external/newton/newton/examples/softbody/example_softbody_franka.py`

- ModelBuilder: `self.scene = ModelBuilder(gravity=-9.81)` at line 68. A separate
  `franka = ModelBuilder()` loads the robot and is added as a world at lines 72-75.
- Rigid bodies: Franka rigid bodies come from URDF import in `create_articulation()` at
  lines 236-247. The scene adds a static table box at lines 77-88 and a ground plane at
  line 112.
- Deformables/cables/cloth: loads a tetrahedral duck mesh from USD and adds it with
  `self.scene.add_soft_mesh(...)` at lines 90-109, then calls `self.scene.color()` at
  line 111.
- Solver: robot uses `SolverFeatherstone` at line 141. Soft body uses
  `SolverVBD(..., integrate_with_external_rigid_solver=True,
  particle_enable_self_contact=False, particle_collision_detection_interval=-1, ...)` at
  lines 146-157.
- Contacts: creates a `CollisionPipeline(..., soft_contact_margin=self.soft_body_contact_margin)`
  at lines 131-136. Each substep runs robot kinematics with particles disabled, then
  `collision_pipeline.collide()` and `soft_solver.step(...)` at lines 337-363.
- Controls: sets up an analytic GPU IK solver at lines 172-226. Each frame updates IK
  targets, computes joint velocities, writes `state_0.joint_qd`, and advances the robot
  with Featherstone before VBD soft-body simulation at lines 284-361.

## Patterns Useful For RoboLab VBD Work

- VBD examples generally call `builder.color()` before `finalize()`; particle/cloth
  examples sometimes use `include_bending=True` when bending constraints matter.
- Rigid-only VBD examples use `model.contacts()` plus `model.collide()` and may tune
  shape material `ke`, `kd`, and friction higher than XPBD.
- Particle/cloth/softbody scenes set `model.soft_contact_ke`, `soft_contact_kd`, and
  `soft_contact_mu` after finalization.
- Explicit `CollisionPipeline` is used when examples need custom broad phase/contact
  matching or soft-contact margins (`cloth_bending`, `cloth_poker_cards`, `cloth_franka`,
  `softbody_franka`, `cable_pile`, `cable_cross_slide_table`).
- If contacts are not refreshed every substep, the rigid VBD examples call
  `solver.set_rigid_history_update(refresh_contacts)`.
- Robot/deformable examples use `integrate_with_external_rigid_solver=True` and split the
  frame into a Featherstone robot step followed by VBD particle/deformable contact solve.

# All Newton Example Solver Survey

Repository inspected: `_external/newton`

Scope: runnable Python examples under `_external/newton/newton/examples`, discovered the
same way the example runner does, by scanning `example_*.py` files in the category
subdirectories. This section treats `newton.solvers.*` classes as physics solvers.
`newton.ik.IKSolver` is listed separately as an auxiliary kinematic optimizer when it is
used to generate robot joint targets, because it does not integrate physics state by
itself.

Reference solver capabilities from `_external/newton/newton/solvers.py`:

- `SolverMuJoCo`: rigid bodies and generalized-coordinate articulations; can use MuJoCo
  contacts internally, or consume Newton contacts when `use_mujoco_contacts=False`.
- `SolverFeatherstone`: rigid bodies and generalized-coordinate articulations; used in
  examples as a robot/rigid kinematic or dynamics integrator before a deformable solve.
- `SolverKamino`: rigid bodies and articulations in maximal coordinates; has optional
  internal collision detector and FK solver.
- `SolverVBD`: implicit solver for rigid bodies, particles, cloth, soft bodies, and a
  limited set of joints; examples often use it for cloth, soft bodies, cable rods, and
  mixed native scenes.
- `SolverXPBD`: implicit native solver for rigid bodies, maximal-coordinate
  articulations, particles, cloth, and experimental soft bodies.
- `SolverSemiImplicit`: semi-implicit native solver for rigid bodies, maximal-coordinate
  articulations, particles, cloth without self-collision, and soft bodies.
- `SolverStyle3D`: implicit cloth/particle solver; no rigid-body or joint solve.
- `SolverImplicitMPM`: implicit MPM particle/material solver; no rigid-body solve.

## Per-Example Solver Inventory

| Example | Physics solver(s) | Integration notes |
| --- | --- | --- |
| `basic.basic_conveyor` | `SolverXPBD` by default, or `SolverVBD` with `--solver vbd` | One native solver owns the rigid bags, rails, belt, and contacts. The belt is kinematically driven with FK before `model.collide()`, then the selected solver steps. |
| `basic.basic_heightfield` | `SolverXPBD` by default, or `SolverMuJoCo` with `--solver mujoco` | XPBD path uses Newton `model.collide()`. MuJoCo path lets MuJoCo own contacts and then calls `solver.update_contacts()` for visualization/reporting. |
| `basic.basic_joints` | `SolverXPBD` by default, or `SolverVBD` with `--solver vbd` | One native solver handles the rigid joint examples. VBD uses the finalized body poses as structural rest pose; contacts come from `model.collide()`. |
| `basic.basic_pendulum` | `SolverXPBD` | Single native rigid/articulation solver with Newton contacts. |
| `basic.basic_plotting` | `SolverMuJoCo` | Single MuJoCo articulation solver for a humanoid MJCF scene. The loop also calls Newton `model.collide()` and passes that contact buffer, but the solver is constructed with MuJoCo defaults rather than an explicit Newton-contact configuration. |
| `basic.basic_shapes` | `SolverXPBD` by default, or `SolverVBD` with `--solver vbd` | Single native solver for primitive/mesh rigid shapes. Contacts come from `model.collide()`. |
| `basic.basic_urdf` | `SolverXPBD` by default, or `SolverVBD` with `--solver vbd` | Single native solver for replicated quadruped URDF worlds. Contact refresh can be throttled; VBD synchronizes contact history with `set_rigid_history_update()`. |
| `basic.basic_viewer` | None | Viewer-only visualization of logged shapes and lines; no Newton model integration. |
| `basic.recording` | `SolverMuJoCo` | Single MuJoCo solver for a humanoid recording. It steps without an explicit `Contacts` object and logs states to `ViewerFile`. |
| `basic.replay_viewer` | None | Replay UI loads recorded model/state frames; no physics solver is stepped. |
| `cable.cable_bundle_hysteresis` | `SolverVBD` | Single VBD solver for cable rods plus kinematic obstacle bodies. Optional Dahl friction is enabled through VBD custom attributes; contacts are refreshed on a cadence with VBD history updates. |
| `cable.cable_cross_slide_table` | `SolverVBD` | Single VBD solver for guided pulleys, table/slide rigid bodies, and rod cable. Uses explicit `CollisionPipeline` contact matching and direct kinematic body driving. |
| `cable.cable_pile` | `SolverVBD` | Single VBD solver for rod cables and ground. Uses `CollisionPipeline(contact_matching="latest")` and rigid contact history. |
| `cable.cable_twist` | `SolverVBD` | Single VBD solver for twisted rod cables. Kinematic rod endpoints are driven before contact refresh and VBD stepping. |
| `cable.cable_y_junction` | `SolverVBD` | Single VBD solver for a rod graph represented as rigid capsule bodies and cable joints. |
| `cloth.cloth_bending` | `SolverVBD` | Single VBD cloth solver. Uses a `CollisionPipeline` and VBD BVH handling for cloth contacts. |
| `cloth.cloth_franka` | `SolverFeatherstone` plus `SolverVBD` | Featherstone advances the Franka/rigid bodies with particles temporarily disabled. A Newton `CollisionPipeline` then builds cloth-body contacts, and VBD steps the cloth with `integrate_with_external_rigid_solver=True`. |
| `cloth.cloth_h1` | `SolverStyle3D`; auxiliary `IKSolver` | IK computes H1 kinematic poses. The example interpolates robot body transforms for collision processing, calls `model.collide()`, then Style3D advances the cloth. There is no separate rigid dynamics solver. |
| `cloth.cloth_hanging` | `SolverVBD` by default; optional `SolverSemiImplicit`, `SolverStyle3D`, or `SolverXPBD` | One selected cloth solver owns the cloth. Style3D uses Style3D-specific cloth construction and attributes; other paths use native cloth grid construction and Newton contacts. |
| `cloth.cloth_poker_cards` | `SolverVBD` | Single VBD solver for cloth cards interacting with rigid cube/sphere bodies. Uses `CollisionPipeline` and VBD self-contact. |
| `cloth.cloth_rollers` | `SolverVBD` | Single VBD solver for cloth interacting with kinematic roller meshes. Roller motion is authored directly, then VBD solves cloth/contact. |
| `cloth.cloth_style3d` | `SolverStyle3D` | Single Style3D cloth solver. Cloth is authored through Style3D helpers; Newton collision data is passed into the solver. |
| `cloth.cloth_twist` | `SolverVBD` | Single VBD solver for cloth twist/self-contact. It rebuilds or refits the VBD BVH before stepping. |
| `contacts.brick_stacking` | `SolverMuJoCo`; auxiliary `IKSolver` | IK generates Franka targets for a brick-stacking task. MuJoCo integrates robot and rigid bricks, but contacts come from a Newton `CollisionPipeline` because `use_mujoco_contacts=False`. |
| `contacts.contacts_rj45_plug` | `SolverVBD` | Single VBD solver for a plug/latch/cable contact scene. Uses VBD custom attributes and Newton contacts. |
| `contacts.nut_bolt_hydro` | `SolverMuJoCo` by default, or `SolverXPBD` with `--solver xpbd` | Both paths use a Newton `CollisionPipeline` with hydroelastic-style geometry/contact setup. MuJoCo is configured with `use_mujoco_contacts=False`, so Newton contacts feed the MuJoCo step. |
| `contacts.nut_bolt_sdf` | `SolverMuJoCo` by default, or `SolverXPBD` with `--solver xpbd` | Same integration pattern as `nut_bolt_hydro`, but with SDF collision geometry: Newton collision pipeline first, then selected solver. |
| `contacts.pyramid` | `SolverXPBD` | Single XPBD rigid contact stack. Uses an explicit Newton `CollisionPipeline` and passes its contacts to XPBD. |
| `diffsim.diffsim_ball` | `SolverSemiImplicit` | Single differentiable semi-implicit particle solver, with a Newton collision pipeline for the ball/ground contact. |
| `diffsim.diffsim_bear` | `SolverSemiImplicit` | Single differentiable semi-implicit soft-body solver for a tetrahedral bear; triangle contacts are disabled in the solver and collision is handled through the configured pipeline. |
| `diffsim.diffsim_cloth` | `SolverSemiImplicit` | Single differentiable semi-implicit cloth solver. |
| `diffsim.diffsim_drone` | Two `SolverSemiImplicit` instances | One solver advances batched rollout worlds for optimization and another advances the displayed drone. They are separate models/solves, not a coupled multiphysics exchange. |
| `diffsim.diffsim_soft_body` | `SolverSemiImplicit` | Single differentiable semi-implicit soft-body solver with a Newton collision pipeline. |
| `diffsim.diffsim_spring_cage` | `SolverSemiImplicit` | Single differentiable semi-implicit particle/spring solver. |
| `ik.ik_cube_stacking` | `SolverMuJoCo`; auxiliary `IKSolver` | IK generates Franka targets. MuJoCo integrates the robot, cubes, and table. Contacts use either MuJoCo contacts when `--use-mujoco-contacts` is enabled or Newton `model.collide()` otherwise. |
| `ik.ik_custom` | Auxiliary `IKSolver` only | Demonstrates custom IK objectives and FK visualization. No physics solver advances dynamics. |
| `ik.ik_franka` | Auxiliary `IKSolver` only | Solves Franka IK and visualizes FK; no physics solver. |
| `ik.ik_h1` | Auxiliary `IKSolver` only | Solves H1 IK and visualizes FK; no physics solver. |
| `kamino.kamino_basic_dr_testmech` | `SolverKamino` | Single Kamino maximal-coordinate solver. Collision detector and FK solver are explicitly disabled. |
| `kamino.kamino_basic_fourbar` | `SolverKamino` | Single Kamino solver with internal collision detector and FK solver enabled. Contacts are updated from Kamino after the step for visualization/reporting. |
| `kamino.kamino_basic_heterogeneous` | `SolverKamino` | Single Kamino solver for heterogeneous replicated assets, with Kamino collision detector and FK solver enabled. |
| `kamino.kamino_robot_anymal_d` | `SolverKamino` | Single Kamino robot solver. With Kamino contacts enabled it passes contacts into Kamino; otherwise it can use an external Newton `CollisionPipeline`. Contacts are updated after stepping. |
| `kamino.kamino_robot_dr_legs` | `SolverKamino` | Same Kamino robot pattern as ANYmal D, with optional internal Kamino contacts or external Newton collision pipeline. |
| `mpm.mpm_anymal` | `SolverMuJoCo` plus `SolverImplicitMPM` | MuJoCo advances ANYmal. Implicit MPM advances sand in the same Newton model, using robot bodies as kinematic colliders via `setup_collider()`. The steps alternate robot substeps and a sand step rather than one monolithic solver. |
| `mpm.mpm_beam_twist` | `SolverImplicitMPM` | Single implicit MPM solver for particle material deformation. The `--solver` option chooses the MPM linear/rheology solve sequence inside `SolverImplicitMPM`, not a different Newton solver class. |
| `mpm.mpm_grain_rendering` | `SolverImplicitMPM` | Single implicit MPM solver for granular particles; rendering-focused. |
| `mpm.mpm_granular` | `SolverImplicitMPM` | Single implicit MPM solver for granular flow with optional static/kinematic colliders. |
| `mpm.mpm_multi_material` | `SolverImplicitMPM` | Single implicit MPM solver for mixed material particles. |
| `mpm.mpm_snow_ball` | `SolverImplicitMPM` | Single implicit MPM solver for snow/plastic material. The `--solver` sequence configures MPM internal solves. |
| `mpm.mpm_twoway_coupling` | `SolverMuJoCo` plus `SolverImplicitMPM` | MuJoCo advances rigid objects. MPM advances sand in a separate sand model while reading rigid colliders from the rigid model. MPM collider impulses are collected and applied back to rigid body velocities before the next rigid step, giving explicit two-way coupling. |
| `mpm.mpm_viscous` | `SolverImplicitMPM` | Single implicit MPM solver for viscous material. Internal MPM solver settings are controlled by command-line options. |
| `multiphysics.rigid_soft_contact` | `SolverXPBD` by default; optional `SolverSemiImplicit` or `SolverVBD` | Single selected native solver owns both the rigid sphere and soft grid. This is not split into a rigid solver plus deformable solver. |
| `multiphysics.softbody_dropping_to_cloth` | `SolverVBD` | Single VBD solver for a soft grid dropping onto cloth. |
| `multiphysics.softbody_gift` | `SolverVBD` | Single VBD solver for soft blocks plus cloth straps. |
| `robot.robot_allegro_hand` | `SolverMuJoCo` | Single MuJoCo solver for Allegro hand articulation. Contacts are Newton-generated because `use_mujoco_contacts=False`. |
| `robot.robot_anymal_c_walk` | `SolverMuJoCo` | Single MuJoCo solver for ANYmal C with policy/control logic. Contacts can be MuJoCo-owned or Newton-generated depending on `--use-mujoco-contacts`. |
| `robot.robot_anymal_d` | `SolverMuJoCo` | Single MuJoCo solver for ANYmal D. Contacts can be MuJoCo-owned or Newton-generated depending on `--use-mujoco-contacts`. |
| `robot.robot_cartpole` | `SolverMuJoCo` | Single MuJoCo solver for cartpole. Commented alternatives show SemiImplicit/Featherstone, but runtime uses MuJoCo. |
| `robot.robot_g1` | `SolverMuJoCo` | Single MuJoCo solver for G1 humanoid. Contacts can be MuJoCo-owned or Newton-generated depending on `--use-mujoco-contacts`. |
| `robot.robot_h1` | `SolverMuJoCo` | Single MuJoCo solver for H1 humanoid. Contacts can be MuJoCo-owned or Newton-generated depending on `--use-mujoco-contacts`. |
| `robot.robot_panda_hydro` | `SolverMuJoCo`; auxiliary `IKSolver` | IK generates Panda targets. MuJoCo integrates robot and rigid objects; hydroelastic/contact data comes from a Newton `CollisionPipeline` because `use_mujoco_contacts=False`. |
| `robot.robot_policy` | `SolverMuJoCo` | Single MuJoCo solver for a policy-controlled robot. Policy output writes controls; MuJoCo integrates the articulation. |
| `robot.robot_ur10` | `SolverMuJoCo` | Single MuJoCo solver for UR10 articulation with contacts disabled. |
| `selection.selection_articulations` | `SolverMuJoCo` | Single MuJoCo solver for selection/filtering demonstrations. The generic selection code has a non-MuJoCo collision path, but this example constructs MuJoCo. |
| `selection.selection_cartpole` | `SolverMuJoCo` | Single MuJoCo solver with contacts disabled for cartpole selection. |
| `selection.selection_materials` | `SolverMuJoCo` | Single MuJoCo solver for material/selection demonstration. |
| `selection.selection_multiple` | `SolverMuJoCo` | Single MuJoCo solver for multiple selected assets. |
| `sensors.sensor_contact` | `SolverMuJoCo` | Single MuJoCo solver. The example calls `update_contacts()` to feed the contact sensor output from MuJoCo-resolved contacts. |
| `sensors.sensor_imu` | `SolverMuJoCo` | Single MuJoCo solver for moving rigid bodies with IMU sensors; contact data is updated from the solver. |
| `sensors.sensor_tiled_camera` | None | Builds a static scene and renders tiled camera outputs from a fixed state; no physics solver step. |
| `softbody.softbody_franka` | `SolverFeatherstone` plus `SolverVBD`; auxiliary `IKSolver` | IK generates Franka targets. Featherstone advances the robot with particles disabled so body velocities are available. A Newton `CollisionPipeline` then computes soft-body contacts, and VBD steps the tetrahedral soft body with `integrate_with_external_rigid_solver=True`. |
| `softbody.softbody_hanging` | `SolverVBD` | Single VBD soft-body solver for hanging volumetric grids. |

## Multi-Solver Integration Patterns

- `SolverFeatherstone` plus `SolverVBD` appears in `cloth.cloth_franka` and
  `softbody.softbody_franka`. The pattern is: solve or author robot joint targets,
  advance the robot/rigid state with Featherstone while temporarily disabling particles,
  restore particle counts/gravity, run a Newton `CollisionPipeline`, then step VBD with
  `integrate_with_external_rigid_solver=True`. VBD owns the deformable solve and uses the
  externally advanced rigid state for contacts/friction.
- `SolverMuJoCo` plus `SolverImplicitMPM` appears in `mpm.mpm_anymal` and
  `mpm.mpm_twoway_coupling`. `mpm_anymal` is mostly one-way from robot to sand: MuJoCo
  advances the robot and MPM treats robot bodies as kinematic colliders. `mpm_twoway_coupling`
  is explicit two-way coupling: MPM reads rigid colliders, collects collider impulses, and a
  kernel applies those impulses back to the rigid-body velocities before the next MuJoCo step.
- `SolverMuJoCo` plus Newton collision is common but is not two physics solvers. Examples
  such as `contacts.brick_stacking`, `robot.robot_panda_hydro`, `robot.robot_allegro_hand`,
  and the nut/bolt examples set `use_mujoco_contacts=False`, run Newton collision/contact
  generation, and pass those contacts into the MuJoCo dynamics step.
- `IKSolver` plus a physics solver is target generation, not coupled dynamics. IK examples
  without a physics solver only solve joint coordinates and visualize FK. Robot manipulation
  examples use IK to write `control.joint_target_q` or state velocities, then a physics
  solver integrates the scene.
- `SolverKamino` examples are single-solver scenes, but Kamino can choose whether contacts
  come from its internal detector or from an external Newton `CollisionPipeline`.
- Solver choice exposed as `--solver` is usually mutually exclusive, not simultaneous:
  examples select exactly one of XPBD, VBD, SemiImplicit, Style3D, MuJoCo, or an internal
  MPM linear/rheology solve mode for the whole scene.

## Final Summary

### Solver types used for robots

- Main robot dynamics solver: `SolverMuJoCo`. This is the dominant choice for articulated
  robots in `robot.*`, manipulation examples, sensors tied to articulated bodies, and IK
  demos that continue into dynamics. It is used because it supports generalized-coordinate
  articulations, joint targets, armature, actuator-like settings, and robot contact workflows.
- Robot/deformable coupling solver: `SolverFeatherstone` for the robot plus `SolverVBD`
  for cloth or soft bodies. This appears when the deformable must be handled by VBD, while
  the robot still needs a generalized-coordinate rigid/articulation update and useful body
  velocities for friction/contact.
- Alternative robot solver: `SolverKamino` in the `kamino.*` examples. These examples are
  specifically about Kamino's maximal-coordinate robot/articulation path and optional
  internal collision detector.
- Native rigid-articulation examples sometimes use `SolverXPBD` or `SolverVBD` for simple
  joints or URDF replication, but that is not the main pattern for full robot examples.
- `IKSolver` is frequently paired with robot examples, but only to compute target joint
  configurations; it is not the physics integrator.

### Solver types used for rigid objects

- `SolverMuJoCo` is used for rigid objects when they share a scene with robots or when the
  example is focused on MuJoCo contacts/integration. Rigid contacts may be MuJoCo-internal
  or Newton-generated and passed in.
- `SolverXPBD` is the common default native rigid/contact solver for simple rigid examples:
  pendulum, shapes, conveyor, heightfield, pyramid, nut/bolt optional paths, and mixed
  rigid-soft demos.
- `SolverVBD` is used for rigid objects when they are part of a VBD-centric scene: cable
  rods, cloth/soft-body contact bodies, or optional VBD versions of basic rigid scenes.
- `SolverSemiImplicit` appears mostly in differentiable examples and a few optional
  multiphysics/cloth paths.
- `SolverKamino` handles rigid objects and articulations only inside Kamino-specific
  examples.
- Static or kinematic rigid colliders in MPM-only examples are not integrated by a rigid
  solver; they serve as collision geometry for `SolverImplicitMPM`.

### Solver types used for deformable objects

- Cloth: `SolverVBD` is the dominant cloth solver for self-contact, cloth-rigid contact,
  and richer cloth scenes. `SolverStyle3D` is used for Style3D-specific cloth workflows and
  H1 clothing. `SolverXPBD` and `SolverSemiImplicit` appear as simpler optional cloth paths,
  especially in `cloth_hanging` and differentiable cloth examples.
- Soft bodies: `SolverVBD` is the main non-differentiable soft-body solver, including
  soft grids, tet meshes, and mixed softbody/cloth scenes. `SolverSemiImplicit` is used in
  differentiable soft-body examples. `SolverXPBD` appears as an experimental/native option
  in mixed rigid-soft contact.
- MPM/granular/snow/viscous materials: `SolverImplicitMPM` owns these scenes. When robots
  or rigid objects are present, coupling is explicit: MuJoCo integrates rigid/articulated
  bodies and MPM reads collider state, optionally feeding impulses back.
- Cable examples in this checkout are modeled as rigid rod bodies and cable joints, not as
  particle deformables. They consistently use `SolverVBD`, which supports the required
  cable-joint path and contact history features.
