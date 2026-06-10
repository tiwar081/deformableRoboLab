# SolverVBD Example Inventory

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
