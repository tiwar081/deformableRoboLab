# Newton Adds Contact-Rich Manipulation and Locomotion Capabilities for Industrial Robotics

**Source:** NVIDIA Technical Blog — <https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/>
**Published:** March 16, 2026 (modified April 2, 2026)
**Authors:** Philipp Reist, Miguel Zamora Mora, JC Chang, Rishabh Chadha, Mohammad Mohajerani
**Category:** Robotics · Intermediate Technical · Tutorial · GTC 2026

> *This is a paraphrased reference summary with verbatim code snippets, prepared for context/reference use. Prose has been reworded; visual assets are described rather than embedded.*

---

## Overview

Physics is the backbone of robotic simulation — it lets you model motion and interaction realistically. Locomotion and manipulation in particular demand that a simulator cope with hard dynamics: contact forces, deformable objects, and so on. Most engines force a trade-off between speed and realism. Newton, a GPU-accelerated open-source simulator, aims to deliver both at once.

Newton 1.0 GA was announced at NVIDIA GTC 2026 as a production-ready foundation for dexterous manipulation and locomotion. It's an extensible physics engine built on top of [NVIDIA Warp](https://developer.nvidia.com/warp-python) and [OpenUSD](https://www.nvidia.com/en-us/glossary/openusd/), and it lets robots learn complex tasks with more precision, speed, and extensibility — usable alongside frameworks like NVIDIA Isaac Lab and NVIDIA Isaac Sim.

Newton is modular: it pulls together multiple solvers and simulation components under one unified architecture. Instead of being locked to a single scene format, it supports a broad runtime data model covering common robotics descriptions — MJCF, URDF, and OpenUSD — which makes it easier to wire in existing robot assets and workflows. Teams can mix and match collision detection, contact models, sensors, control, and solver backends (rigid-body, deformable, and custom solvers) while keeping a consistent simulation stack for robot learning and development.

### 🖼️ Figure 1 — Newton Architecture Diagram

*Visual description:* An architecture diagram showing Newton sitting on top of **Warp** and **OpenUSD** as its foundational layers. From there, integration points branch out to **MuJoCo Warp**, **Kamino**, **Isaac Sim**, and **Isaac Lab**. The diagram emphasizes Newton as a modular physics-simulation framework that unifies multiple solvers and components across robotics and physics workloads.

---

## Release Highlights

1. **Stable API** — A stable, unified interface across modeling, solving, controlling, and sensing within the simulation.

2. **Versatile rigid-body solvers** — Newton ships with several rigid-body solvers. The two with the most advanced and complementary capabilities are:
   - **[Kamino](http://disneyresearch.github.io/kamino)** (Disney Research): handles complex mechanisms — robotic hands, legged systems with closed-loop linkages, passive actuation. It frees mechanical designers from worrying about whether a design is even simulatable, and opens the door to scalable reinforcement learning.
   - **[MuJoCo 3.5 (MJWarp)](https://github.com/google-deepmind/mujoco/discussions/3094)** (Google DeepMind): builds on the accuracy/stability the robotics community already trusts in MuJoCo, now extended with GPU-scale throughput for thousands of parallel training environments. New optimizations let MuJoCo Warp speed up [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) by **252×** for locomotion and **475×** for manipulation tasks on the [NVIDIA RTX PRO 6000 Blackwell Series](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/).

3. **Rich deformable simulation** — The Vertex Block Descent (VBD) solver handles linear deformables (cables), thin deformables (cloth), and volumetric deformables (rubber parts), covering common industrial materials. The Implicit Material Point Method (iMPM) handles particle simulation (granular material), relevant to rough-terrain locomotion. VBD and MPM can be explicitly coupled with MuJoCo Warp to support deformable manipulation and locomotion with robots.

4. **Collision library** — A flexible, fast collision-detection pipeline that lets you pick the right broadphase/narrowphase approach for the scene's complexity. It's reusable and can accelerate custom solver development. Notable contact-generation/modeling features:
   - **Signed distance field (SDF)–based collision**: captures complex geometry straight from CAD-exported meshes, with no need for mesh approximation. Critical for tight-tolerance tasks like connector insertion or in-hand manipulation.
   - **Hydroelastic contacts**: inspired by the [Drake contact model](https://medium.com/toyotaresearch/rethinking-contact-simulation-for-robot-manipulation-434a56b5ec88), these use a continuous pressure distribution across finite-area contact patches instead of discrete contact points — giving higher fidelity and more robust interaction for tactile sensing and manipulation policies, and ultimately better sim-to-real transfer.

   ### 🎥 Video 1 — Hydroelastic Tactile Contact
   *Visual description:* A GPU-accelerated hydroelastic contact demo, showing Newton generating and scaling high-quality, realistic dexterous tactile data. (`Tactile.mp4`)

5. **OpenUSD and Isaac integration** — With OpenUSD as a shared data layer, Newton integrates natively with [Isaac Sim 6.0](https://github.com/isaac-sim/IsaacSim) and [Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab) (early access), enabling faster workflows from robot description → trained policy → evaluation, across RL and imitation-learning pipelines.

6. **Tiled camera sensor** — A Warp-based tiled camera sensor for high-throughput simplified rendering, with channels for RGB, depth, albedo, surface normals, and instance segmentation. Built to scale vision-based RL policies and run end-to-end perceptive training on the [NVIDIA DGX platform](https://www.nvidia.com/en-us/data-center/dgx-platform/). The rendering backend is ray-tracing-based and supports multiple scene representations, including triangle meshes and Gaussian splats.

   ### 🎥 Video 2 — Tiled Camera Sensor
   *Visual description:* The tiled camera sensor producing batched visual observations across many parallel environments for perceptive RL training. (`Warp.mp4`)

### Governance & Partners

Newton is a [Linux Foundation project](https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning) founded by NVIDIA, Google DeepMind, and Disney Research. **Lightwheel** (simulation infrastructure for physical AI) is contributing to solver calibration, the SimReady standard, and next-gen physically grounded SimReady assets. **Toyota Research Institute (TRI)** — developer of the [Drake physics engine](https://drake.mit.edu/) — is partnering to advance solver development and contact modeling.

---

## Simulating Complex Mechanisms with Kamino

The Kamino solver handles complex, intricate closed-chain mechanisms — e.g. robots whose kinematics include parallel-linkage mechanisms. That makes it possible to simulate things like multi-link walking robots, where each leg's kinematics can contain several closed loops.

A concrete example is **Dr. Legs**, a closed-chain robotic leg mechanism in the [Newton asset repository](https://github.com/newton-physics/newton-assets/tree/main/disneyresearch/dr_legs), which shows how Kamino handles articulated structures with multiple closed loops.

### 🎥 Video 3 — Dr. Legs
*Visual description:* The Dr. Legs closed-chain robotic leg mechanism simulated with the Kamino solver. (`kamino-dr-legs.mp4`)

**Workflow pattern:** Newton workflows follow a consistent shape — build or import a model, initialize state, apply controls (joint targets or forces), and step a solver (e.g. Kamino) to advance the physics, with results shown in the viewer.

```python
import newton

# Import the articulation model from USD
builder = newton.ModelBuilder()

# Register attributes to be parsed specific to Kamino
newton.solvers.SolverKamino.register_custom_attributes(builder)

# Import USD asset
builder.add_usd("robot.usd")

# Finalize the model (upload to GPU)
model = builder.finalize()

# Create Kamino solver
solver = newton.solvers.SolverKamino(model)

# Create state and control objects
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

# Simulation loop
for i in range(num_steps):
    state_0.clear_forces()
    model.collide(state_0, contacts)

    # Forward dynamics
    solver.step(state_0, state_1, control, contacts, sim_dt)

    # Swap states
    state_0, state_1 = state_1, state_0
```

---

## Manipulation Workflow with Newton and Isaac Lab

### 🖼️ Figure 2 — Franka Cube-Picking in Isaac Lab
*Visual description:* Multiple parallel simulation environments, each showing a Franka robotic arm picking up a cube — illustrating GPU-scale parallel RL training.

NVIDIA Isaac Lab is an open-source framework for robot learning. You can define simulation environments, set up RL and imitation-learning pipelines, and train policies at GPU scale. You define a scene (robots, objects, terrain), an MDP (observations, actions, rewards, terminations), and a simulation backend. Newton plugs into Isaac Lab as a new physics and camera-sensor backend, extending its capabilities.

A key point: in an Isaac Lab RL workflow, everything *above* the physics layer — task definition, PPO training loop, observation and reward functions — stays identical. Only the simulation backend changes. So you can author an environment once and validate it across different physics engines, building confidence in policy robustness before real-world deployment.

Below is the three-layer physics configuration Isaac Lab uses for the Franka cube-manipulation environment. Once the scene is set up, you configure the physics settings:

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

# Configure Newton MJWarp simulation for the Franka Cube Env
FrankaCubeEnvCfg.solver_cfg = MJWarpSolverCfg(
    solver="newton",
    integrator="implicitfast",
    njmax=2000,
    nconmax=1000,
    impratio=1000.0,
    cone="elliptic",
    update_data_interval=2,
    iterations=20,
    ls_iterations=100,
    ls_parallel=True,
)

FrankaCubeEnvCfg.newton_cfg = NewtonCfg(
    solver_cfg=FrankaCubeEnvCfg.solver_cfg,
    num_substeps=2,
    debug_mode=False,
)

FrankaCubeEnvCfg.sim = SimulationCfg(
    dt=1 / 120,
    render_interval=FrankaCubeEnvCfg.decimation,
    physics=FrankaCubeEnvCfg.newton_cfg,
)
```

Everything else — applying actions, getting rewards, resetting the environment — stays the same.

---

## How Newton Is Being Used in Industrial Applications

Two examples show Newton's capabilities coming together in production robotics workflows: one focused on rigid-body precision assembly, the other on dexterous manipulation of deformable materials.

### GPU Rack Assembly Automation

[Skild AI](https://www.skild.ai/) is training RL policies for GPU rack assembly for industrial end-users — one of the most demanding contact-rich tasks in electronics manufacturing. Connector insertion, board placement, and fastening all need stable collision and contact, reliable force feedback, and full-fidelity geometric representation that most simulators can't deliver at training scale.

Skild uses Isaac Lab with the Newton backend for their electronics-assembly automation tasks. In their workflows, SDF-based collision detection and hydroelastic contact modeling are used to bypass MuJoCo Warp's native collision/contact pipeline, achieving higher contact fidelity. Shapes are configured with precomputed SDFs built from the original CAD geometry, so Newton can operate on non-convex tri-mesh models that accurately represent assembly components.

SDF collision suits rigid, non-compliant interactions with complex geometry, allowing precise contact queries against connectors, boards, and other tightly-toleranced parts.

#### 🎥 Video 4 — GPU Rack Assembly
*Visual description:* Robotic GPU rack assembly tasks — connector insertion and component placement — used for RL policy training. (`skild-busbar.mp4`)

For richer contact dynamics, hydroelastic modeling adds compliance that yields distributed-pressure contacts instead of point-contact approximations. The larger contact areas capture frictional behavior, including torsional friction effects that arise during complex manipulation sequences. Together, the SDF geometry representation and the hydroelastic contact model give the fidelity needed to train policies that transfer reliably to real industrial assembly systems.

The following snippet shows how SDF collisions and hydroelastic contacts are configured:

```python
import newton
from newton.geometry import HydroelasticSDF

# --- 1. Shape configuration: enable hydroelastic contact ---
shape_cfg = newton.ModelBuilder.ShapeConfig(
    mu=1.0,             # friction coefficient
    gap=0.01,           # contact detection margin [m]
    ke=5.0e4,           # elastic contact stiffness (MJWarp fallback)
    kd=5.0e2,           # contact damping
    kh=1.0e11,          # hydroelastic stiffness [Pa/m] — controls
                        # pressure vs. penetration across the contact patch
)

# --- 2. Build SDF on each collision mesh ---
# Precompute a sparse signed-distance field so Newton can find
# sub-voxel contact surfaces via marching cubes at runtime.
for mesh in assembly_meshes:
    mesh.build_sdf(
        max_resolution=128,                # SDF grid resolution
        narrow_band_range=(-0.01, 0.01),   # ±10 mm band around surface
        margin=shape_cfg.gap,
    )

# --- 3. Mark shapes as hydroelastic ---
# When both shapes in a colliding pair carry this flag, Newton
# routes them through the SDF-hydroelastic pipeline instead of
# MJWarp's native point-contact solver.
for shape_idx in range(builder.shape_count):
    builder.shape_flags[shape_idx] |= newton.ShapeFlags.HYDROELASTIC

# --- 4. Create the collision pipeline with hydroelastic config ---
collision_pipeline = newton.CollisionPipeline(
    model,
    reduce_contacts=True,       # contact-reduction for stable solving
    broad_phase="explicit",     # precomputed shape pairs (few shapes)
    sdf_hydroelastic_config=HydroelasticSDF.Config(
        output_contact_surface=False,   # skip surface mesh export
    ),
)

# --- 5. Simulation loop (unchanged from standard Newton) ---
# The solver receives distributed contact patches transparently.
collision_pipeline.collide(state, contacts)
solver.step(state_0, state_1, control, contacts, dt)
```

---

## Cable Manipulation for Refrigerator Assembly

**Samsung** will use Newton for physically-grounded synthetic data generation (SDG) to train their vision-language-action (VLA) models.

**Lightwheel** is applying Newton to generate SimReady assets, tuned and verified against real-world measurements. This enables complex industrial assembly tasks — including cable manipulation in Samsung manufacturing workflows. Cables are among the hardest objects to simulate reliably: they exhibit complex 1D deformable behavior, self-collision, and force-dependent shape changes that canonical solvers can't capture accurately.

The Samsung/Lightwheel work shows how Newton's deformable simulation stack — spanning cables through volumetric solids — enables SDG and policy training across the full range of materials found on real electronics assembly lines.

### 🎥 Video 5 — Cable Insertion (Refrigerator Assembly)
*Visual description:* An RB-Y1 robot performing a cable-insertion task for refrigerator assembly, simulated with two-way coupled MuJoCo Warp and a VBD cable solver. (`waterhose.mp4`)

Newton's VBD solver simulates linear deformables such as cables. Two-way coupling with rigid-body solvers like MuJoCo Warp lets robot motion physically interact with cable deformation during simulation. Combined with Newton's stable collision and high-fidelity contact modeling, this enables tasks like inserting a refrigerator water-hose connector into its housing. The snippet below shows how VBD and MuJoCo Warp are coupled:

```python
import warp as wp
import newton
from newton.solvers import SolverMuJoCo, SolverVBD

# --- Universe A: MuJoCo rigid-body robot ---
robot_model = robot_builder.finalize()

mj_solver = SolverMuJoCo(
    robot_model,
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    iterations=20,
    ls_iterations=10,
    ls_parallel=True,
    impratio=1000.0,
)

robot_state_0 = robot_model.state()
robot_state_1 = robot_model.state()
control = robot_model.control()
mj_collision_pipeline = newton.CollisionPipeline(
    robot_model,
    reduce_contacts=True,
    broad_phase="explicit",
)
mj_contacts = mj_collision_pipeline.contacts()

# --- Universe B: VBD deformable cable ---
cable_builder = newton.ModelBuilder()

cable_builder.add_rod(
    positions=cable_points,          # polyline vertices [m]
    quaternions=cable_quats,         # parallel-transport frames
    radius=0.003,                    # cable cross-section radius [m]
    stretch_stiffness=1e12,          # EA [N]
    bend_stiffness=3.0,              # EI [N*m^2]
    stretch_damping=1e-3,
    bend_damping=1.0,
)

# --- Proxy bodies: robot links mirrored into VBD ---
for body_id in proxy_body_ids:
    # Effective mass: reflects the inertia of the full articulated
    # chain when applicable, optionally scaled for coupling stability.
    proxy_id = cable_builder.add_body(
        xform=robot_state_0.body_q[body_id],
        mass=effective_mass[body_id],
    )
    for shape in shapes_on_body(robot_model, body_id):
        cable_builder.add_shape(body=proxy_id, **shape)

    robot_to_vbd[body_id] = proxy_id

cable_model = cable_builder.finalize()

vbd_solver = SolverVBD(
    cable_model,
    iterations=10,
)

vbd_state_0 = cable_model.state()
vbd_state_1 = cable_model.state()
vbd_control = cable_model.control()
vbd_collision_pipeline = newton.CollisionPipeline(cable_model)
vbd_contacts = vbd_collision_pipeline.contacts()

proxy_forces = wp.zeros(robot_model.body_count, dtype=wp.spatial_vector)
coupling_forces_cache = wp.zeros_like(proxy_forces)


@wp.kernel
def sync_proxy_state(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    src_body_q: wp.array(dtype=wp.transform),
    src_body_qd: wp.array(dtype=wp.spatial_vector),
    dst_body_q: wp.array(dtype=wp.transform),
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
    proxy_forces: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
):
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]

    # Copy pose and velocity from robot to proxy
    dst_body_q[pid] = src_body_q[rid]
    qd = src_body_qd[rid]

    # Undo coupling forces + gravity on proxy velocity
    f = proxy_forces[rid]
    delta_v = dt * body_inv_mass[pid] * wp.spatial_top(f)
    r = wp.transform_get_rotation(dst_body_q[pid])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[pid] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    qd = qd - wp.spatial_vector(delta_v + dt * body_inv_mass[pid] * gravity, delta_w)

    dst_body_qd[pid] = qd


# --- Coupled step (staggered, one-step lag) ---

# Step 1 -- Apply lagged VBD-to-MuJoCo wrenches
robot_state_0.clear_forces()
coupling_forces_cache.assign(proxy_forces)
robot_state_0.body_f.assign(robot_state_0.body_f + coupling_forces_cache)

# Step 2 -- Advance MuJoCo (rigid-body robot)
mj_collision_pipeline.collide(robot_state_0, mj_contacts)
mj_solver.step(robot_state_0, robot_state_1, control, mj_contacts, dt)
robot_state_0, robot_state_1 = robot_state_1, robot_state_0

# Step 3 + 4 -- Sync proxy poses/velocities and undo coupling forces (single kernel)
wp.launch(
    sync_proxy_state,
    dim=len(proxy_body_ids),
    inputs=[
        robot_ids_wp, proxy_ids_wp,
        robot_state_0.body_q, robot_state_0.body_qd,
        vbd_state_0.body_q, vbd_state_0.body_qd,
        coupling_forces_cache,
        cable_model.body_inv_mass, cable_model.body_inv_inertia,
        gravity, dt,
    ],
)

# Step 5 -- Advance VBD (cable deformation + cable-proxy contacts)
vbd_collision_pipeline.collide(vbd_state_0, vbd_contacts)
vbd_solver.step(vbd_state_0, vbd_state_1, vbd_control, vbd_contacts, dt)

# Step 6 -- Harvest contact wrenches from proxy bodies (applied at next step)
proxy_forces = harvest_proxy_wrenches(vbd_solver, vbd_state_1, vbd_contacts, dt)

vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
```

The Samsung/Lightwheel work demonstrates how Newton's deformable simulation stack enables synthetic data generation and policy training across the full material range found on real electronics assembly lines.

---

## How to Get Started with Newton

Newton is free to use, modify, and extend.

- Explore the [`newton-physics/newton`](https://github.com/newton-physics/newton) GitHub repo for [standalone Newton examples](https://github.com/newton-physics/newton?tab=readme-ov-file#examples) and [documentation](https://newton-physics.github.io/newton/stable/).
- Try the dexterous manipulation and locomotion workflows on the [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab/tree/feature/newton?tab=readme-ov-file) GitHub (feature/newton branch).

### Relevant GTC 2026 Sessions
- Disney's Robotic Characters: From the Screen to Reality via Physical AI
- An Introduction to Newton Physics Engine for Robotics
- Accelerate Robot Learning with NVIDIA Isaac Lab and Newton
- Build Robot-Ready Assets for Physically Accurate Simulations With Lightwheel
- How to use NVIDIA Warp to Build GPU-Accelerated Computational Physics Simulations

---

## Key Terms / Glossary (quick reference)

| Term | Meaning |
|------|---------|
| **Newton** | GPU-accelerated, open-source, modular physics engine for robotics; built on Warp + OpenUSD. Linux Foundation project (NVIDIA, Google DeepMind, Disney Research). |
| **Warp** | NVIDIA's Python framework for GPU-accelerated kernels; Newton's compute foundation. |
| **OpenUSD** | Universal Scene Description; shared data layer enabling Isaac Sim/Lab integration. |
| **Kamino** | Disney Research rigid-body solver; specializes in closed-chain / parallel-linkage mechanisms. |
| **MuJoCo Warp (MJWarp)** | GPU-scaled MuJoCo 3.5 (Google DeepMind); ~252× locomotion / ~475× manipulation speedups over MJX on RTX PRO 6000 Blackwell. |
| **VBD** | Vertex Block Descent solver; for linear (cable), thin (cloth), and volumetric (rubber) deformables. |
| **iMPM** | Implicit Material Point Method; particle/granular simulation (e.g. rough-terrain locomotion). |
| **SDF collision** | Signed-distance-field collision from CAD meshes; precise contact for tight tolerances, no mesh approximation. |
| **Hydroelastic contact** | Continuous pressure distribution over finite-area patches (vs. point contacts); inspired by Drake; better tactile fidelity + sim-to-real. |
| **Tiled camera sensor** | Warp-based ray-traced sensor; RGB/depth/albedo/normals/segmentation channels; supports meshes + Gaussian splats. |
| **Isaac Lab** | Open-source robot-learning framework (RL + imitation); Newton plugs in as a physics + camera backend. |
| **Isaac Sim** | NVIDIA robotics simulation app; native Newton integration via OpenUSD (v6.0 early access). |

---

*Reference file generated from the NVIDIA Technical Blog article (Mar 16, 2026). Prose paraphrased; code reproduced for reference. Visual/video assets described, not embedded.*