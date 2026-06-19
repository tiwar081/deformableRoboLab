# Newton + Isaac Lab — Franka / Cloth Manipulation (Extract)

**Source:** NVIDIA Technical Blog — *Train a Quadruped Locomotion Policy and Simulate Cloth Manipulation with NVIDIA Isaac Lab and Newton*
<https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/>

> *Scope of this extract: only the portions relevant to object manipulation with a Franka robot arm (cloth/deformable manipulation). The quadruped locomotion training walkthrough has been omitted. Prose is paraphrased; code snippets and shell commands are reproduced verbatim. Visuals described, not embedded.*

---

## Background: Newton solver model (relevant bits)

Newton is an open-source, extensible physics engine (NVIDIA, Google DeepMind, Disney Research; managed by the Linux Foundation), built on [NVIDIA Warp](https://developer.nvidia.com/warp-python) and [OpenUSD](https://www.nvidia.com/en-us/omniverse/usd/). It's compatible with robot-learning frameworks such as MuJoCo Playground and Isaac Lab.

Key points that matter for manipulation work:

- **Solver modules** are the core — they handle numerical integration and constraint solving. Solvers may be constraint- or force-based, direct or iterative, and may use maximal or reduced coordinate representations.
- **A common interface and shared data model** mean you interact with Newton the same way regardless of whether you run MuJoCo Warp, the Disney Research Kamino solver, or a custom solver. This lets you reuse collision handling, inverse kinematics, state management, and time-stepping logic without rewriting application code.
- **Tensor-based API** exposes physics states as PyTorch- and NumPy-compatible arrays for efficient batching and integration with learning frameworks. Via the Selection API, training scripts can query joint states, apply actions, and feed results back into learning algorithms through one consistent interface.
- **MuJoCo Warp** (Google DeepMind) is fully integrated as a Newton solver and also powers MJX and Playground, so models/benchmarks move across Newton, Isaac Lab, and MuJoCo with minimal friction.
- Newton and its solvers are **Apache 2.0** licensed.

### Beta-release highlight relevant to cloth/deformables
- Extended performance and stability of the **Vertex Block Descent (VBD)** solver for *thin deformables such as clothing*, plus the implicit **Material Point Method (MPM)** solver for granular materials.

### 🖼️ Figure 1 — Architecture
*Visual description:* Architecture diagram with stacked sections labeled **Isaac Lab → Newton → MuJoCo → Warp**, showing Newton as a standalone Python package providing GPU-accelerated interfaces for describing the physical model and state of robotic systems.

---

## Multiphysics with the Newton standalone engine

Multiphysics simulation captures coupled interactions between **rigid bodies** (e.g. a robot hand/arm) and **deformable objects** (e.g. cloth) within one framework, enabling more realistic evaluation and data-driven optimization of robot design, control, and task performance.

Although Newton works inside Isaac Lab, you can also use it directly from Python in **standalone mode** to experiment with complex physical systems. The example below is a rigid Franka robot arm manipulating a deformable cloth — demonstrating how the Newton API lets you combine multiple physics solvers in a single real-time simulation.

### Step 1 — Launch the interactive demo

Newton ships with a suite of runnable examples. The Franka arm + cloth demo launches with a single command from the root of the Newton repo.

First, set up the environment:

```bash
# Set up the uv environment for running Newton examples
uv sync --extra examples
```

Then run the cloth-manipulation example:

```bash
# Launch the Franka arm and cloth demo
uv run -m newton.examples cloth_franka
```

This opens an interactive viewer showing the GPU-accelerated simulation in real time. The demo uses a **GPU-based VBD Cloth solver**, runs at roughly **30 FPS on an RTX 4090**, and guarantees **penetration-free contact** throughout the simulation. Compared with other GPU simulators that also enforce penetration-free dynamics (e.g. GPU-IPC, a GPU Incremental Potential Contact solver), this example reportedly achieves **over 300× higher performance**, making it one of the fastest fully penetration-free cloth-manipulation demos currently available.

### 🎥 Video 3 — Cloth manipulation demo
*Visual description:* The Newton standalone engine running the cloth-manipulation demo, combining rigid-body and deformable physics; a Franka arm folding cloth, rendered in NVIDIA Omniverse Kit. (`robot-arm-folding-cloth.mp4`)

---

### Step 2 — Understanding the multiphysics coupling

This is a clean example of multiphysics: systems with different dynamical behaviors interacting, achieved by assigning a **specialized solver to each component**. In `example_cloth_franka.py`, the solvers are initialized like this:

```python
# Initialize a Featherstone solver for the robot
self.robot_solver = SolverFeatherstone(self.model, ...)

# Initialize a Vertex-Block Descent (VBD) solver for the cloth
self.cloth_solver = SolverVBD(self.model, ...)
```

You can swap the robot solver simply by changing `SolverFeatherstone` to another solver that supports rigid-body simulation, such as `SolverMuJoCo`.

The coordination happens in the simulation loop. This example uses **one-way coupling** — the rigid body affects the deformable, but not vice versa — which is acceptable for cloth manipulation, where the cloth's effect on robot dynamics can be neglected. The loop logic:

- **Update the robot:** `robot_solver` advances the Franka arm's state; the arm acts as a kinematic object.
- **Detect collisions:** the engine checks for collisions between the newly positioned robot and the cloth particles.
- **Update the cloth:** `cloth_solver` simulates the cloth's motion, reacting to collisions from the robot.

```python
# A simplified view of the simulation loop in example_cloth_franka.py

def simulate(self):
    for _step in range(self.sim_substeps):

        # 1. Step the robot solver forward
        self.robot_solver.step(self.state_0, self.state_1, ...)

        # 2. Check for contacts between the robot and the cloth
        self.contacts = self.model.collide(self.state_0, ...)

        # 3. Step the cloth solver, passing in robot contact information
        self.cloth_solver.step(self.state_0, self.state_1, ..., self.contacts, ...)
```

This explicit, user-controlled loop is the point: the Newton API gives fine-grained control over how different physical systems are coupled.

**Roadmap note:** the team plans deeper, more integrated coupling — including **two-way coupling** for scenarios where each system's effect on the other is significant (e.g. a robot locomoting on deformable soil/mud, where the terrain exerts forces back on the rigid bodies), and **implicit coupling** for select solver combinations to more automatically manage force exchange between systems.

---

## Ecosystem work relevant to manipulation / deformables

These collaborations are the manipulation- and cloth-relevant subset of the broader Newton ecosystem:

- **Peking University (PKU)** — extending Newton into tactile domains by integrating their IPC-based solver, **Taccel**, to simulate vision-based tactile sensing for robotic manipulators. Leveraging Newton's GPU-accelerated, differentiable architecture to model fine-grained contact interactions critical for tactile and deformable manipulation.
  - *Video 6:* Taccel simulation of Tac-Man manipulation aligns closely with real-world execution (small sim-real gap).

- **Style3D** — bringing cloth and soft-body simulation expertise to Newton for high-fidelity modeling of garments and deformable objects with complex interactions. A simplified version of the Style3D solver is already integrated into Newton, with plans to expose APIs for full-scale simulations involving millions of vertices.
  - *Video 7:* High-fidelity garment/deformable modeling using Newton.

- **Technical University of Munich (TUM)** — using Newton to run trained dexterous manipulation policies (validated on real robots) back in simulation, a first step toward closing the sim-real loop. Training with 4,000 parallel environments in MuJoCo Warp already works; next milestone is transferring policies to hardware, then extending to fine manipulation with a spatially resolved tactile skin.
  - *Video 8:* Newton running trained dexterous manipulation policies (validated on real robots) back in simulation.

---

## Getting started

- Standalone Newton Beta: [`newton-physics/newton`](https://github.com/newton-physics/newton) GitHub repo.
- Newton in Isaac Lab: [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) (feature/newton branch).
- Additional resources: [Newton Developer](https://developer.nvidia.com/newton-physics).

---

## Quick reference — manipulation-relevant API surface

| Symbol / command | Role |
|---|---|
| `uv sync --extra examples` | Set up the uv env for Newton examples. |
| `uv run -m newton.examples cloth_franka` | Launch the Franka arm + cloth demo. |
| `example_cloth_franka.py` | Source file for the demo (solver init + sim loop). |
| `SolverFeatherstone` | Reduced-coordinate rigid-body solver used for the Franka arm in the demo. |
| `SolverMuJoCo` | Drop-in alternative rigid-body solver (swap for `SolverFeatherstone`). |
| `SolverVBD` | Vertex Block Descent solver for thin deformables (cloth). Penetration-free, GPU-accelerated. |
| `self.model.collide(state, ...)` | Collision detection between robot and cloth particles; returns contacts. |
| `solver.step(state_0, state_1, ...)` | Advance a solver one substep (state double-buffering pattern). |
| `self.contacts` | Contact data passed from collision step into the cloth solver. |

*Coupling in the demo is one-way (rigid → deformable). Two-way and implicit coupling are on the roadmap.*

---

*Extract generated from the NVIDIA Technical Blog article. Prose paraphrased; code and shell commands reproduced verbatim for reference. Locomotion-only content omitted by request.*