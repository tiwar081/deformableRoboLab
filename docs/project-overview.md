# Project overview — the pipeline at a high level

> This doc is a stable map of the end-to-end pipeline. It changes only on MAJOR codebase
> restructuring — never for tuning, fixes, or per-area detail (those live in the sibling `docs/`
> subfolders, indexed in AGENTS.md). It deliberately overlaps AGENTS.md as little as possible:
> AGENTS.md holds the vision, standards, and gotchas; this holds the shape of the system.

## What this project is

A simulation environment for Franka manipulation on Newton physics in which **deformable objects
(cables, cloth, bags, soft FEM bodies) are first-class citizens**, built toward generating custom
scenes and tasks from natural language and simulating robot policies in them with physical
fidelity — and, eventually, high graphical realism.

## The pipeline, end to end

```
asset catalog ──> scene gen ──> task gen ──> env gen ──> [trajectory gen] ──> physics sim ──> render
                  (agentic generation: agent_pipeline.py)   (in flight)        (example.py)      │
                                                                                                 ▼
                                                                                          success eval
```

1. **Asset catalog** — `assets/objects/scene_catalog.json`: the object library (procedural
   primitives, YCB/imported meshes, deformables), each entry carrying the kwargs for its physics
   config dataclass. Asset meshes and per-asset annotations (grasp records, regions) live under
   `assets/`.

2. **Agentic generation** (`agent_pipeline.py` + `agentic_pipeline/`) — three LLM stages, each
   validated before the next runs:
   - **Scene gen**: prompt → object selection + placement → spatial solver → headless physics
     settle check.
   - **Task gen**: scene → a feasibility-checked manipulation task (instruction variants, goal
     predicate, subtasks) → plus the robot placement (edge-mounted stand, reach-checked).
   - **Env gen**: background, table material, lighting, cameras.

3. **Trajectory generation** (in flight — `docs/trajPipeline/`) — turns a task + scene into robot
   motion. Precomputed per-asset grasp candidates (`deformableManipulationTools/grasp_library.py`
   + `grasp_passes/`) are selected at run time (`grasp_select/`); the stage that emits waypoints
   from a task does not exist yet — demo policies are still hand-authored.

4. **Physics simulation** — a demo is a pure data file (`examples/<name>.py` declaring a
   `DemoSpec`: scene + policy) played by the one runner, `example.py`. The framework
   (`deformableManipulationTools/`) owns everything physical: solver routing (rigid-only →
   MuJoCo; any deformable → split MuJoCo robot + VBD objects with a proxy contact bridge), the
   force-controlled grasp, materials, and all object/robot parameters.

5. **Rendering** (`robolabViz/`) — Newton USD, lightweight mp4, or the RoboLab-look ray-traced
   mp4, selected per run; the look is customizable per demo via `RenderSpec`.

6. **Success evaluation** (`agentic_pipeline/success.py`) — scores a rollout's final `SceneState`
   against the task's goal predicate (where a verified evaluator exists).

## Code layout (one line each)

- `deformableManipulationTools/` — ALL physics: params, framework, solvers, gripper, assets,
  grasp library/passes/selection.
- `agentic_pipeline/` — scene/task/env generation, settle check, success scoring, prompt templates.
- `robolabViz/` — the renderer and render-look configuration.
- `examples/` + `example.py` — demo data files and the single runner.
- `assets/` — robots, object meshes, catalogs, per-asset grasp/region annotations.
- `settings.yaml` — repo-wide switches (active robot, default render style, device).
- `docs/` — per-area depth, organized by pipeline stage; `docs/ONGOING.md` is the live log of
  in-flight work.
