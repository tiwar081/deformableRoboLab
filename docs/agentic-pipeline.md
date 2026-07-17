# Agentic pipeline (`agent_pipeline.py` + `agentic_pipeline/`)

The RoboLab-style generation stack restructured into three stages with a clean division of labor
(and deformables as first-class citizens throughout). Run end-to-end by `agent_pipeline.py`, or
interactively as a Claude Code session via `agentic_pipeline/SKILL.md` (registered under
`.claude/skills/agentic-pipeline`, RoboLab-skill style).

| stage | decides | checks |
|---|---|---|
| **scene gen** (`agentic_pipeline/scene_gen.py`) | object selection + placement ONLY (per prompt; relations incl. on/in stacking) | grammar + spatial solver + headless PHYSICS SETTLE CHECK (`agentic_pipeline/settle.py`: NaN, fell-off-table, >5 cm drift, residual deformable motion) with one feedback retry |
| **task gen** (`agentic_pipeline/task_gen.py`) | the manipulation task (reuses `agentic_pipeline/task_generator.py`'s predicates + deformable-aware feasibility) AND the ROBOT PLACEMENT | edge alignment (robot table touches, never overlaps, ≥5 cm edge contact; overhang allowed) + task-object reachability from the ACTUAL base (nearest-point ≤0.80 m); on failure the agent moves the robot or redesigns the task |
| **env gen** (`agentic_pipeline/env_gen.py`) | background HDRI, table material, lighting (dome + key light), exterior camera | camera bounds (0.5–3.5 m from table center, above the top); wrist camera ALWAYS on; no user camera spec → REPORTED default front bird's-eye view (opposite the robot, ~2 m up, whole workspace framed) |

**Robot placement** (`agentic_pipeline/geometry.py` — the hardcoded spatial layer): the robot is
always default-mounted on its own robot table (the franka_stand, measured 0.76 m wide × 0.90 m
deep; base 0.089 m behind its front edge). A placement is just `(edge, anchor)`: which workspace
table edge the stand's front edge touches (`back`/`front` long edges, `left`/`right` short edges)
and the world coordinate along it where the stand is centred — touch-without-overlap is baked into
the math (2 mm clearance), so only edge contact and reach need checking. The DEFAULT is the middle
of the **back long edge** (the legacy framework mount was the right short edge, from which the far
half of the table is out of reach). Reach intuition is preserved for the agent via
`edge_reach_text`: a per-(edge, anchor) table of which scene objects have some part within the
0.80 m reach radius. `robolabViz` renders the stand under any yawed base (the fixture follows
`robot_base_xform`).

**Direction words are ROBOT-POV** everywhere (`geometry.direction_vectors`): "in front" = outward
from the base into the workspace, "behind" = toward the base, left/right = the robot's hand sides.
Scene gen (which runs before placement is chosen) anchors them to the already-fixed placement
(scene_init) or the default mount. The relation solver rotates with the placement
(`resolve_placements(facing_yaw_deg=...)`).

**Prompts are data, not code**: every agent prompt is a `$slot` template in
`agentic_pipeline/prompts/*.md` (scene_system, task_system, placement_section, env_system,
self_prompt) — tune behavior there, not in Python.

## Modes

```bash
.venv/bin/python agent_pipeline.py "<description>" [--count N] [--placement default|task]
                                   [--camera "..."] [--verify] [--out-dir DIR]
.venv/bin/python agent_pipeline.py --user                # RoboLab-style stdin interview
.venv/bin/python agent_pipeline.py                       # userless: agent invents the prompt;
                                                         # default placement/camera; no verify
.venv/bin/python agent_pipeline.py --scene_init RUN_DIR  # rearrange: exact same object multiset,
                                                         # same robot placement + environment,
                                                         # NEW layout and a NEW task
```

Post-render visual verification is OPT-IN (`--verify`, or the interview question) — the default,
including userless, skips it to save time. Artifacts per run
(`outputs/agenticPipeline/<name>/`, numbered — never overwrites): `scene.json`, `task.json`
(embeds `robot_placement`), `env.json`, `pipeline.json` (manifest incl. settle + feasibility
reports), `pipeline_<name>.py` (a standard demo data file: `build.demo_from_dir` assembles the
DemoSpec, so the whole runner/renderer is reused), `scene_overview.png`, `scene_wrist.png`.

Boundary notes: task gen only IDEATES (no grasp force/trajectory — downstream pipeline);
`agentic_pipeline/scene_generator.py`/`agentic_pipeline/task_generator.py` remain the standalone one-shot tools (the pipeline
imports their internals; the legacy `_base_xy` origin bug is fixed to the true framework-default
mount).
