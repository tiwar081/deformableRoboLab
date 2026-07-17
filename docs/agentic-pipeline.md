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

Artifacts per run (`outputs/agenticPipeline/<name>/`, numbered — never overwrites): `scene.json`,
`task.json` (embeds `robot_placement` + the executable `success_spec`), `env.json`, `pipeline.json`
(manifest: settle report, `scene_metrics`, `feasibility`, `success_spec`, camera), `pipeline_<name>.py`
(a standard demo data file: `build.demo_from_dir` assembles the DemoSpec, so the whole
runner/renderer is reused), `scene_overview.png`, `scene_wrist.png`.

## RoboLab-parity features (all on by default)

- **Elongated-aware collision** (`packing.obb_penetration`): the spatial solver resolves overlaps
  with an OBB/SAT narrow phase (circle broad-phase prefilter), so a yawed banana conflicts along
  its true footprint, not a max(w,d) circle — strictly better than RoboLab (whose "convex hull"
  solver is actually circle-only).
- **Support-ratio stacking** (`task_generator.check_support`, `packing.support_ratio`): an
  on-top/stacked goal is rejected if < 50 % of the object's rotated footprint is over the support.
- **Multi-object container packing** (`packing.pack_into_container`, `check_group_fits`): RoboLab's
  ellipse-mouth (0.43×dims) upward-layered packing; `object_groups_in_containers` checks the whole
  group fits.
- **Partial containment** (`PARTIAL_CONTAINMENT=0.35`): an elongated rigid object (banana) counts
  as "fits the bowl" when its short axis fits even though the long axis protrudes.
- **Settled-pose write-back** (RoboLab `--replace`): after the settle check, rigid objects'
  `x/y/yaw` in `scene.json` are overwritten with their SETTLED poses (spawn poses preserved under
  `spawn_*`). `--no-settle-writeback` keeps spawn poses.
- **Scene quality metrics** (`scene_metrics.py`): RoboLab's compactness / diversity / has_container
  / coverage formulas + qualitative notes, stored in the manifest.
- **Executable success predicates** (`success.py`): the goal compiles to a runtime geometric test
  (open-top convex-hull containment, footprint-support, robot-POV 45° cone, deformable proxies) —
  `success.evaluate(predicate, params, SceneState)` scores task success during a rollout. The spec
  is embedded in `task.json` as `success_spec`.
- **Catalog ops** (`catalog_ops.py`): `--ingest <dir>` scans a directory of USD files into the
  scene catalog (USD AABB dims, authored class/description, mass/friction with inference notes);
  `--regen` re-extracts dims. `--container` marks ingested objects as open-top containers.
- **Containers**: the catalog carries open-top containers via a `container: true` flag —
  bowl/mug/pitcher plus vendored VoMP tool bins (`parts_bin` 16 cm, `tool_bin` 23 cm,
  `long_tray_bin` 30 cm; all NON-articulated, coacd-decomposed so the cavity holds objects and they
  settle cleanly on the rigid MuJoCo path). `_is_container` reads the flag, so new containers
  register automatically. (A thin-walled utility bucket was trialed but sinks on the rigid contact
  path — the settle check correctly flags it — so it was dropped; the bins cover tool-container
  scenes.)

### Flags (all heavyweight/optional features default ON)

| flag | effect |
|---|---|
| `--user` | interactive RoboLab-style stdin interview |
| `--scene_init RUN_DIR` | rearrange: reuse a run's exact object multiset + robot placement + env; new layout + different task |
| `--placement default\|task` | robot placement mode (userless default: `default`) |
| `--camera "..."` | exterior-camera specification (else the reported bird's-eye default) |
| `--count N` | approximate object count |
| `--name` / `--out-dir` / `--model` / `--device` / `--seed` | run identity + engine knobs |
| `--no-render` | **[heavyweight]** skip the final PBR render |
| `--no-verify` | **[heavyweight]** skip the post-render visual verification (agent inspects the image); ON by default |
| `--skip-settle-check` | **[heavyweight]** skip the headless physics settle check + its ≤3 retries |
| `--no-settle-writeback` | keep spawn poses instead of writing settled poses back |

Non-heavyweight checks (grammar, spatial/OBB solver, feasibility incl. reach/affordance/fit/support,
quality metrics, success-spec compile) are cheap and always run.

Boundary notes: task gen only IDEATES (no grasp force/trajectory — downstream pipeline);
`agentic_pipeline/scene_generator.py` / `agentic_pipeline/task_generator.py` remain the standalone
one-shot tools (the pipeline imports their internals; the legacy `_base_xy` origin bug is fixed to
the true framework-default mount).
