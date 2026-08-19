# Agentic pipeline (`agent_pipeline.py` + `agentic_pipeline/`)

The RoboLab-style generation stack restructured into three stages with a clean division of labor
(and deformables as first-class citizens throughout). Run end-to-end by `agent_pipeline.py`, or
interactively as a Claude Code session via `agentic_pipeline/SKILL.md` (registered under
`.claude/skills/agentic-pipeline`, RoboLab-skill style).

| stage | decides | checks |
|---|---|---|
| **scene gen** (`agentic_pipeline/scene_gen.py`) | object selection + placement ONLY (per prompt; relations incl. on/in stacking) | grammar + spatial solver + headless PHYSICS SETTLE CHECK (`agentic_pipeline/settle.py`: NaN, fell-off-table, >5 cm drift, residual deformable motion) with one feedback retry |
| **task gen** (`agentic_pipeline/task_gen.py`) | the manipulation task (reuses `agentic_pipeline/task_generator.py`'s predicates + deformable-aware feasibility) AND the ROBOT PLACEMENT. **Scene reuse**: `--tasks N` (default 3) generates N DIFFERENT tasks for the one scene (`task.json`, `task_2.json`, … each avoiding all earlier goals, sharing task 1's placement, each with its own demo file). Tasks may be **MULTI-STEP** (optional `subgoals` chain of 2–4 single-object goals — "put all the cans in the bin"); each subgoal is feasibility-checked and compiled into `subgoal_specs`. Task gen sees NO grasp-candidate data (only the physical jaw-width ceiling) — grasp-difficulty evidence goes to `assets/low_graspability.md`, written by the trajectory stage, never read by generation | edge alignment (robot table touches, never overlaps, ≥5 cm edge contact; overhang allowed) + task-object reachability from the ACTUAL base (nearest-point ≤0.80 m); on failure the agent moves the robot or redesigns the task |
| **env gen** (`agentic_pipeline/env_gen.py`) | background HDRI, table material, lighting (dome + key light), exterior camera | camera bounds (0.5–3.5 m from table center, above the top); wrist camera ALWAYS on; no user camera spec → REPORTED default front bird's-eye view (opposite the robot, ~2 m up, whole workspace framed) |

Temporary solver/catalog guard: scene gen permits at most **one total** among a deformable bag,
cloth garment, and squishy FEM object — defined ONCE as
`scene_generator.MAX_PARTICLE_DEFORMABLES` (see "The particle-deformable limit is centralized"
below); enforced in `validate_scene`, not only in the prompt. All pairings and duplicates across
those three families are rejected until multi-particle-material scenes are explicitly supported
(exception: a USER-requested `--substitute` phrase overrides it). A cable may coexist, subject to
its separate one-cable limit.

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
self_prompt, self_change) — tune behavior there, not in Python.

## Modes

```bash
.venv/bin/python agent_pipeline.py "<description>" [--count N] [--placement default|task]
                                   [--camera "..."] [--no-verify] [--out-dir DIR]
.venv/bin/python agent_pipeline.py --user                # RoboLab-style stdin interview
.venv/bin/python agent_pipeline.py                       # userless: agent invents the prompt;
                                                         # default placement/camera (verify still ON)
.venv/bin/python agent_pipeline.py --scene_init RUN_DIR ["<change request>"] [--substitute "..."]
                                                         # rearrange: same scenario prompt, objects,
                                                         # robot placement + environment;
                                                         # NEW layout and a NEW task
```

**Prompt vs description.** The `prompt` is a BROAD situation — "a messy office desk after lunch",
"dirty laundry in my laundry bin in my room" — not an inventory. Which objects exist, how many, and
where they go is the scene agent's job, and the concrete result is stated in the scene's
`description` field ("a mug left of the bin, two cans in it, …"). `prompts/self_prompt.md` holds
the userless prompt-inventing agent to that shape explicitly (one sentence, no counts/coordinates/
relations/materials; catalog shown only so the situation stays buildable).

**Rearrange mode changes the ARRANGEMENT, never the scenario.** `--scene_init` carries the source
run's `prompt` over VERBATIM (so chained rearranges keep one stable scenario) and re-composes the
same objects into a materially different layout — new positions, orientations, and on/in relations
— so that a *different* task becomes the natural one. The scene agent is shown the PREVIOUS layout
and told to differ from it substantially ("nudging the old layout is NOT a rearrangement"); only
the `description` is rewritten. A positional prompt (or the interview's first question) in this
mode is **not** a new scenario but a CHANGE REQUEST ("turn the bucket on its side"), recorded in
the manifest as `change_request`. Left blank, `self_change_request()` invents one from the original
prompt + the previous layout + the task to avoid (`prompts/self_change.md`) — note this is a
*semantic twist*, not the only pressure to differ: the rearrange system-prompt block demands a
material re-arrangement with or without it.

**Object substitutions** happen ONLY under the `--substitute` flag (valid only with
`--scene_init`; also reachable via the interview's second rearrange question). Two regimes:

- **User-written phrase** (flag value or interview answer) — followed VERBATIM, no matter what
  (`substitutions_free`): adds, removes, any number of swaps, and results past the composition
  limits (extra deformables included) are all valid if requested. No multiset or composition
  enforcement runs; only names/bounds/relations are still checked, and the system prompt tells the
  agent the user's request overrides the hard constraints.
- **Bare flag** — `infer_substitutions()` (`prompts/self_substitute.md`) picks 1-`MAX_SUBSTITUTIONS`
  (2) swaps itself and writes the phrase, under the STRICT regime: one-for-one only, count
  invariant (never add/remove), composition limits enforced (`scene_gen._multiset_errors` +
  `check_composition=True`) — grandfathered against the source's deformable count
  (`deformable_baseline_of`: a scene already over the limit is never forced to shed, only barred
  from adding more). A bag/garment can only be swapped into a scene of ≤ `CLOTH_SCENE_MAX` (4)
  objects; the inference prompt steers around it in bigger scenes.

Without the flag the object set is exact; the userless change-request agent never proposes swaps.
Either regime re-opens the full catalog in the schema. The manifest records `substitutions` and
`substitutions_source` (`"user"` / `"agent"` / null).

**The particle-deformable limit is centralized** in `scene_generator.py`:
`MAX_PARTICLE_DEFORMABLES` (= 1 today) + `particle_family` / `count_particle_deformables` /
`deformable_limit_text` are the ONE definition — validation (`_composition_errors`), both scene
prompts (`$deformable_rule` slot in `prompts/scene_system.md`; the standalone `_agent_system`), and
the substitution-inference prompt all route through them, and `CLOTH_SCENE_MAX` lives beside them.
To change the limit, change that constant (prose copies to update by hand are listed at its
definition).

`scene_init_reuse()` (in `agent_pipeline.py`) is the one place that reads what a source run LOCKS —
prompt, object multiset, previous layout, robot placement, environment (camera included, when the
source has an `env.json`) — and both entry points defer to it, so an input whose answer would be
discarded is never collected: `--count` / `--placement` are a hard CLI error with `--scene_init`
(`--camera` too unless the source has no `env.json`), and `--user` asks only the live questions
(change request, substitutions, verification, output dir). Inside scene gen the object count is
stated as FIXED rather than as a range, and — when no substitution is permitted — the multiset-only
validations (count range, max-2-duplicates, one-deformable, cable count, cloth-scene cap) are
skipped: the multiset is copied from a scene that already passed them, so the multiset-equality
check subsumes them (`validate_scene(check_composition=False)`).

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
  `spawn_*`). Always on — it only ever adds information.
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
  `long_tray_bin` 30 cm) and a `bucket` (utility bucket, 27 cm); all NON-articulated,
  coacd-decomposed so the cavity holds objects and they settle flat on the rigid MuJoCo path.
  `_is_container` reads the flag, so new containers register automatically. (The bucket's USD is a
  MULTI-mesh asset — decal + body + handle prims — and depends on the multi-mesh merge in
  `mesh_collision.load_usd_mesh`; see the import rules in
  [scene-generator.md](scene-generator.md).)

### Flags (all heavyweight/optional features default ON)

| flag | effect |
|---|---|
| `--user` | interactive RoboLab-style stdin interview |
| `--scene_init RUN_DIR` | rearrange: reuse a run's prompt + exact object multiset + robot placement + env; new layout + different task. **Rejects `--count`/`--placement`** (and `--camera` when the source has an `env.json`) — they are inherited, not chosen. The positional argument becomes the CHANGE REQUEST |
| `--substitute ["..."]` | `--scene_init` only: change the object set. A phrase is followed VERBATIM (adds/removes/any swaps, limits waived); bare flag = the agent picks 1-2 strict one-for-one swaps itself; without the flag the object set is fixed |
| `--placement default\|task` | robot placement mode (userless default: `default`); invalid with `--scene_init` |
| `--camera "..."` | exterior-camera specification (else the reported bird's-eye default); invalid with `--scene_init` unless the source run has no `env.json` |
| `--count N` | approximate object count; invalid with `--scene_init` (the count is the source multiset's) |
| `--tasks N` | tasks to generate for the ONE scene (scene reuse; default 3; also asked in the `--user` interview). Each extra task gets `task_k.json` + `pipeline_<slug>__t<k>.py`; the trajectory stage runs them all |
| `--name` / `--out-dir` / `--model` / `--device` / `--seed` | run identity + engine knobs |
| `--no-render` | **[heavyweight]** skip the final PBR render |
| `--no-verify` | **[heavyweight]** skip the post-render visual verification (agent inspects BOTH the over-the-shoulder and the wrist-camera stills); ON by default |
| `--skip-settle-check` | **[heavyweight]** skip the headless physics settle check + its ≤3 retries |

(Settled-pose write-back is always on — spawn poses are preserved under `spawn_*`, so it only adds
information.)

Non-heavyweight checks (grammar, spatial/OBB solver, feasibility incl. reach/affordance/fit/support,
quality metrics, success-spec compile) are cheap and always run.

Boundary notes: task gen only IDEATES (no grasp force/trajectory — downstream pipeline);
`agentic_pipeline/scene_generator.py` / `agentic_pipeline/task_generator.py` remain the standalone
one-shot tools (the pipeline imports their internals; the legacy `_base_xy` origin bug is fixed to
the true framework-default mount).

## Agent-transport constraints (measured — apply to any new stage)

- **Raw HTTPS, not the Anthropic SDK.** Requests go over `urllib` (`scene_generator._messages_request`,
  reused by every stage) because the SDK conflicts with the venv's isaacsim `typing_extensions==4.12.2`
  pin. Credentials: `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`, else the Claude Code OAuth token in
  `~/.claude/.credentials.json`. With an OAuth token the FIRST system block must be the Claude Code
  identity line (`CLAUDE_CODE_IDENTITY`) or inference is rejected; it is harmless with an API key.
- **The structured-output schema REJECTS array bounds** (HTTP 400): `maxItems` always, and
  `minItems` for any value other than 0 or 1 (measured 2026-08-12 in the trajectory stage's retry
  schema). State the exact count in the prompt/description and enforce it in Python (truncate/pad
  after parsing). A new stage that needs "exactly/at most N of X" has to do both.
- Model default `claude-fable-5` with a `claude-opus-4-8` fallback on HTTP/parse failure; each stage
  retries with the validator's natural-language feedback appended (≤3 attempts).

**Validated end-to-end** (2026-07-22): a userless run
(`outputs/agenticPipeline/workshop_bench_cable` — self-invented prompt, default back-long-edge
placement, reported bird's-eye camera, settle clean, feasible cable-coiling task); `--user` interview
parsing; `--scene_init` rearrange including a live user bag-swap into a 7-object scene (multiset
preserved, composition clean) and an agent-inferred strict swap; yawed `robot_base_xform` placements
(IK/FK/physics verified, render fixture follows base yaw); offline unit tests for
placements/alignment/reach/schemas.
