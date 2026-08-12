---
name: agentic-pipeline
description: >
  Generate a complete simulation setup — scene (objects on the workspace table), task (manipulation
  goal + robot placement), and environment (background, table material, lighting, cameras) — from a
  natural-language description, with deformable objects (cloth, cables, squishy) as first-class
  citizens. Use when the user wants to create a scene, a task, or a full demo setup, or to
  rearrange an existing generated scene.
---

# Agentic Pipeline (scene gen → task gen → environment gen)

Everything runs through `agent_pipeline.py` (repo root) and the `agentic_pipeline/` package. The
pipeline is RoboLab's scene/task generation restructured for this repo, plus deformables:

- **scene gen** — object selection + placement ONLY (spatial solver + headless physics settle
  check). No look decisions.
- **task gen** — the manipulation task (instruction variants, goal predicate, subtasks,
  deformable-aware feasibility) AND the robot placement: the robot table touches one edge of the
  workspace table; reach + edge-alignment are checked; on failure the agent moves the robot or
  redesigns the task.
- **env gen** — background HDRI, table material, lighting, cameras. The wrist camera is ALWAYS on;
  the exterior camera follows the user's spec, else a front bird's-eye default (opposite the
  robot, ~2 m up) — and you must TELL the user when the default was used.

Direction words (in prompts, relations, and anything you write to the user) are from the ROBOT'S
point of view: "in front" = outward from the robot base into the workspace, "behind" = toward the
robot, "left"/"right" = the robot's hand sides. Agent prompt templates live in
`agentic_pipeline/prompts/*.md` — edit those, not the Python, to tune agent behavior.

## When Invoked

Display this message (fill the object list from `assets/objects/scene_catalog.json`):

---

I'll generate a full simulation setup (scene → task + robot placement → environment). I need:

1. **Scene description** — a BROAD description of the situation, not an inventory (e.g. "a messy
   office desk after lunch", "dirty laundry in my laundry bin in my room", "a workshop bench
   halfway through a repair"). The pipeline decides which objects appear and where they go; the
   concrete layout comes back in the scene's `description`. A user who does name specific objects
   is of course honored.
2. **Number of objects** — blank = 3-5 for simple scenes (hard range 1-7).
3. **Robot placement** — `default` (middle of the back long edge of the workspace table) or
   `task` (the agent picks the best edge/anchor for the task it designs).
4. **Exterior camera** — describe a view, or blank for the default front bird's-eye view
   (opposite the robot, ~2 m above the table, whole workspace in frame). A wrist camera is always
   mounted regardless.
5. **Visual verification?** — post-render check where the agent inspects BOTH rendered stills
   (over-the-shoulder + wrist); extra model calls + possible re-render. Default ON; answer `n` to
   skip.
6. **Output directory** — default `outputs/agenticPipeline/`.

**Objects I can pick from** (`assets/objects/scene_catalog.json`): rigid YCB/HOT3D/Objaverse items
(banana, bowl, mug, cans, boxes, pitcher, wood block, rubik's cube, steel cube, apple), cloth
garments (gray/green t-shirt, blue dress), cables (power cable, nylon rope), and squishy FEM
objects (sponge, foam brick, soft banana, raspberry cube), plus three Objaverse-derived deformable
bags (plastic recycle, woven tote, paper grocery). Generated scenes currently allow at most one
total item across bag, garment cloth, and squishy families.

---

Then wait for the answers. Reuse the user's previous answers for later runs in the same session
instead of asking again (except the scene description).

**Rearranging an existing run (`--scene_init`) — ask FEWER questions.** The source run supplies the
scenario PROMPT (verbatim — the scenario does not change), the object multiset, the robot
placement, and the environment (camera included), so questions 2, 3 and 4 are dead: their answers
would be discarded, and `--count` / `--placement` / `--camera` are a hard CLI error alongside
`--scene_init`. What changes is the ARRANGEMENT — positions, orientations, what sits on/in what —
enough that a different task becomes natural. Ask only:

1. **Any specific change to the arrangement?** — optional; blank = the pipeline invents one. This
   is a CHANGE REQUEST ("stand the bucket on its side", "clear the bins"), NOT a new scenario; pass
   it as the positional argument.
2. **Any object-set changes?** — optional. A described change is followed EXACTLY as stated —
   swaps, adds, or removals, even past the usual composition limits (pass it as
   `--substitute "..."`). An answer of just "yes" means pass the flag BARE: the pipeline picks 1-2
   strict one-for-one swaps itself (count invariant, limits respected). This is the ONLY way the
   object set changes; without the flag the objects are fixed.
3. **Visual verification?** (default ON) 4. **Output directory**.

State what is being inherited (the scenario prompt, objects, robot placement, environment) instead
of asking about it, and that the new task will differ from the source's. Only if the source run has
no `env.json` does the camera question come back.

## Running the pipeline

Compose the command from the answers (long timeout — a render takes minutes; cloth scenes longest):

```bash
.venv/bin/python agent_pipeline.py "<scene description>" \
    [--count N] [--placement default|task] [--camera "<camera description>"] \
    [--no-verify] [--out-dir <dir>] [--device cuda:0]
```

- Userless run (no description given at all): `.venv/bin/python agent_pipeline.py` — the agent
  invents the prompt; defaults apply (default placement, default camera; verification still ON).
- Rearrange an existing run: `.venv/bin/python agent_pipeline.py --scene_init <run_dir>` — reuses
  that run's scenario prompt, exact objects, robot placement, and environment; generates a NEW
  arrangement and a NEW task. A positional argument here is a CHANGE REQUEST for the arrangement,
  not a new scenario (blank = the pipeline invents one); `--substitute "<changes>"` is followed
  verbatim (bare flag = the pipeline picks 1-2 strict swaps). Do NOT pass `--count`,
  `--placement` or `--camera` with it (inherited; the CLI rejects them).

The run prints progress per stage (`[pipeline/scene]`, `[pipeline/task]`, `[pipeline/env]`) and
writes `outputs/agenticPipeline/<name>/`:
`scene.json`, `task.json` (includes `robot_placement`), `env.json`, `pipeline.json` (manifest),
`pipeline_<name>.py` (runnable demo data file), `scene_overview.png`, `scene_wrist.png`.

## After the run

1. Read `pipeline.json` and report, in plain language: the task instruction, the robot placement
   (edge/anchor + why acceptable: alignment + reach results from `feasibility`), the settle-check
   result, and the camera choice — EXPLICITLY tell the user when the camera defaulted
   (`camera_report` is non-null in that case).
2. Show `scene_overview.png` and `scene_wrist.png` to the user with the Read tool.
3. Sanity-check the overview still yourself: every object visible on the table, nothing fallen or
   floating, arrangement matches the description. If something is off, say so and offer to re-run
   (a different seed, fewer objects, or more spread).
4. Offer next steps:
   - **Different task, same scenario**: `--scene_init <run_dir>` keeps the prompt, assets,
     placement and env, and re-arranges the objects so a different task fits.
   - **Watch the settle**: `outputs/<robot>/pipeline_<name>/simulation_advanced.mp4`.
   - A trajectory/grasp pipeline is downstream work — task gen only IDEATES the task (the one
     grasp knob later will be `GraspWindow.force_target`).

## Failure handling

- Scene/task/env agents already retry with feedback internally (3-4 attempts). If the run still
  fails: fewer objects, more spread, or a simpler task usually fixes it — rerun with the adjusted
  description.
- Settle-check failures print `[pipeline/scene] the physics settle check failed: ...` and retry
  once automatically; if the warning persists, inspect `pipeline.json`'s `settle` block and tell
  the user which object misbehaved.
- If an object the task needs is out of reach with `--placement default`, rerun with
  `--placement task` (lets the agent move the robot) — or accept the agent's redesigned task.
