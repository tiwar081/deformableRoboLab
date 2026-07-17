# Agentic task generator (`agentic_pipeline/task_generator.py`)

RoboLab-style "task gen" for this repo. Where `agentic_pipeline/scene_generator.py` composes a tabletop *scene*
(objects + placement + look), the task generator reads that scene and an agent (Claude, default
`claude-fable-5`, fallback `claude-opus-4-8`) proposes ONE **meaningful, physically feasible
manipulation task**: an instruction (three phrasings), a goal predicate over the scene objects, a
subtask decomposition, and competency attributes. It mirrors RoboLab's task-gen in STRUCTURE; the
key difference is that our scenes contain **deformables**, so the feasibility checks are
deformable-aware.

The task generator only **ideates** the task. It does NOT decide grasp force, grip width, or the
trajectory — a downstream trajectory-generation pipeline owns those. Feasibility here answers "can a
sensible task of this shape exist in this scene?", not "can the gripper mechanically achieve this
grasp?".

## RoboLab's task-gen setup (the thing we mirror)

In RoboLab a **task = `{scene, instruction, terminations, subtasks, attributes}`**
(`_external/RoboLab/robolab/core/task/task.py`). The success condition is a **termination
predicate** — an executable function drawn from a ~50-entry conditional library
(`conditionals.py`): `object_in_container`, `object_on_top`, `stacked`, `object_left_of`/`right_of`/
`in_front_of`/`behind`, `object_groups_in_containers` (sorting), `object_outside_of`, plus negative
event predicates (`object_dropped`, `wrong_object_grabbed`, …). A concrete task
(`banana_in_bowl_task.py`) pairs a `DoneTerm(func=object_in_container, params={...})` with three
instruction variants (`default`/`vague`/`specific`) and a `subtasks=[pick_and_place(...)]`
decomposition for partial-credit scoring, tagged with competency `attributes`.

Task-gen itself is a **Claude Code skill** (`skills/robolab-taskgen/SKILL.md`), not runtime Python:
scene + goal → identify objects (must match scene prim names) → **map the instruction verb to a
termination conditional** ("Put X in Y" → `object_in_container`, "Stack" → `stacked`, "left of" →
`object_left_of`, "Sort" → `object_groups_in_containers`, "Take out" → `object_outside_of`) →
auto-generate the 3 instruction variants → decompose into subtasks → assign attributes/difficulty →
emit the task file. **Feasibility validation is STATIC/structural only** (`verify_task_valid`: the
success predicate's args are all supplied, and every object it names exists in the scene). There is
**no reachability, IK, graspability, or task-success simulation** anywhere — task feasibility is
assumed. (RoboLab's physics validate-and-refine loop — the 300-step settle that flags objects
displacing >2 cm — lives in *scene*-gen, `llm_scene_gen/physical_solver.py`, not task-gen.)

Our generator reproduces this structure exactly: the `Task` dataclass, the predicate library
(`GOAL_PREDICATES`), the verb→predicate mapping done by the LLM, the 3 instruction variants, the
subtask list, the attributes/difficulty, and the validate-and-refine feedback loop. We then slot the
deformable-aware feasibility checks where RoboLab's `verify_task_valid` sits.

## Our task-gen setup

Pipeline (mirrors `agentic_pipeline.scene_generator.call_scene_agent`):
LLM (structured output, JSON schema) → structural validation (`validate_task`) → **feasibility
checks** (`check_feasibility`) → on failure, natural-language feedback is appended and the agent
retries (≤3 attempts). The scene comes from `agentic_pipeline.scene_generator` — either an existing `outputs/` scene
(`--scene`) or one generated on the fly from the prompt (`sg.call_scene_agent`, no render).

The transport/credentials are shared with `agentic_pipeline.scene_generator` (raw HTTPS via `urllib`, Claude Code
OAuth token or `ANTHROPIC_API_KEY`; the Anthropic SDK conflicts with the venv's isaacsim
`typing_extensions` pin). `agentic_pipeline.task_generator` imports `agentic_pipeline.scene_generator` and reuses `_messages_request`,
`_response_text`, `catalog_by_name`, `workspace_bounds`, `call_scene_agent`, `write_scene`.

Run standalone → decide one task and write `task.json` next to the scene:

    .venv/bin/python -m agentic_pipeline.task_generator "fold the shirt and put it in the box"
    .venv/bin/python -m agentic_pipeline.task_generator "tidy the desk" --scene outputs/sceneGen/lunch_tidyup_desk

Helper API (for the full pipeline): `GOAL_PREDICATES`, `KIND_CATEGORY`, `call_task_agent`,
`validate_task`, `check_static`, `check_affordance`, `check_reachable`, `check_fits_in`,
`check_feasibility`, `task_to_dict`, `load_scene`, `write_task`, `generate_task` (end-to-end).

### The predicate library (`GOAL_PREDICATES`) — our analogue of `conditionals.py`

A data table naming each goal we support, with its required object params, a container flag, a human
verb, and — the deformable-relevant part — the **object categories** each predicate applies to.
Shared (rigid + deformable): `object_in_container`, `object_on_top`, the four relational
`object_{left_of,right_of,in_front_of,behind}`, `object_outside_of`. Rigid-only: `stacked`,
`object_groups_in_containers`. **Deformable-only** (the affordances that make deformables special):
`cloth_folded`, `cloth_draped_over` (cloth), `cable_coiled`, `cable_routed_through` (cable),
`object_compressed` (squishy). The LLM maps the instruction onto one predicate; the checks consult
the table.

### The additional feasibility checks (the differentiator)

All are static/geometric — no physics sim (matching RoboLab's static-only stance) — and each is a
pure `(ok, reason)` function the pipeline can call à la carte. Object categories come from
`KIND_CATEGORY` (`ycb_mesh`/`rigid_box`/`rubiks_cube` → rigid; `cloth`; `cable`;
`soft_mesh`/`soft_block` → squishy). Geometry comes from the catalog `dims` footprint and
`params.TABLE`/`FRANKA`.

1. **`check_static`** — RoboLab-parity structural check: the predicate is known, all its params are
   supplied and name scene objects, and any container param actually names an **open-top container**
   (bowl/mug/pitcher — a can/box/block is rigid but is *not* a container).

2. **`check_affordance`** — *the core deformable nuance.* The predicate's allowed categories must
   include the manipulated object's category. Only a **cloth** can be folded/draped, only a
   **cable** coiled/routed, only a **squishy** object compressed, only **rigid** objects stacked.
   This rejects "fold the banana", "stack the cable", "coil the cube" — tasks that are grammatically
   fine but physically meaningless for that object type. This is the check RoboLab has no analogue
   for, because RoboLab has no deformables.

3. **`check_reachable`** — coarse Cartesian reachability (RoboLab has none, so this already exceeds
   parity; still no IK): every object the task touches must sit inside the usable tabletop
   (`workspace_bounds()`) and within the arm's ~0.8 m reach radius from the base; objects outside the
   ~0.35–0.65 m sweet-spot (RoboLab's sensitivity finding, `docs/robolab.md`) are warned, not
   rejected.

4. **`check_fits_in`** — *folded-volume container fit, the second deformable nuance.* Container
   opening ≈ 85% of its footprint (wall inset); depth = its height. A **rigid** object must fit the
   opening laid flat. A **cloth/cable** that does NOT fit flat may fit **folded/coiled**: each fold
   halves the larger footprint axis and doubles the layer count — folding trades footprint for
   thickness, **conserving volume**. The check accepts the smallest fold count whose footprint fits
   the opening *and* whose stacked thickness fits the depth, and reports it ("fits after 2 coils").
   Crucially the volume constraint can still reject a fit: a whole T-shirt folds down to a bowl's
   opening but its fabric volume overfills the shallow depth — an honest, useful "no". This is what
   lets "put the cable in the bowl" (fits after 2 coils) succeed where the flat object wouldn't, and
   correctly distinguishes it from stuffing a garment into a coffee mug.

`check_feasibility` orchestrates 1–4 for the task's predicate + objects and returns
`{ok, checks:[{name,ok,reason}], summary}`, stored on `Task.feasibility` and printed in the report.
On a rejection the `summary` becomes the natural-language feedback for the agent's next attempt.

### Task-gen vs the trajectory pipeline (boundary)

The task generator stops at the ideated task. It deliberately does not touch grasp force
(`GraspWindow.force_target`), grip width, waypoints, or the `DemoSpec` policy — those belong to the
downstream trajectory-generation pipeline, which consumes `task.json` + `scene.json`. So the
feasibility checks here are about *task existence*, not *grasp mechanics* (there is intentionally no
grasp-force or contact-sim check).
