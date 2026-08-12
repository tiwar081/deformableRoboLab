"""agentic_pipeline — the three-stage agentic generation pipeline (RoboLab-style, deformable-aware).

Stages (each an agent + deterministic checks; run in succession by ``agent_pipeline.py`` or
interactively via ``agentic_pipeline/SKILL.md``):

  1. scene gen (``scene_gen``)  — object SELECTION + PLACEMENT only, from the user prompt, plus the
                                  physics placement checks (spatial solver + headless settle check).
                                  No backgrounds/tables/cameras — that is environment gen's job.
  2. task gen  (``task_gen``)   — the manipulation TASK (instruction/goal/subtasks, reusing
                                  ``task_generator``'s deformable-aware feasibility) + ROBOT
                                  PLACEMENT (which workspace-table edge the robot table touches,
                                  and where), with edge-alignment + reachability checks.
  3. env gen   (``env_gen``)    — everything else: background HDRI, table texture, lighting, and
                                  cameras (wrist camera always; exterior camera from the user spec
                                  or the front bird's-eye default).

Shared spatial primitives (edges, stand geometry, reach, robot-POV direction words) live in
``geometry``; ``build`` assembles the final DemoSpec from the three stage artifacts
(scene.json / task.json / env.json). All agent PROMPTS live as templates in
``agentic_pipeline/prompts/*.md`` — the code only fills slots and points the model at them.
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

PIPELINE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PIPELINE_DIR / "prompts"
GOAL_PREDICATES_PATH = PIPELINE_DIR / "goal_predicates.json"

_REQUIRED_PREDICATE_FIELDS = ("params", "container_param", "obj_categories", "verb",
                              "attributes", "driver")


def load_prompt(name: str, **slots) -> str:
    """Load ``prompts/<name>.md`` and substitute ``$slot`` placeholders (string.Template — safe
    with literal JSON braces in the markdown). Unknown ``$slots`` in the file raise KeyError so a
    template/code mismatch fails loudly."""
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return Template(text).substitute(**slots)


def load_goal_predicates() -> tuple[dict, dict]:
    """Load ``goal_predicates.json`` — the goal predicate library (DATA, not code; see the file's
    ``_comment``). Returns ``(predicates, kind_category)``. ``obj_categories`` is converted to a set
    (JSON has no set type). Missing fields raise, so a malformed table fails loudly at import
    rather than silently skipping an affordance gate.

    Kept dependency-free (stdlib only) so both ``task_generator`` and the lightweight ``success``
    evaluator can read the same table without importing each other."""
    data = json.loads(GOAL_PREDICATES_PATH.read_text())
    kind_category = data["kind_category"]
    categories = set(data.get("categories", kind_category.values()))
    predicates: dict[str, dict] = {}
    for name, spec in data["predicates"].items():
        missing = [f for f in _REQUIRED_PREDICATE_FIELDS if f not in spec]
        if missing:
            raise ValueError(f"goal predicate {name!r} is missing field(s) {missing} "
                             f"in {GOAL_PREDICATES_PATH}")
        role_categories = spec.get("role_categories", {})
        unknown = (set(spec["obj_categories"])
                   | {cat for allowed in role_categories.values() for cat in allowed}) - categories
        if unknown:
            raise ValueError(f"goal predicate {name!r} names unknown object categories "
                             f"{sorted(unknown)}; known: {sorted(categories)}")
        unknown_roles = set(role_categories) - set(spec["params"])
        if unknown_roles:
            raise ValueError(f"goal predicate {name!r} gates roles not present in params: "
                             f"{sorted(unknown_roles)}")
        # A predicate is always GENERATABLE; ``driver`` only says whether it is SCORABLE yet.
        # null = no verified evaluator (the build queue lives in docs/agenticPipeline/success-evaluators.md, not in the table
        # — task gen never reads it). A NAMED driver must really exist in success.py: that is what
        # stops the withdrawn shape heuristics being re-declared by a data-only edit.
        if spec["driver"] is not None and spec["driver"] not in _success_drivers():
            raise ValueError(
                f"goal predicate {name!r} declares driver {spec['driver']!r}, which has no "
                f"evaluator in success.py (known: {sorted(_success_drivers())}). Either implement "
                f"it — with a probe showing it separates a true-success from a true-failure state "
                f"— or set driver to null (see docs/agenticPipeline/success-evaluators.md for the pending-evaluator queue).")
        predicates[name] = {**spec, "obj_categories": set(spec["obj_categories"])}
    return predicates, kind_category


def _success_drivers() -> set:
    """Driver names ``success.evaluate`` actually implements. Imported lazily and cached: ``success``
    reads this table, so a top-level import would cycle."""
    global _DRIVER_CACHE
    if _DRIVER_CACHE is None:
        from .success import DRIVERS
        _DRIVER_CACHE = DRIVERS
    return _DRIVER_CACHE


_DRIVER_CACHE = None
