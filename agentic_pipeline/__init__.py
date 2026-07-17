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

from pathlib import Path
from string import Template

PIPELINE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PIPELINE_DIR / "prompts"


def load_prompt(name: str, **slots) -> str:
    """Load ``prompts/<name>.md`` and substitute ``$slot`` placeholders (string.Template — safe
    with literal JSON braces in the markdown). Unknown ``$slots`` in the file raise KeyError so a
    template/code mismatch fails loudly."""
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return Template(text).substitute(**slots)
