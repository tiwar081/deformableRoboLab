"""Pipeline stage 2 — task gen: the manipulation TASK + the ROBOT PLACEMENT.

Reuses ``task_generator``'s predicate library and deformable-aware feasibility checks (affordance,
container fit, reach — reach now measured from the placement's actual base). Adds what RoboLab's
task gen never had to decide: WHERE THE ROBOT STANDS. The robot is always default-mounted on its
own robot table; the agent picks which workspace-table edge the robot table touches and where
along it (``geometry.robot_placement``); code checks edge alignment + task-object reachability and
feeds concrete reach numbers back, letting the agent EITHER move the robot OR redesign the task
(the user's requirement). Placement modes:

  "task"    — the agent chooses (edge, anchor) jointly with the task (the default for --user runs)
  "default" — the middle of the back LONG edge (``geometry.default_placement``); task must fit it
  "fixed"   — an externally supplied placement (scene_init mode); task must fit it

The agent prompt templates live in ``prompts/task_system.md`` + ``prompts/placement_section.md``.
"""
from __future__ import annotations

import json
import urllib.error

from . import scene_generator as sg
from . import task_generator as tg
from . import geometry, load_prompt


def _objects_text(scene: dict) -> str:
    present = tg._scene_by_name(scene)
    lines = []
    for nm, info in present.items():
        e = info["entry"]
        tag = f"{tg._category(e)}, {e['class']}"
        if tg._is_container(e):
            tag += ", CONTAINER (open-top)"
        d = e["dims"]
        pl = info["placement"]
        lines.append(f'- {nm} [{tag}, footprint {d[0]:.2f}x{d[1]:.2f} m] at '
                     f'({pl["x"]:.2f}, {pl["y"]:.2f}) — {e["description"]}')
    return "\n".join(lines)


def _placement_mode_text(mode: str, placement: dict | None) -> str:
    if mode == "task":
        return ("Choose the placement yourself: fill \"placement\": {\"edge\", \"anchor\"} in your "
                "JSON. Prefer a placement from which every task object is COMFORTABLY reachable "
                "(nearest points well inside the reach radius), using the reference table above.")
    p = placement or geometry.default_placement()
    return (f"The robot placement is FIXED for this run ({p['source']}): edge {p['edge']!r}, anchor "
            f"{p['anchor']:.2f} (base at ({p['base'][0]:.2f}, {p['base'][1]:.2f})). Set "
            f"\"placement\": null and design a task whose objects are reachable from THAT placement "
            f"(see the reference-table rows for edge {p['edge']!r}).")


def task_system(scene: dict, mode: str, placement: dict | None) -> str:
    x0, x1, y0, y1 = sg.workspace_bounds(margin=0.0)
    by_name = sg.catalog_by_name()
    placement_section = load_prompt(
        "placement_section",
        stand_w=f"{2 * geometry.STAND_HALF_WIDTH:.2f}",
        stand_d=f"{geometry.STAND_FRONT + geometry.STAND_BACK:.2f}",
        edges=", ".join(sorted(geometry.EDGES)),
        x0=f"{x0:.2f}", x1=f"{x1:.2f}", y0=f"{y0:.2f}", y1=f"{y1:.2f}",
        min_contact=f"{geometry.MIN_EDGE_CONTACT:.2f}",
        reach=f"{geometry.REACH_MAX:.2f}",
        base_off=f"{geometry.STAND_FRONT + geometry.CLEARANCE:.2f}",
        reach_table=geometry.edge_reach_text(scene, by_name),
        placement_mode=_placement_mode_text(mode, placement),
    )
    return load_prompt("task_system", objects=_objects_text(scene),
                       predicates=tg._predicate_table_text(),
                       placement_section=placement_section)


def _task_schema(scene: dict, mode: str) -> dict:
    schema = tg._task_schema(scene)
    schema["properties"]["placement"] = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {
                    "edge": {"type": "string", "enum": sorted(geometry.EDGES)},
                    "anchor": {"type": "number"},
                },
                "required": ["edge", "anchor"],
                "additionalProperties": False,
            },
        ],
        "description": ("robot placement (task mode) or null (fixed/default mode)"
                        if mode == "task" else "must be null — the placement is fixed for this run"),
    }
    schema["required"].append("placement")
    return schema


def call_task_agent(scene: dict, *, mode: str = "task", placement: dict | None = None,
                    avoid_goal=None, model: str = sg.DEFAULT_MODEL,
                    attempts: int = 4, verbose: bool = True) -> tuple[tg.Task, dict]:
    """Scene -> (Task, robot placement), with structural validation + base-aware feasibility +
    edge-alignment checks, and feedback retries in which the agent may move the robot or redesign
    the task. ``avoid_goal`` is a goal dict — or a LIST of them (scene reuse: the same scene gets
    several tasks) — that the new task must DIFFER from. Returns the accepted (or best-effort)
    pair; the placement is also embedded in the task dict written by ``write_task``."""
    if mode not in ("task", "default", "fixed"):
        raise ValueError(f"unknown placement mode {mode!r}")
    fixed = placement or geometry.default_placement() if mode in ("default", "fixed") else placement
    system, schema = task_system(scene, mode, fixed), _task_schema(scene, mode)
    by_name = sg.catalog_by_name()
    avoid = [avoid_goal] if isinstance(avoid_goal, dict) else list(avoid_goal or [])
    avoid_txt = ""
    if avoid:
        listed = "; ".join(f"{g.get('predicate')} {g.get('params')}" for g in avoid)
        avoid_txt = (f" Tasks ALREADY generated for this scene (or its previous run): {listed} — "
                     f"propose a DIFFERENT task this time (a different goal predicate, or the "
                     f"same predicate on different objects).")
    messages = [{"role": "user", "content":
                 f"Scene '{scene.get('name', '')}' ({scene.get('description', '')}). "
                 f"Original request: {scene.get('prompt', '(none)')}. Propose one task"
                 + (" and the robot placement." if mode == "task" else ".") + avoid_txt}]
    task = chosen = None
    use_model = model
    for attempt in range(attempts):
        try:
            resp = sg._messages_request(system, messages, use_model, schema)
            text = sg._response_text(resp)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[pipeline/task] {use_model} failed ({exc}); falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
        data = json.loads(text)
        task = tg._task_from_json(data, scene)
        task.model, task.prompt = use_model, scene.get("prompt", "")
        if mode == "task" and data.get("placement"):
            chosen = geometry.robot_placement(data["placement"]["edge"],
                                              float(data["placement"]["anchor"]), source="agent")
        else:
            chosen = fixed or geometry.default_placement()

        errs = tg.validate_task(task, scene)
        align_ok, align_msg = geometry.alignment_check(chosen)
        if not align_ok:
            errs.append(align_msg)
        if any(task.goal.get("predicate") == g.get("predicate")
               and task.goal.get("params") == g.get("params") for g in avoid):
            errs.append("this is the SAME goal as an already-generated task — choose a different one")
        feedback = "; ".join(errs)
        if not errs:
            report = tg.check_feasibility(task, scene, base_xy=tuple(chosen["base"][:2]))
            task.feasibility = report
            if report["ok"]:
                if verbose:
                    print(f"[pipeline/task] task {task.name!r} accepted on attempt {attempt + 1} "
                          f"(robot at {chosen['edge']!r} edge, anchor {chosen['anchor']:.2f})")
                return task, chosen
            names = sorted(set(task.goal.get("params", {}).values()))
            reach = geometry.reach_report(chosen, scene, by_name, names=names)
            reach_txt = "; ".join(f'{r["name"]}: nearest {r["d_nearest"]:.2f} m'
                                  f'{" (REACHABLE)" if r["reachable"] else " (OUT OF REACH)"}'
                                  for r in reach)
            feedback = (f"{report['summary']}. Reach from your placement "
                        f"({chosen['edge']!r}, anchor {chosen['anchor']:.2f}): {reach_txt}. "
                        + ("You may move the robot (new placement) or change the task."
                           if mode == "task" else
                           "The placement is fixed — change the task to use reachable objects."))
        if verbose:
            print(f"[pipeline/task] attempt {attempt + 1} rejected: {feedback}")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your task was rejected: {feedback}. "
                                                 f"Return a corrected task JSON."}]
    if task is None:
        raise RuntimeError("task agent produced no usable task")
    task.feasibility = tg.check_feasibility(task, scene, base_xy=tuple(chosen["base"][:2]))
    if verbose:
        print("[pipeline/task] accepting best-effort task after retries (feasibility may be partial)")
    return task, chosen


def task_dict(task: tg.Task, placement: dict) -> dict:
    d = tg.task_to_dict(task)
    d["robot_placement"] = placement
    # The EXECUTABLE success spec (RoboLab termination check): predicate + params + which geometric
    # driver evaluates it, so a downstream rollout can score success without re-deriving it.
    from . import success
    d["success_spec"] = success.compile_success_spec(task.goal)
    if task.subgoals:                    # multi-step: one compiled spec per subgoal, same contract
        d["subgoal_specs"] = [success.compile_success_spec(g) for g in task.subgoals]
    return d
