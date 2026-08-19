"""Per-scene trajectory annotations (``annotations.json``) — the VLA training-data record.

One file per scene run dir, one row per EXECUTED trajectory (a scene now carries several tasks,
and a task may execute several rounds under visual-verification tuning — every round that ran to
completion is a row, because an executed, physically valid trajectory is training data for
whatever it actually did). Consistency rule: the ``instructions`` of a row always describe the
OUTCOME the video shows — when the visual check finds a mismatch, the row is RELABELED with the
achieved instruction set (the original stays under ``intended_instructions``), never deleted.

Row contents (chosen for VLA post-training): instruction set (vague/default/specific), goal
predicate + params, the executed grasp (candidate/source/width/force/pre-shape), phase timeline
(the temporal segmentation labels), rollout metrics, verification status, and the artifact paths
(traj json + video + task file) — everything needed to pair frames with language and actions.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load(run_dir: Path) -> dict:
    p = Path(run_dir) / "annotations.json"
    if p.exists():
        return json.loads(p.read_text())
    scene = json.loads((Path(run_dir) / "scene.json").read_text())
    return {
        "scene": scene.get("name"),
        "prompt": scene.get("prompt"),
        "description": scene.get("description"),
        "objects": [{"name": o.get("name"), "x": o.get("x"), "y": o.get("y"),
                     "yaw_deg": o.get("yaw_deg")} for o in scene.get("objects", [])],
        "fps": 60,
        "trajectories": [],
    }


def upsert(run_dir: Path | str, row: dict) -> Path:
    """Insert or update (by ``id``) one trajectory row; returns the annotations path."""
    run_dir = Path(run_dir)
    data = _load(run_dir)
    rows = data["trajectories"]
    for i, r in enumerate(rows):
        if r.get("id") == row.get("id"):
            rows[i] = row
            break
    else:
        rows.append(row)
    if "robot_placement" not in data and row.get("robot_placement"):
        data["robot_placement"] = row.pop("robot_placement")
    p = run_dir / "annotations.json"
    p.write_text(json.dumps(data, indent=1))
    return p


def build_row(*, task: dict, task_name: str, report: dict, round_no: int,
              traj_file: str | None, video: str | None, verdict: dict | None) -> dict:
    """One trajectory row from the stage report (+ the visual verdict, when one ran).

    Relabeling: when the verdict says the outcome does NOT match the instruction, the row's
    ``instructions`` become the ACHIEVED set from the verdict and the original moves to
    ``intended_instructions`` — the label always tells the truth about the video."""
    final = report.get("final") or {}
    last = (report.get("attempts") or [{}])[-1]
    ev = last.get("evaluation") or {}
    plan_pick = last.get("plan_pick") or {}
    tag_id = f"{Path(task_name).stem}_r{round_no}"

    instructions = dict(task.get("instruction") or {})
    intended = None
    relabeled = False
    if verdict is not None and not verdict.get("ok") and verdict.get("achieved_instruction"):
        intended = instructions
        instructions = dict(verdict["achieved_instruction"])
        relabeled = True

    row = {
        "id": tag_id,
        "task_file": task_name,
        "round": round_no,
        "traj_file": traj_file,
        "video": video,
        "executed": bool(report.get("ok")),
        "aborted": report.get("aborted"),
        "instructions": instructions,
        "relabeled": relabeled,
        "intended_instructions": intended,
        "goal": task.get("goal"),
        "subgoals": task.get("subgoals") or None,
        "segments_metrics": ev.get("segments"),
        "subgoal_checks": ev.get("subgoal_checks"),
        "subtasks": task.get("subtasks"),
        "attributes": task.get("attributes"),
        "difficulty": task.get("difficulty"),
        "target": (task.get("goal", {}).get("params") or {}).get("object"),
        "grasp": {k: plan_pick.get(k) for k in
                  ("id", "source", "seat_mode", "width", "yaw_deg", "tilt_deg", "position",
                   "approach", "adjusted")} if plan_pick else None,
        "phases": (json.loads(Path(report["run_dir"], traj_file).read_text()).get("phases")
                   if traj_file and Path(report["run_dir"], traj_file).exists() else None),
        "metrics": {k: ev.get(k) for k in
                    ("held", "carried", "failure", "close_drift", "lift_rise", "place_xy_err",
                     "final_pos")} if ev else None,
        "goal_check_geometric": ev.get("goal"),
        "visual_verification": ({k: verdict.get(k) for k in
                                 ("ok", "observed_outcome", "reasoning", "model")}
                                if verdict is not None else None),
        "attempts_to_grasp": len(report.get("attempts") or []),
        "final_candidate": final.get("candidate_id"),
        "robot_placement": task.get("robot_placement"),
        "timings": {"stage_s": report.get("duration_s"),
                    "rollout_s": ev.get("rollout_s"),
                    **(report.get("timings") or {})},
    }
    return row
