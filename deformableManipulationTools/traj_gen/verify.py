"""Post-trajectory VISUAL verification — did the world end up the way the task says?

The rollout's geometric goal check is exact where a driver exists, but it can mislabel what a
human would see (wrong role in a generated predicate, an unevaluable goal, an object teetering on
a rim) — so after the final video is rendered, a VLM looks at the FINAL frame (same RoboLab look
as ``scene_overview.png``) next to the initial scene still and answers: does the outcome match the
instruction?

On a mismatch the stage does NOT scrap the executed trajectory — an executed, physically valid
trajectory is training data for WHATEVER it actually did. The verdict carries (a) the ACHIEVED
instruction set (vague/default/specific) so the annotation can be RELABELED to the true outcome,
and (b) a bounded world-frame place nudge; the place-phase waypoints are shifted and the demo
re-executed, up to ``MAX_VISUAL_RETRIES`` (2) times per demo. Every executed round keeps its plan,
video, and (possibly relabeled) annotation row.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

MAX_VISUAL_RETRIES = 2        # tuned re-executions per demo after a visual mismatch
MAX_NUDGE_CM = 8.0            # clamp on the place-correction shift per axis
# Waypoints belonging to the place column (standoff / release / post-release hold) are the ones a
# nudge moves — identified by xy proximity to the plan's place point.
_PLACE_XY_TOL = 0.02

SYSTEM = """You are the outcome-verification agent of a robot manipulation pipeline. A simulated
Franka arm just executed a pick-and-place trajectory; you see the scene BEFORE (first image) and
AFTER (second image) the trajectory, plus the object coordinates measured by the simulator.
Judge whether the AFTER state satisfies the task instruction — as a human would judge it
(the right object in/on/beside the right place; direction words are from the ROBOT's point of
view, spelled out in the text).

Answer with:
- ok: true/false — does the outcome match the instruction?
- observed_outcome: one factual sentence saying where the manipulated object actually ended up.
- achieved_instruction (REQUIRED when ok=false): the instruction set this trajectory ACTUALLY
  performed — vague / default / specific phrasings, written exactly like task instructions
  ("Move the tuna can to the right of the spam can."). Honest relabeling turns a mis-executed
  demo into valid training data; describe what happened, never what was intended.
- nudge_cm (optional, only when ok=false and a small placement shift would fix it): {dx, dy} in
  WORLD centimetres (the coordinate frame of the numbers in the text) to move the RELEASE point.
- reasoning: one or two sentences."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "observed_outcome": {"type": "string"},
        "achieved_instruction": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"vague": {"type": "string"}, "default": {"type": "string"},
                           "specific": {"type": "string"}},
        },
        "nudge_cm": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}},
        },
        "reasoning": {"type": "string"},
    },
    "required": ["ok", "observed_outcome", "reasoning"],
}


def _img(png: Path) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(png).read_bytes()).decode()}}


def final_still(demo_py: Path, camera_hint: str = "overview") -> Path | None:
    """The LAST rendered still of the trajectory's advanced render (the final world state)."""
    from ..params import FRANKA
    frames = Path(demo_py).resolve().parent.parent.parent.parent \
        / "outputs" / FRANKA.short_name / Path(demo_py).stem / "frames"
    stills = sorted(p for p in frames.glob("*_*.png")
                    if camera_hint in p.name or "exterior" in p.name)
    if not stills:
        stills = sorted(frames.glob("*_*.png"))
    return stills[-1] if stills else None


def visual_verify(task: dict, evaluation: dict, placement: dict, before_png: Path | None,
                  after_png: Path, *, model: str | None = None, verbose: bool = True) -> dict:
    """One structured verification verdict over the before/after stills + measured coordinates."""
    from agentic_pipeline import geometry, scene_generator as sg   # lazy transport

    goal = task.get("goal", {})
    lines = [
        f"TASK INSTRUCTION: {task.get('instruction', {}).get('default', task.get('name'))}",
        f"  (specific phrasing: {task.get('instruction', {}).get('specific', '')})",
        f"GOAL PREDICATE: {goal.get('predicate')} {goal.get('params')}",
    ]
    if task.get("subgoals"):
        lines.append("SUBGOALS (multi-step task — judge the OVERALL instruction; every step "
                     "matters): " + "; ".join(f"{g.get('predicate')} {g.get('params')}"
                                              for g in task["subgoals"]))
    lines += [
        "",
        geometry.directions_text(placement),
        "",
        "SIMULATOR MEASUREMENTS at the end of the trajectory:",
        f"  manipulated object {evaluation.get('target')!r} final position: "
        f"{evaluation.get('final_pos')} "
        f"({(evaluation.get('final_z_above_table') or 0) * 100:.1f} cm above the tabletop)",
        f"  geometric goal check: {evaluation.get('goal', {})}",
    ]
    for seg in evaluation.get("segments") or []:
        if len(evaluation.get("segments") or []) > 1:
            lines.append(f"  step {seg['segment']} ({seg['target']}): final {seg['final_pos']}, "
                         f"failure={seg['failure']}")
    lines += [
        "",
        "The FIRST image is the scene BEFORE the trajectory; the SECOND is AFTER. Judge the "
        "outcome and answer per the schema.",
    ]
    content: list = []
    if before_png is not None and Path(before_png).exists():
        content.append(_img(before_png))
    content.append(_img(after_png))
    content.append({"type": "text", "text": "\n".join(lines)})

    use_model = model or sg.DEFAULT_MODEL
    for _ in range(2):
        try:
            resp = sg._messages_request(SYSTEM, [{"role": "user", "content": content}],
                                        use_model, SCHEMA, max_tokens=20000)
            verdict = json.loads(sg._response_text(resp))
            break
        except Exception as exc:                       # noqa: BLE001 - one model fallback
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[trajGen/verify] {use_model} failed ({exc}); "
                          f"falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
    verdict["model"] = use_model
    if verdict.get("nudge_cm"):
        n = verdict["nudge_cm"]
        verdict["nudge_cm"] = {"dx": max(-MAX_NUDGE_CM, min(MAX_NUDGE_CM, float(n.get("dx", 0)))),
                               "dy": max(-MAX_NUDGE_CM, min(MAX_NUDGE_CM, float(n.get("dy", 0))))}
    if verbose:
        print(f"[trajGen/verify] visual verdict: ok={verdict.get('ok')} — "
              f"{verdict.get('observed_outcome')} ({verdict.get('reasoning')})")
    return verdict


def tune_place(run_dir: Path, traj_name: str, dx: float, dy: float, *, round_no: int) -> dict:
    """Shift the plan's place column (standoff, release, post-release) by (dx, dy) metres and
    rewrite the traj file. Returns the updated plan dict."""
    p = Path(run_dir) / traj_name
    plan = json.loads(p.read_text())
    old = np.asarray(plan["place"]["xy"], dtype=float)
    for w in plan["waypoints"]:
        if abs(w["pos"][0] - old[0]) < _PLACE_XY_TOL and abs(w["pos"][1] - old[1]) < _PLACE_XY_TOL:
            w["pos"][0] = round(w["pos"][0] + dx, 4)
            w["pos"][1] = round(w["pos"][1] + dy, 4)
    plan["place"]["xy"] = [round(float(old[0] + dx), 4), round(float(old[1] + dy), 4)]
    plan["place"]["visual_tune"] = {"round": round_no, "dx": round(dx, 4), "dy": round(dy, 4)}
    p.write_text(json.dumps(plan, indent=1))
    return plan


def archive_round(run_dir: Path, tag: str, round_no: int) -> dict:
    """Preserve the executed round's plan + video under round-suffixed names (never scrapped)."""
    out = {}
    for src, kind in ((f"traj{tag}.json", "traj"), (f"trajectory{tag}.mp4", "video")):
        s = Path(run_dir) / src
        if s.exists():
            dst = Path(run_dir) / f"{s.stem}.r{round_no}{s.suffix}"
            dst.write_bytes(s.read_bytes())
            out[kind] = dst.name
    return out
