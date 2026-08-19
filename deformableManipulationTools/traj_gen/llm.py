"""The grasp-failure LLM loop: when the rolled-out grasp fails, ask a multimodal LLM to fix it.

Failed grasps are the trajectory stage's primary failure mode (the full-catalog shake holds 24.5%
even offline), so after a failed rollout the stage shows an LLM everything a human debugging the
grasp would look at — kept deliberately LIGHT (no extra ray-traced renders):

- the run's existing ``scene_overview.png`` still (scene layout, camera already paid for);
- a matplotlib grasp-attempt snapshot (:mod:`.viz`): the object's mesh with the attempted pad
  chords + approach arrow + the object's MEASURED path through the phases;
- the trajectory plan (phases, timing, routing stats) and the rollout's measurements (close-window
  drift, lift rise, where the object ended up, failure word);
- the attempted candidate's stored facts (seat mode, width, span, measured shake quality) and the
  TOP-K alternative candidates from the physics re-rank, each with its own facts.

The LLM answers with ONE structured action: ``switch`` to a named alternative candidate, or
``adjust`` the attempted grasp (small grasp-frame position offset, width, force target). It gets
**2 attempts** per trajectory (module constant ``MAX_LLM_ATTEMPTS``); if the second corrected
rollout still fails the trajectory is ABORTED — reported honestly, never papered over.

Transport = ``agentic_pipeline.scene_generator._messages_request`` (raw HTTPS + Claude Code OAuth,
JSON-schema output, model fallback) — the one place that knows how to authenticate, imported
lazily exactly like ``vlm_regions`` does.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

MAX_LLM_ATTEMPTS = 2          # corrected rollouts per trajectory; then abort
ALTERNATIVES_SHOWN = 8        # top-K re-ranked candidates offered for "switch"
MAX_SHIFT_MM = 20.0           # clamp on any adjust offset component
PROMPT_VERSION = 1

SYSTEM = """You are the grasp-recovery agent of a robot trajectory pipeline. A Franka parallel-jaw
gripper (max jaw width 80 mm) just FAILED to grasp an object during a simulated pick-and-place;
you are shown the scene, the attempted grasp, the measured outcome, and alternative candidates.
Decide ONE corrective action:

- "switch": try a DIFFERENT candidate — set candidate_id to one of the offered alternatives.
- "adjust": retry the SAME grasp pose with small corrections — any of: a position offset in the
  GRASP frame (millimetres; x = along the jaw axis, y = across the pads, z = along the approach,
  +z means DEEPER toward/into the object), a new pre-close jaw width (mm), a new force target (N).

Grasp-frame facts you need: the TCP sits at the fingertip tips; the finger pads extend from 0.7 mm
to 54.5 mm BEHIND the TCP along the approach, so material must sit within ~54 mm behind the TCP to
be between the pads; the palm limits seating deeper than 45 mm. The fingers pre-shape to
(width + 10 mm) before the approach. A close-window drift means the pads pushed the object out
(off-centre or too-shallow contact); a lift with no rise means the pads closed on too little
material or missed; a drop in transit means the hold was too weak (force target, thin contact).

Prefer "switch" to a measured-held alternative when one exists and its approach differs from the
failed one. Prefer "adjust" when the failure looks like a small offset or a too-light squeeze.
Be concrete and physical in the rationale (one or two sentences)."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["switch", "adjust"]},
        "candidate_id": {"type": "string",
                         "description": "for switch: one of the offered alternative ids"},
        "adjust": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # NOTE: the structured-output API rejects minItems > 1 (like maxItems — measured),
                # so "exactly 3" lives in the description and is enforced by padding after parse.
                "shift_mm": {"type": "array", "items": {"type": "number"},
                             "description": "EXACTLY 3 numbers: grasp-frame [jaw, lateral, "
                                            "approach] offset in mm"},
                "width_mm": {"type": "number", "description": "new pre-close jaw width in mm"},
                "force_target_n": {"type": "number", "description": "new grasp force target in N"},
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["action", "rationale"],
}


def _image_block(png: Path) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(png).read_bytes()).decode()}}


def _candidate_line(r, failed_approach=None) -> str:
    """One alternative, compact: id, physics tier, score, geometry (+ same-approach warning)."""
    g = r.grasp
    c = g.candidate
    q = getattr(c, "quality", None) or {}
    held = q.get("object_in_gripper")
    phys = ("HELD in shake test" if held is not None and float(held) >= 0.5 else
            "DROPPED in shake test" if held is not None else "not physics-tested")
    p = g.position
    same = ""
    if failed_approach is not None:
        a = np.asarray(g.pose)[:3, 2]
        if float(np.dot(a, failed_approach)) > 0.9:
            same = "  << SAME approach direction as the failed attempt — likely blocked the same way"
    return (f"- {g.id}: {phys}; cost {r.cost:.3f}; seat {getattr(c, 'seat_mode', '?')}; "
            f"width {float(c.width) * 1000:.0f} mm; span {float(getattr(c, 'span', 0)) * 1000:.0f} mm; "
            f"approach tilt {math.degrees(g.command.tilt):.0f} deg, yaw "
            f"{math.degrees(g.command.yaw):.0f} deg; world pos "
            f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}){same}")


def build_package(*, task: dict, plan: dict, evaluation: dict, pick, alternatives,
                  history: list, scene_png: Path | None, grasp_png: Path | None) -> list:
    """The user-message content blocks for one retry call."""
    goal = task.get("goal", {})
    lines = [
        f"TASK: {task.get('instruction', {}).get('default', task.get('name', '?'))}",
        f"GOAL: {goal.get('predicate')} {goal.get('params')}",
        "",
        f"ATTEMPTED GRASP (candidate {plan['pick']['id']}, attempt {plan['attempt']}):",
        f"  seat {plan['pick']['seat_mode']}, width {plan['pick']['width'] * 1000:.0f} mm, "
        f"tilt {plan['pick']['tilt_deg']:.0f} deg, yaw {plan['pick']['yaw_deg']:.0f} deg, "
        f"physics tier {plan['pick']['tier']} "
        f"(held={plan['pick'].get('quality_held')}), score {plan['pick']['score']}",
        f"  adjustment already applied: {plan['pick'].get('adjusted')}",
        f"  force target {plan['grasp_window']['force_target']} N, "
        f"pre-shape width {plan['grasp_window']['preshape_width'] * 1000:.0f} mm",
        "",
        "TRAJECTORY (phases in seconds): " + "; ".join(
            f"{p['name']} {p['t0']:.1f}-{p['t1']:.1f}" for p in plan["phases"]),
        f"  bezier routing: {plan['routing']}",
        "",
        "MEASURED OUTCOME of the rollout:",
        f"  failure = {evaluation.get('failure')}",
        f"  object drift during the close window: "
        f"{(evaluation.get('close_drift') or 0) * 1000:.0f} mm",
        f"  rise through the lift phase: {(evaluation.get('lift_rise') or 0) * 1000:.0f} mm "
        f"(held = {evaluation.get('held')})",
        f"  carried to the place point: {evaluation.get('carried')} "
        f"(xy error at carry end {(evaluation.get('carry_xy_err') or 0) * 100:.1f} cm)",
        f"  TCP distance from the planned grasp pose at close time: "
        f"{(evaluation.get('tcp_grasp_err') or 0) * 1000:.0f} mm "
        f"(large = the arm never reached the pose — an execution problem, not a contact one)",
        f"  final object position: {evaluation.get('final_pos')} "
        f"({(evaluation.get('final_z_above_table') or 0) * 100:.1f} cm above the table)",
    ]
    if history:
        lines += ["", "EARLIER ATTEMPTS this trajectory (do not repeat them):"]
        for h in history:
            lines += [f"  attempt {h['attempt']}: candidate {h['candidate_id']} "
                      f"adjust={h.get('adjusted')} -> {h['failure']} "
                      f"(close drift {(h.get('close_drift') or 0) * 1000:.0f} mm, "
                      f"lift rise {(h.get('lift_rise') or 0) * 1000:.0f} mm)"]
    failed_approach = None
    if evaluation.get("failure") == "approach_missed":
        failed_approach = np.asarray(pick.approach, dtype=float)
        lines += ["", "IMPORTANT: the arm was PHYSICALLY BLOCKED on this approach (the TCP "
                      "stopped short of the pose while the IK itself was feasible — something is "
                      "in the arm's way). Retrying a candidate with a SIMILAR approach direction "
                      "will fail the same way; switch to one whose approach/tilt differs "
                      "substantially (e.g. top-down instead of side, or the opposite side)."]
    lines += ["", f"ALTERNATIVE CANDIDATES (top {len(alternatives)} by physics-tiered score; "
                  f"'switch' must name one of these):"]
    lines += [_candidate_line(r, failed_approach) for r in alternatives]
    lines += ["", "The first image is the scene overview camera; the second is the attempted grasp "
                  "drawn on the object mesh (red = pad chords, blue arrow = approach, purple = the "
                  "object's measured path, green x = the place point)."]

    content: list = []
    if scene_png is not None and Path(scene_png).exists():
        content.append(_image_block(scene_png))
    if grasp_png is not None and Path(grasp_png).exists():
        content.append(_image_block(grasp_png))
    content.append({"type": "text", "text": "\n".join(lines)})
    return content


def retry_verdict(content: list, *, model: str | None = None, verbose: bool = True) -> dict:
    """One structured retry decision. Raises on transport failure (a missing verdict must abort the
    retry, never be invented)."""
    from agentic_pipeline import scene_generator as sg   # lazy: transport lives there

    use_model = model or sg.DEFAULT_MODEL
    messages = [{"role": "user", "content": content}]
    for attempt in range(2):
        try:
            # 20k tokens: a reasoning model can spend the default 8k THINKING before any text
            # (measured 2026-08-12 in the llm_retry pass — same transport, same failure mode).
            resp = sg._messages_request(SYSTEM, messages, use_model, SCHEMA, max_tokens=20000)
            verdict = json.loads(sg._response_text(resp))
            break
        except Exception as exc:                          # noqa: BLE001 - one model fallback, then raise
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[trajGen/llm] {use_model} failed ({exc}); "
                          f"falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
    verdict["model"] = use_model
    if verdict.get("action") == "adjust":
        adj = verdict.get("adjust") or {}
        shift = ([float(v) for v in (adj.get("shift_mm") or [])] + [0.0, 0.0, 0.0])[:3]
        adj["shift_mm"] = [max(-MAX_SHIFT_MM, min(MAX_SHIFT_MM, v)) for v in shift]
        if adj.get("width_mm") is not None:
            adj["width_mm"] = max(2.0, min(80.0, float(adj["width_mm"])))
        if adj.get("force_target_n") is not None:
            adj["force_target_n"] = max(1.0, min(40.0, float(adj["force_target_n"])))
        verdict["adjust"] = adj
    if verbose:
        print(f"[trajGen/llm] verdict: {verdict.get('action')} "
              f"{verdict.get('candidate_id') or verdict.get('adjust')} — "
              f"{verdict.get('rationale')}")
    return verdict


def apply_verdict(verdict: dict, pick, ranking, tried_ids) -> "object":
    """Turn a verdict into the next :class:`~.policy.PickSpec` (never re-trying a tried id)."""
    from dataclasses import replace

    from .policy import pick_from_ranked

    if verdict["action"] == "switch":
        r = ranking.by_id(str(verdict.get("candidate_id", "")))
        if r is None or r.id in tried_ids:
            r = next((x for x in ranking.ranked if x.id not in tried_ids), None)
            if r is None:
                return None
        return pick_from_ranked(r)
    adj = verdict.get("adjust") or {}
    shift = np.asarray(adj.get("shift_mm") or [0.0, 0.0, 0.0], dtype=float) / 1000.0
    pose = np.asarray(pick.pose, dtype=float).copy()
    pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ shift
    width = float(adj["width_mm"]) / 1000.0 if adj.get("width_mm") is not None else pick.width
    return replace(pick, pose=pose, width=width,
                   adjusted={"shift_mm": adj.get("shift_mm"), "width_mm": adj.get("width_mm"),
                             "force_target_n": adj.get("force_target_n"),
                             "rationale": verdict.get("rationale")})
