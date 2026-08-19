"""ONLINE grasp proposal for cloth sheets and cables — the LLM picks the grasp, physics judges it.

Cloths and cables have no offline grasp records (a settled sheet or coiled cable is nothing like
its rest shape), so the stage skips straight to a multimodal LLM: it sees the run's
``scene_overview.png``, the settled-material snapshot (:mod:`.deform_snapshot` — numbered world-
frame sample points, image + text grounding each other), the task, and any earlier attempts'
measured outcomes, and proposes ONE grasp — world position, approach, jaw axis, width, force.
The proposal becomes an ordinary :class:`~.policy.PickSpec`; the SAME plan machinery then builds
the trajectory spline about it (Bezier legs, collision-driven control points, straight validated
approach) and the same rollout measures it. **Up to 3 proposals per trajectory, with feedback in
between**; then the stage aborts honestly.

Physics facts baked into the prompt (from the cloth/cable demos, docs/physicsEngine):
- a flat SHEET is pinched by pressing the fingertips slightly INTO the support (the mu*N anchoring
  recipe) with a LOW force target (~2-4 N) — the admittance regulator converges to a stable pinch;
- a CABLE is gripped perpendicular to its local direction (jaw axis ~ along the cable), force
  ~15-30 N;
- the TCP is at the fingertip tips; pads extend 54 mm behind it; jaw stroke <= 80 mm.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

from ..grasp_library import MAX_JAW_WIDTH, grasp_transform
from ..grasp_select.projection import project_pose
from ..params import TABLE
from .policy import PickSpec

MAX_ONLINE_ATTEMPTS = 3       # LLM proposals per trajectory (feedback in between), then abort
PROMPT_VERSION = 1
# A proposal must grasp MATERIAL: within this distance of some sampled deformable point [m].
NEAR_MATERIAL = 0.10
# Sheet-press allowance: the fingertips may command this far below the tabletop [m] (the measured
# cloth recipe presses ~5 mm; the pad-sweep gate for deformables uses the same number).
PRESS_ALLOWANCE = 0.008

SYSTEM = """You are the grasp-planning agent of a robot trajectory pipeline, choosing where a
Franka parallel-jaw gripper (jaw width <= 80 mm; TCP at the fingertip tips, pads extending 54 mm
behind them along the approach) should grasp a DEFORMABLE object — a cloth sheet or a cable —
lying on a table in its settled (possibly folded/coiled) state. You are shown the scene camera
view, an orthographic snapshot of the settled material with numbered sample points, and those
points' world coordinates in metres. Propose ONE grasp:

- position: world [x, y, z] in metres — ON the material (near the sampled points). For a flat
  sheet on the table, grasp WELL INSIDE the fabric — 3-6 cm in from the nearest edge, so BOTH
  fingertips press on fabric (the pressing pads GATHER a wad of sheet between them as they slide
  closed; an edge/corner pinch closes half on bare table and captures NOTHING — measured, five
  edge pinches in a row all failed while the validated demo grasps 4 cm inside the torso). Pick
  the interior point on the side nearest where the material must go. Command z 4-6 mm BELOW THE
  TABLETOP plane itself (the tabletop z is given in the text; e.g. tabletop at 0.070 ->
  z = 0.064-0.066): the fingertip must press INTO the support so friction anchors the fabric
  while the pinch closes — a pinch at or just under the CLOTH surface closes on air.
- approach: unit [x, y, z] pointing INTO the material (top-down = [0, 0, -1]; slight tilts are
  fine, the arm handles up to 90 deg).
- jaw_axis: unit [x, y, z], the line the two fingertips close along. For a cable set it ALONG the
  local cable direction is WRONG — the jaws must close ACROSS the cable: set jaw_axis
  perpendicular to the cable's local direction at the grasp point. For a sheet any horizontal
  direction works; prefer one that keeps the jaw over material.
- width_mm: pre-close jaw width. Sheet: 8-20 mm (a thin pinch). Cable: cable diameter + 10-20 mm.
- force_n: grasp force target. Sheet/garment: 4-5 N (4 N is the MEASURED value for a knit shirt;
  at 2-3 N a wadded pinch under-engages and the lift leaves the sheet behind; above ~6 N the jaw
  grinds to zero gap and expels the sheet). Cable: 15-30 N.
- rationale: one or two sentences, concrete and physical.

Pick a grasp point that (a) is on accessible material (not buried under the object's own folds
where the fingers cannot reach), (b) leaves a clear vertical approach, and (c) suits the TASK —
e.g. to move the object into a container, grasp near an edge/end so most material hangs free."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "position": {"type": "array", "items": {"type": "number"},
                     "description": "EXACTLY 3 numbers: world grasp position [x, y, z] in metres"},
        "approach": {"type": "array", "items": {"type": "number"},
                     "description": "EXACTLY 3 numbers: unit approach direction, INTO the material"},
        "jaw_axis": {"type": "array", "items": {"type": "number"},
                     "description": "EXACTLY 3 numbers: unit closing direction of the jaws"},
        "width_mm": {"type": "number"},
        "force_n": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["position", "approach", "width_mm", "force_n", "rationale"],
}


def _img(png: Path) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(png).read_bytes()).decode()}}


def _vec3(v, default=(0.0, 0.0, -1.0)) -> np.ndarray:
    a = np.asarray(([float(x) for x in (v or [])] + list(default))[:3], dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else np.asarray(default, dtype=float)


def build_content(task: dict, snapshot: dict, history: list, scene_png: Path | None,
                  extra_pngs=()) -> list:
    pts = snapshot["points"]
    lines = [
        f"TASK: {task.get('instruction', {}).get('default', task.get('name', '?'))}",
        f"GOAL: {task.get('goal', {}).get('predicate')} {task.get('goal', {}).get('params')}",
        "",
        f"OBJECT: a settled {snapshot['kind']} — {snapshot['n_material_points']} material points, "
        f"xy extent {snapshot['extent_xy'][0]:.2f} x {snapshot['extent_xy'][1]:.2f} m, "
        f"z from {snapshot['z_min']:.3f} to {snapshot['z_max']:.3f} m "
        f"(tabletop at z = {TABLE.top_z:.3f} m"
        + (f"; a sheet pinch must command z = {TABLE.top_z - 0.006:.3f}-{TABLE.top_z - 0.004:.3f}"
           f" — pressed into the support, below the tabletop plane)"
           if snapshot["kind"] == "cloth" else ")") + ".",
        f"SAMPLED WORLD POINTS (metres; the red numbered points in the snapshot"
        + ("; cable nodes in CHAIN ORDER, so consecutive points give the local cable direction"
           if snapshot["kind"] == "cable" else "") + "):",
    ]
    lines += [f"  {k}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})" for k, p in enumerate(pts)]
    if history:
        lines += ["", "EARLIER ATTEMPTS this trajectory (do not repeat them; learn from the "
                      "measured outcome):"]
        for h in history:
            lines += [f"  attempt {h['attempt']}: grasp at {h['position']} width "
                      f"{h['width_mm']:.0f} mm force {h['force_n']:.1f} N -> {h['outcome']}"]
    lines += ["", "The first image is the scene camera; the next is the settled-material snapshot "
                  "(top + side views, numbered points, dashed line = tabletop)."
                  + (" Further images show earlier failed attempts drawn on the material."
                     if len(history) else ""),
              "Propose the grasp now."]
    content: list = []
    if scene_png is not None and Path(scene_png).exists():
        content.append(_img(scene_png))
    if snapshot.get("png") and Path(snapshot["png"]).exists():
        content.append(_img(snapshot["png"]))
    for p in extra_pngs:
        if p is not None and Path(p).exists():
            content.append(_img(p))
    content.append({"type": "text", "text": "\n".join(lines)})
    return content


def propose(content: list, *, model: str | None = None, verbose: bool = True) -> dict:
    """One structured grasp proposal (transport failure raises — never invent a grasp)."""
    from agentic_pipeline import scene_generator as sg   # lazy: transport lives there

    use_model = model or sg.DEFAULT_MODEL
    messages = [{"role": "user", "content": content}]
    for _ in range(2):
        try:
            resp = sg._messages_request(SYSTEM, messages, use_model, SCHEMA, max_tokens=20000)
            out = json.loads(sg._response_text(resp))
            out["model"] = use_model
            if verbose:
                print(f"[trajGen/deform] proposal: pos={out.get('position')} "
                      f"width={out.get('width_mm')}mm force={out.get('force_n')}N — "
                      f"{out.get('rationale')}")
            return out
        except Exception as exc:                          # noqa: BLE001 - one model fallback
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[trajGen/deform] {use_model} failed ({exc}); "
                          f"falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise


def pick_from_proposal(proposal: dict, snapshot: dict, attempt: int) -> tuple:
    """(PickSpec, None) for a geometrically valid proposal, else (None, why) for the feedback."""
    pos = np.asarray(([float(v) for v in proposal.get("position", [])] + [0.0] * 3)[:3])
    approach = _vec3(proposal.get("approach"), (0.0, 0.0, -1.0))
    if approach[2] > -0.1:
        return None, "approach must point DOWNWARD into the material (z component <= -0.1)"
    jaw = _vec3(proposal.get("jaw_axis"), (1.0, 0.0, 0.0))
    width = min(max(float(proposal.get("width_mm", 15.0)) / 1000.0, 0.004), MAX_JAW_WIDTH)
    force = min(max(float(proposal.get("force_n", 3.0)), 1.0), 40.0)

    pts = np.asarray(snapshot["points"], dtype=float)
    d = float(np.min(np.linalg.norm(pts - pos, axis=1)))
    if d > NEAR_MATERIAL:
        return None, (f"grasp position is {d * 100:.0f} cm from the nearest material point — "
                      f"it must be ON the object (within {NEAR_MATERIAL * 100:.0f} cm)")
    if pos[2] < TABLE.top_z - PRESS_ALLOWANCE:
        return None, (f"grasp z {pos[2]:.3f} is {(TABLE.top_z - pos[2]) * 1000:.0f} mm below the "
                      f"tabletop — at most {PRESS_ALLOWANCE * 1000:.0f} mm of press is physical")
    if pos[2] > snapshot["z_max"] + 0.05:
        return None, "grasp z is well above the material — the jaws would close on air"
    try:
        pose = grasp_transform(pos, approach, jaw)
    except ValueError as exc:
        return None, str(exc)
    cmd = project_pose(np.asarray(pose)[:3, :3])
    if not cmd.accepted:
        return None, (f"approach needs {math.degrees(cmd.distortion):.0f} deg more tilt than the "
                      f"arm can execute — use a more vertical approach")
    pick = PickSpec(id=f"llm_online/a{attempt}", pose=np.asarray(pose, dtype=float),
                    yaw=float(cmd.yaw), tilt=float(cmd.tilt), tilt_axis=tuple(cmd.tilt_axis),
                    width=width, seat_mode="llm", source="llm_online", tier=1, score=0.5,
                    adjusted={"force_target_n": force,
                              "rationale": proposal.get("rationale", "")})
    return pick, None


def outcome_line(evaluation: dict | None, authoring_reason: str | None) -> str:
    """One line of feedback per attempt for the next proposal round."""
    if authoring_reason:
        return f"REJECTED before rollout: {authoring_reason}"
    e = evaluation or {}
    if e.get("held") and not e.get("carried"):
        return (f"grasped and lifted {(e.get('lift_rise') or 0) * 100:.1f} cm but LOST in "
                f"transit (failure {e.get('failure')})")
    if not e.get("held"):
        return (f"grasp FAILED ({e.get('failure')}): tracked material point moved "
                f"{(e.get('close_drift') or 0) * 1000:.0f} mm during the close and rose only "
                f"{(e.get('lift_rise') or 0) * 1000:.0f} mm in the lift")
    return f"carried but goal not met: {e.get('goal', {}).get('detail')}"
