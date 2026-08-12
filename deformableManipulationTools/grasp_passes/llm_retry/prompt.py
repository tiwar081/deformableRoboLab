"""The LLM ask: object context + tried-grasp history in, up to ten NEW grasp poses out.

Transport is ``vlm_regions.prompt``'s, verbatim in shape: one multimodal call through
``agentic_pipeline.scene_generator._messages_request`` (Claude Code OAuth, JSON-schema structured
output, ``DEFAULT_MODEL`` → ``FALLBACK_MODEL``), with the import LAZY inside the call — the
dependency between the packages runs the other way, and a module-level import would create a cycle
and drag the whole pipeline into every grasp-pass run.

Schema discipline (learned the hard way in ``vlm_regions``): the structured-output endpoint 400s
on JSON-schema ``minimum``/``maximum``/``minItems``/``maxItems``. NONE appear here — the prompt
prose states "exactly 10" and the numeric bounds, and the pass enforces everything in code
(truncate past 10, accept fewer, drop out-of-range widths at authoring, raise on zero valid).

The prompt's frame contract: every position the model returns is in METRES in the CANONICAL object
frame — the same frame the view renders pose and the tried-candidate report is written in — and
the hand-geometry paragraph is generated from ``grasp_library``'s measured constants, never
hand-typed numbers.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ...grasp_library import (MAX_JAW_WIDTH, PAD_FAR_Z, PAD_LENGTH, PAD_NEAR_Z, PALM_NEAR_Z,
                              PREGRASP_MARGIN, SEAT_DEEPEST_Z)
from ..vlm_regions.regions import REGION_LABELS
from ..vlm_regions.views import VIEW_NAMES

# Bumped when the wording, the schema, or the content package changes in a way that would change
# answers. Part of the state-cache key, so bumping it RE-ARMS the retry for every asset.
# v2 (2026-08-12): v1's first live round (banana_soft) was measured invalid as an attempt — the
# primary model died at the transport's 8000 max_tokens with NO text (budget spent reasoning) and
# the fallback hovered 9/10 TCPs ABOVE the material ("gripping air" authoring drops). v2 raises
# the token budget (LLM_MAX_TOKENS below) and adds the worked TCP-past-the-surface example to
# _hand_text. The bump deliberately re-arms banana_soft: its consumed round was an infrastructure
# failure, not a fair blind attempt.
PROMPT_VERSION = 2

# Token budget for the generation call. Reasoning models think before they emit; 8000 was measured
# (2026-08-12) to be entirely consumed by thinking on this prompt, failing the call with no text.
LLM_MAX_TOKENS = 20000

# What a candidate's semantic label may be: the shared region vocabulary plus "body" for a grasp on
# the undistinguished body of the object. The generating call stamps the label itself (the design's
# deliberate no-second-annotator choice — docs/trajPipeline/llm-retry.md "Annotation").
ALLOWED_LABELS = tuple(REGION_LABELS) + ("body",)

# How many candidates a round may emit. Stated as "exactly 10" in the prose; enforced (<=) in code.
MAX_CANDIDATES = 10

SYSTEM = """You are the LAST-RESORT grasp proposer for a robot manipulation dataset. The robot is a
Franka arm with a two-finger parallel gripper. You are only consulted for objects on which EVERY
automatic grasp generator has already run and physics validation has confirmed that NOT ONE of the
generated grasps holds. Your proposals go back through the exact same physics validation: a
free-floating gripper closes on the pose with force control, gravity switches on, the hand shakes,
and the object either stays in the jaws or it does not.

You will be shown the object from six sides, its exact measured geometry and hand constraints, and
EVERYTHING that was already tried with how each attempt failed. Study the failures before
proposing: repeating a failed pose with cosmetic changes wastes one of only two rounds this object
will ever get.

Propose grasps a careful engineer would: place the jaw axis across a locally thin, roughly
parallel-sided section that fits the jaw stroke; put real material between the pads, seated toward
the palm rather than at the fingertip tips; avoid approaches that force the hand through other
parts of the object. Precision matters — your numbers are executed literally, with no correction
step."""


def _image_block(png: Path) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(png).read_bytes()).decode()}}


def _frame_text(name: str, kind: str, entry: dict, frame) -> str:
    ext = np.asarray(frame.extents, dtype=float)
    ext_m = ", ".join(f"{e:.4f}" for e in ext)
    ext_cm = ", ".join(f"{100 * e:.1f}" for e in ext)
    ambiguous = ("\nNOTE: this object's canonical frame is AMBIGUOUS (its bounding box is "
                 "near-degenerate under rotation), so do not read meaning into fine orientation "
                 "about the long axis — rely on what you see in the images." if frame.ambiguous
                 else "")
    cfg = entry.get("config", {})
    mass = cfg.get("target_mass")
    mu = cfg.get("mu", cfg.get("soft_contact_mu"))
    props = ", ".join(filter(None, [
        f"mass {mass} kg" if mass is not None else None,
        f"density {cfg.get('density')} kg/m^3" if mass is None and cfg.get("density") else None,
        f"friction mu {mu}" if mu is not None else None,
        f"catalog dims (AABB) {entry.get('dims')} m" if entry.get("dims") else None,
        "soft/deformable (FEM) body" if kind in ("soft_mesh", "soft_block") else None,
    ]))
    return f"""Object: {name!r} (catalog kind {kind!r}, class {entry.get('class', '?')!r})
Catalog entry: {props}
Description: {entry.get('description') or entry.get('label') or '(none)'}

CANONICAL FRAME — every position you return, and every position in the data below, uses it:
origin at the centre of the object's oriented bounding box; +x along the LONGEST extent, +y the
middle, +z the SHORTEST; axis signs are fixed by mesh asymmetry and are NOT guaranteed to point
"up" (for some objects the rendered top view looks upside down — trust the images).
OBB extents: [{ext_m}] m  =  [{ext_cm}] cm  (x, y, z).
ALL positions you return must be in METRES in this frame.{ambiguous}"""


def _hand_text() -> str:
    return f"""THE HAND, measured (all along the approach axis, relative to the TCP — the grasp-frame origin
your `position` places):
- The TCP sits at the fingertip TIPS. The finger pads span z = {PAD_NEAR_Z * 1000:.1f} mm to
  {PAD_FAR_Z * 1000:.1f} mm — the ENTIRE pad lies BEHIND the TCP (pad length {PAD_LENGTH * 1000:.1f} mm).
- The palm's forward face is at z = {PALM_NEAR_Z * 1000:.1f} mm. The nearest object material in the
  jaw column may sit no deeper than z = {SEAT_DEEPEST_Z * 1000:.1f} mm (deeper buries the palm in
  the object).
- So: choose `position` such that the material you want to hold starts between
  {SEAT_DEEPEST_Z * 1000:.0f} mm and 0 mm behind the TCP along the approach. A TCP short of the
  material grips air (rejected); material deeper than {SEAT_DEEPEST_Z * 1000:.0f} mm is rejected too.
- THE #1 ERROR TO AVOID (it rejected 9 of 10 candidates in a prior round): placing the TCP ON or
  ABOVE the surface you approach, as if `position` were a hover point before the grasp. It is NOT —
  `position` is the FINAL pose of the fingertip tips, so the tips must have PASSED the first
  material by however much you want between the pads. WORKED EXAMPLE: an object occupies
  z in [-18, +18] mm and you pinch it top-down (approach = [0,0,-1]). The first material met is
  the top surface z = +18 mm. To take 20 mm of it between the pads, put the TCP 20 mm PAST that
  surface: position z = 18 - 20 = -2 mm (0.018 - 0.020 = -0.002 m). Setting position z = +35 mm
  would leave every bit of material in front of the tips — gripping air, rejected. The same
  arithmetic applies along any approach axis: TCP coordinate = (first-surface coordinate along
  the approach) advanced INTO the object by your chosen depth (up to {-SEAT_DEEPEST_Z * 1000:.0f} mm).
- Jaw stroke: 0 < width <= {MAX_JAW_WIDTH * 1000:.0f} mm ({MAX_JAW_WIDTH} m). A grasped section
  wider than that cannot be closed on — do not propose it.
- PRE-SHAPED APPROACH CONTRACT: the gripper arrives already shaped to
  min(width + {PREGRASP_MARGIN * 1000:.0f} mm, {MAX_JAW_WIDTH * 1000:.0f} mm) before the final
  approach. The collision check and the physics trial both pose the fingers at that aperture, so
  lateral clearance is needed only for the pre-shaped profile, but any object bulge inside that
  profile along the approach path will (correctly) block the grasp.

POSE FIELDS you return, per candidate:
- `position` [x,y,z] metres, canonical frame: where the TCP goes.
- `approach` [x,y,z]: the direction the hand TRAVELS INTO the grasp (from wrist toward object).
  Need not be unit length; must not be parallel to `jaw_axis`.
- `jaw_axis` [x,y,z]: the direction the two fingers close along. It is orthogonalized against the
  approach for you.
- `width` metres: the jaw opening at contact — slightly more than the material thickness spanned.
- `label`: one of {', '.join(ALLOWED_LABELS)} ("body" = the undistinguished body).
- `rationale`: one short sentence — what this pose grips and why it should hold."""


def _views_text() -> str:
    return f"""The {len(VIEW_NAMES)} images above are rendered views of the object, in this order:
{', '.join(VIEW_NAMES)} (each labelled bottom-left). Each view is the SAME object turned to face
you under identical lighting, so whatever faces the camera is well lit, including the inside of any
opening. "front" presents the canonical +x face, "back" -x, "left" +y, "right" -y, "top" +z,
"bottom" -z. The faint 10x10 grid is a reading aid for judging proportions only — your answer is in
3D metres, not image coordinates."""


def _regions_text(name: str) -> str:
    from ..vlm_regions.regions import load_regions
    try:
        doc = load_regions(name)
    except Exception:                          # noqa: BLE001 - a malformed store must not kill the ask
        doc = None
    if doc is None:
        return ""
    if not doc.regions:
        return ("\nSEMANTIC REGIONS: a prior annotation pass found no semantically distinguished "
                f"grasp region on this object ({doc.empty_reason or 'plain body'}).")
    lines = ["\nSEMANTIC REGIONS previously identified on this object (canonical frame, metres):"]
    for r in doc.regions:
        c = ", ".join(f"{float(x):.3f}" for x in r.center)
        lines.append(f"  - {r.id} ({r.label}): centre [{c}], radius {r.radius:.3f} m, "
                     f"confidence {r.confidence:.2f} — {r.rationale}")
    return "\n".join(lines)


def _ask_text(round_key: str) -> str:
    stage = ("This is ROUND A — your first and second-to-last chance for this object."
             if round_key == "a" else
             "This is ROUND B — the FINAL round for this object, informed by the round-A feedback "
             "above. If these candidates also fail, the object is marked unusable.")
    return f"""{stage}

Propose EXACTLY {MAX_CANDIDATES} grasp candidates. Make them genuinely diverse — different features,
faces, approaches, and widths — and make every one geometrically deliberate: check it against the
extents and the hand constraints before you write it down. Do not repeat a pose that already
failed above unless you change what actually made it fail, and say so in the rationale."""


def round_a_content(asset, views, report: str, *, regions_text: str | None = None) -> list:
    """The full round-A message content: six view images + one context/ask text block."""
    content = [_image_block(v.png) for v in views]
    text = "\n\n".join(filter(None, [
        _views_text(),
        _frame_text(asset.name, asset.kind, asset.entry, asset.frame),
        _hand_text(),
        report,
        (regions_text if regions_text is not None else _regions_text(asset.name)).strip(),
        _ask_text("a"),
    ]))
    content.append({"type": "text", "text": text})
    return content


def round_b_content(asset, views, report: str, fb_images: list, fb_report: str) -> list:
    """Round B: the round-A context, then one labelled marker image per tried round-A candidate
    with the per-candidate failure report."""
    content = [_image_block(v.png) for v in views]
    context = "\n\n".join(filter(None, [
        _views_text(),
        _frame_text(asset.name, asset.kind, asset.entry, asset.frame),
        _hand_text(),
        report,
        _regions_text(asset.name).strip(),
        "Below: one image per candidate YOU proposed in round A, drawn as a gripper marker on the "
        "object (three orthographic panels of the canonical frame), each titled with its id and "
        "outcome, followed by the full failure report.",
    ]))
    content.append({"type": "text", "text": context})
    for png in fb_images:
        content.append({"type": "text", "text": f"Round-A candidate {Path(png).stem}:"})
        content.append(_image_block(png))
    content.append({"type": "text", "text": fb_report + "\n\n" + _ask_text("b")})
    return content


def _schema() -> dict:
    # No minimum/maximum/minItems/maxItems ANYWHERE: the structured-output endpoint rejects each
    # of them with a 400 (measured, one by one, in vlm_regions). Bounds live in the prose; the
    # authoring validation enforces what matters in code.
    def vec(desc):
        return {"type": "array", "items": {"type": "number"},
                "description": f"{desc} — exactly 3 numbers [x, y, z], canonical frame"}
    cand = {
        "type": "object",
        "properties": {
            "position": vec("TCP position in METRES"),
            "approach": vec("direction the hand travels into the grasp"),
            "jaw_axis": vec("direction the fingers close along"),
            "width": {"type": "number",
                      "description": f"jaw opening at contact [m]; 0 < width <= {MAX_JAW_WIDTH}"},
            "label": {"type": "string", "enum": list(ALLOWED_LABELS)},
            "rationale": {"type": "string", "description": "one short sentence"},
        },
        "required": ["position", "approach", "jaw_axis", "width", "label", "rationale"],
        "additionalProperties": False,
    }
    return {"type": "object",
            "properties": {"candidates": {"type": "array", "items": cand,
                                          "description": f"exactly {MAX_CANDIDATES} candidates"}},
            "required": ["candidates"], "additionalProperties": False}


def ask_candidates(name: str, content: list, *, model: str | None = None,
                   verbose: bool = True) -> tuple[dict, str]:
    """One multimodal structured call; returns ``(answer, model_used)``.

    RAISES on transport failure / refusal / malformed JSON rather than returning an empty answer:
    "the LLM proposed nothing" must never be cached as a finding (the caller caches only after
    this returns). Model fallback mirrors ``vlm_regions.prompt.ask_regions``."""
    from agentic_pipeline import scene_generator as sg          # lazy: see the module docstring

    messages = [{"role": "user", "content": content}]
    schema = _schema()
    wanted = model or sg.DEFAULT_MODEL
    last: Exception | None = None
    for use_model in (wanted, sg.FALLBACK_MODEL):
        try:
            resp = sg._messages_request(SYSTEM, messages, use_model, schema,
                                        max_tokens=LLM_MAX_TOKENS)
            answer = json.loads(sg._response_text(resp))
            n = len(answer.get("candidates") or [])
            if verbose:
                print(f"  {use_model}: {n} candidate(s) proposed")
            return answer, use_model
        except Exception as exc:                 # transport, refusal, or malformed JSON
            last = exc
            if verbose:
                print(f"  {use_model} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"llm_retry generation failed for {name!r}: {last}")
