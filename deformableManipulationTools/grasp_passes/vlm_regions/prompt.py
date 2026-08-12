"""The VLM ask: six rendered views in, named grasp regions out.

The request goes through the SAME transport the scene generator's visual-verification stage uses
(``agentic_pipeline.scene_generator._messages_request`` — Claude Code OAuth credentials, JSON-schema
structured output, model fallback), so there is one place that knows how to authenticate and one
place that knows the response shape. The import is deliberately LAZY, inside the call: the
dependency between these packages runs the other way (the pipeline imports
``deformableManipulationTools``), and a module-level import here would make that a cycle and drag
the whole pipeline into every grasp-pass run.

The prompt's contract with the renderer is the view names and the image-fraction coordinate system;
both live in :mod:`.views`.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from .regions import REGION_LABELS
from .views import VIEW_NAMES

# Bumped when the wording, the schema, or the view set changes in a way that would change answers.
# Part of the cache key, so bumping it re-annotates every asset.
PROMPT_VERSION = 2

SYSTEM = """You are annotating 3D object assets for a robot manipulation dataset. The robot is a
Franka arm with a two-finger parallel gripper (maximum jaw opening 80 mm).

Your job is to point at SEMANTICALLY MEANINGFUL GRASP REGIONS: parts of an object that exist to be
held, or that a person would naturally take hold of. Handles, rims, necks, stems, spouts, knobs,
pull tabs, the shaft of a tool.

You are NOT proposing grasps. You do not choose an approach direction, a finger placement, or how
hard to squeeze — a later stage does that. You are labelling WHERE the meaningful parts are.

The single most important rule: MOST OBJECTS HAVE NO SUCH REGION, and for those the correct answer
is an empty list. A soup can, a sugar box, a rubber ball, a wooden block, an apple, a sponge — these
are grasped anywhere on the body, so they have no distinguished region. Reporting "the middle of the
can" as a region is WRONG: it is a forced guess, and a forced guess is worse than nothing here
because a downstream scorer will treat it as real evidence. Only report a region when you can name
the feature and it is clearly visible in the images. When in doubt, leave it out."""


def _ask_text(name: str, kind: str, extents_m, entry: dict) -> str:
    ex = [f"{100.0 * float(e):.1f}" for e in extents_m]
    return f"""These are {len(VIEW_NAMES)} rendered views of ONE object asset from the catalog, shown in this
order: {', '.join(VIEW_NAMES)}. Each image is labelled with its view name in the bottom-left corner.

Object: {name!r} (catalog kind {kind!r})
Bounding-box size: {ex[0]} x {ex[1]} x {ex[2]} cm (longest to shortest axis)
Catalog description: {entry.get('description') or entry.get('label') or '(none)'}

Each view is the SAME object turned to face you from a different side, under identical lighting, so
whatever faces the camera is always well lit — including the inside of any opening. 'front' and
'back' are opposite sides of the object, as are 'left'/'right' and 'top'/'bottom'. Because the
object is turned rather than orbited, a feature can appear at a different place and orientation in
each view; identify it afresh in each one rather than assuming it stays put.

Look carefully at whether a face is OPEN or CLOSED. A container's opening shows its interior wall
as a ring of shading inside the outline, and often the inner floor deeper in; a solid end is an
unbroken surface. Getting this backwards — calling a closed base a rim — is a serious error.

Each image has a faint 10x10 grid drawn over it with its lines labelled 0.1 ... 0.9. Use it to read
off positions: (u, v) are fractions of the image, u across from the LEFT edge (0.0) to the RIGHT
edge (1.0), v down from the TOP edge (0.0) to the BOTTOM edge (1.0).

First decide: does this object have any semantically meaningful grasp region at all? If it is a
plain body a gripper would take anywhere — a can, a box, a ball, a block, a piece of fruit — answer
has_regions=false, regions=[], and say why in empty_reason. That is a normal and frequent answer.

If it DOES have such regions, report each one:
  * label   — one of: {', '.join(REGION_LABELS)}. Use 'other' only for a real named feature that
              none of the rest fit, and name it in 'detail'.
  * picks   — point AT the feature in every view where you can see it clearly, one pick per view,
              at least one and up to {len(VIEW_NAMES)}. Put the pick ON the object's surface at the
              feature, not on the background and not on a different part. Picks in several views are
              much better than one: each is traced onto the 3D surface and they are combined, so
              several views average out an inaccurate pick, while a single one stands alone.
  * confidence — 0..1, how sure you are that this feature is really there and really is that part.
  * rationale  — one short sentence: what you can see that makes this a {REGION_LABELS[0]}/rim/etc.

Do not report the same physical feature twice. Do not report a region you cannot see in any view.
Do not invent a feature because the object "should" have one."""


def _schema() -> dict:
    # Numeric bounds are stated in the prompt and enforced in code, never in the schema: the
    # structured-output endpoint rejects `minimum`/`maximum` (and array `minItems`/`maxItems`).
    # An out-of-range pick is dropped by the back-projection, which is the same outcome.
    pick = {
        "type": "object",
        "properties": {
            "view": {"type": "string", "enum": list(VIEW_NAMES)},
            "u": {"type": "number", "description": "0.0 at the left image edge, 1.0 at the right"},
            "v": {"type": "number", "description": "0.0 at the top image edge, 1.0 at the bottom"},
        },
        "required": ["view", "u", "v"],
        "additionalProperties": False,
    }
    region = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(REGION_LABELS)},
            "detail": {"type": "string", "description": "short free-text name, required for 'other'"},
            # No minItems/maxItems: the structured-output endpoint rejects array bounds. The prompt
            # states the limits and _annotate enforces what matters (a region with no pick that hit
            # the mesh is dropped), so the schema does not need to.
            "picks": {"type": "array", "items": pick},
            "confidence": {"type": "number", "description": "0.0 (guess) to 1.0 (certain)"},
            "rationale": {"type": "string"},
        },
        "required": ["label", "detail", "picks", "confidence", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "has_regions": {"type": "boolean"},
            "empty_reason": {"type": "string",
                             "description": "why there is no meaningful region; '' when there are"},
            "regions": {"type": "array", "items": region},
        },
        "required": ["has_regions", "empty_reason", "regions"],
        "additionalProperties": False,
    }


def _image_block(png: Path) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(png).read_bytes()).decode()}}


def ask_regions(name: str, kind: str, extents_m, entry: dict, views, *,
                model: str | None = None, verbose: bool = True) -> tuple[dict, str]:
    """Show the rendered views to the VLM and return ``(answer, model_used)``.

    Raises on transport failure rather than returning an empty answer: "this asset has no regions"
    and "the annotator could not be reached" must not collapse into the same stored result, since
    the first is cached as a finding and the second would be cached as a lie."""
    from agentic_pipeline import scene_generator as sg          # lazy: see the module docstring

    content = [_image_block(v.png) for v in views]
    content.append({"type": "text", "text": _ask_text(name, kind, extents_m, entry)})
    messages = [{"role": "user", "content": content}]
    schema = _schema()

    wanted = model or sg.DEFAULT_MODEL
    last: Exception | None = None
    for use_model in (wanted, sg.FALLBACK_MODEL):
        try:
            resp = sg._messages_request(SYSTEM, messages, use_model, schema)
            answer = json.loads(sg._response_text(resp))
            if verbose:
                n = len(answer.get("regions", []))
                print(f"  {use_model}: {n} region(s)"
                      + (f" — {answer.get('empty_reason')}" if not n else ""))
            return answer, use_model
        except Exception as exc:                 # transport, refusal, or malformed JSON
            last = exc
            if verbose:
                print(f"  {use_model} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"grasp-region annotation failed for {name!r}: {last}")
