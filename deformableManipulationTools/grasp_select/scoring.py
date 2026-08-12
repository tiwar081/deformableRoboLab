"""The four score terms, their weights, and the temperature sampler.

Every term is a COST in ``[0, 1]``, lower is better, and every term reports whether it actually
KNEW anything. That second part carries most of the design: three of the four inputs are optional in
practice — physics quality is null on every record today (no evaluation pass has run), semantic
regions exist for some assets and are correctly empty for most, and containment can be absent on an
in-memory candidate that never went through ``pad_seat`` (on disk the schema requires it). A scorer
that treated absence as badness would rank assets against each other by how much annotation they
happen to have, which is not a property of the grasps. So an unknown term scores :data:`NEUTRAL`,
is flagged, and is COUNTED in the result's stats — the absence is visible, and it never silently
reorders anything, because a constant added to every candidate of an asset cannot change their
order.

The weighted mean is over the terms' weights regardless of knownness, again so that ranking within
one asset is unaffected: renormalising to only the known terms would let one candidate's known
region hit be diluted differently from another's, which is a ranking change caused by bookkeeping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..grasp_library import QUALITY_FIELDS
from .projection import geodesic_angle

# What an unknown term scores. 0.5, not 0 and not 1: an unmeasured grasp is neither proven good nor
# proven bad, and the value is identical for every candidate that shares the gap.
NEUTRAL = 0.5

# Physics-quality normalisation [m] / [rad] — the motion at which a metric counts as fully bad.
# Anchored on the shake protocol's own tolerances: a grasp that lets the object walk a jaw width
# during closing, or 2 cm during shaking, is not holding it.
QUALITY_SCALES = {
    "object_motion_during_closing_linear": 0.010,
    "object_motion_during_closing_angular": math.radians(20.0),
    "object_motion_during_shaking_linear": 0.020,
    "object_motion_during_shaking_angular": math.radians(30.0),
}

# Radius slack added to an annotated region when asking "is this grasp on it" [m]. A VLM pick is
# back-projected from image space and the store itself calls its balls "all the precision an
# image-space pick supports"; 25 mm is the couple of centimetres that buys, and it is deliberately
# smaller than the half-jaw-width (40 mm) the region store suggests, so a hit means the pads are on
# the feature rather than merely near it.
REGION_MARGIN = 0.025

# Containment is read from the candidate's ``seat_mode`` — a REQUIRED schema field since
# grasp_library SCHEMA_VERSION 2. The label/prose string-matching this replaced (2026-08-06) broke
# the moment a third mode (retreated) existed: prose worded per generator cannot carry an enum.


@dataclass(frozen=True)
class Weights:
    """Relative weight of each term. Defaults, and why they are not all equal:

    ``quality`` and ``region`` are evidence about the GRASP — did it hold, is it on the handle — so
    they lead. ``awkwardness`` and ``containment`` are about how comfortable the execution is; a
    contorted or shallow grasp is a real cost but it should not outrank a grasp that is known to
    work. These are a starting point for a downstream tuner, not a measured optimum, and nothing in
    the package depends on their exact values."""
    quality: float = 1.0
    region: float = 1.0
    awkwardness: float = 0.7
    containment: float = 0.5

    def total(self) -> float:
        return self.quality + self.region + self.awkwardness + self.containment


@dataclass(frozen=True)
class Term:
    """One score term: its cost, whether it was known, and a human-readable reason."""
    cost: float
    known: bool
    detail: str = ""

    @staticmethod
    def unknown(detail: str) -> "Term":
        return Term(cost=NEUTRAL, known=False, detail=detail)


def quality_term(candidate) -> Term:
    """Cost from the measured physics quality (ACRONYM's ``flex`` metric names).

    Null on every record today — validation has not run — so this degrades to NEUTRAL rather than
    treating "not measured" as "does not hold"."""
    q = {k: candidate.quality.get(k) for k in QUALITY_FIELDS} if candidate.quality else {}
    if not candidate.quality_source or all(v is None for v in q.values()):
        return Term.unknown("physics quality not measured")
    held = q.get("object_in_gripper")
    if held is not None and float(held) <= 0.0:
        return Term(cost=1.0, known=True, detail="object was not in the gripper")
    motions = [min(1.0, abs(float(q[k])) / s) for k, s in QUALITY_SCALES.items()
               if q.get(k) is not None]
    if not motions:
        return Term(cost=1.0 - float(held), known=True, detail=f"held={float(held):.2f}")
    cost = float(np.mean(motions))
    return Term(cost=cost, known=True,
                detail=f"{len(motions)} motion metric(s), mean normalised {cost:.2f}"
                       f" [{candidate.quality_source}]")


def region_term(candidate, doc, *, margin: float = REGION_MARGIN) -> Term:
    """Cost from proximity to the asset's annotated semantic regions.

    The join is the region store's own: candidate positions and region centres are both in the
    canonical frame, so no transform is involved — only the margin, which stands in for how coarse
    an image-space pick is. An asset with no annotation, or one honestly annotated as having no
    meaningful region, is UNKNOWN rather than uniformly bad.

    The cost is FLAT inside the region's ball and decays only across the margin outside it. Grading
    by distance within the ball would rank one grasp above another on sub-centimetre differences
    that a back-projected image pick cannot support — the store calls its balls "all the precision
    an image-space pick supports", and this respects that rather than mining it for a tiebreak."""
    if doc is None or not getattr(doc, "regions", ()):
        return Term.unknown("no annotated regions for this asset")
    p = np.asarray(candidate.position, dtype=float)
    best, best_d = None, float("inf")
    for r in doc.regions:
        d = float(np.linalg.norm(p - np.asarray(r.center, dtype=float)))
        if d < best_d:
            best, best_d = r, d
    reach = float(best.radius) + float(margin)
    inside = max(0.0, 1.0 - max(0.0, best_d - float(best.radius)) / max(margin, 1.0e-9))
    conf = min(1.0, max(0.0, float(best.confidence)))
    cost = 1.0 - inside * conf
    where = "on" if best_d <= reach else "off"
    return Term(cost=float(min(1.0, max(0.0, cost))), known=True,
                detail=f"{where} {best.label!r} ({best_d * 1000:.0f} mm from centre, "
                       f"r={best.radius * 1000:.0f} mm, conf={conf:.2f})")


def awkwardness_term(rotation, hand_rot) -> Term:
    """Cost from how far the wrist must turn from the robot's canonical hand pose.

    The geodesic SO(3) distance (see :mod:`.projection`) normalised by π, measured against the pose
    the arm will actually hold — the projected command, not the stored request, since that is the
    orientation the wrist has to reach."""
    theta = geodesic_angle(rotation, hand_rot)
    return Term(cost=float(theta / math.pi), known=True,
                detail=f"{math.degrees(theta):.0f} deg from the home hand pose")


def containment_term(candidate) -> Term:
    """Cost from how the pads hold the object, read off the schema's ``seat_mode`` field.

    ``span_flush`` — the whole measured span sits between the pads (flush against the fingertip
    end): the best hold, cost 0. ``clamped_deep`` — jaws as deep as safe, object protruding past
    the fingertips — and ``retreated`` — backed off along the approach until the hand cleared, so
    the pads hold only the near material — both fail the "fully enclosed" property this term
    measures and cost 1.
    (Which of the two is the stronger hold varies per candidate — a barely-retreated flush pose
    beats a deep clamp — and this term does not pretend to rank what it cannot see; the physics
    quality term is where that difference lands once measured.)

    ``seat_mode`` is required on disk (schema v2+), so UNKNOWN only occurs for in-memory
    candidates that never went through ``pad_seat``."""
    mode = getattr(candidate, "seat_mode", None)
    if mode == "span_flush":
        return Term(cost=0.0, known=True, detail="full span between the pads (tip-flush)")
    if mode == "centred":
        return Term(cost=0.0, known=True, detail="full span between the pads (centred)")
    if mode == "clamped_deep":
        # The record also carries `overhang` (how far the material protrudes past the fingertips,
        # merge-derived). It is TRACKED, deliberately NOT a scoring input — gating or grading on
        # it is a future, evidence-backed decision (grasp-library.md "Candidate statuses").
        return Term(cost=1.0, known=True,
                    detail="clamped deep — object protrudes past the fingertips")
    if mode == "retreated":
        return Term(cost=1.0, known=True,
                    detail="retreated to clear the hand — pads hold only the near material")
    return Term.unknown("seat_mode not set on this candidate")


@dataclass(frozen=True)
class Score:
    """The four terms and their weighted mean."""
    total: float
    terms: dict = field(default_factory=dict)
    weights: Weights = field(default_factory=Weights)

    @property
    def unknown_terms(self) -> tuple:
        return tuple(sorted(k for k, t in self.terms.items() if not t.known))

    def describe(self) -> str:
        return "  ".join(f"{k}={self.terms[k].cost:.2f}{'' if self.terms[k].known else '?'}"
                         for k in ("quality", "region", "awkwardness", "containment"))


def score_candidate(candidate, rotation, hand_rot, doc, weights: Weights = Weights(),
                    *, region_margin: float = REGION_MARGIN) -> Score:
    terms = {
        "quality": quality_term(candidate),
        "region": region_term(candidate, doc, margin=region_margin),
        "awkwardness": awkwardness_term(rotation, hand_rot),
        "containment": containment_term(candidate),
    }
    w = {"quality": weights.quality, "region": weights.region,
         "awkwardness": weights.awkwardness, "containment": weights.containment}
    total = sum(terms[k].cost * w[k] for k in terms) / max(weights.total(), 1.0e-12)
    return Score(total=float(total), terms=terms, weights=weights)


def sample_index(costs, temperature: float = 0.0, rng=None) -> int:
    """Pick one entry of ``costs``. ``temperature == 0`` is argmin; higher is softer.

    The softmin is ``p ∝ exp(-cost / T)``. Costs are already normalised to ``[0, 1]``, so T is in
    the same units: T = 0.05 is nearly greedy, T = 0.3 samples freely among comparable grasps. Ties
    at T = 0 resolve to the lowest index, which keeps a selection reproducible without an RNG."""
    c = np.asarray(costs, dtype=float)
    if c.size == 0:
        raise ValueError("no candidates to sample from")
    if temperature <= 0.0:
        return int(np.argmin(c))
    logits = -(c - c.min()) / float(temperature)
    p = np.exp(logits)
    p = p / p.sum()
    gen = rng if rng is not None else np.random.default_rng(0)
    return int(gen.choice(len(c), p=p))
