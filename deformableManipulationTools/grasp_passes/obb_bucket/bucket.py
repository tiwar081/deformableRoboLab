"""Source-agnostic OBB face bucketing: which face of the canonical box a grasp enters.

A bucket is a property of the POSE, not of the pass that produced it — the only input is the
approach column of the transform and the canonical frame it is expressed in. So one function buckets
a hand-placed fixture, a face-aligned box grasp and (later) an antipodal or learned candidate
identically, and the answer is defined for poses that no generator would emit.

The primary bucket always comes from :meth:`ObjectFrame.face_of` rather than a reimplementation:
``grasp_library.validate_record`` re-derives every stored ``face`` with that method, so a second
nearest-axis rule here could only ever disagree with the schema.

**Two kinds of ambiguity, deliberately kept apart.**

*Frame-ambiguous* (``frame.ambiguous``) — the OBB's orientation is not intrinsically determined
(extents tie, a sign came from the asset's authoring rather than from mesh asymmetry, or the box
optimum is near-degenerate as it is for any body of revolution). Behaviour: **bucket normally, then
label the candidate** ``face_ambiguous``. Reason: the frame is still bit-reproducible for THIS mesh,
and every consumer — the generator, the record, and the runtime pruner — composes that same stored
frame with the placement, so the bucket is geometrically exact for this record. What ambiguity costs
is MEANING across re-exports (a re-scanned mug may land in a different frame, and then these labels
describe a different face), not correctness here. Merging or suppressing buckets on an ambiguous
frame would throw away a partition that is real and useful, and would silently disagree with the
``face`` the schema already stores. The label is what tells a consumer not to read intent into it.

*Bucket-borderline* (a diagonal approach) — the approach is not close to any face normal, e.g. a
45 deg pose entering the +z/+y edge. ``face_of`` takes an argmax, so it answers with the lowest
axis index on an exact tie; that answer is deterministic and schema-conforming, but the runner-up
face is just as true. Behaviour: **keep the argmax as the primary bucket** (it must equal the stored
``face``), and additionally record the runner-up(s) as ``face_alt:<bucket>`` plus a
``face_borderline`` flag. That is what lets the runtime pruner be safe: pruning is a coarse
pre-filter, so it drops a borderline candidate only when EVERY bucket it plausibly belongs to has
been ruled out.
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import FACE_BUCKETS, ObjectFrame

# A component within this fraction of the largest one does not decide the face. 0.05 -> a pose is
# borderline once it comes within ~1.4 deg of an exact 45 deg edge on two equal extents, and any
# genuine face-aligned or near-face-aligned grasp is far outside it.
BORDERLINE_RTOL = 0.05

LABEL_PREFIX = "face:"
ALT_PREFIX = "face_alt:"
BORDERLINE_LABEL = "face_borderline"
AMBIGUOUS_LABEL = "face_ambiguous"


def _unit(approach) -> np.ndarray:
    a = np.asarray(approach, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1.0e-12:
        raise ValueError("approach direction is degenerate")
    return a / n


def bucket_of(frame: ObjectFrame, approach) -> str:
    """The face bucket an approach direction enters. Thin alias for :meth:`ObjectFrame.face_of`,
    kept so callers bucket through this module rather than reimplementing the rule."""
    return frame.face_of(approach)


def plausible_buckets(frame: ObjectFrame, approach, *, rtol: float = BORDERLINE_RTOL) -> tuple:
    """Every face bucket the approach could reasonably be said to enter, primary FIRST.

    One entry for a clean approach; two (or, for a body-diagonal, three) when the approach is
    borderline between faces. The primary is always :func:`bucket_of`."""
    a = _unit(approach)
    primary = frame.face_of(a)
    mag = np.abs(a)
    cut = (1.0 - rtol) * float(mag.max())
    out = [primary]
    for k in np.argsort(-mag):
        if mag[k] < cut:
            break
        face = f"{'-' if a[k] > 0 else '+'}{'xyz'[int(k)]}"
        if face not in out:
            out.append(face)
    return tuple(out)


def face_labels(frame: ObjectFrame, approach, *, rtol: float = BORDERLINE_RTOL) -> tuple:
    """The bucket labels for one grasp: ``face:<primary>``, then ``face_borderline`` +
    ``face_alt:<bucket>`` for a diagonal approach, then ``face_ambiguous`` on an ambiguous frame.

    Order is fixed so the labels of two runs (and of two candidates) diff cleanly."""
    buckets = plausible_buckets(frame, approach, rtol=rtol)
    labels = [f"{LABEL_PREFIX}{buckets[0]}"]
    if len(buckets) > 1:
        labels.append(BORDERLINE_LABEL)
        labels.extend(f"{ALT_PREFIX}{b}" for b in buckets[1:])
    if frame.ambiguous:
        labels.append(AMBIGUOUS_LABEL)
    return tuple(labels)


def buckets_from_labels(labels, fallback: str | None = None) -> tuple:
    """Read back what :func:`face_labels` wrote: ``(primary, *alternates)``.

    ``fallback`` (a candidate's stored ``face``) is used when the labels carry no ``face:`` entry —
    i.e. when the bucket pass has not run over that candidate. Nothing downstream should have to
    care whether the labels are present, only whether alternates are known."""
    primary, alts = None, []
    for lab in labels or ():
        if lab.startswith(LABEL_PREFIX) and not lab.startswith(ALT_PREFIX):
            primary = lab[len(LABEL_PREFIX):]
        elif lab.startswith(ALT_PREFIX):
            alts.append(lab[len(ALT_PREFIX):])
    primary = primary or fallback
    if primary is None:
        return tuple(alts)
    return (primary, *[a for a in alts if a != primary])


def is_bucket(label: str) -> bool:
    return label in FACE_BUCKETS
