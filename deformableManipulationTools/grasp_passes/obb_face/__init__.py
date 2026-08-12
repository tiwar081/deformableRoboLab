"""``obb_face`` pass — face-aligned box grasps from the canonical OBB.

The cheapest honest generator there is: the canonical frame already IS an oriented bounding box, so
for each of its six faces the jaw can be aligned to one of the two in-plane axes and driven straight
in along the inward normal. Six faces x two jaw axes x a few offsets along the third axis gives a
coverage GRID over the object's box — no sampling, no randomness, and every pose is exactly
axis-aligned in ``ctx.frame``, which is what makes the face bucket of the result exact rather than
nearest-axis (see the ``obb_bucket`` pass for what happens to poses that are not).

What it is not: an antipodal search. It reasons about the BOX, and the only thing it knows about the
mesh is the material actually found between the jaws (measured, never assumed from the extent) — so
on a mug or a bowl most of the grid is correctly rejected for not fitting the jaw, and what survives
are the grasps the box shape genuinely offers. Quality is left empty for an evaluation pass.

Geometry, per emitted candidate (all in the canonical frame):

* **approach** = the inward normal of the face, i.e. ``-n_face``; this is the candidate's face bucket.
* **jaw** = one of the two canonical axes lying in that face.
* **grasp centre** = the box centre plane along the approach, offset along the third axis by the
  sweep fractions, RE-CENTRED on the material actually found along the jaw axis, and finally
  PAD-SEATED along the approach by :func:`grasp_library.pad_seat` so the object sits between the
  fingers instead of on their tips (``POSE_CONVENTION`` v2). Depth along the approach is gripper
  geometry, so this pass does not choose it.
* **width** = the measured material span along the jaw axis at that point, plus :data:`CLEARANCE`.
  Over the jaw limit -> the candidate is dropped (never clipped: for a rigid object a clipped width
  would describe a grasp the gripper cannot make). The ONE exception is
  :data:`COMPRESSION_TOLERANCE` on a compressible kind — see below.

Compressible objects (:data:`COMPRESSIBLE_KINDS`) are allowed to be :data:`COMPRESSION_TOLERANCE`
wider than the jaw. The rejection exists because the jaws cannot open far enough; for a deformable
that is only true up to the compression the object actually accepts, and the grip is force-
controlled (``GraspWindow.force_target``), so the stored width is a starting opening rather than a
number the controller depends on. The tolerance is deliberately at the scale of the measurement's
own noise — a few percent, i.e. an object that IS jaw-width to within rounding — and NOT a claim
about how far foam squashes: 3 % passes a 82 mm foam brick and still rejects a 116 mm sponge and a
152 mm banana, which are not near-misses but genuinely-too-wide grasps. Such a candidate stores
``width = MAX_JAW_WIDTH`` (the widest the hand can actually open, so the record stays commandable
and inside the schema), carries the ``compressed`` label, and records the measured span and the
required squeeze in its notes — nothing downstream has to infer that compression was assumed.
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import (CENTRED_VARIANT_LABEL, CENTRED_VARIANT_SUFFIX,
                              MAX_JAW_WIDTH, SEAT_BLOCKED_LABEL, centred_variant,
                              make_candidate, pad_seat)
from ..base import GraspPass, PassContext, PassOutput

# RETIRED 2026-08-05: FINGER_DEPTH (40 mm inserted from the entry face, capped at the box centre
# plane). It aimed at "grip a fixed depth in from the face", which is a different goal from seating
# the object between the pads, and it did not achieve the latter — measured residual +27 to +89 mm
# across banana/mug candidates, and for anything thinner than ~2x the pad length the cap put the
# pose back on the material centre. Depth along the approach is now grasp_library.pad_seat(), once,
# from the hand's own pad geometry.
# Extra jaw opening on top of the measured span [m] — the same 6 mm the hand-placed fixtures use.
CLEARANCE = 0.006
# Half-size of the pad-sized column used to decide what material sits between the jaws [m].
PAD_HALF = 0.010
_MIN_SAMPLES = 8            # material samples a column needs before its span means anything
_MAX_GROWTH = 6             # column-growth attempts (mesh density spans 3 orders across the catalog)
_GROWTH = 1.6
# Sweep positions along the third axis, as fractions of that axis's half-extent. Symmetric about the
# centre so the grid does not favour either end of the object.
SWEEP_FRACTIONS = (-0.5, 0.0, 0.5)
# An offset closer than this to the centre one is not a distinct grasp — a short axis collapses the
# grid to its centre column instead of emitting three near-identical poses.
_MIN_OFFSET = 0.005
_MIN_SPAN = 0.001           # below this there is nothing between the pads worth calling a grasp
# Catalog kinds that yield under the pads. Deformables with a persistent rest shape only. `cloth`
# and (since 2026-08-11) `cable` are out of scope for the whole pipeline; while cables were still in
# scope, `cable` was absent here anyway — a rod's cross-section sits far inside the jaw (nothing to
# relax).
COMPRESSIBLE_KINDS = ("soft_mesh", "soft_block")
# How far over the jaw a compressible span may go and still be emitted (fraction of MAX_JAW_WIDTH).
# 3 % = 2.4 mm: the width of a measurement artefact, not of a squeeze — see the module docstring.
COMPRESSION_TOLERANCE = 0.03

_AXIS_NAME = "xyz"
_EYE = np.eye(3)


def _face_frames():
    """The 12 (face, jaw axis, sweep axis) combinations of a box, in a fixed order.

    Yields ``(face_label, k_face, sign, k_jaw, k_sweep)`` — ``k_*`` index the canonical axes."""
    for k in range(3):
        for sign in (1, -1):
            face = f"{'+' if sign > 0 else '-'}{_AXIS_NAME[k]}"
            for k_jaw in (i for i in range(3) if i != k):
                k_sweep = ({0, 1, 2} - {k, k_jaw}).pop()
                yield face, k, sign, k_jaw, k_sweep


def measure_span(canonical_vertices, point, jaw, approach, *, pad_half: float = PAD_HALF):
    """Material span along the jaw axis in a pad-sized column at ``point``.

    Returns ``(lo, hi, pad)`` as offsets from ``point`` along ``jaw``, or None when the column holds
    no material even after growing — an empty column means the grid point is in free space (the
    hollow of a mug, past the end of a banana), which is a candidate to DROP, not an error.

    The column is grown until it holds :data:`_MIN_SAMPLES`, for the same reason the fixture pass
    grows its own: catalog meshes range from ~10k-vertex YCB scans to 235-vertex tet meshes, and a
    fixed pad-sized query finds nothing at all inside the coarse ones. The span runs from the nearest
    to the furthest material in the column — the jaws must clear EVERYTHING between them, so an
    object whose walls straddle the column is honestly reported as too wide rather than pinched."""
    du = np.asarray(canonical_vertices, dtype=float) - np.asarray(point, dtype=float)
    third = np.cross(jaw, approach)
    a_off, t_off, j_off = du @ approach, du @ third, du @ jaw
    pad = float(pad_half)
    for _ in range(_MAX_GROWTH):
        m = (np.abs(a_off) <= pad) & (np.abs(t_off) <= pad)
        if int(m.sum()) >= _MIN_SAMPLES:
            j = j_off[m]
            return float(j.min()), float(j.max()), pad
        pad *= _GROWTH
    return None


class ObbFacePass(GraspPass):
    """Generate face-aligned box grasps for every supported catalog kind."""

    name = "obb_face"
    source = "obb_face"
    # v2 (2026-08-05): depth along the approach comes from grasp_library.pad_seat() instead of the
    # retired FINGER_DEPTH constant, so every stored pose moves — the cached v1 sidecars are stale.
    # v3 (2026-08-05): compressible kinds may exceed the jaw by COMPRESSION_TOLERANCE.
    # v4 (2026-08-06): uncontained poses use pad_seat's clamped deep-seat (see grasp_library).
    # v5 (2026-08-06): pad_seat's collision-aware retreat — a pose whose rule seat puts the hand
    # inside the object backs off to the deepest clear depth; candidates carry seat_mode (schema
    # v2), and a pose clear at no depth keeps its rule seat with the seat_blocked label.
    # v6 (2026-08-07): the span_flush seat rule (schema v3) — the retired centred rule
    # commanded the fingertips past the object's far surface (into the table for top-down
    # grasps on resting objects); candidates now carry the measured span.
    # v7 (2026-08-07): schema v4 — seat_depth stored per candidate, and the hand collision
    # checks pose the fingers at the PRE-SHAPED aperture (width + PREGRASP_MARGIN), not fully
    # open, matching the trajectory's pre-grasp; retreat/blocked outcomes move.
    # v8 (2026-08-07): schema v5 depth-variant pairs — a palm-safe span also stores its
    # CENTRED seat as a separate '_ctr' candidate; online selection picks between the pair.
    version = 8
    kinds = ()               # every supported kind: they all have an OBB
    requires = ()

    def run(self, ctx: PassContext) -> PassOutput:
        verts = ctx.asset.canonical_vertices()
        faces = ctx.asset.faces
        half = np.asarray(ctx.asset.half_extents, dtype=float)
        out, dropped_empty, dropped_wide, dropped_unseated, uncontained = [], 0, 0, 0, 0
        compressed, retreated, blocked = 0, 0, 0
        compressible = ctx.kind in COMPRESSIBLE_KINDS

        for face, k, sign, k_jaw, k_sweep in _face_frames():
            approach = -sign * _EYE[k]
            jaw = _EYE[k_jaw]
            # Start on the box centre plane along the approach. Depth along the approach is NOT this
            # pass's decision any more: pad_seat() sets it from the gripper's own pad geometry.
            base = np.zeros(3)
            for frac in self._offsets(float(half[k_sweep])):
                point = base + frac * float(half[k_sweep]) * _EYE[k_sweep]
                span = measure_span(verts, point, jaw, approach)
                if span is None:
                    dropped_empty += 1
                    continue
                lo, hi, pad = span
                if hi - lo < _MIN_SPAN:
                    dropped_empty += 1
                    continue
                width = span_width = (hi - lo) + CLEARANCE
                squeeze = 0.0
                if width > MAX_JAW_WIDTH:
                    if not (compressible
                            and width <= MAX_JAW_WIDTH * (1.0 + COMPRESSION_TOLERANCE)):
                        dropped_wide += 1
                        continue
                    # Within the tolerance on a yielding object: command the widest opening the hand
                    # has and let the object give up the remainder. Storing the measured span
                    # instead would put a width the gripper cannot reach into the record.
                    squeeze = width - MAX_JAW_WIDTH
                    width = MAX_JAW_WIDTH
                    compressed += 1
                # Re-centre on the material: the box axis need not run through the middle of what
                # the pads actually close on (a banana's shaft, a mug's wall).
                centred = point + 0.5 * (lo + hi) * jaw
                # ...then seat it along the APPROACH so the object sits between the pads rather than
                # on their tips. POSE_CONVENTION v2 requires this of every stored pose.
                seat = pad_seat(centred, approach, jaw, width, verts, faces)
                if seat is None:
                    dropped_unseated += 1
                    continue
                uncontained += not seat.contained
                retreated += seat.seat_mode == "retreated"
                blocked += seat.blocked
                out.append(make_candidate(
                    ctx.frame, self._id(face, k_jaw, frac), seat.position, approach, jaw,
                    width=width, source=self.source, seat_mode=seat.seat_mode,
                    span=seat.span[1] - seat.span[0], seat_depth=seat.seat_depth,
                    labels=("obb_face", f"jaw_{_AXIS_NAME[k_jaw]}")
                           + (("compressed",) if squeeze else ())
                           + ((SEAT_BLOCKED_LABEL,) if seat.blocked else ()),
                    notes=(f"face-aligned box grasp: enters the {face} face, jaws along "
                           f"{_AXIS_NAME[k_jaw]}; span {(hi - lo) * 1000:.1f} mm measured over a "
                           f"{pad * 2000:.0f} mm column; pad-seated {seat.advance * 1000:+.1f} mm "
                           f"along the approach"
                           + ("" if not squeeze else
                              f"; span+clearance {span_width * 1000:.1f} mm exceeds the "
                              f"{MAX_JAW_WIDTH * 1000:.0f} mm jaw by {squeeze * 1000:.1f} mm "
                              f"({squeeze / MAX_JAW_WIDTH * 100:.1f} %) — width stored at the jaw "
                              f"limit, the object compresses the remainder")
                           + ("" if seat.contained else
                              f" (object {(seat.span[1] - seat.span[0]) * 1000:.0f} mm deep, "
                              f"longer than the pads — not fully enclosed)")
                           + ("" if seat.seat_mode != "retreated" else
                              f"; retreated {seat.retreat * 1000:.0f} mm to clear the hand")
                           + ("" if not seat.blocked else
                              "; hand collides at EVERY depth on this approach"))))
                ctr = centred_variant(seat, seat.position, approach, jaw, width, verts, faces)
                if ctr is not None:
                    out.append(make_candidate(
                        ctx.frame, self._id(face, k_jaw, frac) + CENTRED_VARIANT_SUFFIX,
                        ctr.position, approach, jaw, width=width, source=self.source,
                        seat_mode=ctr.seat_mode, span=ctr.span[1] - ctr.span[0],
                        seat_depth=ctr.seat_depth,
                        labels=("obb_face", f"jaw_{_AXIS_NAME[k_jaw]}", CENTRED_VARIANT_LABEL)
                               + (("compressed",) if squeeze else ()),
                        notes=f"centred depth variant of {self._id(face, k_jaw, frac)} "
                              f"[{ctr.seat_mode}]"))

        return PassOutput(candidates=out,
                          notes=(f"{len(out)} face-aligned grasp(s) over the canonical OBB; dropped "
                                 f"{dropped_empty} empty-column, {dropped_wide} over-jaw and "
                                 f"{dropped_unseated} unseatable grid point(s); {uncontained} deeper "
                                 f"than the pads; {retreated} retreated to clear the hand; "
                                 f"{blocked} seat_blocked (no clear depth)"
                                 + (f"; {compressed} within {COMPRESSION_TOLERANCE:.0%} of the jaw "
                                    f"and emitted at the jaw limit (compressible kind)"
                                    if compressed else "")))

    @staticmethod
    def _offsets(half_sweep: float) -> tuple:
        """Sweep fractions that are actually distinct on this object's third axis."""
        if not any(abs(f) * half_sweep >= _MIN_OFFSET for f in SWEEP_FRACTIONS):
            return (0.0,)
        return tuple(f for f in SWEEP_FRACTIONS if f == 0.0 or abs(f) * half_sweep >= _MIN_OFFSET)

    @staticmethod
    def _id(face: str, k_jaw: int, frac: float) -> str:
        sign = "c" if frac == 0.0 else ("p" if frac > 0 else "m")
        tag = "c" if frac == 0.0 else f"{sign}{abs(frac) * 100:.0f}"
        return f"{'p' if face[0] == '+' else 'n'}{face[1]}_jaw{_AXIS_NAME[k_jaw]}_{tag}"


PASS = ObbFacePass()
