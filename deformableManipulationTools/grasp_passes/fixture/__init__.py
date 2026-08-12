"""``fixture`` pass — hand-placed grasp candidates, so passes that CONSUME candidates can be
developed and tested before any generator exists.

It is a real pass (same interface, same sidecar, same merge) rather than a special case, so a
consumer pass developed against it works unchanged once a generator replaces it: declare
``requires = ("fixture",)`` while developing, then switch to the generator's pass name.

What it emits is honest but limited — a handful of plausible, geometrically valid grasps per object
from the declarative table in :mod:`.candidates`. It is NOT a generator: no antipodal search, no
coverage guarantee, no quality. Widths ARE measured from the real mesh at the grasp point, so a
consumer that reasons about jaw opening sees realistic numbers.
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import (CENTRED_VARIANT_LABEL, CENTRED_VARIANT_SUFFIX,
                              MAX_JAW_WIDTH, SEAT_BLOCKED_LABEL, centred_variant,
                              make_candidate, pad_seat)
from ..base import GraspPass, PassContext, PassOutput
from .candidates import FIXTURES, FixtureGrasp

# Half-size of the square pad footprint [m] used to decide which material sits between the jaws when
# measuring a grasp's width. ~2 cm across, the scale of a Franka fingertip pad.
PAD_HALF = 0.010

_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}


def _direction(word: str) -> np.ndarray:
    """``"-z"`` / ``"+y"`` / ``"y"`` -> a unit canonical-frame vector."""
    w = word.strip()
    sign = -1.0 if w.startswith("-") else 1.0
    axis = w.lstrip("+-")
    if axis not in _AXES:
        raise ValueError(f"bad axis word {word!r}; expected one of +x -x +y -y +z -z")
    return sign * _AXES[axis]


# A gap wider than this along the jaw axis separates one lump of material from the next — used to
# find the lobe a "pinch" closes on (e.g. one mug wall, not the far wall too).
LOBE_GAP = 0.004
_MIN_SAMPLES = 8


def _column(canonical_vertices, point, jaw, approach, pad_half):
    """Vertices in a pad-sized column around the grasp point, as offsets along the jaw axis.

    The column is GROWN until it holds enough samples: mesh density varies by three orders of
    magnitude across the catalog (a YCB scan has ~10k vertices, a coarse fTetWild sponge has 235,
    and a fixed pad-sized query finds nothing at all in the sponge's interior). The pad actually
    used is returned so the candidate can record it instead of hiding the adjustment."""
    third = np.cross(jaw, approach)
    third = third / max(np.linalg.norm(third), 1.0e-12)
    du = np.asarray(canonical_vertices, dtype=float) - point
    a_off, t_off, j_off = du @ approach, du @ third, du @ jaw
    pad = pad_half
    for _ in range(6):
        m = (np.abs(a_off) <= pad) & (np.abs(t_off) <= pad)
        if m.sum() >= _MIN_SAMPLES:
            return np.sort(j_off[m]), pad
        pad *= 1.6
    raise ValueError(
        f"no material within a {pad * 2:.3f} m column at {np.round(point, 4).tolist()} — the grasp "
        f"point is in empty space")


def measure_width(canonical_vertices, point, jaw, approach, *, pad_half: float = PAD_HALF,
                  local: bool = False) -> tuple:
    """Span of material along the jaw axis at a grasp point → ``(span [m], pad used [m])``.

    Measured from the real mesh rather than taken from the OBB extent, which would be wrong
    wherever the object is not box-shaped. Two intents, because they need different answers:

    * ``local=False`` (**span**) — the jaws must clear EVERYTHING in the column, so the span runs
      from the nearest to the furthest material. This is the honest answer for grasping a whole
      body, and it is what makes an over-wide grasp fail loudly.
    * ``local=True`` (**pinch**) — the jaws close on the one lump of material at the grasp point,
      with the rest of the object outside the jaws (pinching a mug wall: one pad inside the cup,
      one outside). The span covers only the lobe containing the grasp point, where a lobe is
      material unbroken by a gap wider than :data:`LOBE_GAP`.
    """
    proj, pad = _column(canonical_vertices, point, jaw, approach, pad_half)
    if not local:
        return float(proj[-1] - proj[0]), pad
    # Walk outward from the grasp point (offset 0) until a gap wider than LOBE_GAP breaks contact.
    i = int(np.searchsorted(proj, 0.0))
    lo = hi = min(max(i, 0), len(proj) - 1)
    while lo > 0 and proj[lo] - proj[lo - 1] <= LOBE_GAP:
        lo -= 1
    while hi < len(proj) - 1 and proj[hi + 1] - proj[hi] <= LOBE_GAP:
        hi += 1
    return float(proj[hi] - proj[lo]), pad


class FixturePass(GraspPass):
    name = "fixture"
    source = "fixture"
    # v2 (2026-08-05): poses are pad-seated via grasp_library.pad_seat(), so every stored pose moves
    # and the cached v1 sidecars are stale. Until this landed, fixture was the last v1 producer and
    # its six assets could not merge at all — the schema refuses to mix v1 and v2 poses, which are
    # indistinguishable by inspection.
    # v3 (2026-08-06): uncontained poses use pad_seat's clamped deep-seat (see grasp_library).
    # v4 (2026-08-06): pad_seat's collision-aware retreat — a pose whose rule seat puts the hand
    # inside the object backs off to the deepest clear depth; candidates carry seat_mode (schema
    # v2), and a pose clear at no depth keeps its rule seat with the seat_blocked label.
    # v5 (2026-08-07): the span_flush seat rule (schema v3) — the retired centred rule
    # commanded the fingertips past the object's far surface (into the table for top-down
    # grasps on resting objects); candidates now carry the measured span.
    # v6 (2026-08-07): schema v4 — seat_depth stored per candidate, and the hand collision
    # checks pose the fingers at the PRE-SHAPED aperture (width + PREGRASP_MARGIN), not fully
    # open, matching the trajectory's pre-grasp; retreat/blocked outcomes move.
    # v7 (2026-08-07): schema v5 depth-variant pairs — a palm-safe span also stores its
    # CENTRED seat as a separate '_ctr' candidate; online selection picks between the pair.
    # v8 (2026-08-11): cables out of the pipeline's scope — power_cable fixtures removed.
    version = 8
    kinds = ()          # any supported kind; the table decides which assets actually get entries

    def applies_to(self, name: str, kind: str) -> bool:
        return name in FIXTURES

    def run(self, ctx: PassContext) -> PassOutput:
        verts = ctx.asset.canonical_vertices()
        faces = ctx.asset.faces
        half = ctx.asset.half_extents
        out = []
        for spec in FIXTURES[ctx.name]:
            out.extend(self._resolve(ctx, spec, verts, faces, half))
        return PassOutput(candidates=out,
                          notes="hand-placed placeholders; widths measured from the mesh, "
                                "poses pad-seated")

    def _resolve(self, ctx: PassContext, spec: FixtureGrasp, verts, faces, half):
        approach = _direction(spec.approach)
        jaw = _direction(spec.jaw)
        if abs(float(jaw @ approach)) > 1.0e-9:
            raise ValueError(f"{ctx.name}/{spec.id}: jaw {spec.jaw!r} is parallel to approach "
                             f"{spec.approach!r} — the jaws would close along the approach")
        point = np.asarray(spec.at, dtype=float) * half
        span, pad = measure_width(verts, point, jaw, approach, local=spec.local)
        width = span + spec.clearance
        if width > MAX_JAW_WIDTH:
            # An authoring bug, not something to clip: a clipped width would describe a grasp the
            # gripper physically cannot make, and validation would happily accept it.
            raise ValueError(
                f"{ctx.name}/{spec.id}: measured span {span * 1000:.1f} mm + clearance "
                f"{spec.clearance * 1000:.1f} mm exceeds the {MAX_JAW_WIDTH * 1000:.0f} mm jaw. "
                f"Move the grasp, choose a different jaw axis, or set local=True if the jaws are "
                f"meant to pinch one wall rather than span the whole body.")
        # Seat the pose along the approach so the object sits BETWEEN the pads instead of on their
        # tips. POSE_CONVENTION v2 requires this of every stored pose, and it is not this pass's
        # decision to make: pad_seat() reads it off the hand's own pad geometry, the same call
        # obb_face and geometric make. Without it these placeholders sat ~28 mm proud of the pads —
        # measured, and enough on its own to lose a 16 mm cable out of the fingertips.
        seat = pad_seat(point, approach, jaw, width, verts, faces)
        if seat is None:
            raise ValueError(
                f"{ctx.name}/{spec.id}: no material in the jaw column at "
                f"{np.round(point, 4).tolist()} — the grasp point is in empty space")
        kind = "pinch" if spec.local else "span"
        seat_note = {
            "span_flush": "",
            "clamped_deep": ", deeper than the pads",
            "retreated": f", retreated {seat.retreat * 1000:.0f} mm to clear the hand",
        }[seat.seat_mode]
        if seat.blocked:
            seat_note = ", hand collides at EVERY depth on this approach"
        cands = [make_candidate(
            ctx.frame, spec.id, seat.position, approach, jaw, width=width, source=self.source,
            seat_mode=seat.seat_mode, span=seat.span[1] - seat.span[0], seat_depth=seat.seat_depth,
            labels=spec.labels + ((SEAT_BLOCKED_LABEL,) if seat.blocked else ()),
            notes=(spec.notes or "")
                  + f" [fixture: {kind} {span * 1000:.1f} mm measured over a "
                    f"{pad * 2000:.0f} mm column; seated {seat.advance * 1000:+.1f} mm along the "
                    f"approach{seat_note}]")]
        ctr = centred_variant(seat, seat.position, approach, jaw, width, verts, faces)
        if ctr is not None:
            cands.append(make_candidate(
                ctx.frame, spec.id + CENTRED_VARIANT_SUFFIX, ctr.position, approach, jaw,
                width=width, source=self.source, seat_mode=ctr.seat_mode,
                span=ctr.span[1] - ctr.span[0], seat_depth=ctr.seat_depth,
                labels=spec.labels + (CENTRED_VARIANT_LABEL,),
                notes=f"centred depth variant of {spec.id} [{ctr.seat_mode}]"))
        return cands


PASS = FixturePass()
