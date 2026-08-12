"""``geometric`` pass — grasp CANDIDATES for rigid catalog assets, from two independent methods.

Both methods answer the same question from opposite directions, which is why the pass runs both
rather than picking one:

* :mod:`.medial` works from the INSIDE. A medial-axis skeleton carries the local radius of the
  solid at every point along it; where twice that radius fits the jaw, the object is locally a
  graspable tube and the grasp is the one perpendicular to the local axis. This is what finds the
  neck of a bottle, the shaft of a tool, the length of a banana — features defined by a centre line,
  which a face-pair search only stumbles onto.
* :mod:`.sweep` works from the OUTSIDE. Cross-sections cut along each canonical axis are searched
  for boundary that faces itself across a gap the jaw can span. This is what finds the flat parallel
  sides of a box, a wall to pinch, a rim — features defined by a pair of surfaces, which a skeleton
  smears into a sheet and discards.

**Every candidate is tagged with the method that produced it**, in ``labels``: ``medial_axis`` or
``cross_section`` (:data:`METHOD_LABELS`). Both share the pass's single ``source`` tag, ``geometric``
— the merge deduplicates by source, so one pass gets one tag — and the id prefix (``medial_`` /
``xsec_``) mirrors the label so a candidate is identifiable from its id alone.

**Scope: rigid kinds only** (``ycb_mesh``, ``rubiks_cube``, ``rigid_box``). The deformable kinds the
schema does support (``soft_mesh``, ``soft_block``) are excluded deliberately: both methods measure
a span the jaws close on, and for a body that deforms under the grasp the rest-shape span is not
that number. (Cables are out of the whole pipeline's scope since 2026-08-11 — see the SCOPE
paragraph in ``grasp_library.py`` — not merely excluded here.)

**What this pass does NOT decide.** ``face`` is derived from the approach by ``make_candidate`` (the
schema requires it to agree with the transform, so it cannot be left unset), but the FACE BUCKET as
a reachability filter and every ``quality`` field are left for the passes that own them — nothing
here is measured in simulation, and no candidate carries a ``quality_source``. A candidate from this
pass is a geometrically plausible place to put the jaws, not a grasp known to work.

Inspect one asset's output with the viewer::

    .venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz banana

Run it like any pass::

    .venv/bin/python -m deformableManipulationTools.grasp_passes run geometric --asset banana
    .venv/bin/python -m deformableManipulationTools.grasp_passes run geometric --check-idempotent
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import (CENTRED_VARIANT_LABEL, CENTRED_VARIANT_SUFFIX,
                              MAX_JAW_WIDTH, SEAT_BLOCKED_LABEL, centred_variant,
                              make_candidate, pad_seat)
from ..base import GraspPass, PassContext, PassOutput
from . import select
from .config import DEFAULT, GeometricConfig
from .medial import medial_axis
from .meshprep import prepare
from .sweep import cross_section_pairs

METHOD_LABELS = ("medial_axis", "cross_section")

# Catalog kinds this pass handles — the rigid ones. See the SCOPE paragraph above.
RIGID_KINDS = ("ycb_mesh", "rubiks_cube", "rigid_box")

__all__ = ["GeometricPass", "GeometricConfig", "METHOD_LABELS", "PASS"]


def _basis(axis):
    """Two unit vectors spanning the plane perpendicular to ``axis``, chosen deterministically."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(a)))] = 1.0       # the world axis least aligned with `a`
    u = np.cross(a, seed)
    u = u / np.linalg.norm(u)
    return u, np.cross(a, u)


class GeometricPass(GraspPass):
    """Two-method geometric candidate generator. See the module docstring."""

    name = "geometric"
    source = "geometric"
    # v3 (2026-08-06): uncontained poses use the CLAMPED deep-seat (near material at
    # SEAT_DEEPEST_Z) instead of midpoint-centring, which pushed the overhang into the palm.
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
    version = 7
    requires = ()
    kinds = RIGID_KINDS

    def __init__(self, cfg: GeometricConfig = DEFAULT):
        self.cfg = cfg

    # ---------------------------------------------------------------------------------------------
    def run(self, ctx: PassContext) -> PassOutput:
        cfg = self.cfg
        prepared = prepare(ctx.name, ctx.asset.canonical_vertices(), ctx.asset.faces, cfg)
        scale = float(np.linalg.norm(ctx.asset.half_extents))

        medial, medial_note = self._medial_candidates(ctx, prepared)
        xsec, xsec_note = self._sweep_candidates(ctx, prepared)

        medial = select.cap(select.dedup(medial, cfg), cfg.max_per_method, scale)
        xsec = select.cap(select.dedup(xsec, cfg), cfg.max_per_method, scale)

        medial, m_seat = self._seat(ctx, medial)
        xsec, x_seat = self._seat(ctx, xsec)

        note = (f"geometry: {'watertight mesh' if prepared.watertight else 'OPEN surface'}"
                f"{', subdivided' if prepared.subdivided else ''}, width measured by "
                f"{'ray cast' if prepared.watertight else f'{cfg.voxel_pitch * 1000:.0f} mm voxel march'}. "
                f"medial_axis: {len(medial)} kept ({medial_note}; {m_seat}). "
                f"cross_section: {len(xsec)} kept ({xsec_note}; {x_seat}). "
                f"quality unmeasured — a later pass fills it.")
        return PassOutput(candidates=medial + xsec, notes=note)

    # ---- pad seating (POSE_CONVENTION v2) --------------------------------------------------------
    def _seat(self, ctx: PassContext, candidates):
        """Advance every pose along its approach until the object sits between the PADS.

        ``grasp_library.pad_seat`` is the one place hand geometry is applied, and v2 poses are what
        it returns — the TCP sits at the fingertip TIPS with all pad material behind it, so a pose
        left where the geometry search found it (on the object's material) grips with the very tips
        and leaves about half the object forward of the fingers. Measured on this catalog: the seat
        advances by 20–30 mm.

        Run AFTER dedup and the cap purely for cost — ``pad_seat`` rebuilds a BVH per call (~130 ms
        on a 15k-triangle YCB scan), so seating the raw stream would mean thousands of calls per
        asset instead of at most ``2 * max_per_method``. Every pose that is STORED is seated, which
        is what rule 2 requires; nothing is emitted unseated."""
        verts, faces = ctx.asset.canonical_vertices(), ctx.asset.faces
        out, dropped, deep, retreated, blocked, advances = [], 0, 0, 0, 0, []
        for c in candidates:
            seat = pad_seat(c.position, c.approach, c.jaw_axis, c.width, verts, faces)
            if seat is None:
                dropped += 1          # no material in the jaw column — drop, never store a pose
                continue              # seated on nothing (rule 2)
            advances.append(seat.advance)
            extra = f"; pad-seated {seat.advance * 1000:+.1f} mm along the approach [{seat.method}]"
            if not seat.contained:
                # Not a reason to drop — the clamp is still the best placement — but the consumer
                # should know the pads cannot enclose this object however it is placed.
                deep += 1
                extra += (f", object is {(seat.span[1] - seat.span[0]) * 1000:.0f} mm deep and the "
                          f"pads CANNOT enclose it")
            if seat.seat_mode == "retreated":
                retreated += 1
                extra += f", retreated {seat.retreat * 1000:.0f} mm to clear the hand"
            labels = c.labels
            if seat.blocked:
                # A measured failure, kept and marked — the codebase's convention. The shake pass's
                # pre-check will independently confirm the collision and skip the trial.
                blocked += 1
                labels = labels + (SEAT_BLOCKED_LABEL,)
                extra += ", hand collides at EVERY depth on this approach"
            out.append(make_candidate(
                ctx.frame, c.id, seat.position, c.approach, c.jaw_axis, width=c.width,
                source=self.source, seat_mode=seat.seat_mode,
                span=seat.span[1] - seat.span[0], seat_depth=seat.seat_depth, labels=labels,
                notes=c.notes + extra))
            ctr = centred_variant(seat, seat.position, c.approach, c.jaw_axis, c.width,
                                  verts, faces)
            if ctr is not None:
                out.append(make_candidate(
                    ctx.frame, c.id + CENTRED_VARIANT_SUFFIX, ctr.position, c.approach,
                    c.jaw_axis, width=c.width, source=self.source, seat_mode=ctr.seat_mode,
                    span=ctr.span[1] - ctr.span[0], seat_depth=ctr.seat_depth,
                    labels=c.labels + (CENTRED_VARIANT_LABEL,),
                    notes=c.notes + f"; centred depth variant [{ctr.seat_mode}]"))
        summary = (f"pad-seated by {np.median(advances) * 1000:.1f} mm median"
                   if advances else "nothing to seat")
        if dropped:
            summary += f", {dropped} dropped as unseatable"
        if deep:
            summary += f", {deep} deeper than the pads"
        if retreated:
            summary += f", {retreated} retreated to clear the hand"
        if blocked:
            summary += f", {blocked} seat_blocked (no clear depth)"
        return out, summary

    # ---- method 1 --------------------------------------------------------------------------------
    def _medial_candidates(self, ctx: PassContext, prepared):
        """Grasps perpendicular to the local skeleton axis, wherever the local radius fits the jaw."""
        cfg = self.cfg
        axis = medial_axis(prepared, cfg)
        if not len(axis):
            return [], f"no usable skeleton nodes [{axis.backend}: {axis.detail}]"

        # Build every (node, roll) query first and measure them in ONE batched probe call: the ray
        # backend pays a fixed cost per call, and a per-candidate loop turns that into seconds.
        points, jaws, approaches, ids, nodes = [], [], [], [], []
        for i in range(len(axis)):
            tangent = axis.tangents[i]
            u, v = _basis(tangent)
            for k in range(cfg.roll_samples):
                phi = 2.0 * np.pi * k / cfg.roll_samples
                approach = np.cos(phi) * u + np.sin(phi) * v      # perpendicular to the local axis
                jaw = np.cross(tangent, approach)                 # closes ACROSS the tube
                jaw = jaw / np.linalg.norm(jaw)
                points.append(axis.points[i])
                jaws.append(jaw)
                approaches.append(approach)
                ids.append(f"medial_{i:03d}_r{k}")
                nodes.append(i)

        points = np.array(points)
        span, centre, ok = prepared.span_probe.measure(points, np.array(jaws))
        # The skeleton radius is what GATES a node (the method's own criterion); the measured span is
        # what a candidate REPORTS, because a real cross-section is not a circle and the chord the
        # jaws actually close along is rarely 2r.
        fits = ok & (span >= cfg.min_span) & (span + cfg.clearance <= MAX_JAW_WIDTH)
        # ...and the node must sit near the middle of that chord, or the radius was not describing
        # this direction and the "grasp" is a chord across a corner. See config.max_centring_error.
        with np.errstate(divide="ignore", invalid="ignore"):
            offset = np.linalg.norm(centre - points, axis=1) / np.maximum(span, 1.0e-9)
        centred = offset <= cfg.max_centring_error
        fits &= centred

        out = []
        for n, keep in enumerate(fits):
            if not keep:
                continue
            i = nodes[n]
            out.append(make_candidate(
                ctx.frame, ids[n], centre[n], approaches[n], jaws[n], width=float(span[n]),
                # Pre-seat intermediate: _seat() re-makes every stored candidate with the mode and
                # span pad_seat actually measured; this placeholder never reaches a sidecar.
                source=self.source, seat_mode="span_flush", span=float(span[n]), seat_depth=-0.02,
                labels=("medial_axis",),
                notes=(f"perpendicular to the medial axis; local radius {axis.radii[i] * 1000:.1f} mm, "
                       f"linearity {axis.linearity[i]:.2f}, measured span {span[n] * 1000:.1f} mm "
                       f"centred to {offset[n]:.2f} of it [{axis.backend}]")))
        return out, (f"{len(axis)} node(s) x {cfg.roll_samples} roll(s) -> {len(out)} "
                     f"({int((~centred).sum())} off-centre) [{axis.detail}]")

    # ---- method 2 --------------------------------------------------------------------------------
    def _sweep_candidates(self, ctx: PassContext, prepared):
        """Grasps where a cross-section shows near-parallel opposing faces within the jaw."""
        cfg = self.cfg
        pairs = cross_section_pairs(prepared, cfg, MAX_JAW_WIDTH)
        out = []
        counter: dict = {}
        for pair in pairs:
            # id encodes WHERE it came from — sweep axis, slice index, and its order within that
            # slice — so it stays meaningful in a diff and needs no hashing to be unique.
            slot = (pair.axis, pair.height_index)
            n = counter.get(slot, 0)
            counter[slot] = n + 1
            cid = f"xsec_{'xyz'[pair.axis]}{pair.height_index:02d}_{n:03d}"
            out.append(make_candidate(
                ctx.frame, cid, pair.centre, pair.approach, pair.jaw, width=pair.width,
                # Pre-seat intermediate, same as the medial branch: replaced in _seat().
                source=self.source, seat_mode="span_flush", span=pair.width, seat_depth=-0.02,
                labels=("cross_section",),
                notes=(f"opposing faces in a section normal to {'xyz'[pair.axis]} at "
                       f"{pair.height * 1000:+.1f} mm; normals antiparallel to within "
                       f"{pair.opposing_deg:.1f} deg, gap {pair.width * 1000:.1f} mm")))
        return out, (f"{cfg.slices_per_axis} slice(s) x 3 axes -> {len(pairs)} pair(s), "
                     f"{len(out)} candidate(s)")


PASS = GeometricPass()
