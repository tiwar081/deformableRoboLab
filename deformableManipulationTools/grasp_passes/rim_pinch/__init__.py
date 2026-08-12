"""``rim_pinch`` pass — top-down pinches on the lip of an open container.

**The regime this exists for.** Seven rigid catalog assets are open containers (mug, bowl, pitcher,
bucket, tool_bin, parts_bin, long_tray_bin) and they are the catalog's worst performers at the
pre-grasp collision check. The cause is dimensional, not incidental: their bodies measure 88–152 mm
across against an 80 mm jaw stroke, and the 204 mm palm sits only 47 mm behind the TCP, so any pose
deep enough to straddle the body puts a finger or the palm through it. The one regime that fits is a
shallow pinch on the RIM WALL itself — 3–6 mm of material, which the jaw clears with room to spare.

**Why the existing generators miss it.** ``obb_face`` derives approaches from bounding-box faces and
``geometric`` slices along the canonical axes; a rim is a thin annular edge aligned with neither. It
is not that they reject these grasps, it is that nothing in either search is shaped like a lip.

**Rims are LOCATED, not inferred.** The seed is a ``vlm_regions`` region (``rim``/``lip``/``spout``),
because "which edge of this mesh is the opening" is a semantic question and guessing it from geometry
is how a pass ends up confidently pinching the base of a bowl. What the store gives is a ball with no
extent, so :mod:`.rim` turns each seed into a run of sites along the real edge — measuring the wall
frame locally and walking the tangent. Where an asset has no such region, this pass emits NOTHING and
says why, the same refusal ``vlm_regions`` itself makes: an absent annotation is not licence to guess.

**Seat mode.** These are pad-seated like every other pose (rule 2), and they are expected to come
back ``retreated`` or ``clamped_deep`` rather than ``span_flush``: the material column below a lip
runs the full height of the container wall, which no 53.8 mm pad can enclose. That is the correct
description of a lip pinch, not a defect — the pads hold the near material and the rest of the
container hangs below them.

Run it::

    .venv/bin/python -m deformableManipulationTools.grasp_passes run rim_pinch
    .venv/bin/python -m deformableManipulationTools.grasp_passes run rim_pinch --asset mug
    .venv/bin/python -m deformableManipulationTools.grasp_passes.rim_pinch --report
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import (CENTRED_VARIANT_LABEL, CENTRED_VARIANT_SUFFIX,
                              MAX_JAW_WIDTH, SEAT_BLOCKED_LABEL, centred_variant,
                              make_candidate, pad_seat)
from ..base import GraspPass, PassContext, PassOutput
from .config import DEFAULT, RimPinchConfig
from .rim import RimTracer

__all__ = ["RimPinchPass", "RimPinchConfig", "CONTAINER_KINDS", "PASS"]

# Catalog kinds that can carry a rim. Only mesh kinds: the procedural box kinds are solid by
# construction, and the deformables are out of this pass's scope for the same reason they are out of
# `geometric`'s — a lip that deforms under the pinch is not the lip the rest geometry describes.
CONTAINER_KINDS = ("ycb_mesh",)

# The one label every candidate carries, so a consumer can select this regime in one token.
METHOD_LABEL = "rim_pinch"


def _unsigned(v):
    """A direction with its sign fixed — for dedup keys where the axis is a line, not an arrow."""
    a = np.asarray(v, dtype=float)
    k = int(np.argmax(np.abs(a)))
    return -a if a[k] < 0 else a


class RimPinchPass(GraspPass):
    """Top-down lip pinches seeded from ``vlm_regions``. See the module docstring."""

    name = "rim_pinch"
    source = "rim_pinch"
    # v2 (2026-08-07): the span_flush seat rule (schema v3) — the retired centred rule
    # commanded the fingertips past the object's far surface (into the table for top-down
    # grasps on resting objects); candidates now carry the measured span.
    # v3 (2026-08-07): schema v4 — seat_depth stored per candidate, and the hand collision
    # checks pose the fingers at the PRE-SHAPED aperture (width + PREGRASP_MARGIN), not fully
    # open, matching the trajectory's pre-grasp; retreat/blocked outcomes move.
    # v4 (2026-08-07): schema v5 depth-variant pairs — a palm-safe span also stores its
    # CENTRED seat as a separate '_ctr' candidate; online selection picks between the pair.
    version = 4
    # Seeded from the regions STORE, not from another pass's candidates: regions are their own
    # field, written by vlm_regions independently of any generator, so this is not an upstream
    # dependency in the harness sense and declaring one would only force a needless run order.
    requires = ()
    kinds = CONTAINER_KINDS

    def __init__(self, cfg: RimPinchConfig = DEFAULT):
        self.cfg = cfg

    # ---------------------------------------------------------------------------------------------
    def run(self, ctx: PassContext) -> PassOutput:
        from ..vlm_regions import load_regions

        cfg = self.cfg
        doc = load_regions(ctx.name)
        if doc is None:
            return PassOutput(notes=(
                f"no regions annotation for {ctx.name!r} — this pass locates rims from the "
                f"vlm_regions store and does not infer them from geometry, so it produces nothing "
                f"rather than guessing which edge is an opening. Run the vlm_regions pass first."))

        seeds = [r for r in doc.regions
                 if r.label in cfg.region_labels and r.confidence >= cfg.min_confidence]
        if not seeds:
            have = sorted({r.label for r in doc.regions})
            return PassOutput(notes=(
                f"{ctx.name!r} is annotated but carries no {'/'.join(cfg.region_labels)} region "
                f"(labels present: {have or 'none'}"
                + (f"; annotator said: {doc.empty_reason}" if doc.empty_reason else "")
                + "). Not an open container as far as the store is concerned, so nothing is emitted."))

        mesh = self._mesh(ctx)
        # The canonical frame's origin IS the OBB centre, so it is the body centre by construction —
        # and unlike a vertex mean it is not dragged sideways by a handle.
        tracer = RimTracer(mesh, cfg, body_centre=np.zeros(3))

        raw, traced = [], {}
        for region in sorted(seeds, key=lambda r: r.id):
            samples = tracer.walk(region)
            traced[region.id] = len(samples)
            raw.extend(samples)

        kept = self._dedup(raw)
        if len(kept) > cfg.max_per_asset:
            kept = kept[:cfg.max_per_asset]
        candidates, seat_note = self._emit(ctx, kept)

        per_seed = ", ".join(f"{rid}={n}" for rid, n in sorted(traced.items()))
        note = (f"lip pinches seeded from {len(seeds)} vlm_regions region(s) [{per_seed}]; "
                f"{len(raw)} site(s) traced, {len(kept)} after dedup, {len(candidates)} stored. "
                f"{seat_note} Wall thickness is measured, not assumed; quality unmeasured.")
        return PassOutput(candidates=candidates, notes=note)

    # ---- geometry ---------------------------------------------------------------------------------
    def _mesh(self, ctx: PassContext):
        """The asset's canonical-frame surface, welded.

        ``load_usd_mesh`` returns vertices per face-corner, so an unwelded YCB scan is topologically
        a pile of loose triangles — the KD-tree patches and the surface snap both need real
        adjacency, and the ray probe needs a coherent surface."""
        import trimesh

        if ctx.asset.faces is None:
            raise ValueError(f"{ctx.name}: no surface triangles; a rim pinch needs a surface to "
                             f"measure a wall against")
        mesh = trimesh.Trimesh(vertices=ctx.asset.canonical_vertices(),
                               faces=np.asarray(ctx.asset.faces, dtype=np.int64).reshape(-1, 3),
                               process=False)
        mesh.merge_vertices()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _dedup(self, samples) -> list:
        """First-wins dedup on a quantized pose key, in a deterministic order.

        Seeds on one rim walk into each other — the mug carries four regions on a single circle — so
        without this most candidates would be the same lip site found from two directions."""
        cfg = self.cfg
        ordered = sorted(samples, key=lambda s: (s.region_id, s.arc))
        seen, out = set(), []
        for s in ordered:
            key = (tuple(np.round(s.point / cfg.dedup_position).astype(int)),
                   tuple(np.round(np.asarray(s.approach) / cfg.dedup_direction).astype(int)),
                   tuple(np.round(_unsigned(s.jaw) / cfg.dedup_direction).astype(int)))
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    # ---- emit ---------------------------------------------------------------------------------------
    def _emit(self, ctx: PassContext, samples):
        """Seat every pinch and build the candidates. Rule 2: nothing is stored unseated."""
        cfg = self.cfg
        verts, faces = ctx.asset.canonical_vertices(), ctx.asset.faces
        out, dropped_wide, dropped_seat, blocked, modes = [], 0, 0, 0, {}
        for i, s in enumerate(samples):
            width = float(s.thickness) + cfg.clearance
            if width > MAX_JAW_WIDTH:
                dropped_wide += 1           # never clip: that would describe an impossible grasp
                continue
            seat = pad_seat(s.point, s.approach, s.jaw, width, verts, faces)
            if seat is None:
                dropped_seat += 1           # no material in the column — drop, don't store a pose
                continue                    # seated on nothing
            modes[seat.seat_mode] = modes.get(seat.seat_mode, 0) + 1
            if seat.blocked:
                blocked += 1
            arc_mm = s.arc * 1000.0
            cid = f"rim_{s.region_id}_{'p' if s.arc >= 0 else 'm'}{abs(arc_mm):03.0f}"
            out.append(make_candidate(
                ctx.frame, cid, seat.position, s.approach, s.jaw, width=width,
                source=self.source, seat_mode=seat.seat_mode,
                span=seat.span[1] - seat.span[0], seat_depth=seat.seat_depth,
                labels=(METHOD_LABEL, s.region_label)
                       + ((SEAT_BLOCKED_LABEL,) if seat.blocked else ()),
                notes=(f"lip pinch on {s.region_id} ({s.region_label}) at arc {arc_mm:+.0f} mm; "
                       f"wall {s.thickness * 1000:.1f} mm, jaw {width * 1000:.1f} mm; "
                       f"seated {seat.advance * 1000:+.1f} mm [{seat.seat_mode}]"
                       + (", NO clear depth on this approach" if seat.blocked else ""))))
            ctr = centred_variant(seat, seat.position, s.approach, s.jaw, width, verts, faces)
            if ctr is not None:
                out.append(make_candidate(
                    ctx.frame, cid + CENTRED_VARIANT_SUFFIX, ctr.position, s.approach, s.jaw,
                    width=width, source=self.source, seat_mode=ctr.seat_mode,
                    span=ctr.span[1] - ctr.span[0], seat_depth=ctr.seat_depth,
                    labels=(METHOD_LABEL, s.region_label, CENTRED_VARIANT_LABEL),
                    notes=f"centred depth variant of {cid} [{ctr.seat_mode}]"))
        note = (f"seat modes {modes or '{}'}"
                + (f", {blocked} blocked" if blocked else "")
                + (f", {dropped_wide} wider than the jaw" if dropped_wide else "")
                + (f", {dropped_seat} unseatable" if dropped_seat else "") + ".")
        return out, note


PASS = RimPinchPass()
