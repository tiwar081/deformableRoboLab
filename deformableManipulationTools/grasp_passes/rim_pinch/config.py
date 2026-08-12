"""Tuning knobs for the ``rim_pinch`` pass, with the reasons attached.

Offline generator parameters — how far around a rim to walk and what still counts as a lip. Nothing
here is physics; the physical constants (jaw stroke, pad and palm geometry) are imported from
``grasp_library`` rather than restated, because a copy here could drift out of step with the hand.

Every length is in metres.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RimPinchConfig:
    """Where to look for a rim, how far to walk it, and what still counts as a pinchable lip."""

    # ---- which regions locate a rim --------------------------------------------------------------
    region_labels: tuple = ("rim", "lip", "spout")
    """``vlm_regions`` labels treated as an openable edge. All three are the boundary of an opening
    described from a different angle — the store's own vocabulary calls ``lip`` "a thin flanged edge
    that a jaw can hook or pinch" and ``spout`` a pour lip. A ``handle`` is deliberately absent: it
    is a graspable feature, but it is not a thin annular edge and a different pass should own it."""

    min_confidence: float = 0.0
    """Minimum region confidence to seed from. 0 accepts everything the annotator kept, which is the
    right default while the store is small — the geometric checks below reject a bad seed far more
    reliably than the VLM's own confidence does."""

    # ---- the local wall frame ---------------------------------------------------------------------
    patch_radius: float = 0.009
    """Radius [m] of the surface neighbourhood the local wall frame is fitted over. A rim is locally
    a thin SHEET, so the smallest principal direction of the patch is across the wall — that is the
    jaw axis, and it is reliable precisely because the thickness (3–6 mm) is much smaller than the
    patch. Too large a patch starts to curve around the rim and the thin direction blurs."""

    min_patch_points: int = 12
    """Points a patch needs before its PCA is trusted. The patch grows (up to
    :attr:`patch_growth_steps` times) until it has this many — catalog mesh density varies by two
    orders of magnitude, and a fixed radius finds nothing on the coarse assets."""

    patch_growth_steps: int = 5
    patch_growth: float = 1.5

    # ---- what counts as a pinchable lip ------------------------------------------------------------
    max_wall: float = 0.015
    """Thickest wall [m] still treated as a lip. The catalog's container walls measure 3–6 mm; past
    ~15 mm the walk has left the rim and is on the solid body, where this pass has no business
    emitting anything — the body is what the other generators already cover (and fail on)."""

    min_wall: float = 0.0005
    """Thinner than this [m] is the same surface hit twice, not a wall.

    Set well below any real wall on purpose. The catalog's containers are SCANNED SHELLS, not
    modelled solids, and they are thinner than intuition suggests: the YCB bowl measures 0.94–1.01 mm
    and the mug 1.1–2.6 mm. A 1.5 mm floor — which sounds conservative — rejected every bowl seed and
    produced zero candidates for it. The real noise this needs to exclude is a duplicated coincident
    face, which reports microns."""

    wall_search_depths: tuple = (0.0, 0.006, 0.012)
    """Depths [m] below a seed to look for the wall, tried in order, first hit wins.

    A seed does not always land where the wall is measurable. Where the annotator pointed at the
    flat TOP of a rim that is wider than :attr:`patch_radius` — the pitcher's flanged lip — the patch
    is a horizontal plate, its thin direction is vertical, and the thickness probe fires straight
    down through the whole vessel (measured: 129 mm). Stepping a little way down the wall lands on
    the side, where the thin direction is across the wall again. 0.0 is tried first so seeds that
    are already on the wall are used exactly as given."""

    probe_standoff: float = 0.015
    """How far outside the surface [m] the thickness ray starts. Comfortably clear of a 6 mm wall
    without reaching across the opening to the far side, which would measure the whole container."""

    # ---- walking the rim ----------------------------------------------------------------------------
    sample_spacing: float = 0.018
    """Arc spacing [m] between pinches along the rim. Slightly under the ~20 mm pad so consecutive
    grasps are distinguishable places on the lip rather than the same one described twice."""

    max_arc: float = 0.13
    """How far [m] to walk in each direction from a seed region before stopping. Enough to carry a
    single seed most of the way round a mug (≈140 mm circumference) while keeping a stray seed on a
    bin's flat wall from wrapping the whole perimeter."""

    rim_band: float = 0.012
    """How far [m] a walked sample may drift along the descent axis from its seed and still count as
    the same rim. This is what stops the walk sliding down the wall onto the body: the rim is a
    curve, and a step that leaves its height band has left it."""

    # ---- output --------------------------------------------------------------------------------------
    clearance: float = 0.004
    """Feasibility margin on the jaw [m]. A pinch is kept only when ``thickness + clearance`` fits
    the stroke; on a 3–6 mm lip this is never the binding constraint, and it is here so the check
    exists rather than because a rim ever fails it."""

    dedup_position: float = 0.008
    """Pinches closer than this [m] and aligned in direction collapse to one. Seeds on the same rim
    walk into each other — the mug carries four rim regions on one circle — and the overlap is the
    same grasp found twice, not coverage."""

    dedup_direction: float = 0.25
    """Direction quantum for the dedup key (unit-vector components rounded to this, ~14 deg)."""

    max_per_asset: int = 96
    """Cap on candidates per asset. Reached only where several seeds each open a long walk."""


DEFAULT = RimPinchConfig()
