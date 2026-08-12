"""Hand-placed fixture grasps, as a declarative spec per catalog object.

Each entry names a point on the object in CANONICAL-frame terms and the two directions that define
the grasp; the pass resolves those against the real mesh (measuring the actual jaw span at that
point) rather than storing raw matrices, so the data stays readable and stays correct if the frame
is recomputed.

Canonical axes, always: **x = longest extent, y = middle, z = shortest** — so ``±z`` are the
object's two largest faces. Signs come from mesh asymmetry, which is not intuitive: the mug's ``+z``
points through its solid BASE (the material-dense side), so its open rim faces ``−z``. Check
``show <asset>`` and the axis profile before adding entries rather than assuming an orientation.

These are PLACEHOLDERS to unblock consumer passes — plausible and geometrically valid, but not
validated grasps and not a sample of what a generator should produce. Labels stay descriptive
("mid_span", "near_end") rather than semantic ("handle", "stem"): a semantic label nobody verified
is worse than none.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureGrasp:
    """One hand-placed grasp.

    ``at``       grasp centre as fractions of the half-extent along canonical (x, y, z);
                 ``(0, 0, 0)`` is the OBB centre, ``(0.6, 0, 0)`` is 60 % of the way to the +x face.
    ``approach`` direction the TCP travels INTO the grasp, e.g. ``"-z"`` = straight down onto the
                 +z face. Becomes the candidate's face bucket.
    ``jaw``      axis the jaws close along, e.g. ``"y"``. Must not be parallel to ``approach``.
    ``clearance`` extra jaw opening [m] on top of the measured span at that point.
    ``local``    False (default): the jaws must span ALL material in the column — a whole-body
                 grasp. True: the jaws pinch only the lump of material at the grasp point, the rest
                 of the object staying outside them (one mug wall, one pad in the cup, one out).
    """
    id: str
    at: tuple
    approach: str
    jaw: str
    clearance: float = 0.006
    labels: tuple = ()
    notes: str = ""
    local: bool = False


# Assets chosen to span the geometry cases a consumer pass has to cope with: an unambiguous
# elongated frame (banana), an ambiguous one where the box yaw is arbitrary (mug, can), a plain box,
# a thin pad too wide to grasp across its face (sponge), and a tet-mesh kind (sponge). The former
# procedural-rod entry (power_cable) was removed 2026-08-11 when cables left the pipeline's scope.
# Widths below are measured from the mesh at run time, not written here.
FIXTURES: dict = {
    "banana": (
        FixtureGrasp("mid_span_top", (0.0, 0.0, 0.0), "-z", "y", labels=("mid_span",),
                     notes="straight down onto the widest face, across the middle of the shaft"),
        FixtureGrasp("near_end_top", (0.6, 0.0, 0.0), "-z", "y", labels=("near_end",)),
        FixtureGrasp("mid_span_side", (0.0, 0.0, 0.0), "-y", "z", labels=("mid_span", "side")),
    ),
    "mug": (
        # The body is wider than the jaw in every direction, so the only fixture grasps are wall
        # pinches near the rim (-z is the open side). This is itself a useful case for a consumer:
        # an object with no whole-body grasp.
        FixtureGrasp("rim_wall_px", (0.85, 0.0, -0.8), "+z", "x", clearance=0.004, local=True,
                     labels=("wall", "rim"), notes="pinch the wall near the rim, +x side"),
        FixtureGrasp("rim_wall_py", (0.0, 0.85, -0.8), "+z", "y", clearance=0.004, local=True,
                     labels=("wall", "rim")),
    ),
    "tomato_soup_can": (
        FixtureGrasp("mid_across_y", (0.0, 0.0, 0.0), "-z", "y", labels=("mid_span",),
                     notes="across the barrel at mid height (x is the can axis)"),
        FixtureGrasp("mid_across_z", (0.0, 0.0, 0.0), "-y", "z", labels=("mid_span",)),
        FixtureGrasp("near_end_across_y", (0.7, 0.0, 0.0), "-z", "y", labels=("near_end",)),
    ),
    "sugar_box": (
        FixtureGrasp("mid_across_thin", (0.0, 0.0, 0.0), "-y", "z", labels=("mid_span", "thin_axis"),
                     notes="across the box's thin axis; across y would exceed the jaw"),
        FixtureGrasp("near_end_across_thin", (0.6, 0.0, 0.0), "-y", "z", labels=("near_end",)),
    ),
    "sponge": (
        FixtureGrasp("mid_across_thin", (0.0, 0.0, 0.0), "-y", "z", labels=("mid_span", "thin_axis"),
                     notes="a thin pad: only the thin axis fits the jaw"),
        FixtureGrasp("near_end_across_thin", (0.7, 0.0, 0.0), "-y", "z", labels=("near_end",)),
    ),
}
