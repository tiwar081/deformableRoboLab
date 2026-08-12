"""A parallel-jaw gripper marker mesh, for drawing candidate poses on an object.

Adapted from ``create_gripper_marker`` in NVIDIA's ACRONYM tools (``acronym_tools/acronym.py``, MIT
licence, Copyright (c) 2020 NVIDIA Corporation) — the same four-cylinder stick figure their dataset
is visualized with, so a candidate rendered here looks like a candidate rendered there.

VENDORED rather than imported: ``_external/`` is a reference checkout that the repo must keep
working without (see AGENTS.md), and this is forty lines of ``trimesh.creation`` calls with no state.

Two deliberate changes to the original:

* **The jaw half-width is a parameter.** ACRONYM hard-codes the fingers at ``x = +/-0.041`` — the
  Franka's maximum opening — because their marker illustrates the hand, not the grasp. Ours is
  drawn at the candidate's OWN ``width``, so how far the jaws are actually open is visible.
* **The origin is moved to the TCP.** ACRONYM's marker frame sits at the hand base, with the finger
  tips :data:`grasp_library.ACRONYM_HAND_TO_TCP` further along ``+z``; our ``POSE_CONVENTION`` puts
  the origin at the grasp centre between the pads. Shifting the geometry by that documented offset
  means a candidate's ``transform`` can be applied to the marker directly, with no correction at the
  call site to get wrong.

The axes are unchanged and already match ours: ``+/-x`` jaws, ``+z`` approach.
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import ACRONYM_HAND_TO_TCP

# ACRONYM's finger geometry, verbatim: the finger cylinders run from z = 0.066 to the tip at
# z = 0.11217 (= ACRONYM_HAND_TO_TCP), behind which the wrist stem runs back to z = 0.
_FINGER_BASE_Z = 0.066
_TUBE_RADIUS = 0.002

# Method -> RGBA, so the two generators are told apart at a glance in both the GLB and the plot.
METHOD_COLORS = {
    "medial_axis": (0, 132, 255, 255),        # blue  — found from the inside, along a centre line
    "cross_section": (255, 96, 0, 255),       # orange — found from the outside, across a face pair
}
DEFAULT_COLOR = (128, 128, 128, 255)


def gripper_marker(width: float, color=DEFAULT_COLOR, tube_radius: float = _TUBE_RADIUS,
                   sections: int = 6):
    """A gripper stick figure opened to ``width`` [m], posed in ``tcp_z_approach_x_jaw_v1``.

    Origin at the grasp centre, ``+z`` the approach, jaws on ``+/-x``. Apply a candidate's
    ``transform`` to place it."""
    import trimesh

    half = 0.5 * float(width)
    tip = ACRONYM_HAND_TO_TCP
    # Shift so the finger TIPS (the pads, at z = tip) land on the origin.
    dz = -tip

    def cyl(a, b):
        return trimesh.creation.cylinder(radius=tube_radius, sections=sections,
                                         segment=[[a[0], a[1], a[2] + dz], [b[0], b[1], b[2] + dz]])

    parts = [
        cyl([0.0, 0.0, 0.0], [0.0, 0.0, _FINGER_BASE_Z]),                       # wrist stem
        cyl([-half, 0.0, _FINGER_BASE_Z], [half, 0.0, _FINGER_BASE_Z]),         # crossbar
        cyl([half, 0.0, _FINGER_BASE_Z], [half, 0.0, tip]),                     # +x finger
        cyl([-half, 0.0, _FINGER_BASE_Z], [-half, 0.0, tip]),                   # -x finger
    ]
    marker = trimesh.util.concatenate(parts)
    marker.visual.face_colors = np.asarray(color, dtype=np.uint8)
    return marker


def color_for(candidate, fallback=DEFAULT_COLOR):
    """The colour a candidate's method label earns it."""
    for label in candidate.labels:
        if label in METHOD_COLORS:
            return METHOD_COLORS[label]
    return fallback


def method_of(candidate) -> str:
    """The method label a candidate carries, or ``"?"`` if it has none."""
    for label in candidate.labels:
        if label in METHOD_COLORS:
            return label
    return "?"
