"""Tuning knobs for the ``geometric`` pass, in one frozen dataclass with the reasons attached.

These are OFFLINE GENERATOR parameters — how finely to sample geometry and what counts as a
graspable feature — not physics. Physics lives in ``deformableManipulationTools/params.py``; nothing
here is read at simulation time. The one physical constant the pass depends on is imported rather
than restated: :data:`grasp_library.MAX_JAW_WIDTH`.

Every length is in metres.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricConfig:
    """Sampling and acceptance parameters for both methods.

    The defaults are sized to the catalog's rigid objects (YCB-scale meshes, 5–27 cm across, ~15k
    triangles) and produce a few dozen candidates per asset per method — enough to cover an object's
    distinct graspable features, few enough that a downstream evaluation pass can simulate them all.
    """

    # ---- shared -------------------------------------------------------------------------------
    clearance: float = 0.005
    """Feasibility margin on the jaw [m]. A grasp is kept only when ``span + clearance <=
    MAX_JAW_WIDTH``: the jaws must open WIDER than the object to get around it, so a span of exactly
    80 mm is not a grasp the Franka can make. The recorded ``width`` is the contact span itself (see
    the module docstring) — this margin gates, it does not pad."""

    min_span: float = 0.003
    """Spans below this [m] are discarded as mesh noise rather than emitted as 1 mm pinches. Roughly
    the triangle scale of a YCB scan."""

    subdivide_max_edge: float = 0.006
    """Triangles longer than this [m] are subdivided before analysis. Procedural kinds
    (``rigid_box``, ``rubiks_cube``) are 12 triangles; skeletonization and the boundary resampling
    both need vertices to work with. Subdivision is deterministic and never moves the surface."""

    # ---- method 1: medial-axis skeleton ---------------------------------------------------------
    node_spacing: float = 0.015
    """Minimum arc distance [m] between the skeleton nodes kept as grasp sites. A raw skeleton has a
    node every few millimetres; without thinning, one tube yields dozens of grasps that differ by
    less than the pad is wide."""

    roll_samples: int = 4
    """Approach directions sampled around the local axis at each node. The jaw axis is fixed
    perpendicular to the local axis, leaving the approach free to roll about it: this samples that
    roll uniformly over the full circle (4 -> every 90 deg). The jaw axis follows as
    ``tangent x approach``, so no two samples describe the same grasp."""

    min_linearity: float = 0.55
    """PCA linearity ``L0 / (L0+L1+L2)`` a node's skeleton neighbourhood must reach to be used. At a
    branch point or on a medial SHEET (the interior of a box is a sheet, not a curve) there is no
    well-defined local axis, so "perpendicular to the local axis" is meaningless — those nodes are
    dropped rather than given an arbitrary tangent. 0.55 keeps gently curving tubes and rejects
    junctions."""

    max_centring_error: float = 0.25
    """How far off-centre the skeleton node may sit in the chord the jaws close on, as a fraction of
    the span.

    **This is on the JAW axis and is not related to pad seating.** Placement along the APPROACH is
    ``grasp_library.pad_seat``'s job and this pass adds nothing of its own there (rule 2). This is a
    REJECTION test on a different axis entirely: it decides whether a candidate is emitted at all,
    and moves nothing.

    It is the medial method CHECKING ITS OWN PREMISE. A node's radius describes the
    solid around it only where the node is the centre of the local cross-section — true of a tube by
    definition, false wherever the skeleton is really a sheet or the backend's radius estimate is
    off. The measured chord always passes through the node (the probe fires from it), so a node
    sitting far from the chord's midpoint says the radius did not describe THIS direction.

    Concretely, this is what rejects a corner nip. The YCB ``wood_block`` is 90 mm thick — nothing
    about it fits an 80 mm jaw — but the wavefront ring radius collapses near the block's ends, so
    the jaw gate passes there and a ray fired across a corner does find a 35 mm chord between two
    PERPENDICULAR faces, which the pads would simply cam off. Its midpoint is nowhere near the
    skeleton, so this rejects it and the block correctly yields nothing."""

    tangent_neighbourhood: float = 0.03
    """Radius [m] of the skeleton-node neighbourhood the local tangent and linearity are fitted
    over. Wide enough to average out skeleton jitter, short enough to follow a banana's curve."""

    voxel_pitch: float = 0.004
    """Voxel size [m] for the non-watertight fallback. 4 mm resolves the catalog's thinnest graspable
    rigid feature (a bucket wall) while keeping the grid under ~10^5 cells for a 30 cm object."""

    # ---- method 2: cross-section width sweep ----------------------------------------------------
    slices_per_axis: int = 12
    """Cross-sections taken along each of the three canonical axes. 12 puts a slice every ~8% of an
    extent, which resolves a bottle's neck/shoulder/body as separate features."""

    boundary_spacing: float = 0.004
    """Arc-length spacing [m] of the samples walked around each cross-section boundary. Each sample
    is one candidate contact point; 4 mm is well under the ~20 mm Franka pad, so no flat facet wide
    enough to grasp is stepped over."""

    antipodal_angle_deg: float = 20.0
    """How near-antiparallel the two contacts' outward normals must be [deg] for the faces to count
    as opposing. This is a FRICTION bound, not an arbitrary tolerance: a contact resists slip while
    the surface normal lies within ``atan(mu)`` of the grasp line. The catalog's rigid objects carry
    ``mu`` 0.40–0.50, coupled to the rubber pad's 0.8 by the framework's geometric-mean law
    (``sqrt(0.8 * 0.4) = 0.57``, ``sqrt(0.8 * 0.5) = 0.63``), giving cone half-angles of 29.5–32.3
    deg. 20 deg therefore stays inside the least-frictional catalog object's cone with margin, so an
    accepted pair is opposing in a sense the simulator will agree with."""

    slice_margin: float = 0.02
    """Fraction of the extent trimmed off each end of a sweep axis. The extreme slices of a closed
    body degenerate to points or slivers whose "faces" are meaningless."""

    # ---- deduplication and budget ---------------------------------------------------------------
    dedup_position: float = 0.006
    """Grasp centres closer than this [m] AND aligned in direction are treated as one candidate."""

    dedup_direction: float = 0.25
    """Direction quantum used with :attr:`dedup_position` (unit-vector components rounded to this
    step, ~14 deg). Coarse on purpose: two grasps a pad-width apart in position and a few degrees
    apart in roll are the same grasp for every downstream purpose."""

    max_per_method: int = 64
    """Cap on the candidates each method contributes. Reached only on large, feature-rich meshes;
    the surplus is thinned by farthest-point selection in pose space (deterministic), so the cap
    costs coverage of near-duplicates rather than coverage of distinct features."""


DEFAULT = GeometricConfig()
