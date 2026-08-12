"""Method 1 — grasps perpendicular to a medial-axis skeleton.

The medial axis is the locus of centres of maximal inscribed spheres, so a skeleton node carries
exactly the quantity a parallel jaw cares about: the LOCAL RADIUS of the solid there. Where twice
that radius fits the jaw, the object is locally a graspable tube, and the grasp that closes on it
is the one whose jaws lie perpendicular to the local axis.

Two backends produce the skeleton, chosen by whether the welded surface came out closed:

* **``skeletor.skeletonize.by_wavefront``** (watertight) — the mesh-native method. It propagates a
  wave over the surface graph and takes each vertex ring's centre and radius, so radii come out of
  the same pass as the nodes and are true surface distances. (It re-seeds numpy's legacy global RNG
  to a fixed constant internally, which is why it is reproducible without our passing ``origins``.)
* **voxelize + ``skimage.morphology.skeletonize``** (not watertight) — the fallback. Rasterize to an
  occupancy grid, thin it to a one-voxel-wide skeleton, and read radii off the Euclidean distance
  transform, which IS the inscribed-sphere radius at each skeleton voxel. Topology-blind, so a
  surface with holes (the vomp bins and bucket) skeletonizes fine where the mesh method cannot run.

Both backends hand back the same :class:`MedialAxis`, and the local axis is fitted the same way for
both — by PCA over each node's skeleton neighbourhood — so a candidate's geometry does not depend on
which one ran. That fit also yields the LINEARITY of the neighbourhood, which is what lets the
method decline to answer where there is no local axis: the medial set of a box is a sheet and of a
junction a branch, and "perpendicular to the local axis" means nothing at either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...grasp_library import MAX_JAW_WIDTH
from .config import GeometricConfig
from .meshprep import PreparedMesh


@dataclass(frozen=True)
class MedialAxis:
    """A skeleton reduced to what the grasp sampler needs, in canonical-frame coordinates.

    ``points`` are the retained skeleton nodes, ``radii`` the local inscribed-sphere radius at each,
    ``tangents`` the unit local axis, and ``linearity`` the PCA measure of how curve-like (rather
    than sheet- or junction-like) the node's neighbourhood is. ``backend`` names which method ran."""
    points: np.ndarray            # (n,3)
    radii: np.ndarray             # (n,)
    tangents: np.ndarray          # (n,3)
    linearity: np.ndarray         # (n,)
    backend: str
    detail: str = ""

    def __len__(self) -> int:
        return len(self.points)


# =================================================================================================
# Backends
# =================================================================================================
def _skeleton_from_mesh(prepared: PreparedMesh):
    """``skeletor`` wavefront skeleton of a watertight mesh -> (points, radii)."""
    import skeletor as sk

    skel = sk.skeletonize.by_wavefront(prepared.mesh, waves=1, step_size=1, progress=False)
    pts = np.asarray(skel.vertices, dtype=float).reshape(-1, 3)
    rad = np.asarray(skel.radius, dtype=float).reshape(-1)
    good = np.isfinite(rad) & np.isfinite(pts).all(axis=1) & (rad > 0.0)
    return pts[good], rad[good], f"skeletor.by_wavefront, {int(good.sum())}/{len(pts)} nodes usable"


def _skeleton_from_voxels(prepared: PreparedMesh, cfg: GeometricConfig):
    """Voxelized + thinned skeleton of an open surface -> (points, radii).

    Radii come from the Euclidean distance transform of the occupancy grid, which at a skeleton
    voxel is by definition the radius of the largest sphere that fits inside the solid there — the
    same quantity the wavefront backend reports, obtained a different way."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    grid = prepared.span_probe.grid          # already built (and filled where possible) by VoxelSpan
    occupancy = np.asarray(grid.matrix, dtype=bool)
    if not occupancy.any():
        return np.zeros((0, 3)), np.zeros(0), "voxel grid empty"

    thinned = skeletonize(occupancy)
    edt = ndimage.distance_transform_edt(occupancy) * cfg.voxel_pitch
    idx = np.argwhere(thinned)
    if not len(idx):
        return np.zeros((0, 3), dtype=float), np.zeros(0), "thinning produced no skeleton"

    # Voxel indices -> canonical points, through the grid's own placement matrix.
    xform = np.asarray(grid.transform, dtype=float)
    pts = idx @ xform[:3, :3].T + xform[:3, 3]
    rad = edt[thinned]
    return (np.asarray(pts, dtype=float), np.asarray(rad, dtype=float),
            f"voxelize({cfg.voxel_pitch * 1000:.0f} mm) + skimage.skeletonize, {len(idx)} voxels")


# =================================================================================================
# Local axis
# =================================================================================================
def _local_axes(points, cfg: GeometricConfig):
    """Unit local axis and linearity at each node, by PCA over its skeleton neighbourhood.

    Returns ``(tangents, linearity)``. A node with fewer than three neighbours gets linearity 0 (no
    axis can be fitted) and is dropped by the caller's gate rather than given a made-up direction."""
    from scipy.spatial import cKDTree

    n = len(points)
    tangents = np.zeros((n, 3))
    linearity = np.zeros(n)
    if n < 3:
        return tangents, linearity

    tree = cKDTree(points)
    for i, neigh in enumerate(tree.query_ball_point(points, cfg.tangent_neighbourhood)):
        if len(neigh) < 3:
            continue
        block = points[neigh] - points[neigh].mean(axis=0)
        # Symmetric 3x3 -> eigh is deterministic and ordered ascending; take the dominant direction.
        vals, vecs = np.linalg.eigh(block.T @ block)
        total = float(vals.sum())
        if total <= 0.0:
            continue
        axis = vecs[:, -1]
        # Sign is arbitrary out of eigh; fix it deterministically so re-runs agree bit for bit.
        k = int(np.argmax(np.abs(axis)))
        if axis[k] < 0:
            axis = -axis
        tangents[i] = axis
        linearity[i] = float(vals[-1]) / total
    return tangents, linearity


def _thin_by_spacing(points, order, spacing: float):
    """Greedily keep nodes at least ``spacing`` apart, visiting them in ``order``.

    Deterministic by construction: the visit order is supplied, not discovered."""
    kept = []
    kept_pts = np.zeros((0, 3))
    for i in order:
        if len(kept_pts) and float(np.linalg.norm(kept_pts - points[i], axis=1).min()) < spacing:
            continue
        kept.append(int(i))
        kept_pts = np.vstack([kept_pts, points[i]])
    return np.asarray(kept, dtype=int)


def medial_axis(prepared: PreparedMesh, cfg: GeometricConfig) -> MedialAxis:
    """Skeletonize an asset and reduce it to well-spaced nodes that have a local axis."""
    if prepared.watertight:
        pts, rad, detail = _skeleton_from_mesh(prepared)
        backend = "skeletor_wavefront"
    else:
        pts, rad, detail = _skeleton_from_voxels(prepared, cfg)
        backend = "voxel_skeletonize"

    if not len(pts):
        return MedialAxis(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0),
                          backend, detail)

    # Deterministic node order: lexicographic on rounded coordinates. Everything downstream (the
    # spacing thin, the roll sampling, the id numbering) inherits it, so the pass is reproducible
    # regardless of what order a backend happened to emit nodes in.
    key = np.round(pts, 6)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    pts, rad = pts[order], rad[order]

    tangents, linearity = _local_axes(pts, cfg)

    # Gate BEFORE thinning: a node that cannot host a grasp should not occupy a spacing slot and
    # push out a neighbour that could.
    fits = 2.0 * rad + cfg.clearance <= _max_width()
    usable = np.flatnonzero(fits & (linearity >= cfg.min_linearity))
    if not len(usable):
        return MedialAxis(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0), backend,
                          f"{detail}; no node both fits the jaw and has a local axis")

    keep = _thin_by_spacing(pts, usable, cfg.node_spacing)
    return MedialAxis(points=pts[keep], radii=rad[keep], tangents=tangents[keep],
                      linearity=linearity[keep], backend=backend,
                      detail=f"{detail}; {len(keep)} node(s) kept of {len(pts)}")


def _max_width() -> float:
    from ...grasp_library import MAX_JAW_WIDTH
    return MAX_JAW_WIDTH
