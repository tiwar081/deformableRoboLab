"""Method 2 — grasps from a cross-section width sweep.

Slice the object with parallel planes, and in each 2-D cross-section look for what a parallel jaw
actually needs: two stretches of boundary that FACE EACH OTHER and are no further apart than the
jaw opens. Reduced to 2-D the test is exact and cheap — walk the section's boundary, and from each
sample fire a ray straight into the material along the inward normal. Wherever it strikes boundary
whose outward normal points back along the ray, those two facets are opposing, and the distance
between them is the width the jaws would close to.

``trimesh.Trimesh.section_multiplane`` does the slicing (one call per axis, all heights at once) and
hands back ``Path2D`` sections together with the ``to_3D`` transform that puts them back where they
came from. It works on open surfaces too — the shapely polygons it builds close small gaps — so
unlike the skeleton method this one needs no fallback.

**The sweep runs along all three canonical axes, and that is what makes the approach directions
complete.** A grasp found in a section has its jaw axis lying IN the section plane, and the approach
is then a roll about that jaw axis; this samples the two rolls that also lie in the plane, where the
geometry was actually measured. Any approach excluded by one sweep axis is found by another — a box
sliced along z yields side approaches closing across x, and the top-down approach closing across
that same x comes from the sweep along y.

Interior rings count as boundary. A mug section is an annulus, and the pair "outer wall, inner wall"
is a real grasp — one pad in the cup, one outside — that a search over the outline alone would miss.
Every hit along the ray is considered, not just the first, so the same sample also yields the wider
grasp that spans the whole section with the void between the jaws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeometricConfig
from .meshprep import PreparedMesh

_EPS = 1.0e-12
# Two ray hits closer together than this [m] are the same contact seen through the two segments that
# meet at a boundary vertex.
_HIT_MERGE = 1.0e-7


@dataclass(frozen=True)
class AntipodalPair:
    """One opposing-face pair found in a cross-section, already lifted to canonical 3-D.

    ``centre`` is the midpoint of the two contacts, ``jaw`` the unit closing axis (pointing from the
    centre toward the first contact), ``approach`` one of the two in-plane directions perpendicular
    to it, and ``width`` the measured separation. ``opposing_deg`` is how far the two outward normals
    fall short of exactly antiparallel — 0 for a perfect slab."""
    centre: np.ndarray
    jaw: np.ndarray
    approach: np.ndarray
    width: float
    opposing_deg: float
    axis: int
    height_index: int
    height: float


# =================================================================================================
# 2-D boundary sampling
# =================================================================================================
def _resample_ring(coords, spacing: float):
    """Walk a closed ring at fixed arc length -> (points (k,2), outward normals (k,2)).

    The ring must already be oriented with MATERIAL ON THE LEFT of travel (shapely's ``orient`` with
    a positive sign does this: exterior counter-clockwise, holes clockwise). The outward normal —
    pointing away from the material — is then the right-hand perpendicular of the edge direction."""
    ring = np.asarray(coords, dtype=float)[:, :2]
    if len(ring) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if np.linalg.norm(ring[0] - ring[-1]) > _EPS:
        ring = np.vstack([ring, ring[0]])

    edges = ring[1:] - ring[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    live = lengths > _EPS
    if not live.any():
        return np.zeros((0, 2)), np.zeros((0, 2))
    edges, lengths, starts = edges[live], lengths[live], ring[:-1][live]

    # Right-hand perpendicular of the travel direction = away from the material on the left.
    normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / lengths[:, None]

    ends = np.cumsum(lengths)
    total = float(ends[-1])
    count = max(int(np.ceil(total / spacing)), 3)
    # Half-open, offset by half a step: samples land mid-interval, never on a vertex where the
    # normal is ambiguous.
    at = (np.arange(count) + 0.5) * (total / count)
    seg = np.clip(np.searchsorted(ends, at), 0, len(lengths) - 1)
    before = np.concatenate([[0.0], ends[:-1]])[seg]
    pts = starts[seg] + ((at - before) / lengths[seg])[:, None] * edges[seg]
    return pts, normals[seg]


def _ring_segments(coords):
    """A ring as (A (m,2), B (m,2), outward normals (m,2)) — the ray target set."""
    ring = np.asarray(coords, dtype=float)[:, :2]
    if len(ring) < 2:
        return None
    if np.linalg.norm(ring[0] - ring[-1]) > _EPS:
        ring = np.vstack([ring, ring[0]])
    a, b = ring[:-1], ring[1:]
    e = b - a
    ln = np.linalg.norm(e, axis=1)
    live = ln > _EPS
    if not live.any():
        return None
    a, b, e, ln = a[live], b[live], e[live], ln[live]
    n = np.column_stack([e[:, 1], -e[:, 0]]) / ln[:, None]
    return a, b, n


def _antipodal_in_polygon(polygon, cfg: GeometricConfig, max_width: float):
    """Opposing-face pairs inside one shapely polygon (exterior + holes).

    Yields ``(centre2d, jaw2d, width, opposing_deg)``."""
    from shapely.geometry.polygon import orient

    poly = orient(polygon, 1.0)                       # exterior CCW, holes CW -> material on the left
    rings = [poly.exterior] + list(poly.interiors)

    samples, sample_normals, seg_a, seg_b, seg_n = [], [], [], [], []
    for ring in rings:
        p, n = _resample_ring(ring.coords, cfg.boundary_spacing)
        if len(p):
            samples.append(p)
            sample_normals.append(n)
        segs = _ring_segments(ring.coords)
        if segs is not None:
            seg_a.append(segs[0])
            seg_b.append(segs[1])
            seg_n.append(segs[2])
    if not samples or not seg_a:
        return []

    p = np.vstack(samples)                            # (n,2) contact candidates
    pn = np.vstack(sample_normals)                    # (n,2) outward normals there
    a = np.vstack(seg_a)                              # (m,2)
    e = np.vstack(seg_b) - a                          # (m,2)
    sn = np.vstack(seg_n)                             # (m,2)

    d = -pn                                           # fire INTO the material

    # Drop samples whose ray does not actually enter the solid — it runs ALONG the boundary instead.
    # A sample landing ON a vertex takes the normal of one of the two edges meeting there and then
    # fires parallel to, and flush against, the other. The segment maths happily reports the far end
    # of that second edge as an "opposing contact": on a 50 mm cube, a 50 mm chord lying exactly on
    # a face, whose jaws would close on nothing and slide off. (Not a rare accident — it happens
    # whenever the boundary length is close to a multiple of the sample spacing, which for a box it
    # exactly is.)
    #
    # The test is to step a short way along the ray and require that point to be inside the material
    # BY A MARGIN. Containment alone is not enough: the grazing probe misses the boundary by ~1e-11,
    # so shapely rightly calls it inside. The margin is what distinguishes a contact that bites from
    # one that skims. One test per SAMPLE, not per hit.
    import shapely

    depth = 0.25 * cfg.min_span
    probe = p + depth * d
    enters = (np.asarray(shapely.contains_xy(poly, probe[:, 0], probe[:, 1]))
              & (np.asarray(shapely.distance(poly.boundary, shapely.points(probe))) >= 0.5 * depth))
    if not enters.any():
        return []

    # Solve p + t d = a + s e for every (sample, segment) pair.  t = (w x e)/(d x e),
    # s = (w x d)/(d x e), with (u x v) = u_x v_y - u_y v_x.
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    w = a[None, :, :] - p[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / denom
        s = (w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]) / denom

    cos_tol = np.cos(np.radians(cfg.antipodal_angle_deg))
    # Opposing faces have OPPOSITE outward normals: the far facet's normal points back along the ray.
    opposing = (pn @ sn.T) <= -cos_tol
    valid = (np.abs(denom) > _EPS) & (s >= -1e-9) & (s <= 1 + 1e-9)
    valid &= opposing & (t >= cfg.min_span) & (t + cfg.clearance <= max_width)
    valid &= enters[:, None]

    ii, jj = np.nonzero(valid)
    if not len(ii):
        return []
    tt = t[ii, jj]
    order = np.lexsort((tt, ii))                      # by sample, then by distance along the ray
    ii, jj, tt = ii[order], jj[order], tt[order]
    # Drop the duplicate hit produced where two segments meet at a boundary vertex.
    keep = np.ones(len(ii), dtype=bool)
    keep[1:] = (ii[1:] != ii[:-1]) | (np.abs(tt[1:] - tt[:-1]) > _HIT_MERGE)

    out = []
    for i, j, width in zip(ii[keep], jj[keep], tt[keep]):
        n_here = pn[i]
        centre = p[i] + 0.5 * width * d[i]
        # Jaw axis points from the centre out to the sampled contact; sign is fixed later.
        cosang = float(np.clip(-float(n_here @ sn[j]), -1.0, 1.0))
        out.append((centre, n_here, float(width), float(np.degrees(np.arccos(cosang)))))
    return out


# =================================================================================================
# The sweep
# =================================================================================================
def cross_section_pairs(prepared: PreparedMesh, cfg: GeometricConfig, max_width: float) -> list:
    """Sweep all three canonical axes and return every opposing-face pair, as :class:`AntipodalPair`.

    Two pairs are emitted per geometric find — the same jaw axis approached from either in-plane
    side — because which side the hand comes from is a different grasp, not a different description
    of one."""
    mesh = prepared.mesh
    centre = mesh.bounds.mean(axis=0)
    extents = mesh.extents
    found = []

    for axis in range(3):
        normal = np.zeros(3)
        normal[axis] = 1.0
        half = 0.5 * float(extents[axis]) * (1.0 - 2.0 * cfg.slice_margin)
        if half <= 0.0:
            continue
        heights = np.linspace(-half, half, cfg.slices_per_axis)
        try:
            sections = mesh.section_multiplane(plane_origin=centre, plane_normal=normal,
                                               heights=heights)
        except Exception as exc:                      # noqa: BLE001 - one axis failing is survivable
            raise RuntimeError(
                f"{prepared.name}: section_multiplane failed on axis {'xyz'[axis]}: "
                f"{type(exc).__name__}: {exc}") from exc

        for height_index, (height, section) in enumerate(zip(heights, sections)):
            if section is None:
                continue
            to_3d = np.asarray(section.metadata["to_3D"], dtype=float)
            rot, off = to_3d[:3, :3], to_3d[:3, 3]
            for polygon in section.polygons_full:
                for c2, n2, width, opposing in _antipodal_in_polygon(polygon, cfg, max_width):
                    centre3 = rot @ np.array([c2[0], c2[1], 0.0]) + off
                    jaw3 = rot @ np.array([n2[0], n2[1], 0.0])
                    # In-plane perpendicular to the jaw: the roll the section actually measured.
                    perp3 = rot @ np.array([-n2[1], n2[0], 0.0])
                    jaw3 = jaw3 / max(float(np.linalg.norm(jaw3)), _EPS)
                    perp3 = perp3 / max(float(np.linalg.norm(perp3)), _EPS)
                    for sign in (1.0, -1.0):
                        found.append(AntipodalPair(
                            centre=centre3, jaw=jaw3, approach=sign * perp3, width=width,
                            opposing_deg=opposing, axis=axis, height_index=height_index,
                            height=float(height)))
    return found
