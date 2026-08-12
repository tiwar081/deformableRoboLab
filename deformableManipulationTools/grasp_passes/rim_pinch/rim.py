"""Tracing a container's rim from a ``vlm_regions`` seed, and measuring the lip to pinch.

The store says WHERE a rim is — a ball a VLM pointed at, one per view that saw it, with no extent
and no reliable axis. This turns that locator into a run of pinch sites along the actual edge.

Everything rests on one local fact: **a rim is a thin sheet, and its thinnest direction is the wall
thickness.** Fit a small surface patch around any point on a lip and the smallest principal
direction of that patch is across the wall, whichever face the point sits on — outer wall, inner
wall, or the annular top. That direction is the jaw axis, and it is the reliable one precisely
because the thickness (3–6 mm here) is far smaller than the patch. It is measured, not taken from
the region's stored ``normal``, which is a mean surface normal at the picks and points radially
outward on one region and straight up the annulus on the next — see the mug, whose four rim regions
disagree by 90 degrees.

The other two axes come from the body rather than from the patch, because a spherical patch on a
rim is about equally wide along the edge and down the wall, so the second and third principal
directions are a coin toss. Instead: the DESCENT is the direction from the point toward the body
centre, projected into the plane across the wall — for a rim that is "down the wall" whichever way
the asset happens to be canonically oriented (the mug's canonical frame has it upside down). The
TANGENT completes the frame and is the direction the walk follows.

The walk then steps along the tangent, snaps back onto the surface, and re-fits — following a curve
it never has to model. Two things end it: a sample that drifts out of the seed's height band has
left the rim, and one whose wall exceeds :attr:`RimPinchConfig.max_wall` has reached the solid body.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RimPinchConfig

_EPS = 1.0e-12


@dataclass(frozen=True)
class RimSample:
    """One pinch site on a lip, in canonical-frame coordinates.

    ``point`` is the mid-thickness centre of the wall, ``jaw`` the outward across-thickness axis the
    pads close along, ``approach`` the descent into the body, and ``thickness`` the measured wall.
    ``arc`` is the signed distance walked from the seed region's centre — 0 for the seed itself —
    which is what makes the ids stable and readable."""
    point: np.ndarray
    jaw: np.ndarray
    approach: np.ndarray
    tangent: np.ndarray
    thickness: float
    arc: float
    region_id: str
    region_label: str


class RimTracer:
    """Local wall geometry on one prepared mesh. Built once per asset, queried per sample."""

    def __init__(self, mesh, cfg: RimPinchConfig, body_centre):
        from scipy.spatial import cKDTree

        self.mesh = mesh
        self.cfg = cfg
        self.centre = np.asarray(body_centre, dtype=float).reshape(3)
        self.points = np.asarray(mesh.vertices, dtype=float)
        self.tree = cKDTree(self.points)

    # ---- local frame -------------------------------------------------------------------------------
    def frame_at(self, p):
        """``(jaw, approach, tangent)`` at a surface point, or None if no wall can be fitted.

        ``jaw`` points OUTWARD (away from the body centre) so the thickness probe can start outside
        the material; ``approach`` points into the body, which is the direction the hand descends."""
        cfg = self.cfg
        p = np.asarray(p, dtype=float).reshape(3)
        radius = cfg.patch_radius
        idx: list = []
        for _ in range(cfg.patch_growth_steps + 1):
            idx = self.tree.query_ball_point(p, radius)
            if len(idx) >= cfg.min_patch_points:
                break
            radius *= cfg.patch_growth
        if len(idx) < cfg.min_patch_points:
            return None

        patch = self.points[idx]
        block = patch - patch.mean(axis=0)
        # Symmetric 3x3: eigh is deterministic and orders eigenvalues ascending, so column 0 is the
        # thin direction. This is the whole trick — see the module docstring.
        _vals, vecs = np.linalg.eigh(block.T @ block)
        jaw = vecs[:, 0]

        outward = p - self.centre
        if float(jaw @ outward) < 0.0:
            jaw = -jaw                      # point away from the body, so the probe starts outside

        # Descent: toward the body centre, with the across-wall component removed.
        inward = self.centre - p
        approach = inward - float(inward @ jaw) * jaw
        n = float(np.linalg.norm(approach))
        if n < 1.0e-6:
            return None                     # directly "above" the centre along the wall normal: the
        approach = approach / n             # descent is undefined here, so decline rather than guess
        tangent = np.cross(jaw, approach)
        tn = float(np.linalg.norm(tangent))
        if tn < 1.0e-6:
            return None
        return jaw, approach, tangent / tn

    # ---- wall thickness ----------------------------------------------------------------------------
    def wall_at(self, p, jaw):
        """``(mid-thickness centre, thickness)`` of the wall at ``p``, or None.

        Cast from just outside the surface straight back along the jaw axis and take the FIRST TWO
        crossings: those are the near and far faces of this wall. Deliberately not the full span —
        a third crossing is the opposite side of the container, and a pinch that spanned to it would
        be a grasp of the whole body, which is the regime that does not fit this gripper."""
        cfg = self.cfg
        origin = np.asarray(p, dtype=float) + cfg.probe_standoff * jaw
        hits, _ray, _tri = self.mesh.ray.intersects_location(
            origin[None, :], (-jaw)[None, :], multiple_hits=True)
        if len(hits) < 2:
            return None
        t = np.sort((hits - origin) @ (-jaw))
        t = t[t > 0.0]
        if len(t) < 2:
            return None
        thickness = float(t[1] - t[0])
        if not (cfg.min_wall <= thickness <= cfg.max_wall):
            return None
        centre = origin - 0.5 * (t[0] + t[1]) * jaw
        return centre, thickness

    def toward_body(self, p):
        """Unit direction from a surface point to the body centre — the pre-frame notion of "down"."""
        d = self.centre - np.asarray(p, dtype=float)
        n = float(np.linalg.norm(d))
        return None if n < 1.0e-9 else d / n

    def site_at(self, p):
        """``(point, jaw, approach, tangent, thickness)`` for the lip at or just below ``p``.

        Tries the point as given first, then progressively further down the wall
        (:attr:`RimPinchConfig.wall_search_depths`) — a seed on a broad flat lip has no measurable
        thickness where it sits, because the patch there is a horizontal plate whose thin direction
        points down through the vessel. Returns None if no depth yields a wall."""
        down = self.toward_body(p)
        if down is None:
            return None
        for depth in self.cfg.wall_search_depths:
            q = np.asarray(p, dtype=float) if depth == 0.0 else self.snap(p + depth * down)
            frame = self.frame_at(q)
            if frame is None:
                continue
            jaw, approach, tangent = frame
            wall = self.wall_at(q, jaw)
            if wall is None:
                continue
            centre, thickness = wall
            return centre, jaw, approach, tangent, thickness
        return None

    def snap(self, q):
        """Nearest point on the surface to ``q`` — how the walk stays on the wall while following a
        curve it never models."""
        import trimesh

        pt, _dist, _tri = trimesh.proximity.closest_point(self.mesh, np.asarray(q).reshape(1, 3))
        return np.asarray(pt[0], dtype=float)

    # ---- the walk ------------------------------------------------------------------------------------
    def walk(self, region) -> list:
        """Pinch sites along the rim through one region, ordered by signed arc.

        The seed is the region's centre snapped onto the surface; the walk then steps along the
        tangent in both directions. It stops when the wall stops looking like a lip, so a seed that
        the VLM put somewhere that is not actually an edge yields nothing rather than a run of
        candidates on the body."""
        cfg = self.cfg
        seed = self.snap(np.asarray(region.center, dtype=float))
        first = self.site_at(seed)
        if first is None:
            return []
        descent0 = first[2]

        out = []
        for direction in (0, +1, -1):
            steps = 1 if direction == 0 else int(cfg.max_arc / cfg.sample_spacing)
            # The CURSOR walks the surface at the seed's own height; the site measured at each stop
            # may sit a little below it (see site_at). Keeping them separate stops a per-step
            # correction from accumulating into a slow slide down the wall.
            cursor = seed.copy()
            for step in range(steps):
                if direction != 0:
                    site = self.site_at(cursor)
                    if site is None:
                        break
                    cursor = self.snap(cursor + direction * cfg.sample_spacing * site[3])
                    # Left the seed's height band -> this is no longer the same rim.
                    if abs(float((cursor - seed) @ descent0)) > cfg.rim_band:
                        break
                site = self.site_at(cursor)
                if site is None:
                    break                   # too thick to be a lip, or no second face: off the rim
                centre, jaw, approach, tangent, thickness = site
                out.append(RimSample(
                    point=centre, jaw=jaw, approach=approach, tangent=tangent,
                    thickness=thickness,
                    arc=direction * (step + 1) * cfg.sample_spacing if direction else 0.0,
                    region_id=region.id, region_label=region.label))
        return sorted(out, key=lambda s: s.arc)
