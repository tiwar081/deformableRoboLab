"""Mesh conditioning and the two width-measurement backends the ``geometric`` pass measures with.

One place decides how a catalog asset becomes an analysable surface, because both methods must see
the SAME geometry — a skeleton fitted to one mesh and cross-sections cut from another would put the
two methods' candidates in subtly different places.

Two things need doing to catalog geometry before either method works:

* **Weld it.** ``load_usd_mesh`` returns vertices per FACE-CORNER (a YCB scan arrives with 16.7k
  vertices for 15.7k triangles), so the surface is topologically 15k disconnected triangles. Every
  connectivity-based algorithm — skeletonization above all — sees dust. Welding coincident vertices
  turns the catalog's YCB meshes watertight; it is what makes the primary skeleton backend usable at
  all.
* **Refine it.** ``rigid_box`` and ``rubiks_cube`` are 12 triangles. Skeletonization and boundary
  resampling both need vertices to sample; subdivision adds them without moving the surface.

Width is then measured in one of two ways, chosen by whether welding produced a closed surface:

* :class:`RaySpan` (watertight) — cast a ray each way along the jaw axis and take the two first
  hits. Exact to the triangle.
* :class:`VoxelSpan` (not watertight) — march an occupancy grid outward until it empties. Accurate
  only to the voxel pitch, but a hole in the surface cannot make it report a span across empty
  space, which is exactly the failure a ray suffers on the vomp bin/bucket meshes.

The same watertight test picks the skeletonization backend in :mod:`.medial`, so an asset is
analysed end-to-end by one consistent set of tools.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeometricConfig

# A ray fired from a skeleton node starts strictly inside the solid, so any hit closer than this [m]
# is numerical noise on a coincident triangle rather than the far wall.
_RAY_EPS = 1.0e-6


@dataclass(frozen=True)
class PreparedMesh:
    """A catalog asset conditioned for analysis, in CANONICAL-frame coordinates.

    ``mesh`` is a welded (and, if it was coarse, subdivided) ``trimesh.Trimesh``. ``watertight``
    records whether welding closed the surface — it selects both the skeletonization backend and the
    width-measurement backend, and it is reported in every candidate's notes so a reader can tell
    which path produced a grasp."""
    name: str
    mesh: object                 # trimesh.Trimesh
    watertight: bool
    subdivided: bool
    span_probe: object           # RaySpan | VoxelSpan

    @property
    def backend(self) -> str:
        return "mesh" if self.watertight else "voxel"


def prepare(name: str, canonical_vertices, faces, cfg: GeometricConfig) -> PreparedMesh:
    """Weld, clean, and (if coarse) subdivide an asset's canonical-frame surface.

    ``faces`` must not be None — the ``geometric`` pass only runs on kinds that carry surface
    triangles, and :meth:`GeometricPass.applies_to` enforces that before we get here."""
    import trimesh

    if faces is None:
        raise ValueError(f"{name}: no surface triangles (point-set geometry); the geometric pass "
                         f"needs a surface to skeletonize and to section")

    mesh = trimesh.Trimesh(vertices=np.asarray(canonical_vertices, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3), process=False)
    # process=True on the constructor also merges, but doing it explicitly keeps the ORDER of
    # operations (and therefore the vertex indexing, and therefore the skeleton) reproducible.
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()          # consistent outward winding; the sweep reads face normals

    subdivided = False
    if len(mesh.faces) and float(mesh.edges_unique_length.max()) > cfg.subdivide_max_edge:
        # Only worth it for genuinely coarse geometry: a YCB scan's longest edge is already ~2 mm,
        # and subdividing 15k triangles would cost a lot for nothing.
        if len(mesh.faces) < 2000:
            mesh = mesh.subdivide_to_size(cfg.subdivide_max_edge)
            subdivided = True

    watertight = bool(mesh.is_watertight)
    probe = RaySpan(mesh) if watertight else VoxelSpan(mesh, cfg.voxel_pitch)
    return PreparedMesh(name=name, mesh=mesh, watertight=watertight, subdivided=subdivided,
                        span_probe=probe)


# =================================================================================================
# Width measurement
# =================================================================================================
class RaySpan:
    """Material span along a direction, by ray casting. Requires a closed surface.

    ``measure`` answers, for each (point, direction) pair, the question the jaws ask: starting from
    a point inside the solid, how far is it to the surface each way along the closing axis? The
    grasp centre is returned as the MIDPOINT of the two hits rather than the query point, because a
    skeleton node is the centre of the local cross-section, not of the particular chord the jaws
    happen to close along."""

    def __init__(self, mesh):
        self.mesh = mesh

    def measure(self, points, directions):
        """(n,3), (n,3) -> (span (n,), centre (n,3), ok (n,) bool). ``span``/``centre`` are
        meaningless where ``ok`` is False."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        n = len(p)

        # One batch: the +d hits then the -d hits, so trimesh's per-call overhead is paid twice, not
        # 2n times.
        origins = np.vstack([p, p])
        dirs = np.vstack([d, -d])
        loc, ray_idx, _tri = self.mesh.ray.intersects_location(origins, dirs, multiple_hits=False)

        hit = np.zeros((2 * n, 3))
        got = np.zeros(2 * n, dtype=bool)
        if len(ray_idx):
            # intersects_location drops missed rays and may reorder; index_ray maps back.
            hit[ray_idx] = loc
            got[ray_idx] = True

        fwd, bwd = hit[:n], hit[n:]
        ok = got[:n] & got[n:]
        span = np.linalg.norm(fwd - bwd, axis=1)
        centre = 0.5 * (fwd + bwd)
        ok &= span > _RAY_EPS
        return span, centre, ok


class VoxelSpan:
    """Material span along a direction, by marching a filled occupancy grid, then snapping to the mesh.

    The fallback for a surface welding could not close. The march alone is robust but coarse, and
    coarse in one direction: ``voxelized`` marks every voxel the surface passes through, so the
    filled region reaches up to a pitch BEYOND the true surface and the raw march over-reports a
    span by about a pitch per side (measured against ray ground truth on watertight catalog meshes:
    +8 mm mean at a 4 mm pitch, which is 10% of the jaw).

    So the two are combined, each doing what it is good at. The march BRACKETS the exit — it cannot
    leak through a hole, which is the whole reason a ray was unusable here — and then the mesh's own
    triangles are consulted for a real intersection near that bracket to pin the endpoint exactly.
    Where the surface is missing there (the hole the mesh is open at) no intersection is found and
    the bracketed value stands, flagged only by staying pitch-accurate."""

    # How far from the marched bracket a ray hit may sit and still be taken as the same crossing, in
    # pitches. Slightly over 1 covers the voxelization's outward bias without reaching past a thin
    # wall to the surface behind it.
    _SNAP_PITCHES = 1.25

    def __init__(self, mesh, pitch: float):
        self.mesh = mesh
        self.pitch = float(pitch)
        grid = mesh.voxelized(pitch=self.pitch)
        try:
            grid = grid.fill()
        except Exception:                      # noqa: BLE001 - an unfillable shell still voxelizes
            # fill() needs a closed shell to flood from; without it the surface voxels alone still
            # bound the object, which is enough to march to a wall. Reported in the pass notes.
            pass
        self.grid = grid
        # March step: half a voxel, so a wall one voxel thick cannot be stepped over.
        self.step = 0.5 * self.pitch
        self.max_steps = int(np.ceil(float(np.linalg.norm(mesh.extents)) / self.step)) + 2

    def _filled(self, points):
        return np.asarray(self.grid.is_filled(np.asarray(points, dtype=float).reshape(-1, 3)))

    def _walk(self, points, directions):
        """Distance from each point to the last filled sample along its direction."""
        p = np.asarray(points, dtype=float)
        n = len(p)
        dist = np.zeros(n)
        live = self._filled(p)          # a query point outside the solid has no span to report
        for k in range(1, self.max_steps + 1):
            if not live.any():
                break
            t = k * self.step
            still = np.zeros(n, dtype=bool)
            still[live] = self._filled(p[live] + t * directions[live])
            # Once a ray leaves the material it stays left: no re-entry, so a grasp across a gap is
            # not silently reported as one solid span.
            live = live & still
            dist[live] = t
        return dist

    def _snap(self, p, d, bracket):
        """Pull each marched exit distance onto a real triangle crossing near it, where one exists.

        Returns the refined distances; entries with no nearby intersection keep the bracket's own
        midpoint estimate (the surface lies between the last filled sample and the first empty one)."""
        refined = bracket + 0.5 * self.step
        loc, ray_idx, _tri = self.mesh.ray.intersects_location(p, d, multiple_hits=True)
        if not len(ray_idx):
            return refined
        # Distance of every hit along its own ray, then keep the hit nearest that ray's bracket.
        t = np.einsum("ij,ij->i", loc - p[ray_idx], d[ray_idx])
        gap = np.abs(t - bracket[ray_idx])
        tol = self._SNAP_PITCHES * self.pitch
        good = (t > 0.0) & (gap <= tol)
        if not good.any():
            return refined
        ray_idx, t, gap = ray_idx[good], t[good], gap[good]
        # Sort so the closest hit for each ray lands last, then scatter — a deterministic argmin per
        # ray without a Python loop.
        order = np.lexsort((-gap, ray_idx))
        refined[ray_idx[order]] = t[order]
        return refined

    def measure(self, points, directions):
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        fwd = self._snap(p, d, self._walk(p, d))
        bwd = self._snap(p, -d, self._walk(p, -d))
        span = fwd + bwd
        centre = p + 0.5 * (fwd - bwd)[:, None] * d
        ok = span > _RAY_EPS
        return span, centre, ok
