"""Stable Newton collision geometry from concave meshes, via coacd convex decomposition.

Centralizes the docs/physicsEngine/SOLVERS.md §4 fix: a raw *concave* mesh (e.g. the YCB bowl) on a rigid
body gives contradictory contact normals → a single-substep VBD penalty spike → the whole
solve ejects to infinity. coacd splits the mesh into convex hull pieces (consistent normals,
cavity preserved); the full mesh is kept for rendering only. This is the shared helper any
mesh-loading demo reuses — the primitive/FEM demos never call it, so they can't regress.

Public ``newton`` / ``trimesh`` / ``pxr`` API only (CLAUDE.md: never depend on ``_external`` at
runtime). coacd is NOT imported here — it segfaults when co-loaded with Newton, so the
decomposition runs in a subprocess (``coacd_worker.py``) and this process only reads the cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import newton
from newton import Mesh

_CACHE_DIR = Path(os.environ.get("COACD_CACHE_PATH", "/tmp/coacd-cache"))
_WORKER = Path(__file__).resolve().parent / "coacd_worker.py"


def load_usd_mesh(path: Path) -> Mesh:
    """Load a USD file's geometry as one Newton ``Mesh``, MERGING every ``UsdGeom.Mesh`` prim with
    each prim's local-to-world xform baked into its vertices. Assets store points in prim-local
    units and pose them via xformOps (the objaverse apple: a 0.01 scale + rotate — its raw points
    are 100x too big); ``Mesh.create_from_usd`` reads raw points only, so the bake is required.

    A single-mesh asset (every legacy YCB/HOT3D object) is byte-identical to before — one prim, its
    xform baked (identity for those, so verts + coacd cache keys are unchanged). Multi-prim assets
    (the VoMP utility bucket = a decal + body + handle + handle-joint; the objaverse apple = 2
    prims) previously loaded only the FIRST prim in traversal order — for the bucket that was the
    flat 141-vert decal sticker, so both collision and viz used a flat panel (it rendered as loose
    sheets and sank through the table). Merging recovers the true geometry (bucket: 7.5k verts,
    0.34x0.33x0.26 m matching the catalog)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    # Merge only the OBJECT's own mesh prims: those under the stage default prim (the asset root).
    # Some USDs ship stray sibling prims that must NOT be merged — the objaverse apple carries a
    # /GroundPlane/CollisionMesh (a 0.9x0.9 m plane) alongside /apple_01/mesh; merging it would blow
    # the apple's bbox to ~0.9 m. Scoping to the default prim's subtree drops those. (When there is
    # no default prim, fall back to the whole stage.)
    root = stage.GetDefaultPrim()
    prims = Usd.PrimRange(root) if root and root.IsValid() else stage.Traverse()
    verts_all: list[np.ndarray] = []
    idx_all: list[np.ndarray] = []
    offset = 0
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # Skip proxy/guide-purpose prims (render/collision stand-ins); keep default + render.
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        if purpose in (UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide):
            continue
        sub = Mesh.create_from_usd(prim)
        v = np.asarray(sub.vertices, dtype=np.float64)
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if not np.allclose(m, np.eye(4), atol=1e-9):
            v = (v @ m[:3, :3]) + m[3, :3]
        verts_all.append(v)
        idx_all.append(np.asarray(sub.indices) + offset)
        offset += len(v)
    if not verts_all:
        raise ValueError(f"no UsdGeom.Mesh prims in {path}")
    if len(verts_all) == 1:                       # fast path: one prim -> unchanged (cache-key stable)
        return Mesh(verts_all[0], idx_all[0])
    return Mesh(np.concatenate(verts_all), np.concatenate(idx_all))


def decimate_mesh(mesh: Mesh, target_faces: int) -> Mesh:
    """Quadric-decimate a mesh to ~``target_faces`` (for a low-poly RENDER copy; never for
    collision hull pieces — decimation breaks convexity, see SOLVERS.md §4)."""
    import trimesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.indices).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    try:
        d = tm.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        d = tm.simplify_quadric_decimation(target_faces)
    return Mesh(d.vertices.astype(np.float32), d.faces.flatten().astype(np.int32))


def _cache_key(v: np.ndarray, f: np.ndarray, cfg) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(f, dtype=np.int32).tobytes())
    h.update(repr((cfg.coacd_threshold, cfg.coacd_max_convex_hull,
                   cfg.coacd_preprocess_resolution, cfg.coacd_max_ch_vertex,
                   cfg.coacd_seed)).encode())
    return h.hexdigest()


def convex_decompose(mesh: Mesh, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """coacd convex decomposition → list of (verts float32, faces int32). Disk-cached (the
    decomposition is ~30 s/mesh). Runs coacd in a subprocess so its native lib never co-loads
    with Newton in this process."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{_cache_key(v, f, cfg)}.npz"
    if not cache.exists():
        fd, req = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            np.savez(req, verts=v, faces=f,
                     threshold=cfg.coacd_threshold, max_convex_hull=cfg.coacd_max_convex_hull,
                     preprocess_resolution=cfg.coacd_preprocess_resolution,
                     max_ch_vertex=cfg.coacd_max_ch_vertex, seed=cfg.coacd_seed)
            subprocess.run([sys.executable, str(_WORKER), req, str(cache)], check=True)
        finally:
            os.unlink(req)
    d = np.load(cache)
    return [(d[f"v{i}"], d[f"f{i}"]) for i in range(int(d["count"]))]


def add_collision_pieces(builder: newton.ModelBuilder, body: int, mesh: Mesh, cfg,
                         *, scale=(1.0, 1.0, 1.0)) -> list[int]:
    """Attach coacd convex-hull pieces as invisible ``CONVEX_MESH`` collision shapes on ``body``.
    Each piece is re-hulled via ``Mesh(maxhullvert=…)`` (convex-preserving) so its BVH stays small.
    Returns the shape indices."""
    coll_cfg = newton.ModelBuilder.ShapeConfig(
        density=cfg.density, ke=cfg.ke, kd=cfg.kd, mu=cfg.mu, is_visible=False)
    shapes = []
    for i, (pv, pf) in enumerate(convex_decompose(mesh, cfg)):
        piece = Mesh(pv.astype(np.float32), pf.flatten().astype(np.int32),
                     maxhullvert=cfg.piece_maxhullvert)
        shapes.append(builder.add_shape_convex_hull(
            body=body, mesh=piece, scale=scale, cfg=coll_cfg, label=f"hull_{i}"))
    return shapes


def add_visual_mesh(builder: newton.ModelBuilder, body: int, mesh: Mesh, color,
                    *, target_faces: int | None = None, scale=(1.0, 1.0, 1.0)) -> int:
    """Add a render-only ``GeoType.MESH`` (collision OFF) so the smooth mesh shows while the
    convex pieces do the colliding. ``target_faces`` decimates the render copy if set."""
    import warp as wp

    render = mesh if target_faces is None else decimate_mesh(mesh, target_faces)
    vis_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, has_shape_collision=False, has_particle_collision=False)
    return builder.add_shape(
        body=body, type=int(newton.GeoType.MESH), xform=wp.transform_identity(),
        cfg=vis_cfg, scale=scale, src=render, color=wp.vec3(*color), label="visual")


def rescale_body_mass(model, body: int, target_mass: float) -> None:
    """Scale a finalized body to an exact ``target_mass`` [kg] (and its inertia by the same
    ratio), so a mass knob is independent of the coacd piece volume (SOLVERS.md §5: realistic
    masses keep light bodies from being flung). Run after ``finalize``."""
    bm = model.body_mass.numpy()
    cur = float(bm[body])
    if cur <= 0.0 or target_mass <= 0.0:
        return
    s = target_mass / cur
    bi = model.body_inertia.numpy()
    inv_m = model.body_inv_mass.numpy()
    inv_i = model.body_inv_inertia.numpy()
    bm[body] = target_mass
    bi[body] = bi[body] * s
    inv_m[body] = 1.0 / target_mass
    inv_i[body] = inv_i[body] / s
    model.body_mass.assign(bm)
    model.body_inertia.assign(bi)
    model.body_inv_mass.assign(inv_m)
    model.body_inv_inertia.assign(inv_i)
