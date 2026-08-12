"""SHARED — catalog + mesh resolution for grasp passes. Read-only for pass authors.

Turns a catalog NAME into everything a pass needs: the entry, the rest-shape geometry, and the
canonical frame. One place resolves geometry per catalog ``kind`` so every pass sees the same
vertices and therefore the same frame — if two passes disagreed about the geometry, their candidate
poses would not be comparable and the merge would silently mix frames.

Reads ``assets/objects/scene_catalog.json`` directly rather than importing
``agentic_pipeline.scene_generator``: the dependency runs the other way (the pipeline imports this
package), and this only needs name/kind/config/dims.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..grasp_library import ObjectFrame, UNSUPPORTED_KINDS, canonical_frame, mesh_digest
from ..settings import REPO_ROOT

OBJECTS_DIR = REPO_ROOT / "assets" / "objects"
CATALOG_PATH = OBJECTS_DIR / "scene_catalog.json"

# Catalog kinds a pass can be run on: rigid bodies + soft (FEM) objects only — see the SCOPE
# paragraph in grasp_library.py. ``cloth`` is absent on purpose (a garment/bag has no persistent
# rest shape, so a frame fitted to its source mesh describes the asset file rather than the settled
# object); ``cable`` was removed 2026-08-11 on the measured record (0/62 held at rest-shape spans;
# the medial probe recovered the synthetic rod's own construction axis, not structure).
SUPPORTED_KINDS = ("ycb_mesh", "rubiks_cube", "rigid_box", "soft_mesh", "soft_block")


class AssetError(ValueError):
    """The catalog entry cannot be resolved into rest geometry."""


@dataclass(frozen=True)
class CatalogAsset:
    """One resolved catalog object — the input every pass receives.

    ``vertices`` are the asset's REST geometry in its own (body) frame, exactly as the runtime
    spawns it; ``faces`` is None for point-set kinds (tet meshes carry no surface triangles, and the
    OBB over the volume samples is the right fit anyway). ``frame`` is the canonical frame those
    vertices produce — never recompute it in a pass, take this one."""
    name: str
    kind: str
    entry: dict
    vertices: np.ndarray
    faces: np.ndarray | None
    frame: ObjectFrame
    mesh_sha1: str
    file: str | None
    scale: float = 1.0

    @property
    def extents(self) -> np.ndarray:
        return self.frame.extents

    @property
    def half_extents(self) -> np.ndarray:
        return self.frame.half_extents()

    def canonical_vertices(self) -> np.ndarray:
        """The rest geometry in canonical-frame coordinates — what a pass measures against."""
        return self.frame.to_canonical(self.vertices)


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def catalog_entry(name: str) -> dict:
    for o in load_catalog()["objects"]:
        if o["name"] == name:
            return o
    known = ", ".join(sorted(o["name"] for o in load_catalog()["objects"]))
    raise AssetError(f"no catalog object named {name!r}. Known: {known}")


def asset_names(kinds=None) -> list:
    """Catalog names a pass can run on, sorted. ``kinds`` filters further (default: all supported)."""
    allow = tuple(kinds) if kinds else SUPPORTED_KINDS
    return sorted(o["name"] for o in load_catalog()["objects"]
                  if o["kind"] in allow and o["kind"] not in UNSUPPORTED_KINDS)


def _box_vertices(hx: float, hy: float, hz: float):
    v = np.array([[sx * hx, sy * hy, sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
    return v, f


def _resolve_geometry(entry: dict):
    """(vertices, faces, file, scale) of a catalog entry's rest geometry, by kind."""
    kind, cfg = entry["kind"], dict(entry.get("config", {}))
    if kind in UNSUPPORTED_KINDS:
        raise AssetError(
            f"{entry['name']!r} is kind {kind!r}, which precomputed grasp candidates do not model "
            f"(no persistent rest shape — see the SCOPE paragraph in grasp_library.py)")

    if kind == "ycb_mesh":
        from ..mesh_collision import load_usd_mesh
        sub = cfg["usd_subpath"]
        mesh = load_usd_mesh(OBJECTS_DIR / sub)
        return (np.asarray(mesh.vertices, dtype=float),
                np.asarray(mesh.indices).reshape(-1, 3), sub, 1.0)

    if kind == "soft_mesh":
        from ..assets import load_tet_mesh
        from ..params import SoftMeshConfig
        c = SoftMeshConfig(**cfg)
        verts, _tets = load_tet_mesh(OBJECTS_DIR / c.tet_subpath)
        return np.asarray(verts, dtype=float) * c.scale, None, c.tet_subpath, c.scale

    if kind == "rubiks_cube":
        from ..params import RUBIKS_CUBE
        from dataclasses import replace
        c = replace(RUBIKS_CUBE, **cfg) if cfg else RUBIKS_CUBE
        v, f = _box_vertices(c.half_extent, c.half_extent, c.half_extent)
        return v, f, None, 1.0

    if kind == "rigid_box":
        from ..params import RIGID_CUBE
        from dataclasses import replace
        c = replace(RIGID_CUBE, **cfg) if cfg else RIGID_CUBE
        v, f = _box_vertices(c.half_extent, c.half_extent, c.half_extent)
        return v, f, None, 1.0

    if kind == "soft_block":
        from ..params import SOFT_BLOCK
        from dataclasses import replace
        c = replace(SOFT_BLOCK, **cfg) if cfg else SOFT_BLOCK
        h = [0.5 * c.dim[i] * c.cell for i in range(3)]
        v, f = _box_vertices(*h)
        return v, f, None, 1.0

    raise AssetError(f"{entry['name']!r}: no rest-geometry resolver for kind {kind!r}")


_CACHE: dict = {}


def load_asset(name: str, *, use_cache: bool = True) -> CatalogAsset:
    """Resolve a catalog name into rest geometry + canonical frame.

    Cached per process: the canonical frame runs the OBB search three times (once plus two drift
    probes), so a run over many passes should not repeat it per pass."""
    if use_cache and name in _CACHE:
        return _CACHE[name]
    entry = catalog_entry(name)
    verts, faces, file, scale = _resolve_geometry(entry)
    asset = CatalogAsset(
        name=name, kind=entry["kind"], entry=entry, vertices=verts, faces=faces,
        frame=canonical_frame(verts, faces),
        mesh_sha1=mesh_digest(verts, faces if faces is not None else np.zeros((0, 3), dtype=int)),
        file=file, scale=scale)
    if use_cache:
        _CACHE[name] = asset
    return asset


def clear_cache() -> None:
    _CACHE.clear()
