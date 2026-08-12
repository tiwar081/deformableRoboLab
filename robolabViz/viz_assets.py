"""Visual-only re-reader of the vendored object assets, for the advanced render tier.

The PHYSICS pipeline loads these USDs without UVs (``mesh_collision.load_usd_mesh``),
and it must stay that way: loading with UVs vertex-splits faceVarying assets (the
objaverse apple goes 75k -> 397k verts), which would change the mesh fed to the
coacd convex decomposition — a physics change and a poisoned decomposition cache.
So the advanced raycast tier re-reads the SAME asset here, purely for rendering:
full-res mesh + per-vertex UVs + the bound base-color texture, keyed by the body
label (``add_ycb_mesh``/``add_rubiks_cube`` label bodies with the asset's catalog
name: banana, bowl, mug, apple_01, rubiks_cube).

Never imported by the physics package; failures degrade to the flat-color look.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VizVisual:
    """A textured render mesh in the asset's local frame (verts/uvs 1:1 aligned)."""

    verts: np.ndarray            # (n, 3) float32
    tris: np.ndarray             # (m, 3) int32
    uvs: np.ndarray              # (n, 2) float32
    texture: np.ndarray          # (tile, tile, 3) uint8 sRGB
    roughness: float = 0.5
    metallic: float = 0.0


def _shader_asset_input(prim, usd_path: Path) -> Path | None:
    """Bound-material walk for assets whose texture ``newton.usd.get_mesh`` does
    not auto-resolve (gltf-style MDL, e.g. the objaverse apple): find the first
    Asset-typed shader input under the bound material and resolve its path."""
    from pxr import Sdf, Usd, UsdShade

    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat or not mat.GetPrim():
        return None
    for sp in Usd.PrimRange(mat.GetPrim()):
        if not sp.IsA(UsdShade.Shader):
            continue
        for inp in UsdShade.Shader(sp).GetInputs():
            if inp.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            ap = inp.Get()
            if not ap:
                continue
            if ap.resolvedPath:
                return Path(ap.resolvedPath)
            return (usd_path.parent / ap.path).resolve()
    return None


def _load_visual(usd_path: Path, tile: int) -> VizVisual | None:
    import cv2

    import newton.usd as newton_usd
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    # Pick the LARGEST mesh prim under the asset root (the body), not the first in traversal order.
    # A multi-mesh asset (the VoMP bucket = decal + body + handle) would otherwise texture only the
    # flat decal sticker (its first prim) — the flat-panel look. The viz mesh carries ONE texture,
    # so the body (most faces, main texture) is the right single prim to show; the collision path
    # (mesh_collision.load_usd_mesh) merges all prims for physics. Scope to the default prim so a
    # stray sibling (the apple's GroundPlane) is not considered.
    from pxr import Usd as _Usd
    root = stage.GetDefaultPrim()
    walk = _Usd.PrimRange(root) if root and root.IsValid() else stage.Traverse()

    def _n_faces(pr):
        fvc = UsdGeom.Mesh(pr).GetFaceVertexCountsAttr().Get()
        return len(fvc) if fvc else 0

    meshes = [p for p in walk if p.IsA(UsdGeom.Mesh)
              and UsdGeom.Imageable(p).ComputePurpose() not in (UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)]
    prim = max(meshes, key=_n_faces, default=None)
    if prim is None:
        return None
    # load_uvs handles both vertex- and faceVarying-interpolated primvars:st
    # (the latter via vertex splitting, keeping verts/uvs 1:1).
    mesh = newton_usd.get_mesh(prim, load_uvs=True)
    uvs = getattr(mesh, "uvs", None)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    # Bake the prim's local-to-world xform, mirroring mesh_collision.load_usd_mesh: prim-local
    # points can be posed/scaled by xformOps (objaverse apple: 0.01 scale). Identity for the
    # legacy assets, so their visuals are unchanged.
    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    if not np.allclose(m, np.eye(4), atol=1e-9):
        verts = ((verts.astype(np.float64) @ m[:3, :3]) + m[3, :3]).astype(np.float32)
    if uvs is None or len(uvs) != len(verts):
        return None
    tex_path = getattr(mesh, "texture", None) or _shader_asset_input(prim, usd_path)
    if not tex_path or not Path(tex_path).exists():
        return None
    bgr = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    texture = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tile, tile), interpolation=cv2.INTER_AREA)
    return VizVisual(
        verts=verts,
        tris=np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3),
        uvs=np.asarray(uvs, dtype=np.float32),
        texture=texture,
        roughness=float(getattr(mesh, "roughness", 0.5) or 0.5),
        metallic=float(getattr(mesh, "metallic", 0.0) or 0.0),
    )


def catalog_visual(body_label: str, tile: int) -> VizVisual | None:
    """The textured render mesh for a body whose label names a vendored catalog
    asset, or None (unknown label / asset unreadable / no UVs or texture)."""
    from .config import _object_catalog

    name = str(body_label).split("/")[-1]
    asset = _object_catalog().get(name)
    if asset is None:
        return None
    try:
        return _load_visual(asset.usd_path, tile)
    except Exception as exc:  # visual-only path: never take the run down
        print(f"[robolabViz] textured visual for {name!r} unavailable ({exc!r}); keeping flat color.")
        return None
