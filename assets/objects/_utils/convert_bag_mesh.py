"""Convert data-driven bag meshes into simulation-ready Newton cloth USDA assets.

The per-asset choices live in a JSON manifest rather than this script.  Each entry supplies the
source mesh, canonical orientation/size, display color, provenance, and the ``ClothConfig`` values
that are authored into the USD.  At runtime ``deformableManipulationTools.assets.add_cloth`` reads
the triangle mesh while the same manifest values are registered as ``ClothConfig`` constructor
arguments in ``scene_catalog.json``.

The conversion deliberately keeps the largest connected surface component.  Render-oriented bag
downloads often contain detached straps, labels, or metal hardware; without sewn constraints those
pieces would become independent cloth bodies and fall away.  The retained bag shell is welded,
made edge-manifold, isotropically remeshed, and required to be one connected surface with regular
boundary loops.

Asset-prep dependencies (not runtime dependencies)::

    .venv/bin/pip install pymeshlab==2025.7.post1
    .venv/bin/python assets/objects/_utils/convert_bag_mesh.py \
        assets/objects/objaverse_bags/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _edge_topology(mesh):
    import numpy as np

    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    nonmanifold = unique[counts > 2]
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    irregular = [v for v, neighbours in adjacency.items() if len(neighbours) != 2]
    loops = 0
    unseen = set(adjacency)
    while unseen:
        loops += 1
        stack = [unseen.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    return unique, counts, boundary, nonmanifold, irregular, loops


def _drop_extra_nonmanifold_faces(mesh):
    """Resolve edges used by >2 triangles, deterministically retaining the first face pair.

    Objaverse GLBs occasionally contain a tiny four-face seam even after exact vertex welding.  A
    cloth constraint graph cannot use such an edge.  Removing only the surplus pair preserves the
    surrounding surface and turns that seam into an ordinary boundary; the isotropic remesher then
    regularizes it.  The final topology checks remain authoritative.
    """
    import numpy as np

    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    drop: set[int] = set()
    face_edge_ids = inverse.reshape(-1, 3)
    for edge_id in np.flatnonzero(counts > 2):
        incident = sorted(np.unique(np.nonzero(face_edge_ids == edge_id)[0]).tolist())
        drop.update(incident[2:])
    if drop:
        mesh.update_faces(np.asarray([i not in drop for i in range(len(mesh.faces))]))
        mesh.remove_unreferenced_vertices()
    return len(drop)


def _orient_and_scale(mesh, up_axis: str, target_height: float, lay_flat: bool,
                      geometry_z_scale: float):
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    if up_axis == "x":
        vertices = vertices[:, [1, 2, 0]]
    elif up_axis == "y":
        # Sketchfab/GLB is normally Y-up.  Map Y to USD Z and flip the new Y to preserve winding.
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 1] *= -1.0
    elif up_axis != "z":
        raise ValueError(f"up_axis must be x, y, or z, got {up_axis!r}")

    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    if height <= 0.0:
        raise ValueError("source mesh has zero height")
    vertices *= float(target_height) / height
    # An unsupported soft bag cannot honestly stand open at rest. Rotate its standing height into
    # the tabletop plane, then compress only the authored front/back separation. This preserves the
    # large wall panels (and their areal mass) instead of shrinking the whole bag vertically.
    if lay_flat:
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] *= -1.0
    vertices[:, 2] *= float(geometry_z_scale)
    vertices[:, :2] -= 0.5 * (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0))
    vertices[:, 2] -= vertices[:, 2].min()
    mesh.vertices = vertices


def _isotropic_remesh(mesh, target_edge: float, iterations: int):
    import numpy as np
    import pymeshlab
    import trimesh

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
    )
    mesh_set.meshing_isotropic_explicit_remeshing(
        iterations=int(iterations),
        adaptive=False,
        targetlen=pymeshlab.PureValue(float(target_edge)),
        featuredeg=60.0,
        checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(0.5 * float(target_edge)),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    out = mesh_set.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=True)


def prepare_mesh(source: Path, *, up_axis: str, target_height: float, target_edge: float,
                 lay_flat: bool, geometry_z_scale: float, remesh_iterations: int,
                 expected_boundary_loops: int):
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(source), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"{source} did not load as a non-empty triangle mesh")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    components = list(mesh.split(only_watertight=False))
    if not components:
        raise ValueError(f"{source} has no connected surface components")
    mesh = max(components, key=lambda part: float(part.area)).copy()
    removed_faces = _drop_extra_nonmanifold_faces(mesh)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.fix_normals()

    _orient_and_scale(mesh, up_axis, target_height, lay_flat, geometry_z_scale)
    mesh = _isotropic_remesh(mesh, target_edge, remesh_iterations)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()

    # Re-anchor after remeshing; do not restore the standing height because the output intentionally
    # represents the collapsed tabletop pose described by geometry_z_scale.
    mesh.vertices[:, :2] -= 0.5 * (
        mesh.vertices[:, :2].min(axis=0) + mesh.vertices[:, :2].max(axis=0)
    )
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

    unique, _, boundary, nonmanifold, irregular, loops = _edge_topology(mesh)
    if int(mesh.body_count) != 1:
        raise ValueError(f"converted mesh has {mesh.body_count} connected components (expected 1)")
    if len(nonmanifold):
        raise ValueError(f"converted mesh has {len(nonmanifold)} non-manifold edges")
    if irregular:
        raise ValueError(f"converted mesh has {len(irregular)} irregular boundary vertices")
    if loops != int(expected_boundary_loops):
        raise ValueError(
            f"converted mesh has {loops} boundary loops (expected {expected_boundary_loops})"
        )

    lengths = np.linalg.norm(mesh.vertices[unique[:, 0]] - mesh.vertices[unique[:, 1]], axis=1)
    report = {
        "source_components": len(components),
        "discarded_components": len(components) - 1,
        "removed_nonmanifold_faces": removed_faces,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "boundary_loops": loops,
        "boundary_edges": int(len(boundary)),
        "edge_median_m": float(np.median(lengths)),
        "edge_p95_m": float(np.percentile(lengths, 95)),
        "edge_max_m": float(lengths.max()),
        "surface_area_m2": float(mesh.area),
        "extents_m": [float(x) for x in mesh.extents],
    }
    return mesh, report


def _author_float(prim, name: str, value: float):
    from pxr import Sdf

    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=True).Set(float(value))


def write_usda(output: Path, mesh, entry: dict, report: dict) -> None:
    from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Bag")
    Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
    stage.SetDefaultPrim(root.GetPrim())

    usd_mesh = UsdGeom.Mesh.Define(stage, "/Bag/Cloth")
    usd_mesh.CreatePointsAttr(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertices]
    )
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr([int(i) for i in mesh.faces.reshape(-1)])
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    usd_mesh.CreateDoubleSidedAttr(True)
    bounds = mesh.bounds
    usd_mesh.CreateExtentAttr(
        [Gf.Vec3f(*[float(v) for v in bounds[0]]), Gf.Vec3f(*[float(v) for v in bounds[1]])]
    )
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in entry["color"]])])

    cloth = entry["cloth"]
    total_mass = float(cloth["density"]) * float(report["surface_area_m2"])
    UsdPhysics.MassAPI.Apply(usd_mesh.GetPrim()).CreateMassAttr(total_mass)

    material = UsdShade.Material.Define(stage, "/Bag/Looks/Material")
    shader = UsdShade.Shader.Define(stage, "/Bag/Looks/Material/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(c) for c in entry["color"]])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(entry.get("roughness", 0.8)))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(float(cloth["soft_contact_mu"]))
    physics_material.CreateDynamicFrictionAttr(float(cloth["soft_contact_mu"]))
    physics_material.CreateRestitutionAttr(0.0)

    # Core USD has rigid mass/material schemas but no portable cloth constitutive schema.  Author
    # the exact Newton ClothConfig values as explicit custom attributes; the catalog carries the
    # same values into add_cloth(), which is the runtime authority for VBD.
    for key in (
        "density", "particle_radius", "tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd",
        "contact_margin", "soft_contact_ke", "soft_contact_kd", "soft_contact_kf",
        "soft_contact_mu", "self_contact_radius", "self_contact_margin",
        "self_contact_rest_exclusion_radius",
    ):
        if key in cloth:
            _author_float(usd_mesh.GetPrim(), f"newton:cloth:{key}", cloth[key])
    usd_mesh.GetPrim().CreateAttribute(
        "newton:cloth:self_contact", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(bool(cloth.get("self_contact", True)))
    usd_mesh.GetPrim().CreateAttribute(
        "newton:cloth:self_contact_filter_threshold", Sdf.ValueTypeNames.Int, custom=True
    ).Set(int(cloth.get("self_contact_filter_threshold", 1)))

    for key, value in {
        "source:dataset": "Objaverse 1.0",
        "source:uid": entry["uid"],
        "source:license": entry["license"],
        "source:url": entry["source_url"],
        "source:author": entry["author"],
    }.items():
        usd_mesh.GetPrim().CreateAttribute(key, Sdf.ValueTypeNames.String, custom=True).Set(value)

    stage.GetRootLayer().customLayerData = {
        "generator": "assets/objects/_utils/convert_bag_mesh.py",
        "assetName": entry["name"],
        "objaverseUid": entry["uid"],
        "license": entry["license"],
    }
    stage.GetRootLayer().Save()


def convert_entry(entry: dict, source_dir: Path, output_dir: Path) -> dict:
    source = source_dir / entry["source_file"]
    output = output_dir / entry["output_file"]
    mesh, report = prepare_mesh(
        source,
        up_axis=entry.get("up_axis", "y"),
        target_height=float(entry["target_height_m"]),
        target_edge=float(entry.get("target_edge_m", 0.012)),
        lay_flat=bool(entry.get("lay_flat", False)),
        geometry_z_scale=float(entry.get("geometry_z_scale", 1.0)),
        remesh_iterations=int(entry.get("remesh_iterations", 8)),
        expected_boundary_loops=int(entry.get("expected_boundary_loops", 1)),
    )
    write_usda(output, mesh, entry, report)
    report.update({"name": entry["name"], "uid": entry["uid"], "output": str(output)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-dir", type=Path, help="default: <manifest-dir>/source")
    parser.add_argument("--output-dir", type=Path, help="default: <manifest-dir>")
    parser.add_argument("--asset", action="append", help="convert only this manifest name (repeatable)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    source_dir = args.source_dir or args.manifest.parent / "source"
    output_dir = args.output_dir or args.manifest.parent
    selected = set(args.asset or [])
    entries = [e for e in manifest["assets"] if not selected or e["name"] in selected]
    missing = selected - {e["name"] for e in entries}
    if missing:
        raise SystemExit(f"unknown --asset names: {', '.join(sorted(missing))}")

    reports = [convert_entry(entry, source_dir, output_dir) for entry in entries]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
