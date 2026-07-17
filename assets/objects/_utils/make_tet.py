"""Offline asset-prep tool: surface mesh -> IsaacGym-format ``.tet`` (DefGraspSim convention:
``v x y z`` vertices + ``t i0 i1 i2 i3`` 0-indexed tetrahedra), consumed at runtime by
``deformableManipulationTools.assets.load_tet_mesh`` / ``add_soft_mesh_object``.

This reproduces DefGraspSim's own pipeline (surface mesh -> **fTetWild** -> .tet via their
``mesh_to_tet.py``) using the ``wildmeshing`` python bindings. fTetWild is the right tool for raw
scan surfaces (YCB): it produces WELL-CONDITIONED tets from dirty/non-watertight input. tetgen
surface-constrained tets were tried first and REJECTED — measured: they carry sliver tets
(min |vol| 3e-16..1e-10 m^3, worse WITH quality switches) whose ill-conditioning makes the VBD FEM
solve gain energy at rest and NaN within seconds, at any contact stiffness.

Run OFFLINE with the project venv python + wildmeshing on PYTHONPATH (it is an asset-prep
dependency only, never a runtime dependency — install with
``.venv/bin/pip install --target <dir> wildmeshing`` and ``PYTHONPATH=<dir>``; delete the target
dir's bundled numpy so the venv's own numpy is used):

    PYTHONPATH=<wmdir> .venv/bin/python assets/objects/_utils/make_tet.py <in_mesh> <out.tet> \
        [--edge-rel 0.1] [--stop-quality 10] [--scale 1.0]

``--edge-rel`` is fTetWild's ideal edge length relative to the bbox diagonal (bigger = coarser =
fewer FEM particles; 0.1 gives a few hundred vertices on a YCB object, the SOFT_BLOCK scale).
The .tet vertex units are the input mesh units (YCB google_16k = metres).
"""
import argparse


def make_tet(in_mesh: str, out_tet: str, edge_rel: float = 0.1, stop_quality: float = 10.0,
             scale: float = 1.0, epsilon: float = 0.005) -> tuple[int, int]:
    import numpy as np
    import trimesh
    import wildmeshing

    tm = trimesh.load(in_mesh, force="mesh")
    # epsilon (surface-envelope tolerance, rel. bbox diag) is the main COARSENESS knob: the default
    # 1e-3 chases scan detail into thousands of verts; 5e-3 lands YCB objects at the SOFT_BLOCK
    # scale (a few hundred particles) while keeping fTetWild's well-conditioned elements.
    tetra = wildmeshing.Tetrahedralizer(stop_quality=stop_quality, edge_length_r=edge_rel,
                                        epsilon=epsilon)
    tetra.set_mesh(np.asarray(tm.vertices, dtype=np.float64) * scale,
                   np.asarray(tm.faces, dtype=np.int32))
    tetra.tetrahedralize()
    nodes, elems = tetra.get_tet_mesh()[:2]
    nodes, elems = np.asarray(nodes), np.asarray(elems, dtype=np.int64)
    # Belt-and-braces: drop any residual degenerate tet (a near-zero rest volume NaNs the FEM).
    a, b, c, d = (nodes[elems[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    keep = np.abs(vol) > max(1e-12, 1e-4 * float(np.median(np.abs(vol))))
    if (~keep).any():
        print(f"dropping {int((~keep).sum())} degenerate tets (|vol| < 1e-4 x median)")
        elems = elems[keep]
    used = np.unique(elems)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, elems = nodes[used], remap[elems]
    with open(out_tet, "w") as out:
        out.write(f"# Tetrahedral mesh generated using fTetWild (wildmeshing) from {in_mesh}\n\n")
        out.write(f"# {len(nodes)} vertices\n")
        for p in nodes:
            out.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        out.write(f"\n# {len(elems)} tetrahedra\n")
        for t in elems:
            out.write(f"t {t[0]} {t[1]} {t[2]} {t[3]}\n")
    return len(nodes), len(elems)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_mesh")
    ap.add_argument("out_tet")
    ap.add_argument("--edge-rel", type=float, default=0.1)
    ap.add_argument("--stop-quality", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.005)
    a = ap.parse_args()
    n, t = make_tet(a.in_mesh, a.out_tet, a.edge_rel, a.stop_quality, a.scale, a.epsilon)
    print(f"{a.out_tet}: {n} vertices, {t} tets")
