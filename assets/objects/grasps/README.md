# Precomputed grasp candidates

One JSON sidecar per catalog object, named by its **`scene_catalog.json` name** — `<name>.json`.
Keyed by catalog name rather than asset filename because two catalog entries can share one USD and
procedural kinds (`rigid_box`, `rubiks_cube`) have no file at all.

Schema, loader, validation, and the canonical-frame utility all live in
[`deformableManipulationTools/grasp_library.py`](../../../deformableManipulationTools/grasp_library.py);
the record layout mirrors the ACRONYM dataset's HDF5 records (grasp transforms + per-grasp
physics-simulation quality fields). Read that module's docstring before adding or editing anything
here — in particular the two versioned conventions (`obb_extent_desc_v1` for the object frame,
`tcp_z_approach_x_jaw_v1` for the grasp pose).

**Scope: rigid objects and soft-bodied (FEM) objects only — NOT cloth, NOT bags, and (since
2026-08-11) NOT cables.** A garment has no persistent rest shape, so a box fitted to its source
mesh says nothing about the settled object; cables failed the same rest-shape premise on the
measured record (0/62 held at rest-shape spans). `validate_record` rejects `kind: "cloth"` and
`kind: "cable"` outright — see the SCOPE paragraph in `grasp_library.py`.

These files are **generated, not hand-written**. The frame block is derived from the mesh and
guarded by `object.mesh_sha1`; editing it by hand desynchronizes the candidates from the geometry
they were computed for. Check a record with:

```bash
.venv/bin/python -m deformableManipulationTools.grasp_library --list
.venv/bin/python -m deformableManipulationTools.grasp_library --selfcheck --asset ycb/banana.usd
```
