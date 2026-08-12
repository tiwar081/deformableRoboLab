# `geometric` — grasp candidates for rigid assets, from two methods

Produces grasp CANDIDATES for the catalog's rigid kinds (`ycb_mesh`, `rubiks_cube`, `rigid_box`) by
two independent geometric searches, each tagged with the method that found it.

| method | label | how it searches | what it is good at |
|---|---|---|---|
| medial-axis skeleton | `medial_axis` | from the INSIDE — skeletonize, and where twice the local radius fits the jaw, grasp perpendicular to the local axis | centre-line features: a bottle neck, a tool shaft, the length of a banana |
| cross-section sweep | `cross_section` | from the OUTSIDE — slice along each canonical axis and find boundary that faces itself across a jaw-sized gap | surface-pair features: flat parallel box sides, a wall to pinch, a rim |

Both carry the pass's single `source` tag `geometric` (the merge deduplicates by source, so one pass
gets one tag); the method is in `labels`, and the id prefix — `medial_` / `xsec_` — mirrors it so a
candidate is identifiable from its id alone.

`face` is derived from the approach by `make_candidate`, because the schema requires it to agree
with the transform. **Everything else another pass owns is left alone:** every `quality` field is
null and no candidate carries a `quality_source`. Nothing here is measured in simulation — a
candidate is a geometrically plausible place to put the jaws, not a grasp known to hold.

Every stored pose goes through `grasp_library.pad_seat` (rule 2 — `POSE_CONVENTION` **v2**), which
advances the grasp centre along its approach until the material sits between the pads instead of on
the fingertips. On this catalog that is a **20–30 mm** move. Since pass v4 the seat also stamps the
required `seat_mode` (`centred`/`clamped_deep`/`retreated`) and the `seat_blocked` label from
`pad_seat`'s collision-aware retreat — the modes and their measured rationale live in
`grasp_library.py` and `docs/trajPipeline/grasp-library.md`. The pass adds no offset of its own; the
seat is applied after dedup and the cap only because `pad_seat` rebuilds a BVH per call (~130 ms on
a 15k-triangle scan), so seating the raw stream would be thousands of calls per asset rather than at
most `2 × max_per_method`. A candidate `pad_seat` cannot seat is dropped, never stored.

`config.max_centring_error` is **not** related to this. It is a rejection test on the **jaw** axis
(is the skeleton node centred in the chord?) and it moves nothing; pad seating is placement along the
**approach**. Two different axes, two different jobs.

## Run it

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes run geometric              # all rigid assets
.venv/bin/python -m deformableManipulationTools.grasp_passes run geometric --asset mug
.venv/bin/python -m deformableManipulationTools.grasp_passes run geometric --check-idempotent
```

~100 s for the whole rigid catalog (18 assets), 60–130 candidates each.

## Look at one asset

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz banana
.venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz mug --method cross_section
.venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz bucket --verify --limit 60
```

Writes to `outputs/grasp_viz/<asset>/`:

* **`<asset>_grasps.glb`** — the object with a gripper marker at every candidate pose, blue for
  `medial_axis`, orange for `cross_section`. Open in any glTF viewer (VS Code has one) and inspect
  from any angle. The marker is ACRONYM's four-cylinder gripper, vendored into `marker.py` and drawn
  at each candidate's own jaw opening.
* **`<asset>_grasps.png`** — three orthographic panels of the canonical frame, each grasp drawn as
  its jaw chord plus a stub along the approach. The repo has no GL context, so the still is drawn
  with matplotlib rather than by rendering the scene.

`viz.pad_containment()` re-runs `pad_seat` on each STORED pose and reports the fraction of the local
material span lying between the pads. Two things it catches: a stored pose that was never seated
shows a non-zero re-seat advance (~28 mm — a v1 pose or a skipped rule-2 call), and an object deeper
than `PAD_LENGTH` (53.8 mm) shows a fraction of exactly `PAD_LENGTH / depth`, which is the pads being
too short rather than the seat being wrong.

`--verify` also re-measures the candidates against the mesh and prints how far the pad points sit
from the surface. That is the check a picture cannot do: a wrong-but-consistent frame still looks
tidy, but shows up here as pads floating off the surface. Watertight assets come back at 0.00 mm
median / ≤0.05 mm max; the voxel-fallback assets at 0.00 mm median / ≤4.4 mm max, which is the
fallback's pitch-limited accuracy.

`grasp centre inside the mesh: 34/40` is not a failure count — a grasp spanning a hollow object has
material at both pads and void between them, so its centre is legitimately in air.

## Layout

```
geometric/
  __init__.py    the pass: assembles both methods, dedups, caps       <- start here
  config.py      every tuning knob, with the reason attached
  meshprep.py    welding/subdivision + the two width-measurement backends
  medial.py      method 1 — skeletonization (two backends) and the local axis fit
  sweep.py       method 2 — the cross-section antipodal search
  select.py      dedup + farthest-point thinning
  marker.py      vendored ACRONYM gripper marker (MIT, NVIDIA)
  viz.py         the inspection CLI
```

## Things worth knowing

* **Catalog meshes arrive unwelded.** `load_usd_mesh` returns vertices per face-corner, so a YCB
  scan is topologically 15k loose triangles and skeletonization sees dust. `meshprep.prepare` welds
  first — that is what makes the YCB meshes watertight and the primary skeleton backend usable.
* **Skeletonization has two backends, chosen by watertightness.** `skeletor.by_wavefront` for closed
  surfaces; voxelize + `skimage.morphology.skeletonize` with radii from the distance transform for
  open ones (the vomp bins and bucket weld to 5 loose bodies). `skeletor` is installed from
  `_external/skeletor` NON-editable, so it lives in site-packages and the repo does not read
  `_external/` at runtime.
* **The width backend follows the same split.** Ray casting where the surface is closed; where it is
  not, an occupancy march BRACKETS the exit (a ray leaks through the holes) and the mesh's own
  triangles then pin the endpoint exactly. The march alone over-reports by about a pitch per side —
  `voxelized` marks every voxel the surface touches — which measured +8 mm at a 4 mm pitch against
  ray ground truth, 10% of the jaw. With the snap, most accepted queries match ray truth exactly.
* **The medial method checks its own premise.** A node's radius describes the solid only where the
  node is the CENTRE of the local cross-section. `config.max_centring_error` rejects chords whose
  midpoint is far from the skeleton node, which is what stops the YCB `wood_block` — 90 mm thick,
  nothing about it fits an 80 mm jaw — from yielding corner nips between two perpendicular faces
  where the wavefront radius collapses near its ends. It correctly returns almost nothing.
* **`antipodal_angle_deg` is a friction bound, not a tolerance.** 20° sits inside the cone of the
  least-frictional catalog object once its `mu` is coupled to the rubber pad's 0.8 by the framework's
  geometric-mean law. See `config.py`.
* **Not every object has a whole-body grasp** — the mug is wider than the jaw in every direction, so
  its candidates are wall pinches and rim grasps. Both methods return nothing rather than something
  when nothing fits, and the pass notes say why.
