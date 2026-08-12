# Grasp passes — the framework and the seven passes

Candidates are generated and annotated by **passes** under
`deformableManipulationTools/grasp_passes/`, built in parallel by separate agents. The authoring
contract is `grasp_passes/README.md` (read it before writing a pass); per-pass design detail
lives in each pass's own README/docstrings. This doc is the system view: how the framework
composes, and the measured findings + dead ends per pass that the in-code docs don't carry.
The data layer they all share: [grasp-library.md](grasp-library.md).

## Framework facts beyond the authoring contract

- Producers fill `candidates`; consumers declare `requires` and fill `annotations`. Resolved run
  order today: fixture → geometric → obb_face → rim_pinch → vlm_regions → obb_bucket →
  shake_validate (→ `llm_retry`, trigger-gated — see below).
- **The merge is also a DERIVATION layer (2026-08-11)** — three record-level facts are computed
  centrally in `merge.py` so no generator or consumer re-implements them (spec:
  [grasp-library.md](grasp-library.md) "Candidate statuses"): `seat_blocked` candidates are
  DISCARDED from the record (the sidecar keeps the measurement; their orphaned annotations are
  dropped with a count, not an error); `retreated` candidates get the `weak_grasp_option` label
  stamped; `clamped_deep` candidates get their `overhang` computed and stored (tracked, never
  scored). The record-level `unusable` verdict (LLM retry exhausted, still nothing holds) is
  derived here too. All additive at schema v5 — a `merge --all` regenerates every record with no
  pass re-runs.
- **`merge.COMPATIBLE_VERSIONS`** — an explicit per-pass allowlist of sidecar versions the
  version gate accepts besides the current one. Exists for exactly one shape of change: a pass
  bump that only NARROWS what the pass chooses to run on, leaving poses and procedure unchanged,
  so older sidecars remain true measurements (first entry: `shake_validate: {3, 4}`). Anything
  that changes what a stored number MEANS still forces a re-run; extending the allowlist requires
  that justification in a comment next to the entry.
- **Two dynamically-discovering consumers recurse into each other's `requires` and blow the
  stack.** Fixed once, centrally: `base.DynamicUpstreamPass` + `discover_producers()` — a class
  flag lets discovery skip other discovering consumers *without reading their `requires`*, which
  is what composes for any number of them. The earlier asymmetric workaround held only while
  exactly one side of one pair evaluated the other. Discovering consumers do not require each
  other (annotators consume producers, not each other).
- **A pass reporting on the library's state is reporting on whatever it last read.** Measured
  twice: `obb_face`'s "20/24 fail to merge on v1 sidecars" was stale within 38 minutes (the same
  session's merge completed 24/24); two agents' library reports disagreed until a raw-JSON sweep
  of disk settled it. Sweep the files — and sweep them by raw read, not through the loader, which
  rejects v1 and hides the picture.
- Concurrent-agent hygiene, learned the hard way: pool counts move under a running measurement
  (re-derive from disk at use time, not at plan time); two agents running the same pass into the
  same sidecars is safe only because output is deterministic; `pgrep -f <pattern>` self-matches
  its own command line and waits forever; `pkill -f <script>` also matches the harness's waiter
  shells.

## `fixture` — hand-placed development candidates

12 candidates over 5 assets (banana, mug, tomato_soup_can, sugar_box, sponge) covering every
geometry path (the power_cable rod fixtures were removed 2026-08-11 with cables leaving the
pipeline's scope — fixture v8). PLACEHOLDERS, not validated grasps and not a model of generator
output — they exist so a consumer pass can be developed with `requires=("fixture",)` and switch
to a real producer with no other change. Widths verified against known geometry (mug wall 8.1 mm,
sponge 17.2 mm; historically also the cable's Ø15.5 mm).

## `geometric` — medial-axis + cross-section generator (rigid kinds, 1330 candidates)

Design + knobs: `grasp_passes/geometric/README.md` and `config.py` (every knob carries its
reason). Two methods, each candidate labelled `medial_axis` or `cross_section`:

- **Medial**: skeleton via `skeletor.by_wavefront` for watertight meshes, else voxelize (4 mm) +
  `skimage.skeletonize` with distance-transform radii; grasps ⊥ the local axis wherever the local
  radius fits the jaw; 4 roll samples per node; PCA linearity gate 0.55 (a bowl's medial set is a
  sheet — bowl yields 0 by design). `config.max_centring_error` **stays**: it gates the JAW axis
  (does the medial node sit near the middle of the chord the jaws close on) and is orthogonal to
  `pad_seat`'s approach-axis move; without it `wood_block` yields 6 corner nips instead of 2.
- **Sweep**: `section_multiplane` along all 3 canonical axes (12 slices/axis), boundary resampled
  at 4 mm, near-parallel opposing faces within the jaw. The 20° antipodal angle is a **friction
  bound, not a tolerance**: catalog µ 0.40–0.50 coupled to the pad's 0.8 by the geometric-mean
  law gives cone half-angles 29.5–32.3°, so 20° stays inside the least-frictional cone. Raw pair
  counts are huge (bucket 11346); dedup at 6 mm/~14° then deterministic farthest-point thinning
  to ≤64/method.

Measured facts / dead ends (do not re-walk):

- **USD meshes arrive unwelded** (per-face-corner vertices; mug 16763→7860 verts on weld) —
  welding is what makes YCB scans watertight and the primary skeleton backend usable. `bucket`
  welds to 5 loose bodies and stays open: the catalog's only voxel-fallback consumer.
- **The voxel march over-reports spans by ~1 pitch per side** (`voxelized` marks every touched
  voxel): banana bias +8.4 mm at 4 mm pitch vs ray ground truth. Fix is hybrid — voxels bracket
  the exit (rays leak through holes), mesh triangles pin the endpoint → bias +1.2, median 0.00.
  The residual heavy tail was a test artifact (outlier queries sat ~1 mm from a wall, where the
  pass's own centring gate rejects anyway).
- **Grazing chords**: on `steel_cube` the sweep accepted a 50 mm chord lying ON a 50 mm face
  (jaws closing on nothing) — boundary samples land on corners and the interior probe sits
  ~1e-11 inside, so `contains_xy` passes. Containment alone cannot catch this; the probe needs a
  real margin from the boundary.
- Verification tooling is part of the pass: `...grasp_passes.geometric.viz <asset> [--verify]`
  writes a GLB with the vendored ACRONYM gripper marker per pose (blue=medial, orange=sweep) +
  orthographic PNGs (no GL context in this repo — pyrender is unusable), and
  `viz.pad_containment()` is the numeric check. Pad-to-surface distance: watertight assets median
  0.00 mm / max 0.29 mm; bucket (voxel fallback) 5.0 mm, pitch-limited. "Grasp centre inside the
  mesh 34/40" is not a failure count — a grasp spanning a hollow legitimately centres in air.
- Timing: full 18-asset run 1m40s pre-seating; seating adds ~16.5 s/asset (seat after dedup/cap,
  never on the raw stream). 18/18 `--check-idempotent`.
- **Where pad containment is <100%, the pads are too short, not the seat wrong**: with re-seat
  residual 0 the covered fraction is identically `min(1, PAD_LENGTH/span)` — every sub-90% case
  has material deeper than 59.8 mm (cheez_it median 186 mm, long_tray_bin 139, sugar_box 127).
  This metric predates the shake finding that deep-seated poses fail differently — it measures
  seating, not holdability.

## `obb_face` — OBB-face generator (all 24 supported kinds, 422 candidates)

Aligns the jaw to each canonical-OBB face: 6 faces × 2 jaw orientations × 3 positions along the
face, width measured FROM THE MESH in a pad-sized column grown to the sampling density (catalog
spans 17k-vertex scans to a 235-vertex tet mesh), over-jaw candidates dropped, never clipped.
`wood_block` correctly yields ZERO (89×89 mm cross-section — nothing fits the 80 mm jaw) and
`pitcher` 0 — left visible rather than papered over. Its original `FINGER_DEPTH`=40 mm insertion
rule is retired (see the v1 defect in [grasp-library.md](grasp-library.md)).

**Compressible kinds get a 3% width tolerance** (`COMPRESSION_TOLERANCE=0.03`,
`COMPRESSIBLE_KINDS=("soft_mesh","soft_block")`): measured probe showed margin matters more than
the flag — foam_brick's 6 rejections are 1–2 mm over (1–2% compression, clearly valid),
sponge's +8 need 31% (unproven), banana_soft's +12 are end-on long-axis spans needing 47–60%
("not a grasp"). Only foam_brick's 6 were admitted, stored at `width = MAX_JAW_WIDTH` (keeping
the record commandable) with a `compressed` label + measured span/overshoot in the notes — gated
on a tolerance, not an asset name, so it re-derives when a mesh changes. Exactly 6 `compressed`
candidates catalog-wide.

## `obb_bucket` — face buckets + runtime pruning (annotator, still v1)

Assigns every candidate a bucket `±x/±y/±z` from its approach column alone — source-agnostic,
always delegated to `ObjectFrame.face_of` (never reimplemented; the pass cross-checks derived vs
stored `face` for every candidate and raises on disagreement). `pruning.surviving_buckets(frame,
placement, approach, half_angle)` composes the 6 face normals to world and rejects whole buckets
by dot product before anything expensive runs; survivors best-first. Design decisions:

- **Borderline** (`BORDERLINE_RTOL=0.05`, ~1.4° from an exact 45° edge): keep the argmax as
  primary (must equal the stored `face`), add `face_borderline` + `face_alt:<bucket>` labels. The
  load-bearing rule: pruning is a coarse pre-filter, so **a candidate is dropped only when EVERY
  plausible bucket is ruled out** — the gate can never remove a grasp the expensive stage would
  have accepted.
- **Frame-ambiguity is not bucket-ambiguity**: an `ambiguous` frame buckets normally + a
  `face_ambiguous` label — never merged or suppressed, because generator, record, and pruner all
  compose the SAME stored frame, so the bucket is geometrically exact; what ambiguity costs is
  meaning across re-exports. 1112 of 1767 annotated candidates (63%) carry it, 39 (2.2%) are
  borderline.
- The cone test carries `_COS_EPS=1e-9` slack: at exactly 90° the limit is cos(π/2)≈6e-17, and a
  perfectly horizontal face (a box on a table) would otherwise land on the wrong side.
- Candidates the pass never saw fall back to their stored `face` in `prune_candidates` — which is
  what currently covers `rim_pinch`'s 233 (merged after the last bucket run; re-run pending).
- **Bucket invariance under seating is confirmed** (0 face changes, 0 label-set changes over 399
  shared ids across the v1→v2 re-seat): a seat is a translation along the approach and
  `bucket_of` reads only the approach column — which is also why the pass has never needed a
  version bump.
- The scene-validation SAT (`agentic_pipeline/packing.py:obb_penetration`) is a 2D
  placement-overlap test and is NOT applicable here; the OBB reuse is `grasp_library.ObjectFrame`
  itself. Stated so nobody re-litigates the "reuse SAT" instruction.

## `vlm_regions` — semantic grasp regions (rigid kinds, separate `_regions/` store)

Renders six canonical views, asks a VLM for semantically meaningful grasp regions (closed
10-label vocabulary: handle, rim, neck, stem, spout, knob, lip, tab, shaft, other), single-linkage
clusters the picks (3.5 cm), back-projects onto the mesh, and stores named regions in the
canonical frame at `assets/objects/grasps/_regions/<name>.json` — a SEPARATE artifact, joined by
proximity (`regions_near`) by whoever scores; the annotator never sees a candidate and stays
valid when generators re-run. "No meaningful region" is a real recorded answer (`empty_reason`) —
7 of 18 stores are legitimately empty, and the apple's stem was SEEN and rejected ("too thin to
hold the fruit by"): the empty answer is a judgement, not blindness.

Design facts worth knowing before touching it:

- **The camera is FIXED on the key light's direction and the OBJECT rotates** to present each
  face (`views.LIT_DIRECTION`, mirrored from `robolabViz.raycast._render_kernel`'s fixed world
  lights). An orbiting camera looks into unlit cavities half the time — the mug's open end
  rendered as a flat dark disc and the VLM labelled the SOLID BASE as the rim at confidence 0.8,
  with the canonical z sign exactly backwards. If that light ever moves in `robolabViz`, `views.py`
  must follow; the symptom is dark cavities. Flat shading tier on purpose (HDRI/PBR specular +
  shadows read as false features); ~60 ms/view steady-state.
- Views are 640×640 at ~44° FOV (deliberately narrower than scene cameras, so silhouette-adjacent
  picks land on the surface they meant), framed at 1.18× the bounding sphere, pivoted on the
  bounding-sphere centre of the VERTICES (the OBB centre is not where the material is on hollow
  assets). A labelled 10×10 grid overlay is a reading aid; back-projection uses the fraction.
- Clustering exists because **averaging an annular feature's picks puts it in mid-air** (mug rim
  picks averaged to 8 mm from the axis — the middle of the hole); each cluster is its own region
  (`rim_0`, `rim_1`, …). Radius = pick spread clamped to 1–6 cm; confidence < 0.35 → `dropped`
  with reason; region `normal` is the mean pick normal, suppressed when ‖mean‖ ≤ 0.2 (opposite-
  side picks cancel — noise must not be reported as a direction).
- Back-projection is verified against the renders themselves (40×40 = 1600 rays vs silhouette,
  99.6–99.8% agreement; residual is boundary pixels). Picks are still only good to ~2 cm — a pick
  near a silhouette edge can land on the outer wall below a rim (seen on long_tray_bin).
  **Consumers join with a margin, never treat a centre as a grasp point** — and the stored
  `normal` is unreliable for geometry (the mug's four rim normals disagree by 90°; `rim_pinch`
  measures its own wall frames instead).
- Cached once per asset on (store format, `mesh_sha1`, prompt version, pass version) — the cache
  is also what makes a nondeterministic annotator pass `--check-idempotent` (18/18). The MODEL is
  deliberately excluded from the key (a new model on an unchanged mesh isn't worth 6 renders +
  a call; `--refresh` exists). `ask_regions` RAISES on transport failure — "no regions" is cached
  as a finding, "annotator unreachable" must never be.
- The structured-output endpoint rejects JSON-schema `minimum`/`maximum`/`maxItems` (each
  discovered as its own 400) — state bounds in the prompt, enforce in code.
- Environment dead ends: no third-party offscreen renderer exists here (pyrender/open3d/vtk/
  moderngl/usdrt all absent) — `robolabViz.raycast` is the renderer; `CATEGORIES` uses the token
  `"objects"`, not `"object"`. The pass emits empty `candidates` by design (regions are not
  candidates; `PassOutput` has no regions field — extending `base.py` was raised and declined,
  the store stays beside the sidecars).

Annotation roster (2026-08-04): 44 regions over 11 assets — handles on mug/bucket/pitcher, rims
on the seven containers, spouts on bucket/mustard/pitcher, stem on banana, low-confidence lid
`tab`s on tomato_soup_can/spam_can (0.60–0.65, kept above the floor, flagged for review); empty
with reason: apple, cheez_it, rubiks_cube, steel_cube, sugar_box, tuna_can, wood_block.

## `rim_pinch` — top-down lip pinches on open containers (233 candidates)

The regime the box-derived generators are structurally blind to: seven catalog assets are open
containers whose bodies (88–152 mm) exceed the 80 mm jaw, so the only fitting grasp is a shallow
pinch on the rim WALL itself (3–6 mm of material). Rims are **located, not inferred** — seeded
from `vlm_regions` (`rim`/`lip`/`spout`); no region → the pass emits nothing with a reason (an
absent annotation is not licence to guess). The jaw axis is the *measured* smallest principal
direction of a local wall patch, walking the tangent from each seed (the stored region normal is
too unreliable — see above). Design detail: `grasp_passes/rim_pinch/README.md`.

Measured at generation (2026-08-07): 233 made, **220 clear the pre-check (94.4%)** vs 64.0%
(393/614) for the existing passes on the same seven assets — mug 33/33, bowl 38/38, bucket 37/37,
long_tray_bin 30/31, tool_bin 41/45, parts_bin 31/37, pitcher 10/12 (identical under the v3
span_flush reseat). Seat modes (v3): 142 clamped_deep / 70 retreated / 21 span_flush. Gotchas: **YCB container walls
are far thinner than intuition** (bowl 0.94–1.01 mm, mug 1.1–2.6 mm) — a "conservative" 1.5 mm
thickness floor zeroed the bowl; it is 0.5 mm, needed only to exclude duplicated coincident
faces. And **a seed on a flat flange has no measurable wall thickness where it sits** (the
pitcher's rim_0 probe fired down through the vessel and returned 129 mm) — hence
`wall_search_depths` steps down the wall from the seed.

## `shake_validate` — physics validation in our own simulator (annotator)

The authoritative doc is `grasp_passes/shake_validate/README.md` (protocol, metrics, force
derivation, tuning, known limits). Facts from the build worth keeping that it doesn't carry:

- **CUDA-graph capture is a correctness hazard here, not a speed knob**: the coupling's gravity
  flag and the shake window times are kernel inputs — a graph captured before the anchor is fixed
  replays stale values for the whole disturbance. Capture arms only after the anchor is set
  (and CUDA + even substeps). Capture was explicitly exonerated as the cause of the sponge/cable
  drops (re-run with capture off, same result) — do not re-walk that hypothesis.
- **Divergence vs drop is a speed test** (`DIVERGENCE_SPEED` 5 m/s on the average rate since
  gravity-on; free fall covers the 0.15 m escape radius at well under 1 m/s). The one observed
  solver ejection: mug rim-wall overlap at 9.0 m/s. No trial has ever diverged under the shake
  itself. The escape test is displacement-from-start, not distance-from-grasp-centre (a long
  cable gripped near one end fails the latter at rest).
- **Shake fidelity is reported as achieved/commanded amplitude** (`shake_ratio_*`, healthy trials
  ~96%/96%), replacing a base-tracking-error metric that at 2 Hz measured phase lag, not
  amplitude loss. A partial ratio is diagnostic: 96%/19% = dropped part-way through the angular
  shake. Drop LATENCY separates failure modes: load failures go in 0.15–0.28 s after gravity;
  a real shake failure held 2.8 s in.
- **Raising `FORCE_SAFETY` to fix drops backfires** (the full grid behind the README's table):
  the soup can is ejected at safety 5–20 while holding at 2 (curved-surface normal tilt — the
  SOLVERS §6 mechanism); the banana improves monotonically to 0.78 mm slip at 27.9 N; the sugar
  box saturates the 40 N clamp from safety 5 so the sweep measures the clamp. There is no single
  value that makes every sound grasp hold — hence safety stays at the physics-derived 2.
- Free-gripper rig numbers: base `LINEAR_KE 1e5` (≈50 Hz corner, 25× the shake) / `KD 1e3`
  (≈1.6× critical, overdamped so the fixture never rings into the object), angular 3e3/3e1,
  effort cap 1e4 (must never shape the shake); the REAL finger actuators come across
  (ke=300, effort=20 N asserted in selfcheck); fps 60 × 16 substeps. Repeatability measured:
  3×3 repeats byte-identical.
- Cost for sizing: ~60–85 s/trial on an A100 (close-time failures exit fast; ~1/3 is model
  build + MuJoCo compile since each candidate bakes its own base frame). `--check-idempotent`
  on one asset is ~20 min of wall clock — budget for it.
- Its sidecars are DELETED whenever stored poses move (twice so far: v1→v2 seating, then the
  clamp/retreat regeneration) — quality on disk is only meaningful for the pose generation that
  produced it. The pre-check (added as pass v2) shares the hand definition with `pad_seat` via
  `grasp_library.pregrasp_collision`; skips carry `shake_skipped`/`pregrasp_collision` and NO
  quality — an unmeasured `object_in_gripper=0` would be indistinguishable from a measured one.
- **Since pass v3 the trial matches the pre-shaped-approach procedure**: both the pre-check and
  the rig pose the jaws at `min(width + PREGRASP_MARGIN, 80 mm)` instead of the full ACRONYM
  aperture — the trajectory pre-shapes the gripper before the approach, so this is what actually
  executes. Side effect worth knowing: the close starts ≤5 mm per finger from contact, so
  close-time lateral travel (the "pinch slides off the lip" mechanism) is much reduced.
- **The rig is TABLE-LESS by design, and that blindness has a measured cost class.** Two
  instances so far: the centred-seat table-command bug (schema v3 story in grasp-library.md),
  and the 2026-08-12 traj-stage finding that flat cylinders (tuna_can, 33 mm tall) pass shake
  while having NO executable table-top grasp — every candidate jaws across the height, so the
  lower finger must occupy the space between can bottom and tabletop. A "held" verdict here
  means the PINCH holds, never that the pose is executable against a support surface; support
  feasibility is the consumer's check (`traj_gen`'s `pads_clear_table` gate; the offline fix on
  record is a flat-cylinder chord-pinch generator — ONGOING Next queue).
- **Pass v4 (2026-08-11 spec) changes WHICH candidates get trials, not how a trial runs.** Two
  behaviors: (1) `retreated` candidates are excluded from trial selection outright — they are
  weak grasp options, not grasps (3% hold at n=628), and a trial on one is ~60–85 s of GPU spent
  confirming a settled verdict. No annotation is written for them (not even a skip: nothing was
  chosen to run, which is different from "unreachable"). (2) **Incremental annotation**: for
  candidates whose id + pose + width match the pass's own previous sidecar, the previous
  annotation is carried forward verbatim and no trial runs; only new/changed candidates are
  simulated. Without this, any new producer landing (e.g. 10 `llm_retry` candidates) changes the
  upstream digest and re-runs an entire asset's trial matrix to reproduce numbers already on
  disk. v3 sidecars stay valid through `merge.COMPATIBLE_VERSIONS` (poses and trial procedure
  unchanged — v4 only narrows selection); their historical measured-retreated rows remain true
  and remain on disk.

## `llm_retry` — LLM-proposed candidates for objects with no passing grasp (trigger-gated)

The last-resort producer, specified in [llm-retry.md](llm-retry.md) (protocol, trigger, rounds,
annotation design choice, unusable marking — that doc is the contract; read it before touching
the pass). Framework-relevant facts only: it runs ONLY for assets where
`grasp_library.needs_llm_retry` is true (shake-covered record, zero legitimate holds); it is the
one pass allowed to read the MERGED record instead of `ctx.upstream` (its input is the composed
state including shake's annotations); its output is bounded (10 candidates round A, ≤10 round B,
then the merge derives `unusable` and the trigger goes False forever); poses are stored exactly
as the LLM gives them (`seat_mode: "llm"`, span/seat_depth MEASURED at the pose via
`measure_span_at`, no seating algorithm, no retreat); LLM responses are cached like
`vlm_regions`' so `--check-idempotent` holds. Orchestration:
`python -m deformableManipulationTools.grasp_passes.llm_retry cycle`.
