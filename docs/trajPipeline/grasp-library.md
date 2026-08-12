# The grasp library — records, canonical frame, pose convention, pad seating

Per-asset grasp candidates, PRECOMPUTED OFFLINE per catalog object and looked up at
task/trajectory-generation time. The API contract lives in
`deformableManipulationTools/grasp_library.py` (frozen: schema, frame, both pose conventions,
`pad_seat`, hand geometry — passes import from it and must not edit it) — read its module
docstring first. This doc holds the decisions, the measured numbers behind them, and the dead
ends that must not be re-walked. Generation itself: [grasp-passes.md](grasp-passes.md); runtime
selection: [grasp-selection.md](grasp-selection.md).

**SCOPE: rigid objects and soft (FEM) bodies — NOT cloth, NOT bags, and (since 2026-08-11) NOT
CABLES.** A garment/bag has no persistent rest shape, so an OBB describes the asset file, not what
the gripper meets; cloth targets must be resolved against live particle state, a different
mechanism. Cables are skipped the same way, on the measured record: the pipeline's rest-shape
premise fails twice for them — the synthetic capsule-chain probe recovered its own construction
axis (108 identical grasps up to translation/roll), and the full-catalog shake held **0/62** cable
trials at rest-shape spans. A cable generator needs a method that reads real geometry AND measures
its span under load; until one exists, cables are out of the pipeline entirely (no records, no
sidecars, `kind: "cable"` rejected). The canonical statement + enforcement is `grasp_library.py`
(`validate_record` rejects `UNSUPPORTED_KINDS = ("cloth", "cable")`).

## Storage and naming

- One JSON record per catalog object at `assets/objects/grasps/<catalog_name>.json`, keyed by
  **catalog name**, not asset file: two entries can share one USD (`gray_tshirt`/`green_tshirt`),
  procedural kinds have no file, and the name is already the handle in scene/task/success specs.
- **Not** inside `scene_catalog.json` — that file is rendered verbatim into the scene agent's
  prompt and rewritten wholesale by `catalog_ops --ingest/--regen`. No `index.json` — the catalog
  is the index. Sidecars live under `_passes/`, regions under `_regions/`: the underscore prefix
  keeps them out of `available_grasps()`'s non-recursive `*.json` glob.
- Layout mirrors the **ACRONYM** dataset (grasp transforms + five `grasps/qualities/flex/*` metric
  names verbatim, empty until an evaluation pass fills them and names itself in `quality_source`).
  Two departures: poses live in OUR canonical frame (ACRONYM's meshes come canonically posed from
  ShapeNetSem; the catalog's don't), and JSON not HDF5 (git-reviewable, no h5py at runtime, N is
  tens not thousands). ACRONYM's grasp origin sits at the hand base, `ACRONYM_HAND_TO_TCP`
  = 0.112 m behind ours.

## Canonical frame (`obb_extent_desc_v1`)

Origin at the OBB centre; axes = OBB axes sorted by extent descending; signs from mesh asymmetry.
Measured facts:

- `trimesh.bounds.oriented_bounds` at its default `angle_digits=1` **wobbles up to 2.0°** between
  re-posings of the same mesh (1.7 mm frame drift on the banana). `angle_digits=3` is exact at
  1.1 s → 7.6 s per mesh — paid once offline, take the accuracy.
- **A near-degenerate OBB is the real hazard and extent ties do NOT catch it.** A body of
  revolution has a nearly flat volume optimum about its axis: the mug (extents 12% apart, no tie)
  lands on orientations ~1.5° apart → **10 cm** of canonical-frame drift. The frame therefore
  measures its own stability (`ObjectFrame.drift`, fixed probe rotations) and flags `ambiguous`.
  Three detectors, each catching cases the others miss — extent tie, sign-rule fallback, drift
  over tolerance (rubik's cube: first two only; mug: third only). `drift` is a detector, not a
  bound (finite probes under-sample). In practice **63% of stored candidates sit on ambiguous
  frames** — consumers must not read intent into an ambiguous frame's yaw.
- The OBB earns its keep over the catalog's AABB `dims`: banana OBB 0.198×0.069×0.036 vs AABB
  0.109×0.178×0.037 — `dims` smears the long axis across x/y, so a jaw width from it is wrong.
  (Verified in passing: bucket dims are 27 cm, not the 34 cm two docs claimed; both corrected.)
- **Axis signs are counter-intuitive**: they come from asymmetry, so the mug's `+z` points through
  its solid BASE and the rim faces `−z`. Renders that look "upside down" are the frame, not a bug.
  Inspect before assuming an orientation.

Runtime composition: an asset's body frame IS its baked mesh frame (`add_ycb_mesh` spawns at
`transform(pos, quat_z(yaw))`), so `grasp_in_world` composes placement ∘ `asset_from_canonical` ∘
`candidate.transform` and nothing regenerates per scene.

## Pose convention (`tcp_z_approach_x_jaw_v2`) and the measured hand

Origin at the **TCP** (the point `WP.pos` commands), `+z` approach, `+x` jaw axis — a stored pose
is command-ready with no offset math. Measured off the active robot (do not re-derive):

- Finger pads span the approach axis from `PAD_NEAR_Z` **−0.7 mm** to `PAD_FAR_Z` **−54.5 mm**
  relative to the grasp centre — the TCP is at the fingertip TIPS and the entire pad lies behind
  it (`PAD_LENGTH` 53.8 mm, `PAD_MID_Z` −27.6 mm, `PAD_HALF_WIDTH` 10 mm).
- The palm's forward face is at `PALM_NEAR_Z` **−47.0 mm** — *in front of* the pad root; the last
  7.5 mm of pad overlaps the palm's z-range, so deep seating clamps to `SEAT_DEEPEST_Z` −45.0 mm
  (palm + 2 mm), not to the pad root. Palm hull bounding box **204×63×92 mm**; the whole hand is
  only ~54 mm deep. Jaw stroke `MAX_JAW_WIDTH` = 80 mm (ONE definition, here; `task_generator`
  imports it).
- **Hand collision volumes are the true convex hulls, never AABBs.** The palm's AABB spans
  204×63×92 mm around a much smaller solid; with boxes the pre-grasp check falsely rejected 52% of
  banana and 99% of mug candidates. The link shapes are `CONVEX_MESH`, so the hull is exact — and
  the volumes are measured off the active robot at import (`hand_volumes()`; `HandGeometry` in
  `robot.py`), because `settings.yaml` picks the robot and the two hands differ here.
- **The hand is checked PRE-SHAPED, not fully open** (`PREGRASP_MARGIN` = 10 mm): the fingers are
  posed at `min(width + 10 mm, 80 mm)` — a pure translation of the measured hulls along the jaw
  axis, verified against probe grids. This is a CONTRACT with the trajectory stage: **the gripper
  must be pre-shaped to that aperture before the approach**, or the collision check validated a
  different procedure than the one executed. Two consequences, both intended: lateral clearances
  shrink (a rim pinch checks a ~15 mm hand profile instead of 80 mm), and object bulges outside
  the grasped chord that a fully-open hand cleared now honestly block the pre-shaped approach —
  which is why the retreat/blocked counts ROSE when this landed (v3→v4: retreated 448→596,
  blocked 371→398). The shake rig starts its trials at the same pre-shaped aperture.

**The v1 defect this convention fixed** (2026-08-05, kept for the numbers): v1 stored the grasp
centre ON the material, so the pads — all behind the TCP — gripped with their tips and left ~half
the object forward of the fingers. Median advance still needed: `geometric` +27.5 mm (banana,
tightly clustered p10–p90 +22.7…+33.3), `obb_face` under its retired `FINGER_DEPTH` rule
+61.0 mm and scattered (+27…+89) — two different wrong rules, replaced by ONE shared function
since pad placement is a property of the hand, not of how a grasp was found. v1 is rejected **by
name** in `validate_record`: v1/v2 share axes, so a v1 pose loads and merges cleanly and is wrong
only by ~28 mm nothing downstream can detect. Like-for-like effect on the identical 85 banana
candidates: pre-check pass **4.7% → 47.1%**.

## `pad_seat` and the three seat modes

`pad_seat(centre, approach, jaw, width, vertices, faces)` raycasts a 5×5 grid through the jaw
column for the material span, seats the pose, then tests the hand's own hulls at the result (the
same test as the shake pre-check — one definition of "where is the hand", both at the pre-shaped
aperture). The depth is ONE rule — **`advance = min(fingertip cap, palm cap)`**, as deep as
possible subject to both — and `seat_mode` names which cap bound. `seat_mode`, the measured
`span`, and `seat_depth` (the near material edge in the grasp frame) are **required** schema
fields (`SCHEMA_VERSION` 4; `make_candidate` refuses to default them — an unmeasured pose must
not store a claim):

- **`span_flush`** (fingertip cap bound; spans ≲ 44 mm): the whole span sits flush against the
  fingertip end of the pads (far material edge on `PAD_NEAR_Z`) — the fingertips do not extend
  past the object. This replaced the retired **centred** rule (span midpoint on `PAD_MID_Z`),
  which commanded the tips `27.6 − span/2` mm PAST the object's far surface — for a top-down
  grasp on a resting object that is INTO THE TABLE (9.6 mm on the 36 mm banana; measured on the
  v2 library: **91 candidates across 11 assets commanded the TCP below the tabletop**, worst
  −21 mm) — and no online check could see it (clearance samples only the corridor above the
  pose; the shake rig has no table). The trade accepted: the object sits at the tip end instead
  of centred, giving up the `27.6 − span/2` mm slide-out margin under load — and that margin is
  MEASURED to matter: under gravity-along-approach (the lift load of a top-down pick) the flush
  stratum holds 4.2% vs the centred seat's 25.0% (2026-08-07 gates, n=24 each). RESOLUTION
  (schema v5): the two depths are stored as a **DEPTH-VARIANT PAIR** — see below.
- **`centred`** (the `_ctr` depth variant): where the span admits a palm-safe centred seat
  (`pad_seat(rule="centred")`), the measured-stronger centred depth is stored as a SEPARATE
  candidate — id + `_ctr`, label `centred_variant` — collision-checked (with retreat) and
  shake-validated offline like any pose. A variant that retreats keeps its intermediate depth (a
  partial recovery); one that is blocked or converges back onto the flush seat is not stored.
  ONLINE, `grasp_select`'s **depth stage** picks between the pair: it probes the space the
  centred fingertips would occupy BEYOND the object's far surface (`beyond_clearance` — table
  plane + obstacle boxes along the protruded finger lines); room → the centred variant
  supersedes its flush sibling; no room (e.g. the object rests on the table under a top-down
  grasp) → the variant is rejected and the always-executable flush pose stands.
- **`clamped_deep`** (palm cap bound): **"jaws as deep as safe", NOT "object centred"** — near
  material at `SEAT_DEEPEST_Z`, remainder protruding forward past the fingertips. The midpoint
  fallback it replaced was catastrophic and measured: centring an over-deep span pushes the
  overhang backwards into the palm — the 62.9 mm apple skipped **116/116** pre-checks as
  midpoint-seated vs 2/116 with no advance at all. Catalog-wide skip rate fell 75.5% → 45.7%
  with the clamp (10 assets went from 100%-skipped to none). (The centred rule also violated the
  palm cap outright for 44–54 mm spans, which survived only by falling through to the retreat —
  the unified min-rule seats them palm-safe directly.)
  Measured at full-catalog scale (2026-08-08, 1719 trials): **hold rate tracks material depth in
  the jaws** — thin-span pinches (median 17 mm) hold ~1–4% at ANY depth, near-pad-length spans
  (35–43 mm) hold ~52%, deeper-than-pads clamps 42%. The variant pair's head-to-head favours
  centred where anything holds, but thin spans dominate the pair population.
- **`retreated`** (rule seat leaves the hand inside the object): back off along the approach in
  1 mm steps to the deepest collision-free depth that keeps material between the pads. No clear
  depth → `seat.blocked`; the candidate keeps its rule seat and carries `seat_blocked` — marked,
  never deleted. Effect at introduction (on the v2 library): skip 45.7% → **20.3%** (808→359 of
  1767; the 359 remaining ARE the seat_blocked set — seat and pre-check agree
  candidate-by-candidate). Idempotent (retreat grid anchored to the rule seat; re-seating stored
  candidates advances 0.0000 mm) and non-invasive (every previously-clear pose byte-identical).
  The retreat distance is BAKED INTO the stored transform — the pose is final and command-ready;
  `seat_mode` records how the depth was chosen, not an adjustment still to apply.

With `seat_depth` + `span` a consumer knows where the material sits relative to the pads without
re-measuring the mesh — material occupies `[seat_depth, seat_depth + span]` in the grasp frame
for EVERY mode, including `retreated` (which `seat_mode` + `span` alone cannot locate). Caveat
either way: `span` is an ENVELOPE — on a multi-run column (the mug's handle → void → wall) it
includes the voids, so it can overstate material by tens of mm on the worst approaches.

**The vessel-class findings behind the retreat** (mug 83/96, bowl 72/76, pitcher 66/70 pre-check
skips diagnosed read-only, 2026-08-06 — the full method + scripts are reproducible;
`skip_diag.py`/`raycast_runs.py`/`depth_sweep.py` in that session's scratchpad):

- **Nothing ever collides forward of the fingertips — 0% on all three assets.** Collisions are
  lateral (mug 68%, bowl 78%, pitcher 66% of colliding vertices) against the near wall and rim,
  or across the palm (bowl: 75% of colliding vertices in the palm zone — the bowl is 55 mm tall
  against the palm's 47 mm standoff, so 46/72 of its skips are palm-only).
- **Deeper NEVER helps**: the depth sweep found zero candidates on any of the three assets
  rescued by advancing; every rescue is a retreat. Per-asset nearest-clear medians: mug
  −34.5 mm (window ~8 mm), bowl −41.0 mm (window ~**3 mm** — the 1 mm retreat step is near the
  minimum that finds it), pitcher −28.0 mm (window ~16 mm). ONGOING's old "median retreat
  ~34 mm" was the mug's number, not the catalog's.
- **The span-segmentation hypothesis was premise-true, conclusion-false**: the raycast envelope
  really does swallow handle→void→wall as one span (mug spans to 108.8 mm with voids to 62.7 mm
  inside), but re-seating on the first material run rescues only **4/83** mug, 2/72 bowl, 0/66
  pitcher — 10 of the 15 handle-approach candidates were already *contained* seats and skipped
  anyway. The arithmetic: handle depth ~12–15 mm + void < 53.8 mm pad length, so any pad-centred
  handle seat pushes fingertips into the wall's z-range, where an 88–94 mm body can't fit between
  80 mm jaws. Rejected in favour of the retreat (44+15+27 rescues).
- Diagnostic heuristic that generalizes: **lateral-contact fraction predicts fixability** (mug
  fixable median 0.92 vs unfixable 0.59; palm-alone + low-lateral + contained + single-run means
  the seat is already right and no depth helps).
- Post-retreat container skip rates (supersede the pre-retreat 86.5/94.7/94.3% figures): mug
  40.6%, bowl 75.0%, pitcher 55.7%, bucket 47.1%, long_tray_bin 12.2%, parts_bin 11.6%,
  tool_bin 3.9%.

Two implementation constraints: `pad_seat` needs the MESH, so it cannot live inside
`make_candidate` — generators must call it explicitly (README rule 2; nothing enforces it, and a
skipped call produces poses that validate and merge while being ~28 mm wrong). Point-set kinds
(tet meshes, the procedural rod) have no faces — the raycast falls back to projected vertices and
GROWS the footprint until it holds material (the 235-vertex sponge needs an 82 mm footprint
against a 20 mm pad; without growth every soft-mesh candidate is silently dropped). Cost:
~127 ms/call (rebuilds the ray BVH per call — memoize per asset in bulk audits); seat after
dedup/caps, not on the raw candidate stream.

## Candidate statuses (2026-08-11, user-directed) — legitimate / weak / discarded

The full-catalog shake settled what the weak seat modes are for. Three statuses, derived
CENTRALLY at merge time (`grasp_passes/merge.py`) so no generator or consumer re-implements the
accounting:

- **Legitimate** — every stored candidate whose `seat_mode` is not `retreated`. Only these count
  toward "does this object have a working grasp" (`grasp_library.legitimate_candidates`,
  `record_holds`).
- **Weak** — `retreated` candidates are marked as a *weak grasp option*: the merge stamps
  `WEAK_GRASP_LABEL` (`"weak_grasp_option"`) on them. They stay in the record (they are honest
  reachability statements, and the LLM retry stage feeds on them) but they are NOT legitimate
  candidates: **no physics testing is spent on them** (`shake_validate` v4 excludes them from
  trial selection — measured basis: 3% hold at n=628), `grasp_select`'s default pool excludes
  them, and they never satisfy the zero-hold trigger.
- **Discarded** — `seat_blocked` candidates (hand collides at EVERY depth on the approach) are
  **dropped from the merged record automatically**: no collision-free grasp exists, so there is
  nothing for any consumer to command. The measurement is not lost — the generator's sidecar
  still carries the candidate + label — but the record, which is the library consumers read, no
  longer contains them. Annotations addressed to a discarded candidate (the old `shake_skipped`
  entries) are dropped by the merge with a count in the record notes, not an error.

Two record-level verdicts, both derived at merge, both durable results a consumer must treat as
"no grasp": **out of reach** (empty record — nothing fit the jaw; `is_out_of_reach`) and
**unusable** (`is_unusable` — the LLM retry stage ran both its rounds and nothing held; see
[llm-retry.md](llm-retry.md)).

**`overhang` is tracked, not scored.** For every `clamped_deep` candidate the merge computes and
stores `overhang` [m] — the material protruding forward past the fingertips,
`(seat_depth + span) − PAD_NEAR_Z`, floored at 0 — because the full-catalog shake showed
clamped-deep holds cluster on modest-overhang bodies and the number should be on disk for
analysis. It is deliberately NOT a `grasp_select` scoring input; gating on it is a future,
evidence-backed decision.

**Why the schema stays at v5**: sidecars are read through the same strict
`schema_version == SCHEMA_VERSION` gate as records, so a bump would orphan every sidecar on disk —
including the 1719 measured shake trials, whose poses these changes do not move. All of the above
is therefore ADDITIVE (a new optional `overhang` field, new labels, the new `llm` seat mode) and
DERIVED at merge; regenerating the records is one cheap `merge --all`, no pass re-runs.

## Schema and versioning rules

- **The CONVENTION names the pose format; the PASS VERSION names the rule that produced the
  numbers; only the pair identifies what a stored pose means.** The merge gates on both:
  `merge._check_agreement` rejects a sidecar whose recorded pass version differs from the
  registered pass's current version ("re-run the pass; do not relabel") — closed after a restored
  pre-clamp backup sidecar (same mesh, frame, convention label) merged cleanly while meaning a
  different seating rule. `run_pass`'s version-keyed skip cannot catch a restored file.
  Unregistered passes are exempt (selfcheck throwaways); retiring a pass's output = deleting its
  sidecars.
- **Zero-candidate assets are recorded as OUT OF REACH, not failures** (`merge.OUT_OF_REACH_NOTE`,
  `is_out_of_reach()`): an empty record with the note is distinguishable from "nothing has run
  yet", which a consumer must be able to tell apart.
- Reading records stays **stdlib+numpy**; SEATING/pre-checking pulls the robot build lazily
  (first `pad_seat`/`pregrasp_collision` call per process).
- Executable invariants (the repo has no test framework):
  `.venv/bin/python -m deformableManipulationTools.grasp_library --selfcheck [--asset ...]`.

## Catalog coverage

30 catalog entries → **22 supported** (−6 cloth family: 3 garments, 3 bags; −2 cables:
`power_cable`, `nylon_rope`, removed 2026-08-11 — see the SCOPE paragraph) → **18 rigid**
(`ycb_mesh` 16 + `rubiks_cube` + `rigid_box`), the kinds the geometric/vlm generators declare —
both measure a rest-shape span, which is not the right number for a body that deforms under the
grasp. The 4 supported-but-not-rigid (`sponge`, `foam_brick`, `banana_soft`, `raspberry_cube`)
are covered only by `obb_face` (+`fixture`); closing that gap is a generator that measures a span
under load, not a `kinds` widening. (The cable-probe artifact is recorded in the SCOPE paragraph:
the medial probe on the synthetic capsule chain recovered `catalog._rod_vertices`' own
construction axis, not structure — kept here so nobody re-walks it if cables re-enter.)
