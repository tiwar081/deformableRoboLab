# The LLM retry stage — last-resort grasp candidates for objects nothing else can hold

Designed 2026-08-11 (user-directed). This doc is the CONTRACT for the stage: agents implementing
or consuming it build against what is written here. Code lives in
`deformableManipulationTools/grasp_passes/llm_retry/`; the shared constants and helpers it names
live in `deformableManipulationTools/grasp_library.py`.

## What it is for

When every generator has run and the physics validation (`shake_validate`) has covered the result,
some objects end with **no legitimate candidate that holds** (`object_in_gripper == 1`). For those
objects — and ONLY those — a multimodal LLM is asked to study the object and everything that was
already tried, and to propose new candidates that go back through the same physics pipeline. If two
LLM rounds both fail, the object is **deemed unusable** and marked so downstream stages can skip it
without re-deriving the verdict.

This is a LAST-RESORT stage, not a peer generator: it runs only on the trigger below, it is
bounded (two rounds, ever), and its output is validated by exactly the same physics as everyone
else's.

## Candidate statuses (the accounting the trigger runs on)

Defined in [grasp-library.md](grasp-library.md) ("Candidate statuses") and implemented centrally —
summarized here because the trigger depends on them:

- **legitimate** — any stored candidate whose `seat_mode` is not `retreated`. Only legitimate
  candidates count toward "does this object have a working grasp".
- **weak** (`seat_mode == "retreated"`, label `weak_grasp_option`, stamped at merge) — kept in the
  record as reachability information and as raw material for THIS stage, but never counted as a
  grasp, never physics-tested (shake v4 skips them), and excluded from `grasp_select`'s default
  pool. (Measured basis: 3% hold at n=628 — reachability statements, not grasps.)
- **discarded** (`seat_blocked`) — a pose whose approach clears the hand at NO depth has no
  collision-free grasp; the merge drops it from the record entirely (the generator sidecar keeps
  the measurement). These never reach this stage or any consumer.

## Trigger — `grasp_library.needs_llm_retry(record)`

True iff ALL of:

1. the record's kind is supported (rigid or soft FEM; cloth/bags/cables are out of scope of the
   whole pipeline);
2. the record is not already marked unusable (`is_unusable`);
3. every legitimate candidate is shake-covered — carries measured quality
   (`quality_source` set) or an honest skip (`shake_skipped` label). An untested candidate is
   "not yet known", not "failed", and the LLM must not be invoked over it;
4. **zero** legitimate candidates hold (`quality.object_in_gripper == 1.0`).

An EMPTY record (out-of-reach: every generator ran, nothing fit the jaw) satisfies 3–4 vacuously
and IS in scope — "no candidates at all" is exactly the population the stage exists for.

Weak (retreated) candidates play no part in the trigger: an object whose only holds sit on
retreated poses still triggers (those holds are not legitimate grasps).

## Protocol — two rounds, then unusable

State lives in the `llm_retry` sidecar like any pass's output; rounds are distinguished by
candidate id prefix (`a00…a09` round A, `b00…b09` round B; namespaced `llm_retry/a00` etc.).
The sidecar is CUMULATIVE: round B re-emits round A's candidates unchanged so their measured
quality (which lives in shake's sidecar, keyed by id) stays attached through the merge.

**Round A — blind generation from the record.** Input package to one multimodal call:

- the object's catalog entry (kind, mass, mu, dims, description);
- the canonical frame: extents, ambiguity flag, and an explanation of the frame convention;
- the SIX canonical view renders (reuse `grasp_passes.vlm_regions.views` — same renderer, same
  lit-rotation scheme, same view names);
- the tried candidates from the merged record, serialized compactly per candidate: id, source,
  seat_mode, position/approach/jaw axis, width, span, seat_depth, and its OUTCOME (held /
  failure labels + motion metrics / pre-check skip). This is "the corresponding json file" in
  distilled form — the LLM sees everything that was tried and how it failed;
- the WEAK (retreated) candidates, explicitly flagged as reusable raw material: the LLM may adopt
  or modify them (they were never physics-tested; their poses are collision-free by construction);
- stored semantic regions from `_regions/<name>.json` when present;
- the measured hand geometry it needs to choose a seat: TCP at the fingertip tips, pads spanning
  `PAD_NEAR_Z −0.7 mm … PAD_FAR_Z −54.5 mm` behind it, palm face at `PALM_NEAR_Z −47 mm`,
  deepest safe near-material edge `SEAT_DEEPEST_Z −45 mm`, jaw stroke ≤ 80 mm, and the
  pre-shaped-approach contract.

Output (JSON-schema structured): exactly **10 candidates**, each `position` (metres, canonical
frame), `approach`, `jaw_axis`, `width`, a semantic `label` from the `vlm_regions` vocabulary (or
`body`), and a one-line `rationale`.

**No seating algorithm runs on these poses.** The LLM chooses the seat depth itself; the pose is
stored EXACTLY as given with `seat_mode: "llm"`. What DOES run is measurement and validation:

- `span`/`seat_depth` are MEASURED at the given pose by the same ray machinery `pad_seat` uses
  (`grasp_library.measure_span_at` — probe only, no pose motion), because they are required
  schema fields and consumers locate material with them;
- a pose with no material in the jaw column, a near-material edge in front of the TCP (gripping
  air), a seat deeper than the palm allows (`seat_depth < −0.05`), or `width > MAX_JAW_WIDTH` is
  DROPPED at authoring and recorded as an authoring failure — it re-enters the round-B feedback
  so the LLM learns it was geometrically invalid;
- the shake pre-check and trial run unchanged: a pose that buries the hand in the object is a
  `pregrasp_collision` skip like anyone else's.

**Annotation of the new candidates.** The OBB face bucket is derived mechanically
(`make_candidate` + an `obb_bucket` run — nothing new). The SEMANTIC annotation is done by the
SAME LLM in the SAME call — it stamps its `label` on each candidate directly. Design choice,
deliberate: the generating call already holds the renders and the geometry context, so a separate
VLM-designated agent would add a second transport, a proximity join, and no parallelism win at
this population size (a handful of assets, by construction). The `_regions/` store is not written
by this stage.

**Round B — visual feedback, once.** Runs ONLY if every round-A candidate is shake-covered and
none holds. The LLM is shown:

- a render of EVERY grasp it tried, drawn as pad markers on the object (reuse
  `grasp_passes.geometric.viz` — one labelled image per tried candidate);
- per candidate, what happened: dropped-at-authoring reason, `pregrasp_collision` skip, or the
  shake outcome (failure labels, close/shake motion, drop latency where available).

It then emits up to 10 NEW candidates (`b*` ids, same schema, same authoring validation). These go
through the same obb_bucket + shake cycle.

**Exhaustion.** Round B is the ONE retry. When round B's candidates are all shake-covered and
still nothing holds, the pass stamps `llm-retry-exhausted` (constant
`grasp_library.RETRY_EXHAUSTED_TOKEN`) in its sidecar notes; the MERGE derives the record-level
verdict from that plus the zero-hold accounting and appends `UNUSABLE_NOTE` to the record's notes.
`grasp_library.is_unusable(record)` reads it back. Like `OUT_OF_REACH`, unusable is a RESULT: task
generation and selection must treat the object as having no grasp, and nothing re-invokes the LLM
for it (`needs_llm_retry` returns False).

## Transport, caching, determinism

- The call goes through `agentic_pipeline.scene_generator._messages_request` (Claude Code OAuth,
  JSON-schema structured output, `DEFAULT_MODEL` → `FALLBACK_MODEL` fallback) — the same one
  place that knows how to authenticate, exactly as `vlm_regions.prompt` does, with the same lazy
  import to avoid the package cycle.
- Responses are CACHED per (mesh_sha1, prompt version, pass version, round, feedback digest),
  mirroring `vlm_regions`: re-running the pass re-reads the cache, which is what lets a
  nondeterministic annotator pass `--check-idempotent`. Transport failure RAISES — "the LLM
  proposed nothing" must never be cached as a finding.
- The pass is a normal producer otherwise: it writes only its own sidecar, its ids are
  namespaced, the merge dedups by its source tag.

**One deliberate exception to the pass rules**: `llm_retry` reads the MERGED record
(`load_grasps`) — read-only — instead of `ctx.upstream`. Its trigger and its input package need
the composed state (all producers' candidates WITH shake's annotations applied), which is exactly
what the record is. It still never writes the record. This exception is stated in
`grasp_passes/README.md` and applies to this pass alone.

## Orchestration

One command drives the whole cycle for one asset or the full trigger population:

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle [--asset NAME]
```

Steps per asset, each through the existing `run_pass`/`merge` machinery: trigger check → round A
generation → merge → `obb_bucket` → `shake_validate` (v4: trials ONLY the new candidates — see
below) → merge → re-check; if still zero holds → round B → merge → obb_bucket → shake → merge →
re-check; if still zero holds → exhausted (the merge stamps unusable). Idempotent at every step:
re-running `cycle` on a completed asset does nothing.

## Interaction with `shake_validate` v4

Two v4 behaviors this stage depends on (spec in [grasp-passes.md](grasp-passes.md)):

- **Incremental annotation**: shake carries forward its previous sidecar's annotations for
  candidates whose id + pose + width are unchanged, and trials only new/changed candidates.
  Without this, adding 10 LLM candidates changes the upstream digest and re-runs a whole asset's
  trial matrix (~an hour of GPU per asset) to reproduce numbers already on disk.
- **Retreated candidates are excluded from trial selection** (they are weak options, not grasps).
  Historical v3 measurements of retreated poses remain on disk and remain true; the merge accepts
  v3 sidecars alongside v4 via `merge.COMPATIBLE_VERSIONS` because v4 only narrows trial
  selection — poses and procedure are unchanged.

## Bounds and non-goals

- Two LLM rounds per object, EVER. The unusable mark is durable; clearing it is a human decision
  (delete the `llm_retry` sidecar to re-arm).
- The stage proposes CANDIDATES only — force targets, trajectories, and task feasibility remain
  downstream concerns.
- Out of scope with the rest of the pipeline: cloth, bags, and (since 2026-08-11) CABLES — see
  the scope statement in [grasp-library.md](grasp-library.md). The empirical cable verdict
  (0/62 held at rest-shape spans; a cable generator needs span-under-load + real geometry) is
  recorded there; nothing about this stage changes it.
