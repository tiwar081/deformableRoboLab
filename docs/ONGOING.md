# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Conventions for writing entries: [docs/SKILLS/update-ongoing.md](SKILLS/update-ongoing.md).
Promoting durable content out to `docs/` and resetting this file (done per big task):
[docs/SKILLS/promote-ongoing-to-docs.md](SKILLS/promote-ongoing-to-docs.md).

## In flight: per-asset grasp candidates (starting 2026-08-04)

Precompute grasp candidates offline per catalog object, look them up at task/trajectory-generation
time. **The substrate is landed and documented** — record store + canonical frame + pad seating
([docs/trajPipeline/grasp-library.md](trajPipeline/grasp-library.md)), the pass framework and all
seven passes ([docs/trajPipeline/grasp-passes.md](trajPipeline/grasp-passes.md)), and selection
over a placed object ([docs/trajPipeline/grasp-selection.md](trajPipeline/grasp-selection.md)).
SCOPE (rigid + FEM; **no cloth, no bags, no cables** — cables removed 2026-08-11, see the
DESIGNED entry below) is stated and enforced in
`deformableManipulationTools/grasp_library.py`. What remains in flight: filling `quality` by
physics validation, deciding what the weak seat modes are for, and the trajectory stage itself.
**Nothing of this pipeline is committed** — it is all uncommitted working tree on top of
`bd3e187`.

### LANDED (2026-08-07): the `span_flush` seat rule (schema v3) — user-directed

The user asked whether the online phase commands the stored seat depth verbatim (it does — v2
poses are command-ready by design) and spotted the consequence: a CENTRED seat on a thin object
commands the fingertips `27.6 − span/2` mm PAST the object's far surface — **into the table** for
a top-down grasp on a resting object, and nothing online can see it (clearance samples only the
corridor above the pose; the pre-check tests hand-vs-object; the shake rig has no table).
Measured on the v2 library before the fix: **91 candidates across 11 assets commanded the TCP
below the tabletop** (banana 30, worst −21 mm).

What landed (docs: [trajPipeline/grasp-library.md](trajPipeline/grasp-library.md)):

- `pad_seat` depth is ONE rule, `advance = min(fingertip cap, palm cap)`; the mode names which
  cap bound: **`span_flush`** (span seated flush against the fingertip end — replaces `centred`)
  or `clamped_deep`. Side fix: 44–54 mm spans, centred, violated the palm cap outright and
  survived only via the collision retreat — the min-rule seats them palm-safe directly.
- **`span` is a REQUIRED candidate field** (the measured material extent along the approach), so
  a consumer knows where material sits relative to the pads without re-measuring the mesh.
  `SCHEMA_VERSION` 2→3 (v2 records and the `centred` mode are rejected by name); producers
  bumped (fixture v5, geometric v5, obb_face v6, rim_pinch v2). The whole store was regenerated
  from clean (v2 backup in the 2026-08-07 session scratchpad, `grasps_v2_backup/`) — old-schema
  sidecars cannot even be READ (`run_pass`'s skip check validates them), so regeneration means
  delete-and-rerun, not rerun-over.

Verified after regeneration: below-table commands **91 → 11** (the residual are oblique-approach
edge cases where the TCP dips below the plane near an object's edge — an online-clearance
concern, not a seat-rule failure); re-seating stored poses moves 0.0000 mm (idempotent); both
selfchecks pass.

### LANDED (2026-08-07): schema v4 — pre-shaped collision check + `seat_depth` (user-directed)

Two changes on top of the span_flush work, per the user's review:

- **The hand collision checks pose the fingers at the PRE-SHAPED aperture** —
  `min(width + PREGRASP_MARGIN, 80 mm)`, `PREGRASP_MARGIN` = 10 mm — instead of fully open, in
  BOTH places that test the hand (`pad_seat`'s retreat, the shake pre-check), implemented as a
  verified pure translation of the measured finger hulls. **This is a CONTRACT with the future
  trajectory stage: the gripper must be pre-shaped to that aperture before the approach.** The
  shake rig (pass v3) starts its trials at the same aperture, so validation runs the procedure
  that will execute. Effect on the library: retreat/blocked ROSE (retreated 448→596, blocked
  371→398) — not a regression: bulges outside the grasped chord that a fully-open hand cleared
  genuinely obstruct a pre-shaped approach, and the check now says so.
- **`seat_depth` is a required candidate field** (near material edge in the grasp frame):
  material occupies `[seat_depth, seat_depth + span]` for every mode — closing the gap where a
  `retreated` pose's material could not be located from `span` + `seat_mode` alone.

### Library state on disk (swept 2026-08-07, post-v5 regeneration)

- 24 merged records, **2117 candidates** (2000 + 117 `_ctr` depth variants): geometric 1387 ·
  obb_face 464 · rim_pinch 246 · fixture 20 (variants included, measured from disk). All
  `tcp_z_approach_x_jaw_v2`, schema v5 (v2/v3/v4 backups in the 2026-08-07 session scratchpad).
- seat_mode: 266 `span_flush` / 85 `centred` / 1138 `clamped_deep` / 628 `retreated`; 398 carry
  `seat_blocked`. **1719 pre-check clear.** Below-table commands on primaries: 12 (the
  oblique-approach residual; was 91 under the old always-centred rule — `_ctr` variants are
  deeper by design and gated online by the depth stage instead).
- Regions store intact (18 files, 44 regions); `obb_bucket` covers all 2000 (rim_pinch included).
- **Quality: tool_bin is the first fully-measured record** (see the retest below). All other
  assets carry no measured quality; sidecars deleted at each pose migration.

### MEASURED (2026-08-07): tool_bin end-to-end retest — the zero-hold verdict is OVERTURNED

Full `shake_validate` pass on the regenerated tool_bin record under the complete new procedure
(span_flush/min-cap seating + pre-shaped pre-check + pre-shaped trial start): 121 candidates,
16 pre-check skips, **105 trials, 15 HELD (14.3%)** — against **0 holds in ~6 prior trials**
across both earlier gates. Held by family: rim_pinch clamped_deep 6 · geometric clamped_deep 5 ·
geometric retreated 2 · span_flush 1 · obb_face 1. The six rim/lip pinches that hold are exactly
the family whose close-time lip slides made tool_bin the evidence-backed zero-hold — the
pre-shaped close (≤5 mm of travel per finger) is the change aimed at that mechanism. Best holds
shake at 0.3–2 mm slip; the rim pinches hold with 19–75 mm of settle-then-hold motion. Drops are
still dominated by close-time push-outs (78 of 90). tool_bin's record now carries measured
quality on all 105 tried candidates (`quality_source: shake_validate`).

### MEASURED (2026-08-07): v3 gate, 86 trials — the flush seat's margin loss is LARGE and real

Same design as the v2 gate below (round-robin across assets, rim_pinch its own group, pre-check
clear only); full rows in the 2026-08-07 session scratchpad (`strat3_shake.json`).

| stratum | n | held | hold% | v2 gate (centred library) |
|---|---|---|---|---|
| span_flush | 24 | 1 | **4.2%** | centred 25.0% (6/24) |
| clamped_deep | 20 | 9 | **45.0%** | 20.0% (4/20) |
| retreated | 24 | 0 | 0.0% | 4.2% (1/24) |
| rim_pinch | 18 | 8 | 44.4% | 38.9% (7/18) |

- **`span_flush` holds 4.2% — the tip-flush seat collapses under load, and that is physics, not
  noise** (1/24 vs 6/24, Fisher p≈0.04). The shake's gravity is along the approach — exactly the
  LIFT load of a top-down pick — and with the material flush at the fingertip end, any slip
  immediately starts leaving the pads: the `27.6 − span/2` mm margin the centred seat had is
  precisely what the load phase consumes. The old centred seat held 25% but was uncommandable
  against a table (91 poses through the tabletop). **Neither offline rule is right on its own;
  the depth question is now an ONLINE one** — see the open decision below.
- **`clamped_deep` is the best non-container family at 45%** — more than doubled, plausibly
  because it absorbed the 44–54 mm ex-centred spans that now seat palm-safe with real pad
  engagement instead of bouncing off the retreat. Membership changed, so the 20%→45% jump is
  confounded with composition; the level, not the delta, is the finding.
- **`retreated` 0/24** — confirms the v2 verdict: not a grasp family; reachability statements.
- **`rim_pinch` stable-to-better (44.4%)** — the container answer holds up under the reseat.

### DECIDED + LANDED (2026-08-07): depth-variant pairs, schema v5 (user's design)

The seat-depth question is RESOLVED, by a cleaner design than the continuous online advance:
**both depths are stored as separate, individually validated candidates.** Where the span admits
a palm-safe centred seat, generators emit the centred depth as a `_ctr` companion candidate
(label `centred_variant`) alongside the span_flush primary; both go through the collision
check/retreat and the shake test offline. ONLINE, `grasp_select`'s new **depth stage** makes a
binary pick: `beyond_clearance` probes the space the centred fingertips would occupy past the
object's far surface (table + obstacle boxes) — room → centred supersedes flush; no room →
flush stands. Landed: `pad_seat(rule="centred")` + `centred_variant()` in grasp_library
(schema v5, `centred` re-enters SEAT_MODES), all four generators emit variants (fixture v7,
geometric v7, obb_face v8, rim_pinch v4), `beyond_clearance` + the depth stage + centred
containment scoring in grasp_select; all three selfchecks pass. Regenerated: **2117 candidates**
= 2000 + 117 `_ctr` variants (85 seated clean `centred`, 32 landed `retreated` — partial depth
recovery, kept; blocked/duplicate variants not stored). Pair geometry spot-checked (flush
−34.3 mm → centred −44.4 mm on a 34 mm span). v4 store backed up in the session scratchpad.

### SUPERSEDED (2026-08-07): the v2 centred-library gate (kept for comparison)

Round-robin across assets, 82.9 min; rows in `strat2_shake.json`.

| stratum | n | held | hold% | drops during close / under load |
|---|---|---|---|---|
| centred | 24 | 6 | 25.0% | 7 / 11 |
| clamped_deep | 20 | 4 | 20.0% | 6 / 10 |
| retreated | 24 | 1 | **4.2%** | **19 / 4** |
| rim_pinch | 18 | 7 | **38.9%** | 9 / 2 |

Verdicts restated and confirmed by the v3 gate above (retreated = not a grasp family; rim_pinch
= the container answer, structured bowl/bucket good vs tray/pitcher/tool_bin lips bad). Two
caveats that carry forward to BOTH gates: **absolute rates are weak at these n** (the banana
centred control moved 53.8% → 30% across regenerations; any decision hanging on an absolute rate
needs a ~30-trial control first), and trial mean is ~58–85 s depending on failure mix — use
60–85 s/trial for sizing.

### Zero-hold roster + failure taxonomy (2026-08-06/07)

13 of 24 objects had no held grasp across the earlier gates — but only **tool_bin (0/6, incl.
0/3 rim_pinch)** and **pitcher (0/4)** carried real evidence of "no working grasp".
**tool_bin's verdict is OVERTURNED by the full retest above (15/105 held under the v4
procedure)** — which also weakens the pitcher verdict by analogy (same close-time failure
mechanism, n=6; retest it before treating it as out of reach).
**steel_cube (0/5)** looks like a capability limit, not a sampling gap: 1 kg of solid steel puts
its derived force target on the 40 N clamp. The other ten are under-sampled (1–6 trials each) —
do not read them as "ungraspable" yet. Six failure modes observed across the gates: pushed out
during close · axial-out lever under load · in-place retention loss · force-cap non-convergence ·
pre-check rejection · divergence. Attribution that matters: retreat-mode deaths happen DURING THE
CLOSE (mechanism problem), lever deaths UNDER LOAD (pose-quality problem). Held-grasp renders:
`outputs/grasp_viz/holds/` (18 PNGs + `holds_overview.png`).

### Deep sweep on the zero-hold objects: launched 2026-08-07 ~01:21, never reported

Best-mode-first (centred → rim_pinch → clamped_deep → retreated), up to 12 pre-check-clear poses
per object, early-stop at 2 holds, recording retreat distance / span depth / squeeze-vs-target /
escape direction. **No artifact ever landed on disk — treat it as NOT RUN**; re-launch if the
zero-hold question still matters after the open decision below.

### Open decisions (the user's)

- **What uncontained/`retreated` candidates are FOR.** Options: keep + label for a selector that
  prefers contained; restrict generators to contained poses; or treat "graspable at all" as
  requiring a contained pose — which makes deep bodies (cheez_it, parts_bin, pitcher, wood_block)
  out-of-reach statements, matching intuition for an 80 mm-jaw parallel gripper.
- **When to commit.** The entire grasp pipeline (library, passes, selection, records, docs) is
  uncommitted working tree.

### MEASURED (2026-08-08): the FULL-CATALOG shake — 1719 trials, every record quality-filled

Ran as 4 parallel shards on disjoint assets (7.2–12.9 h each; apple the long pole), one merge
after all four. **422 of 1719 held (24.5%)**; 398 pre-check skips (the seat_blocked set). Every
candidate in the library now carries measured quality or an honest skip.

| cut | held/tried | hold% |
|---|---|---|
| clamped_deep | 356/840 | **42%** |
| span_flush (all) | 42/166 | 25% |
| centred `_ctr` variants | 3/85 | 4% |
| retreated | 21/628 | **3%** |
| obb_face | 180/445 | 40% |
| rim_pinch | 76/228 | 33% |
| geometric | 159/1026 | 15% |

- **Material depth in the jaws is THE hold predictor.** Flush poses that admit a distinct
  centred variant are thin pinches (span median 17 mm) and hold **1%**; flush poses whose
  centred depth coincides with flush (span 35–43 mm, near pad length) hold **52%**;
  deeper-than-pads clamps hold 42%. Thin-span pinches fail at ANY depth — which is why the
  centred variants (same thin population) also sit at 4%.
- **The depth-variant head-to-head still favours centred where anything holds**: among the 117
  tried pairs — variant held 5, flush held 2, both dead 112 (thin spans). The online chooser
  stays correct; its measured payoff is small because the pair population is thin-span by
  construction.
- **The v3 gate's flush 4.2% is superseded**: under the pre-shaped close the flush stratum
  holds 25% overall. Population AND procedure both changed between gates — the tool_bin
  before/after (0/6 → 15/105 on the same asset) is the clean evidence that the pre-shape
  itself is a large win.
- **retreated is definitively dead as a grasp family** (3% at n=628) — reachability statements.
- **Zero-hold roster, now evidence-backed**: pitcher **0/45 — its verdict STANDS** under the
  latest procedure; power_cable 0/34 and nylon_rope 0/28 (both cables), banana_soft 0/26,
  wood_block 0/2 (nothing fits the jaw). **steel_cube's old "capability limit" hypothesis is
  DEAD — it holds 41/112 (37%)**; raspberry_cube holds 36/36. The empirical generator gap is
  CABLES + the floppy banana_soft, matching the "span under load" gap already on record.
- Weakest survivors worth attention: mug 3/85, mustard 2/77, foam_brick 1/36.

### DESIGNED (2026-08-11): candidate statuses + LLM retry stage + cables OUT — framework first, user-directed

The user's directives, turned into a documented framework BEFORE implementation so parallel
agents can build against it. **The contract docs are the authority**:
[trajPipeline/llm-retry.md](trajPipeline/llm-retry.md) (new — the whole retry stage) and
[trajPipeline/grasp-library.md](trajPipeline/grasp-library.md) "Candidate statuses". Summary:

- **Statuses, derived at MERGE** (additive at schema v5 — a bump would orphan every sidecar
  incl. the 1719 shake trials, since sidecars read through the same strict version gate):
  `retreated` = **weak grasp option** (label `weak_grasp_option`, NOT a legitimate candidate, no
  physics testing ever again, excluded from grasp_select's default pool); `seat_blocked` =
  **discarded from the record** (sidecars keep the measurement; orphaned annotations dropped with
  a count); `clamped_deep` gets **`overhang` tracked in the record JSON, not scored**.
- **`llm_retry` pass + orchestrator** (`grasp_passes/llm_retry/`): for assets with ZERO held
  legitimate candidates after full shake coverage — INCLUDING assets whose only candidates (or
  only holds) are retreated, and empty out-of-reach records. Round A: 10 candidates from the
  merged record + 6 canonical renders (same LLM does the semantic annotation inline — design
  choice, one multimodal call already holds all context; obb_bucket stays mechanical). No
  seating algorithm: pose stored as given, `seat_mode: "llm"`, span/seat_depth MEASURED at the
  pose (`measure_span_at`). Round B (once): renders of every tried grasp + failure detail → ≤10
  new. Still nothing holds → record marked **unusable** (merge-derived, like out-of-reach).
- **shake_validate v4**: skips retreated at trial selection + INCREMENTAL (carries forward its
  own previous annotations for unchanged candidates — else 10 new LLM candidates re-run a whole
  asset's matrix). v3 sidecars stay valid via the new `merge.COMPATIBLE_VERSIONS` allowlist.
- **CABLES OUT of the whole pipeline** (like cloth/bags): `power_cable`/`nylon_rope` records +
  sidecars deleted, `kind: "cable"` moves to `UNSUPPORTED_KINDS`, `catalog._rod_vertices` + the
  cable branch removed, fixture's power_cable entries dropped (fixture v8). Basis already
  measured: 0/62 held; the probe read the rod's construction axis. Catalog coverage 24 → 22.

Implementation task list (parallel-safe split): (1) core contracts in `grasp_library.py` +
`merge.py` derivation layer — LANDED 2026-08-11 (statuses/trigger helpers, `measure_span_at`,
`overhang` round-trip, merge discard/stamp/derive + `COMPATIBLE_VERSIONS`); (2) cable removal
sweep; (3) shake_validate v4; (4) `llm_retry` package per llm-retry.md; (5) grasp_select
weak-pool filter + selftest expectations — (2)–(5) running as four parallel agents. Then:
re-run fixture, `merge --all` (rewrites every record JSON with the new statuses), selfchecks,
and RUN the llm_retry cycle over the trigger population.

**MEASURED (2026-08-11), swept from disk with the landed `needs_llm_retry`** — the trigger
population is **banana_soft, mug, pitcher, wood_block**: banana_soft 22 legit / 0 held;
**mug 8 legit / 0 held — its 3 catalog-shake holds all sat on RETREATED poses**, exactly the
only-weak-holds case the user called out, so it enters the population; pitcher 0 legit (45 weak
+ 40 blocked) — vacuous trigger; wood_block 1 legit / 0 held. Cable records excluded (load now
rejects kind "cable"; deletion in flight).

### LANDED (2026-08-11): the whole framework — all five tasks + regeneration, verified

All four parallel agents delivered; integration swept from disk. **The library reconciles
EXACTLY: 2117 − 398 seat_blocked (merge-discarded) − 62 cable = 1657 candidates over 22
records**; all 628 retreated stamped `weak_grasp_option`, 0 blocked remaining, all 832
clamped_deep carry `overhang`. Selfchecks: grasp_passes, grasp_library, grasp_select (70
checks), shake_validate, llm_retry selftest — all pass. Notables beyond the design: shake v4's
carry-forward re-annotates a covered asset in ~5 s with ZERO trials (tuna_can: 40 carried, 38
weak + 26 blocked skipped; forced re-run byte-identical); its `selfcheck.py` had a pre-existing
broken import (`_rot` gone from `hand.py`) — fixed with `mathutils.quat_to_matrix_xyzw`;
`grasp_select` drops `seat_blocked` even under `include_weak=True`; `llm_retry`'s round-B call
re-sends the full round-A context (a fresh call has no memory), renders `soft_mesh` via a
derived tet-boundary surface (measurement stays on `ctx.asset.faces`), and shows the LLM the
historical v3 measured outcomes on retreated poses rather than hiding them. The llm_retry cycle
over the trigger population launched 2026-08-11 (log: session scratchpad
`llm_retry_cycle.log`); results below when it lands.

### FIXED (2026-08-12): first live round was an INFRASTRUCTURE failure — transport + prompt, not the object

banana_soft's round A (prompt v1) was measured invalid as an attempt, and the cycle was STOPPED
before its round B or any other asset consumed a round: (1) the primary model died at
`_messages_request`'s hardcoded `max_tokens: 8000` with `stop_reason=max_tokens` and NO text
block — a reasoning model spends the budget THINKING before any text, so the call failed over to
the fallback; (2) the fallback's 10 candidates hovered 9/10 TCPs ON/ABOVE the surface (position
z=+35 mm on a body whose surface is z=+18 mm, approach −z) — "gripping air" authoring drops: it
read `position` as a hover point, not the final fingertip-tips pose. Fixes: `_messages_request`
gained a `max_tokens` parameter (default 8000 unchanged for scene-gen; llm_retry passes
`LLM_MAX_TOKENS = 20000`), and prompt v2 adds the worked TCP-past-the-surface example. The
PROMPT_VERSION bump re-arms banana_soft's round A by design — a transport-crippled round is not
a fair blind attempt, and the once-only bound applies to fair rounds. One shake trial (~1 min)
was discarded with the superseded round-A candidate. Cycle relaunched under prompt v2
(`llm_retry_cycle2.log`).

### MEASURED (2026-08-12): the llm_retry cycle CLEARED the trigger population — 3 of 4 now hold

Full cycle over banana_soft / mug / pitcher / wood_block under prompt v2, all verdicts
merge-derived and re-verified from disk (records reload clean; `needs_llm_retry` is now FALSE
catalog-wide; library 1715 candidates = 1657 + 58 llm over 22 records; selfchecks pass):

- **pitcher: HELD, round A** (a07/a08, both **handle** grasps, 35 mm width / 160–167 mm span,
  22–29 mm settle-then-hold) — the one evidence-backed hard case (0/45 across every generator)
  cleared on the LLM's FIRST blind round, and by exactly the handle-pinch experiment the Next
  queue had proposed. Its record now carries legitimate measured holds.
- **mug: HELD, round A** (a04/a06 **rim** pinches at 0.8/4.2 mm slip — among the cleanest holds
  in the library — plus a08 **handle**). The object that entered the population only because its
  3 old holds sat on retreated poses now holds legitimately.
- **banana_soft: HELD, round B** (b04/b05/b08/b09, body pinches 22 mm width / ~36 mm span,
  23–77 mm settle-then-hold) — all 10 blind-round candidates failed shake; the VISUAL-FEEDBACK
  round then produced 4 holds. First live proof of the round-B mechanism, and the first holds
  ever on this asset (obb_face-only coverage held 0/26).
- **wood_block: UNUSABLE** (merge-derived note; `is_unusable` true) — 18 valid LLM candidates
  over both rounds, physics covered everything, nothing holds. The honest designed outcome for
  89×89 mm against an 80 mm jaw.

Ops notes: fable answered 3 of 5 calls; on wood_block's image-heavy round B it hit even the
raised 20k `max_tokens` (all spent reasoning, no text) and the opus fallback carried the round
with 8/10 authoring-valid — the v2 worked example fixed the systematic hover error for BOTH
models (40/40 authoring-valid on rounds where fable answered, 8/10 on the fallback round).
If a future prompt grows further, raise `LLM_MAX_TOKENS` before trusting a fallback-only round.

### LANDED (2026-08-12): the TRAJECTORY STAGE itself (`deformableManipulationTools/traj_gen/`)

The stage between task gen and simulation exists and is documented —
[trajPipeline/trajectory-generation.md](trajPipeline/trajectory-generation.md) is the authority.
One command runs select → plan → rollout → LLM retries → mp4 on a pipeline run dir:
`.venv/bin/python -m deformableManipulationTools.traj_gen outputs/agenticPipeline/<run>`.

- **Online selection** = `grasp_select` at the settled placement (weak/retreated excluded — score
  0 by decree; seat_blocked discarded) + a PHYSICS-TIERED re-rank (measured held → untested →
  measured drop, additive penalties 0/0.35/0.90) + score-weighted sampling
  (`p ∝ exp(-cost/0.08)`, retries draw without replacement).
- **Plan** = fixed phase skeleton; Bezier transport legs between ELEVATED endpoints with
  collision-driven control-point insertion (inflated boxes; the approach/lift legs stay straight —
  they are the corridor the library validated). Force target derived (shake law at 3 m/s²);
  `GraspWindow.preshape_width` is NEW central support for the pre-shaped-approach contract (grip
  kernel holds the aperture pre-close and starts the close ramp from it on both solver paths;
  `None` = legacy fully-open, bit-exact).
- **Rollout** (subprocess, settle-harness pattern) measures held/carried/close-drift/goal
  (`success_spec` on the final state); **LLM loop** gets 2 corrected rollouts (switch candidate /
  adjust ≤20 mm + width + force) with scene still + matplotlib grasp snapshot; then the
  trajectory ABORTS honestly. `traj.json` (last executed plan) + `traj_result.json` (all
  attempts) land in the run dir; `build.demo_from_dir` plays `traj.json` when present (settle-only
  fallback otherwise, unchanged).
- Selftest: `python -m deformableManipulationTools.traj_gen.selftest` (32 checks, no GPU).

### VALIDATED (2026-08-12): trajectory stage end-to-end — 3 full scene→task→traj demos with video

Five scenes were generated for three demo slots; **3 succeeded, 2 failed on the same real library
gap**. Successes (each has `trajectory.mp4` + `traj_result.json` in its run dir under
`outputs/agenticPipeline/`): `demo_kitchen_fruit` (apple → bowl, attempt 0, place err 2.4 cm),
`demo_toy_tidy` (rubik's cube → tool_bin, attempt 0, place err 2.4 mm), `demo_pantry_tray`
(soup can → long_tray_bin, attempt 2 — two side grasps arm-blocked, the LLM switched to the
top-down measured-held candidate; the carry leg inserted a Bezier control point over the
sugar_box). The pre-existing `tool_sorting_station` run also succeeded (attempt 1,
LLM-recovered) as the smoke test.

Live rollouts bought four gates/fixes the offline pipeline could not see, all landed + documented
in [trajPipeline/trajectory-generation.md](trajPipeline/trajectory-generation.md):

- **The flat-can gap (the 2 failed scenes, both on `tuna_can`)**: EVERY stored candidate for a
  3.3 cm-tall can jaws across its HEIGHT — the lower finger must occupy the space between can
  bottom and tabletop, impossible for a resting object; the table-less shake rig and the corridor
  clearance are both blind to it. The stage now rejects these cheaply
  (`policy.pads_clear_table`); the remaining "medial" rim grasps measured 1–2.5 cm of lift then
  slip-out. **Flat cylinders need a chord-pinch generator** (new generator-gap entry).
- **Arm-feasibility is a path property**: the per-pose ladder solve accepted poses the executor's
  chained path IK then missed (different branch), so the gate runs `solve_gripper_ik_path` on the
  ASSEMBLED plan (gripped edge weights included) and FK-verifies every waypoint. Contact-blocked
  arms (side grasp near the table edge, 21 cm TCP jam) remain invisible to any IK gate — that is
  exactly what the rollout + LLM retry catch, and did.
- **Goal eval needs the container MESH** (`hull_points` at the settled pose) — an AABB has no
  cavity.
- **Duplicate-instance choice**: with two apples, the first-in-scene one sat inside the goal bowl
  already; `choose_target_index` picks an instance that does not already satisfy the goal.

### LANDED (2026-08-13): trajectory stage round 2 — scene reuse, multi-step, deformables online, visual verification, annotations (user-directed)

Seven user directives, all in [trajPipeline/trajectory-generation.md](trajPipeline/trajectory-generation.md)
(the authority) + [agenticPipeline/agentic-pipeline.md](agenticPipeline/agentic-pipeline.md):

1. **Generators stay grasp-blind** (verified: prompts carry only the catalog; the one
   grasp-adjacent import is the physical jaw-width ceiling). Consistent grasp-failure evidence
   from full-pipeline runs goes to **`assets/low_graspability.md`** (stage-written, object-bound
   evidence only, deduped; seeded with tuna_can's flat-can class — 2 runs, 16-17 pad-sweep
   rejections + 4 pure grasp failures).
2. **Trajectory videos now render in the scene_overview look** (`mp4_advanced` default in the
   traj CLI; `mp4` stays as a fast preview).
3. **Cloth/cable targets get their grasp ONLINE** (`deform_grasp.py` + `deform_snapshot.py`):
   settle-state snapshot (numbered world-frame material points + scatter PNG) → LLM proposes
   position/approach/jaw/width/force → same spline/collision/IK machinery + rollout →
   **3 proposals with measured feedback**, then honest abort. Cloth press allowance (8 mm) in the
   pad gate; hang-below-TCP from material extent; rollout tracks the material point nearest the
   grasp (cable = capsule bodies, cloth = particles). Bags stay out.
4. **Scene reuse**: `agent_pipeline.py --tasks N` (default 3, interview-asked) → N different
   tasks per scene (multi-goal avoid in `call_task_agent`), per-task demo files
   (`pipeline_<slug>__t<k>.py`), and the traj CLI iterates them all (`traj<k>.json` etc.).
5. **Post-trajectory VISUAL verification** (`verify.py`): a VLM judges the before/after stills +
   measured coordinates against the instruction; mismatch → the executed round is ARCHIVED and
   its annotation RELABELED with the achieved instruction set (never scrapped — training data for
   what it actually did), then a bounded place nudge re-executes, ≤2 rounds per demo.
6. **`annotations.json` per scene** (`annotate.py`): one row per executed trajectory
   (instructions incl. relabels, goal+subgoals, grasp, phase timeline, metrics, verification,
   artifact paths) — the VLA training-data record; labels always match the video.
7. **Multi-step tasks**: task gen may emit a `subgoals` chain (schema + prompt + per-subgoal
   feasibility, `subgoal_specs` compiled); `policy.plan_segments` chains one pick-place segment
   per subgoal (per-segment GraspWindows — the grip kernel iterates them; placed objects become
   obstacles for later segments); the rollout scores every segment (the first failing segment
   names the failure; the LLM retry targets exactly that segment); subgoals are geometrically
   checked at the final state. First live result: the can-sorting scene produced 3/3 feasible
   tasks including two genuine multi-step ones on the first attempt.

Selftest now 45 checks. NOTE: previous demo runs were archived to
`outputs/agenticPipeline_old/`; new runs land in a fresh `outputs/agenticPipeline/`.

**VALIDATED 2026-08-13 (three fully TIMED end-to-end runs; timing is now instrumented at every
level — pipeline.json `duration_s`, traj_result `duration_s` + per-rollout/render/verify, CLI
per-task + per-run totals):**

- `demo_cans_full` (pipeline 112 s + traj 1068 s): multi-step machinery works — the per-segment
  LLM retry fixed segment 0 mid-task (2 mm TCP hit after a switch); all 3 tasks then failed
  HONESTLY (tuna_can flat-can gap ×2; a stack target 0.30 m from the base — too close for the
  arm, caught by the gates in 79 s with zero rollouts). Bought three fixes: aborts now report
  `arm_rejected` reasons; the per-pose IK gate FK-resets to home (verdicts were order-dependent —
  same candidate 2 mm in one task, "85 mm" in another); approach_missed feedback now steers the
  LLM away from same-approach candidates.
- `demo_laundry_full` (pipeline 186 s; final task 363 s): the ONLINE CLOTH PATH works end-to-end
  — shirt retrieval grasped on the first proposal after the prompt gained the measured sheet
  recipe (each clause bought by failed rollouts: interior wad not edge pinch; z below the
  TABLETOP not the cloth surface; 4-5 N not 2-3 N; 2 s close + 3 s press-hold), advanced-look
  video + visual verification passed. Also bought: the crumple set-down footprint (0.4× flat
  extent) and OUTCOME-BASED SUCCESS (goal met at final state counts even if the sheet slipped
  late in the drag — every "dropped" attempt had satisfied `object_retrieved`).
- `demo_fruit_full` (pipeline 170 s + traj 2019 s): task 2 (apple→bowl) SUCCEEDED and was
  visually verified with a precise verdict; task 1 (multi-step "all fruits") executed one of
  three segments, the primary groups predicate OVER-PASSED (single-name schema), the VISUAL
  verifier caught it and the annotation was RELABELED to "Put the apple into the bowl" — the
  training-label consistency loop working as designed. Fix landed: the multi-step geometric goal
  now ANDs the primary with every evaluable subgoal. Task 3 aborted honestly in 6 s (banana =
  flat-object class + place point off the table edge).
- Ledger state: tuna_can (stands), banana (new — flat-object class), green_tshirt (superseded in
  place: recipe-sensitive, not low-graspability).

### Next

- **Flat-cylinder chord-pinch generator** (from the 2026-08-12 traj-stage validation above):
  tuna_can-class objects (33 mm tall) have NO executable table-top grasp in the library — every
  stored candidate jaws across the height, lower finger under the can; the table-less shake rig
  and corridor clearance are both blind to it (same blindness class as the span_flush
  table-command bug). The traj stage gates them online (`policy.pads_clear_table`, measured:
  survivors lift 1–2.5 cm then slip out); the OFFLINE fix is a generator producing chord pinches
  in the top-face plane. Candidate assets: tuna_can (+ any future flat cylinder).
- ~~**Selection can now rank on real quality everywhere**~~ (RESOLVED 2026-08-12: task 5
  updated the selftest expectations against measured records, and "down-weight retreated"
  became full default-pool exclusion via `weak_grasp_option`).
- ~~**Pitcher**: the one evidence-backed hard case left~~ (RESOLVED 2026-08-12: the llm_retry
  round A held 2 handle grasps — see the MEASURED entry above; the standalone handle-pinch
  generator is no longer needed for pitcher, though it remains a valid idea if other
  handled-vessel assets join the catalog).
- ~~**Cables** need a span-under-load generator~~ (SUPERSEDED 2026-08-11: cables are OUT of the
  pipeline's scope entirely — see the DESIGNED entry above; re-opening cables starts from the
  scope paragraph in [grasp-library.md](trajPipeline/grasp-library.md)).
- **Bigger centred control** (~30 banana trials) before trusting any absolute hold rate.
- ~~**Gate `clamped_deep` by overhang depth**~~ (PARTIALLY SUPERSEDED 2026-08-11: `overhang` is
  now tracked in the record JSON by the merge, deliberately NOT scored; gating stays a future,
  evidence-backed decision).
- ~~**`grasp_select`: down-rank `seat_blocked`**~~ (SUPERSEDED 2026-08-11: blocked candidates are
  discarded from records at merge; the selection change that remains is EXCLUDING
  `weak_grasp_option` from the default pool — task 5 of the DESIGNED entry).
- **Deformable-span generator gap**: sponge / foam_brick / banana_soft / raspberry_cube are
  covered only by `obb_face` — needs a span-under-load method.
- **The trajectory stage itself** (task + scene + record → `DemoSpec` waypoints). The IK-
  vocabulary question that used to block it is CLOSED — `grasp_select/projection.py` proves the
  `(yaw, tilt)` vocabulary covers all of SO(3) with only tilt magnitude limited
  ([grasp-selection.md](trajPipeline/grasp-selection.md)). The seam to replace is
  `agentic_pipeline/build.py:demo_from_dir`'s settle-only parked waypoint.

Adjacent, deliberately not started: the `*_through` aperture metadata and the bag mouth-rim loop
in [success-evaluators.md](agenticPipeline/success-evaluators.md) are the same shape of problem
(per-asset annotation, precomputed offline, looked up at runtime) and should share whatever store
this task establishes.
