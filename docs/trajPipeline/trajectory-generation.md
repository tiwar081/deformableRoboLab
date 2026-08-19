# Trajectory generation — the executable stage (`deformableManipulationTools/traj_gen/`)

The stage between task generation and simulation, landed 2026-08-12 (extended 2026-08-13 with
scene reuse, multi-step tasks, the online deformable grasp, visual verification, and training
annotations): a pipeline run (`scene.json` + its tasks) becomes EXECUTED, MEASURED, ANNOTATED
pick-and-place policies. It replaces the seam named in [README.md](README.md):
`agentic_pipeline/build.py:demo_from_dir` reads a `traj<k>.json` beside the run's demo files and
plays the generated policy through the standard runner/renderer (no traj file → the old
settle-only parked arm, which the settle check and still renders still use).

```bash
.venv/bin/python -m deformableManipulationTools.traj_gen outputs/agenticPipeline/<run> \
    [--task-file task_2.json] [--device cuda:0] [--seed N] [--temperature 0.08] \
    [--llm-attempts 2] [--no-render | --render-anyway] [--no-visual-verify] \
    [--output-style mp4_advanced]
python -m deformableManipulationTools.traj_gen.selftest      # no-GPU invariants (45 checks)
```

**Scene reuse (one scene, several tasks).** `agent_pipeline.py --tasks N` (default 3; asked in the
`--user` interview) makes the task agent produce N DIFFERENT tasks for the one scene
(`task.json`, `task_2.json`, … — each later task avoids all earlier goals and inherits the first
task's robot placement), each with its own runnable demo file (`pipeline_<slug>__t<k>.py`). The
traj CLI runs EVERY task of a run dir by default, producing `traj<k>.json` /
`traj_result<k>.json` / `trajectory<k>.mp4` per task.

**Multi-step tasks.** Task gen may fill the optional `subgoals` field — an ordered chain of 2–4
single-object goals decomposing a broad instruction ("put all the cans in the bin"); each subgoal
is feasibility-checked like a goal and compiled into `subgoal_specs`. The planner
(`policy.plan_segments`) chains one pick-place SEGMENT per subgoal into a single trajectory: one
`GraspWindow` per segment (the grip kernel iterates windows), per-segment place resolution, and
objects placed by earlier segments becoming obstacle boxes for later ones. The rollout scores
every segment (the FIRST failing segment names the failure, and the retry loop targets exactly
that segment's grasp); every subgoal is also geometrically checked at the final state.

Artifacts per run dir: `traj<k>.json` (the LAST executed plan — re-rendering reproduces exactly
what was measured; superseded visual-tuning rounds archived as `traj<k>.r<n>.json`),
`traj_result<k>.json` (every attempt's plan + rollout metrics + LLM verdicts),
`grasp_attempt*<n>.png` (failure snapshots shown to the LLM), `trajectory<k>.mp4` (rendered in
the SAME RoboLab look as `scene_overview.png` — `mp4_advanced` is the default; earlier rounds
kept as `trajectory<k>.r<n>.mp4`), and `annotations.json` (below).

## Visual outcome verification + honest relabeling (`verify.py`)

After the video render, a VLM compares the BEFORE/AFTER stills (the after frame is the advanced
render's final still) plus the simulator's measured final coordinates against the task
instruction — catching what the geometric check mislabels (wrong role in a generated predicate,
unevaluable goals, rim-teetering). On a mismatch the executed trajectory is NEVER scrapped: its
plan + video are archived, its annotation row is RELABELED with the ACHIEVED instruction set
(the original preserved under `intended_instructions` — a mis-executed demo is valid training
data for what it actually did), and — for single-step tasks — the verdict's bounded world-frame
nudge (≤ 8 cm/axis) shifts the place column and the demo re-executes, up to 2 rounds per demo.
Multi-step mismatches keep the relabeled row (per-segment tuning is future work).

## Training annotations (`annotate.py` — `annotations.json` per scene)

One row per EXECUTED trajectory across all tasks and verification rounds: instruction set
(vague/default/specific — relabeled when the visual check demanded it), goal + subgoals,
subtasks/attributes/difficulty, the executed grasp(s), the phase timeline (temporal segmentation
for VLA training), per-segment rollout metrics, geometric + visual verification results, and the
traj/video paths. The consistency rule: a row's instructions ALWAYS describe the outcome its
video shows.

## The low-graspability ledger (`assets/low_graspability.md`)

The scene/task generators see NO grasp-candidate information (verified: their prompts carry only
the object catalog; the only grasp-adjacent import is the physical jaw-width ceiling) — grasp
difficulty must never bias generation. Instead, when a full-pipeline run produces OBJECT-BOUND,
placement-independent evidence that an object's grasp will keep failing (a library
unusable/out-of-reach verdict; every candidate's jaw sweeping below the tabletop; all rollouts
failing at the grasp itself), the stage appends the evidence to `assets/low_graspability.md` —
one section per object, deduped per run. Seeded 2026-08-12 with `tuna_can` (the flat-can class).

## Online deformable grasps (`deform_grasp.py` + `deform_snapshot.py`)

Cloth sheets and cables have no offline records (a settled sheet/coil is nothing like a rest
shape), so their targets skip STRAIGHT to a multimodal LLM: a settle-state SNAPSHOT (subprocess
re-settle → numbered world-frame material points + a top/side scatter PNG, image and text
grounding each other) plus the scene still and the task; the LLM proposes position / approach /
jaw axis / width / force (with the measured physics facts in the prompt: sheet pinches press a
few mm into the support at 2–4 N; cables are gripped across their local direction at 15–30 N).
The proposal becomes an ordinary `PickSpec`; the SAME spline/collision/IK machinery builds the
trajectory (the pad-sweep gate relaxed by the sheet-press allowance; the hang-below-TCP depth
estimated from the material extent; the set-down footprint at 0.4× the flat extent — a dropped
sheet crumples) and the same rollout measures it — tracking the material point nearest the grasp
(a cable is capsule BODIES; a cloth is particles). **Up to 3 proposals per trajectory with
measured feedback in between**, then an honest abort. Bags stay out of scope.

The prompt carries the MEASURED sheet recipe, each clause bought by failed rollouts
(2026-08-13, the shirt-retrieval validation): grasp 3–6 cm INSIDE the fabric so both pads press
on cloth and gather a wad (five edge pinches in a row captured nothing — the validated demo
grasps 4 cm inside the torso); command z 4–6 mm below the TABLETOP (pinches at the cloth surface
close on air); force 4–5 N (2–3 N under-engages a wadded knit); and the planner gives cloth
segments the demo's timing (2 s close + 3 s press-hold — `CLOTH_CLOSE_DUR` /
`CLOTH_POST_CLOSE_DWELL`).

**Success is outcome-based**: a trajectory succeeds when the transport metrics pass OR the task's
own goal predicate is MET at the final state — a sheet that slips late in a drag but ends where
the task asked is a successful demo (measured: every "dropped_in_transit" shirt attempt satisfied
`object_retrieved`); the visual verification then confirms it as a human would.

## Online grasp selection (`selection.py`) — offline evidence, re-ranked at the placement

`grasp_select` runs unchanged at the object's SETTLED placement (settled x/y/yaw from the scene
write-back, body z from the pipeline's settle report): pool assembly per the candidate-status
taxonomy — **`retreated` candidates are excluded outright (score 0 by decree, never sampled) and
`seat_blocked` candidates are discarded** — then face-bucket pruning, reach, the straight approach
corridor against table + scene obstacle boxes, projection into the executor's `(yaw, tilt)`
vocabulary, the four-term score, and the depth-variant resolution. On top of that result the stage
adds what the full-catalog shake made possible:

- **Physics-tiered re-rank** — the primary sort key is the measured shake outcome: tier 0 =
  measured HELD, tier 1 = never tested (neutral, not penalized as bad), tier 2 = measured DROP.
  Within a tier the `grasp_select` score orders. Implemented as an additive sampling cost
  (`TIER_PENALTY` = 0 / 0.35 / 0.90 on top of the ~[0,1] score), so a spectacular untested
  candidate can still edge past a mediocre held one, but held evidence dominates.
- **Score-weighted random sampling** — the pick is drawn `p ∝ exp(-cost/T)` (default `T = 0.08`;
  `T = 0` is the deterministic argmin, pinned by the selftest). Retries draw WITHOUT replacement
  (a tried id is never re-drawn).
- **Pad-sweep-vs-table gate** (`policy.pads_clear_table`) — a candidate whose PRE-SHAPED jaw
  sweep dips below the tabletop cannot close on a resting object whatever the arm does (a flat
  3.3 cm tuna can's whole library jaws across its HEIGHT — the lower finger would occupy the
  space under the can; the table-less shake rig and the corridor check are both blind to this;
  measured cost before the gate: three arm-jammed rollouts).
- **Arm-feasibility gate** (`reach.py`) — two levels, both FK-verified. A cheap per-pose ladder
  solve pre-filters each drawn candidate's grasp + pre-grasp; then the ASSEMBLED plan's whole
  waypoint list is solved with the EXECUTOR'S OWN path IK (`solve_gripper_ik_path`, same gripped
  edge weights) and every waypoint's TCP error measured — > 15 mm on the grasp-to-release stretch
  (40 mm on transit knots) rejects the candidate and the stage draws the next one. Both levels
  are measured needs (2026-08-12, first live rollouts): a side grasp 0.30 m from the base passed
  the radius test and projection but the executor missed it by 21 cm, and the per-pose ladder
  itself accepted a pose the chained path solve then missed (different branch); a place pose at
  0.65 m horizontal with a top-down wrist missed by 12 cm — candidate-dependent, hence rejection,
  not abort.

## The plan (`policy.py` + `curve.py`) — Bezier legs, collision-driven control points, no LLM

Fixed phase skeleton, times derived from arc length at demo-calibrated speeds:

    settle (1.5 s) -> cruise -> descend -> close -> lift -> carry -> place -> release -> park

- **Transport legs are single Bezier segments** between ELEVATED endpoints (each endpoint's column
  raised to its clear height). The sampled spline (~1 cm spacing) is tested against a
  `CollisionField` — tabletop plane + scene obstacle boxes, inflated by the hand's half-profile
  (cruise) or the held object's half-extent + its hang-below-TCP depth (carry) — and every
  colliding stretch inserts ONE control point above its deepest sample, with per-iteration
  overshoot (a Bezier bends toward, not through, its control points). Iterate to clear;
  an unroutable leg ABORTS the plan (never executes a colliding path).
- **The approach and lift legs never bend**: straight runs along the candidate's approach axis —
  exactly the corridor `grasp_select`'s clearance validated (bending them would execute a
  procedure the library never measured). The vertical connectors above pre-grasp/standoff are
  deliberately outside the routed field: the inflation is conservative box-swelling, and the low
  final segment is validated against the TRUE boxes by the corridor check.
- **The pre-shape contract is honored centrally**: `GraspWindow.preshape_width` (new, derived —
  `min(width + PREGRASP_MARGIN, 80 mm)`) makes the grip kernel hold the pre-shaped aperture
  through the whole approach and start the close ramp from it, on BOTH solver paths. Hand-written
  demos (`preshape_width=None`) keep the legacy fully-open behavior bit-exactly.
- **The force target is derived, not tuned** — the shake pass's Coulomb law at transport
  accelerations: `F = 2·m·(g + 3 m/s²)/(2·µ_eff)`, `µ_eff = sqrt(µ_obj·µ_pad)`, clamped to the
  same [1, 40] N envelope. The LLM retry may override it (clamped).
- **Goal placement from the predicate** (13 supported): put-in → above the container mouth
  (release 3 cm above, gravity drops it in); on-top/stacked → above the support top (release just
  above contact); robot-POV direction words → beside the reference along
  `geometry.direction_vectors`; push/pull-to → adjacent; retrieved → in front of the base;
  cleared-from/outside-of → away from the reference. A free-spot spiral search keeps the set-down
  out of other objects' footprints, on the table, in reach. Deformable-SHAPE goals (fold/coil/...)
  are not executable here and abort the stage honestly.

## Rollout measurement (`rollout.py`)

A headless subprocess (settle-harness pattern) runs the demo and tracks the target body through
the plan's phases: `close_drift` (push-out detector), `lift_rise` → **held** (> 4 cm through the
lift), **carried** (over the place point at carry end, still elevated), final placement error, and
the task's `success_spec` evaluated on the final `SceneState` (`evaluable: false` = no score,
never a failure). One-word failure classification for the retry loop: `push_out` / `never_held` /
`dropped_in_transit` / `misplaced` / `diverged`.

## The LLM recovery loop (`llm.py`) — 2 attempts, then abort

Failed grasps are the primary failure mode (the offline shake holds 24.5%), so a failed rollout
consults a multimodal LLM with a deliberately LIGHT package: the run's existing
`scene_overview.png`, a matplotlib grasp-attempt snapshot (object mesh + pad chords + approach
arrow + the object's measured path + place point — no ray tracing), the plan's phases/routing, the
rollout's measurements, the attempted candidate's stored facts, and the top-8 untried alternatives
from the re-rank. It answers with ONE structured action: **switch** (a named alternative) or
**adjust** (grasp-frame offset ≤ 20 mm/axis, width, force target — clamped). Two corrected
rollouts per trajectory (`MAX_LLM_ATTEMPTS = 2`); a third failure ABORTS the trajectory —
`traj_result.json` says so, and the failure stays visible (no scripted rescue). Transport is
`scene_generator._messages_request` (raw HTTPS + OAuth, structured output, model fallback),
imported lazily exactly like `vlm_regions`.

## Boundaries

- Targets: rigid + soft-FEM catalog objects with grasp records; cloth/bags/cables and record-less
  or `unusable`/out-of-reach objects abort at the gate (same scope as the grasp library).
- The stage never touches grasp records or sidecars; it is a pure CONSUMER of the library.
- The demo file stays a data file: `traj.json` is data, `demo_from_dir` is the one assembly point.
