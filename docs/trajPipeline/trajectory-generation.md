# Trajectory generation — the executable stage (`deformableManipulationTools/traj_gen/`)

The stage between task generation and simulation, landed 2026-08-12: a pipeline run
(`scene.json` + `task.json`) becomes an EXECUTED, MEASURED pick-and-place policy. It replaces the
seam named in [README.md](README.md): `agentic_pipeline/build.py:demo_from_dir` now reads a
`traj.json` beside the run's demo file and plays the generated policy through the standard
runner/renderer (no `traj.json` → the old settle-only parked arm, which the settle check and still
renders still use).

```bash
.venv/bin/python -m deformableManipulationTools.traj_gen outputs/agenticPipeline/<run> \
    [--device cuda:0] [--seed N] [--temperature 0.08] [--llm-attempts 2] \
    [--no-render | --render-anyway] [--output-style mp4]
python -m deformableManipulationTools.traj_gen.selftest      # no-GPU invariants (30 checks)
```

Artifacts per run dir: `traj.json` (the LAST executed plan — re-rendering reproduces exactly what
was measured), `traj_result.json` (every attempt's plan + rollout metrics + LLM verdicts),
`grasp_attempt_<n>.png` (the failure snapshots shown to the LLM), `trajectory.mp4` (the final
render, copied from `outputs/<robot>/pipeline_<name>/`).

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
