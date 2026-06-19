# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just changed
and not yet settled, and any working hypotheses. Keep it lean — when something is proven and
durable, promote it to CLAUDE.md (or the relevant `docs/` file) and delete it here. Reset this file
at the start of each new big task.

## Current state

**In flight (2026-06-19): rubik's-cube asymmetric stick at release in `pickplace_ycb_franka`.**
The cube clings to one gripper pad ~0.15 s longer than the other before dropping over the bowl.

Diagnosis (confirmed empirically with the new `--log-grip` diagnostic, per-pad `[left,right]` N):
- The grip is fully symmetric in config/control; the asymmetry is emergent in the VBD solve. VBD is
  Gauss-Seidel (order-dependent across color groups), so the cube settles a hair off-center → a
  deterministic per-pad penetration/force imbalance. At release the less-penetrated pad loses normal
  force first; the more-penetrated pad holds the light cube (~0.2 kg, ~2 N) by friction (μ≈1).
- **The dominant driver is grip OVER-FIRMNESS, not the geometric residual.** The cube grip is grossly
  over-firm: ~155 N at 12 iters. The friction reserve μ·N ≫ the cube weight, so the last-contacting
  pad clings well past where a realistic grip would drop it.

**Rejected lever — raising VBD iterations makes it WORSE (verified):** AVBD hard contacts accumulate
the augmented-Lagrangian multiplier λ once per iteration (cold-started each step, history off), so
more iterations converge the grip to a HIGHER force: rubik's cube ~155 N @12 → ~350 N @24. The
release cling lengthened from ~0.15 s (free by t≈6.70) to >0.5 s (still in contact past t=7.0). So
155 N was an under-converged underestimate; the true converged grip on the cube is ~350 N, and
convergence aggravates the over-firmness. Also rejected: `rigid_contact_history` (inflates steady λ
~10–20× at alpha=0.95), low `rigid_avbd_contact_alpha` (ejection risk on the light cube).

Also note: the proxy grip force is **decoupled from the 20 N `finger_effort`** — the proxy is a
rigidly position-slaved body, so its penetration force (`ke·pen + λ` over the flat-face multi-point
patch) balloons to hundreds of N regardless of the actuator limit. This is why the cube reads 350 N.

Done so far (infra only, behavior-preserving):
- Consolidated the per-demo `--vbd-iterations` (all were 12) into ONE shared base-parser
  `--solver-iterations` (default 12) in `examples/__init__.py`; framework reads `args.solver_iterations`.
- Added `--log-grip` (default OFF, read-only) — per-frame per-pad grip force print in `examples.run()`.

**OPEN — needs user decision (the iteration fix failed as predicted):** the fix must reduce the grip
over-firmness toward a physical force. Candidate centralized levers under discussion: (A) per-object
pad clearance for the flat-faced cube (raise its `gripper_closed` → lower bite → ~10–30 N); (B) lower
global `grasp_interference`/`proxy_ke` (risks the already-marginal banana grip); (C) make the proxy
grip force physically bounded/compliant so it can't balloon with patch size or iterations
(architectural, most faithful, affects every demo). No grip change made yet — awaiting direction.

(Object set unified across all examples on 2026-06-19 — one `SOFT_BLOCK`, one 1 kg `RIGID_CUBE`,
the centralized `PLATE`, distinct YCB objects; auto material-restore in the framework. Durable
notes live in CLAUDE.md + docs/deformables.md + docs/examples.md. All 6 demos pass `--test` and
re-rendered to outputs/.)
