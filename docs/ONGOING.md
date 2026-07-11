# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## No task in flight

Last completed (2026-07-10/11): contact-material ownership + the central band-limited harmonic
coupling, the realistic per-object re-authorship of EVERY object, retirement of the cloth friction
exception, the material-restore bug fix, and the legacy cleanup. All durable knowledge lives in its
docs: rules in CLAUDE.md "CONTACT-MATERIAL OWNERSHIP"; mechanics + register-gotcha + noise bands in
[solver-architecture.md](solver-architecture.md) ("Contact materials", "Verification standard");
the fold recipe + its measured history in [cloths.md](cloths.md); the raspberry-like block, cable
values, and the delicate-grasp findings in [deformables.md](deformables.md); per-demo notes in
[examples.md](examples.md).

## Standing open items (live in their docs)

- **No static friction in the rigid grasp (current config, by choice)** →
  [SOLVERS.md](SOLVERS.md) §6 trade study. Symptoms: the ~2 kg plate slowly pivots about the jaw
  axis while carried (`soft_compression_franka`, ~18°); the banana's edge grasp is intermittent at
  any force (its wedge converts squeeze into self-ejection in the same proportion at any target —
  measured at 10/30/45/80 N; only compliant fingertips or a flatter grasp point truly fix it).
  The measured option, `rigid_contact_hard=False, friction_epsilon=0.05`, is RESERVED for an
  explicit future re-opening of the slippage problem (WARNING: jerkier initial grasp; both knobs
  stay central in framework.py, never per-demo — §6).
- **Squeeze-signal under-read on tilted contact normals** → [SOLVERS.md](SOLVERS.md) §6 (last
  paragraph). The regulated signal projects each pad's force onto the jaw axis; on wedge faces it
  under-reads ~3× and the regulator over-squeezes. Physical fix: per-pad contact-NORMAL force as
  the signal — but the cable cage currently depends on the projected signal's behaviour, so the
  change needs a full-demo revalidation.
- Residual penalty-contact ring at impact/grasp moments (pre-existing) →
  [SOLVERS.md](SOLVERS.md) §5.
- **Two demos' narratives changed with the raspberry-like block** ([examples.md](examples.md)):
  `soft_compression` (the 2 kg plate now flattens the fruit) and `rigidCube_soft` (the steel cube
  squashes it and rolls off). Both are honest physics; if their *stories* should be preserved, add
  a second, firmer canonical soft object (params.py convention explicitly allows distinct named
  instances) rather than de-realizing the berry.
- **Harness-grade validation coverage**: `cloth_franka`, `cable_soft`, `soft_pickplace` are
  metric-verified at the final materials; the other five demos are render-verified only.
