# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## No task in flight

Latest (2026-07-09/10): the arm's sensitivity to the harvested EE wrench was audited and fixed
centrally — `RobotConfig.arm_target_ke/kd` 420/42 → **400/80, the EXACT RoboLab/DROID Franka
actuator values** (RoboLab is the sim2real reference, so the arm must feel loads the same way;
the old gains had matching stiffness at HALF the damping → under-damped transients = the visible
jolt on grasp/load). Measured on the 2 kg plate carry: hand rattle ↓~3× (speed swing 0.058→0.021),
held-object rattle ↓~7×, grip force now HOLDS its 30 N target through the carry (was decaying to
9.4 N), pivot 24.5°→17.9°. The `body_f`→`xfrc` wrench path itself was verified correct; Newton's own
Franka demos sidestep the issue by driving the arm kinematically (infinitely stiff — ours is
softer than Newton's by construction, accepted); a real stiff position-controlled Franka is
~5–10× stiffer still (a 20 N payload sags our/RoboLab's TCP ~1–2.5 cm — accepted for RoboLab
consistency). High-payload demos re-rendered at the final gains. Context: this investigation was
prompted by the reverted static-friction trade study (below) — the stiction jolt rode the same
EE-feedback path.

## Standing open items (live in their docs)

- **No static friction in the rigid grasp (current config, by choice)** →
  [SOLVERS.md](SOLVERS.md) §6 trade study. Symptoms: the ~2 kg plate slowly pivots about the jaw
  axis while carried (`soft_compression_franka`; ~18° after the 2026-07-10 arm-damping fix,
  was 24.5°); the banana's edge grasp is intermittent at any force (its wedge converts squeeze
  into self-ejection in the same proportion at any target — measured at 10/30/45/80 N; only
  compliant fingertips or a flatter grasp point truly fix it).
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
