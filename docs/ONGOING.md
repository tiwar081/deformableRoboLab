# ONGOING

In-flight work. Durable facts are promoted to CLAUDE.md; this file holds what is
currently unresolved or recently changed.

## RESOLVED (this work): physical force-limited two-way cable grip

The cable grip in `cable_soft_franka` now uses NVIDIA's **dynamic finite-mass proxy**
two-way coupling with a **physical, bounded grip force (~30–95 N)** and **no force cap** —
the original goal. `test_final` passes (grasp → lift → sweep), CUDA-graph capture works.

### What was actually wrong (two compounding causes, not one)

The "gripper presses arbitrarily hard" artifact (raw 4e5–2.6e6 N) had **two** independent
causes; both had to be fixed:

1. **`rigid_avbd_contact_alpha=0.0` + `rigid_contact_history=True`** — the augmented-Lagrangian
   multiplier accumulates across steps against the position-held pads → 1e4–1e6 N. Removed
   (now NVIDIA's plain default-hard contacts: `alpha=0.95`, no cross-step history).
2. **Overdamped contact damping `kd`** — the Newton "absolute VBD damping" change reinterpreted
   `kd` as absolute [N·s/m]; the carried-over values (cable `20·ke=4e5`, proxy `1e2·ke=5e6`) are
   ~1e4× the proxy↔cable critical damping. Once cause (1) was removed, this **velocity-proportional
   damping force dominated** (~4e4 N during the lift) and is what actually diverged the dynamic
   proxy. Re-derived to a physical `kd≈1e2` (`Example.contact_kd`).

With both fixed: grip force is a physical ~30–95 N, the dynamic proxy is **stable** through the
whole grasp/lift/sweep.

### Correcting the prior log's conclusions (do not trust the old narrative)

- **"Free-body dynamic proxies NaN structurally in this Newton build" was a MISDIAGNOSIS.** The
  divergence was the overdamped `kd` (cause 2) feeding the runaway force into the finite-mass body,
  *not* a fundamental fragility of dynamic proxies. With physical `kd` the dynamic proxy works.
- The old "M2 blocked at t≈4.35 s" / "soften proxy↔cable ke" / "effort-cap as stabilizer" framing is
  obsolete — none of it was the real fix.

### Final design (single path; `cable_soft_franka.py` + `cable_coupling.py`)

- **DYNAMIC finite-mass proxies** (`proxy_effective_mass=10 kg`) mirroring the fingers, re-pinned
  each substep with the momentum-consistent gravity+lagged-wrench velocity undo (`_sync_proxy_state_kernel`).
- **Contact**: NVIDIA default `SolverVBD` (drop `alpha=0`/history) + **physical `kd≈1e2`** on the
  proxy & cable shapes.
- **Two-way feedback = NET-TO-EE** (`_apply_coupling_to_ee_kernel`): the summed pad wrench cancels
  the internal squeeze and feeds the real external cable load (weight + sweep reaction) to the arm,
  so the arm feels the cable while the position-held fingers keep their grip. **Per-finger feedback
  was tried and destroys the grasp** (pushes the pads open → grip lost) — this is the documented
  "no continuous feedback into the gripper DOF" invariant; confirmed empirically.
- **No force cap, no effort cap.** Grip force = position-controlled squeeze (interference) against
  bounded contact; tune via `grasp_interference` / `finger_effort` / contact `ke` if a different
  force is wanted. Per-pad tactile force is read via `TwoWayProxyCoupling.raw_force_norms()`.
- Single faithful path — the old `force_limited_grip` toggle, kinematic pose-mirror, EMA/clamp
  filter, and per-finger feedback kernel are all removed.

### Verification

`python -m examples cable_soft_franka --viewer null --device cuda:0 --num-frames 720 --test`
passes (capture on). `CABLE_DIAG=1` prints a per-frame `grip=(left,right)N cableZ=[…]` health line;
`CABLE_NO_CAPTURE=1` runs the uncaptured substep loop (clearer NaN reporting than the captured
segfault).

## Still open / next

- **`cable_rigidCube_franka.py`** shares `cable_coupling.py` and must be migrated to the new
  `TwoWayProxyCoupling` signature + dynamic proxy + physical `kd` + default contacts (currently
  out of date with the coupling module).
- **Generalize** the bounded-physical-grip approach to the rigid/soft examples (`grip_force.py`
  `GripForceClamp`/`SoftGripWidth` were post-hoc clamps for the OLD runaway; with physical contact
  damping they may be replaceable by the same honest-contact approach). The new upstream
  `example_softbody_franka.py` + `SolverVBD(integrate_with_external_rigid_solver=True)` is the
  reference for robot↔soft-particle grip (no proxy needed there).
- **Re-derive remaining damping** for other examples if they carry the old absolute-damping `kd`
  values (same class of bug as cause 2 above).
- Sub-docs (`gripper.md`, `deformables.md`, `solver-architecture.md`) still describe the old
  kinematic-proxy / force-cap / alpha=0 design in places — update as touched.
