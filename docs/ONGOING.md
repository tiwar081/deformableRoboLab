# ONGOING

In-flight work. Durable facts are promoted to CLAUDE.md; this file holds what is
currently unresolved.

## Changes implemented so far (this work)

**New modules**

- `examples/grip_force.py` — centralized **force-limited gripper for RIGID + SOFT
  objects** (separate from the cable). Contact-driven (no baked object dimensions),
  grasp-window-gated, **0→15 N linear ramp over 0.5 s, capped at 15 N**, public Newton
  API only. Components:
  - `GripForceClamp` (rigid): on first *detected* contact, applies an **explicit
    two-point capped-Coulomb clamp** to the grasped body's `body_f` — normal ramps
    0→15 N + friction hold; the proxy penalty squeeze is relieved (pads ease out) so it
    cannot over-squeeze/eject.
  - `RigidGripWidth`: contact-driven width control — creep to first contact, then ease
    out to relax the stiff penalty.
  - `SoftGripWidth`: squeeze-to-force servo — close until the *measured* per-pad soft
    reaction reaches the ramped 0→15 N setpoint.
  - `GraspTarget` dataclass.
- `examples/cable_coupling.py` — **two-way MuJoCo↔VBD finger-proxy coupling for the
  cable** (`TwoWayProxyCoupling`): M1 kinematic mode + M2 dynamic-proxy mode (WIP),
  harvest / momentum-consistent-undo / feedback kernels, and raw-vs-clamped diagnostics.
  Details in the three sections below.

**Modified examples — rigid/soft force-limit (via `grip_force.py`)**

- `pickplace_ycb_franka.py` (rigid cube + banana) — `RigidGripWidth` + `GripForceClamp`.
  Verified: grip force ramps 0→15 N, no over-squeeze/ejection (carry |v|≈0.3 vs 43 m/s),
  orientation drift ≤5°. (`test_final` is blocked only by a **pre-existing** passive
  resting-object ejection — out of scope per the user.)
- `rigidCube_soft_franka.py` (rigid ~1 kg cube) — `RigidGripWidth` + `GripForceClamp`;
  the original "cube slips/ejects" is fixed (carry |v|≈0.37 vs 43 m/s); `test_final` PASS.
- `soft_compression_franka.py` (rigid ~1.95 kg sheet) — `RigidGripWidth` + `GripForceClamp`;
  `test_final` PASS.
- `soft_pickplace_franka.py` (soft block) — `SoftGripWidth` (squeeze-to-15 N); lifts &
  places, `test_final` PASS. Caveat: the small block is physically too soft to hold a
  *steady* 15 N (force chatters near full compression) — a lower target gives a cleaner hold.

**Modified examples — cable two-way coupling (via `cable_coupling.py`)**

- `cable_soft_franka.py` — M1 kinematic two-way coupling (**default**, verified: arm
  feels the cable, grasp/lift/sweep, `test_final` PASS) **+** the M2 force-limited
  dynamic-proxy WIP behind the `force_limited_grip` toggle (dynamic proxy, per-pad
  finger feedback, gripper effort cap, proxy↔table collision filter, softened
  proxy↔cable `ke`). M2 currently **OFF** (blocked — see §3).
- `cable_rigidCube_franka.py` — M1 kinematic two-way coupling only (verified: cable
  lifted Z 0.08→0.35, swept, `test_final` PASS).

**Docs**

- `CLAUDE.md` — added the **"Newton version (environment gotcha)"** section (the
  `_external/newton` drift to absolute VBD damping).
- `ONGOING.md` — cleared the prior-session log; this file.

## Focus: force-limited gripper↔object contact for the CABLE

Goal: a physically faithful, **force-limited (~15 N)** cable grasp. The cable is a
VBD rod (`add_rod`, `wrap_in_articulation=True`); the robot is MuJoCo. The grip force
the cable *actually feels* is presently uncontrolled — the raw VBD penalty/ALM pinch,
measured at **1e4–1e6 N** (most of it the internal squeeze that cancels as a net, so
the cable doesn't fly, but the magnitude is unphysical). We are trying to adopt
NVIDIA's two-way dynamic-proxy coupling so the grip can be limited at the **gripper
actuator** (the cause), not clamped after the fact (the effect).

Reference article: developer.nvidia.com "Newton adds contact-rich manipulation…".

NOTE — the **rigid/soft** examples already have a working, separate force-limit in
`examples/grip_force.py` (explicit capped-Coulomb clamp for rigid; squeeze-to-force
for soft; 0→15 N ramp). This document is about the **cable**, which that approach
destabilized, motivating the two-way coupling below.

---

### 1. NVIDIA's gripper↔object contact implementation (the target)

Two independent worlds, coupled with a one-step lag:

- **Universe A — MuJoCo rigid robot.** `SolverMuJoCo(solver="newton",
  integrator="implicitfast", cone="elliptic", iterations=20, ls_iterations=10,
  ls_parallel=True, impratio=1000.0)`.
- **Universe B — VBD cable.** `add_rod(radius=0.003, stretch_stiffness=1e12 (EA),
  bend_stiffness=3.0 (EI), stretch_damping=1e-3, bend_damping=1.0)`; `SolverVBD(iterations=10)`.
- **Proxy bodies — robot links mirrored into the VBD model as DYNAMIC finite-mass
  bodies**: `cable_builder.add_body(xform=robot.body_q[bid], mass=effective_mass[bid])`
  where the effective mass "reflects the inertia of the full articulated chain…
  optionally scaled for coupling stability." (i.e. NOT kinematic, NOT a light guess.)
- **Staggered coupled step (one-step lag):**
  1. apply the *previous* step's harvested proxy wrench onto `robot_state.body_f`;
  2. `collide` + `mj_solver.step` the robot; swap;
  3. `sync_proxy_state` kernel: copy robot **pose AND velocity** into each proxy, then
     **subtract the velocity change that gravity + the lagged coupling force will
     impart over dt** (`Δv = dt·inv_m·f_lin + dt·g`, `Δw = dt·R·inv_I·R⁻¹·f_tor`).
     This momentum-consistent "undo" keeps the dynamic proxy slaved to the robot while
     it still participates as a finite-mass contact body in VBD;
  4. `collide` + `vbd_solver.step` the cable (+ cable↔proxy contacts);
  5. **harvest the proxy contact wrenches** (applied at the next step). Swap.

Net: genuine two-way coupling — the arm feels the cable, and the grasp force emerges
from contact + the (force/effort-limited) gripper actuator. Stability is *structural*
(reflected-inertia mass + stiff solver + momentum-consistent undo), not a force cap.

---

### 2. Our current implementation (`examples/cable_coupling.py`)

Same split: `SolverMuJoCo` robot + `SolverVBD` object model (cable + finger proxies +
visible table + soft block). Proxies = the two finger pads (copied finger collision
shapes), `has_particle_collision=False`.

- **M1 — kinematic two-way (DEFAULT, verified, `force_limited_grip=False`):**
  - Proxies are **KINEMATIC** (zero-mass), slaved to the finger poses each substep
    (pure pose mirror — `_sync_proxy_pose_kernel`).
  - Harvest the proxy↔cable contact reaction via PUBLIC
    `SolverVBD.collect_rigid_contact_forces` (the penalty/ALM force `k·C+λ` is
    **mass-independent**, so it is harvestable even from a kinematic body), then feed
    the **NET of both pads** (the squeeze cancels, leaving the real cable load) onto
    the **EE/arm body** (`_apply_coupling_to_ee_kernel`), with EMA filter + clamp +
    gain as *safety bounds*.
  - Result (both `cable_soft_franka` and `cable_rigidCube_franka`): arm feels the
    cable; grasp/lift/sweep stable; `test_final` PASS; CUDA-graph capture OK. **Grip
    force itself UNCONTROLLED** (raw 1e4–1e6 N). This is the stable fallback.
  - `cable_rigidCube_franka` is **M1 only** (no M2 toggle wired).

- **M2 — force-limited dynamic proxy (WIP, currently OFF, only in `cable_soft_franka`):**
  single switch `self.force_limited_grip` (off → clean M1 revert) + independently-tunable
  `self.cable_grip_force_limit = 15.0`. When ON it assembles NVIDIA's recipe:
  - DYNAMIC finite-mass proxies (`proxy_effective_mass=10 kg`, `inertia=0.1`),
    `_sync_proxy_state_kernel` with the momentum-consistent **raw** undo (uses the raw,
    un-clamped lagged force — the clamp is only for the arm-feedback safety bound);
  - per-pad reaction routed to the **finger DOFs** (`_apply_coupling_to_fingers_kernel`)
    so the effort-limited gripper feels the cable and can back off;
  - gripper prismatic **`effort_limit` capped at `cable_grip_force_limit`** — this is
    meant to be *both* the stabilizer (gripper backs off → penetration/λ resolve) and
    the force limit (cable feels ~the cap);
  - stiffer robot (`impratio=1000`, `iterations=20`; `ls_parallel` deprecated in this
    Newton build, omitted);
  - **fallback step 1**: filter proxy↔table collision (`add_shape_collision_filter_pair`);
  - **fallback step 2**: soften the proxy↔cable pair (`cable_grip_contact_ke=5e3` on
    both proxy and cable shapes so `η<1`).

Diagnostics live in `cable_coupling.py`: `raw_force_norms()` (actual cable-felt force,
un-clamped) vs `wrench_norms()` (clamped arm feedback).

---

### 3. Integration issue: dynamic finite-mass proxy diverges; kinematic proxy is stable

The object solver uses `rigid_avbd_contact_alpha=0.0` (full per-step penetration
correction) + `rigid_contact_history=True` (augmented Lagrangian: `lam_new = k·C + lam`
**accumulates** while penetration persists; `f_n = ke·pen + lam_n`). The gripper holds
the pads at a commanded width, so penetration is re-created every substep → `λ`/`f_n`
ramp without bound. The kinematic-vs-dynamic split is decisive:

- **Why KINEMATIC works:** a kinematic body (`inv_mass==0`) **early-outs** in Newton's
  rigid integrator (`rigid_vbd_kernels.py` ~L1847) — the runaway contact force is
  *computed* (hence harvestable, the 1e4–1e6 N) but **never applied to the proxy's
  velocity**. The pad is immovable; nothing diverges. Grip force uncontrolled but stable.

- **Why DYNAMIC fails:** a finite-mass proxy **does** get `v += f_n·inv_m·dt` applied,
  so the runaway `f_n` blows up its velocity. The momentum-consistent undo only cancels
  the **one-step-lagged** force; while `λ` ramps `lagged ≪ current`, leaving a residual
  `∝ inv_m`. Two distinct blow-ups observed (substep-level instrumented):
  - **Descent, t≈2.78 s (FIXED):** the re-pinned dynamic proxy hits the **STATIC table**
    (which can't move to resolve penetration) → `λ` explodes `0 → 8.8e4 → 6.6e14 → inf`
    in ~3 substeps. Heavy mass (10 kg) only slowed it. **Fix that worked:** *filter
    proxy↔table collision* (redundant — the robot-side `robot_contact_table` already
    stops the real fingers). Descent explosion gone.
  - **Close→hold, t≈4.35 s (UNRESOLVED):** a second blow-up remains. It is
    **ke-INDEPENDENT** (changing the proxy↔cable `ke` 4× did not move the timing) and
    **raw contact force reads 0 throughout** (the grip never cleanly engages), and it
    drags the whole object model down (the soft block falls through the table). So it is
    **NOT** the cable-pinch-`η` that the "soften contact" fallback (step 2) targets —
    that fallback addressed the wrong mechanism. Leading hypothesis: an instability in
    the **force-limited control loop itself** during close/hold (dynamic proxy +
    per-pad finger feedback + effort-limited gripper), not the contact stiffness.

**Status:** M2 force-limited dynamic proxy is **blocked** at the t≈4.35 s close/hold
blow-up. M1 kinematic is the stable in-use default. The documented fallback
(filter table + soften contact) did **not** resolve it.

**Next diagnostic (not yet run):** localize the 4.35 s blow-up before any further
change — (a) does the ARM or the OBJECT diverge first; (b) toggle per-pad finger
feedback off (net-to-EE) while keeping the dynamic proxy + effort cap, to test whether
the feedback→finger loop is the trigger; (c) check the effort cap's actual effect.
Constraint from the user: do **not** add a post-hoc clamp on the grip force to force
stability; the limit must come from the actuator effort.
