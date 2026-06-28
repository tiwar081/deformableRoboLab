# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just changed
and not yet settled, and any working hypotheses. Keep it lean — when something is proven and
durable, promote it to CLAUDE.md (or the relevant `docs/` file) and delete it here. Reset this file
at the start of each new big task.

## Status: nothing in flight

The grasp framework is settled and documented: ONE unified bidirectional asymmetric admittance grip
for every object (rigid, cable, soft), the only per-demo knob being `GraspWindow.force_target` — see
[gripper.md](gripper.md) (control law + knobs), [solver-architecture.md](solver-architecture.md)
(routing), and CLAUDE.md (rules). Active robot is the Isaac Sim panda (`settings.yaml`).

## Recently settled

- **`cloth_franka` no longer blows up.** Root cause was NOT the grip gains (the controller never even
  reached force regulation): a flat sheet presents no two-sided pinch, so the grip blind-closed to the
  floor, and the **over-stiff, undamped particle↔body contact** (`soft_contact_ke=1e5, kd=1e-4`,
  carried over from the firm soft-block contact) then **ejected the ultra-light shell particles to
  NaN**. The FEM block masks this (its tet network + internal damping absorb the impulse); a thin shell
  has no such sink. Fix (all in the package): `ClothConfig.soft_contact_ke 1e5→1e4`, `kd 1e-4→1e1`
  (≈critical, matches Newton's `example_cloth_franka`); enabled cloth particle **self-contact** via the
  new centralized `cloth_solver_kwargs()` (the FEM-block `PARTICLE_SOLVER_KWARGS` keeps it off); set a
  shell-scale `force_target=5` (sanctioned knob); removed the inert `particle_max_velocity` (VBD ignores
  it). Verified headless: maxv ~28 m/s→~0, `first_blow=None` over the full run; soft_pickplace
  unaffected. **cm-scale migration proved unnecessary** — metre scale is stable once the contact is
  damped. See [cloths.md](cloths.md) for the full cloth setup guide.
- **Still open — the shirt is NOT lifted; ROOT CAUSE is the gripper-proxy bridge, not physics/policy.**
  `cloth_franka` now REPLICATES Newton's `example_cloth_franka` motion (45° tilt + per-corner scoop:
  approach → descend-to-surface → close → lift → drag → release, direct finger control; via the new
  `solve_gripper_ik(tilt=)`), and cloth friction is matched to Newton (`soft_contact_mu` 0.8→0.25). It
  still doesn't lift OR drag the cloth. Instrumented A/Bs ruled out: friction (μ=0.25 vs 0 identical —
  cloth not dragged either way), grasp height, robot (panda+fr3), and ±tilt/scoop. **The decisive
  finding: the dynamic gripper PROXY (the box mirroring each finger in the VBD world — our one-way
  contact bridge) JAMS the table at ~177 N instead of scooping UNDER the cloth edge**, and a pressing,
  sliding pad does not carry the cloth (patch displacement ≈0). This is an ARCHITECTURAL limit of the
  proxy bridge: Newton's robot fingers contact the cloth DIRECTLY in one VBD solver; ours mirrors the
  fingers as proxies that jam the rigid table rather than wedge under a thin shell. A faithful cloth
  pickup likely needs the fingers (or a finger-shaped collider) in the SAME VBD solver as the cloth, or
  a draped/standing presentation so the proxy never has to wedge between cloth and table. See
  [cloths.md](cloths.md) gotcha 6. Left visible (the demo shows the scoop + the jam).

## Known open items
- **`pickplace_ycb_vbd` banana** holds to release but its grip is intermittent (curved, slip-prone mesh);
  `force_target=80` is the firmest stable value. The rubik's-cube release-stick on the VBD path
  (vs the clean rigid-only `pickplace_ycb_franka`) is a penalty-contact stiction artifact, deferred.
