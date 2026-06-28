# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just changed
and not yet settled, and any working hypotheses. Keep it lean — when something is proven and
durable, promote it to CLAUDE.md (or the relevant `docs/` file) and delete it here. Reset this file
at the start of each new big task.

## In flight: `cloth_franka` — the gripper cannot even MOVE the cloth

Goal: a Franka picks up / drags a flat T-shirt on the table (the first cloth-manipulation demo,
`examples/cloth_franka.py`). The sim is **stable** — the blow-up fight is settled and now lives in
[cloths.md](cloths.md) (≈critically-damped particle↔body contact + cloth self-contact; metre scale
is fine). The live blocker is the grasp itself.

**Symptom (current): the cloth does not move at all — not lifted, not dragged, not even nudged.**
`cloth_franka` now replicates Newton's `example_cloth_franka` motion exactly: a ~45° tilt + per-corner
SCOOP (approach → descend-to-surface OPEN → close → lift+translate → drag → release → retract),
driving all 9 DOFs from IK keyframes (Newton's position-activation gripper), **not** the force
`GripController`. The pads close on/over the shirt edge and the shirt stays put.

**Decisive A/B: gripper friction OFF behaves IDENTICALLY to friction ON.** `ClothConfig.soft_contact_mu`
= 0.25 (matched to Newton) vs 0 are indistinguishable — cloth patch displacement ≈ 0 either way. So
this is **not** a friction/μ-tuning problem: a pad that never really grips the shell can't drag it no
matter how sticky it is. Earlier instrumented sweeps already ruled out grasp height, ±tilt/scoop, and
robot (panda vs fr3) — all identical, no motion.

**Working root cause — the gripper-PROXY bridge, an architectural limit (NOT physics/policy/friction).**
Our split design mirrors each MuJoCo finger as a dynamic box PROXY in the cloth's VBD world (the
one-way contact bridge — see [gripper.md](gripper.md)). Instrumented: that proxy **jams the rigid
table at ~177 N** instead of wedging UNDER the thin shell edge, and a pressing+sliding pad carries no
cloth. Newton's `example_cloth_franka` succeeds because its robot fingers contact the cloth DIRECTLY
inside ONE VBD solver — there is no proxy that must squeeze between cloth and table. With friction
ruled out, this is the remaining explanation.

**Next directions (unverified):**
- Put a finger-shaped collider IN the cloth's VBD solver (not a mirrored proxy), so the fingers
  contact the shell directly the way Newton's do. Most faithful, but breaks the split MuJoCo-robot ↔
  VBD-object architecture the rest of the framework relies on.
- Or change the cloth PRESENTATION so no under-wedge is needed: drape/stand a corner over a table
  edge or backstop, or pre-lift a corner, so the proxy grasps a free feature instead of pinching a
  flat sheet against the table.
- Keep the failure VISIBLE in the demo (it shows the scoop + the jam); do not fake the grasp.

Full cloth setup + the two grasp-limit gotchas: [cloths.md](cloths.md) §6 (flat-sheet top-down-pinch
limit) and §7 (the proxy-bridge architectural blocker).

## Known open items (other demos)
- **`pickplace_ycb_vbd` banana** holds to release but its grip is intermittent (curved, slip-prone
  mesh); `force_target=80` is the firmest stable value. The rubik's-cube release-stick on the VBD path
  (vs the clean rigid-only `pickplace_ycb_franka`) is a penalty-contact stiction artifact, deferred.
</content>
</invoke>
