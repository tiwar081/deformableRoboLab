# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## In flight: cloth_franka force-grip trial (2026-07-08)

`cloth_franka` was shortened to ONE hot-dog fold and switched from the explicit `finger_schedule`
to a `GraspWindow` force grip. Target ladder measured: 1 N and 5 N latch too loose and shed the
wad mid-drag; **`force_target=8 N` holds through the whole fold** (details promoted to
[gripper.md](gripper.md) "trial results"). Rendered with 8 N, awaiting verdict.

Render-iteration findings (all fixed, all measured): (1) arm "dragged down" mid-fold = the 50 cm
drag segment's joint-space blend bowing the TCP 10.5 cm (fingers plowed the table) — fixed with
drag via-points; (2) choppy motion = the per-segment smoothstep pausing at every via-point — fixed
centrally with `WP.via=True` (linear/constant-velocity segments through pass-through points);
(3) dropped shirt = loose latch (1 N/5 N targets) + a mid-drag tilt blend twisting the pinch —
fixed with `force_target=8 N`, tilt 0 throughout (the tilted pose is q6-limit-bound past x≈0; the
untilted far grasp needed pulling in to x=0.35), and a longer stationary tightening dwell.

Remaining known imperfection: with 8 N unreachable for a shell, the jaw creeps to the 2 mm
`min_close_width` floor by release (gradual contact shedding, no expulsion; fold completes) — the
shell-aware stop is the standing retune item in [gripper.md](gripper.md).

## Standing open items (promoted to their docs)

- Box-slice proxy pad sheds the cloth wad at drag onset → [cloths.md](cloths.md) gotcha 8.
- ycb-VBD banana intermittent grip + rubik's release-stick → [examples.md](examples.md); after the
  one-table remap the RIGID ycb twin's cube also bounces off the bowl rim at its drop (candidate
  fix: release ~3 cm more bowl-central).
