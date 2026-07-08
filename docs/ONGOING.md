# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## In flight: target-relative admittance retune (2026-07-08)

The GripController's admittance constants are now derived per grasp window from `force_target`
(`GripConfig.window_params`: gain ∝ 1/target capped, deadband ∝ target floored; anchored bit-exact
at the long-tested 30 N constants — [gripper.md](gripper.md) "Knobs"). Physical robot properties
(`grip_rate_max`, `engage_floor`, `force_filter_tau`, actuator gains) stay fixed for sim2real.
This closes the cloth force-grip caveat: `cloth_franka` now targets an ACHIEVABLE 2 N and the
regulator converges to a stable ~8–9 mm pinch instead of creeping shut. Retargets: cloth 8→2,
soft block 5→4, plate 50→30 (anchor); others unchanged. Re-rendering all affected demos —
awaiting verdict on the videos.

## Standing open items (promoted to their docs)

- ycb-VBD banana intermittent grip + rubik's release-stick → [examples.md](examples.md); after the
  one-table remap the RIGID ycb twin's cube also bounces off the bowl rim at its drop (candidate
  fix: release ~3 cm more bowl-central). Watch the banana under the new (slower at 80 N) cage
  tightening in the re-render.
