# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## In flight: gain-asymmetry redesign + target-relative retune (2026-07-08)

The open/close asymmetry moved from the rate caps into the GAINS: `grip_rate_open` (2e-4, a
controller choice disguised as a hardware limit) is DELETED; `grip_rate_max` is now the symmetric
physical jaw-speed cap (both directions), and `window_params` returns `(k_close, k_open, deadband)`
with `k_open = k_close/k_open_ratio` (ratio 20 — 10 lost the cable cage in cable_soft). Canary validations: cable cage (spike-dominated
worst case), banana, soft block @5 N. Awaiting verdict on the re-rendered videos.

### Earlier in this task: target-relative retune

The GripController's admittance constants are now derived per grasp window from `force_target`
(`GripConfig.window_params`: gain ∝ 1/target capped, deadband ∝ target floored; anchored bit-exact
at the long-tested 30 N constants — [gripper.md](gripper.md) "Knobs"). Physical robot properties
(`grip_rate_max`, `engage_floor`, `force_filter_tau`, actuator gains) stay fixed for sim2real.
This closes the cloth force-grip caveat: `cloth_franka` now targets an ACHIEVABLE 2 N and the
regulator converges to a stable ~8–9 mm pinch instead of creeping shut. Retargets: cloth 8→2,
plate 50→30 (anchor); others unchanged. Re-rendering all affected demos —
awaiting verdict on the videos.

## Standing open items (promoted to their docs)

- ycb-VBD banana intermittent grip + rubik's release-stick → [examples.md](examples.md); after the
  one-table remap the RIGID ycb twin's cube also bounces off the bowl rim at its drop (candidate
  fix: release ~3 cm more bowl-central). Watch the banana under the new (slower at 80 N) cage
  tightening in the re-render.
