# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## No task in flight

The cloth-grasp task (2026-07-07) is resolved and promoted: root causes, the unit-conversion law,
Newton's grasp recipe, and the measured proxy-vs-kinematic comparison live in
[cloths.md](cloths.md); the demos reproduce Newton's exact folding sequence
([examples.md](examples.md)). Standing open items were promoted to their homes:

- box-slice proxy pad sheds the cloth wad at drag onset → [cloths.md](cloths.md) gotcha 8.
- force-mode `GripController` not yet usable for cloth (rigid-scale engage floor) →
  [gripper.md](gripper.md) "Known limitation".
- ycb-VBD banana intermittent grip + rubik's release-stick → [examples.md](examples.md)
  (`pickplace_ycb_vbd_franka`).
