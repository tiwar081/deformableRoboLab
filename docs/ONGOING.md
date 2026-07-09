# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## No task in flight

The grip-controller redesign (2026-07-08) is done and promoted: target-relative gains/deadband +
the close/open GAIN asymmetry with a symmetric physical rate cap live in [gripper.md](gripper.md)
("Knobs" + "The law"); the centralized effective harvest ke (avg of particle ke and the pads' final
material, identical for every particle object) is documented in [gripper.md](gripper.md) (harvest
bullet) and [cloths.md](cloths.md). All demos re-validated and re-rendered.

## Standing open items (live in their docs)

- ycb banana intermittent grip (curved, slip-prone mesh; 80 N is the firmest stable target) →
  [examples.md](examples.md).
