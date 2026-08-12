You invent ONE realistic scenario prompt for a robot-manipulation scene generator (the "userless"
mode: no human gave a prompt, so you write the kind of prompt a human would).

Write a BROAD description of a situation — the setting, and what has just happened there. A
downstream scene generator picks the actual objects and where they go, so you do not need to
describe any objects; if one or two come up naturally in the phrasing, that is fine.

Prompts of the right shape:

- "a workshop tool-sorting station"
- "a messy office desk after lunch"
- "dirty laundry in my laundry bin in my room"
- "the kitchen counter right after unpacking the groceries"
- "a workbench halfway through a repair job"
- "a hotel housekeeping cart mid-round"

Rules:
- ONE short sentence, phrased the way a person would say it out loud.
- The situation must be BUILDABLE from the catalog below — that is the only reason you are shown it.
- Prefer situations that naturally involve a DEFORMABLE object (a garment, bag, cable/rope, or
  squishy item) — that is this simulator's specialty.
- Vary your choice of setting; do not always reach for the same one.

The catalog the generator will build from:
$catalog

Respond with the JSON only.
