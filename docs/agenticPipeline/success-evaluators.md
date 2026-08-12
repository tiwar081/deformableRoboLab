# Success evaluators (`agentic_pipeline/success.py`)

The **scoring** layer of the agentic pipeline. Task gen ideates a goal
([task-generator.md](task-generator.md)); this layer answers "did a rollout achieve it?" —
`success.evaluate(predicate, params, SceneState)` runs a geometric test against a live sim snapshot,
and `compile_success_spec` embeds the executable spec in `task.json` as `success_spec`.

## How scoring is wired

- **`SceneState`** is the snapshot a predicate reads: `bodies[name]` = `{pos, aabb, yaw_deg}`,
  `particles[name]` = an (N,3) point set (cloth/FEM particles or cable nodes), plus `base_xy` /
  `facing_yaw_deg` for robot-POV cones. Everything an evaluator may use must come from here.
- **`DRIVERS`** is the set of implemented evaluators. A goal in
  `agentic_pipeline/goal_predicates.json` names one via its `driver` field, or `null`.
- **`driver: null` means NOT SCORABLE, not "fails"**. `success.evaluate` returns
  `{"ok": false, "evaluable": false}` and `compile_success_spec` carries `evaluable: false` — a
  rollout must treat that as *no score*, never as a failed attempt. `pending_evaluators()` lists them.
- **The guard**: `load_goal_predicates()` rejects a `driver` naming an evaluator absent from
  `success.DRIVERS`, so a phantom evaluator can't be declared by a data-only edit. This is what
  stops the withdrawn heuristics below being quietly re-declared.

Generation is unaffected by scoring: every predicate in the table is generatable and
feasibility-checked whether or not it can be measured.

## 16 of 27 scorable — why 11 are null (2026-07-27)

The deformable **shape/topology** evaluators were withdrawn after probing each against a
true-success *and* a true-failure state. Every one either false-positived or could never fire.
Common cause, and the rule to carry forward:

> **An axis-aligned bounding box cannot express SHAPE or TOPOLOGY.** Do not re-attempt any of these
> with AABB extents/thickness heuristics. Each needs the real measurement below, plus a probe
> showing it separates the two states, before its `driver` goes back in.

The deformable **position** goals were kept — they reuse the generic geometry and were verified
alongside it.

| Predicate | How the withdrawn version failed | What to build instead |
|---|---|---|
| `cloth_folded` | a *crumpled* ball scored identically to a neat fold (both compact + thick) | spawn-time flat footprint as reference → require shrink to ~1/2^k for k folds **and** particle **layering** (cluster by z into k+1 planar sheets of similar area, stacked in register). Layer structure is what separates folded from crumpled. |
| `cloth_spread` | failed BOTH ways: a shirt folded in half passed; a small napkin lying perfectly flat failed (absolute thresholds 0.12 m / 0.12 flatness) | compare to the garment's **own** spawn-flat area (scale-free): spread iff current area ≥ ~0.9 × reference **and** single-layer thickness |
| `cloth_draped_over` | xy-overlap only → a box sitting **on** a flat cloth read as the cloth draped over the box (relationship inverted) | cloth particles **above** the target's top face within its footprint **and** descending below that face on two opposing sides (the hanging skirt) |
| `cable_coiled` | demanded thickness > 1 cm, which a cable coiled **flat** on a table hasn't got → a perfect coil scored `False`; could never fire | node-chain **turning angle ≥ 2π** (sum of per-node heading changes), or end-to-end ≪ arc length within a small radius. Winding is the physical quantity, not extent. |
| `cable_straightened` | PCA linearity is direction-blind: a cable folded back on itself (U-shape) scored 0.986 → "straight" | end-to-end distance ≥ ~0.95 × arc length along the node chain — the definition of straight, and it rejects a fold |
| `cable_routed_through` | xy-overlap → a cable lying 10 cm **above** a ring read as threaded through it | **needs authored aperture metadata** on targets (hole centre/radius/plane). Then: a node inside the aperture disc **and** the chain crossing the aperture plane with nodes on both sides in sequence. |
| `cable_threaded_through` | same evaluator, same failure | same aperture machinery, plus tighter tolerance and a minimum protruding length past the far side — build with the above |
| `object_compressed` | **hardcoded `return True`** — every task scored as success | compare to the object's **own** rest height/volume (spawn-settled reference), e.g. ≤ ~0.7 × rest. An elastic FEM body rebounds on release, so this likely needs evaluating over a **window** (peak compression), not at a terminal state. |
| `bag_opened` | read `mouth_open`/`mouth_points` metadata **nothing in the sim emits** → always `False` in a real rollout | tag the bag mesh's **mouth-rim vertex loop at asset-import time** (named vertex group in the catalog entry), then measure the live rim loop's enclosed area / min diameter vs its flattened-shut area |
| `bag_mouth_accessible` | same missing metadata | same tagged rim, plus a clearance test (no other object's AABB intrudes into the column above it) — build after `bag_opened` |
| `object_in_bag` | `open_top_hull` is right for a **rigid** container, but a collapsed cloth bag's convex hull is a flat sheet → an object resting **on** the flattened bag read as inside it | gate `open_top_hull` on the bag actually being open (reuse the tagged rim) **and** require the object below the rim plane |

## Build order

- **Cheapest wins, no new infrastructure**: `cable_straightened` and `cable_coiled` — pure
  node-chain math on data `SceneState.particles` already carries.
- **Needs a rest-state reference captured at settle time**: `object_compressed`, and the
  spawn-reference halves of `cloth_folded` / `cloth_spread`. The settle stage already writes settled
  poses back to `scene.json`; a settled *shape* reference (flat footprint, rest height) is the same
  kind of artifact and would serve three predicates at once.
- **Blocked on ASSET ANNOTATION, not geometry code** — budget these as asset work, not evaluator
  work: the `*_through` pair needs **aperture metadata** (hole centre/radius/plane) on targets with
  holes, and the bag-mouth pair (`bag_opened`, `bag_mouth_accessible`, `object_in_bag`) needs a
  **tagged mouth-rim vertex loop** produced at import. Both are per-asset, precomputed-offline,
  looked-up-at-runtime annotations — the same shape of problem as per-asset grasp candidates, so
  they should reuse whatever per-asset annotation store that work establishes rather than inventing
  a second one.

## Landing one

1. Implement the evaluator in `success.py` and add its name to `DRIVERS`.
2. Wire the branch in `evaluate`.
3. Set the predicate's `driver` in `goal_predicates.json` (the loader validates it exists).
4. **Keep the probe** — the true-success/true-failure pair that shows it separates the two states.
   That probe is what the 2026-07-27 withdrawal was for; an evaluator without one is not done.
