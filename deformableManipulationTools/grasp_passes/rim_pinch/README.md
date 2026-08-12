# `rim_pinch` — top-down lip pinches on open containers

Generates grasp candidates for a regime the other generators structurally cannot reach: a shallow
pinch on the **rim wall** of an open container.

## Why it exists

Seven rigid catalog assets are open containers — mug, bowl, pitcher, bucket, tool_bin, parts_bin,
long_tray_bin — and they are the catalog's worst performers at the pre-grasp collision check. The
cause is dimensional:

* their bodies are **88–152 mm** across, against an **80 mm** jaw stroke;
* the **204 mm** palm sits only **47 mm** behind the TCP.

So any pose deep enough to straddle the body puts a finger or the palm through it. The one regime
that fits is the rim wall itself — **3–6 mm** of material (the YCB bowl is under 1 mm), which the jaw
clears with room to spare.

`obb_face` derives approaches from bounding-box faces; `geometric` slices along the canonical axes.
A rim is a thin annular edge aligned with neither. Neither pass *rejects* these grasps — nothing in
either search is shaped like a lip.

## Rims are located, not inferred

The seed is a `vlm_regions` region (`rim`, `lip`, or `spout`). "Which edge of this mesh is the
opening" is a semantic question, and guessing it geometrically is how a pass ends up confidently
pinching the base of a bowl. **Where an asset has no such region, this pass emits nothing and says
why** — the same refusal `vlm_regions` makes; an absent annotation is not licence to guess.

The store gives a ball with no extent, so [`rim.py`](rim.py) turns each seed into a run of sites
along the real edge. It rests on one local fact:

> A rim is a thin sheet, and the **smallest principal direction** of a small surface patch on it is
> the wall thickness — whichever face the point sits on. That is the jaw axis, and it is reliable
> precisely because the thickness is far smaller than the patch.

It is *measured*, not read from the region's stored `normal`: that is a mean surface normal at the
picks, and the mug's four rim regions disagree with each other by 90°.

The other two axes come from the body, not the patch — a spherical patch on a rim is about equally
wide along the edge and down the wall, so the 2nd and 3rd principal directions are a coin toss.
**Descent** is the direction toward the body centre projected across the wall (which is "down the
wall" whichever way the asset is canonically oriented — the mug's canonical frame has it upside
down); **tangent** completes the frame and is what the walk follows, snapping back to the surface
each step so it tracks a curve it never has to model.

## Run it

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes run rim_pinch
.venv/bin/python -m deformableManipulationTools.grasp_passes run rim_pinch --asset mug
.venv/bin/python -m deformableManipulationTools.grasp_passes run rim_pinch --check-idempotent
```

## Seat mode

Poses are pad-seated through `grasp_library.pad_seat` like everything else (rule 2), and they come
back **`retreated` or `clamped_deep`, essentially never `centred`** — by construction, not by
accident. The material column below a lip runs the full height of the container wall, which no
53.8 mm pad can enclose. That is the correct description of a lip pinch: the pads hold the near
material and the rest of the container hangs below them.

## Things worth knowing

* **Catalog containers are scanned shells, thinner than intuition suggests.** The YCB bowl measures
  0.94–1.01 mm and the mug 1.1–2.6 mm. `config.min_wall` is 0.5 mm for that reason — a 1.5 mm floor,
  which sounds conservative, rejected every bowl seed and produced zero candidates for it. What the
  floor actually needs to exclude is a duplicated coincident face, which reports microns.
* **A seed on a flat lip has no measurable thickness where it sits.** Where the annotator pointed at
  the top of a rim wider than `patch_radius` (the pitcher's flanged lip), the patch is a horizontal
  plate, its thin direction is vertical, and the probe fires straight down through the vessel —
  measured 129 mm. `config.wall_search_depths` steps a little way down the wall, where the thin
  direction is across the wall again. The un-nudged point is always tried first.
* **The thickness probe takes the first TWO ray crossings, not the full span.** A third crossing is
  the far side of the container, and a pinch spanning to it would be a grasp of the whole body —
  exactly the regime that does not fit this gripper.
* **The walk stops rather than wanders.** A sample that drifts out of the seed's height band has
  left the rim; one whose wall exceeds `max_wall` has reached the solid body. A seed the VLM put
  somewhere that is not really an edge yields nothing, not a run of candidates on the body.
* **Cursor and measurement point are kept separate.** The cursor walks at the seed's height; the
  measured site may sit slightly below it. Folding the correction back into the cursor would let it
  accumulate into a slow slide down the wall.
