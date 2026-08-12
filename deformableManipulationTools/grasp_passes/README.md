# Writing a grasp pass

A **pass** computes something about one catalog asset's grasps. Passes are built in parallel by
separate agents, so the rule that makes that safe is: **a pass writes only its own sidecar, and
adding a pass touches only its own directory.**

```
grasp_passes/
  __init__.py  base.py  catalog.py  merge.py  selfcheck.py  __main__.py   <- SHARED, DO NOT EDIT
  fixture/          <- an existing pass
  <your_pass>/      <- add exactly this
```

`deformableManipulationTools/grasp_library.py` is **frozen** — the record schema, the canonical
frame, and the two pose conventions live there and passes import from them. If you believe a shared
file must change, stop and raise it rather than editing: another agent is building against it.

## The 30-second version

```python
# grasp_passes/my_pass/__init__.py
import numpy as np
from ...grasp_library import MAX_JAW_WIDTH, make_candidate
from ..base import GraspPass, PassContext, PassOutput

class MyPass(GraspPass):
    name    = "my_pass"        # MUST equal the directory name
    source  = "my_pass"        # unique across all passes; stamped on every candidate
    version = 1                # bump when output changes -> invalidates cached sidecars
    kinds   = ("ycb_mesh",)    # () = every supported kind
    requires = ()              # names of passes whose candidates you consume

    def run(self, ctx: PassContext) -> PassOutput:
        c = make_candidate(ctx.frame, "grasp_0",
                           position=[0, 0, 0], approach=[0, 0, -1], jaw_axis=[0, 1, 0],
                           width=0.03, source=self.source)
        return PassOutput(candidates=[c])

PASS = MyPass()
```

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes list
.venv/bin/python -m deformableManipulationTools.grasp_passes run my_pass --asset banana
.venv/bin/python -m deformableManipulationTools.grasp_passes run my_pass --check-idempotent
.venv/bin/python -m deformableManipulationTools.grasp_passes show banana
```

Discovery is by convention — a subdirectory exporting `PASS` — so there is no registry to edit and
nothing for two agents to conflict on.

## What you get: `ctx`

| field | what it is |
|---|---|
| `ctx.name`, `ctx.kind` | catalog name and kind |
| `ctx.asset.entry` | the raw `scene_catalog.json` entry (mass, mu, dims, …) |
| `ctx.asset.vertices` / `.faces` | rest geometry in the asset/body frame; `faces` is None for tet and point-set kinds |
| `ctx.asset.canonical_vertices()` | the same geometry in canonical coordinates — what you measure against |
| `ctx.frame` | the canonical `ObjectFrame` (rotation, translation, `extents`, `ambiguous`, `drift`) |
| `ctx.asset.half_extents` | convenience, `extents / 2` |
| `ctx.upstream` | candidates from the passes in `requires`, already id-namespaced |

**Never recompute the canonical frame.** Poses from two different frames are not comparable, and the
merge rejects a sidecar whose frame disagrees with the asset's.

## Rules

1. **Emit poses in `ctx.frame`**, following `tcp_z_approach_x_jaw_v2`: origin at the TCP, `+z`
   approach, `+x` jaw axis. `make_candidate` builds one correctly and derives the OBB face bucket
   for you — prefer it to hand-building a matrix.
2. **Pad-seat every pose before you store it** — this is what the `v2` in the convention means, and
   it is not optional:

   ```python
   from ...grasp_library import SEAT_BLOCKED_LABEL, pad_seat, make_candidate

   seat = pad_seat(centre, approach, jaw, width, ctx.asset.canonical_vertices(), ctx.asset.faces)
   if seat is None:
       continue                      # no material in the jaw column — drop, don't store
   cand = make_candidate(ctx.frame, cid, seat.position, approach, jaw, width=width, source=...,
                         seat_mode=seat.seat_mode,
                         labels=(SEAT_BLOCKED_LABEL,) if seat.blocked else ())
   ```

   **The TCP is at the FINGERTIP TIPS.** The pads run from `PAD_NEAR_Z` (−0.7 mm) back to
   `PAD_FAR_Z` (−54.5 mm) *behind* the grasp centre, all of it on the hand side. A pose left on the
   object's material therefore grips with the very tips and leaves about half the object forward of
   the fingers — measured, and it is why v1 records are rejected outright rather than migrated.
   `pad_seat` raycasts the jaw column for the material span and seats the pose, then checks the
   HAND'S OWN COLLISION HULLS at the result (the same test the shake pre-check runs). Three
   outcomes, named by `seat.seat_mode` — a **required** field on every stored candidate (schema
   v2), which `make_candidate` refuses to default: a span that FITS the pads is `centred` between
   them; a DEEPER one gets the `clamped_deep` seat — near material flush with the palm face
   (`SEAT_DEEPEST_Z`), overhang protruding forward past the fingertips — because centring a span
   that cannot fit pushes its overhang backwards into the palm (measured: all 116 apple candidates
   palm-colliding under midpoint seating); and a pose whose rule seat leaves the hand inside the
   object is `retreated` along the approach to the deepest collision-free depth that keeps
   material between the pads (measured 2026-08-06: on bodies wider than the jaw stroke the only
   clear poses are 24–51 mm shallower than any rule seat). If NO depth clears, `seat.blocked` is
   set — keep the candidate at the rule seat and add `SEAT_BLOCKED_LABEL`; a measured failure is
   marked, never deleted **in your sidecar**. (What the MERGE does with these statuses is central
   policy, not yours: blocked candidates are discarded from the record, retreated ones are stamped
   `weak_grasp_option` and count as weak options rather than legitimate candidates — see
   docs/trajPipeline/grasp-library.md "Candidate statuses". Emit and label; do not pre-filter.)
   Do not re-derive any of this or add an offset of your own: where the pads are is a property of
   the hand, not of how a grasp was found, so it lives in exactly one function.
3. **Only your own `source` tag.** The merge deduplicates by source; emitting another pass's tag is
   rejected. Candidate ids are namespaced to `<source>/<your id>` by the harness, so pick ids unique
   within your pass and don't worry about other passes.
4. **Be deterministic.** Same asset in, same output out, including ids. Seed any sampling from the
   asset (e.g. `np.random.default_rng(abs(hash(ctx.name)) % 2**32)`), never from the clock or an
   unseeded RNG. Prove it with `--check-idempotent` before you hand the pass over.
5. **Write nothing.** Return a `PassOutput`; the harness writes the sidecar. Never open
   `assets/objects/grasps/<name>.json` — it is a build artifact composed by `merge`, and anything
   you write there is lost on the next merge. (READING it is banned too, with exactly one
   sanctioned exception: `llm_retry`, whose trigger and input ARE the composed state — see
   docs/trajPipeline/llm-retry.md. Every other pass takes `ctx.upstream`.)
6. **`width` must fit the jaw** (`0 < width <= MAX_JAW_WIDTH`, 0.08 m). If a grasp does not fit,
   drop it — do not clip the width, which would describe a grasp the gripper cannot make.
7. **Cloth, bags, and CABLES are out of scope.** `kind: "cloth"` has no persistent rest shape;
   `kind: "cable"` failed the pipeline's rest-shape premise empirically (0/62 held; the probe
   read the synthetic rod's construction axis, not geometry). Validation rejects records for
   both. See the SCOPE paragraph in `grasp_library.py`. The pipeline covers rigid bodies and
   soft (FEM) objects only.
8. **Fail loudly per asset.** Raising for one asset is fine — the runner reports it and continues.
   Silently emitting nothing is worse than an error.

## Producers and consumers

A **producer** fills `PassOutput.candidates`. A **consumer** declares `requires = ("producer_name",)`,
reads `ctx.upstream`, and fills `PassOutput.annotations` — `{candidate_id: {...}}` — to attach
quality numbers or extra labels to candidates it does not own:

```python
return PassOutput(annotations={
    c.id: {"quality": {"object_in_gripper": 1.0, "object_motion_during_shaking_linear": 0.004},
           "quality_source": "newton_vbd", "labels": ("verified",)}
    for c in ctx.upstream})
```

Quality field names are fixed (`grasp_library.QUALITY_FIELDS`, ACRONYM's `flex` metric names
verbatim) and any value you set needs a `quality_source` naming what measured it. `requires` also
sets run order, so `run-all` runs your producer first.

**Consuming every producer, whoever wrote them**: subclass `DynamicUpstreamPass` instead of setting
`requires` — it resolves to the producers that have a sidecar for the asset in hand, so a generator
that lands after you is picked up automatically and one that has not run for an asset does not fail
it. Cycle exclusion is handled centrally (`base.discover_producers`); any number of discovering
consumers compose.

**Developing a consumer before a generator exists**: `requires = ("fixture",)` gives you the
hand-placed candidates in `fixture/`. Switch to the real producer's name when it lands — nothing
else about your pass changes.

## Idempotency and caching

`run_pass` skips the work when the sidecar was written by the same pass version, from the same mesh
(`mesh_sha1`), with the same upstream. So re-running is cheap as well as harmless. Bump `version`
whenever your output would change, or you will keep reading a stale sidecar.

## Verifying the scaffolding itself

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes selfcheck
```

Asserts the guarantees above on throwaway in-memory passes (isolation, dedup by source, no
accumulation on re-run, annotation routing, and the rejections: wrong source tag, duplicate ids,
orphan annotations, missing producer, contested source tag, stale mesh, mismatched frame). It runs
against a temp sidecar directory and never touches real records.
