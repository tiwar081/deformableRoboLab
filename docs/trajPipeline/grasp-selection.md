# Grasp selection — choosing a candidate for a PLACED object

`deformableManipulationTools/grasp_select/` picks a grasp for an object at a concrete scene
placement: **prune → filter → project → score → sample**. Shipped standalone and tested
(`python -m deformableManipulationTools.grasp_select.selftest`) but **deliberately unwired** —
nothing imports it yet, because no trajectory stage exists. The result carries the full ranked
set, every rejection with its reason, and the missing-evidence stats (candidates with no measured
quality, no region, unrecorded containment, projection distortion) — a caller re-ranks
`result.ranked` itself; nothing hides behind the pick. API detail: the `grasp_select/__init__.py`
docstring.

The stages, cheapest first:

1. **Face-bucket pruning** — six dot products for the whole record, reusing `obb_bucket`'s
   pruner including its borderline handling (a diagonal grasp survives if ANY plausible bucket
   survives). Measured on the mug at one placement: 46 of 93 pruned before anything expensive.
2. **Approach clearance** — a straight run-up corridor test against table + declared obstacles
   (`clearance.py`, numpy only).
3. **Projection into the executor's vocabulary** — see below.
4. **Scoring** — four weighted terms (`scoring.py`): measured physics quality (NEUTRAL 0.5 when
   absent — null quality is counted, never read as "does not hold"), proximity to a
   `vlm_regions` annotation (joined with `REGION_MARGIN` 2.5 cm), rotational awkwardness vs the
   measured home-pose hand rotation, and containment — which reads the first-class `seat_mode`
   field (it replaced string-matching per-generator prose, which could not carry a third mode).
5. **Depth-variant resolution** — for each surviving `_ctr` centred candidate (grasp-library
   schema v5), `beyond_clearance` probes the space its fingertips would occupy past the object's
   far surface (table plane + obstacle boxes): room → the centred variant supersedes its flush
   sibling; blocked → the variant is rejected (`"depth"` stage) and flush stands. This is the
   whole online half of the depth question — a binary pick between two offline-validated poses.
6. **Sampling** — `temperature=0` = argmin; higher samples the softmin.

## The projection result (closes the old "IK vocabulary" open question)

`robot._ik_target_rot` can only express *straight-down, yawed, tilted*. The projection
(`projection.py`, the ONE place this policy lives) proves this vocabulary is **not
orientation-limited**: `R_cmd = Rot(axis,tilt)·Rz(yaw)·R_home` with a free horizontal tilt axis
is a Z-X-Z Euler decomposition, which covers all of SO(3) — every stored rotation has an exact
closed-form `(yaw, tilt, axis)`, no search. The only real limit is tilt MAGNITUDE; the clamp
moves only that angle, so the distortion charged to a candidate is provably `β − TILT_MAX`
(asserted in tests) and is logged per candidate.

- **`TILT_MAX` is 90°, and the earlier 75° cap was a calibration bug** caught only by surveying
  the whole catalog: 733 of 922 survivors were clamped by *exactly* 15.0°, because a pure
  horizontal side grasp needs exactly 90° of tilt — an entire legitimate class sat on the
  rejection boundary. Post-fix: 1036 of 1767 selectable; projection rejections 173 → 59.
- **The jaw's 180° flip cannot reduce tilt** (it rotates about the approach; `(Δ·Rz(π))[2,2] =
  Δ[2,2]`) — a yaw-only symmetry, taken to keep the wrist near zero, never able to rescue an
  over-tilted approach.
- `R_home` is measured by FK at `home_q` (link7 +z along world −z, fingers along −y; both robots
  agree to 3e-6), not assumed.
- The rotational-awkwardness metric is the standard bi-invariant geodesic
  `arccos((tr(R₁ᵀR₂)−1)/2)`. It was requested as "PRISM paper Appendix B.1", which was never
  available to check — if B.1 specifies a variant (e.g. chordal), only `geodesic_angle` changes.

**2026-08-11 — the default pool excludes weak candidates** (closes the old known gap where a
`seat_blocked` candidate ranked #1 on the mug). Per the status taxonomy in
[grasp-library.md](grasp-library.md) "Candidate statuses", pool assembly now precedes stage 1:
weak grasp options (`retreated` seats, merge-stamped `weak_grasp_option`; 3% hold at n=628) are
excluded from the default pool, counted in `stats["weak_excluded"]`, and called out in `report()`
so a reader sees why the pool shrank. The escape hatch is `select_grasps(..., include_weak=True)`
— included weak candidates rank naturally (scoring already penalises the retreated seat; the
scoring math is unchanged). `seat_blocked` candidates no longer reach selection at all (the merge
discards them from records); any met anyway — a pre-migration record or a raw sidecar — are
dropped unconditionally, `include_weak` or not, since no collision-free depth exists to command.
The filter keys on `grasp_library.is_weak` (the `seat_mode` fact), not the stamped label, so pre-
and post-migration records behave identically. (`overhang` on `clamped_deep` candidates is
tracked in the record, deliberately NOT a scoring input.)

Note for the consumer of a selection result: the executed grasp must PRE-SHAPE the fingers to
`candidate.width + PREGRASP_MARGIN` before the approach begins — the library's collision
feasibility and validation numbers all assume that aperture, never a fully-open approach (the
contract is spelled out in [README.md](README.md)).
