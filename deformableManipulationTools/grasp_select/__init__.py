"""Choose a grasp for a PLACED object: prune -> filter -> project -> score -> sample.

    result = select_grasps(load_grasps("mug"), placement={"x": .4, "y": .1, "z": .78, "yaw": .3},
                           robot=RobotState(base_pos=(0, .9, .75)), temperature=0.0)
    result.pick.command.waypoint_kwargs      # -> {"yaw": …, "tilt": …, "tilt_axis": …} for a WP
    print(result.report())                   # what survived, what was rejected, what it cost

Nothing consumes this yet — there is no trajectory generation — so it is deliberately a library with
its own self-test (``python -m deformableManipulationTools.grasp_select.selftest``) rather than a
hook into the demo runner.

The POOL is assembled first, per the status taxonomy in docs/trajPipeline/grasp-library.md
"Candidate statuses": weak grasp options (``seat_mode == "retreated"`` — reachability statements,
3% hold at n=628) are excluded from the default pool and counted in the report
(``include_weak=True`` admits them, where they rank naturally), and any candidate still carrying
``seat_blocked`` is dropped unconditionally (the merge discards those from records; meeting one
here means a pre-migration record or a raw sidecar). The stages then run, in order and each one
cheaper than the next:

1. **Face-bucket pruning** — six dot products for the whole record. A face pointing into the table
   cannot be entered whatever the arm does. Reuses the ``obb_bucket`` pruner, including its
   borderline handling (a diagonal grasp survives if ANY bucket it plausibly belongs to survives).
2. **Approach clearance** — is there a straight run-up to the pose, or does the corridor pass
   through the table or another object (:mod:`.clearance`).
3. **Projection** — the stored rotation expressed as ``(yaw, tilt, tilt_axis)``, the only vocabulary
   the executor has, with the distortion charged and recorded (:mod:`.projection`).
4. **Scoring** — four terms, weighted (:mod:`.scoring`): measured physics quality, proximity to an
   annotated semantic region, rotational awkwardness against the robot's canonical hand pose, and
   whether the pads enclose the object.
5. **Sampling** — ``temperature=0`` returns the argmin; higher samples the softmin.

The result carries the FULL ranked set, its rejections with reasons, and the stats that make the
missing evidence visible (how many candidates had no measured quality, no region annotation, no
recorded containment, and how much orientation the projection had to bend). A caller that wants a
different trade-off re-ranks ``result.ranked`` itself; nothing here is hidden behind the pick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..grasp_library import (CENTRED_VARIANT_SUFFIX, GraspRecord, MAX_JAW_WIDTH,
                             SEAT_BLOCKED_LABEL, body_pose, is_weak)
from ..params import TABLE
from ..grasp_passes.obb_bucket.bucket import buckets_from_labels
from ..grasp_passes.obb_bucket.pruning import DOWN, placement_matrix, surviving_buckets
from .clearance import (ClearanceCheck, DEFAULT_STANDOFF, Obstacle, approach_clearance,
                        beyond_clearance)
from .projection import (HOME_GRASP_ROT, HandCommand, MAX_DISTORTION, TILT_MAX, geodesic_angle,
                         project_pose)
from .scoring import (NEUTRAL, REGION_MARGIN, Score, Term, Weights, sample_index, score_candidate)

__all__ = ["select_grasps", "SelectionResult", "ScoredGrasp", "Rejection", "RobotState", "Weights",
           "Obstacle", "HandCommand", "project_pose", "geodesic_angle", "HOME_GRASP_ROT",
           "TILT_MAX", "MAX_DISTORTION", "REGION_MARGIN", "DEFAULT_STANDOFF"]

log = logging.getLogger(__name__)

# Franka usable reach [m] — the same number agentic_pipeline.geometry places robots by.
REACH_MAX = 0.80


@dataclass(frozen=True)
class RobotState:
    """What selection needs to know about the robot: where it stands and how its hand rests.

    ``hand_rot`` is the CANONICAL hand pose in the grasp-frame convention — the orientation the arm
    holds at ``home_q``, which is what rotational awkwardness is measured against and what the
    executor's yaw/tilt are relative to. It defaults to the measured home pose
    (:data:`projection.HOME_GRASP_ROT`), identical for both robots in ``params.ROBOTS``.

    ``base_pos`` is used only for the reach test; pass ``reach_max=None`` to skip it (the selection
    is still meaningful without a placed robot, which is how the self-test uses it)."""
    base_pos: tuple = (0.0, 0.0, 0.0)
    hand_rot: np.ndarray = field(default_factory=lambda: HOME_GRASP_ROT.copy())
    reach_max: float | None = REACH_MAX
    table_z: float = TABLE.top_z


@dataclass(frozen=True)
class ScoredGrasp:
    """One surviving candidate: where it is in the world, how to command it, what it scored."""
    candidate: object
    pose: np.ndarray                  # 4x4 world <- grasp
    command: HandCommand
    score: Score
    clearance: ClearanceCheck
    rank: int = -1

    @property
    def id(self) -> str:
        return self.candidate.id

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.pose)[:3, 3]

    @property
    def width(self) -> float:
        return float(self.candidate.width)

    def describe(self) -> str:
        p = self.position
        return (f"#{self.rank:<2d} {self.id:34s} cost={self.score.total:.3f}  "
                f"[{self.score.describe()}]  pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
                f"{self.command.describe()}  {self.clearance.describe()}")


@dataclass(frozen=True)
class Rejection:
    """A candidate that did not survive, and which stage removed it."""
    id: str
    stage: str                # "status" | "bucket" | "reach" | "clearance" | "projection" | "depth"
    reason: str

    def describe(self) -> str:
        return f"   {self.id:34s} {self.stage:10s} {self.reason}"


@dataclass(frozen=True)
class SelectionResult:
    """The full outcome: every survivor ranked, every rejection explained, and the evidence gaps."""
    ranked: tuple = ()
    pick: ScoredGrasp | None = None
    rejected: tuple = ()
    buckets: tuple = ()
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ranked)

    def by_id(self, cid: str):
        return next((g for g in self.ranked if g.id == cid), None)

    def report(self) -> str:
        lines = [f"{self.stats.get('asset', '?')}: {len(self.ranked)} of "
                 f"{self.stats.get('candidates', 0)} candidate(s) selectable; "
                 f"buckets kept {list(self.buckets)}"]
        # Pool-assembly exclusions come first: a human reading the report must see WHY the pool is
        # smaller than the record before reading the ranking.
        if self.stats.get("weak_excluded"):
            lines.append(f"  pool: {self.stats['weak_excluded']} weak grasp option(s) (retreated "
                         f"seat) excluded from the default pool — not legitimate candidates; "
                         f"pass include_weak=True to rank them")
        if self.stats.get("seat_blocked_dropped"):
            lines.append(f"  pool: {self.stats['seat_blocked_dropped']} seat_blocked candidate(s) "
                         f"dropped (the merge discards these from records; this record predates "
                         f"the re-merge)")
        for g in self.ranked:
            lines.append(g.describe())
        if self.rejected:
            lines.append(f"  rejected {len(self.rejected)}:")
            lines.extend(r.describe() for r in self.rejected)
        lines.append("  evidence: " + ", ".join(
            f"{k}={self.stats.get(k, 0)}" for k in
            ("quality_unknown", "region_unknown", "containment_unknown")))
        if self.stats.get("distorted"):
            lines.append(f"  projection: {self.stats['distorted']} pose(s) tilt-clamped, "
                         f"max distortion {np.degrees(self.stats['max_distortion']):.1f} deg")
        return "\n".join(lines)


def _regions_for(record: GraspRecord, regions):
    """The region document to score against: what the caller passed, or the store's own lookup."""
    if regions is not None:
        return regions
    try:
        from ..grasp_passes.vlm_regions.regions import load_regions
    except Exception as exc:                      # noqa: BLE001 - the pass is optional
        log.debug("region store unavailable (%s); scoring without semantics", exc)
        return None
    try:
        return load_regions(record.name)
    except Exception as exc:                      # noqa: BLE001 - a missing/broken doc is not fatal
        log.debug("no regions for %r (%s)", record.name, exc)
        return None


def select_grasps(record: GraspRecord, placement, robot: RobotState = None, *,
                  obstacles=(), regions=None, weights: Weights = None, temperature: float = 0.0,
                  seed: int = 0, approach=DOWN, half_angle_deg: float = 90.0,
                  standoff: float = DEFAULT_STANDOFF, region_margin: float = REGION_MARGIN,
                  tilt_max: float = TILT_MAX, max_distortion: float = MAX_DISTORTION,
                  include_weak: bool = False,
                  ) -> SelectionResult:
    """Rank the record's candidates for an object at ``placement`` and pick one.

    ``placement`` is anything :func:`obb_bucket.pruning.placement_matrix` reads — a 4x4 world pose,
    ``(pos, yaw)``, or a ``scene.json`` object entry. ``temperature`` 0 means argmin.

    The default pool holds only LEGITIMATE candidates (grasp-library.md "Candidate statuses"):
    weak grasp options — ``grasp_library.is_weak``, i.e. ``seat_mode == "retreated"`` (3% hold at
    n=628) — are excluded and counted in ``stats["weak_excluded"]``. ``include_weak=True`` admits
    them and they rank naturally (scoring already penalises the retreated seat). Candidates
    carrying ``seat_blocked`` are dropped UNCONDITIONALLY — no collision-free depth exists on
    their approach, so there is nothing to command; the merge discards them from records, and the
    check here is belt over braces for pre-migration records and raw sidecars.

    Returns a :class:`SelectionResult` whose ``ranked`` is the complete surviving set in cost order;
    ``pick`` is the sampled one (None when nothing survives, which is a real answer for an object
    lying on its only graspable face)."""
    robot = robot or RobotState()
    weights = weights or Weights()
    world = placement_matrix(placement)
    doc = _regions_for(record, regions)

    keep = set(surviving_buckets(record.frame, world, approach=approach,
                                 half_angle_deg=half_angle_deg))
    survivors, rejected = [], []
    distorted, max_dist = 0, 0.0
    weak_excluded, seat_blocked_dropped = 0, 0
    unknown = {"quality": 0, "region": 0, "containment": 0}

    for c in record.candidates:
        # STATUS FILTER (grasp-library.md "Candidate statuses") — before any geometry is spent.
        # seat_blocked first and unconditional: no collision-free depth exists on that approach,
        # so there is nothing to command whatever the caller asked for.
        if SEAT_BLOCKED_LABEL in getattr(c, "labels", ()):
            seat_blocked_dropped += 1
            rejected.append(Rejection(c.id, "status", "seat_blocked — no collision-free depth on "
                                                      "this approach (merge-discarded; defensive)"))
            continue
        # Weak grasp options are keyed on is_weak (the seat_mode fact), not the merge-stamped
        # WEAK_GRASP_LABEL, so pre-migration records and raw sidecars filter identically.
        if not include_weak and is_weak(c):
            weak_excluded += 1
            rejected.append(Rejection(c.id, "status", "weak grasp option (retreated seat) — "
                                                      "excluded from the default pool"))
            continue

        if not keep.intersection(buckets_from_labels(getattr(c, "labels", ()), c.face)):
            rejected.append(Rejection(c.id, "bucket", f"face {c.face} is not presented"))
            continue

        pose = record.grasp_in_world(c, world)
        pos, rot = np.asarray(pose)[:3, 3], np.asarray(pose)[:3, :3]

        if robot.reach_max is not None:
            d = float(np.linalg.norm(pos - np.asarray(robot.base_pos, dtype=float)))
            if d > robot.reach_max:
                rejected.append(Rejection(c.id, "reach", f"{d:.2f} m from the base "
                                                         f"(max {robot.reach_max:.2f} m)"))
                continue

        clear = approach_clearance(pos, rot[:, 2], table_z=robot.table_z,
                                   jaw_width=min(float(c.width), MAX_JAW_WIDTH),
                                   obstacles=obstacles, standoff=standoff)
        if not clear.ok:
            rejected.append(Rejection(c.id, "clearance", clear.describe()))
            continue

        cmd = project_pose(rot, home=robot.hand_rot, tilt_max=tilt_max,
                           max_distortion=max_distortion)
        max_dist = max(max_dist, cmd.distortion)
        distorted += bool(cmd.clamped)
        if not cmd.accepted:
            # Logged as well as returned: a pose the arm cannot express is a cost of the executor's
            # vocabulary, and it should be visible without anyone inspecting the result object.
            log.info("%s/%s rejected: needs %.1f deg more tilt than the arm is asked for",
                     record.name, c.id, np.degrees(cmd.distortion))
            rejected.append(Rejection(c.id, "projection", cmd.describe()))
            continue

        score = score_candidate(c, cmd.rotation, robot.hand_rot, doc, weights,
                                region_margin=region_margin)
        for k in unknown:
            unknown[k] += not score.terms[k].known
        survivors.append(ScoredGrasp(candidate=c, pose=pose, command=cmd, score=score,
                                     clearance=clear))

    # DEPTH-VARIANT RESOLUTION (the online half of grasp_library schema v5). A "_ctr" centred
    # candidate is the measured-stronger seat, executable only when nothing occupies the space
    # beyond the object's far surface — checked HERE because only the placement knows. Room ->
    # the centred variant supersedes its flush sibling; no room -> the variant is rejected and
    # the always-executable flush pose stands.
    by_id = {g.id: g for g in survivors}
    drop = set()
    for g in list(survivors):
        if not g.id.endswith(CENTRED_VARIANT_SUFFIX):
            continue
        c = g.candidate
        room = beyond_clearance(g.pose, float(c.seat_depth), float(c.span), float(c.width),
                                table_z=robot.table_z, obstacles=obstacles)
        base = g.id[: -len(CENTRED_VARIANT_SUFFIX)]
        if room.ok:
            if base in by_id:
                drop.add(base)
                rejected.append(Rejection(base, "depth",
                                          f"deeper centred variant preferred ({room.describe()})"))
        else:
            drop.add(g.id)
            rejected.append(Rejection(g.id, "depth",
                                      f"no room beyond the object ({room.describe()})"))
    if drop:
        survivors = [g for g in survivors if g.id not in drop]
        # The evidence counters were accumulated over the pre-drop survivors; recount so the
        # stats describe the ranked set (the selftest asserts exactly this consistency).
        for k in unknown:
            unknown[k] = sum(not g.score.terms[k].known for g in survivors)

    survivors.sort(key=lambda g: (g.score.total, g.id))
    ranked = tuple(ScoredGrasp(candidate=g.candidate, pose=g.pose, command=g.command,
                               score=g.score, clearance=g.clearance, rank=i)
                   for i, g in enumerate(survivors))
    pick = None
    if ranked:
        idx = sample_index([g.score.total for g in ranked], temperature,
                           np.random.default_rng(seed))
        pick = ranked[idx]

    stats = {"asset": record.name, "candidates": len(record.candidates),
             "selectable": len(ranked), "rejected": len(rejected),
             "weak_excluded": weak_excluded, "seat_blocked_dropped": seat_blocked_dropped,
             "quality_unknown": unknown["quality"], "region_unknown": unknown["region"],
             "containment_unknown": unknown["containment"],
             "distorted": distorted, "max_distortion": max_dist,
             "temperature": float(temperature)}
    if weak_excluded:
        log.info("%s: %d weak grasp option(s) (retreated) excluded from the default pool",
                 record.name, weak_excluded)
    if distorted:
        log.info("%s: %d pose(s) tilt-clamped, max distortion %.1f deg",
                 record.name, distorted, np.degrees(max_dist))
    if unknown["quality"] == len(ranked) and ranked:
        log.info("%s: physics quality unmeasured on every candidate — that term is neutral",
                 record.name)
    return SelectionResult(ranked=ranked, pick=pick, rejected=tuple(rejected),
                           buckets=tuple(sorted(keep)), stats=stats)
