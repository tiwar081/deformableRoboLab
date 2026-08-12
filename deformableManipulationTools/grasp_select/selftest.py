"""Self-test for grasp selection — the tests that ship with the API, since nothing consumes it yet.

    .venv/bin/python -m deformableManipulationTools.grasp_select.selftest [asset]

Synthetic fixtures first (a hand-built frame and candidates, so every expected answer is derivable
on paper), then one real merged record if the catalog has one — a failure therefore says whether the
RULE broke or only its application to real data. No sim, no newton: the API is pure geometry over a
stored record.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from ..grasp_library import (GraspCandidate, GraspRecord, ObjectFrame, QUALITY_FIELDS,
                             SEAT_BLOCKED_LABEL, WEAK_GRASP_LABEL, body_pose, grasp_transform,
                             has_grasps, is_weak, load_grasps)
from . import Obstacle, RobotState, SelectionResult, Weights, select_grasps
from .clearance import approach_clearance
from .projection import (HOME_GRASP_ROT, MAX_DISTORTION, TILT_MAX, command_rotation, geodesic_angle,
                         project_pose, rot_axis, rot_z, wrap_angle as wrap_pi,
                         _project_one)
from .scoring import NEUTRAL, Term, containment_term, quality_term, region_term, sample_index

_FAILED = []


def ok(what: str, cond, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        _FAILED.append(what)


# =================================================================================================
# Fixtures
# =================================================================================================
def _frame() -> ObjectFrame:
    """A canonical frame aligned with the world (identity), 20 x 10 x 5 cm."""
    return ObjectFrame(rotation=np.eye(3), translation=np.zeros(3),
                       extents=np.array([0.20, 0.10, 0.05]), axis_order=(0, 1, 2),
                       sign_rules=("skew",) * 3, ambiguous=False, drift=0.0)


def _cand(cid: str, position, approach, jaw, *, width=0.03, labels=(), notes="pad-seated +0.0 mm",
          seat_mode="span_flush", quality=None, quality_source=None) -> GraspCandidate:
    t = grasp_transform(position, approach, jaw)
    face = _frame().face_of(t[:3, 2])
    q = {k: None for k in QUALITY_FIELDS}
    if quality:
        q.update(quality)
    return GraspCandidate(id=cid, transform=t, width=width, face=face, source="selftest",
                          seat_mode=seat_mode, span=0.02, seat_depth=-0.02, quality=q, quality_source=quality_source,
                          labels=tuple(labels), notes=notes)


def _record(candidates, name="selftest") -> GraspRecord:
    return GraspRecord(name=name, kind="rigid_box", frame=_frame(), candidates=tuple(candidates),
                       generator="selftest")


class _Doc:
    """Minimal stand-in for a vlm_regions RegionDoc (the store's own class needs its file)."""
    def __init__(self, regions):
        self.regions = regions


class _Region:
    def __init__(self, label, center, radius=0.02, confidence=0.9):
        self.label, self.center, self.radius, self.confidence = label, center, radius, confidence


TOP = _cand("top", [0.0, 0.0, 0.0], [0, 0, -1], [0, 1, 0])          # straight down onto +z
SIDE = _cand("side", [0.0, 0.0, 0.0], [0, -1, 0], [0, 0, 1])        # horizontally into +y
BOTTOM = _cand("bottom", [0.0, 0.0, 0.0], [0, 0, 1], [0, 1, 0])     # up into -z (from below)


# =================================================================================================
# Projection — the policy that meets robot._ik_target_rot
# =================================================================================================
def check_projection() -> None:
    print("projection")
    home = HOME_GRASP_ROT
    ok("the home pose itself needs no yaw and no tilt",
       (lambda c: abs(c.yaw) < 1e-9 and abs(c.tilt) < 1e-9)(project_pose(home)))
    ok("the home pose is reproduced exactly",
       geodesic_angle(project_pose(home).rotation, home) < 1e-12)

    # Round-trip: every commandable pose must come back as itself, over a spread of the vocabulary.
    # "Itself" is up to the jaw flip: the projection is free to return the equivalent grasp with the
    # nearer wrist, and says so in jaw_flipped — so the comparison is against both representations.
    worst = 0.0
    for yaw in (-2.0, -0.4, 0.0, 0.7, 2.9):
        for tilt in (0.0, 0.2, 0.9, 1.2):
            for alpha in (0.0, 0.6, 2.4):
                r = command_rotation(yaw, tilt, (math.cos(alpha), math.sin(alpha), 0.0), home)
                c = project_pose(r)
                worst = max(worst, min(geodesic_angle(c.rotation, r),
                                       geodesic_angle(c.rotation, r @ np.diag([-1.0, -1.0, 1.0]))))
    # Tolerance 1e-6 rad, not 1e-12: the geodesic goes through arccos, whose slope is infinite at
    # 0, so a bit-exact rotation still reads back ~sqrt(eps) ≈ 2e-8 rad. That is the metric's
    # resolution near identity, not an error in the projection.
    ok("every (yaw, tilt, axis) round-trips to itself", worst < 1e-6,
       f"worst {math.degrees(worst):.2e} deg over 60 poses")

    # Coverage: arbitrary rotations, exactly representable while within the tilt cap.
    rng = np.random.default_rng(7)
    exact, clamped, worst_exact = 0, 0, 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        c = project_pose(r)
        if c.clamped:
            clamped += 1
        else:
            exact += 1
            worst_exact = max(worst_exact, min(geodesic_angle(c.rotation, r),
                                               geodesic_angle(c.rotation,
                                                              r @ np.diag([-1.0, -1.0, 1.0]))))
    ok("an unclamped projection is EXACT, not approximate", worst_exact < 1e-6,
       f"{exact} exact / {clamped} clamped of 200 random rotations")

    # The clamp's error is exactly the tilt it removed — this is why distortion is reportable.
    over = command_rotation(0.3, TILT_MAX + math.radians(20.0), (1.0, 0.0, 0.0), home)
    c = project_pose(over)
    ok("the clamp's distortion equals the tilt it removed",
       abs(c.distortion - math.radians(20.0)) < 1e-9,
       f"{math.degrees(c.distortion):.3f} deg")
    ok("the reported distortion IS the geodesic error of the commanded pose",
       abs(geodesic_angle(c.rotation, over) - c.distortion) < 1e-9)
    ok("a pose past MAX_DISTORTION is rejected, not silently bent", not c.accepted)
    ok("a rejected projection still reports its command and cost",
       c.rotation is not None and c.distortion > MAX_DISTORTION)

    # Jaw symmetry: the flipped grasp is the same contact. It is a YAW-only symmetry — the flip
    # rotates about the approach, which cannot change the tilt — so it buys wrist range, never
    # reachability. Both halves of that are asserted, because the tempting wrong belief (that a flip
    # can rescue an over-tilted pose) would make rejections look avoidable when they are not.
    flip = project_pose(HOME_GRASP_ROT @ np.diag([-1.0, -1.0, 1.0]))
    ok("the 180 deg jaw flip is recognised as the same grasp",
       flip.distortion == 0.0 and abs(flip.tilt) < 1e-9, flip.describe())
    tilted = command_rotation(0.0, math.radians(40.0), (1.0, 0.0, 0.0), home)
    branches = [_project_one(tilted, home, TILT_MAX, False, MAX_DISTORTION),
                _project_one(tilted @ np.diag([-1.0, -1.0, 1.0]), home, TILT_MAX, True,
                             MAX_DISTORTION)]
    ok("the flip never changes the tilt, only the yaw",
       abs(branches[0].tilt - branches[1].tilt) < 1e-9
       and abs(abs(wrap_pi(branches[0].yaw - branches[1].yaw)) - math.pi) < 1e-9,
       f"tilts {math.degrees(branches[0].tilt):.1f}/{math.degrees(branches[1].tilt):.1f} deg, "
       f"yaws {math.degrees(branches[0].yaw):.1f}/{math.degrees(branches[1].yaw):.1f} deg")
    ok("the two jaw representations of one grasp canonicalise to the SAME command",
       project_pose(tilted).yaw == project_pose(tilted @ np.diag([-1.0, -1.0, 1.0])).yaw)
    far_wrist = command_rotation(math.radians(179.0), math.radians(20.0), (1.0, 0.0, 0.0), home)
    ok("the flip is used to keep the commanded wrist yaw near zero",
       abs(project_pose(far_wrist).yaw) <= math.radians(1.0001)
       and project_pose(far_wrist).jaw_flipped, project_pose(far_wrist).describe())
    ok("a flip cannot rescue an over-tilted approach",
       not project_pose(command_rotation(0.0, math.radians(170.0), (1.0, 0.0, 0.0), home)).accepted)

    ok("straight-up (grasping from underneath) is rejected",
       not project_pose(command_rotation(0.0, math.pi, (1.0, 0.0, 0.0), home)).accepted)


# =================================================================================================
# Clearance
# =================================================================================================
def check_clearance() -> None:
    print("clearance")
    c = approach_clearance([0.4, 0.0, 0.80], [0, 0, -1], table_z=0.75, jaw_width=0.03)
    ok("a top-down grasp above the table has a clear corridor", c.ok, c.describe())
    c = approach_clearance([0.4, 0.0, 0.76], [0, 0, 1], table_z=0.75, jaw_width=0.03)
    ok("a grasp approached from BELOW is blocked by the table", not c.ok and c.blocker == "table",
       c.describe())
    box = Obstacle(name="bin", center=(0.4, 0.06, 0.80), half=(0.05, 0.02, 0.05))
    c = approach_clearance([0.4, 0.0, 0.80], [0, -1, 0], table_z=0.75, jaw_width=0.03,
                           obstacles=[box])
    ok("a corridor through a neighbouring object is blocked",
       not c.ok and c.blocker == "bin", c.describe())
    c = approach_clearance([0.4, 0.0, 0.80], [0, 1, 0], table_z=0.75, jaw_width=0.03,
                           obstacles=[box])
    ok("the same grasp from the other side is clear", c.ok, c.describe())
    ok("a yawed obstacle box rotates with its placement",
       Obstacle("b", (0, 0, 0), (0.10, 0.01, 0.01), yaw=math.pi / 2).contains([0.0, 0.05, 0.0])
       and not Obstacle("b", (0, 0, 0), (0.10, 0.01, 0.01)).contains([0.0, 0.05, 0.0]))


# =================================================================================================
# Scoring terms
# =================================================================================================
def check_scoring() -> None:
    print("scoring terms")
    ok("unmeasured physics quality degrades to neutral, flagged",
       quality_term(TOP).cost == NEUTRAL and not quality_term(TOP).known)
    held = _cand("held", [0, 0, 0], [0, 0, -1], [0, 1, 0], quality_source="newton_vbd",
                 quality={"object_in_gripper": 1.0, "object_motion_during_shaking_linear": 0.0})
    dropped = _cand("dropped", [0, 0, 0], [0, 0, -1], [0, 1, 0], quality_source="newton_vbd",
                    quality={"object_in_gripper": 0.0})
    ok("a measured, motionless grasp scores better than an unmeasured one",
       quality_term(held).cost < NEUTRAL and quality_term(held).known)
    ok("a grasp that dropped the object scores worst", quality_term(dropped).cost == 1.0)
    shaky = _cand("shaky", [0, 0, 0], [0, 0, -1], [0, 1, 0], quality_source="newton_vbd",
                  quality={"object_in_gripper": 1.0, "object_motion_during_shaking_linear": 0.02})
    ok("more motion during shaking costs more", quality_term(shaky).cost > quality_term(held).cost)

    doc = _Doc([_Region("handle", (0.08, 0.0, 0.0), radius=0.02, confidence=0.9)])
    on = _cand("on", [0.08, 0.0, 0.0], [0, 0, -1], [0, 1, 0])
    near = _cand("near", [0.095, 0.0, 0.0], [0, 0, -1], [0, 1, 0])
    off = _cand("off", [-0.08, 0.0, 0.0], [0, 0, -1], [0, 1, 0])
    edge = _cand("edge", [0.115, 0.0, 0.0], [0, 0, -1], [0, 1, 0])   # 15 mm outside the ball
    ok("a grasp on an annotated region scores best, and the ball is FLAT inside",
       region_term(on, doc).cost == region_term(near, doc).cost
       < region_term(edge, doc).cost < region_term(off, doc).cost,
       f"on=near={region_term(on, doc).cost:.2f} < edge={region_term(edge, doc).cost:.2f} "
       f"< off={region_term(off, doc).cost:.2f}")
    ok("the margin covers image-space slop rather than the whole object",
       region_term(near, doc).cost < 1.0 and region_term(off, doc).cost == 1.0)
    ok("a lower-confidence region is worth less",
       region_term(on, _Doc([_Region("h", (0.08, 0, 0), 0.02, 0.4)])).cost
       > region_term(on, doc).cost)
    ok("an asset with no regions is unknown, not penalised",
       region_term(on, None).cost == NEUTRAL and not region_term(on, None).known)
    ok("an empty region list is unknown too (the store records honest emptiness)",
       not region_term(on, _Doc([])).known)

    ok("a clamped-deep pose costs more than a span-flush one",
       containment_term(_cand("u", [0, 0, 0], [0, 0, -1], [0, 1, 0],
                              seat_mode="clamped_deep")).cost
       > containment_term(TOP).cost)
    ok("a retreated pose also fails full enclosure",
       containment_term(_cand("v", [0, 0, 0], [0, 0, -1], [0, 1, 0],
                              seat_mode="retreated")).cost == 1.0)
    ok("the schema field is read, not the prose",
       containment_term(_cand("w", [0, 0, 0], [0, 0, -1], [0, 1, 0], seat_mode="span_flush",
                              notes="pads CANNOT enclose")).cost == 0.0)
    ok("a candidate with no seat_mode is unknown, not assumed",
       not containment_term(_cand("x", [0, 0, 0], [0, 0, -1], [0, 1, 0], seat_mode=None)).known)


# =================================================================================================
# Sampling
# =================================================================================================
def check_sampling() -> None:
    print("sampling")
    costs = [0.9, 0.1, 0.5]
    ok("temperature 0 is argmin", sample_index(costs, 0.0) == 1)
    ok("temperature 0 breaks ties at the lowest index", sample_index([0.2, 0.2], 0.0) == 0)
    rng = np.random.default_rng(3)
    draws = [sample_index(costs, 0.05, rng) for _ in range(300)]
    ok("a low temperature stays near the argmin", draws.count(1) > 280, f"{draws.count(1)}/300")
    rng = np.random.default_rng(3)
    hot = [sample_index(costs, 0.5, rng) for _ in range(300)]
    ok("a high temperature explores", len(set(hot)) == 3 and hot.count(1) < 280,
       f"{[hot.count(i) for i in range(3)]}")
    ok("sampling is reproducible from a seed",
       [sample_index(costs, 0.3, np.random.default_rng(11)) for _ in range(5)]
       == [sample_index(costs, 0.3, np.random.default_rng(11)) for _ in range(5)])


# =================================================================================================
# Pool assembly — the candidate statuses (grasp-library.md "Candidate statuses", 2026-08-11)
# =================================================================================================
def check_pool() -> None:
    print("pool assembly (candidate statuses)")
    # Same pose as TOP so any rank difference is the status/scoring, never the geometry. One weak
    # candidate pre-migration (seat_mode only) and one post-migration (merge-stamped label) — the
    # filter keys on is_weak (the seat_mode fact), so both must behave identically.
    weak = _cand("weak", [0.0, 0.0, 0.0], [0, 0, -1], [0, 1, 0], seat_mode="retreated")
    stamped = _cand("weak_stamped", [0.0, 0.0, 0.0], [0, 0, -1], [0, 1, 0],
                    seat_mode="retreated", labels=(WEAK_GRASP_LABEL,))
    blocked = _cand("blocked", [0.0, 0.0, 0.0], [0, 0, -1], [0, 1, 0],
                    labels=(SEAT_BLOCKED_LABEL,))
    rec = _record([TOP, weak, stamped, blocked])
    robot = RobotState(base_pos=(0.0, 0.0, 0.75), reach_max=None, table_z=0.75)
    place = body_pose([0.4, 0.0, 0.80], 0.0)

    res = select_grasps(rec, place, robot)
    ok("weak (retreated) candidates are excluded from the default pool",
       [g.id for g in res.ranked] == ["top"], f"{[g.id for g in res.ranked]}")
    ok("the filter keys on seat_mode, so stamped and un-stamped weak candidates filter alike",
       {r.id for r in res.rejected if r.stage == "status"} >= {"weak", "weak_stamped"},
       f"{[(r.id, r.stage) for r in res.rejected]}")
    ok("excluded candidates are still accounted for, with a reason",
       len(res.ranked) + len(res.rejected) == len(rec.candidates))
    ok("the exclusion is counted in the stats",
       res.stats["weak_excluded"] == 2 and res.stats["seat_blocked_dropped"] == 1)
    ok("the report says how many weak options were excluded, and why the pool shrank",
       "2 weak grasp option(s)" in res.report() and "include_weak" in res.report())

    inc = select_grasps(rec, place, robot, include_weak=True)
    ok("include_weak admits weak options into the ranking",
       {"weak", "weak_stamped"} <= {g.id for g in inc.ranked} and inc.stats["weak_excluded"] == 0,
       f"{[g.id for g in inc.ranked]}")
    ok("an included weak option ranks naturally, below the legitimate pose at the same spot",
       inc.ranked[0].id == "top", f"{[g.id for g in inc.ranked]}")
    ok("seat_blocked is dropped even with include_weak (nothing to command; belt over braces)",
       "blocked" not in {g.id for g in inc.ranked}
       and "blocked" in {r.id for r in inc.rejected if r.stage == "status"})


# =================================================================================================
# End to end
# =================================================================================================
def check_pipeline() -> None:
    print("pipeline")
    rec = _record([TOP, SIDE, BOTTOM])
    robot = RobotState(base_pos=(0.0, 0.0, 0.75), reach_max=None, table_z=0.75)
    place = body_pose([0.4, 0.0, 0.80], 0.0)
    res = select_grasps(rec, place, robot)
    ids = [g.id for g in res.ranked]
    ok("every candidate is accounted for as ranked or rejected",
       len(res.ranked) + len(res.rejected) == len(rec.candidates),
       f"{len(res.ranked)} + {len(res.rejected)}")
    ok("the grasp from underneath is pruned by its face bucket",
       "bottom" in [r.id for r in res.rejected if r.stage == "bucket"], f"{ids}")
    ok("the full ranked set is returned, not just the pick", len(res.ranked) == 2, f"{ids}")
    ok("ranks are dense and in cost order",
       [g.rank for g in res.ranked] == list(range(len(res.ranked)))
       and all(res.ranked[i].score.total <= res.ranked[i + 1].score.total
               for i in range(len(res.ranked) - 1)))
    ok("temperature 0 picks the top of the ranking", res.pick is res.ranked[0])
    ok("the top-down grasp beats the side grasp on awkwardness",
       res.ranked[0].id == "top"
       and res.ranked[0].score.terms["awkwardness"].cost
       < res.ranked[1].score.terms["awkwardness"].cost)
    ok("the pick carries executor-ready waypoint kwargs",
       set(res.pick.command.waypoint_kwargs) == {"yaw", "tilt", "tilt_axis"})
    ok("the world pose is the placement composed with the stored pose",
       np.allclose(res.by_id("top").position, [0.4, 0.0, 0.80], atol=1e-9),
       f"{np.round(res.by_id('top').position, 4).tolist()}")
    ok("missing evidence is counted, not hidden",
       res.stats["quality_unknown"] == len(res.ranked)
       and res.stats["region_unknown"] == len(res.ranked))

    # A yawed placement must rotate the commanded yaw with it, one for one.
    turned = select_grasps(rec, body_pose([0.4, 0.0, 0.80], 0.5), robot)
    ok("a yawed placement yaws the command by the same angle",
       abs(abs(turned.by_id("top").command.yaw - res.by_id("top").command.yaw) - 0.5) < 1e-9,
       f"{turned.by_id('top').command.yaw:.3f} vs {res.by_id('top').command.yaw:.3f}")

    # Upside down: the top grasp's face now points into the table and must be pruned.
    flip = np.eye(4)
    flip[:3, :3] = np.diag([1.0, -1.0, -1.0])
    flip[:3, 3] = [0.4, 0.0, 0.80]
    upside = select_grasps(rec, flip, robot)
    ok("flipping the object flips which grasps survive",
       "top" in [r.id for r in upside.rejected] and "bottom" in [g.id for g in upside.ranked],
       f"{[g.id for g in upside.ranked]}")

    # Reach and obstacles remove candidates with a stated reason rather than silently.
    far = select_grasps(rec, place, RobotState(base_pos=(0.0, 0.0, 0.75), reach_max=0.10,
                                               table_z=0.75))
    ok("an out-of-reach object is rejected at the reach stage",
       far.ranked == () and all(r.stage == "reach" for r in far.rejected if r.stage != "bucket"))
    blocked = select_grasps(rec, place, robot,
                            obstacles=[Obstacle("wall", (0.4, 0.0, 0.90), (0.10, 0.10, 0.01))])
    ok("an obstacle over the object removes the top-down grasp",
       "top" in [r.id for r in blocked.rejected if r.stage == "clearance"],
       f"{[(r.id, r.stage) for r in blocked.rejected]}")

    # Weights are the caller's, and moving them must move the ranking.
    doc = _Doc([_Region("handle", (0.0, -0.05, 0.0), radius=0.02, confidence=1.0)])
    biased = select_grasps(rec, place, robot, regions=doc,
                           weights=Weights(quality=0.0, region=1.0, awkwardness=0.0,
                                           containment=0.0))
    ok("with only the region term weighted, the grasp on the region wins",
       biased.ranked[0].id == "side", f"{[g.id for g in biased.ranked]}")
    ok("selection is deterministic at temperature 0",
       select_grasps(rec, place, robot).pick.id == res.pick.id)
    ok("a report is printable and mentions the rejections",
       "rejected" in res.report() and "evidence" in res.report())


def check_real_record(asset: str) -> None:
    print(f"real record: {asset}")
    if not has_grasps(asset):
        print("  --    no merged record; run the passes first")
        return
    rec = load_grasps(asset, use_cache=False)
    robot = RobotState(base_pos=(0.0, 0.9, 0.75))
    res = select_grasps(rec, {"x": 0.45, "y": 0.45, "z": 0.78, "yaw": 0.4}, robot)
    ok("some candidate survives a normal tabletop placement", len(res.ranked) > 0,
       f"{len(res.ranked)} of {len(rec.candidates)}")
    ok("every candidate is accounted for",
       len(res.ranked) + len(res.rejected) == len(rec.candidates))
    ok("no survivor was silently distorted",
       all(g.command.distortion <= MAX_DISTORTION for g in res.ranked))
    ok("every survivor's corridor clears the table",
       all(g.clearance.ok for g in res.ranked))
    ok("costs are finite and in range",
       all(0.0 <= g.score.total <= 1.0 and np.isfinite(g.score.total) for g in res.ranked))
    # Since the 2026-08-08 full-catalog shake, records carry measured quality — but the store is
    # allowed to be mid-migration here, so assert the COUNTER is honest, not a fixed coverage.
    ok("the quality-unknown stat counts exactly the ranked poses with no measured quality",
       res.stats["quality_unknown"] == sum(not g.score.terms["quality"].known
                                           for g in res.ranked))
    ok("no weak (retreated) candidate reaches the default ranking",
       all(not is_weak(g.candidate) for g in res.ranked))
    ok("the weak exclusions are counted against the record's own retreated population",
       res.stats["weak_excluded"] == sum(is_weak(c) for c in rec.candidates
                                         if SEAT_BLOCKED_LABEL not in c.labels),
       f"{res.stats['weak_excluded']} excluded, "
       f"{res.stats['seat_blocked_dropped']} seat_blocked dropped")
    hot = select_grasps(rec, {"x": 0.45, "y": 0.45, "z": 0.78, "yaw": 0.4}, robot,
                        temperature=0.4, seed=5)
    ok("a hot sample stays inside the ranked set",
       hot.pick.id in {g.id for g in hot.ranked})
    print("  " + res.report().splitlines()[0])
    for line in res.report().splitlines()[1:4]:
        print("  " + line.strip())


def main() -> None:
    asset = sys.argv[1] if len(sys.argv) > 1 else "mug"
    check_projection()
    check_clearance()
    check_scoring()
    check_sampling()
    check_pool()
    check_pipeline()
    check_real_record(asset)
    print(f"\n{'FAILED: ' + ', '.join(_FAILED) if _FAILED else 'all checks passed'}")
    raise SystemExit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
