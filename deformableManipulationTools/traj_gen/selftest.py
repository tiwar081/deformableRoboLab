"""Executable invariants for the trajectory stage (no GPU, no LLM, no rollout).

    .venv/bin/python -m deformableManipulationTools.traj_gen.selftest

Covers: Bezier collision-driven control-point insertion; physics-tiered re-rank + weighted
sampling (retreated exclusion is grasp_select's, asserted there); goal resolution per predicate
incl. the free-spot nudge; full plan assembly (monotonic times, window containment, pre-shape
contract, derived force bounds, JSON round-trip); LLM verdict application (switch + clamped
adjust)."""
from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import numpy as np

from ..grasp_library import PREGRASP_MARGIN, grasp_transform
from ..grasp_select.clearance import Obstacle
from ..grasp_select.projection import project_pose
from ..params import TABLE
from . import curve, llm, policy, selection

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        sys.exit(1)
    PASS += 1


# ---------------------------------------------------------------------------------------------
def test_curve() -> None:
    print("curve: bezier routing")
    fld = curve.CollisionField(obstacles=(), floor_z=0.0, inflate=0.0)
    leg = curve.route((0.0, 0.0, 0.3), (0.6, 0.0, 0.3), fld)
    check("clear leg stays straight", leg.clear and leg.inserted == 0)
    check("clear leg endpoint exact", np.allclose(leg.points[-1], (0.6, 0.0, 0.3), atol=1e-9))

    wall = Obstacle(name="wall", center=(0.3, 0.0, 0.2), half=(0.05, 0.4, 0.2))
    fld = curve.CollisionField(obstacles=(wall,), floor_z=0.05, inflate=0.02)
    leg = curve.route((0.0, 0.0, 0.3), (0.6, 0.0, 0.3), fld)
    check("collision inserts control points", leg.clear and leg.inserted >= 1,
          f"inserted={leg.inserted} iters={leg.iterations} blockers={leg.blockers}")
    check("routed spline clears the box", not any(fld.hit(p) for p in leg.points[1:-1]))
    check("blocker recorded", "wall" in leg.blockers)

    wps = curve.leg_waypoints(leg, 2.0, 0.25)
    ts = [t for t, _, _ in wps]
    check("leg waypoints monotonic", all(b > a for a, b in zip(ts, ts[1:])) and ts[0] > 2.0)
    check("leg waypoints end at endpoint", np.allclose(wps[-1][1], (0.6, 0.0, 0.3), atol=1e-6))
    check("interior waypoints via-marked", all(v for _, _, v in wps[:-1]) and not wps[-1][2])


# ---------------------------------------------------------------------------------------------
def _fake_ranked(cid: str, held, score: float, pose=None) -> selection.RankedGrasp:
    cand = SimpleNamespace(id=cid, width=0.05, span=0.03, seat_mode="span_flush", source="test",
                           quality={"object_in_gripper": held} if held is not None else {},
                           quality_source="shake_validate" if held is not None else None,
                           labels=(), face="+z")
    pose = pose if pose is not None else grasp_transform((0.1, -0.4, 0.12), (0, 0, -1), (1, 0, 0))
    cmd = project_pose(np.asarray(pose)[:3, :3])
    g = SimpleNamespace(candidate=cand, pose=np.asarray(pose, dtype=float), command=cmd,
                        score=SimpleNamespace(total=score, describe=lambda: ""),
                        clearance=None, id=cid,
                        position=np.asarray(pose, dtype=float)[:3, 3])
    tier = selection.physics_tier(cand)
    return selection.RankedGrasp(grasp=g, tier=tier, cost=score + selection.TIER_PENALTY[tier])


def test_selection() -> tuple:
    print("selection: physics re-rank + weighted sampling")
    held = _fake_ranked("a_held", 1.0, 0.60)          # worse score, but measured HELD
    untested = _fake_ranked("b_untested", None, 0.30)  # best raw score, no physics
    dropped = _fake_ranked("c_dropped", 0.0, 0.10)     # best raw score of all, measured DROP
    check("tiers assigned", held.tier == 0 and untested.tier == 1 and dropped.tier == 2)
    ranked = sorted([dropped, untested, held], key=lambda r: (r.cost, r.id))
    ranking = selection.TaskRanking(ranked=ranked)
    check("held outranks better-scored untested/dropped", ranking.ranked[0].id == "a_held",
          " > ".join(r.id for r in ranking.ranked))

    rng = np.random.default_rng(0)
    check("T=0 draw is the best", selection.draw(ranking, temperature=0.0, rng=rng).id == "a_held")
    check("exclusion works",
          selection.draw(ranking, temperature=0.0, rng=rng, exclude=["a_held"]).id != "a_held")
    counts: dict[str, int] = {}
    for _ in range(400):
        cid = selection.draw(ranking, temperature=0.15, rng=rng).id
        counts[cid] = counts.get(cid, 0) + 1
    check("sampling prefers better cost", counts.get("a_held", 0) > counts.get("c_dropped", 0),
          str(counts))
    return ranking, held


# ---------------------------------------------------------------------------------------------
def _fixture() -> tuple:
    from agentic_pipeline import geometry
    placement = geometry.default_placement()
    by_name = {
        "tomato_soup_can": {"kind": "ycb_mesh", "dims": [0.068, 0.068, 0.102],
                            "config": {"target_mass": 0.349, "mu": 0.4,
                                       "usd_subpath": "ycb/tomato_soup_can.usd"}},
        "bucket": {"kind": "ycb_mesh", "dims": [0.27, 0.27, 0.25],
                   "config": {"usd_subpath": "vomp/bucket.usd"}, "container": True},
    }
    scene = {"name": "selftest", "objects": [
        {"name": "tomato_soup_can", "x": 0.10, "y": -0.45, "yaw_deg": 0.0},
        {"name": "bucket", "x": 0.12, "y": -0.32, "yaw_deg": 0.0},
    ]}
    task = {"name": "t", "goal": {"predicate": "object_in_container",
                                  "params": {"object": "tomato_soup_can", "container": "bucket"}},
            "robot_placement": placement, "instruction": {"default": "put the can in the bucket"}}
    return scene, task, placement, by_name


def test_policy(ranking, held) -> dict:
    print("policy: goal resolution + plan assembly")
    scene, task, placement, by_name = _fixture()

    place = policy.resolve_place(scene, task, placement, by_name, "tomato_soup_can", (0.10, -0.45))
    check("drop-in lands on the container mouth",
          np.allclose(place["xy"], (0.12, -0.32), atol=1e-9)
          and abs(place["surface_z"] - (TABLE.top_z + 0.25)) < 1e-9, str(place))

    t2 = {**task, "goal": {"predicate": "object_left_of",
                           "params": {"object": "tomato_soup_can", "reference": "bucket"}}}
    p2 = policy.resolve_place(scene, t2, placement, by_name, "tomato_soup_can", (0.10, -0.45))
    check("beside placement offsets from the reference",
          0.05 < math.hypot(p2["xy"][0] - 0.12, p2["xy"][1] + 0.32) < 0.40, str(p2["xy"]))
    check("beside placement is on the tabletop", abs(p2["surface_z"] - TABLE.top_z) < 1e-9)

    t3 = {**task, "goal": {"predicate": "cloth_folded", "params": {"object": "tomato_soup_can"}}}
    try:
        policy.resolve_place(scene, t3, placement, by_name, "tomato_soup_can", (0.1, -0.45))
        check("unsupported predicate rejected", False)
    except policy.PlanError:
        check("unsupported predicate rejected", True)

    pick = policy.pick_from_ranked(held)
    # place the fake grasp over the actual can
    pose = grasp_transform((0.10, -0.45, TABLE.top_z + 0.08), (0, 0, -1), (1, 0, 0))
    from dataclasses import replace
    pick = replace(pick, pose=np.asarray(pose, dtype=float))
    ok, _ = policy.pads_clear_table(pick)
    check("top-down pad sweep clears the table", ok)
    # a side grasp jawing across a flat 30 mm-tall can: the lower pad must dip below the table
    side = grasp_transform((0.10, -0.45, TABLE.top_z + 0.016), (0, -1, 0), (0, 0, 1))
    low = replace(pick, pose=np.asarray(side, dtype=float), width=0.030)
    ok, zmin = policy.pads_clear_table(low)
    check("height-jaw grasp on a flat can is rejected", not ok,
          f"lowest pad z {zmin:.3f} vs table {TABLE.top_z:.3f}")
    plan = policy.plan_pick_place(scene, task, placement, by_name, pick,
                                  target_bottom_dz=-0.051, attempt=0)
    ts = [w["t"] for w in plan.waypoints]
    check("waypoint times strictly increasing", all(b > a for a, b in zip(ts, ts[1:])))
    w = plan.grasp_window
    check("grasp window ordered", w["close_start"] < w["close_end"] < w["release_start"]
          < w["release_end"] < ts[-1])
    check("pre-shape honors the library contract",
          abs(w["preshape_width"] - (pick.width + PREGRASP_MARGIN)) < 1e-6)
    check("force target derived within bounds", 1.0 <= w["force_target"] <= 40.0,
          f"{w['force_target']} N")
    check("routing legs all clear", all(l["clear"] for l in plan.routing["legs"]))
    hold_wps = [wp for wp in plan.waypoints if w["close_start"] <= wp["t"] <= w["release_start"]]
    grasp_xy = (0.10, -0.45)
    check("hand is over the grasp through the close",
          any(abs(wp["pos"][0] - grasp_xy[0]) < 1e-6 and abs(wp["pos"][1] - grasp_xy[1]) < 1e-6
              for wp in hold_wps))
    d = plan.to_dict()
    check("traj dict JSON round-trips", json.loads(json.dumps(d))["target"] == "tomato_soup_can")
    check("target label + ordinal resolved",
          d["target_label"] == "tomato_soup_can" and d["target_ordinal"] == 0)
    return {"plan": plan, "pick": pick, "ranking": ranking}


# ---------------------------------------------------------------------------------------------
def test_llm(fix: dict) -> None:
    print("llm: verdict application")
    ranking, pick = fix["ranking"], fix["pick"]
    v = {"action": "switch", "candidate_id": "b_untested", "rationale": "r"}
    nxt = llm.apply_verdict(v, pick, ranking, tried_ids=["a_held"])
    check("switch picks the named alternative", nxt is not None and nxt.id == "b_untested")
    v = {"action": "switch", "candidate_id": "a_held", "rationale": "r"}    # already tried
    nxt = llm.apply_verdict(v, pick, ranking, tried_ids=["a_held"])
    check("switch never re-tries a tried id", nxt is not None and nxt.id != "a_held")

    v = {"action": "adjust", "adjust": {"shift_mm": [5.0, 0.0, 10.0], "width_mm": 60.0,
                                        "force_target_n": 12.0}, "rationale": "r"}
    nxt = llm.apply_verdict(v, pick, ranking, tried_ids=[pick.id])
    dp = nxt.position - pick.position
    expect = np.asarray(pick.pose)[:3, :3] @ np.array([0.005, 0.0, 0.010])
    check("adjust shifts in the grasp frame", np.allclose(dp, expect, atol=1e-9), str(dp))
    check("adjust records itself", nxt.adjusted is not None
          and nxt.adjusted["force_target_n"] == 12.0 and abs(nxt.width - 0.06) < 1e-9)

    # package assembly must not require images
    content = llm.build_package(task=_fixture()[1], plan=fix["plan"].to_dict(),
                                evaluation={"failure": "never_held", "held": False,
                                            "carried": False, "close_drift": 0.001,
                                            "lift_rise": 0.0, "final_pos": [0, 0, 0],
                                            "final_z_above_table": 0.0},
                                pick=pick, alternatives=ranking.ranked, history=[],
                                scene_png=None, grasp_png=None)
    check("retry package builds without images",
          len(content) == 1 and content[0]["type"] == "text"
          and "ALTERNATIVE CANDIDATES" in content[0]["text"])


def main() -> None:
    test_curve()
    ranking, held = test_selection()
    fix = test_policy(ranking, held)
    test_llm(fix)
    print(f"traj_gen selftest: all {PASS} checks passed")


if __name__ == "__main__":
    main()
