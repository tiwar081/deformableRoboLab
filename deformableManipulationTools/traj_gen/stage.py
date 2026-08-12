"""The trajectory stage orchestrator: pipeline run dir -> executed, evaluated trajectory.

    report = generate_trajectory("outputs/agenticPipeline/<run>")

Steps, per docs/trajPipeline/trajectory-generation.md:

1. read ``scene.json`` / ``task.json`` (settled poses + robot placement), gate the goal predicate
   and the target's grasp-library support;
2. ONLINE grasp selection (:mod:`.selection`): ``grasp_select`` pool/prune/clearance/projection/
   score at the object's actual placement, physics-tiered re-rank, score-weighted random draw;
3. plan (:mod:`.policy`): Bezier legs + straight validated approach/lift, collision-driven control
   points, derived force target, pre-shaped aperture -> ``traj.json`` (the generated demo file
   picks it up through ``agentic_pipeline.build.demo_from_dir``);
4. headless rollout (:mod:`.rollout`) measures the grasp and the goal;
5. on a failed grasp, the LLM loop (:mod:`.llm`) gets ``MAX_LLM_ATTEMPTS`` (2) corrected rollouts
   (switch candidate / adjust pose+width+force); still failing -> the trajectory is ABORTED and the
   report says so (the failure stays visible; nothing is scripted around it).

Every attempt's plan + evaluation goes into ``traj_result.json`` in the run dir; ``traj.json``
always holds the LAST executed plan, so re-rendering the demo reproduces exactly what was measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..grasp_library import (GraspSchemaError, body_pose, is_unusable, load_grasps,
                             UNSUPPORTED_KINDS)
from ..grasp_select import RobotState
from ..params import TABLE
from . import llm, policy, rollout, selection, viz
from .policy import PlanError

DEFAULT_TEMPERATURE = selection.DEFAULT_TEMPERATURE


class TrajectoryAborted(RuntimeError):
    """The stage could not produce a working trajectory; details in the report."""


def _settled_body_z(run_dir: Path, label: str, ordinal: int, fallback: float) -> float:
    """The target's settled body z from the pipeline's settle report (label + duplicate ordinal),
    else the rest-pose fallback."""
    p = run_dir / "pipeline.json"
    if p.exists():
        settle = (json.loads(p.read_text()).get("settle") or {})
        pool = [s for s in settle.get("settled", []) if s.get("label") == label]
        if ordinal < len(pool):
            return float(pool[ordinal]["z"])
    return fallback


def _asset_world_vertices(asset, tgt: dict, body_z: float) -> np.ndarray:
    import math
    w = body_pose((float(tgt["x"]), float(tgt["y"]), body_z),
                  math.radians(float(tgt.get("yaw_deg", 0.0))))
    v = np.asarray(asset.vertices, dtype=float)
    return v @ w[:3, :3].T + w[:3, 3]


def generate_trajectory(run_dir: Path | str, *, device: str = "cuda:0",
                        llm_attempts: int = llm.MAX_LLM_ATTEMPTS, seed: int = 0,
                        temperature: float = DEFAULT_TEMPERATURE, model: str | None = None,
                        verbose: bool = True) -> dict:
    """Run the whole stage on one pipeline run dir. Returns the report (= ``traj_result.json``)."""
    run_dir = Path(run_dir)
    scene = json.loads((run_dir / "scene.json").read_text())
    task = json.loads((run_dir / "task.json").read_text())
    demo_py = next(iter(sorted(run_dir.glob("pipeline_*.py"))), None)
    if demo_py is None:
        raise FileNotFoundError(f"no pipeline demo file in {run_dir}")
    placement = task.get("robot_placement") or {}
    report: dict = {"run_dir": str(run_dir), "task": task.get("name"),
                    "goal": task.get("goal"), "attempts": [], "ok": False}

    def _abort(reason: str) -> dict:
        report["aborted"] = reason
        (run_dir / "traj_result.json").write_text(json.dumps(report, indent=1))
        if verbose:
            print(f"[trajGen] ABORTED: {reason}")
        return report

    # ---- gate: goal + target support ----
    goal = task.get("goal") or {}
    target = (goal.get("params") or {}).get("object")
    if not target:
        return _abort("task goal names no object")
    if goal.get("predicate") not in policy.SUPPORTED_GOALS:
        return _abort(f"goal predicate {goal.get('predicate')!r} not executable "
                      f"(deformable-shape goals need a different mechanism)")
    from agentic_pipeline.scene_generator import catalog_by_name, load_catalog
    by_name = catalog_by_name(load_catalog())
    kind = by_name.get(target, {}).get("kind")
    if kind in UNSUPPORTED_KINDS or kind in ("cloth",):
        return _abort(f"target {target!r} kind {kind!r} is outside the grasp library's scope")
    try:
        record = load_grasps(target)
    except (FileNotFoundError, GraspSchemaError) as exc:
        return _abort(f"no usable grasp record for {target!r}: {exc}")
    if is_unusable(record) or not record.candidates:
        return _abort(f"grasp record for {target!r} is "
                      f"{'UNUSABLE' if is_unusable(record) else 'out of reach (empty)'}")

    # ---- placement of the target + the selection inputs ----
    from deformableManipulationTools.grasp_passes.catalog import load_asset
    asset = load_asset(target)
    t_idx = policy.object_indices(scene, target)
    if not t_idx:
        return _abort(f"target {target!r} not present in the scene")
    # With duplicate instances, manipulate one that does not already satisfy the goal
    # (e.g. two apples, one already in the target bowl).
    chosen = policy.choose_target_index(scene, task, by_name)
    tgt = scene["objects"][chosen]
    bottom_dz = float(np.asarray(asset.vertices)[:, 2].min())
    from agentic_pipeline.scene_gen import _object_label
    label = _object_label(tgt)
    ordinal = sum(1 for o in scene["objects"][:chosen] if _object_label(o) == label)
    body_z = _settled_body_z(run_dir, label, ordinal, TABLE.top_z - bottom_dz)
    import math
    world_from_body = body_pose((float(tgt["x"]), float(tgt["y"]), body_z),
                                math.radians(float(tgt.get("yaw_deg", 0.0))))
    base = placement.get("base") or (0.0, 0.0, 0.0)
    robot = RobotState(base_pos=tuple(float(v) for v in base))
    obstacles = policy.scene_obstacles(scene, by_name, exclude_indices=t_idx)

    ranking = selection.rank_for_task(record, world_from_body, robot, obstacles=obstacles)
    report["selection"] = {"candidates": len(record.candidates), "selectable": len(ranking),
                           "tiers": {str(k): v for k, v in ranking.tiers().items()},
                           "weak_excluded": ranking.selection.stats.get("weak_excluded", 0),
                           "seat_blocked_dropped":
                               ranking.selection.stats.get("seat_blocked_dropped", 0)}
    if verbose:
        print(f"[trajGen] {ranking.report()}")
    if not len(ranking):
        return _abort(f"no selectable grasp for {target!r} at this placement "
                      f"({len(ranking.selection.rejected)} rejected)")

    rng = np.random.default_rng(seed)
    tried_ids: list[str] = []
    history: list[dict] = []
    scene_png = run_dir / "scene_overview.png"

    # ---- arm-feasibility gate (reach.py): IK-verify a pick before spending a rollout on it ----
    from .reach import checker_for_placement
    arm = checker_for_placement(placement)
    arm_rejected: list[dict] = []

    def _arm_ok(p: policy.PickSpec) -> bool:
        # Cheapest first: a jaw sweep that dips below the tabletop cannot close on a resting
        # object whatever the arm does (the table-less shake rig and the corridor check are both
        # blind to it — measured, see policy.pad_lowest_z).
        ok, zmin = policy.pads_clear_table(p)
        if not ok:
            arm_rejected.append({"id": p.id, "pose": "pad_sweep",
                                 "below_table_mm": round((TABLE.top_z - zmin) * 1000, 1)})
            if verbose:
                print(f"[trajGen] {p.id} rejected: pad sweep dips "
                      f"{(TABLE.top_z - zmin) * 1000:.0f} mm below the tabletop")
            return False
        pre = p.position - 0.10 * p.approach
        for name, pos in (("grasp", p.position), ("pregrasp", pre)):
            ok, err = arm.pose_ok(pos, p.yaw, p.tilt, p.tilt_axis)
            if not ok:
                arm_rejected.append({"id": p.id, "pose": name, "tcp_err": round(err, 4)})
                if verbose:
                    print(f"[trajGen] arm cannot hold {p.id} ({name} misses by "
                          f"{err * 1000:.0f} mm) — rejected before rollout")
                return False
        return True

    # A candidate is only PLANNABLE if the whole waypoint path solves with the EXECUTOR'S OWN
    # path IK (branch-consistent chained solve — a per-pose ladder can accept a pose the chained
    # solve then misses by 20 cm, measured). Critical waypoints (grasp approach through release)
    # must hit tighter than transit knots.
    PATH_TOL_CRITICAL, PATH_TOL_TRANSIT = 0.015, 0.04

    def _plan_checked(p: policy.PickSpec, attempt: int):
        """(plan, None) when the pick plans AND its full path is arm-executable, else (None, why)."""
        try:
            plan = policy.plan_pick_place(scene, task, placement, by_name, p,
                                          target_bottom_dz=bottom_dz, attempt=attempt)
        except PlanError as exc:
            return None, str(exc)
        errs = arm.path_errors(plan.waypoints, plan.grasp_window)
        w0 = float(plan.grasp_window["close_start"]) - 1.5
        w1 = float(plan.grasp_window["release_end"]) + 0.2
        bad = [e for e in errs
               if e["err"] > (PATH_TOL_CRITICAL if w0 <= e["t"] <= w1 else PATH_TOL_TRANSIT)]
        if bad:
            worst = max(bad, key=lambda e: e["err"])
            return None, (f"executor path IK misses {len(bad)} waypoint(s), worst "
                          f"{worst['err'] * 1000:.0f} mm at t={worst['t']:.1f}s")
        return plan, None

    def _draw_planned(attempt: int):
        """Draw candidates (without replacement) until one plans + path-checks; None = exhausted."""
        excluded = list(tried_ids) + [r["id"] for r in arm_rejected]
        while True:
            drawn = selection.draw(ranking, temperature=temperature, rng=rng, exclude=excluded)
            if drawn is None:
                return None, None
            p = policy.pick_from_ranked(drawn)
            if not _arm_ok(p):                     # cheap per-pose pre-filter
                excluded.append(p.id)
                continue
            plan, why = _plan_checked(p, attempt)
            if plan is None:
                arm_rejected.append({"id": p.id, "pose": "path", "reason": why})
                if verbose:
                    print(f"[trajGen] {p.id} rejected at planning: {why}")
                excluded.append(p.id)
                continue
            return p, plan

    pick, plan = _draw_planned(0)
    if pick is None:
        return _abort("no arm-executable grasp among the selectable candidates "
                      f"({len(arm_rejected)} rejected by the IK/plan gate)")

    for attempt in range(1 + llm_attempts):
        tried_ids.append(pick.id)
        (run_dir / "traj.json").write_text(json.dumps(plan.to_dict(), indent=1))
        if verbose:
            w = plan.grasp_window
            print(f"[trajGen] attempt {attempt}: candidate {pick.id} "
                  f"(tier {pick.tier}, seat {pick.seat_mode}, width {pick.width * 1000:.0f} mm, "
                  f"adjusted={pick.adjusted is not None}) — force {w['force_target']} N, "
                  f"pre-shape {w['preshape_width'] * 1000:.0f} mm, "
                  f"{plan.num_frames} frames, routing {plan.routing['legs']}")
        evaluation = rollout.run_rollout(demo_py, device=device, verbose=verbose,
                                         log_path=run_dir / f"rollout_a{attempt}.log")
        attempt_row = {"attempt": attempt, "candidate_id": pick.id,
                       "adjusted": pick.adjusted, "plan_pick": plan.pick,
                       "evaluation": evaluation}
        report["attempts"].append(attempt_row)
        if verbose:
            print(f"[trajGen] attempt {attempt} outcome: failure={evaluation.get('failure')} "
                  f"held={evaluation.get('held')} carried={evaluation.get('carried')} "
                  f"goal={evaluation.get('goal', {}).get('ok')} "
                  f"({evaluation.get('goal', {}).get('detail')})")
        if evaluation.get("held") and evaluation.get("carried"):
            report["ok"] = True
            report["final"] = {"candidate_id": pick.id, "attempt": attempt,
                               "goal": evaluation.get("goal"),
                               "place_xy_err": evaluation.get("place_xy_err")}
            break

        if attempt == llm_attempts:
            report["aborted"] = (f"grasp failed on all {1 + llm_attempts} attempts "
                                 f"(last failure: {evaluation.get('failure')})")
            break

        # ---- the LLM retry ----
        history.append({"attempt": attempt, "candidate_id": pick.id, "adjusted": pick.adjusted,
                        "failure": evaluation.get("failure"),
                        "close_drift": evaluation.get("close_drift"),
                        "lift_rise": evaluation.get("lift_rise")})
        grasp_png = run_dir / f"grasp_attempt_{attempt}.png"
        try:
            verts_w = _asset_world_vertices(asset, tgt, body_z)
            track = {k: np.asarray(v) for k, v in
                     (evaluation.get("track_samples") or {}).items()}
            viz.grasp_snapshot(grasp_png, verts_w, pick, obstacles=obstacles,
                               place_xy=plan.place["xy"], track=track,
                               title=f"attempt {attempt}: {evaluation.get('failure')}")
        except Exception as exc:                       # noqa: BLE001 - the image is an aid, not a gate
            if verbose:
                print(f"[trajGen] grasp snapshot failed ({exc}); retrying without it")
            grasp_png = None
        alternatives = [r for r in ranking.ranked if r.id not in tried_ids][:llm.ALTERNATIVES_SHOWN]
        content = llm.build_package(task=task, plan=plan.to_dict(), evaluation=evaluation,
                                    pick=pick, alternatives=alternatives, history=history,
                                    scene_png=scene_png if scene_png.exists() else None,
                                    grasp_png=grasp_png)
        try:
            verdict = llm.retry_verdict(content, model=model, verbose=verbose)
        except Exception as exc:                       # noqa: BLE001 - no verdict -> abort honestly
            report["aborted"] = f"LLM retry transport failed: {exc}"
            break
        attempt_row["llm_verdict"] = {k: verdict.get(k) for k in
                                      ("action", "candidate_id", "adjust", "rationale", "model")}
        # The LLM's choice must pass the same plan/path gate as any pick; a failing choice falls
        # back to the next best plannable candidate rather than burning the retry on a known miss.
        next_pick = llm.apply_verdict(verdict, pick, ranking, tried_ids)
        next_plan = None
        if next_pick is not None:
            if next_pick.id in tried_ids and next_pick.adjusted is None:
                next_pick = None                       # switch resolved to an already-tried id
            elif not _arm_ok(next_pick):
                next_pick = None                       # fails the pad/IK gate -> draw instead
            else:
                next_plan, why = _plan_checked(next_pick, attempt + 1)
                if next_plan is None:
                    if verbose:
                        print(f"[trajGen] LLM pick {next_pick.id} rejected at planning: {why}")
                    next_pick = None
        if next_pick is None:
            next_pick, next_plan = _draw_planned(attempt + 1)
        if next_pick is None:
            report["aborted"] = "LLM retry exhausted the candidate pool"
            break
        pick, plan = next_pick, next_plan

    report["arm_rejected"] = arm_rejected
    (run_dir / "traj_result.json").write_text(json.dumps(report, indent=1))
    if verbose:
        print(f"[trajGen] {'SUCCESS' if report['ok'] else 'FAILED'} after "
              f"{len(report['attempts'])} attempt(s); report -> {run_dir / 'traj_result.json'}")
    return report
