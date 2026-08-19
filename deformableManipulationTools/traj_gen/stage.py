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

from ..grasp_library import GraspSchemaError, body_pose, is_unusable, load_grasps
from ..grasp_select import RobotState
from ..params import TABLE
from . import deform_grasp, deform_snapshot, llm, policy, rollout, selection, viz
from .policy import PlanError

DEFAULT_TEMPERATURE = selection.DEFAULT_TEMPERATURE


class TrajectoryAborted(RuntimeError):
    """The stage could not produce a working trajectory; details in the report."""


# ---------------------------------------------------------------------------------------------
# Low-graspability ledger — assets/low_graspability.md
# ---------------------------------------------------------------------------------------------
# Written ONLY from full-pipeline evidence, and ONLY when the evidence says the object's grasp
# will fail CONSISTENTLY (object-bound, placement-independent) — a single arm-blocked approach at
# one placement is not that. The scene/task generators deliberately never see grasp-candidate
# information; this file is the human-facing record of what the runs proved.
LOW_GRASPABILITY_MD = Path(__file__).resolve().parents[2] / "assets" / "low_graspability.md"

_LEDGER_HEADER = """\
# Low-graspability ledger

Objects whose grasp the FULL PIPELINE has evidenced will fail consistently (written automatically
by `deformableManipulationTools/traj_gen/stage.py` when a trajectory aborts on object-bound,
placement-independent grasp evidence — library verdicts, jaw sweeps below the tabletop, or every
rollout failing at the grasp itself). Scene/task generation deliberately does NOT read this file
(or any grasp data); it exists so humans and the grasp-generator roadmap know where the gaps are.
One section per object; evidence accumulates per run.
"""


def _consistent_failure_reason(report: dict) -> str | None:
    """A one-line 'this will keep failing' verdict, or None when the abort was circumstantial."""
    reason = report.get("aborted") or ""
    attempts = report.get("attempts", [])
    fails = [a.get("evaluation", {}).get("failure") for a in attempts]
    grasp_fails = [f for f in fails if f in ("never_held", "push_out")]
    pad_sweeps = [r for r in report.get("arm_rejected", []) if r.get("pose") == "pad_sweep"]
    if any(t in reason for t in ("UNUSABLE", "out of reach", "no usable grasp record")):
        return f"grasp-library verdict: {reason}"
    if ("exhausted" in reason or "no arm-executable" in reason) and pad_sweeps \
            and len(grasp_fails) == len(fails):
        rolled = (f"; the {len(fails)} rollout(s) run all failed the grasp itself "
                  f"({', '.join(sorted(set(grasp_fails)))})") if fails else ""
        return (f"candidate pool exhausted — {len(pad_sweeps)} candidate(s) sweep the jaw below "
                f"the tabletop (object geometry, placement-independent){rolled}")
    if len(fails) >= 2 and len(grasp_fails) == len(fails):
        return (f"all {len(fails)} rollout attempts across different candidates failed at the "
                f"grasp itself ({', '.join(sorted(set(grasp_fails)))})")
    return None


def _note_low_graspability(run_dir: Path, report: dict, *, verbose: bool = True) -> None:
    target = ((report.get("goal") or {}).get("params") or {}).get("object")
    verdict = _consistent_failure_reason(report)
    if not target or verdict is None:
        return
    import datetime
    bullet = (f"- {datetime.date.today().isoformat()} `{run_dir.name}` "
              f"({report.get('task_file', 'task.json')}, task {report.get('task')!r}): {verdict}")
    path = LOW_GRASPABILITY_MD
    text = path.read_text() if path.exists() else _LEDGER_HEADER
    section = f"\n## {target}\n"
    if section in text:
        head, _, tail = text.partition(section)
        lines = tail.split("\n\n", 1)
        block = lines[0].rstrip()
        if bullet.split("`")[1] in block and verdict in block:
            return                                    # same run, same verdict — already recorded
        rest = ("\n\n" + lines[1]) if len(lines) > 1 else "\n"
        text = head + section + block + "\n" + bullet + rest
    else:
        text = text.rstrip() + "\n" + section + bullet + "\n"
    path.write_text(text)
    if verbose:
        print(f"[trajGen] low-graspability evidence for {target!r} -> {path}")


# ---------------------------------------------------------------------------------------------
# The ONLINE deformable path: cloth/cable targets get their grasp from an LLM over the settled
# state (deform_grasp.py), the same plan/rollout machinery, and 3 proposals with feedback.
# ---------------------------------------------------------------------------------------------
def _run_deformable(run_dir: Path, *, tag: str, task_name: str, traj_name: str, result_name: str,
                    scene: dict, task: dict, placement: dict, by_name: dict, target: str,
                    report: dict, device: str, model: str | None, verbose: bool) -> dict:
    import time
    t_start = time.time()
    report["mode"] = "deformable_online"

    def _finish(reason: str | None = None) -> dict:
        report["duration_s"] = round(time.time() - t_start, 1)
        if reason:
            report["aborted"] = reason
            if verbose:
                print(f"[trajGen] ABORTED after {report['duration_s']:.0f} s: {reason}")
        (run_dir / result_name).write_text(json.dumps(report, indent=1))
        if not report.get("ok"):
            _note_low_graspability(run_dir, report, verbose=verbose)
        return report

    demo_py = demo_for_task(run_dir, task_name)
    if demo_py is None:
        return _finish(f"no pipeline demo file for {task_name} in {run_dir}")
    try:
        snapshot = deform_snapshot.run_snapshot(run_dir, device=device, tag=tag, verbose=verbose)
    except RuntimeError as exc:
        return _finish(f"deformable snapshot failed: {exc}")
    if not snapshot.get("ok"):
        return _finish(f"deformable snapshot failed: {snapshot.get('reason')}")
    report["snapshot"] = {k: snapshot[k] for k in ("kind", "z_min", "z_max", "extent_xy",
                                                   "n_material_points")}

    from .reach import checker_for_placement
    arm = checker_for_placement(placement)
    PATH_TOL_CRITICAL, PATH_TOL_TRANSIT = 0.015, 0.04
    scene_png = run_dir / "scene_overview.png"
    history: list[dict] = []
    extra_pngs: list = []

    for attempt in range(deform_grasp.MAX_ONLINE_ATTEMPTS):
        content = deform_grasp.build_content(task, snapshot, history,
                                             scene_png if scene_png.exists() else None,
                                             extra_pngs)
        try:
            proposal = deform_grasp.propose(content, model=model, verbose=verbose)
        except Exception as exc:                       # noqa: BLE001 - no proposal -> abort honestly
            return _finish(f"online grasp proposal transport failed: {exc}")
        row: dict = {"attempt": attempt,
                     "proposal": {k: proposal.get(k) for k in
                                  ("position", "approach", "jaw_axis", "width_mm", "force_n",
                                   "rationale", "model")}}
        pick, why = deform_grasp.pick_from_proposal(proposal, snapshot, attempt)
        plan = None
        if pick is not None:
            # Relaxed pad gate: a sheet pinch legitimately presses a few mm into the support.
            ok, zmin = policy.pads_clear_table(
                pick, table_z=TABLE.top_z - deform_grasp.PRESS_ALLOWANCE)
            if not ok:
                why, pick = (f"pad sweep dips {(TABLE.top_z - zmin) * 1000:.0f} mm below the "
                             f"tabletop (max {deform_grasp.PRESS_ALLOWANCE * 1000:.0f} mm press "
                             f"+ tolerance)"), None
        if pick is not None:
            # A grasped sheet/cable HANGS below the TCP by up to its material span from the
            # grasp point (capped — a long drape folds on the support during the set-down).
            pts = np.asarray(snapshot["points"], dtype=float)
            hang = float(min(np.max(np.linalg.norm(pts - pick.position, axis=1)), 0.22))
            try:
                plan = policy.plan_pick_place(scene, task, placement, by_name, pick,
                                              target_bottom_dz=0.0, attempt=attempt,
                                              drop_override=hang)
            except PlanError as exc:
                why, pick = str(exc), None
        if pick is not None:
            errs = arm.path_errors(plan.waypoints, plan.grasp_window)
            w0 = float(plan.grasp_window["close_start"]) - 1.5
            w1 = float(plan.grasp_window["release_end"]) + 0.2
            bad = [e for e in errs
                   if e["err"] > (PATH_TOL_CRITICAL if w0 <= e["t"] <= w1 else PATH_TOL_TRANSIT)]
            if bad:
                worst = max(bad, key=lambda e: e["err"])
                why, pick = (f"the arm cannot execute this pose (path IK misses by "
                             f"{worst['err'] * 1000:.0f} mm at t={worst['t']:.1f}s) — choose a "
                             f"more reachable grasp point or a more vertical approach"), None

        if pick is None:
            row["authoring_rejected"] = why
            report["attempts"].append(row)
            if verbose:
                print(f"[trajGen] proposal {attempt} rejected before rollout: {why}")
            history.append({"attempt": attempt, "position": proposal.get("position"),
                            "width_mm": float(proposal.get("width_mm") or 0),
                            "force_n": float(proposal.get("force_n") or 0),
                            "outcome": deform_grasp.outcome_line(None, why)})
            continue

        (run_dir / traj_name).write_text(json.dumps(plan.to_dict(), indent=1))
        if verbose:
            w = plan.grasp_window
            print(f"[trajGen] online attempt {attempt}: grasp at "
                  f"{plan.pick['position']} width {pick.width * 1000:.0f} mm, force "
                  f"{w['force_target']} N, {plan.num_frames} frames")
        evaluation = rollout.run_rollout(demo_py, device=device, verbose=verbose,
                                         log_path=run_dir / f"rollout{tag}_a{attempt}.log",
                                         task_name=task_name, traj_name=traj_name)
        row["plan_pick"] = plan.pick
        row["evaluation"] = evaluation
        report["attempts"].append(row)
        if verbose:
            print(f"[trajGen] online attempt {attempt}: failure={evaluation.get('failure')} "
                  f"held={evaluation.get('held')} carried={evaluation.get('carried')} "
                  f"goal={evaluation.get('goal', {}).get('ok')}")
        goal_met = bool(evaluation.get("goal", {}).get("evaluable")
                        and evaluation.get("goal", {}).get("ok"))
        if (evaluation.get("held") and evaluation.get("carried")) or goal_met:
            report["ok"] = True
            report["final"] = {"candidate_id": pick.id, "attempt": attempt,
                               "goal": evaluation.get("goal"), "goal_met": goal_met,
                               "place_xy_err": evaluation.get("place_xy_err")}
            return _finish()

        png = run_dir / f"grasp_attempt{tag}_{attempt}.png"
        try:
            track = {k: np.asarray(v) for k, v in (evaluation.get("track_samples") or {}).items()}
            viz.grasp_snapshot(png, np.asarray(snapshot["points"], dtype=float), pick,
                               place_xy=plan.place["xy"], track=track,
                               title=f"attempt {attempt}: {evaluation.get('failure')}")
            extra_pngs = [png]
        except Exception:                              # noqa: BLE001 - the image is an aid
            extra_pngs = []
        history.append({"attempt": attempt, "position": proposal.get("position"),
                        "width_mm": float(proposal.get("width_mm") or 0),
                        "force_n": float(proposal.get("force_n") or 0),
                        "outcome": deform_grasp.outcome_line(evaluation, None)})

    return _finish(f"online deformable grasp failed after "
                   f"{deform_grasp.MAX_ONLINE_ATTEMPTS} proposal(s)")


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


def task_tag(task_name: str) -> str:
    """'' for the primary task.json; '_2' for task_2.json (scene reuse) — suffixes every per-task
    artifact (traj_2.json, traj_result_2.json, rollout logs, snapshots, trajectory_2.mp4)."""
    import re
    m = re.fullmatch(r"task(_\d+)?\.json", task_name)
    if not m:
        raise ValueError(f"unrecognized task file name {task_name!r}")
    return m.group(1) or ""


def demo_for_task(run_dir: Path, task_name: str) -> Path | None:
    """The runnable demo data file that plays this task (distinct file per task of a shared
    scene — the stem also keys the render output folder)."""
    tag = task_tag(task_name)
    if tag:
        p = run_dir / f"pipeline_{run_dir.name}__t{tag[1:]}.py"
        return p if p.exists() else None
    return next((p for p in sorted(run_dir.glob("pipeline_*.py")) if "__t" not in p.stem), None)


def generate_trajectory(run_dir: Path | str, *, task_name: str = "task.json",
                        device: str = "cuda:0",
                        llm_attempts: int = llm.MAX_LLM_ATTEMPTS, seed: int = 0,
                        temperature: float = DEFAULT_TEMPERATURE, model: str | None = None,
                        verbose: bool = True) -> dict:
    """Run the whole stage on one pipeline run dir for ONE of its tasks. Returns the report
    (= ``traj_result<tag>.json``); ``report["duration_s"]`` carries the wall-clock."""
    import time
    t_start = time.time()
    run_dir = Path(run_dir)
    tag = task_tag(task_name)
    traj_name = f"traj{tag}.json"
    result_name = f"traj_result{tag}.json"
    scene = json.loads((run_dir / "scene.json").read_text())
    task = json.loads((run_dir / task_name).read_text())
    demo_py = demo_for_task(run_dir, task_name)
    if demo_py is None:
        raise FileNotFoundError(f"no pipeline demo file for {task_name} in {run_dir}")
    placement = task.get("robot_placement") or {}
    report: dict = {"run_dir": str(run_dir), "task_file": task_name, "task": task.get("name"),
                    "goal": task.get("goal"), "attempts": [], "ok": False}

    arm_rejected: list[dict] = []          # shared with the gate closures below; always reported

    def _abort(reason: str) -> dict:
        report["aborted"] = reason
        report["arm_rejected"] = arm_rejected
        report["duration_s"] = round(time.time() - t_start, 1)
        (run_dir / result_name).write_text(json.dumps(report, indent=1))
        _note_low_graspability(run_dir, report, verbose=verbose)
        if verbose:
            print(f"[trajGen] ABORTED after {report['duration_s']:.0f} s: {reason}")
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
    if kind in ("cloth", "cable"):
        # No offline record CAN exist (a settled sheet/coil is nothing like a rest shape) —
        # the ONLINE path proposes the grasp from a settled-state snapshot instead.
        if by_name.get(target, {}).get("category") == "bag":
            return _abort("bags are out of scope (no validated bag grasp mechanism)")
        return _run_deformable(run_dir, tag=tag, task_name=task_name, traj_name=traj_name,
                               result_name=result_name, scene=scene, task=task,
                               placement=placement, by_name=by_name, target=target,
                               report=report, device=device, model=model, verbose=verbose)
    # ---- subgoal expansion (multi-step tasks chain one pick-place segment per subgoal) ----
    subgoals = [g for g in (task.get("subgoals") or []) if g] or [goal]
    for g in subgoals:
        if g.get("predicate") not in policy.SUPPORTED_GOALS:
            return _abort(f"subgoal predicate {g.get('predicate')!r} not executable")
        if not (g.get("params") or {}).get("object"):
            return _abort(f"subgoal {g.get('predicate')!r} names no object")
    if len(subgoals) > 1 and any(by_name.get(g["params"]["object"], {}).get("kind")
                                 in ("cloth", "cable") for g in subgoals):
        return _abort("multi-step tasks with cloth/cable targets are not supported yet "
                      "(the online deformable path handles single-step tasks)")

    # ---- per-subgoal selection context: record, placement pose, physics-tiered ranking ----
    from deformableManipulationTools.grasp_passes.catalog import load_asset
    from agentic_pipeline.scene_gen import _object_label
    import math
    base = placement.get("base") or (0.0, 0.0, 0.0)
    robot = RobotState(base_pos=tuple(float(v) for v in base))
    ctxs: list[dict] = []
    report["selection"] = []
    for gi, g in enumerate(subgoals):
        target = g["params"]["object"]
        try:
            record = load_grasps(target)
        except (FileNotFoundError, GraspSchemaError) as exc:
            return _abort(f"no usable grasp record for {target!r}: {exc}")
        if is_unusable(record) or not record.candidates:
            return _abort(f"grasp record for {target!r} is "
                          f"{'UNUSABLE' if is_unusable(record) else 'out of reach (empty)'}")
        asset = load_asset(target)
        if not policy.object_indices(scene, target):
            return _abort(f"target {target!r} not present in the scene")
        # With duplicate instances, manipulate one that does not already satisfy the subgoal.
        chosen = policy.choose_target_index(scene, {**task, "goal": g}, by_name)
        tgt = scene["objects"][chosen]
        bottom_dz = float(np.asarray(asset.vertices)[:, 2].min())
        label = _object_label(tgt)
        ordinal = sum(1 for o in scene["objects"][:chosen] if _object_label(o) == label)
        body_z = _settled_body_z(run_dir, label, ordinal, TABLE.top_z - bottom_dz)
        world_from_body = body_pose((float(tgt["x"]), float(tgt["y"]), body_z),
                                    math.radians(float(tgt.get("yaw_deg", 0.0))))
        obstacles = policy.scene_obstacles(scene, by_name,
                                           exclude_indices=policy.object_indices(scene, target))
        ranking = selection.rank_for_task(record, world_from_body, robot, obstacles=obstacles)
        report["selection"].append(
            {"subgoal": gi, "target": target, "candidates": len(record.candidates),
             "selectable": len(ranking),
             "tiers": {str(k): v for k, v in ranking.tiers().items()},
             "weak_excluded": ranking.selection.stats.get("weak_excluded", 0),
             "seat_blocked_dropped": ranking.selection.stats.get("seat_blocked_dropped", 0)})
        if verbose:
            print(f"[trajGen] subgoal {gi} ({target}): {ranking.report()}")
        if not len(ranking):
            return _abort(f"no selectable grasp for {target!r} at this placement "
                          f"({len(ranking.selection.rejected)} rejected)")
        ctxs.append({"goal": g, "target": target, "asset": asset, "tgt": tgt,
                     "bottom_dz": bottom_dz, "body_z": body_z, "ranking": ranking,
                     "obstacles": obstacles, "chosen": chosen})

    rng = np.random.default_rng(seed)
    tried_ids: dict[int, list] = {k: [] for k in range(len(ctxs))}
    history: list[dict] = []
    scene_png = run_dir / "scene_overview.png"

    # ---- arm-feasibility gate (reach.py): IK-verify a pick before spending a rollout on it ----
    from .reach import checker_for_placement
    arm = checker_for_placement(placement)

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

    # A candidate set is only PLANNABLE if the whole waypoint path solves with the EXECUTOR'S OWN
    # path IK (branch-consistent chained solve — a per-pose ladder can accept a pose the chained
    # solve then misses by 20 cm, measured). Critical waypoints (any grasp approach through its
    # release) must hit tighter than transit knots.
    PATH_TOL_CRITICAL, PATH_TOL_TRANSIT = 0.015, 0.04

    def _segment_specs(picks: dict) -> list:
        return [policy.SegmentSpec(goal=ctxs[k]["goal"], pick=picks[k],
                                   target_bottom_dz=ctxs[k]["bottom_dz"],
                                   target_index=ctxs[k]["chosen"])
                for k in range(len(ctxs))]

    def _failing_seg_of(msg: str) -> int:
        import re
        m = re.search(r"\bs(\d+)\.", msg)
        return int(m.group(1)) if m else 0

    def _plan_checked(picks: dict, attempt: int):
        """(plan, None, -1) when all picks plan AND the full path is arm-executable,
        else (None, why, failing_segment)."""
        try:
            plan = policy.plan_segments(scene, task, placement, by_name,
                                        _segment_specs(picks), attempt=attempt)
        except PlanError as exc:
            return None, str(exc), _failing_seg_of(str(exc))
        windows = plan.grasp_windows or [plan.grasp_window]
        spans = [(float(w["close_start"]) - 1.5, float(w["release_end"]) + 0.2) for w in windows]
        errs = arm.path_errors(plan.waypoints, windows)
        bad = [e for e in errs
               if e["err"] > (PATH_TOL_CRITICAL if any(a <= e["t"] <= b for a, b in spans)
                              else PATH_TOL_TRANSIT)]
        if bad:
            worst = max(bad, key=lambda e: e["err"])
            seg_k = next((k for k, (a, b) in enumerate(spans) if a <= worst["t"] <= b), 0)
            return None, (f"executor path IK misses {len(bad)} waypoint(s), worst "
                          f"{worst['err'] * 1000:.0f} mm at t={worst['t']:.1f}s"), seg_k
        return plan, None, -1

    def _draw_pick(k: int, extra_excluded=()) -> policy.PickSpec | None:
        """Draw segment k's next candidate through the cheap per-pose gates."""
        excluded = list(tried_ids[k]) + [r["id"] for r in arm_rejected] + list(extra_excluded)
        while True:
            drawn = selection.draw(ctxs[k]["ranking"], temperature=temperature, rng=rng,
                                   exclude=excluded)
            if drawn is None:
                return None
            p = policy.pick_from_ranked(drawn)
            if _arm_ok(p):
                return p
            excluded.append(p.id)

    def _draw_planned(attempt: int, picks: dict | None = None, redraw_k: int | None = None):
        """Fill/refresh the per-segment picks until the WHOLE plan checks; None = exhausted."""
        picks = dict(picks or {})
        for k in range(len(ctxs)):
            if k not in picks:
                p = _draw_pick(k)
                if p is None:
                    return None, None
                picks[k] = p
        if redraw_k is not None:
            p = _draw_pick(redraw_k, extra_excluded=[picks[redraw_k].id])
            if p is None:
                return None, None
            picks[redraw_k] = p
        for _ in range(12):                        # bounded replan loop over failing segments
            plan, why, seg_k = _plan_checked(picks, attempt)
            if plan is not None:
                return picks, plan
            arm_rejected.append({"id": picks[seg_k].id, "segment": seg_k,
                                 "pose": "path", "reason": why})
            if verbose:
                print(f"[trajGen] s{seg_k} {picks[seg_k].id} rejected at planning: {why}")
            p = _draw_pick(seg_k, extra_excluded=[picks[seg_k].id])
            if p is None:
                return None, None
            picks[seg_k] = p
        return None, None

    picks, plan = _draw_planned(0)
    if picks is None:
        return _abort("no arm-executable grasp set among the selectable candidates "
                      f"({len(arm_rejected)} rejected by the IK/plan gate)")

    for attempt in range(1 + llm_attempts):
        for k, p in picks.items():
            if p.id not in tried_ids[k]:
                tried_ids[k].append(p.id)
        (run_dir / traj_name).write_text(json.dumps(plan.to_dict(), indent=1))
        if verbose:
            picks_txt = ", ".join(f"s{k}:{p.id}" for k, p in sorted(picks.items()))
            print(f"[trajGen] attempt {attempt}: {picks_txt} — "
                  f"{len(plan.segments)} segment(s), {plan.num_frames} frames, "
                  f"routing {plan.routing['legs']}")
        evaluation = rollout.run_rollout(demo_py, device=device, verbose=verbose,
                                         log_path=run_dir / f"rollout{tag}_a{attempt}.log",
                                         task_name=task_name, traj_name=traj_name)
        fail_k = int(evaluation.get("failing_segment") or 0)
        fail_k = min(fail_k, len(ctxs) - 1)
        pick = picks[fail_k]                       # the retry loop's subject
        attempt_row = {"attempt": attempt,
                       "candidate_ids": {str(k): p.id for k, p in sorted(picks.items())},
                       "candidate_id": pick.id, "failing_segment": fail_k,
                       "adjusted": pick.adjusted, "plan_pick": plan.segments[fail_k]["pick"],
                       "evaluation": evaluation}
        report["attempts"].append(attempt_row)
        if verbose:
            print(f"[trajGen] attempt {attempt} outcome: failure={evaluation.get('failure')} "
                  f"(segment {evaluation.get('failing_segment')}) "
                  f"held={evaluation.get('held')} carried={evaluation.get('carried')} "
                  f"goal={evaluation.get('goal', {}).get('ok')} "
                  f"({evaluation.get('goal', {}).get('detail')})")
        # SUCCESS = the transport metrics worked, OR the task's own goal predicate is MET at the
        # final state (outcome over process: a sheet that slips late in a drag but ends where the
        # task asked is a successful demo — measured on the shirt-retrieval task, whose goal
        # passed on every "dropped_in_transit" attempt).
        goal_met = bool(evaluation.get("goal", {}).get("evaluable")
                        and evaluation.get("goal", {}).get("ok"))
        if (evaluation.get("held") and evaluation.get("carried")) or goal_met:
            report["ok"] = True
            report["final"] = {"candidate_id": pick.id,
                               "candidate_ids": {str(k): p.id for k, p in sorted(picks.items())},
                               "attempt": attempt, "goal": evaluation.get("goal"),
                               "goal_met": goal_met,
                               "place_xy_err": evaluation.get("place_xy_err")}
            break

        if attempt == llm_attempts:
            report["aborted"] = (f"grasp failed on all {1 + llm_attempts} attempts "
                                 f"(last failure: {evaluation.get('failure')} on segment "
                                 f"{evaluation.get('failing_segment')})")
            break

        # ---- the LLM retry (scoped to the FAILING segment's grasp) ----
        ctx = ctxs[fail_k]
        seg_task = {**task, "goal": ctx["goal"]}
        history.append({"attempt": attempt, "candidate_id": pick.id, "adjusted": pick.adjusted,
                        "failure": evaluation.get("failure"),
                        "close_drift": evaluation.get("close_drift"),
                        "lift_rise": evaluation.get("lift_rise")})
        grasp_png = run_dir / f"grasp_attempt{tag}_{attempt}.png"
        try:
            verts_w = _asset_world_vertices(ctx["asset"], ctx["tgt"], ctx["body_z"])
            track = {k: np.asarray(v) for k, v in
                     (evaluation.get("track_samples") or {}).items()}
            viz.grasp_snapshot(grasp_png, verts_w, pick, obstacles=ctx["obstacles"],
                               place_xy=plan.segments[fail_k]["place"]["xy"], track=track,
                               title=f"attempt {attempt} (segment {fail_k}): "
                                     f"{evaluation.get('failure')}")
        except Exception as exc:                       # noqa: BLE001 - the image is an aid, not a gate
            if verbose:
                print(f"[trajGen] grasp snapshot failed ({exc}); retrying without it")
            grasp_png = None
        ranking = ctx["ranking"]
        alternatives = [r for r in ranking.ranked
                        if r.id not in tried_ids[fail_k]][:llm.ALTERNATIVES_SHOWN]
        seg_plan_dict = {**plan.to_dict(), "pick": plan.segments[fail_k]["pick"],
                         "grasp_window": plan.segments[fail_k]["grasp_window"],
                         "attempt": attempt}
        content = llm.build_package(task=seg_task, plan=seg_plan_dict, evaluation=evaluation,
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
        next_pick = llm.apply_verdict(verdict, pick, ranking, tried_ids[fail_k])
        next_plan = None
        if next_pick is not None:
            if next_pick.id in tried_ids[fail_k] and next_pick.adjusted is None:
                next_pick = None                       # switch resolved to an already-tried id
            elif not _arm_ok(next_pick):
                next_pick = None                       # fails the pad/IK gate -> draw instead
            else:
                cand = {**picks, fail_k: next_pick}
                next_plan, why, _ = _plan_checked(cand, attempt + 1)
                if next_plan is None:
                    if verbose:
                        print(f"[trajGen] LLM pick {next_pick.id} rejected at planning: {why}")
                    next_pick = None
                else:
                    picks = cand
        if next_pick is None:
            picks2, next_plan = _draw_planned(attempt + 1, picks=picks, redraw_k=fail_k)
            if picks2 is None:
                report["aborted"] = "LLM retry exhausted the candidate pool"
                break
            picks = picks2
        plan = next_plan

    report["arm_rejected"] = arm_rejected
    report["duration_s"] = round(time.time() - t_start, 1)
    (run_dir / result_name).write_text(json.dumps(report, indent=1))
    if not report["ok"]:
        _note_low_graspability(run_dir, report, verbose=verbose)
    if verbose:
        print(f"[trajGen] {'SUCCESS' if report['ok'] else 'FAILED'} after "
              f"{len(report['attempts'])} attempt(s) in {report['duration_s']:.0f} s; "
              f"report -> {run_dir / result_name}")
    return report
