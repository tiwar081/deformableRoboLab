"""Headless trajectory rollout + grasp/goal measurement (``python -m ...traj_gen.rollout <demo>``).

Runs the generated demo (whose folder carries ``traj.json``) with a null viewer, tracks the target
object through the plan's phases, and prints ONE ``TRAJ_EVAL_JSON: {...}`` line:

  held           — the object rose with the hand through the lift phase (the grasp worked)
  carried        — at the end of the carry phase the object is over the place point, still elevated
  close_drift    — object displacement during the close window [m] (push-out detector)
  lift_rise      — object z gain over the lift phase [m]
  place_xy_err   — final object xy distance from the plan's place point [m]
  goal           — ``success.evaluate`` on the task's ``success_spec`` at the final state
                   (``evaluable: false`` is "no score", never a failure)
  failure        — one word for the LLM retry loop: none | push_out | never_held |
                   dropped_in_transit | misplaced | diverged

Run in a SUBPROCESS (``run_rollout``): Newton/warp builds are one-shot per process, exactly like
the settle harness this mirrors.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_TAG = "TRAJ_EVAL_JSON:"

# The object must rise at least this much through the lift phase to count as held [m]
# (the plan lifts by the 10 cm standoff; half of it is decisive).
LIFT_HELD_MIN = 0.04
# Carried = within this xy radius of the place point at carry end [m].
CARRY_RADIUS = 0.12
# Push-out = the object moved this far during the close window [m].
PUSH_OUT_DRIFT = 0.02


def run_rollout(demo_py: Path | str, *, device: str = "cuda:0", verbose: bool = True,
                log_path: Path | str | None = None, task_name: str = "task.json",
                traj_name: str = "traj.json") -> dict:
    """Run the rollout in a subprocess and return the parsed report.

    ``log_path`` keeps the subprocess's full stdout/stderr (IK-miss warnings, solver chatter) —
    the report alone can say WHAT failed but the log says WHY."""
    cmd = [sys.executable, "-m", "deformableManipulationTools.traj_gen.rollout", str(demo_py),
           "--device", device, "--task", task_name, "--traj", traj_name]
    if verbose:
        print("[trajGen] rollout:", " ".join(cmd))
    import time
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if log_path is not None:
        Path(log_path).write_text((res.stdout or "") + "\n--- stderr ---\n" + (res.stderr or ""))
    for line in reversed((res.stdout or "").splitlines()):
        if line.startswith(EVAL_TAG):
            out = json.loads(line[len(EVAL_TAG):])
            out["rollout_s"] = round(elapsed, 1)
            if verbose:
                print(f"[trajGen] rollout took {elapsed:.0f} s")
            return out
    raise RuntimeError(f"trajectory rollout produced no report (exit {res.returncode}):\n"
                       f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def _labels(demo) -> list[str]:
    if demo.object_model is not None:
        raw = list(demo.object_model.body_label)
    elif demo.object_body_start is not None:
        raw = list(demo.robot_model.body_label)[demo.object_body_start:]
    else:
        raw = []
    return [str(lb).split("/")[-1] for lb in raw]


def _body_index(labels: list[str], keep: list[int], label: str, ordinal: int) -> int | None:
    seen = 0
    for i in keep:
        if labels[i] == label:
            if seen == ordinal:
                return i
            seen += 1
    return None


def _yaw_of(q) -> float:
    qx, qy, qz, qw = (float(q[k]) for k in range(4))
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _quat_rotate(q, pts: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) points by an xyzw quaternion (vectorized Rodrigues via the vector part)."""
    v = np.asarray(q[:3], dtype=float)
    w = float(q[3])
    t = 2.0 * np.cross(v, pts)
    return pts + w * t + np.cross(v, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", default="task.json")
    ap.add_argument("--traj", default="traj.json")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    run_dir = Path(args.demo).resolve().parent
    traj = json.loads((run_dir / args.traj).read_text())
    scene = json.loads((run_dir / "scene.json").read_text())
    task = json.loads((run_dir / args.task).read_text())

    import example as ex
    import examples
    from deformableManipulationTools.params import TABLE

    cls, parser = ex.example_for(args.demo)
    sys.argv = [sys.argv[0], "--viewer", "null", "--device", args.device,
                "--output", "graphs", "--quiet"]
    viewer, run_args = examples.init(parser, example_name="traj_rollout")
    demo = cls(viewer, run_args)
    frames = demo.spec.num_frames
    fps = 60.0

    labels = _labels(demo)
    keep = [i for i, lb in enumerate(labels)
            if not any(t in lb.lower() for t in ("proxy", "palm", "finger", "table"))]
    # One tracker per SEGMENT (multi-step tasks chain several pick-places in one trajectory;
    # legacy single-goal traj files synthesize one segment from the top-level fields).
    segments = traj.get("segments") or [{
        "segment": 0, "target": traj["target"], "target_label": traj["target_label"],
        "target_ordinal": traj.get("target_ordinal", 0),
        "target_kind": traj.get("target_kind", "rigid"),
        "grasp_window": traj["grasp_window"], "place": traj["place"], "pick": traj["pick"],
        "goal": task.get("goal") or {}}]
    multi = len(segments) > 1
    cable_ids = [i for i in keep if labels[i] == "vbd_cable"]
    has_particles = demo.object_model is not None and demo.object_model.particle_count
    phase_by_name = {p["name"]: p for p in traj["phases"]}

    def span(k: int, name: str, default: dict) -> dict:
        return phase_by_name.get((f"s{k}." if multi else "") + name, default)

    trackers = []
    for s in segments:
        knd = s.get("target_kind", "rigid")
        trackers.append({
            "seg": s, "kind": knd,
            "close_f": int(float(s["grasp_window"]["close_start"]) * 60),
            "grasp_pos": np.asarray(s["pick"]["position"], dtype=float),
            "tracked_pt": None,        # deformable: material point nearest the grasp at close time
            "body": (_body_index(labels, keep, s["target_label"], int(s.get("target_ordinal", 0)))
                     if knd == "rigid" else None),
            "track": np.full((frames, 3), np.nan)})
    need_pq = has_particles and any(t["kind"] in ("cloth", "soft") for t in trackers)
    tcp_track = np.full((frames, 3), np.nan)

    for f in range(frames):
        demo.step()
        bq = demo.object_body_q()
        pq = demo.object_state_0.particle_q.numpy() if need_pq else None
        for st in trackers:
            if st["kind"] == "rigid":
                if st["body"] is not None and bq.size:
                    st["track"][f] = bq[st["body"], :3]
            elif st["kind"] == "cable" and cable_ids:
                pos = bq[cable_ids, :3]
                if f >= st["close_f"] and st["tracked_pt"] is None:
                    st["tracked_pt"] = int(np.argmin(((pos - st["grasp_pos"]) ** 2).sum(axis=1)))
                st["track"][f] = pos[st["tracked_pt"]] if st["tracked_pt"] is not None \
                    else pos.mean(axis=0)
            elif st["kind"] in ("cloth", "soft") and pq is not None:
                if f >= st["close_f"] and st["tracked_pt"] is None:
                    st["tracked_pt"] = int(np.argmin(((pq - st["grasp_pos"]) ** 2).sum(axis=1)))
                st["track"][f] = pq[st["tracked_pt"]] if st["tracked_pt"] is not None \
                    else pq.mean(axis=0)
        tcp_track[f] = demo.tcp_position(demo.robot_state_0)

    bq = demo.object_body_q()
    finite = bool(np.all(np.isfinite(bq))) if bq.size else True
    if has_particles:
        finite = finite and bool(np.all(np.isfinite(demo.object_state_0.particle_q.numpy())))

    def tcp_at(t: float) -> np.ndarray:
        f = int(min(max(t * fps, 0), frames - 1))
        return tcp_track[f]

    def _seg_tracked(st: dict) -> bool:
        if st["kind"] == "rigid":
            return st["body"] is not None
        return bool(cable_ids) if st["kind"] == "cable" else bool(has_particles)

    report: dict = {"finite": finite, "target": traj["target"],
                    "target_kind": traj.get("target_kind", "rigid"),
                    "target_tracked": all(_seg_tracked(st) for st in trackers),
                    "frames": frames, "n_segments": len(segments)}
    if not report["target_tracked"] or not finite:
        report.update({"held": False, "carried": False, "failure": "diverged" if not finite
                       else "target_untracked", "goal": {"ok": False, "evaluable": False,
                                                         "detail": "no measurement"}})
        print(EVAL_TAG + json.dumps(report))
        return

    # ---- per-segment metrics; the FIRST failing segment names the trajectory's failure ----
    seg_reports = []
    failure, failing_segment = "none", None
    for k, st in enumerate(trackers):
        s = st["seg"]
        win = s["grasp_window"]
        place_xy = np.asarray(s["place"]["xy"], dtype=float)

        def at(t: float) -> np.ndarray:
            f = int(min(max(t * fps, 0), frames - 1))
            return st["track"][f]

        p_close0, p_close1 = at(win["close_start"]), at(win["close_end"])
        lift = span(k, "lift", {"t0": win["close_end"], "t1": win["close_end"] + 1.0})
        carry = span(k, "carry", lift)
        release = span(k, "release", carry)
        p_lift1, p_carry1 = at(lift["t1"]), at(carry["t1"])
        # The segment's object is measured a moment after ITS release (a later segment may bump
        # it, but its placement verdict belongs to this segment).
        p_done = at(release["t1"] + 0.5)
        close_drift = float(np.linalg.norm(p_close1 - p_close0))
        lift_rise = float(p_lift1[2] - p_close1[2])
        held = lift_rise > LIFT_HELD_MIN
        carry_xy_err = float(np.linalg.norm(p_carry1[:2] - place_xy))
        carried = held and carry_xy_err < CARRY_RADIUS and (p_carry1[2] - p_close0[2]) > 0.02
        place_xy_err = float(np.linalg.norm(p_done[:2] - place_xy))
        tcp_grasp_err = float(np.linalg.norm(tcp_at(win["close_start"]) - st["grasp_pos"]))
        if not held:
            seg_fail = "approach_missed" if tcp_grasp_err > 0.05 else \
                ("push_out" if close_drift > PUSH_OUT_DRIFT else "never_held")
        elif not carried:
            seg_fail = "dropped_in_transit"
        elif place_xy_err > 0.18:
            seg_fail = "misplaced"
        else:
            seg_fail = "none"
        seg_reports.append({
            "segment": k, "target": s["target"], "held": bool(held), "carried": bool(carried),
            "failure": seg_fail, "close_drift": round(close_drift, 4),
            "lift_rise": round(lift_rise, 4), "carry_xy_err": round(carry_xy_err, 4),
            "place_xy_err": round(place_xy_err, 4), "tcp_grasp_err": round(tcp_grasp_err, 4),
            "final_pos": [round(float(v), 4) for v in st["track"][-1]]})
        if seg_fail != "none" and failing_segment is None:
            failure, failing_segment = seg_fail, k

    # Top-level metrics mirror the FAILING segment (the retry loop's subject), or the last one.
    focus_k = failing_segment if failing_segment is not None else len(segments) - 1
    focus = seg_reports[focus_k]
    st = trackers[focus_k]
    held = all(r["held"] for r in seg_reports)
    carried = all(r["carried"] for r in seg_reports)
    close_drift, lift_rise = focus["close_drift"], focus["lift_rise"]
    carry_xy_err, place_xy_err = focus["carry_xy_err"], focus["place_xy_err"]
    tcp_grasp_err = focus["tcp_grasp_err"]
    track = st["track"]
    p_final = track[-1]

    def at(t: float) -> np.ndarray:
        f = int(min(max(t * fps, 0), frames - 1))
        return track[f]

    # ---- goal evaluation at the final state ----
    from agentic_pipeline import success
    from agentic_pipeline.scene_gen import _object_label

    state = success.SceneState()
    pl = task.get("robot_placement") or {}
    state.base_xy = tuple(pl.get("base", (0.0, 0.0))[:2])
    state.facing_yaw_deg = float(pl.get("yaw_deg", -90.0))
    from agentic_pipeline.scene_generator import catalog_by_name, load_catalog
    by_name = catalog_by_name(load_catalog())
    ref_names = {s["place"].get("reference") for s in segments if s["place"].get("reference")}
    cursors: dict[str, int] = {}
    mapped: dict[str, str] = {}
    for o in scene.get("objects", []):
        if o["name"] in state.bodies:                 # first instance wins the name (task refs one)
            continue
        lb = _object_label(o)
        k = cursors.get(lb, 0)
        cursors[lb] = k + 1
        i = _body_index(labels, keep, lb, k)
        if i is None:
            continue
        mapped[o["name"]] = labels[i]
        pos = bq[i, :3]
        dims = np.asarray(by_name.get(o["name"], {}).get("dims") or (0.05, 0.05, 0.05), float)
        yaw = math.radians(_yaw_of(bq[i, 3:7]))
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx = 0.5 * (c * dims[0] + s * dims[1])
        hy = 0.5 * (s * dims[0] + c * dims[1])
        lo = [float(pos[0] - hx), float(pos[1] - hy), float(pos[2] - 0.5 * dims[2])]
        hi = [float(pos[0] + hx), float(pos[1] + hy), float(pos[2] + 0.5 * dims[2])]
        state.bodies[o["name"]] = {"pos": [float(v) for v in pos], "aabb": (lo, hi),
                                   "yaw_deg": _yaw_of(bq[i, 3:7])}
        # The open-top containment evaluator needs the container's MESH at its final pose
        # (an AABB has no cavity). Provided for every segment's reference container.
        if o["name"] in ref_names and by_name.get(o["name"], {}).get("container"):
            try:
                from deformableManipulationTools.grasp_passes.catalog import load_asset
                verts = np.asarray(load_asset(o["name"]).vertices, dtype=float)
                state.bodies[o["name"]]["hull_points"] = \
                    (_quat_rotate(bq[i, 3:7], verts) + pos).tolist()
            except Exception:                          # noqa: BLE001 - fall back to AABB corners
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                    for z in (lo[2], hi[2])])
                state.bodies[o["name"]]["hull_points"] = corners.tolist()
    if has_particles:
        soft = [o["name"] for o in scene.get("objects", [])
                if by_name.get(o["name"], {}).get("kind") in ("cloth", "soft_mesh", "soft_block")]
        if len(soft) == 1:                            # one particle deformable -> all particles are it
            state.particles[soft[0]] = demo.object_state_0.particle_q.numpy()
    if cable_ids:                                     # a cable is capsule BODIES, not particles —
        cables = [o["name"] for o in scene.get("objects", [])   # its "point cloud" is the nodes
                  if by_name.get(o["name"], {}).get("kind") == "cable"]
        if len(cables) == 1:
            state.particles[cables[0]] = bq[cable_ids, :3]

    def _eval_goal(goal_dict: dict, ref: str | None) -> dict:
        params = dict(goal_dict.get("params") or {})
        # Task gen occasionally stores the reference under the wrong role name (its feasibility
        # flags it); patch the evaluator's expected role from the resolved reference.
        for role in ("container", "target", "base", "reference"):
            if role not in params and ref:
                params.setdefault(role, ref)
        return success.evaluate(goal_dict.get("predicate", ""), params, state)

    spec = task.get("success_spec") or {}
    goal = _eval_goal(spec, segments[0]["place"].get("reference")) if spec else \
        {"ok": False, "evaluable": False, "detail": "no success_spec"}
    # Multi-step: every SUBGOAL is also checked at the final state, and the overall verdict is
    # the AND of the primary (when evaluable) with ALL evaluable subgoals — the primary alone
    # under-specifies a group (the task schema stores one object name per role, so a groups
    # predicate can pass with one member placed; measured: "all fruits in the bowl" read ok with
    # two fruits still on the table — the visual verifier caught it, this makes the geometric
    # check agree).
    subgoal_checks = []
    if multi:
        for k, s in enumerate(segments):
            r = _eval_goal(s.get("goal") or {}, s["place"].get("reference"))
            subgoal_checks.append({"segment": k, **r})
        evaluable_subs = [r for r in subgoal_checks if r.get("evaluable")]
        if evaluable_subs:
            primary_ok = bool(goal.get("ok")) if goal.get("evaluable") else True
            goal = {"ok": primary_ok and all(r["ok"] for r in evaluable_subs),
                    "evaluable": True,
                    "detail": "; ".join(f"s{r['segment']}: {r['detail']}"
                                        for r in subgoal_checks)}

    pfx = f"s{focus_k}." if multi else ""
    report.update({
        "held": bool(held), "carried": bool(carried), "failure": failure,
        "failing_segment": failing_segment,
        "segments": seg_reports,
        "close_drift": round(close_drift, 4), "lift_rise": round(lift_rise, 4),
        "carry_xy_err": round(carry_xy_err, 4), "place_xy_err": round(place_xy_err, 4),
        "tcp_grasp_err": round(tcp_grasp_err, 4),
        "final_pos": [round(float(v), 4) for v in p_final],
        "final_z_above_table": round(float(p_final[2] - TABLE.top_z), 4),
        "goal": goal,
        "subgoal_checks": subgoal_checks or None,
        "mapped_bodies": mapped,
        "track_samples": {name: [round(float(v), 4) for v in at(phase_by_name[pfx + name]["t1"])]
                          for name in ("cruise", "close", "lift", "carry", "place", "release")
                          if pfx + name in phase_by_name},
        "tcp_samples": {name: [round(float(v), 4) for v in tcp_at(phase_by_name[pfx + name]["t1"])]
                        for name in ("cruise", "descend", "close", "lift", "carry", "place")
                        if pfx + name in phase_by_name},
    })
    print(EVAL_TAG + json.dumps(report))


if __name__ == "__main__":
    main()
