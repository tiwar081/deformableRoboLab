"""CLI for the trajectory stage: select -> plan -> rollout -> LLM retries -> final render.

    .venv/bin/python -m deformableManipulationTools.traj_gen <run_dir> [--device cuda:0]
        [--task-file task_2.json] [--seed N] [--temperature T] [--llm-attempts 2]
        [--no-render | --render-anyway] [--output-style mp4_advanced]

By default the stage runs EVERY task of the run dir (scene reuse: ``task.json``, ``task_2.json``,
... — each with its own demo file, ``traj<k>.json``, ``traj_result<k>.json`` and video);
``--task-file`` restricts to one. The render replays each task's executed ``traj<k>.json`` through
the standard runner, in the SAME RoboLab look as ``scene_overview.png`` (``mp4_advanced``: HDRI-lit
PBR ray tracing — ``mp4`` remains available as a fast preview), and copies the video into the run
dir as ``trajectory<k>.mp4``. By default only a SUCCESSFUL trajectory is rendered;
``--render-anyway`` renders an aborted one too (the failure stays visible — that is the point).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import annotate, rollout, verify
from .stage import demo_for_task, generate_trajectory, task_tag

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def task_files(run_dir: Path) -> list[str]:
    """All task files of a run, primary first (task.json, task_2.json, ...)."""
    out = [p.name for p in run_dir.glob("task_*.json")
           if re.fullmatch(r"task_\d+\.json", p.name)]
    out.sort(key=lambda n: int(n[5:-5]))
    return (["task.json"] if (run_dir / "task.json").exists() else []) + out


def render_trajectory(run_dir: Path, *, task_name: str = "task.json", device: str = "cuda:0",
                      output_style: str = "mp4_advanced", verbose: bool = True) -> Path | None:
    """Render the task's demo (which plays its ``traj<k>.json``) and copy the video into the run
    dir as ``trajectory<k>.mp4``."""
    demo_py = demo_for_task(run_dir, task_name)
    if demo_py is None:
        raise FileNotFoundError(f"no pipeline demo file for {task_name} in {run_dir}")
    cmd = [sys.executable, "example.py", "--demo", str(demo_py), "--output-style", output_style,
           "--viewer", "null", "--device", device, "--quiet"]
    if verbose:
        print("[trajGen] render:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"render failed (exit {res.returncode}):\n{res.stdout[-1500:]}\n"
                           f"{res.stderr[-1500:]}")
    from ..params import FRANKA
    out_dir = REPO_ROOT / "outputs" / FRANKA.short_name / demo_py.stem
    video = out_dir / ("simulation_advanced.mp4" if output_style == "mp4_advanced"
                       else "simulation.mp4")
    if not video.exists():
        raise FileNotFoundError(f"render finished but {video} is missing")
    dst = run_dir / f"trajectory{task_tag(task_name)}.mp4"
    shutil.copy2(video, dst)
    if verbose:
        print(f"[trajGen] video -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="pipeline run folder (outputs/agenticPipeline/<name>)")
    ap.add_argument("--task-file", default=None,
                    help="run only this task (task.json / task_2.json / ...); default: all tasks")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature over the re-ranked scores (default 0.08)")
    ap.add_argument("--llm-attempts", type=int, default=2,
                    help="LLM-corrected rollouts after the first failure (then abort)")
    ap.add_argument("--model", default=None, help="LLM model override for the retry loop")
    ap.add_argument("--no-render", action="store_true", help="skip the final videos")
    ap.add_argument("--render-anyway", action="store_true",
                    help="render even an aborted trajectory (failure visible in the video)")
    ap.add_argument("--no-visual-verify", action="store_true",
                    help="skip the post-trajectory visual outcome verification (+ its tuned "
                         "re-executions); annotations then carry only the geometric check")
    ap.add_argument("--output-style", default="mp4_advanced",
                    choices=["mp4_advanced", "mp4", "usd"],
                    help="video look; mp4_advanced = the scene_overview.png RoboLab look (default)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    names = [args.task_file] if args.task_file else task_files(run_dir)
    if not names:
        raise FileNotFoundError(f"no task files in {run_dir}")
    kwargs = {"device": args.device, "seed": args.seed, "llm_attempts": args.llm_attempts,
              "model": args.model}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    import time
    any_ok = False
    t_all = time.time()
    for task_name in names:
        t_task = time.time()
        tag = task_tag(task_name)
        if len(names) > 1:
            print(f"[trajGen] ===== {task_name} =====")
        report = generate_trajectory(run_dir, task_name=task_name, **kwargs)
        any_ok = any_ok or bool(report.get("ok"))
        task = json.loads((run_dir / task_name).read_text())
        if not report.get("ok") or args.no_render and not args.render_anyway:
            # No executed trajectory to verify/annotate beyond the bookkeeping row.
            annotate.upsert(run_dir, annotate.build_row(
                task=task, task_name=task_name, report=report, round_no=0,
                traj_file=f"traj{tag}.json" if (run_dir / f"traj{tag}.json").exists() else None,
                video=None, verdict=None))
        elif (run_dir / f"traj{tag}.json").exists():
            finalize_task(run_dir, task_name, report, args)
        print(f"[trajGen] {task_name} total wall-clock: {time.time() - t_task:.0f} s")
    print(f"[trajGen] run total wall-clock ({len(names)} task(s)): {time.time() - t_all:.0f} s")
    sys.exit(0 if any_ok else 3)


def finalize_task(run_dir: Path, task_name: str, report: dict, args) -> None:
    """Render -> VISUAL verification -> annotate; on a mismatch, relabel the executed round
    (never scrap it), nudge the place column, and re-execute — up to
    ``verify.MAX_VISUAL_RETRIES`` rounds per demo."""
    import time
    tag = task_tag(task_name)
    task = json.loads((run_dir / task_name).read_text())
    placement = task.get("robot_placement") or {}
    demo_py = demo_for_task(run_dir, task_name)
    scene_png = run_dir / "scene_overview.png"
    cur_report = report
    round_no = 0
    while True:
        t0 = time.time()
        video = render_trajectory(run_dir, task_name=task_name, device=args.device,
                                  output_style=args.output_style)
        render_s = time.time() - t0
        cur_report["video"] = str(video)
        cur_report.setdefault("timings", {})[f"render_r{round_no}_s"] = round(render_s, 1)
        print(f"[trajGen] render took {render_s:.0f} s")
        (run_dir / f"traj_result{tag}.json").write_text(json.dumps(cur_report, indent=1))
        verdict = None
        ev = (cur_report.get("attempts") or [{}])[-1].get("evaluation") or {}
        after = verify.final_still(demo_py) if args.output_style == "mp4_advanced" else None
        if after is not None and not args.no_visual_verify:
            try:
                t0 = time.time()
                verdict = verify.visual_verify(task, ev, placement,
                                               scene_png if scene_png.exists() else None,
                                               after, model=args.model)
                cur_report["timings"][f"verify_r{round_no}_s"] = round(time.time() - t0, 1)
            except Exception as exc:               # noqa: BLE001 - verification is best-effort
                print(f"[trajGen/verify] visual verification unavailable ({exc})")
        archived = verify.archive_round(run_dir, tag, round_no)
        annotate.upsert(run_dir, annotate.build_row(
            task=task, task_name=task_name, report=cur_report, round_no=round_no,
            traj_file=archived.get("traj"), video=archived.get("video"), verdict=verdict))
        if verdict is None or verdict.get("ok") or round_no >= verify.MAX_VISUAL_RETRIES:
            return
        traj_now = json.loads((run_dir / f"traj{tag}.json").read_text())
        if len(traj_now.get("segments") or []) > 1:
            # Multi-step: a single place nudge is ambiguous across segments — keep the honestly
            # relabeled row rather than guessing which step to move.
            print("[trajGen/verify] mismatch on a multi-step task — keeping the relabeled row "
                  "(per-segment tuning not supported yet)")
            return
        nudge = verdict.get("nudge_cm") or {}
        dx, dy = float(nudge.get("dx", 0)) / 100.0, float(nudge.get("dy", 0)) / 100.0
        if abs(dx) + abs(dy) < 0.005:
            print("[trajGen/verify] mismatch but no actionable nudge — keeping the relabeled row")
            return
        round_no += 1
        print(f"[trajGen/verify] tuning the place column by ({dx * 100:+.1f}, {dy * 100:+.1f}) cm "
              f"and re-executing (round {round_no})")
        verify.tune_place(run_dir, f"traj{tag}.json", dx, dy, round_no=round_no)
        ev2 = rollout.run_rollout(demo_py, device=args.device, verbose=True,
                                  log_path=run_dir / f"rollout{tag}_v{round_no}.log",
                                  task_name=task_name, traj_name=f"traj{tag}.json")
        last = dict((cur_report.get("attempts") or [{}])[-1])
        last["evaluation"] = ev2
        last["visual_tune_round"] = round_no
        cur_report = {**cur_report, "ok": bool(ev2.get("held") and ev2.get("carried")),
                      "attempts": (cur_report.get("attempts") or [])[:-1] + [last]}
        if not cur_report["ok"]:
            print(f"[trajGen/verify] tuned re-execution failed ({ev2.get('failure')}) — "
                  f"recording and stopping")
            annotate.upsert(run_dir, annotate.build_row(
                task=task, task_name=task_name, report=cur_report, round_no=round_no,
                traj_file=f"traj{tag}.json", video=None, verdict=None))
            (run_dir / f"traj_result{tag}.json").write_text(json.dumps(cur_report, indent=1))
            return


if __name__ == "__main__":
    main()
