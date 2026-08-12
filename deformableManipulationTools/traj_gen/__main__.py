"""CLI for the trajectory stage: select -> plan -> rollout -> LLM retries -> final mp4 render.

    .venv/bin/python -m deformableManipulationTools.traj_gen <run_dir> [--device cuda:0]
        [--seed N] [--temperature T] [--llm-attempts 2] [--no-render] [--render-anyway]
        [--output-style mp4]

The render replays the LAST executed ``traj.json`` through the standard runner
(``example.py --demo <run>/pipeline_<name>.py``), so the video shows exactly the rollout that was
measured. By default only a SUCCESSFUL trajectory is rendered; ``--render-anyway`` renders an
aborted one too (the failure stays visible — that is the point). The video is copied into the run
dir as ``trajectory.mp4``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .stage import generate_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def render_trajectory(run_dir: Path, *, device: str = "cuda:0", output_style: str = "mp4",
                      verbose: bool = True) -> Path | None:
    """Render the run's demo (which now plays ``traj.json``) and copy the video into the run dir."""
    demo_py = next(iter(sorted(run_dir.glob("pipeline_*.py"))), None)
    if demo_py is None:
        raise FileNotFoundError(f"no pipeline demo file in {run_dir}")
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
    dst = run_dir / "trajectory.mp4"
    shutil.copy2(video, dst)
    if verbose:
        print(f"[trajGen] video -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="pipeline run folder (outputs/agenticPipeline/<name>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature over the re-ranked scores (default 0.08)")
    ap.add_argument("--llm-attempts", type=int, default=2,
                    help="LLM-corrected rollouts after the first failure (then abort)")
    ap.add_argument("--model", default=None, help="LLM model override for the retry loop")
    ap.add_argument("--no-render", action="store_true", help="skip the final video")
    ap.add_argument("--render-anyway", action="store_true",
                    help="render even an aborted trajectory (failure visible in the video)")
    ap.add_argument("--output-style", default="mp4", choices=["mp4", "mp4_advanced", "usd"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    kwargs = {"device": args.device, "seed": args.seed, "llm_attempts": args.llm_attempts,
              "model": args.model}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    report = generate_trajectory(run_dir, **kwargs)

    if not args.no_render and (report.get("ok") or args.render_anyway) \
            and (run_dir / "traj.json").exists():
        video = render_trajectory(run_dir, device=args.device, output_style=args.output_style)
        report["video"] = str(video)
        (run_dir / "traj_result.json").write_text(json.dumps(report, indent=1))
    sys.exit(0 if report.get("ok") else 3)


if __name__ == "__main__":
    main()
