"""Example runner for the RoboLab-VBD Franka demos (registry + run loop, not a
demo itself). Each example script in this package opens with its motion summary
and run command.

Every demo is a single file ``<name>.py``. The ``--output-style`` flag selects
how a run is rendered:

- ``scenic`` (default) — robolabViz renders ``outputs/<name>/`` with a ``frames/``
  folder (a still every ``--frames-per-image`` frames) and ``simulation.mp4``
  (over-shoulder-left + wrist cameras, side by side), on any CUDA GPU.
- ``basic`` — the Newton USD viewer writes ``outputs/<name>.usd`` (no scene look).

Run: python -m examples <name> --device cuda:0   (list: python -m examples --list)
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


USD_EXTENSIONS = {".usd", ".usda", ".usdc"}

# The single-file demos in this package (each defines `Example` + a `__main__`).
EXAMPLE_NAMES = (
    "cable_rigidCube_franka",
    "cable_soft_franka",
    "rigidCube_soft_franka",
    "soft_compression_franka",
    "soft_pickplace_franka",
    "pickplace_ycb_franka",
)


def get_examples() -> dict[str, str]:
    examples_dir = Path(__file__).resolve().parent
    return {
        name: f"examples.{name}"
        for name in EXAMPLE_NAMES
        if (examples_dir / f"{name}.py").exists()
    }


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t", "on")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--output-style",
        type=str,
        default="scenic",
        choices=["basic", "scenic"],
        help="scenic: robolabViz frames/ + simulation.mp4 in outputs/<name>/. "
             "basic: a plain Newton USD at outputs/<name>.usd.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="usd",
        choices=["gl", "usd", "null"],
        help="Viewer for --output-style basic (scenic forces null).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="basic: USD file/dir (dirs receive <name>.usd). scenic: output dir (default outputs/<name>).",
    )
    parser.add_argument("--num-frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    # Diagnostic only: print per-pad grip force [left, right] N each frame. OFF by default (it forces
    # a host readback every frame); enabling it does NOT change the simulation, only observes it.
    parser.add_argument("--log-grip", action=argparse.BooleanOptionalAction, default=False,
                        help="Print per-pad grip force [left, right] N each frame (diagnostic; no physics change).")

    # ---- scenic (--output-style scenic) rendering options ----
    scenic = parser.add_argument_group("scenic rendering (--output-style scenic)")
    scenic.add_argument("--frames-per-image", type=int, default=30,
                        help="Dump a per-camera still into frames/ every N frames (0 = off).")
    scenic.add_argument("--table", default="maple",
                        help="Vendored work-table texture (e.g. maple; see robolabViz.config.available_tables).")
    scenic.add_argument("--background", default="home_office",
                        help="Vendored dome background (e.g. home_office, garage_2k; "
                             "see robolabViz.config.available_backgrounds).")
    scenic.add_argument("--usd", type=_str2bool, nargs="?", const=True, default=False,
                        help="Also write the full time-sampled RoboLab USD scene to outputs/<name>/<name>.usd.")
    scenic.add_argument("--npz", type=_str2bool, nargs="?", const=True, default=False,
                        help="Also write the per-frame state cache + geometry.pkl for robolabViz.rerender.")
    scenic.add_argument("--objectview", type=_str2bool, nargs="?", const=True, default=False,
                        help="Add a fixed object-inspection camera (soft demos only); its frames go to "
                             "frames/ but it is kept out of simulation.mp4.")
    scenic.add_argument("--wrist-eye", type=float, nargs=3, default=None,
                        help="Override wrist camera eye in the hand frame (x y z).")
    scenic.add_argument("--wrist-target", type=float, nargs=3, default=None,
                        help="Override wrist camera look-at target in the hand frame (x y z).")
    return parser


def _resolve_output_path(value: str | None, example_name: str) -> str:
    output_path = Path(value) if value else Path("outputs")
    if output_path.suffix.lower() not in USD_EXTENSIONS:
        output_path = output_path / f"{example_name}.usd"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def init(parser: argparse.ArgumentParser | None = None, example_name: str = "example"):
    import warp as wp

    import newton.viewer

    parser = parser or create_parser()
    args = parser.parse_args()

    if getattr(args, "output_style", "scenic") == "scenic":
        # scenic renders through robolabViz; everything lands in outputs/<name>/.
        args.viewer = "null"
        out_dir = Path(args.output_path) if args.output_path else Path("outputs") / example_name
        if out_dir.suffix.lower() in USD_EXTENSIONS:
            out_dir = out_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(out_dir)
        # Used only by the opt-in --usd / --npz outputs.
        args.output_path = str(out_dir / f"{example_name}.usd")
    else:
        args.output_dir = None
        args.output_path = _resolve_output_path(args.output_path, example_name)

    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        viewer = newton.viewer.ViewerGL(headless=args.headless, paused=args.paused)
    elif args.viewer == "usd":
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    else:
        raise ValueError(f"Unsupported viewer: {args.viewer}")

    return viewer, args


def run(example, args) -> None:
    viewer = example.viewer
    log_grip = getattr(args, "log_grip", False) and hasattr(example, "grip_force_norms")
    while viewer.is_running():
        if viewer.should_step():
            example.step()
            if log_grip:
                # Read-only diagnostic: per-pad grip force [left, right] N (the harvested object
                # reaction). Does not feed back into the sim — purely observes the contact balance.
                left, right = example.grip_force_norms()
                print(f"[grip] t={example.sim_time:6.3f}  left={left:8.3f} N  right={right:8.3f} N", flush=True)
        example.render()

    if args.test:
        if not hasattr(example, "test_final"):
            raise NotImplementedError("Example does not define test_final().")
        example.test_final()

    viewer.close()


def _print_examples(examples: dict[str, str]) -> None:
    print("Available examples:")
    for name in examples:
        print(f"  {name}")


def main() -> None:
    examples = get_examples()

    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m examples <example_name> [options]")
        print("       python -m examples --list")
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        _print_examples(examples)
        sys.exit(0)

    example_name = sys.argv[1] if len(sys.argv) >= 2 else "cable_rigidCube_franka"
    if example_name not in examples:
        print(f"Error: Unknown example '{example_name}'\n")
        _print_examples(examples)
        sys.exit(1)

    sys.argv = [examples[example_name], *sys.argv[2:]]
    runpy.run_module(examples[example_name], run_name="__main__")


__all__ = ["create_parser", "get_examples", "init", "main", "run"]
