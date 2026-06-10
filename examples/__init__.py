from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


USD_EXTENSIONS = {".usd", ".usda", ".usdc"}


def get_examples() -> dict[str, str]:
    examples_dir = Path(__file__).resolve().parent
    example_map: dict[str, str] = {}
    for path in sorted(examples_dir.glob("example_*.py")):
        example_name = path.stem.removeprefix("example_")
        example_map[example_name] = f"examples.{path.stem}"
    return example_map


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--viewer",
        type=str,
        default="usd",
        choices=["gl", "usd", "null"],
        help="Viewer to use.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="USD file or output directory. Directories receive <example>.usd.",
    )
    parser.add_argument("--num-frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
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
    while viewer.is_running():
        if viewer.should_step():
            example.step()
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

    example_name = sys.argv[1] if len(sys.argv) >= 2 else "minimal_cable_franka"
    if example_name not in examples:
        print(f"Error: Unknown example '{example_name}'\n")
        _print_examples(examples)
        sys.exit(1)

    sys.argv = [examples[example_name], *sys.argv[2:]]
    runpy.run_module(examples[example_name], run_name="__main__")


__all__ = ["create_parser", "get_examples", "init", "main", "run"]
