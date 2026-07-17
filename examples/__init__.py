"""Example runner for the RoboLab-VBD Franka demos (registry + run loop, not a
demo itself). Each example script in this package opens with its motion summary
and run command.

Every demo is a single file ``<name>.py``. The ``--output-style`` flag selects
how a run is rendered (all artifacts land in ``outputs/<robot>/<name>/``):

- ``usd`` — lightest: the Newton USD viewer writes a time-sampled ``<name>.usd``
  (no scene look, no frames, no video).
- ``mp4`` (default) — lightweight video: the warp-raycast preview renders
  ``simulation.mp4`` (over-shoulder-left + wrist cameras, side by side) with
  flat shading and no HDRI decode (table textures kept for motion cues), on
  any CUDA GPU.
- ``mp4_advanced`` — the RoboLab look: HDRI-lit PBR ray tracing with shadows,
  textures, and anti-aliasing, written as ``simulation_advanced.mp4``; also dumps
  ``frames/`` stills (every ``--frames-per-image`` frames) and
  ``wrist_coverage.json``. A demo customizes it via ``DemoSpec.render`` (a
  ``robolabViz.RenderSpec``).

Deprecated aliases: ``basic`` -> ``usd``, ``scenic`` -> ``mp4_advanced``.

Run: python -m examples <name> --device cuda:0   (list: python -m examples --list)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from deformableManipulationTools.settings import SETTINGS

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


USD_EXTENSIONS = {".usd", ".usda", ".usdc"}

# Canonical output styles + the deprecated aliases still accepted on the CLI.
OUTPUT_STYLES = ("usd", "mp4", "mp4_advanced")
_STYLE_ALIASES = {"basic": "usd", "scenic": "mp4_advanced"}

# The single-file demos in this package (each defines `Example` + a `__main__`).
EXAMPLE_NAMES = (
    "cable_rigidCube_franka",
    "cable_soft_franka",
    "rigidCube_soft_franka",
    "soft_compression_franka",
    "soft_pickplace_franka",
    "pickplace_ycb_franka",
    "pickplace_ycb_vbd_franka",
    "cloth_franka",
    "cloth_franka_green",
    "cloth_franka_full",
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
    parser.add_argument("--device", type=str, default=SETTINGS.device,
                        help="Override the default Warp device (settings.yaml: device).")
    parser.add_argument(
        "--output-style",
        type=str,
        default=None,
        choices=[*OUTPUT_STYLES, *_STYLE_ALIASES],
        help="usd: a plain Newton USD (lightest). mp4: lightweight robolabViz video (both cameras). "
             "mp4_advanced: the full RoboLab look + frames/ stills + wrist_coverage.json. "
             "Deprecated aliases: basic->usd, scenic->mp4_advanced. "
             "Default: settings.yaml render.style (mp4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mp4",
        choices=["mp4", "graphs", "both"],
        help="What to emit: mp4 (the render), graphs (per-frame gripper physics PNG), or both. "
             "graphs work in every output style (VBD/proxy demos only); graphs-only skips the "
             "raycast render in the mp4 styles.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="usd",
        choices=["gl", "usd", "null"],
        help="Viewer for --output-style usd (the mp4 styles force null; gl gives an interactive window).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output dir for every style (a .usd path pins the USD file instead). "
             "Default: outputs/<robot>/<name>/.",
    )
    parser.add_argument("--num-frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)

    # ---- render options (--output-style mp4 / mp4_advanced) ----
    scenic = parser.add_argument_group("render options (--output-style mp4 / mp4_advanced)")
    scenic.add_argument("--frames-per-image", type=int, default=None,
                        help="Dump a per-camera still into frames/ every N frames (0 = off). "
                             "Default: 30 in mp4_advanced, 0 otherwise.")
    scenic.add_argument("--table", default=None,
                        help="Vendored work-table texture (see robolabViz.config.available_tables). "
                             "Precedence: this flag > the demo's RenderSpec > settings.yaml render.table.")
    scenic.add_argument("--background", default=None,
                        help="Vendored dome background (e.g. home_office, garage_2k; see "
                             "robolabViz.config.available_backgrounds). Precedence: this flag > "
                             "the demo's RenderSpec > settings.yaml render.background.")
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


def init(parser: argparse.ArgumentParser | None = None, example_name: str = "example"):
    import warp as wp

    import newton.viewer

    parser = parser or create_parser()
    args = parser.parse_args()

    style = getattr(args, "output_style", None) or SETTINGS.render_style
    if style in _STYLE_ALIASES:
        print(f"[examples] --output-style {style!r} is deprecated; using {_STYLE_ALIASES[style]!r} "
              f"(choices: {', '.join(OUTPUT_STYLES)}).")
        style = _STYLE_ALIASES[style]
    if style not in OUTPUT_STYLES:
        raise ValueError(f"Unknown output style {style!r} (from --output-style or settings.yaml "
                         f"render.style); choices: {', '.join(OUTPUT_STYLES)}.")
    args.output_style = style  # canonical for everything downstream

    # ONE output home for every style: outputs/<robot>/<name>/ (<robot> = the active robot's
    # short_name, so the two robots' outputs never collide). The <name>.usd slot inside it holds
    # the Newton USD (usd style) or the opt-in --usd RoboLab stage (mp4 styles).
    from deformableManipulationTools.params import FRANKA
    if args.output_path:
        p = Path(args.output_path)
        if p.suffix.lower() in USD_EXTENSIONS:
            out_dir, usd_path = p.parent, p
        else:
            out_dir, usd_path = p, p / f"{example_name}.usd"
    else:
        out_dir = Path("outputs") / FRANKA.short_name / example_name
        usd_path = out_dir / f"{example_name}.usd"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out_dir)
    args.output_path = str(usd_path)
    if style in ("mp4", "mp4_advanced"):
        args.viewer = "null"  # robolabViz renders; the Newton viewer stays inert

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

    # Centralized per-frame gripper physics graphs (--output graphs|both). VBD path only: a demo with
    # no proxy coupling (rigid-only MuJoCo grasp) records nothing even when graphs are requested.
    recorder = None
    if (getattr(args, "output", "mp4") in ("graphs", "both")
            and getattr(example, "coupling", None) is not None):
        # grasp_windows is optional: a force-grip demo supplies it (its close/release times draw the
        # trigger lines); an explicit-finger demo has none, so the recorder just omits those lines
        # (PhysicsRecorder handles grasp_windows=None).
        from deformableManipulationTools.instrumentation import PhysicsRecorder
        recorder = PhysicsRecorder(example.fps, getattr(example, "grasp_windows", None))

    while viewer.is_running():
        if viewer.should_step():
            example.step()
            if recorder is not None:
                try:
                    sample = example.gripper_instrumentation()
                except Exception as exc:
                    print(f"[instrumentation] disabled after error: {exc!r}")
                    recorder = None
                    sample = None
                if recorder is not None and sample is not None:
                    recorder.record(*sample)
        example.render()

    # Save the physics graphs BEFORE test_final so a failing assertion can't discard them.
    if recorder is not None and recorder.frames:
        out_dir = Path(getattr(args, "output_dir", None) or Path(args.output_path).parent)
        png = out_dir / "physics.png"
        recorder.save(png, title=f"{out_dir.name} — gripper physics")
        print(f"[instrumentation] physics graphs -> {png}")

    # Close the viewer BEFORE test_final: ViewerUSD only saves its stage on close, so the
    # usd-style artifact check needs it — and a failing assertion can't discard the USD.
    viewer.close()

    if args.test:
        if not hasattr(example, "test_final"):
            raise NotImplementedError("Example does not define test_final().")
        example.test_final()


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

    # Each example is now a DATA FILE (a DemoSpec); the single runner `example.py` plays it. This is
    # the same as: python example.py --demo examples/<example_name>.py [options]
    sys.argv = [sys.argv[0], *sys.argv[2:]]                 # hand remaining flags to the runner's parser
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import example as _runner
    _runner.run(example_name)


__all__ = ["create_parser", "get_examples", "init", "main", "run"]
