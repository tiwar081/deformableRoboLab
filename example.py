"""Single data-driven demo runner. Every demo is a DATA FILE in ``examples/`` that declares only a
``DemoSpec`` (scene + policy); this script plays it. Nothing here changes when you add a demo — write
one new ``examples/<name>.py`` data file and run it.

    python example.py --demo examples/soft_pickplace_franka.py --device cuda:0
    python -m examples soft_pickplace_franka --device cuda:0      # equivalent shim

See deformableManipulationTools/demo_runner.py for the DemoSpec schema and the shared executor, and
docs/physicsEngine/examples.md for the per-demo descriptions.
"""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from deformableManipulationTools.helper import install_terminal_log

_terminal_log = install_terminal_log()

import examples
from robolabViz.scenic import ScenicGraspExample
from deformableManipulationTools.demo_runner import make_data_driven_example, DemoSpec

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_BaseExample = make_data_driven_example(ScenicGraspExample)


def demo_path(demo: str) -> Path:
    """Resolve a demo given a file path OR a bare name (looked up in examples/)."""
    p = Path(demo)
    if p.suffix == ".py" and p.exists():
        return p.resolve()
    cand = EXAMPLES_DIR / (demo if demo.endswith(".py") else f"{demo}.py")
    if not cand.exists():
        raise FileNotFoundError(f"no demo data file for {demo!r} (looked for {cand})")
    return cand.resolve()


def load_spec(demo: str) -> DemoSpec:
    path = demo_path(demo)
    spec = importlib.util.spec_from_file_location(f"_demo_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "DEMO") or not isinstance(mod.DEMO, DemoSpec):
        raise ValueError(f"{path} must define DEMO = DemoSpec(...)")
    return mod.DEMO


def make_parser(spec: DemoSpec):
    parser = examples.create_parser()
    parser.set_defaults(num_frames=spec.num_frames)
    parser.add_argument("--substeps", type=int, default=spec.substeps,
                        help="Simulation substeps per rendered frame.")
    parser.add_argument("--vbd-iterations", type=int, default=spec.vbd_iterations,
                        help="VBD iterations for the deformable object solver.")
    for name, default in spec.extra_args.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    return parser


def example_for(demo: str):
    """Return (ExampleClass, parser) for a demo name/path — used by the runner + the capture harness."""
    spec = load_spec(demo)
    cls = type("DataDemo", (_BaseExample,), {"spec": spec})
    return cls, make_parser(spec)


def run(demo: str, example_name: str | None = None):
    cls, parser = example_for(demo)
    viewer, args = examples.init(parser, example_name=example_name or demo_path(demo).stem)
    examples.run(cls(viewer, args), args)


if __name__ == "__main__":
    import argparse
    import sys

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--demo", required=True, help="Demo data file in examples/ (path or bare name).")
    ns, rest = pre.parse_known_args()
    name = demo_path(ns.demo).stem
    sys.argv = [sys.argv[0], *rest]                     # hand the remaining flags to the real parser
    run(ns.demo, example_name=name)
