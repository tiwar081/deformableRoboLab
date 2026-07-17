"""Assemble the final physics-ready DemoSpec from the pipeline's three stage artifacts.

A pipeline run folder holds ``scene.json`` (stage 1, required), ``task.json`` (stage 2, carries
``robot_placement``), and ``env.json`` (stage 3, the look + cameras). The generated demo data file
in the folder just calls ``demo_from_dir(<folder>)`` — missing artifacts fall back to defaults, so
the SAME demo file works for the early settle check (scene only) and for the final render (all
three present)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import scene_generator as sg
from . import env_gen, geometry

DEMO_TEMPLATE = (
    '"""GENERATED pipeline demo (agentic_pipeline) — artifacts live beside this file."""\n'
    "from pathlib import Path\n\n"
    "import agentic_pipeline.build\n\n"
    "DEMO = agentic_pipeline.build.demo_from_dir(Path(__file__).parent)\n")


def _read(d: Path, name: str) -> dict | None:
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def placement_from_dir(d: Path | str) -> dict:
    task = _read(Path(d), "task.json")
    if task and task.get("robot_placement"):
        return task["robot_placement"]
    return geometry.default_placement()


def demo_from_dir(d: Path | str):
    """scene.json (+ task.json robot placement, + env.json look) -> DemoSpec."""
    import warp as wp
    from deformableManipulationTools.demo_runner import WP

    d = Path(d)
    scene = _read(d, "scene.json")
    if scene is None:
        raise FileNotFoundError(f"no scene.json in {d}")
    placement = placement_from_dir(d)
    env = _read(d, "env.json") or env_gen.default_env(placement)

    spec = sg.demo_spec_from_scene(scene)
    base = placement["base"]
    spec.robot_base_xform = wp.transform(
        wp.vec3(float(base[0]), float(base[1]), float(base[2])),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(placement["yaw_deg"])))
    # Start the arm at a RAISED, out-of-the-way pose (frame 0) instead of home_q, so the parked
    # end-effector hovers well above the scene and can't touch objects during the settle (the
    # cloth demos' "start high and clear" trick). One waypoint held for the whole settle window.
    spec.waypoints = [WP(0.0, geometry.parked_start_tcp(placement))]
    spec.start_at_first_waypoint = True
    render = env_gen.env_to_render_spec(env, placement)
    render.soft_body_style = spec.render.soft_body_style   # keep the scene's deformable color
    spec.render = render
    return spec


def write_run(scene: dict, out_root: Path | str, name: str | None = None) -> tuple[Path, Path]:
    """Create the run folder (numbered — never overwrites an existing run), write ``scene.json``
    and the generated demo file. Returns ``(run_dir, demo_py)``."""
    import re
    slug = re.sub(r"[^a-z0-9_]+", "_", (name or scene.get("name", "run")).lower()).strip("_") or "run"
    root = Path(out_root)
    out, n = root / slug, 1
    while out.exists():
        n += 1
        out = root / f"{slug}_{n}"
    if n > 1:
        slug = f"{slug}_{n}"
        scene["name"] = slug
    out.mkdir(parents=True)
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    demo_py = out / f"pipeline_{slug}.py"
    demo_py.write_text(DEMO_TEMPLATE)
    return out, demo_py
