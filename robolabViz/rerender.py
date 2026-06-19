"""Re-render preview cameras from a recorded state cache, without re-simulating.

Run the example once in scenic mode with ``--npz`` to produce the inputs:
``outputs/<name>/<name>.state.npz`` (per-frame link/body/particle state) and
``outputs/<name>/geometry.pkl`` (preview geometry). This CLI then replays any
camera configuration against them — the fast path for tuning the wrist mount:

    python -m examples cable_soft_franka --device cuda:0 --npz   # once
    python -m robolabViz.rerender outputs/cable_soft_franka/cable_soft_franka.state.npz \
        --geometry outputs/cable_soft_franka/geometry.pkl \
        --wrist-eye 0.05 0 0.03 --wrist-target 0 0 0.12 \
        --frames 0 170 270 420 600 --output-dir /tmp/wrist_tune
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import warp as wp

from .config import background_dome, droid_scene_config
from .raycast import RaycastPreviewRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_cache", help="<name>.state.npz written by a scenic demo run with --npz.")
    parser.add_argument("--geometry", required=True, help="geometry.pkl written next to the preview renders.")
    parser.add_argument("--output-dir", default="/tmp/robolab_rerender")
    parser.add_argument("--cameras", nargs="+", default=["wrist_camera"])
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Frame indices to render as PNGs (default: all frames as MP4).")
    parser.add_argument("--wrist-eye", type=float, nargs=3, default=None)
    parser.add_argument("--wrist-target", type=float, nargs=3, default=None)
    parser.add_argument("--wrist-up", type=float, nargs=3, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = np.load(args.state_cache, allow_pickle=False)
    link_names = [str(n) for n in data["link_names"]]
    link_tfs_all, body_q_all, particle_q_all = data["link_tfs"], data["body_q"], data["particle_q"]

    with open(args.geometry, "rb") as f:
        geom = pickle.load(f)

    scene = droid_scene_config()
    scene.sim_to_viz_translation = tuple(geom["sim_to_viz_translation"])
    scene.fps = float(data["fps"])
    # Reproduce the run's dome background (the table texture rides along in the
    # pickled instances; the background is sampled from scene.dome_light).
    if geom.get("dome_texture"):
        scene.dome_light = background_dome(geom["dome_texture"])
    if args.wrist_eye is not None:
        scene.wrist_camera.eye = tuple(args.wrist_eye)
    if args.wrist_target is not None:
        scene.wrist_camera.target = tuple(args.wrist_target)
    if args.wrist_up is not None:
        scene.wrist_camera.up = tuple(args.wrist_up)

    frames = args.frames if args.frames else range(len(link_tfs_all))
    renderer = RaycastPreviewRenderer(
        scene=scene,
        object_model=None,
        robot_link_names=None,
        output_dir=args.output_dir,
        device=wp.get_device(args.device),
        camera_names=args.cameras,
        frames_per_image=1 if args.frames else 0,
        instances=geom["instances"],
    )
    for frame in frames:
        link_tfs = {name: link_tfs_all[frame][i] for i, name in enumerate(link_names)}
        particles = wp.array(particle_q_all[frame], dtype=wp.vec3, device=wp.get_device(args.device))
        renderer._frame_idx = int(frame)
        renderer.render_frame(link_tfs, body_q_all[frame], particles)
    summary = renderer.close()
    if summary:
        print("wrist coverage:", {k: round(v, 3) for k, v in summary["wrist_mean_coverage"].items()})


if __name__ == "__main__":
    main()
