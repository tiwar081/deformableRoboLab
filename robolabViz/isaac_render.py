"""Offline RTX rendering of a produced demo USD with Isaac Sim (RoboLab's renderer).

Run this on an RTX-capable machine (RT cores required — the Omniverse RTX
renderer does not run on A100/H100/H200; on those GPUs use the warp raycast
preview the demo writes alongside the USD, i.e. the default scenic output). The
USD is produced by running a demo in scenic mode with ``--usd``:

    OMNI_KIT_ALLOW_ROOT=1 python -m robolabViz.isaac_render \
        outputs/cable_soft_franka/cable_soft_franka.usd \
        --cameras over_shoulder_left_camera wrist_camera \
        --output-dir outputs/robolab_rtx

It opens the time-sampled USD written by ``RoboLabStageWriter``, attaches a
replicator render product to each named camera prim, steps the stage timeline
one time-code per frame, and writes one MP4 per camera — the same RTX pipeline
(dome-lit home-office background, MDL maple table, OmniPBR robot materials)
that RoboLab's ``run_recorded.py`` uses for its videos.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", help="Time-sampled USD produced by a scenic demo run with --usd.")
    parser.add_argument("--cameras", nargs="+", default=["over_shoulder_left_camera", "wrist_camera"])
    parser.add_argument("--output-dir", default="outputs/robolab_rtx")
    parser.add_argument("--rt-subframes", type=int, default=8, help="RTX accumulation subframes per frame.")
    parser.add_argument("--fps", type=float, default=None, help="Video fps (default: stage timeCodesPerSecond).")
    args = parser.parse_args()

    import cv2  # noqa: F401  (import before kit)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "open_usd": args.usd_path})

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    fps = args.fps or stage.GetTimeCodesPerSecond() or 60.0
    start = int(stage.GetStartTimeCode())
    end = int(stage.GetEndTimeCode())

    camera_paths = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera) and prim.GetName() in args.cameras:
            camera_paths[prim.GetName()] = str(prim.GetPath())
    missing = set(args.cameras) - set(camera_paths)
    if missing:
        raise SystemExit(f"Cameras not found in stage: {sorted(missing)}")

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    annotators, writers = {}, {}
    for name, path in camera_paths.items():
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        # resolution from aperture ratio at 720p-equivalent height authored by the writer
        res = (1280, 720) if cam.GetHorizontalApertureAttr().Get() < 10 else (864, 480)
        rp = rep.create.render_product(path, res)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        annotators[name] = (ann, res)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(start / fps)
    timeline.set_end_time(end / fps)

    from robolabViz.video import VideoWriter

    for frame in range(start, end + 1):
        timeline.set_current_time(frame / fps)
        rep.orchestrator.step(rt_subframes=args.rt_subframes, pause_timeline=True)
        for name, (ann, res) in annotators.items():
            data = ann.get_data()
            if data is None or getattr(data, "size", 0) == 0:
                continue
            if name not in writers:
                writers[name] = VideoWriter(os.path.join(args.output_dir, f"{name}.mp4"), fps)
            writers[name].write(data[..., :3])
        if frame % 60 == 0:
            print(f"[robolabViz] rendered frame {frame}/{end}")

    for w in writers.values():
        w.release()
    print(f"[robolabViz] RTX renders written to {args.output_dir}")
    app.close()


if __name__ == "__main__":
    sys.exit(main())
