"""Headless settle-metrics harness (run as ``python -m agentic_pipeline.settle <demo.py>``).

Builds the generated demo, steps its full settle window WITHOUT rendering, and reports physics
placement metrics as one ``SETTLE_JSON: {...}`` line on stdout (warp/newton chatter precedes it):

  finite              — no NaN/Inf anywhere in the object state (blow-up check)
  bodies              — per-body {label, displacement} (gripper proxies excluded)
  off_table           — body labels whose final pose left the tabletop (fell off / below top)
  large_displacement  — body labels that drifted > 5 cm from their spawn (RoboLab's ~2 cm
                        instability flag, loosened because our spawns intentionally DROP a little)
  particle_drift      — mean deformable-particle drift [m] (informational; cloth drapes move)
  residual_motion     — deformables still moving at the end (max particle speed > 0.05 m/s)
  ok                  — finite AND nothing off the table AND no residual motion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=None, help="override the spec's settle length")
    args = ap.parse_args()

    import example as ex
    import examples
    from deformableManipulationTools.params import TABLE

    cls, parser = ex.example_for(args.demo)
    sys.argv = [sys.argv[0], "--viewer", "null", "--device", args.device, "--output", "graphs", "--quiet"]
    viewer, run_args = examples.init(parser, example_name="settle_check")
    demo = cls(viewer, run_args)
    frames = args.frames or demo.spec.num_frames

    labels = list(demo.object_model.body_label) if demo.object_model is not None else []
    keep = [i for i, lb in enumerate(labels)
            if not any(t in str(lb).lower() for t in ("proxy", "palm", "finger"))]
    bq0 = demo.object_body_q()
    pq0 = (demo.object_state_0.particle_q.numpy().copy()
           if demo.object_model is not None and demo.object_model.particle_count else None)

    for _ in range(frames):
        demo.step()

    bq1 = demo.object_body_q()
    finite = bool(np.all(np.isfinite(bq1))) if bq1.size else True
    bodies, off_table, large_disp = [], [], []
    x0 = TABLE.pos[0] - TABLE.half[0] - 0.10
    x1 = TABLE.pos[0] + TABLE.half[0] + 0.10
    y0 = TABLE.pos[1] - TABLE.half[1] - 0.10
    y1 = TABLE.pos[1] + TABLE.half[1] + 0.10
    if bq1.size:
        for i in keep:
            lb = str(labels[i]).split("/")[-1]
            d = float(np.linalg.norm(bq1[i, :3] - bq0[i, :3])) if finite else float("nan")
            bodies.append({"label": lb, "displacement": round(d, 4) if np.isfinite(d) else None})
            if not finite:
                continue
            ex_, ey_, ez_ = bq1[i, :3]
            if not (x0 <= ex_ <= x1 and y0 <= ey_ <= y1) or ez_ < TABLE.top_z - 0.05:
                off_table.append(lb)
            elif d > 0.05:
                large_disp.append(lb)

    particle_drift = residual = None
    if pq0 is not None:
        pq1 = demo.object_state_0.particle_q.numpy()
        pqd = demo.object_state_0.particle_qd.numpy()
        finite = finite and bool(np.all(np.isfinite(pq1)))
        if finite:
            particle_drift = round(float(np.linalg.norm(pq1 - pq0, axis=1).mean()), 4)
            residual = bool(np.abs(pqd).max() > 0.05)
            if float(pq1[:, 2].min()) < TABLE.top_z - 0.05:
                off_table.append("deformable_particles")

    report = {
        "finite": finite,
        "bodies": bodies,
        "off_table": sorted(set(off_table)),
        "large_displacement": sorted(set(large_disp)),
        "particle_drift": particle_drift,
        "residual_motion": bool(residual) if residual is not None else False,
        "frames": frames,
    }
    report["ok"] = bool(finite and not report["off_table"] and not report["residual_motion"])
    print("SETTLE_JSON:" + json.dumps(report))


if __name__ == "__main__":
    main()
