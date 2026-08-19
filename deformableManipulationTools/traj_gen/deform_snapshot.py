"""Settled-state snapshot of a scene's DEFORMABLE (cloth sheet / cable) for the online grasp LLM.

A cloth or cable has no rest-shape grasp record — it may be coiled or folded any way the settle
left it — so the online grasp proposal needs the ACTUAL settled material. This subprocess builds
the run's settle-only scene (the demo spec WITHOUT any traj policy), steps the settle window, and
emits:

- ``DEFORM_SNAPSHOT_JSON:`` — sampled world-frame material points (cable: every node in chain
  order; cloth: boundary extremes + a uniform subsample), centroid, z-range, xy extent;
- ``deform_snapshot<tag>.png`` in the run dir — top + side orthographic scatters with a labelled
  world-coordinate grid, the tabletop line, and the numbered sample points the LLM's prompt lists
  in text (image + numbers ground each other).

Run via :func:`run_snapshot` (subprocess — one Newton build per process, settle-harness pattern).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SNAP_TAG = "DEFORM_SNAPSHOT_JSON:"
SETTLE_FRAMES = 180           # 3 s — the settle checks use the same order of quiet time
_CLOTH_SAMPLES = 48


def run_snapshot(run_dir: Path | str, *, device: str = "cuda:0", tag: str = "",
                 verbose: bool = True) -> dict:
    cmd = [sys.executable, "-m", "deformableManipulationTools.traj_gen.deform_snapshot",
           str(run_dir), "--device", device, "--tag", tag]
    if verbose:
        print("[trajGen] deform snapshot:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in reversed((res.stdout or "").splitlines()):
        if line.startswith(SNAP_TAG):
            return json.loads(line[len(SNAP_TAG):])
    raise RuntimeError(f"deform snapshot produced no report (exit {res.returncode}):\n"
                       f"{res.stdout[-1500:]}\n{res.stderr[-1500:]}")


def _sample_cloth(pq: np.ndarray) -> np.ndarray:
    """Boundary extremes first (the corners the LLM most wants), then a uniform subsample."""
    picks = {int(np.argmin(pq[:, 0])), int(np.argmax(pq[:, 0])),
             int(np.argmin(pq[:, 1])), int(np.argmax(pq[:, 1])),
             int(np.argmax(pq[:, 2])),
             int(np.argmin(pq[:, 0] + pq[:, 1])), int(np.argmax(pq[:, 0] + pq[:, 1])),
             int(np.argmin(pq[:, 0] - pq[:, 1])), int(np.argmax(pq[:, 0] - pq[:, 1]))}
    uniform = np.linspace(0, len(pq) - 1, _CLOTH_SAMPLES - len(picks)).astype(int)
    order = list(dict.fromkeys(list(picks) + list(uniform)))
    return pq[order]


def _snapshot_png(out_png: Path, points: np.ndarray, all_pts: np.ndarray, table_z: float,
                  title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (i0, i1, name) in zip(axs, ((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
        ax.scatter(all_pts[:, i0], all_pts[:, i1], s=1.5, c="#8fa3b8", alpha=0.5, linewidths=0)
        ax.plot(points[:, i0], points[:, i1], ".", ms=5, c="#d1342f")
        for k, p in enumerate(points):
            if k % max(1, len(points) // 16) == 0:
                ax.annotate(str(k), (p[i0], p[i1]), fontsize=6, color="#8a1f1c")
        if i1 == 2:
            ax.axhline(table_z, c="#b98b46", lw=1.0, ls="--")
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(title + " — red numbered points = the sample list in the prompt", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    run_dir = Path(args.run_dir).resolve()

    import example as ex
    import examples
    import agentic_pipeline.build as build
    from deformableManipulationTools.params import TABLE

    # The SETTLE-ONLY spec: point demo_from_dir at a traj file that never exists, so an already-
    # written traj<k>.json from a previous attempt cannot leak its policy into the snapshot.
    spec = build.demo_from_dir(run_dir, traj_name="_no_traj_.json")
    spec.num_frames = SETTLE_FRAMES
    cls = type("DeformSnapshot", (ex._BaseExample,), {"spec": spec})
    parser = ex.make_parser(spec)
    sys.argv = [sys.argv[0], "--viewer", "null", "--device", args.device,
                "--output", "graphs", "--quiet"]
    viewer, run_args = examples.init(parser, example_name="deform_snapshot")
    demo = cls(viewer, run_args)
    for _ in range(SETTLE_FRAMES):
        demo.step()

    if demo.object_model is not None:
        labels = [str(lb).split("/")[-1] for lb in demo.object_model.body_label]
    else:
        labels = []
    bq = demo.object_body_q()
    cable_ids = [i for i, lb in enumerate(labels) if lb == "vbd_cable"]

    if cable_ids:
        all_pts = bq[cable_ids, :3]
        points = all_pts                        # every node, in chain order
        kind = "cable"
    elif demo.object_model is not None and demo.object_model.particle_count:
        all_pts = demo.object_state_0.particle_q.numpy()
        points = _sample_cloth(all_pts)
        kind = "cloth"
    else:
        print(SNAP_TAG + json.dumps({"ok": False, "reason": "no deformable in the scene"}))
        return

    png = run_dir / f"deform_snapshot{args.tag}.png"
    _snapshot_png(png, points, all_pts, TABLE.top_z, f"settled {kind}")
    report = {
        "ok": True, "kind": kind, "png": str(png),
        "points": [[round(float(v), 4) for v in p] for p in points],
        "centroid": [round(float(v), 4) for v in all_pts.mean(axis=0)],
        "z_min": round(float(all_pts[:, 2].min()), 4),
        "z_max": round(float(all_pts[:, 2].max()), 4),
        "extent_xy": [round(float(all_pts[:, 0].max() - all_pts[:, 0].min()), 4),
                      round(float(all_pts[:, 1].max() - all_pts[:, 1].min()), 4)],
        "n_material_points": int(len(all_pts)),
    }
    print(SNAP_TAG + json.dumps(report))


if __name__ == "__main__":
    main()
