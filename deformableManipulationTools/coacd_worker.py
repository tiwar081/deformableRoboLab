"""Standalone coacd convex-decomposition worker (run as a SUBPROCESS, by absolute path).

`import coacd` and `import newton` in the same process segfault (a native-library conflict),
so the decomposition is isolated here: this script imports ONLY numpy + coacd, never newton.
`examples.mesh_collision.convex_decompose` spawns it on a cache miss; the Newton simulation
process only ever reads the resulting cache, so coacd's native lib never co-loads with Newton.

Usage (internal): python /abs/path/coacd_worker.py <request.npz> <out.npz>
  request.npz: verts(float64 Nx3), faces(int32 Mx3), and the run_coacd scalar params.
  out.npz:     count, then v0/f0, v1/f1, ... per convex piece (float32 / int32).
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    req_path, out_path = sys.argv[1], sys.argv[2]
    req = np.load(req_path)
    import coacd

    coacd.set_log_level("error")
    mesh = coacd.Mesh(np.asarray(req["verts"], dtype=np.float64),
                      np.asarray(req["faces"], dtype=np.int32))
    parts = coacd.run_coacd(
        mesh,
        threshold=float(req["threshold"]),
        max_convex_hull=int(req["max_convex_hull"]),
        preprocess_mode="on",  # REQUIRED: 'auto' segfaults on raw non-watertight YCB scans
        preprocess_resolution=int(req["preprocess_resolution"]),
        max_ch_vertex=int(req["max_ch_vertex"]),
        merge=True,
        seed=int(req["seed"]),
    )
    out: dict[str, np.ndarray] = {"count": np.int32(len(parts))}
    for i, (v, f) in enumerate(parts):
        out[f"v{i}"] = np.asarray(v, dtype=np.float32)
        out[f"f{i}"] = np.asarray(f, dtype=np.int32).reshape(-1, 3)
    np.savez_compressed(out_path, **out)


if __name__ == "__main__":
    main()
