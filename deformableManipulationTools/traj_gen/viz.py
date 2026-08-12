"""Lightweight grasp-attempt snapshot for the LLM retry loop — matplotlib, no GL, no ray tracing.

Three orthographic world-frame panels (top x-y, front x-z, side y-z) of the TARGET object's mesh
with the attempted grasp drawn on it: the two pad contact chords, the approach arrow into the TCP,
the tabletop line, nearby obstacle boxes, and the place point. Deliberately crude — the LLM needs
to see WHERE the jaws met the object and where it went, not a photoreal render.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..grasp_library import PAD_FAR_Z, PAD_NEAR_Z
from ..params import TABLE

_MAX_VERTS = 1200


def grasp_snapshot(out_png: Path | str, vertices_world: np.ndarray, pick, *,
                   obstacles=(), place_xy=None, track=None, title: str = "") -> Path:
    """Draw the attempted grasp. ``pick`` is a ``policy.PickSpec``; ``track`` an optional dict of
    labelled world points (e.g. measured object positions at phase ends) drawn as a path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(vertices_world, dtype=float)
    if len(v) > _MAX_VERTS:
        v = v[np.linspace(0, len(v) - 1, _MAX_VERTS).astype(int)]
    p = np.asarray(pick.position, dtype=float)
    a = np.asarray(pick.approach, dtype=float)
    j = np.asarray(pick.jaw_axis, dtype=float)
    half_w = 0.5 * float(pick.width)
    # Pad chords: each pad's centerline segment over the pad length, offset +-half_w along the jaw.
    pads = []
    for side in (-1.0, 1.0):
        c0 = p + side * half_w * j + PAD_NEAR_Z * a
        c1 = p + side * half_w * j + PAD_FAR_Z * a
        pads.append((c0, c1))
    arrow0 = p - 0.09 * a

    axes_pairs = ((0, 1, "top (x-y)"), (0, 2, "front (x-z)"), (1, 2, "side (y-z)"))
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (i0, i1, name) in zip(axs, axes_pairs):
        ax.scatter(v[:, i0], v[:, i1], s=1.2, c="#8899aa", alpha=0.55, linewidths=0)
        for c0, c1 in pads:
            ax.plot([c0[i0], c1[i0]], [c0[i1], c1[i1]], c="#d1342f", lw=2.4)
        ax.annotate("", xy=(p[i0], p[i1]), xytext=(arrow0[i0], arrow0[i1]),
                    arrowprops=dict(arrowstyle="-|>", color="#1a6faf", lw=1.8))
        ax.plot([p[i0]], [p[i1]], "o", ms=4, c="#1a6faf")
        if i1 == 2:
            ax.axhline(TABLE.top_z, c="#b98b46", lw=1.0, ls="--")
        for ob in obstacles:
            ctr, half = np.asarray(ob.center), np.asarray(ob.half)
            lo, hi = ctr - half, ctr + half
            ax.add_patch(plt.Rectangle((lo[i0], lo[i1]), hi[i0] - lo[i0], hi[i1] - lo[i1],
                                       fill=False, ec="#77aa77", lw=0.8))
        if place_xy is not None and i0 == 0 and i1 == 1:
            ax.plot([place_xy[0]], [place_xy[1]], "x", ms=8, c="#3a8a3a", mew=2)
        if track:
            pts = np.array(list(track.values()), dtype=float)
            ax.plot(pts[:, i0], pts[:, i1], ".-", ms=4, lw=0.9, c="#9955bb", alpha=0.9)
            for lbl, q in track.items():
                ax.annotate(lbl, (q[i0], q[i1]), fontsize=6, color="#7744aa")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out
