"""Thinning the raw candidate stream: deduplication, then a budgeted farthest-point selection.

Both methods over-produce by design — a flat facet yields one find per boundary sample, a tube one
per skeleton node — and the surplus is not extra coverage, it is the same grasp described many
times. Two stages fix that, both deterministic:

1. **Quantized dedup.** Grasps agreeing on position to :attr:`GeometricConfig.dedup_position` and on
   direction to :attr:`GeometricConfig.dedup_direction` collapse to the first one seen. Cheap and
   order-stable, which matters because the pass must be reproducible bit for bit.
2. **Farthest-point selection.** If a method is still over budget, keep the candidate set that
   spreads widest in pose space rather than an arbitrary prefix — greedy farthest-point sampling
   seeded from the first candidate of a deterministic sort. Truncating the list instead would bias
   every record toward whichever corner of the object the sweep happened to start in.

Position and direction are made comparable in stage 2 by scaling position by the object's own
radius, so "a centimetre away" and "ten degrees around" trade off at the object's scale rather than
at a hard-coded one.
"""
from __future__ import annotations

import numpy as np

from .config import GeometricConfig


def _unsigned(axis):
    """A direction with its sign fixed, for keys where the axis is a LINE not an arrow (the jaw
    closes the same way whichever pad you call first)."""
    a = np.asarray(axis, dtype=float)
    k = int(np.argmax(np.abs(a)))
    return -a if a[k] < 0 else a


def _key(candidate, cfg: GeometricConfig):
    pos = np.round(np.asarray(candidate.position) / cfg.dedup_position).astype(int)
    app = np.round(np.asarray(candidate.approach) / cfg.dedup_direction).astype(int)
    jaw = np.round(_unsigned(candidate.jaw_axis) / cfg.dedup_direction).astype(int)
    return tuple(pos), tuple(app), tuple(jaw)


def dedup(candidates, cfg: GeometricConfig) -> list:
    """First-wins deduplication over the quantized pose key, preserving input order."""
    seen, out = set(), []
    for c in candidates:
        k = _key(c, cfg)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _features(candidates, scale: float):
    """Pose as a comparable vector: scaled centre, approach, and sign-fixed jaw axis."""
    return np.array([
        np.concatenate([np.asarray(c.position) / scale,
                        np.asarray(c.approach),
                        _unsigned(c.jaw_axis)])
        for c in candidates], dtype=float)


def cap(candidates, limit: int, scale: float) -> list:
    """Thin to ``limit`` candidates by greedy farthest-point selection in pose space.

    Deterministic: the seed is the candidate whose id sorts first, and ties in the greedy step break
    on the lower index. Returns the kept candidates in their ORIGINAL order, so ids stay readable in
    the sidecar."""
    if len(candidates) <= limit:
        return list(candidates)

    order = sorted(range(len(candidates)), key=lambda i: candidates[i].id)
    feats = _features(candidates, max(scale, 1.0e-6))
    picked = [order[0]]
    dist = np.linalg.norm(feats - feats[picked[0]], axis=1)
    while len(picked) < limit:
        nxt = int(np.argmax(dist))                    # argmax takes the lowest index on a tie
        picked.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
    keep = set(picked)
    return [c for i, c in enumerate(candidates) if i in keep]
