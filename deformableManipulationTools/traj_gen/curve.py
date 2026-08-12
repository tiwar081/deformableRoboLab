"""Bezier transport legs with COLLISION-DRIVEN control-point insertion (no LLM, no search).

A trajectory's free-space legs (parked -> pre-grasp, lift-top -> place-standoff) are single Bezier
segments. The control polygon starts as the straight chord; the sampled spline is then tested
against a :class:`CollisionField` (tabletop plane + the scene's obstacle boxes, inflated for
whatever the hand carries), and every colliding stretch inserts ONE control point above its deepest
sample — pushed past the blocking top, with an overshoot that grows per iteration because a Bezier
bends TOWARD a control point without reaching it. Repeat until the sampled spline is clear (or the
iteration cap reports failure honestly; the caller aborts rather than executing a colliding path).

The grasp approach and lift legs are NOT routed here — they are straight runs along the candidate's
approach axis, already validated by ``grasp_select``'s corridor clearance, and bending them would
execute a different procedure than the library measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grasp_select.clearance import Obstacle

# Samples along a routed leg. 96 over a <=1 m leg is ~1 cm spacing — finer than any catalog
# obstacle's smallest half-extent, so a box cannot slip between samples.
ROUTE_SAMPLES = 96
# Clearance added above a blocking obstacle top when inserting a control point [m].
CLEAR_MARGIN = 0.03
# Per-iteration overshoot growth [m]: the curve undershoots its control polygon, so each pass that
# still collides pushes the inserted/raised points further past the blocking top.
OVERSHOOT_STEP = 0.04
# Two control points closer than this in curve parameter are one point (raise, don't duplicate).
_MERGE_DU = 0.08
# Iteration cap — with OVERSHOOT_STEP growth this spans >0.3 m of extra lift, beyond any
# tabletop obstacle; hitting it means the leg is genuinely unroutable (reported, never executed).
MAX_ROUTE_ITERS = 8


def bezier(control: np.ndarray, n: int = ROUTE_SAMPLES) -> np.ndarray:
    """Evaluate one Bezier segment (any control count >= 2) at ``n`` uniform parameters."""
    c = np.asarray(control, dtype=float)
    u = np.linspace(0.0, 1.0, n)
    pts = np.repeat(c[None, :, :], n, axis=0)          # (n, k, 3) de Casteljau
    while pts.shape[1] > 1:
        pts = (1.0 - u)[:, None, None] * pts[:, :-1, :] + u[:, None, None] * pts[:, 1:, :]
    return pts[:, 0, :]


@dataclass
class CollisionField:
    """What a routed leg must stay clear of: the tabletop plane and the scene's obstacle boxes.

    ``inflate`` grows every box isotropically (hand half-profile, plus the held object's largest
    half-extent when carrying); ``floor_z`` is the lowest z the TCP may cruise at (tabletop + hand
    clearance, plus the held object's hang-below-TCP depth when carrying)."""
    obstacles: tuple = ()
    floor_z: float = 0.0
    inflate: float = 0.0
    ceiling_z: float = 10.0        # don't route above the arm's useful workspace

    def blocker(self, p) -> Obstacle | None:
        for ob in self.obstacles:
            if ob.contains(p, self.inflate):
                return ob
        return None

    def hit(self, p) -> bool:
        return p[2] < self.floor_z or self.blocker(p) is not None

    def clear_z(self, p) -> float:
        """The z at which point ``p``'s xy-column is clear: above the floor and above the top of
        every (inflated) obstacle whose column contains it."""
        z = self.floor_z
        probe = np.array([p[0], p[1], 0.0])
        for ob in self.obstacles:
            top = float(ob.center[2]) + float(ob.half[2]) + self.inflate
            probe[2] = min(top - 1.0e-4, float(ob.center[2]))   # test xy membership inside the box
            if ob.contains(probe, self.inflate):
                z = max(z, top)
        return z + CLEAR_MARGIN


@dataclass
class RoutedLeg:
    """One routed transport leg: the final control polygon, the sampled points, and honesty."""
    control: np.ndarray            # (k, 3) control polygon actually used
    points: np.ndarray             # (ROUTE_SAMPLES, 3) sampled spline
    inserted: int                  # control points added by collision resolution
    iterations: int
    clear: bool                    # False -> the cap was hit and the leg still collides
    blockers: tuple = field(default_factory=tuple)   # names seen while resolving

    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


def route(p0, p1, fld: CollisionField, *, samples: int = ROUTE_SAMPLES,
          max_iters: int = MAX_ROUTE_ITERS) -> RoutedLeg:
    """Bezier from ``p0`` to ``p1`` that clears ``fld``, by inserting control points over collisions.

    Endpoints are NEVER moved: they are the pre-grasp / place standoffs whose own clearance the
    selection stage already established."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ctrl: list[tuple[float, np.ndarray]] = [(0.0, p0), (1.0, p1)]   # (parameter, point)
    blockers: list[str] = []
    inserted = 0
    for it in range(max_iters):
        pts = bezier(np.array([c for _, c in ctrl]), samples)
        u = np.linspace(0.0, 1.0, samples)
        colliding = np.array([fld.hit(p) for p in pts])
        colliding[0] = colliding[-1] = False           # endpoints are the caller's contract
        if not colliding.any():
            return RoutedLeg(control=np.array([c for _, c in ctrl]), points=pts,
                             inserted=inserted, iterations=it, clear=True,
                             blockers=tuple(dict.fromkeys(blockers)))
        # One control point per contiguous colliding run, above the run's most-buried sample.
        overshoot = OVERSHOOT_STEP * (it + 1)
        runs = _runs(colliding)
        for a, b in runs:
            i = (a + b) // 2                            # mid-sample of the run
            ob = fld.blocker(pts[i])
            if ob is not None:
                blockers.append(ob.name)
            zc = min(fld.clear_z(pts[i]) + overshoot, fld.ceiling_z)
            cp = np.array([pts[i][0], pts[i][1], zc])
            near = [j for j, (uu, _) in enumerate(ctrl) if 0.0 < uu < 1.0 and abs(uu - u[i]) < _MERGE_DU]
            if near:                                    # raise the existing point, don't stack a new one
                j = near[0]
                ctrl[j] = (ctrl[j][0], np.array([ctrl[j][1][0], ctrl[j][1][1],
                                                 max(ctrl[j][1][2], zc)]))
            else:
                ctrl.append((float(u[i]), cp))
                inserted += 1
        ctrl.sort(key=lambda t: t[0])
    pts = bezier(np.array([c for _, c in ctrl]), samples)
    return RoutedLeg(control=np.array([c for _, c in ctrl]), points=pts, inserted=inserted,
                     iterations=max_iters, clear=False, blockers=tuple(dict.fromkeys(blockers)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of ``mask`` as (start, end) inclusive indices."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def leg_waypoints(leg: RoutedLeg, t0: float, speed: float, *, n_via: int = 5,
                  min_duration: float = 0.4) -> list[tuple[float, np.ndarray, bool]]:
    """Downsample a routed leg into timed waypoints: ``(t, pos, via)`` with arc-length-uniform
    spacing and constant-speed timing. Interior points are via-marked (LINEAR blend — the executor
    eases only into the endpoints, so the TCP doesn't pulse at every knot). The leg's START point is
    NOT emitted (it is the previous leg's endpoint)."""
    pts = leg.points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    duration = max(total / max(speed, 1.0e-6), min_duration)
    if total < 1.0e-6:
        return [(t0 + duration, pts[-1].copy(), False)]
    out = []
    targets = np.linspace(0.0, total, n_via + 2)[1:]    # skip the start point
    for k, sk in enumerate(targets):
        i = int(np.searchsorted(s, sk, side="left").clip(1, len(s) - 1))
        w = (sk - s[i - 1]) / max(s[i] - s[i - 1], 1.0e-9)
        p = (1.0 - w) * pts[i - 1] + w * pts[i]
        out.append((t0 + duration * (sk / total), p, k < len(targets) - 1))
    return out
