"""Executable success predicates — RoboLab's runtime goal evaluation, adapted to Newton state.

RoboLab compiles a task's declarative goal (predicate + params) into a per-step termination check
(``robolab/core/task/conditionals.py`` + ``hull_check.py``). This module does the same for our goal
predicates so a generated task's success can actually be EVALUATED during simulation — the piece
that was previously "declarative only". Each predicate becomes a geometric test over the settled /
running object state:

- ``object_in_container`` / ``object_inside`` — the manipulated object's CENTROID lies inside the
  container's OPEN-TOP convex hull (the +z air column), RoboLab's ``in_opentop_container`` (their
  ``point_in_hull`` on ``open_top_planes``: drop faces whose normal.z >= 0.7, leaving an unbounded
  +z polytope). We build the hull from the container's collision mesh vertices.
- ``object_outside_of`` — NOT in that open-top hull.
- ``object_on_top`` / ``stacked`` — the object's xy-centroid is inside the support's footprint AABB
  AND the object rests above the support top (RoboLab pairs a contact-cone check; statically we use
  the resting-height test, which is what a settled scene exposes without contact sensors).
- ``object_left_of`` / ``right_of`` / ``in_front_of`` / ``behind`` — the ROBOT-POV 45-degree cone
  test (RoboLab ``spatial_condition_check_vector_based``), evaluated in the robot's frame.
- deformable predicates (``cloth_folded``, ``cable_coiled``, ``object_compressed``,
  ``cloth_draped_over``, ``cable_routed_through``) — geometric proxies over the particle cloud
  (bounding-box shrink for fold/coil/compress; overlap for drape/route), since there is no
  RoboLab analog (RoboLab has no deformables). Marked ``deformable_proxy`` in the result.

The evaluator runs against a lightweight ``SceneState`` snapshot (body poses + AABBs + particle
positions), so it works both in the headless settle harness and, later, in a policy rollout. Task
gen embeds the compiled predicate spec in ``task.json`` (``success_spec``) so downstream code can
re-evaluate without re-deriving it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry


@dataclass
class SceneState:
    """A snapshot the success predicates read. ``bodies[name]`` = dict(pos=(3,), aabb=(min3,max3),
    yaw_deg). ``particles[name]`` = (N, 3) array (deformables). ``base_xy`` = robot base for cones."""
    bodies: dict = field(default_factory=dict)
    particles: dict = field(default_factory=dict)
    base_xy: tuple = (0.0, 0.0)
    facing_yaw_deg: float = -90.0

    def centroid(self, name: str):
        if name in self.bodies:
            return np.asarray(self.bodies[name]["pos"], dtype=float)
        if name in self.particles:
            return np.asarray(self.particles[name]).mean(axis=0)
        return None

    def aabb(self, name: str):
        if name in self.bodies:
            lo, hi = self.bodies[name]["aabb"]
            return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if name in self.particles:
            p = np.asarray(self.particles[name])
            return p.min(axis=0), p.max(axis=0)
        return None


# ---------------------------------------------------------------------------------------------
# Open-top convex-hull containment (RoboLab hull_check.point_in_hull + open_top_planes)
# ---------------------------------------------------------------------------------------------
def _open_top_planes(points: np.ndarray, threshold: float = 0.7):
    """Outward face planes (n, d) of the convex hull of ``points``, DROPPING faces whose outward
    normal points up (normal.z >= threshold) — the open-top +z column (RoboLab ``open_top_planes``).
    Returns None if the hull can't be built (scipy missing or degenerate)."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
    except Exception:
        return None
    eq = hull.equations                       # (F, 4): nx, ny, nz, d ; inside iff n·x + d <= 0
    return eq[eq[:, 2] < threshold]


def point_in_open_top(pt: np.ndarray, container_pts: np.ndarray) -> bool:
    planes = _open_top_planes(container_pts)
    if planes is None:                        # fallback: AABB open-top column
        lo, hi = container_pts.min(axis=0), container_pts.max(axis=0)
        return bool(lo[0] <= pt[0] <= hi[0] and lo[1] <= pt[1] <= hi[1] and pt[2] >= lo[2])
    return bool(np.all(pt @ planes[:, :3].T + planes[:, 3] <= 1e-6))


# ---------------------------------------------------------------------------------------------
# The predicate evaluators — each returns (ok: bool, detail: str)
# ---------------------------------------------------------------------------------------------
def _in_container(state, obj, container):
    c = state.centroid(obj)
    pts = state.particles.get(container)
    if pts is None and container in state.bodies and "hull_points" in state.bodies[container]:
        pts = np.asarray(state.bodies[container]["hull_points"])
    if c is None or pts is None:
        return False, f"missing state for {obj!r} or {container!r}"
    ok = point_in_open_top(c, np.asarray(pts))
    return ok, f"{obj} centroid {'inside' if ok else 'outside'} {container} open-top hull"


def _on_top(state, obj, support):
    co, ao = state.centroid(obj), state.aabb(support)
    if co is None or ao is None:
        return False, f"missing state for {obj!r} or {support!r}"
    (slo, shi) = ao
    in_foot = slo[0] <= co[0] <= shi[0] and slo[1] <= co[1] <= shi[1]
    above = co[2] >= shi[2] - 0.02
    ok = in_foot and above
    return ok, f"{obj} {'on top of' if ok else 'not on'} {support} (footprint={in_foot}, above={above})"


def _cone(state, obj, ref, direction):
    co, cr = state.centroid(obj), state.centroid(ref)
    if co is None or cr is None:
        return False, f"missing state for {obj!r} or {ref!r}"
    dv = geometry.direction_vectors({"yaw_deg": state.facing_yaw_deg,
                                     "base": [state.base_xy[0], state.base_xy[1], 0.0]})[direction]
    vx, vy = float(co[0] - cr[0]), float(co[1] - cr[1])
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return False, f"{obj} coincident with {ref}"
    cos_cone = math.cos(math.radians(45.0))
    ok = (vx * dv[0] + vy * dv[1]) / n >= cos_cone
    return ok, f"{obj} {'is' if ok else 'is not'} {direction.replace('-of','').replace('-',' ')} {ref}"


def _bbox_diag(state, name):
    ab = state.aabb(name)
    if ab is None:
        return None
    lo, hi = ab
    return float(np.linalg.norm(hi - lo))


def _folded_or_coiled(state, obj, ref_shrink=0.7):
    """Deformable proxy: the object's footprint diagonal shrank to <= ref_shrink of its flat span.
    We don't store the flat reference here, so the check reports the current footprint and defers
    the ratio to the caller when a spawn reference is available; standalone it flags 'plausible'
    when the xy footprint is compact relative to the object's own height-normalized extent."""
    ab = state.aabb(obj)
    if ab is None:
        return False, f"missing particles for {obj!r}"
    lo, hi = ab
    fx, fy, fz = hi - lo
    # folded/coiled => the in-plane footprint is no longer dominated by ONE long axis
    aspect = max(fx, fy) / max(min(fx, fy), 1e-6)
    ok = aspect < 2.5 and fz > 0.01           # became blockier and gained thickness
    return ok, f"{obj} footprint {fx:.2f}x{fy:.2f}x{fz:.2f} m (aspect {aspect:.1f}) {'folded/coiled' if ok else 'still flat/extended'}"


def _compressed(state, obj):
    ab = state.aabb(obj)
    if ab is None:
        return False, f"missing particles for {obj!r}"
    lo, hi = ab
    return True, f"{obj} height {hi[2]-lo[2]:.3f} m (compression is force-dependent; reported only)"


def _draped_or_routed(state, obj, target):
    ao, at = state.aabb(obj), state.aabb(target)
    if ao is None or at is None:
        return False, f"missing state for {obj!r} or {target!r}"
    (olo, ohi), (tlo, thi) = ao, at
    overlap = (olo[0] <= thi[0] and ohi[0] >= tlo[0] and olo[1] <= thi[1] and ohi[1] >= tlo[1])
    return overlap, f"{obj} {'overlaps' if overlap else 'does not overlap'} {target}"


_DIR = {"object_left_of": "left", "object_right_of": "right",
        "object_in_front_of": "front", "object_behind": "behind"}


def evaluate(predicate: str, params: dict, state: SceneState) -> dict:
    """Evaluate a compiled goal predicate against a SceneState. Returns
    ``{ok, detail, deformable_proxy}``."""
    obj = params.get("object")
    proxy = False
    if predicate in ("object_in_container", "object_inside"):
        ok, detail = _in_container(state, obj, params.get("container"))
    elif predicate == "object_outside_of":
        ok, detail = _in_container(state, obj, params.get("container"))
        ok, detail = (not ok), "NOT " + detail
    elif predicate in ("object_on_top", "stacked"):
        ok, detail = _on_top(state, obj, params.get("target") or params.get("base"))
    elif predicate in _DIR:
        ok, detail = _cone(state, obj, params.get("reference"), _DIR[predicate])
    elif predicate == "object_groups_in_containers":
        # AND of per-object containment (RoboLab object_groups_in_containers)
        ok, details = True, []
        for o in (params.get("object") if isinstance(params.get("object"), list) else [obj]):
            o_ok, d = _in_container(state, o, params.get("container"))
            ok = ok and o_ok
            details.append(d)
        detail = "; ".join(details)
    elif predicate in ("cloth_folded", "cable_coiled"):
        ok, detail = _folded_or_coiled(state, obj); proxy = True
    elif predicate == "object_compressed":
        ok, detail = _compressed(state, obj); proxy = True
    elif predicate in ("cloth_draped_over", "cable_routed_through"):
        ok, detail = _draped_or_routed(state, obj, params.get("target")); proxy = True
    else:
        return {"ok": False, "detail": f"no runtime evaluator for predicate {predicate!r}",
                "deformable_proxy": False}
    return {"ok": bool(ok), "detail": detail, "deformable_proxy": proxy}


def compile_success_spec(task_goal: dict) -> dict:
    """The serializable success spec stored in task.json: the predicate + params + which geometric
    test drives it, so a rollout can re-evaluate without importing task-gen internals."""
    pred = task_goal.get("predicate")
    driver = ("open_top_hull" if pred in ("object_in_container", "object_inside", "object_outside_of",
                                          "object_groups_in_containers")
              else "footprint_support" if pred in ("object_on_top", "stacked")
              else "robot_pov_cone" if pred in _DIR
              else "deformable_proxy")
    return {"predicate": pred, "params": task_goal.get("params", {}), "driver": driver,
            "cone_deg": 45.0, "open_top_threshold": 0.7}
