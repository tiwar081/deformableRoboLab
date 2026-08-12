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
- generic pose tests (no RoboLab analog, but each verified to separate true success from true
  failure): ``proximity`` (xy AABB gap <= tol) for push/pull-to, ``separation`` (xy AABB gap >=
  clearance) for clear-of/uncovered, ``robot_reach`` (distance from base) for retrieved, and
  ``height`` (bottom above support + clearance) for lifted.

SCORING IS DECOUPLED FROM GENERATION. Task gen may propose EVERY predicate in the library; this
module only says whether completion can be MEASURED yet. Predicates whose evaluator is not built
carry ``driver: null`` in ``goal_predicates.json``, and ``evaluate`` returns ``evaluable=False``
for them — never a pass, and never a silent failure either (a rollout must read it as *no score*,
not a failed attempt). ``pending_evaluators()`` lists them; what each one needs built is written up
in ``docs/agenticPipeline/success-evaluators.md``, kept out of the predicate table because task gen never reads it.

The deformable SHAPE evaluators (fold / coil / spread / compress / drape / thread / bag-mouth /
in-bag) were WITHDRAWN 2026-07-27 after probing each against a true-success and a true-failure
state: every one either false-positived or could never fire — a crumpled cloth read as "folded"; a
cloth folded in half read as "spread"; a flat coil never registered (the test demanded thickness a
tabletop coil has not got); a cable lying ABOVE a ring read as "threaded through", as did a box
merely sitting ON a cloth for "draped over"; ``object_compressed`` was hardcoded ``True``; the
bag-mouth tests needed ``mouth_open`` metadata no simulator emits; an object resting on a COLLAPSED
bag read as inside it. Common cause: an AABB cannot express SHAPE or TOPOLOGY. Deformable POSITION
goals (uncovered-from, clear-of, lifted) were kept: they reuse the generic geometry above and were
verified alongside it.

The evaluator runs against a lightweight ``SceneState`` snapshot (body poses + AABBs + particle
positions), so it works both in the headless settle harness and, later, in a policy rollout. Task
gen embeds the compiled predicate spec in ``task.json`` (``success_spec``) so downstream code can
re-evaluate without re-deriving it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry, load_goal_predicates


def _predicates() -> dict:
    """The predicate library (DATA — agentic_pipeline/goal_predicates.json), read here only for
    each predicate's ``driver``. Loaded LAZILY: the loader validates drivers against ``DRIVERS``
    below, so calling it at module scope would cycle."""
    global _PREDICATE_CACHE
    if _PREDICATE_CACHE is None:
        _PREDICATE_CACHE, _ = load_goal_predicates()
    return _PREDICATE_CACHE


_PREDICATE_CACHE = None


@dataclass
class SceneState:
    """A snapshot the success predicates read. ``bodies[name]`` = dict(pos=(3,), aabb=(min3,max3),
    yaw_deg), optionally with bag ``mouth_points``/``mouth_open`` metadata. ``particles[name]`` =
    (N, 3) deformable points (cloth/FEM particles or cable nodes). ``base_xy`` = robot base for
    cones/retrieval."""
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


def _aabb_gap_xy(state, first, second):
    aa, ab = state.aabb(first), state.aabb(second)
    if aa is None or ab is None:
        return None
    (alo, ahi), (blo, bhi) = aa, ab
    dx = max(float(blo[0] - ahi[0]), float(alo[0] - bhi[0]), 0.0)
    dy = max(float(blo[1] - ahi[1]), float(alo[1] - bhi[1]), 0.0)
    return math.hypot(dx, dy)


def _separated(state, obj, target, clearance=0.02):
    gap = _aabb_gap_xy(state, obj, target)
    if gap is None:
        return False, f"missing state for {obj!r} or {target!r}"
    ok = gap >= clearance
    return ok, f"{obj} is {gap:.3f} m clear of {target} (required {clearance:.3f} m)"


def _near(state, obj, target, tolerance=0.05):
    gap = _aabb_gap_xy(state, obj, target)
    if gap is None:
        return False, f"missing state for {obj!r} or {target!r}"
    ok = gap <= tolerance
    return ok, f"{obj} is {gap:.3f} m from {target} (required <= {tolerance:.3f} m)"


def _retrieved(state, obj, reach=0.45):
    c = state.centroid(obj)
    if c is None:
        return False, f"missing state for {obj!r}"
    dist = math.hypot(float(c[0]) - state.base_xy[0], float(c[1]) - state.base_xy[1])
    ok = dist <= reach
    return ok, f"{obj} is {dist:.3f} m from robot base (retrieved <= {reach:.3f} m)"


def _lifted(state, obj):
    ab = state.aabb(obj)
    if ab is None:
        return False, f"missing state for {obj!r}"
    lo, _ = ab
    meta = state.bodies.get(obj, {})
    support_z = float(meta.get("support_z", 0.07))
    clearance = float(meta.get("lift_clearance", 0.05))
    ok = float(lo[2]) >= support_z + clearance
    return ok, (f"{obj} bottom z={lo[2]:.3f} m "
                f"({'lifted' if ok else 'not lifted'}; threshold {support_z + clearance:.3f} m)")


_DIR = {"object_left_of": "left", "object_right_of": "right",
        "object_in_front_of": "front", "object_behind": "behind"}


def evaluate(predicate: str, params: dict, state: SceneState) -> dict:
    """Evaluate a compiled goal predicate against a SceneState. Returns
    ``{ok, evaluable, detail}``.

    ``evaluable=False`` means the goal is legitimate and task gen may propose it, but no VERIFIED
    evaluator exists yet (``driver: null`` in the predicate library) — ``ok`` is then False because
    nothing was measured, NOT because the robot failed. Callers scoring a rollout must treat an
    unevaluable goal as "no score", never as a failed attempt."""
    if predicate in pending_evaluators():
        return {"ok": False, "evaluable": False,
                "detail": f"no verified success evaluator for {predicate!r} yet "
                          f"(see docs/agenticPipeline/success-evaluators.md) — NOT scored, and not a failed attempt"}
    obj = params.get("object")
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
    elif predicate in ("cloth_uncovered_from", "cable_clear_of", "object_cleared_from"):
        ok, detail = _separated(state, obj, params.get("target"))
    elif predicate in ("object_pushed_to", "object_pulled_to"):
        ok, detail = _near(state, obj, params.get("target"))
    elif predicate == "object_retrieved":
        ok, detail = _retrieved(state, obj)
    elif predicate == "bag_lifted":
        ok, detail = _lifted(state, obj)
    else:
        return {"ok": False, "evaluable": False,
                "detail": f"unknown predicate {predicate!r} (not in the predicate library)"}
    return {"ok": bool(ok), "evaluable": True, "detail": detail}


# The drivers ``evaluate`` above actually implements. The loader cross-checks the predicate table
# against this, so a data-only edit cannot claim an evaluator that does not exist — a predicate
# without one must instead declare ``driver: null`` (build queue: docs/agenticPipeline/success-evaluators.md).
DRIVERS = {"open_top_hull", "footprint_support", "robot_pov_cone",
           "proximity", "separation", "robot_reach", "height"}


def pending_evaluators() -> set:
    """Predicates task gen can propose but cannot yet score (``driver: null``). What each one needs
    built is written up in docs/agenticPipeline/success-evaluators.md, deliberately not in the predicate table."""
    return {name for name, spec in _predicates().items() if spec.get("driver") is None}


def compile_success_spec(task_goal: dict) -> dict:
    """The serializable success spec stored in task.json: the predicate + params + which geometric
    test drives it, so a rollout can re-evaluate without importing task-gen internals.

    The driver is read from the predicate library (``goal_predicates.json``) — the SAME table task
    gen validates against — so the two can never disagree about a predicate."""
    pred = task_goal.get("predicate")
    spec = _predicates().get(pred)
    if spec is None:
        if pred != "object_inside":        # the one alias not in the table
            raise KeyError(f"no goal predicate {pred!r} in the predicate library")
        driver = "open_top_hull"
    else:
        driver = spec["driver"]
    # ``evaluable`` travels into task.json so a rollout knows a null driver means "cannot score
    # this goal yet", rather than reading the missing driver as a failed attempt.
    return {"predicate": pred, "params": task_goal.get("params", {}), "driver": driver,
            "evaluable": driver is not None, "cone_deg": 45.0, "open_top_threshold": 0.7}
