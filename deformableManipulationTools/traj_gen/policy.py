"""Task + scene + selected grasp -> the executable trajectory plan (``traj.json``).

The plan is pure DATA — timed TCP waypoints in the executor's ``(pos, yaw, tilt, tilt_axis)``
vocabulary plus ONE ``GraspWindow`` — assembled from straight validated legs and Bezier transport
legs (:mod:`.curve`):

    parked --(bezier cruise)--> pre-grasp --(straight descend)--> grasp [close] --(straight
    retreat)--> pre-grasp --(bezier carry, obstacle boxes inflated by the held object)--> place
    standoff --(straight lower)--> release [open] --> retreat + park

The approach and lift NEVER bend: they are the corridor ``grasp_select`` cleared, along the
candidate's own approach axis. The fingers pre-shape to ``width + PREGRASP_MARGIN`` for the whole
approach (``GraspWindow.preshape_width`` — the grasp library's contract). The grasp force target is
derived, not tuned: the shake pass's Coulomb law at gentler transport accelerations
(F = 2·m·(g + a)/(2·µ_eff), clamped to the same [1, 40] N envelope).

Goal placement is resolved from the task's PREDICATE (put-in -> above the container mouth,
on-top/stacked -> above the support's top, robot-POV direction words -> beside the reference,
push/pull-to -> adjacent, retrieved -> near the base), with a free-spot spiral search so the
set-down never lands inside another object's footprint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..grasp_library import (MAX_JAW_WIDTH, PAD_FAR_Z, PAD_HALF_WIDTH, PAD_NEAR_Z,
                             PREGRASP_MARGIN)
from ..grasp_select import RobotState
from ..grasp_select.clearance import DEFAULT_STANDOFF, Obstacle
from ..params import GRIP, TABLE
from . import curve
from .selection import RankedGrasp

# ---- timing/speed constants (from the hand-written demos' measured comfortable ranges) ----
SETTLE_T = 1.5            # [s] scene settles physically before the arm moves
CRUISE_SPEED = 0.28       # [m/s] free-space transport (empty hand)
CARRY_SPEED = 0.20        # [m/s] transport with the object held
DESCEND_SPEED = 0.07      # [m/s] final approach / place lowering
LIFT_SPEED = 0.12         # [m/s] retreat along the approach after the close
PRE_CLOSE_DWELL = 0.3     # [s] settle at the grasp pose before closing
CLOSE_DUR = 1.1           # [s] the close ramp window
POST_CLOSE_DWELL = 0.5    # [s] hold after the close before moving
# CLOTH pinches take longer, measured (examples/cloth_franka_green): the admittance pinch on a
# pressed sheet closes over ~2 s and must KEEP pressing ~3 s more before the lift, or the lift
# leaves the sheet behind.
CLOTH_CLOSE_DUR = 2.0
CLOTH_POST_CLOSE_DWELL = 3.0
RELEASE_DUR = 0.7         # [s] the reopen ramp
POST_RELEASE_DWELL = 0.4  # [s] hold after release before retreating
END_TAIL = 0.6            # [s] trailing frames after the last waypoint
PLACE_STANDOFF = 0.12     # [m] vertical standoff above the release pose
FPS = 60

# Hand half-profile for the EMPTY-hand cruise inflation [m]: half the measured palm-hull length
# (204 mm — grasp-library.md), the widest thing the cruising hand sweeps.
HAND_HALF_PROFILE = 0.102

# ---- derived force target (the shake pass's law at transport accelerations) ----
FORCE_SAFETY = 2.0
FORCE_MIN, FORCE_MAX = 1.0, 40.0
A_TRANSPORT = 3.0         # [m/s^2] peak planned acceleration while carrying


def derived_force_target(mass: float, mu_object: float) -> float:
    """F = safety * m * (g + a) / (2 * mu_eff); mu_eff couples the object's authored friction with
    the rubber pad by the central pairing-blind geometric-mean law."""
    mu_eff = max(math.sqrt(max(float(mu_object), 1.0e-3) * GRIP.proxy_mu), 1.0e-3)
    f = FORCE_SAFETY * float(mass) * (9.81 + A_TRANSPORT) / (2.0 * mu_eff)
    return float(min(max(f, FORCE_MIN), FORCE_MAX))


@dataclass(frozen=True)
class PickSpec:
    """The one grasp the plan executes: a candidate's world pose in the executor's vocabulary."""
    id: str
    pose: np.ndarray               # 4x4 world <- grasp (v2: +z approach, +x jaw, origin at TCP)
    yaw: float
    tilt: float
    tilt_axis: tuple
    width: float
    seat_mode: str = ""
    source: str = ""
    tier: int = 1
    score: float = 0.5
    quality: dict = field(default_factory=dict)
    adjusted: dict | None = None   # LLM adjustment applied on top of the stored candidate, if any

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.pose)[:3, 3]

    @property
    def approach(self) -> np.ndarray:
        return np.asarray(self.pose)[:3, 2]

    @property
    def jaw_axis(self) -> np.ndarray:
        return np.asarray(self.pose)[:3, 0]

    @property
    def preshape_width(self) -> float:
        return float(min(self.width + PREGRASP_MARGIN, MAX_JAW_WIDTH))


# How far a pad corner may dip below the tabletop [m] — the deliberate press allowance is a CLOTH
# recipe; a rigid pick gets a few mm of tolerance for pad chamfer, no more.
PAD_TABLE_TOL = 0.004
# The finger body extends this far beyond the commanded pad face along the jaw axis [m].
_FINGER_OUTER = 0.012


def pad_lowest_z(pose, width: float) -> float:
    """The lowest world z any pad corner reaches at this grasp pose and jaw width.

    Measured need (2026-08-12, demo_kitchen_putaway): a 3.3 cm-tall tuna can's candidates all jaw
    across its HEIGHT, so the lower finger must occupy the space between the can's bottom and the
    table — impossible for a resting object, invisible to the corridor clearance (which samples
    the approach, not the jaw sweep at the grasp) and to the table-less shake rig. Three arm-
    blocked rollouts (TCP jammed 60 mm short) bought this check."""
    m = np.asarray(pose, dtype=float)
    p, jaw, lat, app = m[:3, 3], m[:3, 0], m[:3, 1], m[:3, 2]
    zmin = float("inf")
    for side in (-1.0, 1.0):
        for along in (PAD_FAR_Z, PAD_NEAR_Z):
            for dl in (-PAD_HALF_WIDTH, PAD_HALF_WIDTH):
                for out in (0.0, _FINGER_OUTER):
                    q = p + (side * (0.5 * float(width) + out)) * jaw + dl * lat + along * app
                    zmin = min(zmin, float(q[2]))
    return zmin


def pads_clear_table(pick: "PickSpec", table_z: float | None = None) -> tuple:
    """(ok, lowest_z): the PRE-SHAPED jaw sweep must stay above the tabletop."""
    z = pad_lowest_z(pick.pose, pick.preshape_width)
    top = TABLE.top_z if table_z is None else float(table_z)
    return z >= top - PAD_TABLE_TOL, z


def pick_from_ranked(r: RankedGrasp) -> PickSpec:
    c = r.grasp.candidate
    return PickSpec(id=r.id, pose=np.asarray(r.grasp.pose, dtype=float),
                    yaw=float(r.grasp.command.yaw), tilt=float(r.grasp.command.tilt),
                    tilt_axis=tuple(r.grasp.command.tilt_axis), width=float(c.width),
                    seat_mode=getattr(c, "seat_mode", ""), source=getattr(c, "source", ""),
                    tier=r.tier, score=float(r.grasp.score.total),
                    quality=dict(getattr(c, "quality", None) or {}))


# =================================================================================================
# Scene geometry helpers
# =================================================================================================
def scene_obstacles(scene: dict, catalog_by_name: dict, *, exclude_indices=()) -> list:
    """World obstacle boxes for the scene's objects (settled x/y/yaw, resting on the tabletop).

    Center z = tabletop + half height: generated scenes rest objects on the table (stacks add a few
    cm we deliberately ignore — the inflation margins dominate). Deformables without dims are
    skipped, like ``grasp_select.clearance.obstacles_from_scene``."""
    out = []
    skip = set(exclude_indices)
    for i, o in enumerate(scene.get("objects", ())):
        if i in skip:
            continue
        dims = catalog_by_name.get(o.get("name"), {}).get("dims")
        if not dims:
            continue
        out.append(Obstacle(name=f"{o['name']}#{i}",
                            center=(float(o.get("x", 0.0)), float(o.get("y", 0.0)),
                                    TABLE.top_z + 0.5 * float(dims[2])),
                            half=(0.5 * float(dims[0]), 0.5 * float(dims[1]),
                                  0.5 * float(dims[2])),
                            yaw=math.radians(float(o.get("yaw_deg", 0.0)))))
    return out


def object_indices(scene: dict, name: str) -> list[int]:
    return [i for i, o in enumerate(scene.get("objects", ())) if o.get("name") == name]


# =================================================================================================
# Goal resolution — predicate -> place pose
# =================================================================================================
# predicate -> (role holding the reference object, placement mode)
SUPPORTED_GOALS = {
    "object_in_container": ("container", "drop_in"),
    "object_groups_in_containers": ("container", "drop_in"),
    "object_on_top": ("target", "set_on"),
    "stacked": ("base", "set_on"),
    "object_left_of": ("reference", "beside:left"),
    "object_right_of": ("reference", "beside:right"),
    "object_in_front_of": ("reference", "beside:front"),
    "object_behind": ("reference", "beside:behind"),
    "object_pushed_to": ("target", "adjacent"),
    "object_pulled_to": ("target", "adjacent"),
    "object_retrieved": (None, "retrieve"),
    "object_cleared_from": ("target", "away"),
    "object_outside_of": ("container", "away"),
}


class PlanError(RuntimeError):
    """The task cannot be turned into an executable plan (unsupported goal, no free spot, ...)."""


def _entry_dims(catalog_by_name: dict, name: str) -> np.ndarray:
    dims = catalog_by_name.get(name, {}).get("dims") or (0.08, 0.08, 0.08)
    return np.asarray(dims, dtype=float)


def _xy_half(dims: np.ndarray) -> float:
    return 0.5 * float(max(dims[0], dims[1]))


def _free_spot(desired_xy, need_radius: float, obstacles, base_xy, *, reach_max: float = 0.78):
    """The nearest point to ``desired_xy`` whose footprint disc is on the table, in reach, and
    outside every obstacle box: the desired point first, then a spiral of nudges."""
    lo = (TABLE.pos[0] - TABLE.half[0] + need_radius + 0.03,
          TABLE.pos[1] - TABLE.half[1] + need_radius + 0.03)
    hi = (TABLE.pos[0] + TABLE.half[0] - need_radius - 0.03,
          TABLE.pos[1] + TABLE.half[1] - need_radius - 0.03)

    def ok(p):
        if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]):
            return False
        if math.hypot(p[0] - base_xy[0], p[1] - base_xy[1]) > reach_max:
            return False
        probe = (p[0], p[1], TABLE.top_z + 0.02)
        return all(not ob.contains(probe, need_radius) for ob in obstacles)

    d = np.asarray(desired_xy, dtype=float)
    if ok(d):
        return d
    for r in np.arange(0.03, 0.25, 0.03):
        for a in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
            p = d + r * np.array([math.cos(a), math.sin(a)])
            if ok(p):
                return p
    raise PlanError(f"no free place spot near ({d[0]:+.2f}, {d[1]:+.2f}) "
                    f"(need {need_radius * 100:.0f} cm clearance)")


def resolve_place(scene: dict, task: dict, placement: dict, catalog_by_name: dict,
                  target_name: str, target_xy, extra_obstacles=()) -> dict:
    """The predicate's place point: ``{xy, surface_z, gap, mode, reference}``.

    ``surface_z`` is where the OBJECT'S BOTTOM should end up (container mouth top for a drop-in,
    support top for a set-on, the tabletop otherwise); ``gap`` is the extra release height above
    it (a drop-in lets go higher, a set-down releases just above contact)."""
    goal = task.get("goal") or {}
    pred = goal.get("predicate")
    if pred not in SUPPORTED_GOALS:
        raise PlanError(f"goal predicate {pred!r} is not executable by the trajectory stage")
    role, mode = SUPPORTED_GOALS[pred]
    params = goal.get("params") or {}
    ref_name = params.get(role) if role else None
    if role and not ref_name:
        # task gen occasionally mislabels the role (its own feasibility flags it); take any param
        # that names a scene object and is not the target.
        ref_name = next((v for k, v in params.items()
                         if k != "object" and object_indices(scene, str(v))), None)
        if not ref_name:
            raise PlanError(f"goal {pred!r} names no resolvable reference object (params {params})")

    from agentic_pipeline import geometry  # lazy: keep the physics package import-light

    base_xy = (float(placement["base"][0]), float(placement["base"][1]))
    obj_dims = _entry_dims(catalog_by_name, target_name)
    obj_half = _xy_half(obj_dims)
    # A dropped SHEET/CABLE crumples: its set-down footprint is a fraction of the flat extent
    # (measured need: a 0.62 m t-shirt demanded a 33 cm free disc no tabletop has; the settled
    # drop occupies far less). Rigid footprints stay exact.
    if catalog_by_name.get(target_name, {}).get("kind") in ("cloth", "cable"):
        obj_half = max(0.05, 0.4 * obj_half)
    obstacles = scene_obstacles(scene, catalog_by_name,
                                exclude_indices=object_indices(scene, target_name)) \
        + list(extra_obstacles)
    surface_z, gap = TABLE.top_z, 0.004
    ref_entry = None
    if ref_name:
        idx = object_indices(scene, str(ref_name))
        if not idx:
            raise PlanError(f"reference object {ref_name!r} is not in the scene")
        ref_entry = scene["objects"][idx[0]]
        ref_dims = _entry_dims(catalog_by_name, str(ref_name))
        ref_xy = np.array([float(ref_entry["x"]), float(ref_entry["y"])])
        ref_half = _xy_half(ref_dims)

    if mode == "drop_in":
        xy = ref_xy
        surface_z = TABLE.top_z + float(ref_dims[2])       # the container's mouth plane
        gap = 0.03                                          # release above the mouth; gravity does the rest
    elif mode == "set_on":
        xy = ref_xy
        surface_z = TABLE.top_z + float(ref_dims[2])
        gap = 0.006
    elif mode.startswith("beside:"):
        word = mode.split(":", 1)[1]
        v = geometry.direction_vectors(placement)[word]
        desired = ref_xy + np.asarray(v) * (ref_half + obj_half + 0.05)
        xy = _free_spot(desired, obj_half + 0.01, obstacles, base_xy)
    elif mode == "adjacent":
        away = np.asarray(target_xy, dtype=float) - ref_xy
        away = away / max(float(np.linalg.norm(away)), 1.0e-6)
        desired = ref_xy + away * (ref_half + obj_half + 0.02)
        xy = _free_spot(desired, obj_half + 0.005, obstacles, base_xy)
    elif mode == "retrieve":
        v = geometry.direction_vectors(placement)["front"]
        desired = np.asarray(base_xy) + np.asarray(v) * 0.38
        xy = _free_spot(desired, obj_half + 0.01, obstacles, base_xy)
    elif mode == "away":
        away = np.asarray(target_xy, dtype=float) - ref_xy
        n = float(np.linalg.norm(away))
        away = away / n if n > 1.0e-6 else np.asarray(geometry.direction_vectors(placement)["front"])
        desired = ref_xy + away * (ref_half + obj_half + 0.15)
        xy = _free_spot(desired, obj_half + 0.01, obstacles, base_xy)
    else:                                                   # pragma: no cover - table is exhaustive
        raise PlanError(f"unhandled placement mode {mode!r}")
    return {"xy": [float(xy[0]), float(xy[1])], "surface_z": float(surface_z), "gap": float(gap),
            "mode": mode, "reference": str(ref_name) if ref_name else None, "predicate": pred}


# =================================================================================================
# The plan
# =================================================================================================
@dataclass
class TrajPlan:
    waypoints: list                 # [{t, pos, yaw, tilt, tilt_axis, via}]
    grasp_window: dict              # {close_start, close_end, release_start, release_end,
                                    #  force_target, preshape_width}
    num_frames: int
    pick: dict                      # serialized PickSpec summary (id, seat_mode, width, ...)
    place: dict                     # resolve_place() output
    phases: list                    # [{name, t0, t1}] for evaluation + reporting
    target: str
    target_label: str
    target_ordinal: int             # duplicate-label ordinal (settle.py label matching)
    routing: dict                   # bezier stats: inserted points, blockers, clear flags
    attempt: int = 0
    target_kind: str = "rigid"      # rigid | soft | cloth | cable — the rollout's tracking mode
    grasp_windows: list = field(default_factory=list)   # one per segment (multi-step tasks)
    segments: list = field(default_factory=list)        # per-segment rows (target/window/place/pick)

    def to_dict(self) -> dict:
        return {"version": 2, "target": self.target, "target_label": self.target_label,
                "target_ordinal": self.target_ordinal, "target_kind": self.target_kind,
                "attempt": self.attempt,
                "pick": self.pick, "place": self.place, "phases": self.phases,
                "grasp_window": self.grasp_window,
                "grasp_windows": self.grasp_windows or [self.grasp_window],
                "segments": self.segments, "num_frames": self.num_frames,
                "routing": self.routing, "waypoints": self.waypoints}


def _wp(t, pos, yaw=0.0, tilt=0.0, tilt_axis=(1.0, 0.0, 0.0), via=False) -> dict:
    return {"t": round(float(t), 3), "pos": [round(float(v), 4) for v in pos],
            "yaw": round(float(yaw), 4), "tilt": round(float(tilt), 4),
            "tilt_axis": [round(float(v), 4) for v in tilt_axis], "via": bool(via)}


def choose_target_index(scene: dict, task: dict, catalog_by_name: dict) -> int:
    """Which scene INSTANCE of the goal object to manipulate.

    Duplicates are legal ('pick up the apple' with two apples); prefer an instance that does NOT
    already satisfy the goal — measured need (demo_kitchen_fruit): the first-in-scene apple sat
    inside the very bowl the task wanted an apple in. Heuristic: skip instances whose xy already
    lies within the reference object's footprint half-extent."""
    goal = task.get("goal") or {}
    target = (goal.get("params") or {}).get("object")
    t_idx = object_indices(scene, str(target))
    if not t_idx:
        raise PlanError(f"task object {target!r} is not in the scene")
    if len(t_idx) == 1:
        return t_idx[0]
    role, _ = SUPPORTED_GOALS.get(goal.get("predicate"), (None, None))
    ref_name = (goal.get("params") or {}).get(role) if role else None
    ref_idx = object_indices(scene, str(ref_name)) if ref_name else []
    if not ref_idx:
        return t_idx[0]
    ref = scene["objects"][ref_idx[0]]
    # "Already at the goal" = within the reference's inner footprint (0.7x half-extent: the
    # cavity/top region — an instance BESIDE a wide container sits near the full half-extent and
    # must still be pickable).
    ref_inner = 0.7 * _xy_half(_entry_dims(catalog_by_name, str(ref_name)))
    for i in t_idx:
        o = scene["objects"][i]
        d = math.hypot(float(o["x"]) - float(ref["x"]), float(o["y"]) - float(ref["y"]))
        if d > ref_inner:
            return i
    return t_idx[0]


def plan_pick_place(scene: dict, task: dict, placement: dict, catalog_by_name: dict,
                    pick: PickSpec, *, target_bottom_dz: float, attempt: int = 0,
                    target_index: int | None = None,
                    drop_override: float | None = None) -> TrajPlan:
    """Single-goal plan — the 1-segment case of :func:`plan_segments` (kept as the stable API).

    ``target_bottom_dz`` is the target's bottom RELATIVE to its body origin (min vertex z of the
    rest mesh) — with the settled body z it gives how far the held object hangs below the TCP,
    which sets both the carry clearance and the release height. ``target_index`` picks the scene
    INSTANCE when the goal object has duplicates (default: :func:`choose_target_index`)."""
    seg = SegmentSpec(goal=task["goal"], pick=pick, target_bottom_dz=target_bottom_dz,
                      drop_override=drop_override, target_index=target_index)
    return plan_segments(scene, task, placement, catalog_by_name, [seg], attempt=attempt)


@dataclass(frozen=True)
class SegmentSpec:
    """One pick-and-place step of a (possibly multi-step) task."""
    goal: dict                       # a single-object goal predicate dict
    pick: PickSpec
    target_bottom_dz: float
    drop_override: float | None = None
    target_index: int | None = None


def plan_segments(scene: dict, task: dict, placement: dict, catalog_by_name: dict,
                  segments: list, *, attempt: int = 0) -> TrajPlan:
    """Chain one pick-and-place SEGMENT per subgoal into a single timed plan.

    Segment 0 starts from the parked pose after the settle hold; each further segment's cruise
    starts from the previous segment's post-release standoff. Every segment gets its own
    ``GraspWindow`` (the executor's grip kernel iterates windows), its own place resolution, and
    its own Bezier-routed legs; objects PLACED by earlier segments become synthetic obstacle boxes
    for later ones (their scene.json boxes still cover their ORIGINAL spots — conservative on both
    ends)."""
    from agentic_pipeline import geometry  # lazy (keep the physics package import-light)
    from agentic_pipeline.scene_gen import _object_label  # the one body-label definition

    parked = np.asarray(geometry.parked_start_tcp(placement), dtype=float)
    neutral = {"yaw": 0.0, "tilt": 0.0, "tilt_axis": (1.0, 0.0, 0.0)}
    multi = len(segments) > 1

    wps: list[dict] = []
    phases: list[dict] = []
    routing = {"legs": []}
    windows: list[dict] = []
    seg_rows: list[dict] = []
    placed_boxes: list[Obstacle] = []
    wait_pen = 0                     # placed-object boxes accumulate; names stay unique
    t = SETTLE_T
    prev_end = parked

    wps.append(_wp(0.0, parked, **neutral))
    wps.append(_wp(SETTLE_T, parked, **neutral))
    phases.append({"name": "settle", "t0": 0.0, "t1": SETTLE_T})

    for k, seg in enumerate(segments):
        pfx = f"s{k}." if multi else ""
        seg_task = {**task, "goal": seg.goal}
        target = seg.goal["params"]["object"]
        t_idx = object_indices(scene, target)
        if not t_idx:
            raise PlanError(f"subgoal object {target!r} is not in the scene")
        chosen = seg.target_index if seg.target_index is not None \
            else choose_target_index(scene, seg_task, catalog_by_name)
        tgt = scene["objects"][chosen]
        target_xy = (float(tgt["x"]), float(tgt["y"]))
        place = resolve_place(scene, seg_task, placement, catalog_by_name, target, target_xy,
                              extra_obstacles=tuple(placed_boxes))

        pick = seg.pick
        entry = catalog_by_name.get(target, {})
        seg_kind = {"cloth": "cloth", "cable": "cable", "soft_mesh": "soft",
                    "soft_block": "soft"}.get(entry.get("kind", "ycb_mesh"), "rigid")
        close_dur = CLOTH_CLOSE_DUR if seg_kind == "cloth" else CLOSE_DUR
        post_dwell = CLOTH_POST_CLOSE_DWELL if seg_kind == "cloth" else POST_CLOSE_DWELL
        obj_dims = _entry_dims(catalog_by_name, target)
        obj_half = _xy_half(obj_dims)
        gpos = pick.position.copy()
        approach = pick.approach / max(float(np.linalg.norm(pick.approach)), 1.0e-9)
        pregrasp = gpos - DEFAULT_STANDOFF * approach

        # How far the held object hangs below the TCP (world), from the settled bottom plane —
        # or the caller's estimate (a grasped SHEET/CABLE hangs by its material extent, which no
        # rigid bottom-plane formula can know; the deformable path passes it explicitly).
        settled_z = float(tgt.get("z", TABLE.top_z - float(seg.target_bottom_dz)))
        obj_bottom_z = settled_z + float(seg.target_bottom_dz)
        drop = max(float(gpos[2] - obj_bottom_z), 0.0) if seg.drop_override is None \
            else max(float(seg.drop_override), 0.0)

        release_z = place["surface_z"] + drop + place["gap"]
        place_pos = np.array([place["xy"][0], place["xy"][1], release_z])
        place_standoff = place_pos + np.array([0.0, 0.0, PLACE_STANDOFF])
        ori = {"yaw": pick.yaw, "tilt": pick.tilt, "tilt_axis": pick.tilt_axis}

        obstacles_all = scene_obstacles(scene, catalog_by_name, exclude_indices=t_idx) \
            + placed_boxes
        ref_idx = object_indices(scene, place["reference"]) if place["reference"] else []
        obstacles_carry = scene_obstacles(scene, catalog_by_name,
                                         exclude_indices=set(t_idx) | set(ref_idx[:1])) \
            + placed_boxes
        t_seg0 = t

        # -- cruise: previous end -> above the pre-grasp (empty hand). Routed Bezier between
        # ELEVATED endpoints; the drop to the pre-grasp is a straight vertical connector (the
        # inflation is conservative box-swelling; the low approach is the corridor grasp_select
        # cleared against the TRUE boxes). --
        cruise_field = curve.CollisionField(
            obstacles=tuple(obstacles_all), floor_z=TABLE.top_z + 0.02,
            inflate=HAND_HALF_PROFILE * 0.5, ceiling_z=TABLE.top_z + 0.55)
        approach_top = np.array([pregrasp[0], pregrasp[1],
                                 max(pregrasp[2], cruise_field.clear_z(pregrasp))])
        leg = curve.route(prev_end, approach_top, cruise_field)
        routing["legs"].append({"name": f"{pfx}cruise", "inserted": leg.inserted,
                                "clear": leg.clear, "blockers": list(leg.blockers)})
        if not leg.clear:
            raise PlanError(f"{pfx}cruise leg cannot be routed clear (blockers: {leg.blockers})")
        t0_leg = t
        for tt, p, via in curve.leg_waypoints(leg, t, CRUISE_SPEED):
            wps.append(_wp(tt, p, via=via, **ori))
            t = tt
        phases.append({"name": f"{pfx}cruise", "t0": t0_leg, "t1": t})

        # -- descend: straight down to the pre-grasp, then the validated corridor --
        drop_to_pre = float(approach_top[2] - pregrasp[2])
        if drop_to_pre > 0.02:
            t += max(drop_to_pre / (DESCEND_SPEED * 2.0), 0.4)
            wps.append(_wp(t, pregrasp, **ori))
        t_desc = t + max(DEFAULT_STANDOFF / DESCEND_SPEED, 0.8)
        wps.append(_wp(t_desc, gpos, **ori))
        phases.append({"name": f"{pfx}descend", "t0": t, "t1": t_desc})

        # -- close + hold --
        close_start = t_desc + PRE_CLOSE_DWELL
        close_end = close_start + close_dur
        t_hold = close_end + post_dwell
        wps.append(_wp(t_hold, gpos, **ori))
        phases.append({"name": f"{pfx}close", "t0": t_desc, "t1": t_hold})

        # -- retreat along the approach, then straight up to carry height --
        t_ret = t_hold + max(DEFAULT_STANDOFF / LIFT_SPEED, 0.6)
        wps.append(_wp(t_ret, pregrasp, **ori))
        carry_field = curve.CollisionField(
            obstacles=tuple(obstacles_carry), floor_z=TABLE.top_z + drop + 0.03,
            inflate=obj_half + 0.02, ceiling_z=TABLE.top_z + 0.55)
        carry_top = np.array([pregrasp[0], pregrasp[1],
                              max(pregrasp[2], carry_field.clear_z(pregrasp))])
        t = t_ret
        rise = float(carry_top[2] - pregrasp[2])
        if rise > 0.02:
            t += max(rise / LIFT_SPEED, 0.4)
            wps.append(_wp(t, carry_top, **ori))
        phases.append({"name": f"{pfx}lift", "t0": t_hold, "t1": t})

        # -- carry: elevated Bezier to above the place standoff --
        place_top = np.array([place_standoff[0], place_standoff[1],
                              max(place_standoff[2], carry_field.clear_z(place_standoff))])
        leg = curve.route(carry_top, place_top, carry_field)
        routing["legs"].append({"name": f"{pfx}carry", "inserted": leg.inserted,
                                "clear": leg.clear, "blockers": list(leg.blockers)})
        if not leg.clear:
            raise PlanError(f"{pfx}carry leg cannot be routed clear (blockers: {leg.blockers})")
        t0_carry = t
        for tt, p, via in curve.leg_waypoints(leg, t, CARRY_SPEED):
            wps.append(_wp(tt, p, via=via, **ori))
            t = tt
        phases.append({"name": f"{pfx}carry", "t0": t0_carry, "t1": t})

        # -- lower to the release pose (straight vertical set-down) --
        drop_to_standoff = float(place_top[2] - place_standoff[2])
        if drop_to_standoff > 0.02:
            t += max(drop_to_standoff / (DESCEND_SPEED * 2.0), 0.4)
            wps.append(_wp(t, place_standoff, **ori))
        t_place = t + max(PLACE_STANDOFF / DESCEND_SPEED, 0.6)
        wps.append(_wp(t_place, place_pos, **ori))
        phases.append({"name": f"{pfx}place", "t0": t, "t1": t_place})

        # -- release; hold until the jaw is open (rising earlier would drag the object) --
        release_start = t_place + 0.2
        release_end = release_start + RELEASE_DUR
        t_open = release_end + 0.1
        wps.append(_wp(t_open, place_pos, **ori))
        t_up = t_open + POST_RELEASE_DWELL
        wps.append(_wp(t_up, place_standoff, **ori))
        phases.append({"name": f"{pfx}release", "t0": t_place, "t1": t_up})
        t = t_up
        prev_end = place_standoff

        entry = catalog_by_name.get(target, {})
        cfg = entry.get("config", {}) or {}
        mass = float(cfg.get("target_mass") or entry.get("mass") or 0.3)
        force_target = derived_force_target(mass, float(cfg.get("mu", 0.5)))
        if pick.adjusted and pick.adjusted.get("force_target_n") is not None:
            force_target = float(pick.adjusted["force_target_n"])   # LLM override (clamped there)
        window = {"close_start": round(close_start, 3), "close_end": round(close_end, 3),
                  "release_start": round(release_start, 3), "release_end": round(release_end, 3),
                  "force_target": round(force_target, 2),
                  "preshape_width": round(pick.preshape_width, 4)}
        windows.append(window)

        pick_summary = {"id": pick.id, "source": pick.source, "seat_mode": pick.seat_mode,
                        "width": round(pick.width, 4), "tier": pick.tier,
                        "score": round(pick.score, 4),
                        "position": [round(float(v), 4) for v in gpos],
                        "approach": [round(float(v), 4) for v in approach],
                        "yaw_deg": round(math.degrees(pick.yaw), 1),
                        "tilt_deg": round(math.degrees(pick.tilt), 1),
                        "quality_held": pick.quality.get("object_in_gripper"),
                        "adjusted": pick.adjusted, "drop_below_tcp": round(drop, 4)}
        label = _object_label(tgt)
        ordinal = sum(1 for o in scene["objects"][:chosen] if _object_label(o) == label)
        kind = entry.get("kind", "ycb_mesh")
        target_kind = {"cloth": "cloth", "cable": "cable",
                       "soft_mesh": "soft", "soft_block": "soft"}.get(kind, "rigid")
        if target_kind == "cable":
            label = "vbd_cable"                 # the one body label assets.add_cable assigns
        seg_rows.append({"segment": k, "goal": seg.goal, "target": target,
                         "target_label": label, "target_ordinal": ordinal,
                         "target_kind": target_kind, "grasp_window": window, "place": place,
                         "pick": pick_summary, "t0": round(t_seg0, 3), "t1": round(t_up, 3)})
        # The placed object now occupies its destination — a box for LATER segments' routing
        # and free-spot searches.
        wait_pen += 1
        placed_boxes.append(Obstacle(
            name=f"placed:{target}#{wait_pen}",
            center=(place["xy"][0], place["xy"][1],
                    place["surface_z"] + 0.5 * float(obj_dims[2])),
            half=(0.5 * float(obj_dims[0]), 0.5 * float(obj_dims[1]), 0.5 * float(obj_dims[2]))))

    # -- park --
    t_end = t + 1.4
    park_end = np.array([parked[0], parked[1], max(parked[2], prev_end[2])])
    wps.append(_wp(t_end, park_end, **neutral))
    phases.append({"name": "park", "t0": t, "t1": t_end})

    num_frames = int(math.ceil((t_end + END_TAIL) * FPS))
    s0 = seg_rows[0]
    return TrajPlan(waypoints=wps, grasp_window=windows[0], num_frames=num_frames,
                    pick=s0["pick"], place=s0["place"], phases=phases, target=s0["target"],
                    target_label=s0["target_label"], target_ordinal=s0["target_ordinal"],
                    routing=routing, attempt=attempt, target_kind=s0["target_kind"],
                    grasp_windows=windows, segments=seg_rows)
