"""Hardcoded spatial layer for the agentic pipeline: workspace-table edges, robot-table (stand)
placement, and reachability — the primitives that let the task-gen agent reason about robot
placement with plain numbers instead of geometry code.

Frames and conventions (single source of truth for the pipeline):
- World frame: the workspace TABLE is fixed (deformableManipulationTools.params.TABLE).
- DIRECTION WORDS are from the ROBOT'S point of view (``direction_vectors``): "in front" = outward
  from the robot base into the workspace (the facing direction), "behind" = back toward the base,
  "left"/"right" = the robot's left/right hand sides. They rotate with the robot placement; scene
  gen (which runs before task gen picks a placement) anchors them to the placement that is already
  fixed (scene_init) or to the DEFAULT mount. For the default mount (facing -y): front = -y,
  behind = +y, left = +x, right = -x.
- Edge NAMES are fixed labels of the workspace table (independent of where the robot ends up),
  chosen from the default-mount viewpoint: "back" = +y long edge (the default robot side),
  "front" = -y long edge, "left" = +x short edge, "right" = -x short edge (the legacy mount side).
- The robot is ALWAYS mounted at the default position on its own robot table (the franka_stand
  fixture, measured from assets/fixtures/franka_table.usd): relative to the base, the stand
  extends STAND_FRONT ahead (toward the workspace), STAND_BACK behind, +-STAND_HALF_WIDTH across.
  A placement chooses only WHERE the stand sits: an edge of the workspace table + an anchor along
  it. The stand's front edge TOUCHES the workspace edge (CLEARANCE apart, never overlapping) —
  that constraint is baked into the placement math, so the only things to CHECK are edge-interval
  alignment (some actual contact; overhang past the ends is allowed) and reach.
- Reach: Franka max reach ~0.855 m; REACH_MAX = 0.80 m usable. An object counts as reachable when
  SOME PART of it is in reach: distance from the base to the NEAREST point of its footprint
  (center minus footprint radius; nearest node for cables) <= REACH_MAX.
"""
from __future__ import annotations

import math

from deformableManipulationTools.params import TABLE

# franka_stand extents RELATIVE TO THE ROBOT BASE in the base's facing frame (measured from
# franka_table.usd with the fixture's translate (-0.087, 0, 0) + 180 deg yaw applied):
STAND_FRONT = 0.089        # base -> stand front edge (toward the workspace table) [m]
STAND_BACK = 0.811         # base -> stand back edge [m]
STAND_HALF_WIDTH = 0.379   # stand half-extent across the facing direction [m]
CLEARANCE = 0.002          # stand front edge sits this far off the workspace edge (touch, no overlap)
REACH_MAX = 0.80           # usable arm reach from the base axis [m] (kinematic max ~0.855)
SWEET_BAND = (0.35, 0.65)  # comfortable center-distance band (RoboLab's ~0.5 m sweet spot)
MIN_EDGE_CONTACT = 0.05    # required overlap between stand edge and workspace edge [m]

_X0, _X1 = TABLE.pos[0] - TABLE.half[0], TABLE.pos[0] + TABLE.half[0]
_Y0, _Y1 = TABLE.pos[1] - TABLE.half[1], TABLE.pos[1] + TABLE.half[1]

# name -> (outward normal, edge axis 'x'|'y', fixed coordinate, (span_lo, span_hi))
EDGES: dict[str, dict] = {
    "back":  {"normal": (0.0, 1.0),  "axis": "x", "coord": _Y1, "span": (_X0, _X1)},
    "front": {"normal": (0.0, -1.0), "axis": "x", "coord": _Y0, "span": (_X0, _X1)},
    "left":  {"normal": (1.0, 0.0),  "axis": "y", "coord": _X1, "span": (_Y0, _Y1)},
    "right": {"normal": (-1.0, 0.0), "axis": "y", "coord": _X0, "span": (_Y0, _Y1)},
}
LEGACY_BASE_XY = (-0.45, -0.45)   # the framework's built-in default mount (right short edge)


def robot_placement(edge: str, anchor: float, source: str = "agent") -> dict:
    """A robot-table placement: the stand's front edge touches the workspace ``edge``, centred at
    ``anchor`` (world coordinate along that edge). Returns the full placement dict the pipeline
    stores in task.json: base position/yaw plus the stand's contact interval for the checks."""
    e = EDGES[edge]
    nx, ny = e["normal"]
    fx, fy = -nx, -ny                            # facing = into the table
    d = STAND_FRONT + CLEARANCE
    if e["axis"] == "x":
        base_x, base_y = float(anchor), e["coord"] + ny * d
    else:
        base_x, base_y = e["coord"] + nx * d, float(anchor)
    yaw_deg = math.degrees(math.atan2(fy, fx))   # identity facing is +x
    return {
        "edge": edge, "anchor": float(anchor), "source": source,
        "base": [base_x, base_y, TABLE.top_z - TABLE.base_drop],
        "yaw_deg": yaw_deg,
        "stand_interval": [float(anchor) - STAND_HALF_WIDTH, float(anchor) + STAND_HALF_WIDTH],
    }


def default_placement() -> dict:
    """The user-facing DEFAULT: middle of the back LONG edge (not the legacy short-edge mount)."""
    return robot_placement("back", TABLE.pos[0], source="default")


def direction_vectors(placement: dict | None = None) -> dict[str, tuple[float, float]]:
    """Robot-POV direction words -> world-frame unit vectors, for the given placement (None = the
    default mount). "front" = the facing direction (outward from the base into the workspace),
    "behind" = toward the base, "left"/"right" = the robot's left/right hand sides."""
    placement = placement or default_placement()
    yaw = math.radians(placement["yaw_deg"])
    fx, fy = math.cos(yaw), math.sin(yaw)
    return {"front": (fx, fy), "behind": (-fx, -fy),
            "left": (-fy, fx), "right": (fy, -fx)}


def directions_text(placement: dict | None = None) -> str:
    """The direction-word semantics block for agent prompts, with the concrete world vectors of
    the reference placement filled in (robot-POV; see ``direction_vectors``)."""
    placement = placement or default_placement()
    dv = direction_vectors(placement)

    def _fmt(v):
        return f"({v[0]:+.0f}x, {v[1]:+.0f}y)" if abs(abs(v[0]) - abs(v[1])) > 0.5 else \
               f"({v[0]:+.2f}x, {v[1]:+.2f}y)"
    return (f"DIRECTION WORDS are from the ROBOT'S point of view (robot on the {placement['edge']!r} "
            f"edge of the workspace table): IN FRONT = outward from the robot into the workspace "
            f"= {_fmt(dv['front'])}; BEHIND = back toward the robot = {_fmt(dv['behind'])}; "
            f"LEFT = the robot's left = {_fmt(dv['left'])}; RIGHT = the robot's right = {_fmt(dv['right'])}.")


def alignment_check(placement: dict) -> tuple[bool, str]:
    """Edge-alignment: the stand's front edge interval must actually CONTACT the workspace edge
    span (>= MIN_EDGE_CONTACT of shared length). Extending past either end is allowed. Overlap of
    the two table AREAS is impossible by construction (the placement math keeps the stand
    CLEARANCE outside the edge)."""
    e = EDGES[placement["edge"]]
    lo, hi = placement["stand_interval"]
    s0, s1 = e["span"]
    contact = min(hi, s1) - max(lo, s0)
    if contact < MIN_EDGE_CONTACT:
        return False, (f"robot table on '{placement['edge']}' at anchor {placement['anchor']:.2f} "
                       f"contacts only {max(contact, 0.0):.2f} m of the workspace edge span "
                       f"[{s0:.2f}, {s1:.2f}] (need >= {MIN_EDGE_CONTACT:.2f} m)")
    return True, f"stand contacts {contact:.2f} m of the '{placement['edge']}' edge"


def _object_points(obj: dict, entry: dict) -> list[tuple[float, float, float]]:
    """(x, y, footprint_radius) sample points for reach: cables contribute every stored node
    (radius ~ the cable radius); everything else its center + half-footprint radius."""
    if entry["kind"] == "cable" and obj.get("nodes"):
        r = float(entry.get("config", {}).get("radius", 0.008))
        return [(float(n[0]), float(n[1]), r) for n in obj["nodes"]]
    r = 0.5 * max(entry["dims"][0], entry["dims"][1])
    return [(float(obj["x"]), float(obj["y"]), r)]


def reach_report(placement: dict, scene: dict, catalog_by_name: dict,
                 names: list[str] | None = None) -> list[dict]:
    """Per-object reach numbers from a placement's base: nearest-point distance (SOME PART of the
    object), center distance, and the reachable / sweet-spot verdicts. ``names`` restricts the
    report to task-relevant objects (default: all scene objects)."""
    bx, by = placement["base"][0], placement["base"][1]
    out = []
    for o in scene.get("objects", []):
        if names is not None and o["name"] not in names:
            continue
        pts = _object_points(o, catalog_by_name[o["name"]])
        d_center = min(math.hypot(x - bx, y - by) for x, y, _ in pts)
        d_nearest = min(max(0.0, math.hypot(x - bx, y - by) - r) for x, y, r in pts)
        out.append({
            "name": o["name"], "d_nearest": round(d_nearest, 3), "d_center": round(d_center, 3),
            "reachable": d_nearest <= REACH_MAX,
            "sweet": SWEET_BAND[0] <= d_center <= SWEET_BAND[1],
        })
    return out


def edge_reach_text(scene: dict, catalog_by_name: dict) -> str:
    """The intuition-preserving reach table for the task-gen agent: for each edge and a few anchor
    positions, which scene objects have SOME PART within reach. Lets the agent pick (edge, anchor)
    by looking at concrete numbers instead of doing trigonometry."""
    lines = []
    for edge, e in EDGES.items():
        s0, s1 = e["span"]
        anchors = [s0 + f * (s1 - s0) for f in (0.25, 0.5, 0.75)]
        for a in anchors:
            p = robot_placement(edge, a)
            rep = reach_report(p, scene, catalog_by_name)
            reach = [r["name"] for r in rep if r["reachable"]]
            miss = [f'{r["name"]}({r["d_nearest"]:.2f}m)' for r in rep if not r["reachable"]]
            lines.append(f"- edge={edge!r} anchor={a:.2f}: reachable={reach}"
                         + (f" out-of-reach={miss}" if miss else ""))
    return "\n".join(lines)


def default_exterior_camera(placement: dict) -> dict:
    """The default exterior camera: on the OPPOSITE side of the workspace table from the robot,
    pulled well back past the edge and raised above the tabletop, looking down at the table center
    so the full workspace is visible with room for the scene background (front-facing, MODERATELY
    bird's-eye — a flatter, less top-down angle than a pure overhead). Uses a narrowed lens
    (focal 5.0 vs the DROID cameras' 2.1 wide-angle) so the table fills the frame from that
    distance. Geometry (measured to frame the 0.9x0.7 m table with background visible): the eye
    sits 1.1 m OUT from the opposite edge and 0.85 m up — a ~28 deg look-down (lowered + angled up
    from the earlier 1.15 m / ~35 deg) so more horizon/background shows above the far table edge."""
    opposite = {"back": "front", "front": "back", "left": "right", "right": "left"}[placement["edge"]]
    e = EDGES[opposite]
    nx, ny = e["normal"]
    s0, s1 = e["span"]
    mid = 0.5 * (s0 + s1)
    if e["axis"] == "x":
        ex, ey = mid, e["coord"]
    else:
        ex, ey = e["coord"], mid
    pos = (ex + nx * 1.1, ey + ny * 1.1, TABLE.top_z + 0.85)
    target = (TABLE.pos[0], TABLE.pos[1], TABLE.top_z)
    return {"name": "overview_camera", "position": [round(v, 3) for v in pos],
            "target": list(target), "focal_length": 5.0, "source": "default"}


PARKED_TCP_HEIGHT = 0.45   # TCP height above the tabletop at the raised parked start pose [m]


def parked_start_tcp(placement: dict) -> tuple[float, float, float]:
    """The raised, out-of-the-way TCP pose the arm STARTS at (frame 0), so the parked end-effector
    hovers well ABOVE the scene instead of at home_q's ~0.17 m (where tall objects can touch it —
    the cloth demos use exactly this "start high and clear" trick, cf. examples/cloth_franka_green).
    ~0.5 m out along the facing direction, PARKED_TCP_HEIGHT above the tabletop."""
    yaw = math.radians(placement["yaw_deg"])
    bx, by = placement["base"][0], placement["base"][1]
    return (bx + 0.50 * math.cos(yaw), by + 0.50 * math.sin(yaw), TABLE.top_z + PARKED_TCP_HEIGHT)


def home_tcp_keepout(placement: dict) -> tuple[float, float, float, float]:
    """(x, y, radius, min_height): the disc under the parked gripper where TALL objects must not
    stand. With the raised parked start pose (``parked_start_tcp``, TCP ~0.45 m up) the fingers
    clear most objects, but a very tall object (pitcher ~0.24 m, wood block ~0.21 m) plus the
    descending approach still wants clearance — keep the keep-out but only against the tallest
    objects (min_height 0.20 m; below the raised parked pose short/mid objects are fine)."""
    x, y, _ = parked_start_tcp(placement)
    return (x, y, 0.12, 0.20)
