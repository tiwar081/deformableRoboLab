"""Agentic scene generator — RoboLab-style "scene gen" for the Newton deformable environment.

An agent (Claude) reads a natural-language prompt and composes a realistic tabletop scene from the
IMPORTED object catalog (``assets/objects/scene_catalog.json`` — rigid YCB/HOT3D/Objaverse meshes
plus deformables: RGBench garments, YCB-derived squishy FEM tets, cable/rope variants), a
background HDRI set, and a table-material set. Mirroring RoboLab's scene_gen: the LLM proposes
objects + coordinates, a spatial solver (circle-collision push-apart + table-bounds clamp) makes
the layout feasible, and validation errors are fed back to the agent for a retry (RoboLab's
FeedbackSystem pattern). Unlike RoboLab, deformables are first-class scene objects and the scene
BAKES the background choice too (RoboLab defers backgrounds/lighting/robot to its environment-gen
layer; we keep lighting + robot placement at the centralized defaults for now).

The output of generation is a *scene dict* (JSON-serializable). ``demo_spec_from_scene`` turns it
into a standard ``DemoSpec`` — a settle-only demo (arm parked at home, fingers open) — so the whole
existing runner (solver routing, proxies, materials, RoboLab-look renderer) is reused unchanged.

Helper API (for the full pipeline):
    load_catalog() / catalog_by_name()          — the object set
    available_background_names() / TABLE_KEYS   — the look sets
    call_scene_agent(prompt, ...)               — prompt -> validated + solver-resolved scene dict
    resolve_placements(scene, catalog)          — the spatial solver (also usable standalone)
    demo_spec_from_scene(scene)                 — scene dict -> DemoSpec (physics-ready)
    write_scene(scene, out_dir)                 — persist scene.json + a runnable demo data file
                                                  (numbered folder if the name already exists)
    render_scene(demo_path, ...)                — run the demo headless, return the last
                                                  over-the-shoulder PNG
    verify_scene_render(scene, png)             — agent looks at the settled render and returns
                                                  {ok, issues, revised} (RoboLab's screenshot check)
    generate_scene(prompt, ...)                 — the full loop incl. one verified refine pass

Run standalone to generate ONE scene and render a single settled over-the-shoulder still:

    .venv/bin/python -m agentic_pipeline.scene_generator "a cluttered breakfast table with a sponge and two cans"

Credentials: ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN env vars, else the Claude Code OAuth token
(~/.claude/.credentials.json — the user's Claude subscription). The Anthropic Python SDK is NOT a
dependency (it conflicts with the venv's isaacsim typing-extensions pin), so requests go over raw
HTTPS via urllib.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from deformableManipulationTools.params import TABLE

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "assets" / "objects" / "scene_catalog.json"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "sceneGen"

DEFAULT_MODEL = os.environ.get("SCENEGEN_MODEL", "claude-fable-5")   # latest Claude model
FALLBACK_MODEL = "claude-opus-4-8"
API_URL = "https://api.anthropic.com/v1/messages"
# OAuth (Claude subscription) tokens are scoped to Claude Code: the first system block must be the
# Claude Code identity line or inference is rejected. Harmless with an API key.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

TABLE_KEYS = ("maple", "oak", "bamboo", "black")
CLOTH_KINDS = {"cloth"}
SQUISHY_KINDS = {"soft_mesh", "soft_block"}
RIGID_KINDS = {"ycb_mesh", "rigid_box", "rubiks_cube"}
OVER_SHOULDER_CAMERA = "over_shoulder_left_camera"

# Relational placement vocabulary (RoboLab predicate analog). 2D relations reposition the subject
# next to its target on the tabletop; "on"/"in" pin the subject's x/y to the target and spawn it
# ABOVE the target so gravity settles the stack / drops it into the container.
RELATIONS_2D = ("left-of", "right-of", "in-front-of", "behind")
RELATIONS_STACK = ("on", "in")


def container_names(catalog: dict | None = None) -> list[str]:
    """Open-top container object names usable as an 'in' target — read from the catalog's
    ``container`` flag so any imported container (bowls, buckets, bins, non-articulated cabinets)
    is recognized without editing this module."""
    catalog = catalog or load_catalog()
    return [o["name"] for o in catalog["objects"] if o.get("container") is True]


# ---------------------------------------------------------------------------------------------
# Catalog + look sets
# ---------------------------------------------------------------------------------------------
def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def catalog_by_name(catalog: dict | None = None) -> dict:
    catalog = catalog or load_catalog()
    return {o["name"]: o for o in catalog["objects"]}


def available_background_names() -> list[str]:
    try:
        from robolabViz.config import available_backgrounds
        return sorted(available_backgrounds())
    except Exception:  # renderer not importable (e.g. agent-only host): the vendored set
        return ["aerodynamics_workshop", "billiard_hall", "brown_photostudio", "carpentry_shop_01_2k",
                "empty_warehouse", "garage_2k", "home_office", "industrial_pipe_and_valve_01_2k",
                "machine_shop_01_2k", "photo_studio_01_2k", "small_hangar_01_2k", "studio_small_03_2k",
                "tv_studio_2k"]


def workspace_bounds(margin: float = 0.04) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) usable tabletop extents [m], world frame (robot base at the origin)."""
    return (TABLE.pos[0] - TABLE.half[0] + margin, TABLE.pos[0] + TABLE.half[0] - margin,
            TABLE.pos[1] - TABLE.half[1] + margin, TABLE.pos[1] + TABLE.half[1] - margin)


# ---------------------------------------------------------------------------------------------
# The agent: prompt -> scene dict (validated, solver-resolved)
# ---------------------------------------------------------------------------------------------
def _agent_system(catalog: dict) -> str:
    x0, x1, y0, y1 = workspace_bounds()
    lines = []
    for o in catalog["objects"]:
        kind = {"ycb_mesh": "rigid", "rigid_box": "rigid", "rubiks_cube": "rigid",
                "cloth": "cloth", "cable": "cable",
                "soft_mesh": "squishy", "soft_block": "squishy"}[o["kind"]]
        d = o["dims"]
        lines.append(f'- {o["name"]} [{kind}, {o["class"]}, footprint {d[0]:.2f}x{d[1]:.2f} m] — {o["description"]}')
    return f"""You compose realistic tabletop scenes for a Franka-arm robot-manipulation simulator with
full deformable-body physics (cloth, cables/ropes, squishy FEM objects) — pick objects, a
background, a table material, and physically sensible placements.

WORKSPACE (metres, world frame; the robot base is at the origin, the table is in front of it):
- usable tabletop: x in [{x0:.2f}, {x1:.2f}], y in [{y0:.2f}, {y1:.2f}]; (x, y) is each object's CENTER.
- DIRECTION WORDS are from the ROBOT'S point of view (default mount, facing the table): IN FRONT =
  outward from the robot = -y, BEHIND = toward the robot = +y, LEFT = +x, RIGHT = -x.
- the sweet spot for reachability is ~0.5 m from the origin; avoid crowding the table edges.
- z is automatic (objects rest on the tabletop / settle under gravity). yaw_deg spins about vertical.

OBJECT CATALOG (reference objects ONLY by these exact names; the same name may appear at most twice):
{chr(10).join(lines)}

RELATIONS (optional per object; use them whenever the prompt implies arrangement):
- "left-of" / "right-of" / "in-front-of" / "behind": place this object next to the target in that
  direction (optional "distance" = center-to-center metres; omit for touching-with-clearance).
- "on": stack this object ON TOP of the target. The target must be a rigid object with a similar or
  larger footprint. Give x/y near the target's; they are pinned to the target automatically.
- "in": drop this object INTO an open-top container. The target must be one of: {", ".join(container_names(catalog))}.
- "target" is the INDEX of the target object in your objects array. Chains ("A on B on C") are
  allowed but keep stacks at most 3 high; never make relation cycles. Cables and garments cannot
  be stacked ON or dropped IN anything, and nothing can target a cable, garment, or squishy item
  as its "on"/"in" support. Use relation: null for free-standing objects.

HARD CONSTRAINTS (the physics solver requires them):
- 1 to 7 objects total.
- At most ONE cloth item and at most ONE squishy item per scene, and NEVER both a cloth item and a
  squishy item in the same scene (one particle-deformable type per scene). At most ONE cable/rope.
- A cloth garment laid flat is large: with a garment, add at most 3 other objects and place them
  OFF the garment's footprint (or deliberately on it, if the prompt asks — they will settle onto it).
- Leave clearance between free-standing objects; the spatial solver only nudges, it cannot untangle
  unintended piles.

Choose the background and table material to MATCH THE PROMPT's setting and mood. Compose like a
set dresser: realistic co-occurrence, natural spread (not a grid), slight yaw variation.
Respond with the scene JSON only."""


def _scene_schema(catalog: dict) -> dict:
    names = [o["name"] for o in catalog["objects"]]
    relation = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(RELATIONS_2D + RELATIONS_STACK)},
                    "target": {"type": "integer",
                               "description": "index of the target object in the objects array"},
                    "distance": {"anyOf": [{"type": "null"}, {"type": "number"}],
                                 "description": "center-to-center metres (2D relations only); null = auto"},
                },
                "required": ["type", "target", "distance"],
                "additionalProperties": False,
            },
        ],
    }
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "short lowercase slug for the scene"},
            "description": {"type": "string", "description": "one-sentence scene description"},
            "background": {"type": "string", "enum": available_background_names()},
            "table": {"type": "string", "enum": list(TABLE_KEYS)},
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "yaw_deg": {"type": "number"},
                        "relation": relation,
                    },
                    "required": ["name", "x", "y", "yaw_deg", "relation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["name", "description", "background", "table", "objects"],
        "additionalProperties": False,
    }


def _credential_headers() -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"x-api-key": key}
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not token:
        cred_file = Path.home() / ".claude" / ".credentials.json"
        if cred_file.exists():
            token = json.loads(cred_file.read_text()).get("claudeAiOauth", {}).get("accessToken")
    if not token:
        raise RuntimeError("No Anthropic credentials: set ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN, "
                           "or log in to Claude Code (~/.claude/.credentials.json).")
    return {"Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"}


def _messages_request(system: str, messages: list, model: str, schema: dict) -> dict:
    body = {
        "model": model,
        "max_tokens": 8000,
        "system": [{"type": "text", "text": CLAUDE_CODE_IDENTITY},
                   {"type": "text", "text": system}],
        "messages": messages,
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                 **_credential_headers()})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _response_text(resp: dict) -> str:
    if resp.get("stop_reason") == "refusal":
        raise RuntimeError("model refused the request")
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    raise RuntimeError(f"no text block in response (stop_reason={resp.get('stop_reason')})")


def validate_scene(scene: dict, catalog: dict) -> list[str]:
    """Grammar/constraint check on the agent's scene (RoboLab FeedbackSystem-style messages)."""
    by_name = catalog_by_name(catalog)
    errs = []
    objs = scene.get("objects", [])
    if not 1 <= len(objs) <= 7:
        errs.append(f"scene has {len(objs)} objects; must be 1-7")
    counts: dict[str, int] = {}
    kinds = []
    for o in objs:
        e = by_name.get(o.get("name"))
        if e is None:
            errs.append(f"unknown object name {o.get('name')!r}")
            continue
        counts[o["name"]] = counts.get(o["name"], 0) + 1
        kinds.append(e["kind"])
    for n, c in counts.items():
        if c > 2:
            errs.append(f"{n} appears {c} times; max 2 instances of a name")
    n_cloth = sum(k in CLOTH_KINDS for k in kinds)
    n_squishy = sum(k in SQUISHY_KINDS for k in kinds)
    if n_cloth > 1:
        errs.append("more than one cloth item")
    if n_squishy > 1:
        errs.append("more than one squishy item")
    if n_cloth and n_squishy:
        errs.append("scene mixes a cloth item with a squishy item (one particle-deformable type per scene)")
    if sum(k == "cable" for k in kinds) > 1:
        errs.append("more than one cable/rope")
    if n_cloth and len(objs) > 4:
        errs.append("a garment scene allows at most 3 other objects")
    x0, x1, y0, y1 = workspace_bounds()
    for o in objs:
        if not (x0 - 0.2 <= o.get("x", 1e9) <= x1 + 0.2 and y0 - 0.2 <= o.get("y", 1e9) <= y1 + 0.2):
            errs.append(f"{o.get('name')} at ({o.get('x')}, {o.get('y')}) is far outside the tabletop "
                        f"x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]")
    errs += _validate_relations(objs, by_name)
    return errs


def _validate_relations(objs: list, by_name: dict) -> list[str]:
    """Relation-specific grammar checks (targets, kinds, container fit, footprints, cycles)."""
    errs = []
    for i, o in enumerate(objs):
        rel = o.get("relation")
        if not rel:
            continue
        rtype, tgt = rel.get("type"), rel.get("target")
        label = f"objects[{i}] ({o.get('name')})"
        if rtype not in RELATIONS_2D + RELATIONS_STACK:
            errs.append(f"{label}: unknown relation type {rtype!r}")
            continue
        if not isinstance(tgt, int) or not 0 <= tgt < len(objs) or tgt == i:
            errs.append(f"{label}: relation target must be the index of ANOTHER object (got {tgt!r})")
            continue
        te = by_name.get(objs[tgt].get("name"))
        se = by_name.get(o.get("name"))
        if te is None or se is None:
            continue                                   # unknown-name error already reported
        if rtype in RELATIONS_STACK:
            if se["kind"] == "cable":
                errs.append(f"{label}: a cable cannot be placed '{rtype}' anything")
            if rtype == "on":
                if te["kind"] not in RIGID_KINDS:
                    errs.append(f"{label}: 'on' target {objs[tgt]['name']} must be a rigid object")
                elif max(se["dims"][0], se["dims"][1]) > 1.6 * max(te["dims"][0], te["dims"][1]):
                    errs.append(f"{label}: footprint too large to stack on {objs[tgt]['name']}")
            if rtype == "in":
                containers = [n for n, e in by_name.items() if e.get("container") is True]
                if not te.get("container"):
                    errs.append(f"{label}: 'in' target must be an open-top container "
                                f"({', '.join(sorted(containers))}), not {objs[tgt]['name']}")
                elif min(se["dims"][0], se["dims"][1]) > 0.75 * max(te["dims"][0], te["dims"][1]):
                    errs.append(f"{label}: too large to fit in {objs[tgt]['name']}")
    # cycle check over the relation graph (subject -> target)
    edges = {i: o["relation"]["target"] for i, o in enumerate(objs)
             if o.get("relation") and isinstance(o["relation"].get("target"), int)}
    for start in edges:
        seen, node = set(), start
        while node in edges:
            if node in seen:
                errs.append(f"relation cycle involving objects[{start}] ({objs[start].get('name')})")
                break
            seen.add(node)
            node = edges[node]
    return errs


def _stack_base_z(idx: int, objs: list, by_name: dict, table_top: float) -> float:
    """The z the subject should spawn/rest ON: the table top, or the accumulated top of its
    'on'/'in' target chain (target heights from catalog dims — physics settles the rest)."""
    z, node, hops = table_top, idx, 0
    while hops < 5:
        rel = objs[node].get("relation")
        if not rel or rel.get("type") not in RELATIONS_STACK:
            break
        tgt = rel["target"]
        z += by_name[objs[tgt]["name"]]["dims"][2]
        node, hops = tgt, hops + 1
    return z


# ---------------------------------------------------------------------------------------------
# Spatial solver — circle-collision push-apart on the tabletop (RoboLab SpatialSolver pattern)
# ---------------------------------------------------------------------------------------------
def _cable_nodes(entry: dict, x: float, y: float, yaw: float) -> list[list[float]]:
    """Bowed cable node path centred at (x, y) with direction ``yaw``, clamped to the tabletop
    (the same layout law as demo_runner.cable_layout, parameterized by placement)."""
    from deformableManipulationTools.params import CableConfig
    cfg = CableConfig(**entry.get("config", {}))
    n, seg, bow, r = cfg.node_count, cfg.segment_length, cfg.bow, cfg.radius
    length = seg * (n - 1)
    dx, dy = math.cos(yaw), math.sin(yaw)
    nx, ny = -dy, dx
    x0, x1, y0, y1 = workspace_bounds()
    sx, sy = x - dx * length / 2.0, y - dy * length / 2.0     # start node
    # shift the whole line so both endpoints stay inside the bounds
    for (lo, hi, s, d) in ((x0, x1, sx, dx * length), (y0, y1, sy, dy * length)):
        pass
    ex, ey = sx + dx * length, sy + dy * length
    shift_x = max(0.0, x0 - min(sx, ex)) - max(0.0, max(sx, ex) - x1)
    shift_y = max(0.0, y0 - min(sy, ey)) - max(0.0, max(sy, ey) - y1)
    sx += shift_x
    sy += shift_y
    z = TABLE.top_z + r
    return [[sx + dx * seg * i + nx * bow * math.sin(math.pi * i / (n - 1)),
             sy + dy * seg * i + ny * bow * math.sin(math.pi * i / (n - 1)), z] for i in range(n)]


def resolve_placements(scene: dict, catalog: dict | None = None, *, margin: float = 0.03,
                       iters: int = 150, seed: int = 0, facing_yaw_deg: float = -90.0,
                       tall_keepout: tuple[float, float, float, float] | None = None) -> tuple[bool, str]:
    """Make the agent's layout feasible in place: apply 2D relations (left-of etc.), pin 'on'/'in'
    subjects to their targets, then clamp every free-standing object to the tabletop and push
    overlapping footprint circles apart (cables are laid first and treated as fixed obstacles).
    Mutates ``scene`` (final x/y; cables gain a ``nodes`` list). Returns ``(ok, feedback)`` —
    ``feedback`` names residual violations for the agent retry loop.

    ``facing_yaw_deg`` orients the ROBOT-POV direction words (in-front-of = the robot's facing
    direction, left-of = the robot's left, ...). The default -90 deg is the default mount on the
    back (+y) long edge facing -y — which makes left-of=+x, in-front-of=-y, the historical
    mapping. The agentic pipeline passes the actual placement's yaw.

    ``tall_keepout`` = (x, y, radius, min_height): an obstacle disc applied ONLY to objects taller
    than ``min_height`` — the parked gripper's home-TCP hover zone (a tall object under the
    fingers starts the sim in collision). Cables/cloths lie flat and are unaffected."""
    from . import packing
    by_name = catalog_by_name(catalog)
    rng = random.Random(seed)
    x0, x1, y0, y1 = workspace_bounds()
    objs = scene["objects"]

    def _radius(o):
        e = by_name[o["name"]]
        return 0.5 * max(e["dims"][0], e["dims"][1]) + margin

    def _obb(m):
        """(cx, cy, yaw, w, d) OBB for a movable entry m=[idx,x,y,r]. Cables/keep-out (idx<0 or
        no dims) fall back to a square of side 2r so the same SAT path handles them."""
        idx = m[0]
        if idx < 0:
            return (m[1], m[2], 0.0, 2 * m[3], 2 * m[3])
        e = by_name[objs[idx]["name"]]
        if e["kind"] == "cable":
            return (m[1], m[2], 0.0, 2 * m[3], 2 * m[3])
        yaw = math.radians(objs[idx].get("yaw_deg", 0.0))
        return (m[1], m[2], yaw, e["dims"][0], e["dims"][1])

    # Apply 2D relations first (targets before subjects — follow chains up to depth 5): the subject
    # is repositioned beside its target's CURRENT position; the push-apart below may still nudge it.
    # Direction words are ROBOT-POV, rotated by the reference placement's facing yaw.
    _fx, _fy = math.cos(math.radians(facing_yaw_deg)), math.sin(math.radians(facing_yaw_deg))
    dir_vec = {"left-of": (-_fy, _fx), "right-of": (_fy, -_fx),
               "in-front-of": (_fx, _fy), "behind": (-_fx, -_fy)}
    for _ in range(5):
        for o in objs:
            rel = o.get("relation")
            if rel and rel.get("type") in RELATIONS_2D and isinstance(rel.get("target"), int):
                t = objs[rel["target"]]
                d = rel.get("distance") or (_radius(o) + _radius(t))
                ux, uy = dir_vec[rel["type"]]
                o["x"] = float(t["x"]) + ux * float(d)
                o["y"] = float(t["y"]) + uy * float(d)

    # 'on'/'in' subjects are pinned to their target: no independent placement, no collision circle
    # (they are MEANT to overlap the target in 2D; gravity resolves them in z).
    pinned = {i for i, o in enumerate(objs)
              if o.get("relation") and o["relation"].get("type") in RELATIONS_STACK}

    movable, obstacles = [], []          # [idx, x, y, r]
    for i, o in enumerate(objs):
        e = by_name[o["name"]]
        yaw = math.radians(o.get("yaw_deg", 0.0))
        if i in pinned:
            continue
        if e["kind"] == "cable":
            o["nodes"] = _cable_nodes(e, float(o["x"]), float(o["y"]), yaw)
            r = float(e["config"].get("radius", 0.008)) + margin
            obstacles += [[i, n[0], n[1], r] for n in o["nodes"][::2]]
        else:
            movable.append([i, float(o["x"]), float(o["y"]), _radius(o)])
    # The home-TCP keep-out repels only TALL free-standing objects (idx -1 = not a scene object).
    tall_obstacle = None
    if tall_keepout is not None:
        kx, ky, kr, kmin = tall_keepout
        tall_obstacle = [-1, float(kx), float(ky), float(kr)]

    def _pair_obstacles(m):
        if tall_obstacle is not None and by_name[objs[m[0]]["name"]]["dims"][2] > tall_keepout[3]:
            return obstacles + [tall_obstacle]
        return obstacles

    for _ in range(iters):
        moved = False
        for m in movable:
            cx = min(max(m[1], x0 + min(m[3], 0.5 * (x1 - x0))), x1 - min(m[3], 0.5 * (x1 - x0)))
            cy = min(max(m[2], y0 + min(m[3], 0.5 * (y1 - y0))), y1 - min(m[3], 0.5 * (y1 - y0)))
            if (cx, cy) != (m[1], m[2]):
                m[1], m[2], moved = cx, cy, True
        for a_i in range(len(movable)):
            for b in (movable[a_i + 1:] + _pair_obstacles(movable[a_i])):
                a = movable[a_i]
                if b[0] == a[0]:
                    continue
                # Broad phase: skip pairs whose bounding circles are clearly apart (cheap), then
                # OBB/SAT narrow phase so an elongated object at a yaw is judged by its true
                # footprint, not its longest-side circle (RoboLab uses only the circle).
                if math.hypot(b[1] - a[1], b[2] - a[2]) > a[3] + b[3]:
                    continue
                overlap, ux, uy = packing.obb_penetration(_obb(a), _obb(b), margin=0.5 * margin)
                if overlap <= 1e-4:
                    continue
                if abs(ux) < 1e-9 and abs(uy) < 1e-9:
                    ang = rng.uniform(0, 2 * math.pi)
                    ux, uy = math.cos(ang), math.sin(ang)
                if b in obstacles or b[0] < 0:      # push only 'a' off a fixed obstacle
                    a[1] -= ux * overlap
                    a[2] -= uy * overlap
                else:
                    a[1] -= ux * overlap / 2
                    a[2] -= uy * overlap / 2
                    b[1] += ux * overlap / 2
                    b[2] += uy * overlap / 2
                moved = True
        if not moved:
            break

    problems = []
    for a_i in range(len(movable)):
        for b in (movable[a_i + 1:] + _pair_obstacles(movable[a_i])):
            a = movable[a_i]
            if b[0] == a[0]:
                continue
            if math.hypot(b[1] - a[1], b[2] - a[2]) > a[3] + b[3]:
                continue
            depth, _, _ = packing.obb_penetration(_obb(a), _obb(b))
            if depth > 0.005:
                na = objs[a[0]]["name"]
                nb = objs[b[0]]["name"] if b[0] >= 0 else "the parked gripper's hover zone (tall object)"
                problems.append(f"{na} overlaps {nb} by {depth:.3f} m")
    for m in movable:
        objs[m[0]]["x"] = round(m[1], 4)
        objs[m[0]]["y"] = round(m[2], 4)
    # Pin stacked/contained subjects onto their target's FINAL position (chains resolve root-first).
    for _ in range(5):
        for i in sorted(pinned):
            t = objs[objs[i]["relation"]["target"]]
            objs[i]["x"], objs[i]["y"] = t["x"], t["y"]
    if problems:
        return False, ("the placements could not be made collision-free: " + "; ".join(problems)
                       + ". Spread the objects further apart (or use fewer/smaller objects).")
    return True, ""


def call_scene_agent(prompt: str, *, model: str = DEFAULT_MODEL, attempts: int = 3,
                     catalog: dict | None = None, seed: int = 0, verbose: bool = True) -> dict:
    """The agent loop: prompt -> scene dict, with grammar validation + the spatial solver, and
    natural-language feedback retries (RoboLab's validate-and-refine pattern)."""
    catalog = catalog or load_catalog()
    system, schema = _agent_system(catalog), _scene_schema(catalog)
    messages = [{"role": "user", "content": prompt}]
    scene, use_model = None, model
    for attempt in range(attempts):
        try:
            resp = _messages_request(system, messages, use_model, schema)
            text = _response_text(resp)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode(errors="replace")[:300]
            if use_model != FALLBACK_MODEL:
                if verbose:
                    print(f"[sceneGen] {use_model} failed ({exc} {detail}); falling back to {FALLBACK_MODEL}")
                use_model = FALLBACK_MODEL
                continue
            raise
        scene = json.loads(text)
        scene["prompt"] = prompt
        scene["model"] = use_model
        errs = validate_scene(scene, catalog)
        feedback = "; ".join(errs)
        if not errs:
            ok, feedback = resolve_placements(scene, catalog, seed=seed)
            if ok:
                if verbose:
                    print(f"[sceneGen] scene {scene['name']!r} accepted on attempt {attempt + 1}")
                return scene
        if verbose:
            print(f"[sceneGen] attempt {attempt + 1} rejected: {feedback}")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your scene was rejected: {feedback}. "
                                                 f"Return a corrected scene JSON."}]
    if scene is None:
        raise RuntimeError("scene agent produced no usable scene")
    resolve_placements(scene, catalog, seed=seed)   # best effort on the final attempt
    if verbose:
        print("[sceneGen] accepting best-effort layout after retries")
    return scene


# ---------------------------------------------------------------------------------------------
# Scene dict -> DemoSpec (physics-ready, rendered like any demo)
# ---------------------------------------------------------------------------------------------
def demo_spec_from_scene(scene: dict):
    """Build the settle-only ``DemoSpec`` for a generated scene: the standard runner then supplies
    solver routing (MuJoCo rigid-only vs split VBD + proxies), materials, and the RoboLab-look
    render. The arm parks at home (one waypoint), fingers open, and the scene settles physically."""
    from dataclasses import replace
    from deformableManipulationTools import (TABLE as _TABLE, RIGID_CUBE, RUBIKS_CUBE, SOFT_BLOCK,
                                             SoftMeshConfig, YcbMeshConfig, CableConfig, ClothConfig)
    from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP
    from robolabViz import ObjectStyle, RenderSpec

    catalog = load_catalog()
    by_name = catalog_by_name(catalog)
    soft_colors = catalog.get("soft_body_colors", {})
    tt = _TABLE.top_z

    objs = [Obj("table", _TABLE)]
    particle_cfg = None          # the ONE particle deformable's config (soft_contact_* owner)
    has_cloth = has_deformable = False
    soft_color = None
    for i, o in enumerate(scene["objects"]):
        e = by_name[o["name"]]
        kind, cfg = e["kind"], dict(e.get("config", {}))
        x, y = float(o["x"]), float(o["y"])
        yaw = math.radians(float(o.get("yaw_deg", 0.0)))
        # base z: the table top, or the accumulated top of the 'on'/'in' target chain (the subject
        # spawns just above it and gravity settles the stack / drops it into the container).
        base_z = _stack_base_z(i, scene["objects"], by_name, tt)
        stacked = base_z > tt
        if kind == "ycb_mesh":
            objs.append(Obj("ycb_mesh", YcbMeshConfig(**cfg), pos=(x, y, base_z + 0.005),
                            rest_on_z=(base_z + 0.01 if stacked else True), yaw=yaw))
        elif kind == "rigid_box":
            objs.append(Obj("rigid_box", RIGID_CUBE,
                            pos=(x, y, base_z + RIGID_CUBE.half_extent + (0.01 if stacked else 0.002)),
                            half=RIGID_CUBE.half_extent))
        elif kind == "rubiks_cube":
            objs.append(Obj("rubiks_cube", RUBIKS_CUBE,
                            pos=(x, y, base_z + RUBIKS_CUBE.half_extent + (0.01 if stacked else 0.002))))
        elif kind == "soft_block":
            particle_cfg = SOFT_BLOCK
            objs.append(Obj("soft_block", SOFT_BLOCK, pos=(x, y, base_z + (0.01 if stacked else 0.002))))
            soft_color = soft_colors.get(o["name"], soft_color)
            has_deformable = True
        elif kind == "soft_mesh":
            particle_cfg = SoftMeshConfig(**cfg)
            objs.append(Obj("soft_mesh", particle_cfg, pos=(x, y, base_z + (0.01 if stacked else 0.002)),
                            yaw=yaw))
            soft_color = soft_colors.get(o["name"], soft_color)
            has_deformable = True
        elif kind == "cloth":
            particle_cfg = replace(ClothConfig(), **cfg)
            objs.append(Obj("cloth", particle_cfg, pos=(x, y, base_z + 0.04), yaw=yaw))
            soft_color = soft_colors.get(o["name"], soft_color)
            has_cloth = has_deformable = True
        elif kind == "cable":
            nodes = o.get("nodes") or _cable_nodes(e, x, y, yaw)
            objs.append(Obj("cable", CableConfig(**cfg), pos=[tuple(n) for n in nodes]))
            has_deformable = True
        else:
            raise ValueError(f"unknown catalog kind {kind!r}")
    if has_deformable:
        objs.append(Obj("proxies"))

    solver_kwargs, pipeline_kwargs, coupling_ke = {}, {}, None
    if particle_cfg is not None:
        coupling_ke = particle_cfg.soft_contact_ke
        solver_kwargs = {"rigid_body_contact_buffer_size": 4096,
                         "rigid_body_particle_contact_buffer_size": 8192}
        pipeline_kwargs = {"soft_contact_margin": particle_cfg.contact_margin}
    substeps, vbd_iters = (10, 5) if has_cloth else (16, 12)
    num_frames = demo_spec_num_frames(scene)

    render = RenderSpec(
        background=scene.get("background"),
        table=scene.get("table"),
        preview_cameras=[OVER_SHOULDER_CAMERA],
        soft_body_style=(ObjectStyle(color=tuple(soft_color), roughness=0.85) if soft_color else None),
    )
    return DemoSpec(
        scene=objs,
        waypoints=[WP(0.0)],                 # arm parked at home; scene settles physically
        render=render,
        coupling_soft_ke=coupling_ke,
        object_solver_kwargs=solver_kwargs,
        object_pipeline_kwargs=pipeline_kwargs,
        substeps=substeps, vbd_iterations=vbd_iters, num_frames=num_frames,
        scenic_check_table=False,
    )


# ---------------------------------------------------------------------------------------------
# Persist + render
# ---------------------------------------------------------------------------------------------
def write_scene(scene: dict, out_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Write ``scene.json`` + a runnable demo DATA FILE for it. Returns (scene_json, demo_py).
    Never overwrites an existing scene: if the default folder already exists, a numeric suffix is
    appended (name, name_2, name_3, ...) and ``scene['name']`` is updated to match (an explicit
    ``out_dir`` is used as-is — that is the revise-in-place path)."""
    name = re.sub(r"[^a-z0-9_]+", "_", scene["name"].lower()).strip("_") or "scene"
    if out_dir is None:
        out, n = OUTPUT_ROOT / name, 1
        while out.exists():
            n += 1
            out = OUTPUT_ROOT / f"{name}_{n}"
        if n > 1:
            name = f"{name}_{n}"
            scene["name"] = name
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_json = out / "scene.json"
    scene_json.write_text(json.dumps(scene, indent=1))
    demo_py = out / f"scene_{name}.py"
    demo_py.write_text(
        '"""GENERATED scene demo (agentic_pipeline.scene_generator) — scene data lives in scene.json."""\n'
        "import json\nfrom pathlib import Path\n\nfrom agentic_pipeline import scene_generator\n\n"
        "DEMO = scene_generator.demo_spec_from_scene(\n"
        "    json.loads((Path(__file__).parent / 'scene.json').read_text()))\n")
    return scene_json, demo_py


def render_scene(demo_py: Path | str, *, device: str = "cuda:0", style: str = "mp4_advanced",
                 out_png: Path | str | None = None, camera: str = OVER_SHOULDER_CAMERA,
                 extra_cameras: dict[str, Path | str] | None = None, verbose: bool = True) -> Path:
    """Run the generated demo headless and return the LAST still from ``camera`` (the settled
    scene). ``extra_cameras`` optionally copies other cameras' final stills too (e.g. the wrist
    camera): {camera_name: destination_png}. Uses the standard runner in a subprocess (clean
    CUDA/warp state)."""
    demo_py = Path(demo_py)
    spec_scene = json.loads((demo_py.parent / "scene.json").read_text())
    n_frames = demo_spec_num_frames(spec_scene)
    cmd = [sys.executable, str(REPO_ROOT / "example.py"), "--demo", str(demo_py),
           "--output-style", style, "--device", device,
           "--frames-per-image", str(max(n_frames - 1, 1)), "--quiet"]
    if verbose:
        print("[sceneGen] rendering:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"render failed (exit {res.returncode}):\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    from deformableManipulationTools.params import FRANKA
    frames_dir = REPO_ROOT / "outputs" / FRANKA.short_name / demo_py.stem / "frames"

    def _last(cam_name: str) -> Path:
        stills = sorted(frames_dir.glob(f"{cam_name}_*.png"))
        if not stills:
            raise RuntimeError(f"no {cam_name!r} stills in {frames_dir}")
        return stills[-1]

    target = Path(out_png) if out_png else demo_py.parent / "scene_over_shoulder.png"
    target.write_bytes(_last(camera).read_bytes())
    for cam_name, dest in (extra_cameras or {}).items():
        try:
            Path(dest).write_bytes(_last(cam_name).read_bytes())
        except RuntimeError as exc:
            if verbose:
                print(f"[sceneGen] {exc} (skipping)")
    return target


def demo_spec_num_frames(scene: dict) -> int:
    """Settle length by scene content (ONE definition — DemoSpec and the PNG cadence share it)."""
    by_name = catalog_by_name()
    kinds = [by_name[o["name"]]["kind"] for o in scene["objects"]]
    n = 60
    if any(k in SQUISHY_KINDS or k == "cable" for k in kinds):
        n = 80
    if any(k in CLOTH_KINDS for k in kinds):
        n = 110
    if any(o.get("relation") and o["relation"].get("type") in RELATIONS_STACK for o in scene["objects"]):
        n = max(n, 100)                  # stacks/drops need longer to come to rest
    return n


# ---------------------------------------------------------------------------------------------
# Post-render visual verification (RoboLab's screenshot sanity check, done by the agent)
# ---------------------------------------------------------------------------------------------
def _verify_schema(catalog: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "revised": {"anyOf": [{"type": "null"}, _scene_schema(catalog)],
                        "description": "full corrected scene when ok=false and fixable, else null"},
        },
        "required": ["ok", "issues", "revised"],
        "additionalProperties": False,
    }


def verify_scene_render(scene: dict, png_path: Path | str, *, model: str = DEFAULT_MODEL,
                        catalog: dict | None = None) -> dict:
    """Show the agent the SETTLED over-the-shoulder render of its own scene and ask for a verdict:
    ``{ok, issues, revised}``. The verifier checks that every object is present and resting sanely
    (nothing fallen off, floating, or interpenetrating), that the arrangement matches the prompt
    and relations, and that the look suits the prompt — the agent analog of RoboLab's post-settle
    screenshot sanity check."""
    import base64
    catalog = catalog or load_catalog()
    shown = {k: v for k, v in scene.items() if k not in ("paths", "verification", "verification_revised")}
    ask = f"""This image is the PHYSICALLY SETTLED render of the scene you composed (over-the-shoulder
camera; the pale robot arm in the foreground is expected; image-left = +x, farther = -y).

Original user prompt: {scene.get('prompt', '(unknown)')}

The scene as generated:
{json.dumps(shown, indent=1)}

Verify the render:
1. Every object in the scene JSON is visible on the table — none missing, fallen off, floating,
   sunken into the table, or interpenetrating another object.
2. The arrangement matches the prompt and every declared relation (stacks stacked, contained
   objects in their container, left/right/front/behind as stated).
3. Background and table material suit the prompt.

If everything is acceptable: ok=true, issues=[], revised=null. If something is wrong AND fixable by
different placements/relations/object choices, set ok=false, list the concrete issues, and return a
FULL corrected scene in 'revised' (same rules and constraints as before, keep the same name). If
wrong but not fixable by re-placement, set ok=false with issues and revised=null."""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.standard_b64encode(Path(png_path).read_bytes()).decode()}},
        {"type": "text", "text": ask},
    ]
    schema = _verify_schema(catalog)
    system = _agent_system(catalog)
    for use_model in (model, FALLBACK_MODEL):
        try:
            resp = _messages_request(system, [{"role": "user", "content": content}], use_model, schema)
            return json.loads(_response_text(resp))
        except (urllib.error.HTTPError, RuntimeError):
            if use_model == FALLBACK_MODEL:
                raise
    raise RuntimeError("unreachable")


def generate_scene(prompt: str, *, model: str = DEFAULT_MODEL, out_dir: Path | str | None = None,
                   render: bool = True, verify: bool = False, device: str = "cuda:0",
                   seed: int = 0, verbose: bool = True) -> dict:
    """End-to-end helper for the pipeline: prompt -> scene dict + files (+ rendered still, and —
    OPT-IN via ``verify=True`` — one agent-verified refine pass). Returns the scene dict, with
    ``paths`` (and ``verification`` when enabled) filled in. On a failed verification with a
    usable revision, the scene is revised IN PLACE (the original is kept as
    ``scene_initial.json``) and re-rendered + re-verified once."""
    catalog = load_catalog()
    scene = call_scene_agent(prompt, model=model, seed=seed, catalog=catalog, verbose=verbose)
    scene_json, demo_py = write_scene(scene, out_dir)
    scene["paths"] = {"scene_json": str(scene_json), "demo": str(demo_py)}
    if not render:
        return scene
    png = render_scene(demo_py, device=device, verbose=verbose)
    scene["paths"]["image"] = str(png)
    if not verify:
        return scene
    verdict = verify_scene_render(scene, png, model=model, catalog=catalog)
    scene["verification"] = {"ok": bool(verdict.get("ok")), "issues": verdict.get("issues", [])}
    if verbose:
        print(f"[sceneGen] verification: ok={verdict.get('ok')} issues={verdict.get('issues', [])}")
    revised = verdict.get("revised")
    if not verdict.get("ok") and revised:
        revised["name"] = scene["name"]              # keep the folder/demo identity
        revised["prompt"], revised["model"] = scene.get("prompt"), scene.get("model")
        errs = validate_scene(revised, catalog)
        ok = not errs and resolve_placements(revised, catalog, seed=seed)[0]
        if ok:
            (demo_py.parent / "scene_initial.json").write_text(json.dumps(scene, indent=1))
            write_scene(revised, out_dir=demo_py.parent)
            png2 = render_scene(demo_py, device=device, verbose=verbose)
            verdict2 = verify_scene_render(revised, png2, model=model, catalog=catalog)
            revised["paths"] = dict(scene["paths"], image=str(png2))
            revised["verification"] = scene["verification"]
            revised["verification_revised"] = {"ok": bool(verdict2.get("ok")),
                                               "issues": verdict2.get("issues", [])}
            if verbose:
                print(f"[sceneGen] revised scene re-verified: ok={verdict2.get('ok')} "
                      f"issues={verdict2.get('issues', [])}")
            scene = revised
        elif verbose:
            print(f"[sceneGen] revision rejected by validation/solver ({'; '.join(errs) or 'overlaps'}); "
                  f"keeping the original scene")
    scene_json.write_text(json.dumps(scene, indent=1))   # persist the verdict(s) alongside the scene
    return scene


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a scene from a prompt and render one "
                                             "over-the-shoulder still of the settled scene.")
    ap.add_argument("prompt", help="natural-language scene description")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=None, help="output directory (default outputs/sceneGen/<name>)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="stop after writing scene.json + demo file")
    ap.add_argument("--verify", action="store_true",
                    help="OPT-IN post-render agent verification / refine pass (extra model calls "
                         "+ a possible re-render; off by default)")
    args = ap.parse_args()

    result = generate_scene(args.prompt, model=args.model, out_dir=args.out_dir,
                            render=not args.no_render, verify=args.verify,
                            device=args.device, seed=args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != "prompt"}, indent=1))
    for label, p in result["paths"].items():
        print(f"{label}: {p}")
