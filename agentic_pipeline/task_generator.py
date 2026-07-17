"""Agentic task generator — RoboLab-style "task gen" for the Newton deformable environment.

Given a scene (from ``agentic_pipeline.scene_generator`` — a tabletop of rigid AND deformable objects), an agent
(Claude) proposes a MEANINGFUL, PHYSICALLY FEASIBLE manipulation task: a natural-language
instruction (default/vague/specific variants), a goal predicate over the scene objects, a subtask
decomposition, and competency attributes. It mirrors RoboLab's task-gen in STRUCTURE (scene + goal
-> map the instruction verb to a termination predicate -> instruction variants + subtasks +
attributes; validate-and-refine with feedback retries). The KEY DIFFERENCE is that our scenes hold
deformables, so the feasibility checks are DEFORMABLE-AWARE:

- affordance gating by object type (``check_affordance``): "fold" is meaningful for a cloth but not a
  cable; "coil" for a cable but not a cube; "stack" only for rigids. The predicate must match the
  object's category (rigid / cloth / cable / squishy).
- folded-volume container fit (``check_fits_in``): a shirt that does NOT fit a container laid flat
  MAY fit folded (folding trades footprint for thickness — volume is conserved), so "put the shirt
  in the bowl" can be feasible where the flat garment isn't. A cable is modelled coiled.

Like RoboLab's own task-gen, feasibility here is STATIC / GEOMETRIC only — no physics simulation. The
task generator only IDEATES the task; it does NOT decide grasp force, grip width, or the trajectory
(a downstream trajectory-generation pipeline owns those). It reproduces RoboLab's static-check
placement (``verify_task_valid``) and adds a coarse Cartesian reachability check RoboLab lacks.

Helper API (for the full pipeline — parallels ``scene_generator``):
    GOAL_PREDICATES / KIND_CATEGORY            — the predicate table + kind->category map
    call_task_agent(scene, ...)                — scene -> validated + feasibility-checked Task
    validate_task(task, scene)                 — grammar/structural errors (RoboLab-parity)
    check_static / check_affordance /          — the individual feasibility checks (pure, à la carte)
        check_reachable / check_fits_in
    check_feasibility(task, scene)             — orchestrator -> aggregate feasibility report
    task_to_dict(task) / generate_task(...)    — serialize / end-to-end (prompt|scene -> Task)

Run standalone to decide ONE task for a scene (generating the scene first if none is pre-specified):

    .venv/bin/python -m agentic_pipeline.task_generator "fold the shirt and put it in the bowl"
    .venv/bin/python -m agentic_pipeline.task_generator "tidy the desk" --scene outputs/sceneGen/lunch_tidyup_desk

Credentials/transport are shared with ``agentic_pipeline.scene_generator`` (raw HTTPS + Claude Code OAuth; the SDK
conflicts with the venv's isaacsim typing-extensions pin).
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_pipeline import scene_generator as sg
from deformableManipulationTools.params import FRANKA, TABLE

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two supported robots are kinematically identical; max PARALLEL-JAW opening is 2x the per-finger
# prismatic limit (params.FRANKA.gripper_open). Used only as a coarse "is this even graspable" width
# ceiling — the task generator does NOT tune grip force/width (that is the trajectory pipeline's job).
MAX_JAW_WIDTH = 2.0 * FRANKA.gripper_open           # ~0.08 m

# kind -> coarse category (reuses scene_generator's own mapping; deformable types split out).
KIND_CATEGORY = {
    "ycb_mesh": "rigid", "rigid_box": "rigid", "rubiks_cube": "rigid",
    "cloth": "cloth", "cable": "cable", "soft_mesh": "squishy", "soft_block": "squishy",
}
DEFORMABLE_CATEGORIES = {"cloth", "cable", "squishy"}

# Max fraction of an object's footprint allowed to protrude from a container mouth and still count
# as "contained" — the partial-containment model (a banana settling in a bowl). 0 = RoboLab-strict.
PARTIAL_CONTAINMENT = 0.35

# Open-top containers a thing can be placed INTO — resolved from the catalog's "container" flag
# (``entry["container"] is True``) so any imported container (bowls, buckets, bins, non-articulated
# cabinets/crates/totes) is recognized automatically. The legacy allow-list below is the fallback
# for catalog entries that predate the flag.
CONTAINER_CLASSES = {"kitchenware", "storage"}
CONTAINER_NAMES = {"bowl", "mug", "pitcher"}         # legacy fallback (pre-"container"-flag entries)


# ---------------------------------------------------------------------------------------------
# The predicate library (our analogue of RoboLab's conditionals.py)
# ---------------------------------------------------------------------------------------------
# Each predicate declares: required object params, the categories each param object may belong to,
# whether it needs a container, and a human verb. ``obj_categories`` gates AFFORDANCE — a predicate
# that names a category set excluding an object's category is infeasible for that object.
GOAL_PREDICATES: dict[str, dict] = {
    # --- shared: rigid AND deformable ---
    "object_in_container": {
        "params": ["object", "container"], "container_param": "container",
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "put ... in", "attributes": ["spatial"]},
    "object_on_top": {
        "params": ["object", "target"], "container_param": None,
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "put ... on", "attributes": ["spatial"]},
    "object_left_of": {
        "params": ["object", "reference"], "container_param": None,
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "move ... left of", "attributes": ["spatial"]},
    "object_right_of": {
        "params": ["object", "reference"], "container_param": None,
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "move ... right of", "attributes": ["spatial"]},
    "object_in_front_of": {
        "params": ["object", "reference"], "container_param": None,
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "move ... in front of", "attributes": ["spatial"]},
    "object_behind": {
        "params": ["object", "reference"], "container_param": None,
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "move ... behind", "attributes": ["spatial"]},
    "object_outside_of": {
        "params": ["object", "container"], "container_param": "container",
        "obj_categories": {"rigid", "cloth", "cable", "squishy"},
        "verb": "take ... out of", "attributes": ["spatial"]},
    # --- rigid-only ---
    "stacked": {
        "params": ["object", "base"], "container_param": None,
        "obj_categories": {"rigid"},
        "verb": "stack ... on", "attributes": ["procedural"]},
    "object_groups_in_containers": {
        "params": ["object", "container"], "container_param": "container",
        "obj_categories": {"rigid"},
        "verb": "sort ... into", "attributes": ["relational", "procedural"]},
    # --- deformable-only (the affordances that make deformables special) ---
    "cloth_folded": {
        "params": ["object"], "container_param": None,
        "obj_categories": {"cloth"},
        "verb": "fold", "attributes": ["deformable", "procedural"]},
    "cloth_draped_over": {
        "params": ["object", "target"], "container_param": None,
        "obj_categories": {"cloth"},
        "verb": "drape ... over", "attributes": ["deformable", "spatial"]},
    "cable_coiled": {
        "params": ["object"], "container_param": None,
        "obj_categories": {"cable"},
        "verb": "coil", "attributes": ["deformable", "procedural"]},
    "cable_routed_through": {
        "params": ["object", "target"], "container_param": None,
        "obj_categories": {"cable"},
        "verb": "route ... through", "attributes": ["deformable", "procedural"]},
    "object_compressed": {
        "params": ["object"], "container_param": None,
        "obj_categories": {"squishy"},
        "verb": "squeeze", "attributes": ["deformable"]},
}

# Every role name used across GOAL_PREDICATES (the fixed key set for the goal.params schema object;
# the API requires an explicit-property object with additionalProperties=false, not a free-form map).
_PARAM_ROLES = sorted({role for spec in GOAL_PREDICATES.values() for role in spec["params"]})


# ---------------------------------------------------------------------------------------------
# Data model (mirror RoboLab's Task)
# ---------------------------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    instruction: dict            # {"default":..., "vague":..., "specific":...}
    goal: dict                   # {"predicate": str, "params": {role: object_name, ...}}
    subtasks: list = field(default_factory=list)
    attributes: list = field(default_factory=list)
    difficulty: str = "simple"   # "simple" | "moderate" | "complex"
    feasibility: dict = field(default_factory=dict)
    scene_name: str = ""
    prompt: str = ""
    model: str = ""


def task_to_dict(task: Task) -> dict:
    return asdict(task)


# ---------------------------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------------------------
def _scene_by_name(scene: dict, catalog: dict | None = None) -> dict:
    """{object_name: {"entry": catalog_entry, "placement": scene_object}} for the scene's objects.

    A name may appear twice in a scene; the placement kept is the FIRST occurrence (feasibility is a
    property of the object type + a representative placement, not of a specific duplicate)."""
    by_name = sg.catalog_by_name(catalog)
    out: dict[str, dict] = {}
    for o in scene.get("objects", []):
        nm = o["name"]
        if nm not in out:
            out[nm] = {"entry": by_name[nm], "placement": o}
    return out


def _category(entry: dict) -> str:
    return KIND_CATEGORY[entry["kind"]]


def _is_container(entry: dict) -> bool:
    """Open-top container? Prefer the catalog's explicit ``container`` flag (set on every imported
    container — bowls, buckets, bins, non-articulated cabinets); fall back to the legacy name list
    for entries that predate the flag."""
    if entry.get("container") is True:
        return True
    return entry["name"] in CONTAINER_NAMES


# ---------------------------------------------------------------------------------------------
# Feasibility checks — pure, individually importable; each returns (ok, reason)
# ---------------------------------------------------------------------------------------------
def check_static(task: Task, scene: dict) -> tuple[bool, str]:
    """RoboLab-parity structural check: predicate known, all its params supplied and naming objects
    present in the scene, and any container param actually names an open-top container."""
    pred = task.goal.get("predicate")
    spec = GOAL_PREDICATES.get(pred)
    if spec is None:
        return False, f"unknown goal predicate {pred!r} (allowed: {sorted(GOAL_PREDICATES)})"
    params = task.goal.get("params", {})
    present = _scene_by_name(scene)
    # The schema exposes the UNION of all role names (a JSON-schema limitation), so the agent can
    # key a value under an equivalent-but-wrong role (e.g. 'reference'/'target' instead of
    # 'container' for object_in_container). Auto-repair by pulling any single stray value into the
    # missing required role before checking — the roles are positionally unambiguous (each
    # predicate has one 'object' + one partner role). Mutates task.goal.params in place so the
    # written task.json and success_spec are correct.
    strays = [v for k, v in params.items() if k not in spec["params"] and v]
    for role in spec["params"]:
        if not params.get(role) and len(strays) == 1:
            params[role] = strays[0]
            params = {k: v for k, v in params.items() if k in spec["params"]}
            task.goal["params"] = params
    for role in spec["params"]:
        nm = params.get(role)
        if not nm:
            leftover = {k: v for k, v in task.goal.get("params", {}).items()
                        if k not in spec["params"] and v}
            hint = (f" (you put {leftover} under other roles — this predicate uses EXACTLY "
                    f"{spec['params']})" if leftover else "")
            return False, f"goal {pred} is missing required param {role!r}{hint}"
        if nm not in present:
            return False, f"goal {pred} names {nm!r} which is not in the scene"
    cp = spec["container_param"]
    if cp is not None:
        cname = params[cp]
        containers = sorted(n for n, i in present.items() if _is_container(i["entry"]))
        if not _is_container(present[cname]["entry"]):
            return False, (f"{cname!r} is not an open-top container (scene containers: "
                           f"{containers}); {pred} needs one")
    return True, "structural checks pass"


def check_affordance(task: Task, scene: dict) -> tuple[bool, str]:
    """DEFORMABLE GATING (core): the predicate's allowed object categories must include the actual
    category of the manipulated object. Rejects fold-a-banana, stack-a-cable, coil-a-cube, etc."""
    pred = task.goal.get("predicate")
    spec = GOAL_PREDICATES.get(pred)
    if spec is None:
        return False, f"unknown predicate {pred!r}"
    present = _scene_by_name(scene)
    # The manipulated object is always the "object" role (every predicate has it).
    obj = task.goal.get("params", {}).get("object")
    if obj is None or obj not in present:
        return False, "affordance: no valid 'object' to manipulate"
    cat = _category(present[obj]["entry"])
    allowed = spec["obj_categories"]
    if cat not in allowed:
        return False, (f"affordance mismatch: '{spec['verb']}' ({pred}) applies to {sorted(allowed)} "
                       f"objects, but {obj!r} is {cat}")
    return True, f"affordance ok: {obj!r} ({cat}) supports '{spec['verb']}'"


def _base_xy() -> tuple[float, float]:
    """The FRAMEWORK-DEFAULT robot base in world XY (the legacy mount at the right short edge —
    NOT the origin; ``GraspExample`` mounts at (-0.45, -0.45) when ``robot_base_xform`` is None).
    The agentic pipeline passes the actual placement's base via ``base_xy`` instead."""
    return -0.45, -0.45


def check_reachable(task: Task, scene: dict, *, base_xy: tuple[float, float] | None = None,
                    reach_max: float = 0.80,
                    sweet: tuple[float, float] = (0.35, 0.65)) -> tuple[bool, str]:
    """Coarse Cartesian reachability (RoboLab has NONE — this exceeds parity; no IK). Every object
    the task manipulates or references must sit inside the usable tabletop AND have SOME PART (the
    nearest point of its footprint; nearest node for a cable) within the arm's reach radius from
    the base. Objects whose center is outside the ~0.5 m sweet-spot are warned, not rejected.
    ``base_xy`` is the robot base (None -> the framework-default legacy mount)."""
    x0, x1, y0, y1 = sg.workspace_bounds()
    bx, by = base_xy if base_xy is not None else _base_xy()
    present = _scene_by_name(scene)
    warnings = []
    for role, nm in task.goal.get("params", {}).items():
        info = present.get(nm)
        if info is None:
            continue
        pl, entry = info["placement"], info["entry"]
        x, y = float(pl.get("x", 0.0)), float(pl.get("y", 0.0))
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False, (f"{nm!r} at ({x:.2f}, {y:.2f}) is off the usable tabletop "
                           f"x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]")
        if entry["kind"] == "cable" and pl.get("nodes"):
            r = float(entry.get("config", {}).get("radius", 0.008))
            d_center = min(math.hypot(nx - bx, ny - by) for nx, ny, *_ in pl["nodes"])
            d_near = max(0.0, d_center - r)
        else:
            footprint_r = 0.5 * max(entry["dims"][0], entry["dims"][1])
            d_center = math.hypot(x - bx, y - by)
            d_near = max(0.0, d_center - footprint_r)
        if d_near > reach_max:
            return False, (f"{nm!r}: nearest point at {d_near:.2f} m exceeds the arm's "
                           f"~{reach_max:.2f} m reach (center {d_center:.2f} m)")
        if not (sweet[0] <= d_center <= sweet[1]):
            warnings.append(f"{nm!r} at reach {d_center:.2f} m is outside the "
                            f"{sweet[0]:.2f}-{sweet[1]:.2f} m sweet-spot")
    return True, ("reachable" + ("; " + "; ".join(warnings) if warnings else ""))


def _container_opening(entry: dict) -> tuple[float, float, float]:
    """(open_x, open_y, depth) [m] for a container, from its catalog footprint. The opening is the
    footprint inset ~15% for wall thickness; depth is the vertical extent."""
    dx, dy, dz = entry["dims"]
    return 0.85 * dx, 0.85 * dy, dz


def _fits_flat(obj_dims, opening) -> bool:
    ox, oy = sorted(obj_dims[:2])          # object footprint, small..large
    cx, cy = sorted(opening[:2])           # opening, small..large
    return ox <= cx and oy <= cy


def check_fits_in(obj_name: str, container_name: str, scene: dict, *,
                  allow_fold: bool = True, max_folds: int = 4) -> tuple[bool, str]:
    """Geometric container fit with the DEFORMABLE folded-volume nuance. A rigid object must fit the
    opening flat. A cloth/cable that does not fit flat MAY fit folded/coiled: each fold halves the
    larger footprint axis and adds a layer of thickness (volume conserved). Accept if some fold count
    makes the footprint fit the opening AND the stacked thickness fits the depth. Returns the fold
    count in the reason ("fits after 2 folds")."""
    present = _scene_by_name(scene)
    if obj_name not in present or container_name not in present:
        return False, f"fit: {obj_name!r} or {container_name!r} not in scene"
    oent, cent = present[obj_name]["entry"], present[container_name]["entry"]
    if not _is_container(cent):
        return False, f"fit: {container_name!r} is not an open-top container"
    odims = list(oent["dims"])
    opening = _container_opening(cent)
    cat = _category(oent)

    if _fits_flat(odims, opening):
        return True, f"{obj_name!r} fits {container_name!r} flat"

    # PARTIAL CONTAINMENT (RoboLab ellipse-mouth model + our partial extension): an elongated rigid
    # object (a banana) may still settle into a bowl if its SHORTER axis fits the mouth even though
    # the long axis protrudes. Accept that before falling through to the fold model / rejection.
    from . import packing
    rx, ry, _ = packing.container_mouth(cent["dims"])
    fx, fy = packing.rotated_extent(odims[0], odims[1], 0.0)
    if packing.fits_container_ellipse(0.0, 0.0, fx, fy, rx, ry, partial=PARTIAL_CONTAINMENT):
        return True, (f"{obj_name!r} fits {container_name!r} with partial containment "
                      f"(long axis protrudes; short axis {min(odims[0], odims[1]):.2f} m fits the "
                      f"{2*min(rx,ry):.2f} m mouth)")

    if not (allow_fold and cat in {"cloth", "cable"}):
        return False, (f"{obj_name!r} footprint {odims[0]:.2f}x{odims[1]:.2f} m does not fit the "
                       f"{container_name!r} opening {opening[0]:.2f}x{opening[1]:.2f} m")

    # Fold model: fold across the LARGER axis; each fold halves it and doubles the layer count. The
    # footprint shrinks but the stack thickens (volume conserved), so a fold that fits the OPENING
    # can still bust the DEPTH — the checker reports whichever constraint binds.
    fx, fy, thick = odims[0], odims[1], max(odims[2], 0.005)
    what = "coils" if cat == "cable" else "folds"
    footprint_ok = False
    for k in range(1, max_folds + 1):
        if fx >= fy:
            fx *= 0.5
        else:
            fy *= 0.5
        layers = 2 ** k
        stack = thick * layers
        if _fits_flat((fx, fy, 0.0), opening):
            footprint_ok = True
            if stack <= opening[2] * 1.5:
                return True, (f"{obj_name!r} fits {container_name!r} after {k} {what} "
                              f"(-> {fx:.2f}x{fy:.2f} m, ~{stack*1000:.0f} mm thick)")
    if footprint_ok:
        return False, (f"{obj_name!r} folds down to the {container_name!r} opening but its fabric "
                       f"volume overfills the ~{opening[2]:.2f} m depth (folding conserves volume)")
    return False, (f"{obj_name!r} is too large for the {container_name!r} opening "
                   f"{opening[0]:.2f}x{opening[1]:.2f} m even after {max_folds} {what}")


def check_support(obj_name: str, support_name: str, scene: dict, *,
                  min_ratio: float = 0.5) -> tuple[bool, str]:
    """SUPPORT-RATIO stacking check (RoboLab flags stacks below ~0.5-0.8 support). The fraction of
    the object's rotated footprint AABB over the support top must exceed ``min_ratio``, else the
    stack topples on settle. Support must be rigid with a similar-or-larger footprint."""
    from . import packing
    present = _scene_by_name(scene)
    if obj_name not in present or support_name not in present:
        return False, f"support: {obj_name!r} or {support_name!r} not in scene"
    oent, sent = present[obj_name]["entry"], present[support_name]["entry"]
    if _category(sent) != "rigid":
        return False, f"support: {support_name!r} is not a rigid object to stack on"
    ratio = packing.support_ratio(oent["dims"][0], oent["dims"][1],
                                  math.radians(present[obj_name]["placement"].get("yaw_deg", 0.0)),
                                  sent["dims"][0], sent["dims"][1])
    ok = ratio >= min_ratio
    return ok, (f"{obj_name!r} on {support_name!r}: {ratio*100:.0f}% supported "
                f"({'stable' if ok else f'< {min_ratio*100:.0f}% -> topples'})")


def check_group_fits(obj_names: list[str], container_name: str, scene: dict) -> tuple[bool, str]:
    """GROUP-PREDICATE feasibility: can ALL of ``obj_names`` fit into ``container_name`` together
    (RoboLab's multi-object ellipse packing with upward layering)? Rigid objects only."""
    from . import packing
    present = _scene_by_name(scene)
    cent = present.get(container_name, {}).get("entry")
    if cent is None or not _is_container(cent):
        return False, f"group-fit: {container_name!r} is not an open-top container"
    dims_list, missing = [], []
    for nm in obj_names:
        info = present.get(nm)
        if info is None:
            missing.append(nm)
        else:
            dims_list.append(list(info["entry"]["dims"]))
    if missing:
        return False, f"group-fit: {missing} not in scene"
    _, ok, reason = packing.pack_into_container(dims_list, cent["dims"], partial=PARTIAL_CONTAINMENT)
    return ok, f"group into {container_name!r}: {reason}"


def check_feasibility(task: Task, scene: dict, *, base_xy: tuple[float, float] | None = None) -> dict:
    """Orchestrator: run every relevant check for this task's predicate + objects and return an
    aggregate report ``{ok, checks:[{name,ok,reason}], summary}`` (stored on ``Task.feasibility``).
    ``base_xy`` = robot base for the reach check (None -> the framework-default legacy mount)."""
    checks = []

    def _run(name, fn):
        ok, reason = fn()
        checks.append({"name": name, "ok": bool(ok), "reason": reason})
        return ok

    ok_static = _run("static", lambda: check_static(task, scene))
    # If the structure is broken, the object-level checks would raise on missing params — stop here.
    if ok_static:
        _run("affordance", lambda: check_affordance(task, scene))
        _run("reachable", lambda: check_reachable(task, scene, base_xy=base_xy))
        pred = task.goal.get("predicate")
        spec = GOAL_PREDICATES.get(pred, {})
        params = task.goal["params"]
        cp = spec.get("container_param")
        if cp is not None:
            obj = params.get("object")
            cont = params.get(cp)
            if pred == "object_groups_in_containers":     # GROUP feasibility: all objects together
                objs = obj if isinstance(obj, list) else [obj]
                _run("group_fits", lambda: check_group_fits(objs, cont, scene))
            else:
                _run("fits_in", lambda: check_fits_in(obj, cont, scene))
        # SUPPORT-RATIO stacking check for on-top / stacked goals.
        if pred in ("object_on_top", "stacked"):
            support = params.get("target") or params.get("base")
            if support:
                _run("support_ratio", lambda: check_support(params.get("object"), support, scene))
    all_ok = all(c["ok"] for c in checks)
    fails = [c for c in checks if not c["ok"]]
    summary = "all feasibility checks pass" if all_ok else "; ".join(c["reason"] for c in fails)
    return {"ok": all_ok, "checks": checks, "summary": summary}


def validate_task(task: Task, scene: dict) -> list[str]:
    """Grammar/structural errors only (RoboLab-parity ``verify_task_valid``). Feasibility is separate
    (``check_feasibility``). Returns a list of human-readable error strings (empty = valid)."""
    errs: list[str] = []
    if not task.instruction or not task.instruction.get("default"):
        errs.append("instruction.default is required")
    for variant in ("default", "vague", "specific"):
        if variant not in (task.instruction or {}):
            errs.append(f"instruction is missing the {variant!r} variant")
    ok, reason = check_static(task, scene)
    if not ok:
        errs.append(reason)
    if not task.subtasks:
        errs.append("at least one subtask is required")
    if task.difficulty not in ("simple", "moderate", "complex"):
        errs.append(f"difficulty {task.difficulty!r} must be simple|moderate|complex")
    return errs


# ---------------------------------------------------------------------------------------------
# The agent: scene -> Task (validated + feasibility-checked, feedback retries)
# ---------------------------------------------------------------------------------------------
def _predicate_table_text() -> str:
    lines = []
    for name, spec in GOAL_PREDICATES.items():
        cats = "/".join(sorted(spec["obj_categories"]))
        lines.append(f'- {name}(params={spec["params"]}) — "{spec["verb"]}"; object must be [{cats}]')
    return "\n".join(lines)


def _agent_system(scene: dict) -> str:
    present = _scene_by_name(scene)
    obj_lines = []
    for nm, info in present.items():
        e = info["entry"]
        cat = _category(e)
        d = e["dims"]
        tag = f"{cat}, {e['class']}"
        if _is_container(e):
            tag += ", CONTAINER (open-top)"
        obj_lines.append(f'- {nm} [{tag}, footprint {d[0]:.2f}x{d[1]:.2f} m] — {e["description"]}')
    return f"""You propose ONE meaningful, physically feasible manipulation task for a Franka-arm robot,
given a tabletop scene that may contain RIGID and DEFORMABLE objects (cloth, cables/ropes, squishy
FEM objects). You choose the goal and describe it; you do NOT decide grasp force or trajectory.

SCENE OBJECTS (reference objects ONLY by these exact names):
{chr(10).join(obj_lines)}

GOAL PREDICATES (choose exactly one; its "object" param is the thing being manipulated):
{_predicate_table_text()}

HARD RULES (the feasibility checker enforces them — violating one gets the task rejected):
- AFFORDANCE: the predicate must match the object's category. Only fold/drape a CLOTH; only
  coil/route a CABLE; only squeeze a SQUISHY object; only stack RIGID objects. Do not fold a banana
  or stack a rope.
- CONTAINER: object_in_container / object_outside_of / sort need an OPEN-TOP container object
  (bowl, mug, pitcher) — a can, box, or block is not a container.
- FIT: to place an object into a container it must fit — a large cloth may need to be FOLDED first
  (fold it in your subtasks); if it cannot fit even folded, pick another goal.
- REACH: only use objects on the tabletop within the arm's reach.

Also produce THREE instruction phrasings (default = clear and complete; vague = terse/underspecified;
specific = names colors/attributes), a SUBTASK list (ordered atomic steps, e.g. grasp -> fold ->
place), competency ATTRIBUTES, and a difficulty (simple|moderate|complex). Prefer tasks that exercise
the deformable objects when the scene has them. Respond with the task JSON only."""


def _task_schema(scene: dict) -> dict:
    names = list(_scene_by_name(scene).keys())
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "short lowercase slug for the task"},
            "instruction": {
                "type": "object",
                "properties": {
                    "default": {"type": "string"},
                    "vague": {"type": "string"},
                    "specific": {"type": "string"},
                },
                "required": ["default", "vague", "specific"],
                "additionalProperties": False,
            },
            "goal": {
                "type": "object",
                "properties": {
                    "predicate": {"type": "string", "enum": list(GOAL_PREDICATES)},
                    "params": {
                        "type": "object",
                        "description": "role -> scene object name; supply exactly the roles the "
                                       "chosen predicate requires (see the predicate table)",
                        "properties": {role: {"type": "string", "enum": names}
                                       for role in _PARAM_ROLES},
                        "additionalProperties": False,
                    },
                },
                "required": ["predicate", "params"],
                "additionalProperties": False,
            },
            "subtasks": {"type": "array", "items": {"type": "string"}},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
        },
        "required": ["name", "instruction", "goal", "subtasks", "attributes", "difficulty"],
        "additionalProperties": False,
    }


def _task_from_json(data: dict, scene: dict) -> Task:
    return Task(
        name=data.get("name", "task"),
        instruction=data.get("instruction", {}),
        goal=data.get("goal", {}),
        subtasks=data.get("subtasks", []),
        attributes=data.get("attributes", []),
        difficulty=data.get("difficulty", "simple"),
        scene_name=scene.get("name", ""),
    )


def call_task_agent(scene: dict, *, model: str = sg.DEFAULT_MODEL, attempts: int = 3,
                    verbose: bool = True) -> Task:
    """The agent loop: scene -> Task, with structural validation + the deformable-aware feasibility
    checks, and natural-language feedback retries (RoboLab's validate-and-refine pattern)."""
    system, schema = _agent_system(scene), _task_schema(scene)
    messages = [{"role": "user", "content":
                 f"Scene '{scene.get('name','')}' ({scene.get('description','')}). "
                 f"Original request: {scene.get('prompt','(none)')}. Propose one task."}]
    task, use_model = None, model
    for attempt in range(attempts):
        try:
            resp = sg._messages_request(system, messages, use_model, schema)
            text = sg._response_text(resp)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            detail = exc.read().decode(errors="replace")[:300] if isinstance(exc, urllib.error.HTTPError) else ""
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[taskGen] {use_model} failed ({exc} {detail}); falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
        task = _task_from_json(json.loads(text), scene)
        task.model = use_model
        task.prompt = scene.get("prompt", "")
        errs = validate_task(task, scene)
        feedback = "; ".join(errs)
        if not errs:
            report = check_feasibility(task, scene)
            task.feasibility = report
            if report["ok"]:
                if verbose:
                    print(f"[taskGen] task {task.name!r} accepted on attempt {attempt + 1}")
                return task
            feedback = report["summary"]
        if verbose:
            print(f"[taskGen] attempt {attempt + 1} rejected: {feedback}")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your task was rejected: {feedback}. "
                                                 f"Return a corrected task JSON (pick a feasible goal)."}]
    if task is None:
        raise RuntimeError("task agent produced no usable task")
    task.feasibility = check_feasibility(task, scene)     # best-effort report on the final attempt
    if verbose:
        print("[taskGen] accepting best-effort task after retries (feasibility may be partial)")
    return task


# ---------------------------------------------------------------------------------------------
# Scene loading + end-to-end
# ---------------------------------------------------------------------------------------------
def load_scene(scene_dir: Path | str) -> dict:
    """Load a scene dict from an existing ``outputs/`` scene directory (or a ``scene.json`` path)."""
    p = Path(scene_dir)
    scene_json = p if p.suffix == ".json" else p / "scene.json"
    if not scene_json.exists():
        raise FileNotFoundError(f"no scene.json at {scene_json}")
    return json.loads(scene_json.read_text())


def write_task(task: Task, scene_dir: Path | str) -> Path:
    """Write ``task.json`` next to the scene's ``scene.json``. Returns the path."""
    p = Path(scene_dir)
    out = p if p.is_dir() else p.parent
    task_json = out / "task.json"
    task_json.write_text(json.dumps(task_to_dict(task), indent=1))
    return task_json


def generate_task(prompt: str | None = None, *, scene: dict | None = None,
                  scene_dir: Path | str | None = None, model: str = sg.DEFAULT_MODEL,
                  seed: int = 0, verbose: bool = True) -> tuple[Task, dict]:
    """End-to-end for the pipeline: give a ready ``scene`` dict, a ``scene_dir`` to load, or a
    ``prompt`` to generate a scene first (scene generator, no render). Returns ``(Task, scene)``."""
    if scene is None:
        if scene_dir is not None:
            scene = load_scene(scene_dir)
        elif prompt is not None:
            if verbose:
                print(f"[taskGen] no scene given — generating one from the prompt: {prompt!r}")
            scene = sg.call_scene_agent(prompt, seed=seed, verbose=verbose)
        else:
            raise ValueError("generate_task needs one of: scene, scene_dir, or prompt")
    task = call_task_agent(scene, model=model, verbose=verbose)
    return task, scene


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decide one feasible manipulation task for a scene "
                                             "(generating the scene from the prompt if none is given).")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="natural-language request (used to generate a scene if --scene is omitted)")
    ap.add_argument("--scene", default=None,
                    help="existing scene dir or scene.json (e.g. outputs/sceneGen/<name>)")
    ap.add_argument("--model", default=sg.DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="print only; do not write task.json")
    args = ap.parse_args()

    if args.scene is None and args.prompt is None:
        ap.error("give a prompt, a --scene, or both")

    task, scene = generate_task(prompt=args.prompt, scene_dir=args.scene, model=args.model, seed=args.seed)

    print(json.dumps(task_to_dict(task), indent=1))
    if not args.no_write:
        target = args.scene
        if target is None:
            # A fresh scene was generated in-memory; persist it so task.json has a home next to it.
            scene_json, _ = sg.write_scene(scene)
            target = scene_json.parent
        path = write_task(task, target)
        print(f"task: {path}")
