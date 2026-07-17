"""Pipeline stage 1 — scene gen: object SELECTION + PLACEMENT only.

Unlike ``agentic_pipeline.scene_generator`` (which also picks a background/table look), this
stage owns exactly what RoboLab's scene gen owns MINUS the look: which objects exist and where
they sit on the workspace table, validated by the spatial solver and a headless PHYSICS SETTLE
CHECK (objects must come to rest ON the table — nothing falls off, blows up, or keeps moving).
Backgrounds/tables/lighting/cameras belong to env gen; the task and robot placement to task gen.

Direction words in prompts and relations are ROBOT-POV (see ``geometry.directions_text``): scene
gen runs before task gen picks a placement, so they are anchored to the placement that is already
fixed (scene_init mode) or to the DEFAULT mount.

The agent prompt template lives in ``prompts/scene_system.md`` — this module only fills its slots.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

from . import scene_generator as sg
from . import geometry, load_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent


def catalog_lines(catalog: dict) -> str:
    kind_map = {"ycb_mesh": "rigid", "rigid_box": "rigid", "rubiks_cube": "rigid",
                "cloth": "cloth", "cable": "cable", "soft_mesh": "squishy", "soft_block": "squishy"}
    lines = []
    for o in catalog["objects"]:
        d = o["dims"]
        lines.append(f'- {o["name"]} [{kind_map[o["kind"]]}, {o["class"]}, footprint '
                     f'{d[0]:.2f}x{d[1]:.2f} m] — {o["description"]}')
    return "\n".join(lines)


def scene_schema(catalog: dict, names: list[str] | None = None) -> dict:
    """The standalone generator's schema MINUS the look fields (background/table are env gen's).
    ``names`` restricts the object enum (scene_init rearrange mode)."""
    schema = sg._scene_schema(catalog)
    for look in ("background", "table"):
        schema["properties"].pop(look)
        schema["required"].remove(look)
    if names is not None:
        schema["properties"]["objects"]["items"]["properties"]["name"]["enum"] = sorted(set(names))
    return schema


def scene_system(catalog: dict, placement: dict | None = None, *, count_hint: int | None = None,
                 fixed_multiset: dict | None = None) -> str:
    x0, x1, y0, y1 = sg.workspace_bounds()
    count_rule = "1 to 7 objects total (default 3-5 for simple scenes)."
    if count_hint:
        count_rule = f"Use about {int(count_hint)} objects (hard range 1-7)."
    kx, ky, kr, kmin = geometry.home_tcp_keepout(placement or geometry.default_placement())
    extra = (f"- The robot's PARKED gripper hovers over ({kx:.2f}, {ky:.2f}): keep objects taller "
             f"than {kmin:.2f} m at least {kr:.2f} m away from that point (short/flat objects are "
             f"fine there).\n")
    if fixed_multiset:
        listing = ", ".join(f"{n} x{c}" for n, c in sorted(fixed_multiset.items()))
        extra += ("- REARRANGE MODE: use EXACTLY this multiset of objects — no more, no fewer, no "
                  f"substitutions: {listing}. Give each a NEW placement/arrangement.\n")
    return load_prompt(
        "scene_system",
        x0=f"{x0:.2f}", x1=f"{x1:.2f}", y0=f"{y0:.2f}", y1=f"{y1:.2f}",
        directions=geometry.directions_text(placement),
        catalog=catalog_lines(catalog),
        containers=", ".join(sg.CONTAINER_NAMES),
        count_rule=count_rule,
        extra_rules=extra,
    )


def _multiset(objs: list) -> dict:
    out: dict[str, int] = {}
    for o in objs:
        out[o["name"]] = out.get(o["name"], 0) + 1
    return out


def call_scene_agent(prompt: str, *, placement: dict | None = None, model: str = sg.DEFAULT_MODEL,
                     attempts: int = 3, seed: int = 0, count_hint: int | None = None,
                     fixed_multiset: dict | None = None, verbose: bool = True) -> dict:
    """Prompt -> scene dict (objects only), with grammar validation + the placement-aware spatial
    solver and feedback retries. ``fixed_multiset`` (scene_init rearrange) forces the exact object
    multiset of a source scene."""
    catalog = sg.load_catalog()
    placement = placement or geometry.default_placement()
    system = scene_system(catalog, placement, count_hint=count_hint, fixed_multiset=fixed_multiset)
    schema = scene_schema(catalog, names=list(fixed_multiset) if fixed_multiset else None)
    keepout = geometry.home_tcp_keepout(placement)
    messages = [{"role": "user", "content": prompt}]
    scene, use_model = None, model
    for attempt in range(attempts):
        try:
            resp = sg._messages_request(system, messages, use_model, schema)
            text = sg._response_text(resp)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[pipeline/scene] {use_model} failed ({exc}); falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
        scene = json.loads(text)
        scene["prompt"], scene["model"] = prompt, use_model
        errs = sg.validate_scene(scene, catalog)
        if fixed_multiset and _multiset(scene.get("objects", [])) != fixed_multiset:
            want = ", ".join(f"{n} x{c}" for n, c in sorted(fixed_multiset.items()))
            errs.append(f"rearrange mode requires EXACTLY these objects: {want}")
        feedback = "; ".join(errs)
        if not errs:
            ok, feedback = sg.resolve_placements(scene, catalog, seed=seed,
                                                 facing_yaw_deg=placement["yaw_deg"],
                                                 tall_keepout=keepout)
            if ok:
                if verbose:
                    print(f"[pipeline/scene] scene {scene['name']!r} accepted on attempt {attempt + 1}")
                return scene
        if verbose:
            print(f"[pipeline/scene] attempt {attempt + 1} rejected: {feedback}")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Your scene was rejected: {feedback}. "
                                                 f"Return a corrected scene JSON."}]
    if scene is None:
        raise RuntimeError("scene agent produced no usable scene")
    sg.resolve_placements(scene, catalog, seed=seed, facing_yaw_deg=placement["yaw_deg"],
                          tall_keepout=keepout)
    if verbose:
        print("[pipeline/scene] accepting best-effort layout after retries")
    return scene


def settle_check(demo_py: Path | str, *, device: str = "cuda:0", verbose: bool = True) -> dict:
    """RoboLab's post-settle stability check, headless (no render): run the scene's demo for its
    settle length in a subprocess and measure — NaN blow-ups, objects that left the table, and
    residual motion. Returns the metrics dict (``ok`` aggregates)."""
    cmd = [sys.executable, "-m", "agentic_pipeline.settle", str(demo_py), "--device", device]
    if verbose:
        print("[pipeline/scene] settle check:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in reversed((res.stdout or "").splitlines()):
        if line.startswith("SETTLE_JSON:"):
            return json.loads(line[len("SETTLE_JSON:"):])
    raise RuntimeError(f"settle check produced no report (exit {res.returncode}):\n"
                       f"{res.stdout[-1500:]}\n{res.stderr[-1500:]}")


def settle_feedback(report: dict) -> str:
    """The natural-language settle feedback for the agent retry (RoboLab FeedbackSystem parity)."""
    parts = []
    if not report.get("finite", True):
        parts.append("the simulation blew up (non-finite state)")
    for b in report.get("off_table", []):
        parts.append(f"{b} fell off the table during settle")
    for b in report.get("large_displacement", []):
        parts.append(f"{b} moved more than 5 cm while settling (unstable placement)")
    if report.get("residual_motion"):
        parts.append("objects were still moving at the end of the settle window")
    return ("the physics settle check failed: " + "; ".join(parts)
            + ". Re-place the affected objects (more clearance, flatter/more supported poses).")
