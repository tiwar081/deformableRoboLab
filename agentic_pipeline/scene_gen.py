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
CLOTH_SCENE_MAX = sg.CLOTH_SCENE_MAX    # single definition in scene_generator (with the deformable limit)


def catalog_lines(catalog: dict) -> str:
    kind_map = {"ycb_mesh": "rigid", "rigid_box": "rigid", "rubiks_cube": "rigid",
                "cloth": "cloth", "cable": "cable", "soft_mesh": "squishy", "soft_block": "squishy"}
    lines = []
    for o in catalog["objects"]:
        d = o["dims"]
        category = o.get("category", kind_map[o["kind"]])
        lines.append(f'- {o["name"]} [{category}, {o["class"]}, footprint '
                     f'{d[0]:.2f}x{d[1]:.2f} m] — {o["description"]}')
    return "\n".join(lines)


def scene_schema(catalog: dict, names: list[str] | None = None) -> dict:
    """The standalone generator's schema MINUS the look fields (background/table are env gen's).
    ``names`` restricts the object enum (scene_init rearrange mode; pass None when substitutions
    are allowed, so the agent can reach the rest of the catalog)."""
    schema = sg._scene_schema(catalog)
    for look in ("background", "table"):
        schema["properties"].pop(look)
        schema["required"].remove(look)
    if names is not None:
        schema["properties"]["objects"]["items"]["properties"]["name"]["enum"] = sorted(set(names))
    return schema


def layout_lines(objs: list) -> str:
    """One line per object of an EXISTING layout — what a rearrange must deliberately differ from."""
    out = []
    for o in objs:
        rel = o.get("relation") or None
        tgt = objs[rel["target"]]["name"] if rel and 0 <= rel.get("target", -1) < len(objs) else None
        where = f", {rel['type']} {tgt}" if tgt else ""
        out.append(f"  * {o['name']} at ({o.get('x', 0.0):+.2f}, {o.get('y', 0.0):+.2f}) "
                   f"yaw {o.get('yaw_deg', 0.0):.0f} deg{where}")
    return "\n".join(out)


def multiset_delta(new: dict, old: dict) -> tuple[dict, dict]:
    """(added, removed) name->count between two multisets. A SUBSTITUTION shows up as one name in
    each with equal totals; an add/drop shows up as unequal totals."""
    added = {n: c - old.get(n, 0) for n, c in new.items() if c > old.get(n, 0)}
    removed = {n: c - new.get(n, 0) for n, c in old.items() if c > new.get(n, 0)}
    return added, removed


def _multiset_errors(got: dict, want: dict, allow_substitutions: int) -> list[str]:
    """Enforce the rearrange object set: identical to the source, except up to
    ``allow_substitutions`` one-for-one SWAPS. Objects are NEVER added or removed — the count is
    invariant in every case (so an infeasible requested swap, e.g. a bag into a 5+ object scene,
    is simply not performed rather than shrinking the scene)."""
    if got == want:
        return []
    listing = ", ".join(f"{n} x{c}" for n, c in sorted(want.items()))
    if not allow_substitutions:
        return [f"rearrange mode requires EXACTLY these objects: {listing}"]
    added, removed = multiset_delta(got, want)
    if sum(got.values()) != sum(want.values()):
        return [f"a substitution never ADDS or REMOVES an object: the count must stay "
                f"{sum(want.values())} (you have {sum(got.values())}). Start from {listing} and "
                f"swap strictly in place."]
    if sum(added.values()) > allow_substitutions:      # counts equal => n_added == n_removed
        return [f"at most {allow_substitutions} substitution(s) are allowed, but you swapped "
                f"{sum(added.values())} (removed {sorted(removed)}, added {sorted(added)}). "
                f"Start from {listing}."]
    return []


def scene_system(catalog: dict, placement: dict | None = None, *, count_hint: int | None = None,
                 fixed_multiset: dict | None = None, prev_objects: list | None = None,
                 change_request: str | None = None, substitution_request: str | None = None,
                 allow_substitutions: int = 0, substitutions_free: bool = False,
                 deformable_baseline: int = 0) -> str:
    x0, x1, y0, y1 = sg.workspace_bounds()
    count_rule = "1 to 7 objects total (default 3-5 for simple scenes)."
    if fixed_multiset and substitutions_free:
        # User-requested object changes: the request rules, including over the count.
        count_rule = (f"The scene starts from the {sum(fixed_multiset.values())} source objects "
                      f"below. Change the object set ONLY as the USER'S REQUESTED CHANGES require "
                      f"(they may add, remove, or swap objects); beyond that, never add or drop.")
    elif fixed_multiset:
        # Rearrange mode: the count is a PROPERTY OF THE SOURCE SCENE, not a choice — say so
        # instead of quoting a range the agent is not free to move within.
        count_rule = (f"The object count is FIXED at {sum(fixed_multiset.values())} by the "
                      f"rearrange multiset below; never add or drop an object."
                      + ("" if allow_substitutions else " Never substitute one either."))
    elif count_hint:
        count_rule = f"Use about {int(count_hint)} objects (hard range 1-7)."
    kx, ky, kr, kmin = geometry.home_tcp_keepout(placement or geometry.default_placement())
    extra = (f"- The robot's PARKED gripper hovers over ({kx:.2f}, {ky:.2f}): keep objects taller "
             f"than {kmin:.2f} m at least {kr:.2f} m away from that point (short/flat objects are "
             f"fine there).\n")
    if fixed_multiset:
        # Rearrange: the SCENARIO (the prompt) is unchanged and is not up for reinterpretation —
        # what changes is the ARRANGEMENT, enough that a different manipulation task becomes the
        # natural one. Showing the previous layout is what lets the agent differ from it on purpose.
        listing = ", ".join(f"{n} x{c}" for n, c in sorted(fixed_multiset.items()))
        extra += (
            "- REARRANGE MODE: this scene ALREADY EXISTS. The prompt above is its ORIGINAL scenario "
            "and does NOT change — do not reinterpret it, do not swap the setting. Start from "
            f"EXACTLY this multiset of objects: {listing}.\n"
            "- Your job is to MEANINGFULLY RE-ARRANGE those same objects so that a DIFFERENT "
            "manipulation task becomes the natural one: move objects to different parts of the "
            "table, change which object is near / on / in which, re-orient containers and elongated "
            "objects (yaw), form or break up a stack. Nudging the old layout is NOT a rearrangement. "
            "Write a fresh 'description' for the NEW arrangement (the scenario stays the same, the "
            "arrangement is what you are describing).\n")
        if substitutions_free and substitution_request:
            # The USER asked for these object-set changes: follow them verbatim — they override
            # every composition limit (count, duplicates, deformable/cable limits, cloth cap).
            extra += (
                f"- OBJECT-SET CHANGES REQUESTED BY THE USER — follow them EXACTLY, no matter "
                f"what: {substitution_request}\n"
                "  These changes OVERRIDE the HARD CONSTRAINTS above wherever they conflict "
                "(object count, duplicate cap, the bag/garment/squishy limit, the cable limit, "
                "the cloth-scene cap): the user explicitly asked. Apply the requested changes and "
                "NOTHING ELSE — every object the request does not touch stays in the scene.\n")
        elif allow_substitutions:
            extra += (
                f"- SUBSTITUTIONS ARE ALLOWED here, at most {allow_substitutions}: you may REPLACE "
                "up to that many objects with different catalog objects, as long as the scenario "
                "still makes sense. A substitution swaps one object for another, strictly "
                "one-for-one — it NEVER adds or removes an object, the count is invariant. "
                "Everything not substituted must stay. The composition limits above still apply "
                "to the RESULT — a swap that brings in a second deformable or cable must swap the "
                "first one out in the same move.\n")
            if sum(fixed_multiset.values()) > CLOTH_SCENE_MAX:
                extra += (
                    f"- A BAG or GARMENT CANNOT be substituted into this scene: a cloth scene is "
                    f"limited to {CLOTH_SCENE_MAX} objects total and this one has "
                    f"{sum(fixed_multiset.values())} (the count never changes). If a requested "
                    f"substitution asks for one, SKIP that substitution (keep the original object) "
                    f"and say so in the description — never shrink the scene to make it fit.\n")
            if substitution_request:
                extra += f"- SUBSTITUTION(S) REQUESTED: {substitution_request}\n"
        else:
            extra += "- No substitutions: every object must be one of the names above.\n"
        if prev_objects:
            extra += ("- PREVIOUS ARRANGEMENT — differ from it substantially:\n"
                      f"{layout_lines(prev_objects)}\n")
        if change_request:
            extra += (f"- CHANGE REQUESTED (honor it in the new arrangement): {change_request}\n")
    return load_prompt(
        "scene_system",
        x0=f"{x0:.2f}", x1=f"{x1:.2f}", y0=f"{y0:.2f}", y1=f"{y1:.2f}",
        directions=geometry.directions_text(placement),
        catalog=catalog_lines(catalog),
        containers=", ".join(sg.container_names(catalog)),
        count_rule=count_rule,
        deformable_rule=sg.deformable_limit_text(deformable_baseline),
        cloth_scene_others=str(CLOTH_SCENE_MAX - 1),
        extra_rules=extra,
    )


def _multiset(objs: list) -> dict:
    out: dict[str, int] = {}
    for o in objs:
        out[o["name"]] = out.get(o["name"], 0) + 1
    return out


def deformable_baseline_of(multiset: dict, catalog: dict | None = None) -> int:
    """How many particle deformables a source multiset already carries — grandfathers the limit
    for rearranges of a scene that (by an earlier user request) already exceeds it."""
    by_name = sg.catalog_by_name(catalog)
    return sum(c for n, c in multiset.items()
               if n in by_name and sg.particle_family(by_name[n]) is not None)


def call_scene_agent(prompt: str, *, placement: dict | None = None, model: str = sg.DEFAULT_MODEL,
                     attempts: int = 3, seed: int = 0, count_hint: int | None = None,
                     fixed_multiset: dict | None = None, prev_objects: list | None = None,
                     change_request: str | None = None, substitution_request: str | None = None,
                     allow_substitutions: int = 0, substitutions_free: bool = False,
                     verbose: bool = True) -> dict:
    """Prompt -> scene dict (objects only), with grammar validation + the placement-aware spatial
    solver and feedback retries.

    ``prompt`` is the BROAD scenario ("a messy office desk after lunch"); the per-object layout is
    the agent's job, and the resulting scene's ``description`` is what states it. In rearrange
    (scene_init) mode the prompt is the SOURCE run's prompt VERBATIM — ``fixed_multiset`` forces its
    exact object multiset, ``prev_objects`` is the layout to differ from, and ``change_request`` is
    the optional tweak. The scenario never changes there; the arrangement does.

    Substitution regimes (rearrange only):
    - ``substitutions_free=True`` — the USER wrote ``substitution_request``: follow it verbatim.
      No multiset or composition enforcement (adds/removes/any swap count/extra deformables are
      all valid if requested); only names/bounds/relations are still checked.
    - ``allow_substitutions`` > 0, not free — an AGENT-inferred request: at most N one-for-one
      swaps, count invariant, composition rules enforced against the source's deformable baseline
      (a source already over the limit is grandfathered, never forced to shed).
    Either way the whole catalog re-opens in the schema."""
    catalog = sg.load_catalog()
    placement = placement or geometry.default_placement()
    baseline = deformable_baseline_of(fixed_multiset, catalog) if fixed_multiset else 0
    system = scene_system(catalog, placement, count_hint=count_hint, fixed_multiset=fixed_multiset,
                          prev_objects=prev_objects, change_request=change_request,
                          substitution_request=substitution_request,
                          allow_substitutions=allow_substitutions,
                          substitutions_free=substitutions_free,
                          deformable_baseline=baseline)
    schema = scene_schema(catalog, names=list(fixed_multiset)
                          if fixed_multiset and not (allow_substitutions or substitutions_free)
                          else None)
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
        # Composition/multiset enforcement by regime:
        # - free (user-requested object changes): NONE — the request rules; names/bounds/relations
        #   below still apply.
        # - inferred substitutions: composition ON (vs the source's deformable baseline — a
        #   swapped-in garment is not pre-validated) + the strict swap-count/invariant-count check.
        # - plain rearrange: the multiset-equality check SUBSUMES the composition rules (the
        #   multiset came from a scene that already passed them).
        # - fresh scene: composition ON, no multiset to hold.
        errs = sg.validate_scene(
            scene, catalog,
            check_composition=(not fixed_multiset or bool(allow_substitutions))
                              and not substitutions_free,
            deformable_baseline=baseline)
        if fixed_multiset and not substitutions_free:
            errs += _multiset_errors(_multiset(scene.get("objects", [])), fixed_multiset,
                                     allow_substitutions)
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


def write_back_settled_poses(scene: dict, report: dict) -> int:
    """RoboLab's settled-pose write-back (their ``--replace``): overwrite each scene object's
    x/y/yaw with its SETTLED pose from the physics run, so the stored scene reflects where objects
    actually came to rest (relations/stacks resolved, small drift baked in). Rigid bodies only
    (deformables have no single pose); matched by label, duplicates by order. Records the original
    spawn pose under ``spawn_x/spawn_y/spawn_yaw_deg`` so it is not lost. Returns the count updated.
    Skipped when the settle was not finite (poses would be garbage)."""
    if not report.get("finite", True):
        return 0
    by_label: dict[str, list] = {}
    for s in report.get("settled", []):
        by_label.setdefault(s["label"], []).append(s)
    cursors: dict[str, int] = {}
    n = 0
    rigid_kinds = {"ycb_mesh", "rigid_box", "rubiks_cube"}
    for o in scene.get("objects", []):
        # Rigid bodies only: a cable/cloth/squishy has no single settled pose (its shape is the
        # particle cloud), and its 'x/y/yaw' is a spawn hint the builder re-derives, not a body pose.
        if sg.catalog_by_name()[o["name"]]["kind"] not in rigid_kinds:
            continue
        # a body is labeled with the asset stem; ycb uses the usd stem, others the kind name.
        label = _object_label(o)
        pool = by_label.get(label)
        if not pool:
            continue
        k = cursors.get(label, 0)
        if k >= len(pool):
            continue
        cursors[label] = k + 1
        s = pool[k]
        o.setdefault("spawn_x", o.get("x"))
        o.setdefault("spawn_y", o.get("y"))
        o.setdefault("spawn_yaw_deg", o.get("yaw_deg"))
        o["x"], o["y"], o["yaw_deg"] = s["x"], s["y"], s["yaw_deg"]
        n += 1
    return n


def _object_label(o: dict) -> str:
    """The physics body label the framework assigns to a scene object (matches settle.py labels)."""
    e = sg.catalog_by_name()[o["name"]]
    kind = e["kind"]
    if kind == "ycb_mesh":
        return Path(e["config"]["usd_subpath"]).stem
    return {"rigid_box": "cube", "rubiks_cube": "rubiks_cube"}.get(kind, kind)


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
