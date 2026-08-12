"""agent_pipeline.py — run the full agentic pipeline: scene gen -> task gen -> environment gen.

Three modes (see ``agentic_pipeline/`` for the stages and ``agentic_pipeline/SKILL.md`` for the
interactive Claude Code session that mirrors RoboLab's skill):

  --user                 RoboLab-style interview on stdin: scene description, object count, robot
                         placement (default mount vs task-specific), camera specification, visual
                         verification, output directory — MINUS whatever --scene_init already
                         fixes. Everything else runs the same pipeline.
  (no flags)             Userless defaults: the agent INVENTS the prompt (conditioned on the object
                         catalog), default robot placement (middle of the back LONG edge), default
                         camera (front bird's-eye opposite the robot).
  --scene_init PATH      Rearrange mode: reuse a previous run's ASSETS (exact object multiset),
                         robot placement, environment AND its scenario PROMPT (verbatim) — what
                         changes is the ARRANGEMENT (positions, orientations, stacks, what sits
                         in/on what), enough that a DIFFERENT task becomes natural. Everything it
                         inherits is DEAD input here: --count and --placement are rejected,
                         --camera too when the source has an env.json, and --user skips those
                         questions (see ``scene_init_reuse``). A positional prompt here is not a
                         new scenario but a CHANGE REQUEST for the rearrangement (blank = the agent
                         invents one). --substitute is the only way the object set may change:
                         with a USER phrase it is followed VERBATIM (adds/removes/any number of
                         swaps/extra deformables all valid -- no enforcement); passed BARE, the
                         agent infers 1-MAX_SUBSTITUTIONS strict one-for-one swaps (count
                         invariant, composition limits enforced).

The scene PROMPT is a broad situation ("a messy office desk after lunch"); the concrete inventory
and layout are the scene agent's job and land in the scene's ``description``.

Artifacts land in ``outputs/agenticPipeline/<name>/``: scene.json, task.json, env.json, the
generated demo data file, pipeline.json (manifest), and the rendered stills
(scene_overview.png + scene_wrist.png). Post-render visual verification is ON by default
(--no-verify skips it); it inspects BOTH stills.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

from agentic_pipeline import (build, env_gen, geometry, load_prompt, scene_gen, scene_generator as sg,
                               scene_metrics, task_gen, task_generator as tg)

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "agenticPipeline"
SETTLE_RETRIES = 3          # RoboLab-parity: re-generate the scene up to 3x on a physics-settle failure
MAX_SUBSTITUTIONS = 2       # rearrange: how many objects a substitution request may swap out


# ---------------------------------------------------------------------------------------------
# Userless default: the agent invents the prompt (conditioned on the catalog)
# ---------------------------------------------------------------------------------------------
def self_prompt(*, model: str = sg.DEFAULT_MODEL, verbose: bool = True) -> str:
    catalog = sg.load_catalog()
    system = load_prompt("self_prompt", catalog=scene_gen.catalog_lines(catalog))
    schema = {"type": "object", "properties": {"prompt": {"type": "string"}},
              "required": ["prompt"], "additionalProperties": False}
    for use_model in (model, sg.FALLBACK_MODEL):
        try:
            resp = sg._messages_request(system, [{"role": "user", "content":
                                                  "Invent one scene prompt."}], use_model, schema)
            prompt = json.loads(sg._response_text(resp))["prompt"]
            if verbose:
                print(f"[pipeline] self-generated prompt: {prompt!r}")
            return prompt
        except (urllib.error.HTTPError, RuntimeError):
            if use_model == sg.FALLBACK_MODEL:
                raise
    raise RuntimeError("unreachable")


def _goal_text(reuse: dict) -> str:
    g = reuse["avoid_goal"]
    return (f"{g['predicate']}({', '.join(str(v) for v in g.get('params', {}).values())})"
            if g else "(no task recorded)")


def _reuse_agent_call(name: str, ask: str, schema: dict, slots: dict, model: str) -> dict:
    """One structured request against a ``prompts/<name>.md`` template, with the model fallback."""
    system = load_prompt(name, **slots)
    for use_model in (model, sg.FALLBACK_MODEL):
        try:
            resp = sg._messages_request(system, [{"role": "user", "content": ask}],
                                        use_model, schema)
            return json.loads(sg._response_text(resp))
        except (urllib.error.HTTPError, RuntimeError):
            if use_model == sg.FALLBACK_MODEL:
                raise
    raise RuntimeError("unreachable")


def self_change_request(reuse: dict, *, model: str = sg.DEFAULT_MODEL,
                        verbose: bool = True) -> str:
    """Userless default for a REARRANGE: invent the change request a human would have given —
    a re-staging of the SAME objects, never a substitution (swaps happen only under the
    --substitute flag; see ``infer_substitutions``). Note the scene agent is already
    hard-instructed to rearrange meaningfully with or without this — the change request adds a
    specific, semantic twist rather than supplying the only pressure to differ."""
    out = _reuse_agent_call(
        "self_change", "Invent one change request.",
        {"type": "object", "properties": {"change": {"type": "string"}},
         "required": ["change"], "additionalProperties": False},
        dict(prompt=reuse["prompt"],
             layout=scene_gen.layout_lines(reuse["objects"]) or "  (no layout recorded)",
             goal=_goal_text(reuse)),
        model)
    if verbose:
        print(f"[pipeline] self-generated change request: {out['change']!r}")
    return out["change"]


def infer_substitutions(reuse: dict, *, model: str = sg.DEFAULT_MODEL,
                        verbose: bool = True) -> str:
    """--substitute given with NO phrase: the agent picks 1-{MAX_SUBSTITUTIONS} appropriate
    one-for-one swaps itself and writes the substitution phrase the human would have written.
    This inferred phrase runs under the STRICT regime (count invariant, composition limits) --
    unlike a user-written phrase, which is followed verbatim (substitutions_free)."""
    out = _reuse_agent_call(
        "self_substitute", "Invent the substitution(s).",
        # NOTE: the structured-output schema rejects maxItems on arrays (HTTP 400); the upper
        # bound lives in the prompt ($max_swaps) and the [:MAX_SUBSTITUTIONS] truncation below.
        {"type": "object",
         "properties": {"substitutions": {"type": "array", "items": {"type": "string"},
                                          "minItems": 1}},
         "required": ["substitutions"], "additionalProperties": False},
        dict(prompt=reuse["prompt"],
             layout=scene_gen.layout_lines(reuse["objects"]) or "  (no layout recorded)",
             goal=_goal_text(reuse),
             catalog=scene_gen.catalog_lines(sg.load_catalog()),
             min_swaps="1", max_swaps=str(MAX_SUBSTITUTIONS),
             cloth_max=str(scene_gen.CLOTH_SCENE_MAX),
             n_objects=str(sum(reuse["multiset"].values())),
             # the CENTRAL deformable limit, grandfathered against what the source already carries
             deformable_rule=sg.deformable_limit_text(
                 scene_gen.deformable_baseline_of(reuse["multiset"]))),
        model)
    text = "; ".join(out["substitutions"][:MAX_SUBSTITUTIONS])
    if verbose:
        print(f"[pipeline] self-generated substitution(s): {text!r}")
    return text


# ---------------------------------------------------------------------------------------------
# Rearrange (--scene_init): what the source run LOCKS
# ---------------------------------------------------------------------------------------------
def scene_init_reuse(scene_init: Path | str) -> dict:
    """Read a --scene_init source run and report everything it LOCKS: the object multiset (so the
    count is fixed), the robot placement, the environment (camera included) when the source has an
    ``env.json``, and the goal the new task must differ from.

    Single source of truth for "what is dead in rearrange mode": the interview and the CLI both
    consult this so a question/flag whose answer would be discarded is never asked or accepted."""
    src = Path(scene_init)
    src_dir = src if src.is_dir() else src.parent
    scene = json.loads((src_dir / "scene.json").read_text())
    env_json, task_json = src_dir / "env.json", src_dir / "task.json"
    return {
        "dir": src_dir,
        "prompt": scene.get("prompt", "(unknown)"),
        "multiset": scene_gen._multiset(scene.get("objects", [])),
        "objects": scene.get("objects", []),                 # the layout the rearrange must differ from
        "placement": build.placement_from_dir(src_dir),      # always fixed (default mount if absent)
        "env": json.loads(env_json.read_text()) if env_json.exists() else None,
        "avoid_goal": json.loads(task_json.read_text()).get("goal") if task_json.exists() else None,
    }


# ---------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------
def run_pipeline(prompt: str | None = None, *, placement_mode: str = "default",
                 fixed_placement: dict | None = None, camera_spec: str | None = None,
                 scene_init: Path | str | None = None, substitutions: str | None = None,
                 count_hint: int | None = None,
                 out_root: Path | str = OUTPUT_ROOT, name: str | None = None,
                 model: str = sg.DEFAULT_MODEL, device: str = "cuda:0", seed: int = 0,
                 render: bool = True, verify: bool = True, settle: bool = True,
                 verbose: bool = True) -> dict:
    """The full scene -> task -> env pipeline. Returns the manifest (also written as
    pipeline.json in the run folder).

    In ``scene_init`` (rearrange) mode the source run LOCKS the object multiset, the robot
    placement, and — when it has an ``env.json`` — the environment/camera, so ``count_hint``,
    ``placement_mode`` and (with a reused env) ``camera_spec`` are IGNORED; they are dropped here
    rather than silently overridden downstream.

    ``substitutions`` (rearrange only; the --substitute flag) is the ONE opt-in that may change the
    object set: ``None`` = off (the multiset is exact), a phrase = up to MAX_SUBSTITUTIONS
    one-for-one swaps as requested, ``""`` (flag given bare) = the agent infers 1-2 appropriate
    swaps and writes the phrase itself. Never an add or a remove in any case."""
    fixed_multiset = prev_objects = change_request = None
    init_env = avoid_goal = None
    allow_substitutions, substitutions_free = 0, False
    if scene_init is not None:
        reuse = scene_init_reuse(scene_init)
        fixed_multiset = reuse["multiset"]
        prev_objects = reuse["objects"]
        fixed_placement = reuse["placement"]
        placement_mode = "fixed"
        init_env = reuse["env"]
        avoid_goal = reuse["avoid_goal"]         # a DIFFERENT task is required in rearrange mode
        count_hint = None                        # the count IS the source multiset's
        if init_env is not None:
            camera_spec = None                   # env gen is skipped; the camera comes with the env
        # The SCENARIO does not change in rearrange mode: the source run's prompt is carried over
        # verbatim (so chained rearranges keep one stable scenario), and anything the user typed is
        # a CHANGE REQUEST for the new ARRANGEMENT, not a new prompt.
        change_request, prompt = prompt, reuse["prompt"]
        if verbose:
            print(f"[pipeline] scene_init from {reuse['dir']} (prompt reused: {prompt!r}; "
                  f"placement fixed: {fixed_placement['edge']!r} @ {fixed_placement['anchor']:.2f}; "
                  f"env {'reused' if init_env is not None else 'MISSING -> regenerated'})")
        if substitutions is not None:            # --substitute: the ONE opt-in that opens the catalog
            if substitutions.strip():
                # USER-written phrase (flag value or interview): followed VERBATIM — may add,
                # remove, swap any number, or exceed the composition limits. No enforcement.
                substitutions_free = True
            else:
                # Bare flag: the agent picks 1-MAX_SUBSTITUTIONS swaps itself, under the strict
                # regime (one-for-one, count invariant, composition limits enforced).
                substitutions = infer_substitutions(reuse, model=model, verbose=verbose)
                allow_substitutions = MAX_SUBSTITUTIONS
        if change_request is None:               # userless: invent the change request (never swaps)
            change_request = self_change_request(reuse, model=model, verbose=verbose)
        elif verbose:
            print(f"[pipeline] requested change: {change_request!r}")
        if substitutions and verbose:
            print(f"[pipeline] object-set changes "
                  f"{'requested by the USER (followed verbatim)' if substitutions_free else f'inferred (max {MAX_SUBSTITUTIONS} swaps)'}: "
                  f"{substitutions!r}")
    if prompt is None:
        prompt = self_prompt(model=model, verbose=verbose)

    # Stage 1 — scene gen (objects only) + settle check with one feedback retry.
    ref_placement = fixed_placement if placement_mode in ("default", "fixed") and fixed_placement \
        else geometry.default_placement()
    scene = scene_gen.call_scene_agent(prompt, placement=ref_placement, model=model, seed=seed,
                                       count_hint=count_hint, fixed_multiset=fixed_multiset,
                                       prev_objects=prev_objects, change_request=change_request,
                                       substitution_request=substitutions,
                                       allow_substitutions=allow_substitutions,
                                       substitutions_free=substitutions_free, verbose=verbose)
    run_dir, demo_py = build.write_run(scene, out_root, name=name)
    settle_report = None
    if settle:
        # RoboLab-parity settle loop: re-generate the layout on physics failure, up to
        # SETTLE_RETRIES times, feeding the settle diagnostics back to the scene agent each time.
        settle_report = scene_gen.settle_check(demo_py, device=device, verbose=verbose)
        for attempt in range(1, SETTLE_RETRIES + 1):
            if settle_report.get("ok"):
                break
            feedback = scene_gen.settle_feedback(settle_report)
            if verbose:
                print(f"[pipeline/scene] settle retry {attempt}/{SETTLE_RETRIES}: {feedback}")
            scene = scene_gen.call_scene_agent(
                f"{prompt}\n\nYour previous layout failed physics: {feedback}",
                placement=ref_placement, model=model, seed=seed + attempt, count_hint=count_hint,
                fixed_multiset=fixed_multiset, prev_objects=prev_objects,
                change_request=change_request, substitution_request=substitutions,
                allow_substitutions=allow_substitutions,
                substitutions_free=substitutions_free, verbose=verbose)
            scene["name"] = run_dir.name                      # keep the folder identity
            scene["prompt"] = prompt                          # the SCENARIO, not the retry message
            # (a rearrange of this run must inherit the scenario, not the settle-failure text)
            (run_dir / "scene.json").write_text(json.dumps(scene, indent=1))
            settle_report = scene_gen.settle_check(demo_py, device=device, verbose=verbose)
        if verbose and not settle_report.get("ok"):
            print(f"[pipeline/scene] settle check still failing after {SETTLE_RETRIES} retries; "
                  f"continuing with a warning")
        # Settled-pose write-back (RoboLab --replace): the stored scene now reflects where objects
        # actually came to rest. Always on — spawn poses are preserved under spawn_* on each object,
        # so this only ever adds information.
        n = scene_gen.write_back_settled_poses(scene, settle_report)
        (run_dir / "scene.json").write_text(json.dumps(scene, indent=1))
        if verbose and n:
            print(f"[pipeline/scene] wrote back {n} settled object pose(s) into scene.json")

    # Stage 2 — task gen (+ robot placement; scene_init requires a DIFFERENT task than the source).
    task, placement = task_gen.call_task_agent(scene, mode=placement_mode,
                                               placement=fixed_placement, avoid_goal=avoid_goal,
                                               model=model, verbose=verbose)
    (run_dir / "task.json").write_text(json.dumps(task_gen.task_dict(task, placement), indent=1))

    # Stage 3 — environment gen (skipped in scene_init mode: the environment is REUSED).
    if init_env is not None:
        env = dict(init_env)
        env["camera"] = geometry.default_exterior_camera(placement) \
            if env.get("camera", {}).get("source") == "default" else env.get("camera")
        env["reused_from"] = str(scene_init)
        if verbose:
            print(f"[pipeline/env] environment reused from {scene_init}")
    else:
        env = env_gen.call_env_agent(scene, task_gen.task_dict(task, placement), placement,
                                     camera_spec=camera_spec, model=model, verbose=verbose)
    (run_dir / "env.json").write_text(json.dumps(env, indent=1))

    metrics = scene_metrics.scene_metrics(scene)
    quality_notes = scene_metrics.quality_feedback(metrics)
    if verbose and quality_notes:
        print(f"[pipeline/scene] quality: {metrics} | {'; '.join(quality_notes)}")

    manifest = {
        "prompt": prompt, "change_request": change_request,
        "substitutions": substitutions or None,
        "substitutions_source": ("user" if substitutions_free else
                                 "agent" if substitutions else None),
        "mode": placement_mode, "scene_init": str(scene_init) if scene_init else None,
        "run_dir": str(run_dir),
        "robot_placement": placement,
        "settle": settle_report,
        "scene_metrics": metrics, "quality_notes": quality_notes,
        "feasibility": task.feasibility,
        "success_spec": task_gen.task_dict(task, placement)["success_spec"],
        "camera": env.get("camera"), "camera_report": env.get("camera_report"),
        "paths": {"scene": str(run_dir / "scene.json"), "task": str(run_dir / "task.json"),
                  "env": str(run_dir / "env.json"), "demo": str(demo_py)},
    }

    # Render (+ the default-on visual verification over BOTH stills).
    if render:
        cam_name = (env.get("camera") or {}).get("name", "overview_camera")
        overview = sg.render_scene(
            demo_py, device=device, camera=cam_name,
            out_png=run_dir / "scene_overview.png",
            extra_cameras={"wrist_camera": run_dir / "scene_wrist.png"}, verbose=verbose)
        manifest["paths"]["overview"] = str(overview)
        manifest["paths"]["wrist"] = str(run_dir / "scene_wrist.png")
        if verify:
            verdict = sg.verify_scene_render(scene, overview, run_dir / "scene_wrist.png", model=model)
            manifest["verification"] = {"ok": bool(verdict.get("ok")),
                                        "issues": verdict.get("issues", [])}
            if verbose:
                print(f"[pipeline] verification: ok={verdict.get('ok')} "
                      f"issues={verdict.get('issues', [])}")

    (run_dir / "pipeline.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------------------------
# --user interview (the RoboLab-skill interaction, on stdin)
# ---------------------------------------------------------------------------------------------
def _ask(q: str, default: str = "") -> str:
    tail = f" [{default}]" if default else ""
    ans = input(f"{q}{tail}: ").strip()
    return ans or default


def interview(scene_init: Path | str | None = None) -> dict:
    """The stdin interview. Returns ONLY the questions that were actually asked — in rearrange
    mode the object count, the robot placement, and (when the source carries an env.json) the
    camera are inherited from the source run, so asking would collect an answer the pipeline
    discards. The caller applies exactly the keys it gets back.

    In rearrange mode the ``prompt`` key is NOT a scene description but an optional CHANGE REQUEST
    for the new arrangement — the scenario prompt is the source run's, verbatim."""
    answers: dict = {}
    reuse = scene_init_reuse(scene_init) if scene_init is not None else None
    n = 0

    def step(q: str, default: str = "") -> str:
        nonlocal n
        n += 1
        return _ask(f"{n}. {q}", default)

    if reuse is not None:
        listing = ", ".join(f"{k} x{v}" for k, v in sorted(reuse["multiset"].items()))
        p, g = reuse["placement"], reuse["avoid_goal"]
        goal_txt = (f"{g['predicate']}({', '.join(str(v) for v in g.get('params', {}).values())})"
                    if g else "no task recorded")
        print(f"Rearrange mode — reusing {reuse['dir']}.\n"
              f"  scenario (unchanged): {reuse['prompt']!r}\n"
              f"  objects (fixed): {listing}\n"
              f"  robot placement (fixed): {p['edge']} edge @ anchor {p['anchor']:.2f}\n"
              f"  environment: {'reused (background/table/lighting/camera)' if reuse['env'] else 'MISSING in the source — will be generated'}\n"
              f"I'll RE-ARRANGE these objects — new positions/orientations/stacks — so a task "
              f"different from the source's ({goal_txt}) becomes the natural one.\n"
              f"(The object set is fixed unless you ask for a substitution below.)\n")
        answers["prompt"] = step("Any specific change to the arrangement? (blank = I'll invent "
                                 "one; the scenario itself stays the same)") or None
        sub = step("Any object-set changes? (followed exactly as you say -- swaps, adds, or "
                   f"removals; 'yes' = I'll pick 1-{MAX_SUBSTITUTIONS} swaps myself; blank = none)")
        answers["substitutions"] = None if not sub else ("" if sub.lower() in ("y", "yes") else sub)
    else:
        catalog = sg.load_catalog()
        names = ", ".join(o["name"] for o in catalog["objects"])
        print("I'll generate a scene, a task (with robot placement), and an environment.\n"
              f"Objects I can pick from ({len(catalog['objects'])}): {names}\n")
        q = "Scene description — what should be on the table?"
        answers["prompt"] = step(q)
        while not answers["prompt"]:                         # required — re-ask under the same number
            answers["prompt"] = _ask(f"{n}. {q}")
        count = step("Number of objects (blank = 3-5 for simple scenes)")
        answers["count_hint"] = int(count) if count.isdigit() else None
        place = step("Robot placement — 'default' (middle of the back long edge) or 'task' "
                     "(chosen per task)", "task")
        answers["placement_mode"] = "task" if place.lower().startswith("t") else "default"

    if reuse is None or reuse["env"] is None:
        camera = step("Exterior camera — describe it, or blank for the default front bird's-eye view")
        answers["camera_spec"] = camera or None
    answers["verify"] = step("Post-render visual verification? (Y/n)", "y").lower().startswith("y")
    answers["out_root"] = step("Output directory", str(OUTPUT_ROOT))
    return answers


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=None,
                    help="scene description (blank + no --user = the agent invents one)")
    ap.add_argument("--user", action="store_true", help="interactive RoboLab-style interview")
    ap.add_argument("--scene_init", default=None,
                    help="existing run dir (or scene.json) whose prompt/assets/placement/env to "
                         "reuse; the positional argument becomes a CHANGE REQUEST for the new "
                         "arrangement (blank = the agent invents one)")
    ap.add_argument("--substitute", nargs="?", const="", default=None,
                    help=f"--scene_init only: change the object set. With a phrase (\"a mug "
                         f"instead of the bowl\") it is followed VERBATIM -- adds, removes, any "
                         f"number of swaps, even past the usual composition limits. Passed bare, "
                         f"the agent picks 1-{MAX_SUBSTITUTIONS} strict one-for-one swap(s) "
                         f"itself. Without the flag the object set is fixed.")
    ap.add_argument("--placement", choices=["default", "task"], default=None,
                    help="robot placement mode (userless default: 'default')")
    ap.add_argument("--camera", default=None, help="exterior-camera specification text")
    ap.add_argument("--count", type=int, default=None, help="approximate object count")
    ap.add_argument("--name", default=None, help="run folder name (default: the scene slug)")
    ap.add_argument("--out-dir", default=str(OUTPUT_ROOT))
    ap.add_argument("--model", default=sg.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    # --- toggles for the HEAVYWEIGHT (time-intensive) stages; all default ON, flags turn them off ---
    # These are the slow parts (each adds seconds-to-minutes): they run by default (RoboLab parity)
    # and can be skipped for a fast iteration. Non-heavyweight checks (grammar, spatial solver, OBB,
    # feasibility, quality metrics, success-spec compile) are cheap and always run.
    ap.add_argument("--no-render", action="store_true",
                    help="[heavyweight] skip the final PBR render of the settled scene")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="[heavyweight] skip the post-render visual verification (agent inspects "
                         "the image; extra model call + possible re-render). On by default.")
    ap.add_argument("--skip-settle-check", dest="settle", action="store_false",
                    help="[heavyweight] skip the headless physics settle check + its retries")
    ap.set_defaults(verify=True, settle=True)
    args = ap.parse_args()

    if args.scene_init:                    # reject flags the source run overrides (see scene_init_reuse)
        dead = [f for f, v in (("--count", args.count), ("--placement", args.placement))
                if v is not None]
        if args.camera and scene_init_reuse(args.scene_init)["env"] is not None:
            dead.append("--camera")
        if dead:
            ap.error(f"{', '.join(dead)} cannot be used with --scene_init: rearrange mode inherits "
                     f"the object multiset, robot placement and environment (camera included) from "
                     f"the source run")
    elif args.substitute is not None:      # nothing to substitute INTO without a source scene
        ap.error("--substitute only applies with --scene_init (without it the scene agent chooses "
                 "the objects freely)")

    kw = dict(placement_mode=args.placement or "default", camera_spec=args.camera,
              count_hint=args.count, out_root=args.out_dir, verify=args.verify,
              substitutions=args.substitute)
    if args.user:
        answers = interview(args.scene_init)                 # only the LIVE questions come back
        args.prompt = answers.pop("prompt")
        kw["out_root"] = answers.pop("out_root")
        kw["verify"] = args.verify and answers.pop("verify")
        for key, cli in (("placement_mode", args.placement), ("camera_spec", args.camera),
                         ("count_hint", args.count), ("substitutions", args.substitute)):
            if key in answers and cli is None:               # an explicit CLI flag still wins
                kw[key] = answers[key]

    manifest = run_pipeline(
        args.prompt, scene_init=args.scene_init, name=args.name, model=args.model,
        device=args.device, seed=args.seed, render=not args.no_render,
        settle=args.settle, **kw)
    print(json.dumps({k: manifest[k] for k in ("run_dir", "robot_placement", "camera_report",
                                               "paths") if manifest.get(k) is not None}, indent=1))
