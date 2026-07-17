"""agent_pipeline.py — run the full agentic pipeline: scene gen -> task gen -> environment gen.

Three modes (see ``agentic_pipeline/`` for the stages and ``agentic_pipeline/SKILL.md`` for the
interactive Claude Code session that mirrors RoboLab's skill):

  --user                 RoboLab-style interview on stdin: scene description, object count, robot
                         placement (default mount vs task-specific), camera specification, optional
                         visual verification, output directory. Everything else runs the same
                         pipeline.
  (no flags)             Userless defaults: the agent INVENTS the prompt (conditioned on the object
                         catalog), default robot placement (middle of the back LONG edge), default
                         camera (front bird's-eye opposite the robot), no visual verification.
  --scene_init PATH      Rearrange mode: reuse a previous run's ASSETS (exact object multiset),
                         robot placement, and environment; generate NEW object placements and a
                         DIFFERENT task.

Artifacts land in ``outputs/agenticPipeline/<name>/``: scene.json, task.json, env.json, the
generated demo data file, pipeline.json (manifest), and the rendered stills
(scene_overview.png + scene_wrist.png). Post-render visual verification is OPT-IN (--verify).
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


# ---------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------
def run_pipeline(prompt: str | None = None, *, placement_mode: str = "default",
                 fixed_placement: dict | None = None, camera_spec: str | None = None,
                 scene_init: Path | str | None = None, count_hint: int | None = None,
                 out_root: Path | str = OUTPUT_ROOT, name: str | None = None,
                 model: str = sg.DEFAULT_MODEL, device: str = "cuda:0", seed: int = 0,
                 render: bool = True, verify: bool = True, settle: bool = True,
                 settle_writeback: bool = True, success_predicate: bool = True,
                 verbose: bool = True) -> dict:
    """The full scene -> task -> env pipeline. Returns the manifest (also written as
    pipeline.json in the run folder)."""
    fixed_multiset = None
    init_env = avoid_goal = None
    if scene_init is not None:
        src = Path(scene_init)
        src_dir = src if src.is_dir() else src.parent
        src_scene = json.loads((src_dir / "scene.json").read_text())
        fixed_multiset = scene_gen._multiset(src_scene.get("objects", []))
        fixed_placement = build.placement_from_dir(src_dir)
        placement_mode = "fixed"
        env_json = src_dir / "env.json"
        init_env = json.loads(env_json.read_text()) if env_json.exists() else None
        src_task = src_dir / "task.json"
        if src_task.exists():                    # a DIFFERENT task is required in rearrange mode
            avoid_goal = json.loads(src_task.read_text()).get("goal")
        if prompt is None:
            prompt = (f"Rearrange these objects into a fresh, differently-composed tabletop scene: "
                      f"{', '.join(f'{n} x{c}' for n, c in sorted(fixed_multiset.items()))}. "
                      f"Original scenario for reference: {src_scene.get('prompt', '(unknown)')}")
        if verbose:
            print(f"[pipeline] scene_init from {src_dir} (placement fixed: "
                  f"{fixed_placement['edge']!r} @ {fixed_placement['anchor']:.2f}; env reused)")
    if prompt is None:
        prompt = self_prompt(model=model, verbose=verbose)

    # Stage 1 — scene gen (objects only) + settle check with one feedback retry.
    ref_placement = fixed_placement if placement_mode in ("default", "fixed") and fixed_placement \
        else geometry.default_placement()
    scene = scene_gen.call_scene_agent(prompt, placement=ref_placement, model=model, seed=seed,
                                       count_hint=count_hint, fixed_multiset=fixed_multiset,
                                       verbose=verbose)
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
                fixed_multiset=fixed_multiset, verbose=verbose)
            scene["name"] = run_dir.name                      # keep the folder identity
            (run_dir / "scene.json").write_text(json.dumps(scene, indent=1))
            settle_report = scene_gen.settle_check(demo_py, device=device, verbose=verbose)
        if verbose and not settle_report.get("ok"):
            print(f"[pipeline/scene] settle check still failing after {SETTLE_RETRIES} retries; "
                  f"continuing with a warning")
        # Settled-pose write-back (RoboLab --replace): the stored scene now reflects where objects
        # actually came to rest. Enabled by default; --no-settle-writeback keeps the spawn poses.
        if settle_writeback:
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
        "prompt": prompt, "mode": placement_mode, "scene_init": str(scene_init) if scene_init else None,
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

    # Render (+ OPT-IN visual verification).
    if render:
        cam_name = (env.get("camera") or {}).get("name", "overview_camera")
        overview = sg.render_scene(
            demo_py, device=device, camera=cam_name,
            out_png=run_dir / "scene_overview.png",
            extra_cameras={"wrist_camera": run_dir / "scene_wrist.png"}, verbose=verbose)
        manifest["paths"]["overview"] = str(overview)
        manifest["paths"]["wrist"] = str(run_dir / "scene_wrist.png")
        if verify:
            verdict = sg.verify_scene_render(scene, overview, model=model)
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


def interview() -> dict:
    catalog = sg.load_catalog()
    names = ", ".join(o["name"] for o in catalog["objects"])
    print("I'll generate a scene, a task (with robot placement), and an environment.\n"
          f"Objects I can pick from ({len(catalog['objects'])}): {names}\n")
    prompt = ""
    while not prompt:
        prompt = _ask("1. Scene description — what should be on the table?")
    count = _ask("2. Number of objects (blank = 3-5 for simple scenes)")
    place = _ask("3. Robot placement — 'default' (middle of the back long edge) or 'task' "
                 "(chosen per task)", "task")
    camera = _ask("4. Exterior camera — describe it, or blank for the default front bird's-eye view")
    verify = _ask("5. Post-render visual verification? (Y/n)", "y").lower().startswith("y")
    out = _ask("6. Output directory", str(OUTPUT_ROOT))
    return {"prompt": prompt, "count_hint": int(count) if count.isdigit() else None,
            "placement_mode": "task" if place.lower().startswith("t") else "default",
            "camera_spec": camera or None, "verify": verify, "out_root": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=None,
                    help="scene description (blank + no --user = the agent invents one)")
    ap.add_argument("--user", action="store_true", help="interactive RoboLab-style interview")
    ap.add_argument("--scene_init", default=None,
                    help="existing run dir (or scene.json) whose assets/placement/env to reuse")
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
    ap.add_argument("--no-settle-writeback", dest="settle_writeback", action="store_false",
                    help="keep spawn poses instead of writing settled poses back into scene.json")
    ap.set_defaults(verify=True, settle=True, settle_writeback=True)
    args = ap.parse_args()

    kw = dict(placement_mode=args.placement or "default", camera_spec=args.camera,
              count_hint=args.count, out_root=args.out_dir, verify=args.verify,
              settle_writeback=args.settle_writeback)
    if args.user:
        answers = interview()
        kw.update(placement_mode=(args.placement or answers["placement_mode"]),
                  camera_spec=args.camera or answers["camera_spec"],
                  count_hint=args.count or answers["count_hint"],
                  out_root=answers["out_root"], verify=args.verify and answers["verify"])
        args.prompt = answers["prompt"]

    manifest = run_pipeline(
        args.prompt, scene_init=args.scene_init, name=args.name, model=args.model,
        device=args.device, seed=args.seed, render=not args.no_render,
        settle=args.settle, **kw)
    print(json.dumps({k: manifest[k] for k in ("run_dir", "robot_placement", "camera_report",
                                               "paths") if manifest.get(k) is not None}, indent=1))
