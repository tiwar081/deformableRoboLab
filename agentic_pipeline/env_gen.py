"""Pipeline stage 3 — environment gen: everything scene/task gen did not decide about the LOOK.

Owns: background HDRI, workspace-table material, lighting (dome intensity + key sphere light), and
cameras. Camera policy (per the pipeline spec): the WRIST camera is always mounted on the robot
and is not configurable; the EXTERIOR camera follows the user's specification when one is given,
otherwise the pipeline REPORTS that no specification was given and uses the default front
bird's-eye view — opposite side of the workspace table from the robot, ~2 m above the tabletop,
looking down at the table center so the full workspace is visible
(``geometry.default_exterior_camera``).

The agent prompt template lives in ``prompts/env_system.md``. ``env_to_render_spec`` converts the
stage's ``env.json`` artifact into the renderer's ``RenderSpec`` (world -> viz-frame camera
conversion happens here: the viz frame is the robot base frame, translation-only).
"""
from __future__ import annotations

import json
import math
import urllib.error

from . import scene_generator as sg
from deformableManipulationTools.params import TABLE
from . import geometry, load_prompt

DOME_RANGE = (200.0, 1500.0)
SPHERE_RANGE = (1000.0, 10000.0)
SPHERE_HEIGHT_RANGE = (0.4, 1.5)
CAM_DIST_RANGE = (0.5, 3.5)


def _env_schema() -> dict:
    xyz = {"type": "array", "items": {"type": "number"},
           "description": "[x, y, z] world metres (exactly 3 numbers)"}
    return {
        "type": "object",
        "properties": {
            "background": {"type": "string", "enum": sg.available_background_names()},
            "table": {"type": "string", "enum": list(sg.TABLE_KEYS)},
            "dome_intensity": {"type": "number"},
            "sphere_light": {
                "type": "object",
                "properties": {"intensity": {"type": "number"}, "height": {"type": "number"}},
                "required": ["intensity", "height"],
                "additionalProperties": False,
            },
            "camera": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"position": xyz, "target": xyz},
                        "required": ["position", "target"],
                        "additionalProperties": False,
                    },
                ],
            },
        },
        "required": ["background", "table", "dome_intensity", "sphere_light", "camera"],
        "additionalProperties": False,
    }


def _validate_camera(cam: dict) -> list[str]:
    errs = []
    for k in ("position", "target"):
        if len(cam.get(k, [])) != 3:
            errs.append(f"camera.{k} must be [x, y, z]")
    if errs:
        return errs
    px, py, pz = cam["position"]
    d = math.hypot(px - TABLE.pos[0], py - TABLE.pos[1])
    if pz <= TABLE.top_z + 0.05:
        errs.append(f"camera height z={pz:.2f} must be above the tabletop ({TABLE.top_z:.2f})")
    if not (CAM_DIST_RANGE[0] <= d <= CAM_DIST_RANGE[1]):
        errs.append(f"camera is {d:.2f} m from the table center; keep it within "
                    f"{CAM_DIST_RANGE[0]:.1f}-{CAM_DIST_RANGE[1]:.1f} m")
    return errs


def call_env_agent(scene: dict, task: dict | None, placement: dict, *,
                   camera_spec: str | None = None, model: str = sg.DEFAULT_MODEL,
                   attempts: int = 2, verbose: bool = True) -> dict:
    """(scene, task, robot placement, optional user camera spec) -> the env dict (env.json).
    Reports the camera default when the user gave no specification (the pipeline requirement)."""
    x0, x1, y0, y1 = sg.workspace_bounds(margin=0.0)
    system = load_prompt(
        "env_system",
        backgrounds=", ".join(sg.available_background_names()),
        tables=", ".join(sg.TABLE_KEYS),
        x0=f"{x0:.2f}", x1=f"{x1:.2f}", y0=f"{y0:.2f}", y1=f"{y1:.2f}", tz=f"{TABLE.top_z:.2f}",
        robot_edge=placement["edge"],
        scene_summary=f"{scene.get('name', '')} — {scene.get('description', '')} "
                      f"(objects: {', '.join(o['name'] for o in scene.get('objects', []))})",
        task_summary=(task or {}).get("instruction", {}).get("default", "(no task)"),
        camera_spec=camera_spec or "(none given)",
    )
    schema = _env_schema()
    messages = [{"role": "user", "content": "Decide the environment JSON."}]
    env, use_model = None, model
    for attempt in range(attempts):
        try:
            resp = sg._messages_request(system, messages, use_model, schema)
            text = sg._response_text(resp)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if use_model != sg.FALLBACK_MODEL:
                if verbose:
                    print(f"[pipeline/env] {use_model} failed ({exc}); falling back to {sg.FALLBACK_MODEL}")
                use_model = sg.FALLBACK_MODEL
                continue
            raise
        env = json.loads(text)
        errs = []
        if camera_spec and env.get("camera") is None:
            errs.append("the user DID give a camera specification — express it as position/target")
        if env.get("camera"):
            errs.extend(_validate_camera(env["camera"]))
        if not errs:
            break
        if verbose:
            print(f"[pipeline/env] attempt {attempt + 1} rejected: {'; '.join(errs)}")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Rejected: {'; '.join(errs)}. Return a corrected "
                                                 f"environment JSON."}]
    if env is None:
        raise RuntimeError("env agent produced no usable environment")

    env["dome_intensity"] = float(min(max(env.get("dome_intensity", 500.0), DOME_RANGE[0]), DOME_RANGE[1]))
    sl = env.get("sphere_light") or {}
    env["sphere_light"] = {
        "intensity": float(min(max(sl.get("intensity", 5000.0), SPHERE_RANGE[0]), SPHERE_RANGE[1])),
        "height": float(min(max(sl.get("height", 0.65), SPHERE_HEIGHT_RANGE[0]), SPHERE_HEIGHT_RANGE[1])),
    }
    if env.get("camera"):
        env["camera"] = {"name": "overview_camera", "position": env["camera"]["position"],
                         "target": env["camera"]["target"], "source": "user"}
    else:
        env["camera"] = geometry.default_exterior_camera(placement)
        env["camera_report"] = ("no user camera specification — using the default front bird's-eye "
                                "view (opposite the robot, ~2 m above the tabletop, whole workspace "
                                "in frame)")
        if verbose:
            print(f"[pipeline/env] {env['camera_report']}")
    env["model"] = use_model
    return env


def default_env(placement: dict) -> dict:
    """The no-agent environment (settle checks, userless fallbacks): centralized-default look +
    the default camera for this placement."""
    return {"background": None, "table": None, "dome_intensity": 500.0,
            "sphere_light": {"intensity": 5000.0, "height": 0.65},
            "camera": geometry.default_exterior_camera(placement),
            "camera_report": "default environment (no env-gen run)"}


def env_to_render_spec(env: dict, placement: dict):
    """env.json + robot placement -> robolabViz.RenderSpec. Cameras/lights are authored in WORLD
    coordinates; the viz frame is the robot base frame (translation only), so world -> viz is a
    subtraction of the base position."""
    from robolabViz import CameraConfig, RenderSpec, SphereLightConfig
    from robolabViz.config import look_at_quat_wxyz

    bx, by, bz = placement["base"]
    cam = env["camera"]
    eye_w, tgt_w = cam["position"], cam["target"]
    eye_viz = (eye_w[0] - bx, eye_w[1] - by, eye_w[2] - bz)
    quat = look_at_quat_wxyz(eye_w, tgt_w, (0.0, 0.0, 1.0))   # rotation is translation-invariant
    camera = CameraConfig(name=cam.get("name", "overview_camera"),
                          position=eye_viz, orientation_wxyz=quat,
                          focal_length=float(cam.get("focal_length") or 4.0))
    sl = env.get("sphere_light", {})
    light_world = (TABLE.pos[0], TABLE.pos[1], TABLE.top_z + float(sl.get("height", 0.65)))
    sphere = SphereLightConfig(
        name="sphere_light",
        position=(light_world[0] - bx, light_world[1] - by, light_world[2] - bz),
        intensity=float(sl.get("intensity", 5000.0)), radius=0.5)
    return RenderSpec(
        background=env.get("background"),
        table=env.get("table"),
        dome_intensity=float(env.get("dome_intensity", 500.0)),
        sphere_lights=[sphere],
        exterior_cameras=[camera],
        preview_cameras=[camera.name, "wrist_camera"],
    )
