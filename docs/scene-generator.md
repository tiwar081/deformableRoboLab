# Agentic scene generator (`agentic_pipeline/scene_generator.py`)

RoboLab-style "scene gen" for this repo, with deformables as first-class scene objects (the exact
gap RoboLab's paper names as its limitation). An agent (Claude, default `claude-fable-5`, fallback
`claude-opus-4-8`) reads a natural-language prompt and composes a tabletop scene from three sets:

- **objects** — `assets/objects/scene_catalog.json` (see below),
- **backgrounds** — the vendored HDRI set (`robolabViz.config.available_backgrounds()`),
- **table materials** — `maple / oak / bamboo / black` (`robolabViz.config.TABLE_TEXTURES`).

Pipeline (mirrors RoboLab's validate-and-refine scene_gen, `docs/robolab.md`):
LLM (structured output, JSON schema) → grammar validation (`validate_scene`) → spatial solver
(`resolve_placements`: circle-collision push-apart + table-bounds clamp; cables are laid first as
fixed obstacle node-chains) → on failure, natural-language feedback is appended and the agent
retries (≤3 attempts). The accepted scene dict is converted by `demo_spec_from_scene` into a
standard settle-only `DemoSpec` (arm parked at one home waypoint, fingers open) so the ENTIRE
existing runner is reused: solver routing (MuJoCo rigid-only vs split VBD + proxies), centralized
materials/coupling, and the RoboLab-look renderer.

**Relations** (RoboLab predicate analog; optional per object, `"relation": {type, target, distance}`,
`target` = index into the objects array): `left-of` / `right-of` / `in-front-of` / `behind`
reposition the subject beside its target — direction words are pinned in the system prompt
(LEFT = +x, RIGHT = −x, FARTHER = −y, NEARER = +y, as the robot/camera sees the table); `on` stacks
the subject on a rigid target, `in` drops it into an open-top container (bowl/mug/pitcher) — both
pin the subject's x/y to the target and spawn it just above the target's catalog height
(`Obj.rest_on_z` accepts an explicit float for this), so GRAVITY forms the actual stack/containment
during settle, per the no-scripted-motion physics rule. Validation rejects cables/garments as stack
subjects, non-rigid stack targets, oversize stacking/containment, and relation cycles.

**Post-render verification** (`verify_scene_render`, on by default; `--no-verify` skips): the agent
is shown the settled over-the-shoulder still of its own scene and returns `{ok, issues, revised}` —
the agent analog of RoboLab's post-settle screenshot sanity check. On `ok=false` with a usable
`revised` scene, the revision is re-validated/re-solved, written IN PLACE (original kept as
`scene_initial.json`), re-rendered, and re-verified ONCE; verdicts land in the scene dict as
`verification` / `verification_revised`.

**No overwrites**: `write_scene` never reuses an existing default folder — `outputs/sceneGen/<name>`
becomes `<name>_2`, `<name>_3`, … with `scene["name"]` renamed to match (an explicit `out_dir` is
honored as-is; that is the revise-in-place path).

Run standalone → one settled over-the-shoulder still:

    .venv/bin/python -m agentic_pipeline.scene_generator "a cluttered breakfast table with a sponge and two cans"
    # outputs/sceneGen/<name>/{scene.json, scene_<name>.py, scene_over_shoulder.png}

Pipeline helpers: `call_scene_agent`, `resolve_placements`, `demo_spec_from_scene`, `write_scene`,
`render_scene`, `generate_scene` (end-to-end). Credentials: `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`, else the Claude Code OAuth token (`~/.claude/.credentials.json`); requests
go over raw HTTPS (the Anthropic SDK conflicts with the venv's isaacsim `typing_extensions` pin).

## Scene-gen vs environment-gen boundary (RoboLab comparison)

In RoboLab, scene gen bakes ONLY objects + table (incl. table material) into a robot-agnostic
`.usda`; **backgrounds, lighting, cameras, and robot placement are the environment/variations
layer** (merged config classes at env-registration time). Here, the scene ALSO picks the
background (per request); **lighting and robot placement stay at the centralized defaults**
(`robolabViz.droid_scene_config` dome + sphere light; robot base from the framework) — they are
environment-gen concerns to be added later, and `RenderSpec` already exposes the hooks
(`dome_intensity`, `sphere_lights`, cameras).

## The catalog (`assets/objects/scene_catalog.json`)

Every entry is an **imported pre-existing object**: `kind` picks the centralized builder,
`config` holds the per-object physical parameters, `dims` drives the placement solver, `source`
is provenance, and `inferred` lists every parameter NOT shipped by the source dataset with how it
was derived. Constraints the generator enforces (solver reality): ≤1 cloth item, ≤1 squishy item,
never cloth+squishy together (`model.soft_contact_*` is per-model — ONE particle-deformable type
per scene), ≤1 cable.

| group | objects | source |
|---|---|---|
| rigid (`ycb_mesh` etc.) | banana, bowl, mug, cheez_it, tomato_soup_can, mustard, sugar_box, tuna_can, spam_can, pitcher, wood_block, apple, rubiks_cube, steel_cube | RoboLab's vendored YCB/HOT3D/Objaverse USDs (official YCB masses; realistic mu — RoboLab's catalog mu=2.0 is a documented grasp cheat) |
| cloth | gray_tshirt (proven Newton shirt), green_tshirt, blue_dress | RGBench (`hwk0809/RGBench-Cloth-Sim2Real-v1`, CC-BY 4.0) garment meshes + MEASURED fabric params (area density, friction); SimWeaver (the requested source) is unreleased — RGBench is its physics source, swap in SimWeaver garments+bags when the repo goes live |
| cable | power_cable (canonical CABLE), nylon_rope | rope variant parameterized from YCB object-67 rope PHYSICAL specs (no scan exists); capsule-rod is NVIDIA's own recommended rope representation |
| squishy (`soft_mesh` FEM tets) | sponge, foam_brick, banana_soft, raspberry_cube | YCB scans tetrahedralized offline (`assets/objects/_utils/make_tet.py`, DefGraspSim `.tet` convention; DefGraspSim's own mesh set is Google-Drive-gated). banana_soft carries DefGraspSim's official FEM params (rho=1000, nu=0.3, mu=0.7, E=2e5) |

New squishy imports: run `make_tet.py` (offline; trimesh+pymeshfix+tetgen — asset-prep deps only)
and add a `soft_mesh` entry; the framework auto-detects tets and applies the FEM particle-solver
config + the entry's `soft_contact_*` centrally (`SoftMeshConfig` mirrors `SoftBlockConfig`).
