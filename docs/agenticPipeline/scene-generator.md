# Agentic scene generator (`agentic_pipeline/scene_generator.py`)

RoboLab-style "scene gen" for this repo, with deformables as first-class scene objects (the exact
gap RoboLab's paper names as its limitation). An agent (Claude, default `claude-fable-5`, fallback
`claude-opus-4-8`) reads a natural-language prompt and composes a tabletop scene from three sets:

- **objects** — `assets/objects/scene_catalog.json` (see below),
- **backgrounds** — the vendored HDRI set (`robolabViz.config.available_backgrounds()`),
- **table materials** — `maple / oak / bamboo / black` (`robolabViz.config.TABLE_TEXTURES`).

Pipeline (mirrors RoboLab's validate-and-refine scene_gen, `docs/external/robolab.md`):
LLM (structured output, JSON schema) → grammar validation (`validate_scene`) → spatial solver
(`resolve_placements`: push-apart with an **OBB/SAT narrow phase** behind a circle broad-phase
prefilter (`packing.obb_penetration`), so an elongated yawed object conflicts along its true
footprint; + table-bounds clamp; cables are laid first as fixed obstacle node-chains; a
`tall_keepout` disc under the parked gripper's home TCP repels objects taller than 0.14 m, which
would otherwise start the sim in collision with the parked fingers) → on failure, natural-language
feedback is appended and the agent
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
layer** (merged config classes at env-registration time).

This standalone generator keeps the RoboLab split loosely: the scene also picks the background, and
lighting + robot placement stay at the centralized defaults (`robolabViz.droid_scene_config` dome +
sphere light; robot base from the framework). **In the three-stage pipeline those are no longer
defaults**: robot placement is decided by task gen (edge + anchor, reach-checked) and
background/table/lighting/cameras by env gen — see
[agentic-pipeline.md](agentic-pipeline.md). `RenderSpec` is the shared surface either way
(`dome_intensity`, `sphere_lights`, cameras).

## The catalog (`assets/objects/scene_catalog.json`)

Every entry is an **imported pre-existing object**: `kind` picks the centralized builder,
`config` holds the per-object physical parameters, `dims` drives the placement solver, `source`
is provenance, and `inferred` lists every parameter NOT shipped by the source dataset with how it
was derived. **Current particle-deformable limitation:** a scene may contain at most **one item
total** from the union of bag, cloth garment, and squishy FEM objects. Thus two bags, bag+garment,
bag+squishy, two garments, and two squishies are all rejected by `validate_scene` (`model.soft_contact_*`
is per-model and the pipeline currently owns one particle-deformable config per scene). A cable may
coexist with that item, but there is still at most one cable.

| group | objects | source |
|---|---|---|
| rigid (`ycb_mesh` etc.) | banana, bowl, mug, cheez_it, tomato_soup_can, mustard, sugar_box, tuna_can, spam_can, pitcher, wood_block, apple, rubiks_cube, steel_cube | RoboLab's vendored YCB/HOT3D/Objaverse USDs (official YCB masses; realistic mu — RoboLab's catalog mu=2.0 is a documented grasp cheat) |
| cloth | gray_tshirt (proven Newton shirt), green_tshirt, blue_dress | RGBench (`hwk0809/RGBench-Cloth-Sim2Real-v1`, CC-BY 4.0) garment meshes + measured fabric params (area density, friction); SimWeaver remains unreleased |
| bag (`kind: cloth`, `category: bag`) | plastic_recycle_bag, woven_tote_bag, paper_grocery_bag | Objaverse 1.0 CC-BY geometry, UID/author/license recorded in `assets/objects/objaverse_bags/README.md`; generic isotropic mesh→USDA conversion via `_utils/convert_bag_mesh.py`; physical values are documented inferences because Objaverse supplies no measured material parameters |
| cable | power_cable (canonical CABLE), nylon_rope | rope variant parameterized from YCB object-67 rope PHYSICAL specs (no scan exists); capsule-rod is NVIDIA's own recommended rope representation |
| squishy (`soft_mesh` FEM tets) | sponge, foam_brick, banana_soft, raspberry_cube | YCB scans tetrahedralized offline (`assets/objects/_utils/make_tet.py`, DefGraspSim `.tet` convention; DefGraspSim's own mesh set is Google-Drive-gated). banana_soft carries DefGraspSim's official FEM params (rho=1000, nu=0.3, mu=0.7, E=2e5) |

Containers were extended with RoboLab's VoMP tool bins (`assets/objects/vomp/`): `parts_bin`
(16 cm), `tool_bin` (23 cm), `long_tray_bin` (30 cm), `bucket` (27 cm utility bucket) — all
non-articulated, coacd-decomposed so the cavity holds objects, and all settle flat. Containers are
recognized from the catalog's `container: true` flag (bowl/mug/pitcher carry it too), so a newly
ingested container registers automatically.

**Importing an object — the two rules that bit:**

- **A USD asset may be MULTI-mesh, and taking the first prim is wrong.** `mesh_collision.load_usd_mesh`
  MERGES every `UsdGeom.Mesh` prim under the stage default prim, baking each prim's local-to-world
  xform into its vertices; `viz_assets._load_visual` picks the largest-face prim (the one carrying
  the texture). Both were first-prim-only, and the utility bucket's USD leads with a flat 141-vertex
  decal sticker — so collision *and* viz used a flat panel (it rendered as loose sheets and sank
  ~5 cm through the table). Scoping the merge to the default prim's subtree is deliberate: it
  excludes stray siblings like the objaverse apple's `/GroundPlane/CollisionMesh`, which would
  otherwise blow the AABB to ~0.9 m. Baking the xform is also required — the apple ships a
  0.01-scale `xformOp`, so its raw points are 100× too big (the first time it was actually placed in
  a scene it threw the robot). Single-prim legacy assets are byte-identical (same coacd cache key).
- **New squishy imports run `make_tet.py` offline** (fTetWild via the `wildmeshing` bindings —
  asset-prep dependency only; **not** tetgen, see [deformables.md](../physicsEngine/deformables.md)) and add a
  `soft_mesh` entry whose `soft_contact_*` is re-derived at that mesh's own per-particle mass. The
  framework auto-detects tets and applies the FEM particle-solver config centrally
  (`SoftMeshConfig` mirrors `SoftBlockConfig`).

Garment meshes are scaled 0.6–0.65 to fit the worktop, and a scene containing a cloth shell (garment
or bag) caps at `CLOTH_SCENE_MAX` = 4 objects total.

New bag imports follow `assets/objects/objaverse_bags/README.md`. Keep `kind: "cloth"` so Newton
uses `ClothConfig`/`add_cloth`, and set semantic `category: "bag"` so task generation exposes bag
affordances. The initial catalog is intentionally only three diverse bags—proportionate to the
three garments—and explicitly identifies Objaverse so more licensed examples can be added later.
Bags start physically collapsed on the table; `container: true` exposes them to task generation,
but scene gen excludes them from initial `in` relation targets. Opening the mouth and inserting an
object must happen through the generated manipulation task, not an impossible pre-open settle pose.
