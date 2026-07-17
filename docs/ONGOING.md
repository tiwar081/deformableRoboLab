# ONGOING

Scratchpad for the **current** in-flight task: what's unresolved right now, what was just tried, and
the working hypotheses. Keep it lean — when something is proven and durable, promote it to CLAUDE.md
(or the relevant `docs/` file) and delete it here. Reset this file at the start of each new big task.

## In flight: three-stage agentic pipeline (2026-07-17)

`agent_pipeline.py` + `agentic_pipeline/` (+ `agentic_pipeline/SKILL.md`, registered at
`.claude/skills/agentic-pipeline`) — see [agentic-pipeline.md](agentic-pipeline.md). Verified:
userless end-to-end run (`outputs/agenticPipeline/workshop_bench_cable` — self-invented prompt,
default back-long-edge robot placement, reported default bird's-eye camera, settle ok, feasible
cable-coiling task); offline unit tests for placements/alignment/reach/schemas; --user interview
parsing; --scene_init rearrange in flight. Key mechanics landed with it:
- `robot_base_xform` placements WORK (yawed base: IK/FK/physics verified; the franka_stand render
  fixture now follows base yaw via `droid_scene_config(robot_yaw_deg=...)`).
- The legacy `task_generator._base_xy` bug is fixed: the framework-default mount is (-0.45,-0.45)
  (right SHORT edge; scene_generator's "base at origin" docstring was wrong), and reach is now
  nearest-point + base-aware (`check_reachable(base_xy=...)`).
- Direction words are ROBOT-POV everywhere (`geometry.direction_vectors`); the relation solver
  takes `facing_yaw_deg`; a `tall_keepout` disc under the parked gripper's home TCP repels tall
  (>0.14 m) objects (a wood block under the parked fingers starts the sim in collision).
- Prompts are DATA: `agentic_pipeline/prompts/*.md` ($-templates), not Python strings.
- Post-render visual verification is OPT-IN everywhere now (scene_generator --verify flag too).

## Prior: agentic scene generator (2026-07-13)

`agentic_pipeline/scene_generator.py` + `assets/objects/scene_catalog.json` + [scene-generator.md](scene-generator.md)
landed and verified end-to-end (rigid+squishy+cable scene, and a garment scene — settled stills in
`outputs/sceneGen/`): prompt → Claude (fable-5, opus-4-8 fallback; raw HTTPS — the anthropic SDK
conflicts with the venv's isaacsim `typing_extensions==4.12.2` pin) → grammar validation → circle
push-apart spatial solver → settle-only DemoSpec → over-the-shoulder still. New centralized pieces:
`SoftMeshConfig`/`add_soft_mesh_object` (imported `.tet` FEM squishies), ClothConfig `mesh_file`
(OBJ garments), `Obj(kind="soft_mesh")`, yaw on `add_ycb_mesh`, xform-baking in `load_usd_mesh` +
`viz_assets` (the objaverse apple carries a 0.01-scale xformOp; its raw points are 100x too big —
the first time the apple was actually placed in a scene it threw the robot).

Findings that must not be re-walked:
- **tetgen tets NaN the VBD FEM** (slivers, min |vol| 3e-16..1e-10 m^3; WORSE with quality
  switches). fTetWild (`wildmeshing`, DefGraspSim's own tool; `assets/objects/_utils/make_tet.py`)
  gives well-conditioned coarse tets → all three squishies settle DEAD still (max |qd| = 0.000).
- **Imported squishies author their contact skin at their own PER-PARTICLE-MASS scale**
  (ke = 2.5e4·(m_p/9e-4), kd ~1.5x pair-critical, kf = 0.01·ke — see catalog `inferred` notes).
  The raspberry-scale skin on 50x-lighter particles measured ~90x-critical pair damping and NaN'd.
- SimWeaver (requested cloth source) is unreleased (empty repo); RGBench (its physics source) is
  the stand-in: measured per-garment params, meshes vendored under `assets/objects/rgbench/`.
  DefGraspSim meshes are Drive-gated (no public mirror) — geometry rebuilt from YCB via fTetWild,
  their documented FEM params (rho=1000, nu=0.3, mu=0.7, E-sweep) applied.

Open: lighting + robot placement stay at centralized defaults (RoboLab keeps them in
environment-gen — add a variations layer later); RenderSpec already exposes the hooks. Garments
are scaled 0.6-0.65 to fit the worktop; cloth scenes cap at ~4 objects.

Last completed (2026-07-10/11): contact-material ownership + the central band-limited harmonic
coupling, the realistic per-object re-authorship of EVERY object, retirement of the cloth friction
exception, the material-restore bug fix, and the legacy cleanup. All durable knowledge lives in its
docs: rules in CLAUDE.md "CONTACT-MATERIAL OWNERSHIP"; mechanics + register-gotcha + noise bands in
[solver-architecture.md](solver-architecture.md) ("Contact materials", "Verification standard");
the fold recipe + its measured history in [cloths.md](cloths.md); the raspberry-like block, cable
values, and the delicate-grasp findings in [deformables.md](deformables.md); per-demo notes in
[examples.md](examples.md).

## Standing open items (live in their docs)

- **No static friction in the rigid grasp (current config, by choice)** →
  [SOLVERS.md](SOLVERS.md) §6 trade study. Symptoms: the ~2 kg plate slowly pivots about the jaw
  axis while carried (`soft_compression_franka`, ~18°); the banana's edge grasp is intermittent at
  any force (its wedge converts squeeze into self-ejection in the same proportion at any target —
  measured at 10/30/45/80 N; only compliant fingertips or a flatter grasp point truly fix it).
  The measured option, `rigid_contact_hard=False, friction_epsilon=0.05`, is RESERVED for an
  explicit future re-opening of the slippage problem (WARNING: jerkier initial grasp; both knobs
  stay central in framework.py, never per-demo — §6).
- **Squeeze-signal under-read on tilted contact normals** → [SOLVERS.md](SOLVERS.md) §6 (last
  paragraph). The regulated signal projects each pad's force onto the jaw axis; on wedge faces it
  under-reads ~3× and the regulator over-squeezes. Physical fix: per-pad contact-NORMAL force as
  the signal — but the cable cage currently depends on the projected signal's behaviour, so the
  change needs a full-demo revalidation.
- Residual penalty-contact ring at impact/grasp moments (pre-existing) →
  [SOLVERS.md](SOLVERS.md) §5.
- **Two demos' narratives changed with the raspberry-like block** ([examples.md](examples.md)):
  `soft_compression` (the 2 kg plate now flattens the fruit) and `rigidCube_soft` (the steel cube
  squashes it and rolls off). Both are honest physics; if their *stories* should be preserved, add
  a second, firmer canonical soft object (params.py convention explicitly allows distinct named
  instances) rather than de-realizing the berry.
- **Harness-grade validation coverage**: `cloth_franka`, `cable_soft`, `soft_pickplace` are
  metric-verified at the final materials; the other five demos are render-verified only.
