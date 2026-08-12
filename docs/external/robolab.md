# RoboLab — NVIDIA's policy-evaluation benchmark (reference summary)

**Paper:** *RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies*
(Yang, Dagli, Zook, Hadfield, Goyal, Birchfield, Ramos, Tremblay — NVIDIA, RSS 2026,
[arXiv:2604.09860](https://arxiv.org/abs/2604.09860))
**Repo:** [`_external/RoboLab`](../_external/RoboLab/) (Apache 2.0) ·
[website](https://research.nvidia.com/labs/srl/projects/robolab/) ·
[GitHub](https://github.com/NVlabs/RoboLab/)

> Paraphrased summary of the paper + a map of the vendored repo, for context/reference use.

## What it is

A benchmarking framework for evaluating **real-world generalist manipulation policies in
simulation**, built on Isaac Lab / Isaac Sim. The core idea: policies are trained ONLY on
real-world data (DROID), and high-fidelity sim is used purely as a controlled *evaluation*
environment — so training and evaluation domains are decoupled and success measures actual
generalization, not overfit to simulator quirks. Scenes/tasks are robot- and policy-agnostic
(robot binding is deferred to environment registration), authored by drag-and-drop or an
LLM pipeline in minutes rather than the ~1 hr/scene of real2sim (Gaussian-splat) approaches.

## Why this repo cares

- **RoboLab is our sim2real reference.** Our Franka arm gains (400/80 stiffness/damping on all
  7 joints, `params.py`) match RoboLab's DROID Franka exactly
  (`_external/RoboLab/robolab/robots/droid.py`), our friction anchors come from RoboLab's
  physics utils (e.g. rubber fingertip pad mu = 0.8), and our rigid grasp follows RoboLab's
  pattern of closing to a fixed target with no object-size preset width.
- **`robolabViz/` reproduces RoboLab's render look** — the DROID rig in
  `_external/RoboLab/examples/run_recorded.py` (home-office dome, sphere key light, maple table,
  external + wrist cameras). See [robolab-graphics.md](../rendering/robolab-graphics.md).
- **RoboLab's stated limitation is exactly this project's vision.** The paper's Limitations
  section: it "currently focuses on rigid-body tabletop scenes and does not fully capture the
  challenges of deformable object manipulation (e.g., cloth, cables, bags)", and contact-rich
  skills needing precise force control are underrepresented. Our Newton deformable environment
  (cables, cloth, FEM, force-controlled grip) targets that gap.
- Standing rule still applies: **never import or depend on `_external/RoboLab` at runtime** —
  copy what's needed (as `params.py` and `robolabViz/` already do).

## The framework (paper §III)

Formalism: a **scene** S = {(bᵢ, pᵢ, qᵢ)} is object instances (from a ~300-asset catalog:
YCB, HOT3D, HOPE, HANDAL, VoMP — each with visual + collision mesh, mass, friction) posed in a
workspace. A **task** T = {S, l} adds a language instruction. An **environment**
E = (T, R, O, A, ξ) binds a robot embodiment, policy obs/action spaces, and scene variations
ξ (camera, lighting, background, pose) at runtime — the same task runs on any Isaac-Lab robot.

Tasks decompose into **sequential subtasks with parallel events** (e.g. PickPlace(x) =
Grasp → Hover → Drop → Done), which drives partial-credit scoring.

**LLM scaling pipelines** (both with validate-and-refine feedback loops):
- *Scenes:* an LLM turns a theme ("messy counter") into objects + spatial predicates
  (`place-on-base`, `place-in`, `place-on`, `cluster-around`) → a spatial solver resolves poses
  (rejection sampling, SAT on OBBs, adaptive margins) → 300-step Isaac Sim settle flags unstable
  objects (>2 cm displacement) → text feedback refines the plan.
- *Tasks:* an LLM generates task code from the scene catalog + predicate library + competency
  templates; syntax/asset/containment checks feed a fix prompt. LLM-judged quality over 812
  generated tasks: 0.91 alignment, 76% fully aligned.

## RoboLab-120 benchmark

120 hand-curated household pick-and-place-family tasks (65 simple / 38 moderate / 18 complex),
multi-labeled across three **competency axes**: *visual* (color, semantics, size — 91 tasks),
*relational* (conjunction, counting, spatial — 44), *procedural* (affordance, reorientation,
stacking — 36). Difficulty = N_subtasks + max skill weight (0 visual ID, 1 spatial,
2 procedural, 3 reorientation/dynamic); simple ≤ 2, moderate 3–4, complex ≥ 5. Each task ships
**three instruction variants** (default / vague / specific) for language-robustness probing.

**Metrics beyond binary success:** normalized graded score Sc(T) = (1/|T|) Σ w_τ Sc(τ)
(partial credit per subtask/event); trajectory quality — SPARC smoothness (spectral arc length
of the EE velocity profile; closer to 0 = smoother), EE speed, path length; and discrete
**event tracking** (wrong object grasped, object dropped, gripper collision) that catches
"successful but sloppy" episodes.

**Sensitivity analysis:** Mixed Neural Posterior Estimation (simulation-based inference) learns
p(scene parameters | success) — a posterior tightly concentrated at the nominal value means the
policy is brittle to that parameter.

## Key results (DROID-finetuned policies, N=10 episodes/task)

| Policy | Succ% / Score |
|---|---|
| π0.5 | 28.0 / 0.43 |
| π0-FAST | 15.5 / 0.27 |
| GR00T N1.6 | 7.2 / 0.17 |
| π0 | 5.0 / 0.12 |
| PaliGemma | 3.4 / 0.10 |

- The score/success gap exposes partial progress: π0.5 gets only 13.5% success on complex tasks
  but 0.44 score — policies understand the task, then fail in late execution.
- Performance degrades with vaguer instructions (π0.5 28.0% → 15.3%), more clutter, and longer
  horizons. Robust to lighting (90–100%) and background/table textures (<5% drop).
- MNPE: success depends critically on the **wrist camera staying near its nominal pose**
  (external camera far more tolerant); object placement sweet spot ~0.5 m from robot base
  (reachability).
- **Real-world validity:** policy ranking matches RoboArena exactly (Spearman ρ = 1.00,
  Pearson r = 0.68) — sim results are a meaningful proxy for real-world ordering.
- Statistical caveat: N=10 gives ±30% CI per task at p=0.5 — per-task comparisons need N≥100;
  only aggregate numbers are tight.

## The repo (`_external/RoboLab`)

Isaac Sim 5.0 + Isaac Lab 2.2.0, Python 3.11, `uv`-managed. ~8 GB (assets ~7 GB), RTX GPU
(48 GB+ VRAM recommended), ~30 GPU h per 100 tasks at 1.4 it/s. Terminology: *scene* (USD) →
*task* (dataclass: scene + instruction + termination predicates + subtasks) → *environment*
(task + robot/camera/lighting/background configs, registered as a Gymnasium env; `--num-envs N`
runs N parallel instances) → *episode* / *run*.

| Path | What it is |
|---|---|
| `robolab/core/` | Environments, scenes, task machinery, observations, sensors, metrics, event logging |
| `robolab/tasks/benchmark/` | The 120 task definitions (one file each; see `robolab/tasks/README.md` for the full table) |
| `robolab/robots/` | DROID + Franka articulation/actuator configs (`droid.py` = our gain reference) |
| `robolab/registrations/` | Task × robot × obs/action registration (built-in: DROID joint-position) |
| `robolab/variations/` | Camera / lighting / background perturbations for robustness sweeps |
| `robolab/eval/` | `InferenceClient` ABC (`base_client.py`), episode runner, results summarizer |
| `policies/` | Server-client policy backends: `pi0_family`, `gr00t`, `cosmos3`, `dreamzero` — model runs as a standalone server, RoboLab connects via WebSocket/ZMQ/HTTP |
| `examples/` | Policy-free runners (`run_empty.py`, `run_recorded.py` — the robolabViz look source, `run_gripper_toggle.py`) |
| `assets/` | Object/scene/background USD libraries |
| `analysis/`, `dashboard/` | Results compilation, MNPE sensitivity analysis, self-contained web dashboard (`uv run robolab-dashboard`, port 8080) |
| `skills/` | `/robolab-scenegen` and `/robolab-taskgen` Claude Code skills (natural-language scene/task authoring) |
| `docs/` | Full documentation index at `docs/README.md` |

Typical run: `python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask
--num-envs 10 --enable-subtask --headless` (policy server started separately; results + episode
videos land in `output/`, browsable in the dashboard).
