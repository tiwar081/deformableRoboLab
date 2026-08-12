# Trajectory pipeline

The stage between task generation and simulation: turning a generated task + scene into robot
motion (grasp choice, waypoints, grip schedule) — i.e. producing the `DemoSpec` policy that used
to be hand-authored per demo. **The trajectory stage LANDED 2026-08-12**
([trajectory-generation.md](trajectory-generation.md), code in
`deformableManipulationTools/traj_gen/`): online selection over the grasp library
(physics-tiered re-rank + score-weighted sampling), Bezier transport legs with collision-driven
control-point insertion, a headless measured rollout, and a bounded 2-attempt LLM recovery loop
for failed grasps. The old seam — generated scenes getting a settle-only parked-arm `DemoSpec`
from `agentic_pipeline/build.py:demo_from_dir` — is now the no-`traj.json` fallback of the same
function; hand-written `examples/*.py` demos are unchanged.

**STANDING CONTRACT for whoever builds the trajectory stage: the fingers must be pre-shaped to
`candidate.width + PREGRASP_MARGIN` (capped at the 80 mm jaw) BEFORE the hand starts moving
toward the object.** Every stored candidate's collision feasibility — `pad_seat`'s retreat, the
`seat_blocked` labels, the shake pre-check, and the shake quality numbers themselves — is
computed for a hand at exactly that aperture, not fully open. A trajectory that approaches with
the jaws fully open executes a procedure nothing in the library validated. The constant and its
rationale live in `deformableManipulationTools/grasp_library.py` (`PREGRASP_MARGIN`).

Docs in this folder:

- [grasp-library.md](grasp-library.md) — the per-asset record store: storage/naming, the
  canonical OBB frame (+ ambiguity detectors), the pad-seated v2 pose convention, `pad_seat` and
  the three seat modes (with the measured vessel-class findings behind the retreat), schema and
  versioning rules, catalog coverage. The frozen API contract is
  `deformableManipulationTools/grasp_library.py`.
- [grasp-passes.md](grasp-passes.md) — the pass framework (parallel-agent-safe sidecar/merge
  system) and the seven passes: `fixture`, `geometric`, `obb_face`, `obb_bucket`, `vlm_regions`,
  `rim_pinch`, `shake_validate` — each with the measured findings and dead ends beyond its
  in-code docs. Authoring contract: `deformableManipulationTools/grasp_passes/README.md`.
- [grasp-selection.md](grasp-selection.md) — `grasp_select/`: prune → clearance → projection →
  score → sample over a placed object; the Z-X-Z projection result that closed the IK-vocabulary
  question; deliberately unwired until a trajectory stage exists.
- [trajectory-generation.md](trajectory-generation.md) — THE STAGE ITSELF
  (`deformableManipulationTools/traj_gen/`): run-dir CLI, online selection re-rank/sampling rules,
  the Bezier/collision plan, the rollout metrics, the 2-attempt grasp-failure LLM loop, and the
  `traj.json`/`traj_result.json` artifacts.
- [llm-retry.md](llm-retry.md) — the last-resort LLM stage for objects with no passing grasp:
  trigger (zero legitimate holds after full shake coverage), two bounded rounds (blind, then
  visual-feedback), the `unusable` verdict, and the candidate status taxonomy it runs on
  (legitimate / weak `retreated` / discarded `seat_blocked`).

Live status of the in-flight work (current library counts, validation campaign, open decisions,
queued runs): [docs/ONGOING.md](../ONGOING.md) — trust it over anything summarized elsewhere.
Scope boundary (**no cloth, no bags, no cables** — rigid bodies and squishy FEM objects only):
stated and enforced in `grasp_library.py`.
