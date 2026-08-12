# llm_retry — LLM-proposed candidates for objects with no passing grasp

**The contract is [docs/trajPipeline/llm-retry.md](../../../docs/trajPipeline/llm-retry.md)**
(trigger, two-round protocol, authoring validation, exhaustion/unusable, the sanctioned
read-the-merged-record exception). This file documents only code-level facts.

## Layout

| file | what |
|---|---|
| `__init__.py` | `PASS` (trigger-gated producer), `derive_round` (authoring validation — the pure function from a cached answer to candidates + drop reasons), `next_step` (round decision) |
| `prompt.py` | `PROMPT_VERSION`, system/user prompt builders, JSON schema (no `minimum`/`maximum`/`minItems`/`maxItems` — the endpoint 400s on them), `ask_candidates` transport (lazy `scene_generator._messages_request`, `DEFAULT_MODEL` → `FALLBACK_MODEL`, raises on failure) |
| `feedback.py` | tried-candidate report for the prompt, round-B marker images (via `geometric.viz.write_png`), `render_geometry` (derives `soft_mesh` boundary faces from the tet file for rendering only) |
| `state.py` | the per-asset state cache |
| `__main__.py` | `cycle` / `status` / `selftest` CLI |

## State cache

One JSON per asset at `assets/objects/grasps/_passes/llm_retry/_cache/<asset>.json` — invisible to
the harness (`sidecar_assets` globs `*.json` non-recursively). Keyed on
`(mesh_sha1, PROMPT_VERSION, pass version)`; a stale key reads as "no cache", which RE-ARMS both
rounds (the contract's re-check-if-the-mesh-changes). Per round it stores the request digest, the
model used, the RAW validated answer, and the authoring-failure list; round B adds the feedback
digest. `run()` re-derives candidates from the cached answer every time — that is what makes a
nondeterministic annotator pass `--check-idempotent`. **Deleting the file re-arms the retry by
hand** (the documented way to clear an `unusable` verdict, together with deleting the sidecar).

## Id scheme

Round A: `a00…a09`, round B: `b00…b09`, positional in response order; an authoring-dropped pose
CONSUMES its id so round-B feedback can name it. The harness namespaces them to `llm_retry/a00`
etc. Labels per candidate: the LLM's semantic label (vlm_regions vocabulary + `body`) plus
`llm_round_a`/`llm_round_b`. `seat_mode` is always `"llm"`.

## CLI

```bash
.venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle [--asset NAME]
.venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle --dry-run   # no network
.venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry status
.venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry selftest          # no LLM
```

`cycle` = per asset: merge → `run_pass(llm_retry, force=True)` → merge → `obb_bucket` → merge →
`shake_validate` → merge → re-evaluate; repeats once if round B became due; prints a verdict
(`HELD via llm candidate` / `UNUSABLE` / `still pending`). `force` matters: this pass has no
upstream, so its digest never changes and `run_pass` would otherwise skip the round-B re-run.
Resumable throughout — cached rounds re-emit without a network call, shake v4 carries forward
prior trials, completed assets are no-ops. Shake runs on `settings.yaml`'s device (the rig's own
default; nothing here plumbs one).

Render artifacts (audit): `outputs/grasp_viz/llm_retry/<asset>/views/*.png` (the six canonical
views) and `outputs/grasp_viz/llm_retry/<asset>/a??.png` (round-B feedback markers).
