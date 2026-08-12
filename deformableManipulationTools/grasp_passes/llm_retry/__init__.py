"""``llm_retry`` pass — LLM-proposed grasp candidates for objects nothing else can hold.

THE CONTRACT IS docs/trajPipeline/llm-retry.md — read it before changing anything here. In one
paragraph: when every generator has run and ``shake_validate`` has covered the result with ZERO
legitimate holds, a multimodal LLM is shown the object (six canonical views), the measured hand
geometry, and everything that was tried with how it failed, and asked for up to ten new candidates
(round A, ids ``a00…a09``). Those go through the same obb_bucket + shake cycle as everyone else's.
If they are all covered and still nothing holds, ONE feedback round runs (round B, ``b00…b09``):
the LLM sees its own round-A poses rendered as markers on the object with each failure spelled
out. After round B the pass's notes carry :data:`grasp_library.RETRY_EXHAUSTED_TOKEN` and the
MERGE derives the durable record-level ``unusable`` verdict. Two rounds, EVER.

Properties that make this a well-behaved pass despite the nondeterministic annotator:

* **Poses are stored EXACTLY as the LLM gives them** — ``seat_mode: "llm"``, no ``pad_seat``, no
  retreat. ``span``/``seat_depth`` are MEASURED at the given pose (:func:`grasp_library.
  measure_span_at`, probe only); authoring VALIDATION drops (and records, for round-B feedback) a
  pose with no material in the jaw column, one gripping air (``seat_depth > 0``), one buried past
  the palm (``make_candidate`` raises), an out-of-jaw width, or a jaw axis parallel to the
  approach.
* **``run()`` is deterministic given the state cache** (:mod:`.state`): the raw LLM answer is
  cached at call time and candidates are re-derived from it on every later run, so re-running
  re-emits byte-identical sidecars (``--check-idempotent`` holds, as in ``vlm_regions``).
  Transport failure RAISES and is never cached. The sidecar is CUMULATIVE: after round B, round
  A's candidates are re-emitted UNCHANGED so shake's annotations (keyed by id in shake's own
  sidecar) stay attached through the merge.
* **The sanctioned exception**: this pass — alone — READS the merged record (``load_grasps``),
  because its trigger (:func:`grasp_library.needs_llm_retry`) and its input package are the
  COMPOSED state, all producers' candidates with shake's annotations applied. It still never
  writes the record, and it consumes no ``ctx.upstream`` (``requires = ()``).

Orchestration (round A -> merge -> obb_bucket -> shake -> merge -> round B -> …):

    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle [--asset NAME]
    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle --dry-run
    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry status
"""
from __future__ import annotations

import numpy as np

from ...grasp_library import (MAX_JAW_WIDTH, RETRY_EXHAUSTED_TOKEN, grasp_transform,
                              has_grasps, is_shake_covered, load_grasps, make_candidate,
                              measure_span_at, needs_llm_retry)
from ..base import GraspPass, PassContext, PassError, PassOutput, namespaced_id
from . import prompt, state

__all__ = ["PASS", "LlmRetryPass", "derive_round", "next_step", "MAX_CANDIDATES", "ROUND_TAGS"]

MAX_CANDIDATES = prompt.MAX_CANDIDATES
# The per-round label every emitted candidate carries beside its semantic label, so a consumer can
# tell which round proposed it without parsing the id.
ROUND_TAGS = {"a": "llm_round_a", "b": "llm_round_b"}


class LlmRetryPass(GraspPass):
    name = "llm_retry"
    source = "llm_retry"
    version = 1
    kinds = ()          # every supported kind; the REAL gate is the trigger in applies_to
    requires = ()       # reads the merged record instead (the sanctioned exception, see docstring)

    def applies_to(self, name: str, kind: str) -> bool:
        """The trigger, verbatim from the contract: a merged record exists AND
        ``needs_llm_retry`` holds on it (supported kind, not already unusable, every legitimate
        candidate shake-covered, zero legitimate holds). Once the merge stamps ``unusable`` — or
        one LLM candidate holds — this goes False and the pass never runs again for the asset;
        its cumulative sidecar stays on disk for the merge to read."""
        return has_grasps(name) and needs_llm_retry(load_grasps(name, use_cache=False))

    def run(self, ctx: PassContext) -> PassOutput:
        st = state.load_state(ctx.name, ctx.asset.mesh_sha1, prompt.PROMPT_VERSION, self.version)
        called_a = st is None or "a" not in st.get("rounds", {})
        if called_a:
            st = self._generate(ctx, "a", st)
        cands_a, drops_a = derive_round(ctx.asset, st["rounds"]["a"]["answer"], "a")

        if "b" not in st["rounds"]:
            # Round-B decision, from the MERGED record: only when every emitted round-A candidate
            # is shake-covered and none held. Freshly generated round-A output cannot be covered
            # yet, so the round it was generated in never cascades into round B.
            if not called_a and _round_a_resolved(ctx.name, cands_a):
                st = self._generate(ctx, "b", st, round_a=(cands_a, drops_a))
            else:
                return PassOutput(
                    candidates=cands_a,
                    notes=(f"round A: {len(cands_a)} candidate(s) emitted, {len(drops_a)} dropped "
                           f"at authoring; round B pends shake coverage of the a* candidates"))

        cands_b, drops_b = derive_round(ctx.asset, st["rounds"]["b"]["answer"], "b")
        # Round A re-emitted UNCHANGED + round B; the exhausted token in the notes is what the
        # merge reads to derive the record-level unusable verdict once coverage completes.
        return PassOutput(
            candidates=cands_a + cands_b,
            notes=(f"round A: {len(cands_a)} emitted ({len(drops_a)} authoring-dropped); "
                   f"round B: {len(cands_b)} emitted ({len(drops_b)} authoring-dropped); "
                   f"both rounds spent — {RETRY_EXHAUSTED_TOKEN}"))

    # ---- the one nondeterministic step, gated by the cache --------------------------------------
    def _generate(self, ctx: PassContext, round_key: str, st: dict | None,
                  round_a: tuple | None = None) -> dict:
        """Build the request, make the ONE call for this round, and cache the raw answer.

        Raises (and caches NOTHING) on transport failure or a structurally empty answer — "the LLM
        proposed nothing" must never become a stored finding."""
        from . import feedback as F

        if st is None:
            st = state.new_state(ctx.name, ctx.asset.mesh_sha1, prompt.PROMPT_VERSION,
                                 self.version)
        record = load_grasps(ctx.name, use_cache=False)
        views = F.render_views_for(ctx.asset)
        entry: dict = {}
        if round_key == "a":
            content = prompt.round_a_content(ctx.asset, views, F.tried_report(record))
        else:
            cands_a, drops_a = round_a
            fb_images, fb_report, fb_digest = F.round_b_feedback(ctx.asset, record, cands_a,
                                                                 drops_a)
            content = prompt.round_b_content(
                ctx.asset, views, F.tried_report(record, exclude_sources=(self.source,)),
                fb_images, fb_report)
            entry["feedback_digest"] = fb_digest

        print(f"  [llm_retry] {ctx.name}: round {round_key.upper()} — calling the LLM")
        answer, model = prompt.ask_candidates(ctx.name, content)
        raws = list(answer.get("candidates") or [])[:MAX_CANDIDATES]
        if not any(_parse_raw(r) is not None for r in raws):
            raise PassError(f"llm_retry round {round_key} for {ctx.name!r}: the response contains "
                            f"zero structurally valid candidates — refusing to cache it")
        cands, drops = derive_round(ctx.asset, answer, round_key)
        entry.update({"request_digest": state.request_digest(content), "model": model,
                      "answer": answer, "authoring_failures": drops})
        st["rounds"][round_key] = entry
        state.save_state(st)
        print(f"  [llm_retry] {ctx.name}: round {round_key.upper()} cached — "
              f"{len(cands)} valid candidate(s), {len(drops)} authoring failure(s)")
        return st


PASS = LlmRetryPass()


# =================================================================================================
# Authoring validation — the LLM's numbers, judged (never corrected)
# =================================================================================================
def _parse_raw(raw) -> dict | None:
    """Structural parse of one response candidate, or None. Structure only — geometric judgement
    belongs to :func:`derive_round`, and 'zero structurally valid' is the raise-don't-cache line."""
    try:
        pos = np.asarray(raw["position"], dtype=float).reshape(3)
        appr = np.asarray(raw["approach"], dtype=float).reshape(3)
        jaw = np.asarray(raw["jaw_axis"], dtype=float).reshape(3)
        width = float(raw["width"])
    except Exception:                          # noqa: BLE001 - malformed is a per-candidate verdict
        return None
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(appr)) and np.all(np.isfinite(jaw))
            and np.isfinite(width)):
        return None
    return {"position": pos, "approach": appr, "jaw_axis": jaw, "width": width,
            "label": str(raw.get("label", "body")), "rationale": str(raw.get("rationale", ""))}


def derive_round(asset, answer: dict, round_key: str) -> tuple[list, list]:
    """``(candidates, authoring_failures)`` for one round's cached answer — pure and deterministic.

    Ids are positional (``a00…``/``b00…``) and a dropped pose CONSUMES its id, so the round-B
    feedback can name exactly which proposal was invalid. The pose is stored EXACTLY as given
    (``seat_mode="llm"``): what runs here is measurement (:func:`measure_span_at` — probe only)
    and validation, never seating, retreat, or any auto-correction. Drop reasons (kept in the
    state cache and re-derivable from it) are the contract's authoring-failure list."""
    verts = asset.canonical_vertices()
    out, drops = [], []
    for i, raw in enumerate(list(answer.get("candidates") or [])[:MAX_CANDIDATES]):
        cid = f"{round_key}{i:02d}"

        def drop(reason: str) -> None:
            drops.append({"id": cid, "reason": reason, "raw": raw})

        p = _parse_raw(raw)
        if p is None:
            drop("malformed: position/approach/jaw_axis must each be 3 finite numbers and width "
                 "a finite number")
            continue
        if not (0.0 < p["width"] <= MAX_JAW_WIDTH + 1.0e-9):
            drop(f"width {p['width'] * 1000:.1f} mm outside (0, {MAX_JAW_WIDTH * 1000:.0f}] mm — "
                 f"the jaw cannot make this opening")
            continue
        try:
            grasp_transform(p["position"], p["approach"], p["jaw_axis"])
        except ValueError as exc:
            drop(f"invalid pose frame: {exc}")
            continue
        seat = measure_span_at(p["position"], p["approach"], p["jaw_axis"], p["width"],
                               verts, asset.faces)
        if seat is None:
            drop("no material in the jaw column at this pose — the jaws would close on nothing")
            continue
        if seat.seat_depth > 0.0:
            drop(f"gripping air: nearest material sits {seat.seat_depth * 1000:.1f} mm IN FRONT "
                 f"of the TCP (it must lie behind it, between -45 mm and 0)")
            continue
        label = p["label"] if p["label"] in prompt.ALLOWED_LABELS else "body"
        try:
            cand = make_candidate(
                asset.frame, cid, p["position"], p["approach"], p["jaw_axis"],
                width=min(p["width"], MAX_JAW_WIDTH), source=PASS.source,
                seat_mode="llm", span=float(seat.span[1] - seat.span[0]),
                seat_depth=float(seat.seat_depth),
                labels=(label, ROUND_TAGS[round_key]),
                notes=f"LLM-proposed ({ROUND_TAGS[round_key]}): {p['rationale']}".strip())
        except ValueError as exc:
            # seat_depth < -0.05 (material buried past the palm cap) or an implausible span.
            drop(str(exc))
            continue
        out.append(cand)
    return out, drops


# =================================================================================================
# Round decision — where one asset stands
# =================================================================================================
def _held(candidate) -> bool:
    q = candidate.quality.get("object_in_gripper") if candidate.quality else None
    return q is not None and float(q) == 1.0


def _round_a_resolved(name: str, cands_a: list) -> bool:
    """Round B is due iff EVERY emitted round-A candidate appears in the merged record
    shake-covered, and none held. Zero emitted candidates (all authoring-dropped) resolve
    vacuously — there is nothing to shake, and the feedback round runs on the drop reasons. A
    held one never reaches round B (the trigger is off and ``applies_to`` gates the pass out)."""
    record = load_grasps(name, use_cache=False)
    by_id = {c.id: c for c in record.candidates}
    for c in cands_a:
        merged = by_id.get(namespaced_id(PASS.source, c.id))
        if merged is None or not is_shake_covered(merged) or _held(merged):
            return False
    return True


def next_step(asset) -> str:
    """Diagnostic view of the round decision for one asset:
    ``round_a`` (no cached round A — the next forced run calls the LLM),
    ``await_shake`` (round A emitted, its candidates not yet shake-covered in the record),
    ``round_b`` (coverage complete, nothing held — the next forced run makes the final call),
    ``exhausted`` (both rounds cached — every run re-emits, no call, ever again)."""
    st = state.load_state(asset.name, asset.mesh_sha1, prompt.PROMPT_VERSION, PASS.version)
    if st is None or "a" not in st["rounds"]:
        return "round_a"
    if "b" in st["rounds"]:
        return "exhausted"
    cands_a, _ = derive_round(asset, st["rounds"]["a"]["answer"], "a")
    return "round_b" if _round_a_resolved(asset.name, cands_a) else "await_shake"
