"""SHARED — compose per-pass sidecars into the final record. Read-only for pass authors.

``assets/objects/grasps/<catalog_name>.json`` is a BUILD ARTIFACT: everything in it comes from a
sidecar, so it can always be rebuilt and no pass ever has to open it. Consequences worth knowing:

  * **Hand edits to a record are lost on the next merge.** Authored candidates belong in a pass
    (that is what ``grasp_passes/fixture`` is).
  * **Deleting a pass's sidecars removes its candidates** from the record on the next merge.

Deduplication is BY SOURCE TAG. A source tag is owned by exactly one pass, the harness namespaces
every candidate id as ``<source>/<id>``, and the merge takes all candidates carrying a source from
that source's current sidecar only. So re-running a pass and merging replaces its contribution
wholesale — candidates never accumulate, and a pass can shrink its output.

Frame agreement is enforced rather than assumed: every sidecar must have been computed from the
same mesh (``mesh_sha1``) and carry the same canonical frame, or the poses are not in comparable
coordinates and merging them would silently mix frames.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..grasp_library import (MAX_JAW_WIDTH, PAD_NEAR_Z, RETRY_EXHAUSTED_TOKEN, SEAT_BLOCKED_LABEL,
                             UNUSABLE_NOTE, WEAK_GRASP_LABEL, GraspRecord, GraspSchemaError,
                             QUALITY_FIELDS, is_shake_covered, is_weak, record_path, write_record)
from .base import existing_pass_dirs, read_sidecar, sidecar_assets, source_of_id
from .catalog import load_asset

_FRAME_TOL = 1.0e-9

# An asset every generator ran on and none found a grasp for is a RESULT, not a failure: the object
# has no feature this gripper can close on (too wide, no parallel faces within the jaw). It gets a
# record with an empty candidate list saying so, rather than no record — "no record" is
# indistinguishable from "nothing has run yet", and a downstream consumer needs to tell those apart
# to know whether to wait for a pass or to skip the object.
OUT_OF_REACH_NOTE = (
    "OUT OF REACH for this gripper: {n} pass(es) ran and none found a grasp within the "
    "{jaw:.0f} mm jaw. This is a measured result, not a failed run — the object offers no feature "
    "the parallel jaws can close on. Re-check if the gripper or MAX_JAW_WIDTH changes."
)


def is_out_of_reach(record: GraspRecord | None) -> bool:
    """True when passes have run for this asset and produced no graspable pose between them."""
    return record is not None and not record.candidates


# Sidecar versions the pass-version gate accepts BESIDES the pass's current one. This allowlist
# exists for exactly one shape of change: a version bump that only NARROWS what the pass chooses
# to run on, leaving poses and trial procedure unchanged, so the older sidecars remain true
# measurements. Anything that changes what a stored number MEANS must NOT be listed — re-run the
# pass instead. Every entry needs its justification here:
#   shake_validate 3: v4 (2026-08-11) only excludes retreated candidates from trial selection and
#   adds carry-forward of its own previous annotations; the trial procedure, the rig, and every
#   stored quality number are identical, so v3's 1719 measured trials stay valid.
COMPATIBLE_VERSIONS = {"shake_validate": (3,)}


class MergeError(RuntimeError):
    """Sidecars cannot be composed (disagreeing frames, stale geometry, contested source tag)."""


def sidecars_for(asset_name: str) -> list:
    """Every pass sidecar that exists for this asset, in pass-name order."""
    out = []
    for pass_name in existing_pass_dirs():
        side = read_sidecar(pass_name, asset_name)
        if side is not None:
            out.append({**side, "pass_name": pass_name})
    return out


def _check_agreement(asset, sides) -> None:
    # PASS-VERSION GATE. A sidecar must have been written by the CURRENT version of its pass, not
    # merely carry the current POSE_CONVENTION: the convention names the pose format, the pass
    # version names the rule that produced the numbers, and the two can diverge — the clamped
    # deep-seat (2026-08-06) changed what an uncontained v2 pose IS while the v2 label stayed. A
    # pre-clamp sidecar restored from a backup passes every other check here (same mesh, same frame,
    # same convention) and would silently mix two seating rules into one record. run_pass's skip
    # logic cannot catch this — it only runs when a pass runs; a restored file just sits there.
    # A sidecar whose pass is no longer registered is skipped (there is no current version to hold
    # it to — deliberately, so the selfcheck's throwaway passes and a mid-development directory
    # don't break the merge; deleting a pass's sidecars is still the way to retire its output).
    from . import discover_passes

    registry = discover_passes()
    for s in sides:
        current = registry.get(s["pass_name"])
        if current is not None and s["pass"].get("version") != current.version \
                and s["pass"].get("version") not in COMPATIBLE_VERSIONS.get(s["pass_name"], ()):
            raise MergeError(
                f"{s['path']}: written by pass {s['pass_name']!r} version "
                f"{s['pass'].get('version')!r}, but the registered pass is version "
                f"{current.version}. The pose convention label alone cannot catch this — the "
                f"seating rule can change under the same label. Re-run the pass; do not relabel.")
    for s in sides:
        rec, where = s["record"], s["path"]
        if rec.mesh_sha1 != asset.mesh_sha1:
            raise MergeError(
                f"{where}: computed from a different mesh (sidecar {rec.mesh_sha1[:12]}, current "
                f"{asset.mesh_sha1[:12]}). The asset changed — re-run pass {s['pass_name']!r}.")
        if np.abs(rec.frame.rotation - asset.frame.rotation).max() > _FRAME_TOL \
                or np.abs(rec.frame.translation - asset.frame.translation).max() > _FRAME_TOL:
            raise MergeError(
                f"{where}: canonical frame disagrees with the asset's. Poses from different frames "
                f"are not comparable; re-run pass {s['pass_name']!r} (and do not recompute the "
                f"frame inside a pass — use ctx.frame).")
    owners: dict = {}
    for s in sides:
        src = s["pass"]["source"]
        if src in owners:
            raise MergeError(
                f"source tag {src!r} is claimed by two passes ({owners[src]!r} and "
                f"{s['pass_name']!r}). A source tag identifies exactly one pass — the merge "
                f"deduplicates by it, so sharing one makes their candidates indistinguishable.")
        owners[src] = s["pass_name"]


def merge_asset(asset_name: str, *, write: bool = True, verbose: bool = True) -> GraspRecord | None:
    """Compose every sidecar for one asset into its final record. Returns None if there are none
    (and then removes a stale record, so deleting the last pass really does clear the artifact)."""
    asset = load_asset(asset_name)
    sides = sidecars_for(asset_name)
    if not sides:
        stale = record_path(asset_name)
        if write and stale.exists():
            stale.unlink()
            if verbose:
                print(f"  remove {stale.name} (no sidecars left)")
        return None
    _check_agreement(asset, sides)

    candidates, provenance = [], []
    for s in sides:
        # Dedup by source: each source's candidates come from its own sidecar and nowhere else, so
        # this concatenation cannot double-count. _check_agreement guarantees one sidecar per source.
        for c in s["record"].candidates:
            if source_of_id(c.id) != s["pass"]["source"]:
                raise MergeError(
                    f"{s['path']}: candidate {c.id!r} is not namespaced under the pass's source "
                    f"{s['pass']['source']!r}")
            candidates.append(c)
        provenance.append(f"{s['pass']['name']}@{s['pass']['version']}")

    # Apply annotations from consumer passes onto the candidates their producers own.
    by_id = {c.id: i for i, c in enumerate(candidates)}
    for s in sides:
        for cid, ann in s["annotations"].items():
            if cid not in by_id:
                raise MergeError(
                    f"{s['path']}: annotates {cid!r}, which no sidecar produces. Either its "
                    f"producer pass has not run for this asset, or the id is stale.")
            i = by_id[cid]
            c = candidates[i]
            quality = dict(c.quality)
            quality.update({k: v for k, v in ann.get("quality", {}).items() if k in QUALITY_FIELDS})
            qsrc = ann.get("quality_source") or c.quality_source
            labels = tuple(dict.fromkeys(tuple(c.labels) + tuple(ann.get("labels", ()))))
            notes = ann.get("notes") or c.notes
            candidates[i] = replace(c, quality=quality, quality_source=qsrc, labels=labels,
                                    notes=notes)

    # ---- DERIVATION layer (2026-08-11) ----------------------------------------------------------
    # Central candidate-status policy, applied to the composed set so no generator or consumer
    # re-implements it (spec: docs/trajPipeline/grasp-library.md "Candidate statuses").
    # 1. seat_blocked candidates are DISCARDED: the hand collides at every depth on their
    #    approach, so no collision-free grasp exists to command. The generator sidecars keep the
    #    measurement (marked, not deleted — there); the record, which consumers read, drops them.
    #    Runs AFTER annotations are applied, so an annotation addressed to a blocked candidate
    #    (shake's old skip entries) is dropped with it rather than raising as an orphan.
    n_blocked = sum(1 for c in candidates if SEAT_BLOCKED_LABEL in c.labels)
    candidates = [c for c in candidates if SEAT_BLOCKED_LABEL not in c.labels]
    # 2. retreated candidates are WEAK GRASP OPTIONS, never legitimate candidates: stamp the one
    #    shared label consumers filter on. 3. clamped_deep candidates get their forward overhang
    #    computed and TRACKED (never scored).
    derived = []
    for c in candidates:
        if is_weak(c):
            c = replace(c, labels=tuple(dict.fromkeys(tuple(c.labels) + (WEAK_GRASP_LABEL,))))
        elif c.seat_mode == "clamped_deep" and c.span is not None and c.seat_depth is not None:
            c = replace(c, overhang=max(0.0, float(c.seat_depth) + float(c.span) - PAD_NEAR_Z))
        derived.append(c)
    candidates = derived

    base_note = "GENERATED from per-pass sidecars in _passes/ — do not hand-edit."
    if n_blocked:
        base_note = (f"{n_blocked} seat_blocked candidate(s) discarded (hand collides at every "
                     f"depth — no collision-free grasp; measurements remain in the generator "
                     f"sidecars). ") + base_note
    if candidates:
        notes = base_note
    else:
        notes = OUT_OF_REACH_NOTE.format(n=len(sides), jaw=MAX_JAW_WIDTH * 1000) + " " + base_note
    # The record-level UNUSABLE verdict: the llm_retry stage exhausted both its rounds, every
    # legitimate candidate is shake-covered, and the composed record still holds nothing. Derived
    # here — the record is a build artifact, so a durable verdict must be recomputable from
    # sidecars alone (docs/trajPipeline/llm-retry.md). The coverage condition matters: without it
    # the record would read unusable in the window between round B's generation and its trials.
    retry_exhausted = any(s["pass_name"] == "llm_retry"
                          and RETRY_EXHAUSTED_TOKEN in (s["pass"].get("notes") or "")
                          for s in sides)
    legit = [c for c in candidates if not is_weak(c)]
    all_covered = all(is_shake_covered(c) for c in legit)
    any_held = any(c.quality and c.quality.get("object_in_gripper") is not None
                   and float(c.quality["object_in_gripper"]) == 1.0 for c in legit)
    if retry_exhausted and all_covered and not any_held:
        notes = UNUSABLE_NOTE + " " + notes
    record = GraspRecord(
        name=asset.name, kind=asset.kind, frame=asset.frame, candidates=tuple(candidates),
        file=asset.file, scale=asset.scale, mesh_sha1=asset.mesh_sha1,
        generator="grasp_passes.merge: " + ", ".join(provenance),
        notes=notes)
    if write:
        out = write_record(record)
        if verbose:
            if candidates:
                print(f"  merge {asset_name}: {len(candidates)} candidate(s) from "
                      f"{len(sides)} pass(es) -> {out.name}")
            else:
                print(f"  merge {asset_name}: OUT OF REACH — {len(sides)} pass(es) ran, "
                      f"0 candidates within the {MAX_JAW_WIDTH * 1000:.0f} mm jaw -> {out.name}")
    return record


def merged_assets() -> list:
    """Assets that have at least one sidecar, i.e. everything the merge would write."""
    names = set()
    for pass_name in existing_pass_dirs():
        names.update(sidecar_assets(pass_name))
    return sorted(names)


def merge_all(*, write: bool = True, verbose: bool = True) -> dict:
    out, failed = {}, []
    for name in merged_assets():
        try:
            rec = merge_asset(name, write=write, verbose=verbose)
        except (MergeError, GraspSchemaError) as exc:
            print(f"  FAIL  {name}: {exc}")
            out[name] = None
            failed.append(name)
            continue
        out[name] = rec
    if verbose:
        unreachable = sorted(n for n, r in out.items() if is_out_of_reach(r))
        total = sum(len(r.candidates) for r in out.values() if r is not None)
        print(f"  {len(out) - len(failed)} asset(s) merged, {total} candidate(s); "
              f"out of reach: {', '.join(unreachable) if unreachable else 'none'}"
              + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return out
