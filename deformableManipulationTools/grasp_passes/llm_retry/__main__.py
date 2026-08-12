"""CLI for the ``llm_retry`` pass — the CYCLE that drives an asset from trigger to verdict.

    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle [--asset NAME]
    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry cycle --dry-run
    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry status
    .venv/bin/python -m deformableManipulationTools.grasp_passes.llm_retry selftest

``cycle`` runs, per triggered asset (or the one named), each step through the existing
``run_pass``/``merge`` machinery:

    merge (freshen) -> llm_retry (force) -> merge -> obb_bucket -> merge -> shake_validate ->
    merge -> re-evaluate; if round B is now due, repeat the sequence once -> verdict.

``force=True`` on llm_retry matters: its ``requires`` is empty so its upstream digest never
changes, and without force ``run_pass`` would skip the round-B re-run forever. Determinism comes
from the state cache, so a forced re-run without a due round is a byte-identical no-op. The whole
command is RESUMABLE: re-running continues where it left off, completed assets are no-ops
(``needs_llm_retry`` is False once a verdict exists), and every already-cached round re-emits
without a network call.

Shake trials run wherever ``settings.yaml`` points (``SETTINGS.device`` — the rig's own default;
this CLI adds no device plumbing for it) and cost ~60-85 s each on an A100; a full fresh round is
up to 10 trials. ``--dry-run`` assembles the complete round-A request for ONE asset — renders the
six views, builds the tried-candidate report, the messages and the schema — prints a summary and
writes the prompt text + image list into the state-cache directory, WITHOUT any network call.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ...grasp_library import (has_grasps, is_unusable, is_weak, load_grasps, needs_llm_retry,
                              record_holds)
from ..base import run_pass
from ..catalog import asset_names, load_asset
from ..merge import merge_asset
from . import PASS, derive_round, next_step, prompt, state

# The task-directed default for --dry-run: a currently-triggered asset with a rich failure history.
DRY_RUN_DEFAULT = "pitcher"


def _triggered_assets() -> list:
    """Every supported catalog asset whose merged record currently trips the trigger."""
    out = []
    for name in asset_names():
        if not has_grasps(name):
            continue
        if needs_llm_retry(load_grasps(name, use_cache=False)):
            out.append(name)
    return out


def _verdict(record) -> str | None:
    """The final per-asset verdict, or None while the cycle still has work to do."""
    llm_held = [c.id for c in record.candidates
                if c.source == PASS.source and not is_weak(c)
                and c.quality.get("object_in_gripper") == 1.0]
    if llm_held:
        return f"HELD via LLM candidate(s): {', '.join(llm_held)}"
    if record_holds(record):
        return "HELD (a non-LLM legitimate candidate holds — trigger is off)"
    if is_unusable(record):
        return "UNUSABLE — both LLM rounds spent, physics covered everything, nothing holds"
    return None


# =================================================================================================
# cycle
# =================================================================================================
def _cycle_asset(name: str) -> None:
    from .. import get_pass

    merge_asset(name)                               # freshen the derived record first
    record = load_grasps(name, use_cache=False)
    if not needs_llm_retry(record):
        done = _verdict(record)
        print(f"  no-op: {done or 'does not trigger (untested candidates remain, or unsupported)'}")
        return

    bucket, shake = get_pass("obb_bucket"), get_pass("shake_validate")
    for seq in (1, 2):
        step = next_step(load_asset(name))
        print(f"  [seq {seq}] llm_retry step: {step}")
        run_pass(PASS, name, force=True)            # force: this pass's upstream digest never moves
        merge_asset(name)
        print(f"  [seq {seq}] obb_bucket")
        run_pass(bucket, name)
        merge_asset(name)
        print(f"  [seq {seq}] shake_validate (GPU per settings.yaml; ~60-85 s per NEW trial, "
              f"previously measured candidates carry forward)")
        run_pass(shake, name)
        merge_asset(name)
        record = load_grasps(name, use_cache=False)
        verdict = _verdict(record)
        if verdict is not None:
            print(f"  VERDICT {name}: {verdict}")
            return
        if next_step(load_asset(name)) != "round_b":
            break                                   # nothing more this invocation can decide
    verdict = _verdict(record) or f"still pending ({next_step(load_asset(name))})"
    print(f"  VERDICT {name}: {verdict}")


def cmd_cycle(args) -> int:
    if args.dry_run:
        return _dry_run(args.asset or DRY_RUN_DEFAULT, device=args.device)
    targets = [args.asset] if args.asset else _triggered_assets()
    if not targets:
        print("no asset currently triggers llm_retry (needs_llm_retry is False everywhere)")
        return 0
    print(f"llm_retry cycle over {len(targets)} asset(s): {', '.join(targets)}")
    failed = []
    for name in targets:
        print(f"\n=== {name} ===")
        try:
            _cycle_asset(name)
        except Exception as exc:                    # noqa: BLE001 - one asset must not stop the rest
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            failed.append(name)
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
    return 1 if failed else 0


# =================================================================================================
# --dry-run: the full round-A request, assembled and inspected, zero network
# =================================================================================================
_BANNED_SCHEMA_KEYS = {"minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"}


def _scan_schema(node, found: set) -> None:
    if isinstance(node, dict):
        found.update(set(node) & _BANNED_SCHEMA_KEYS)
        for v in node.values():
            _scan_schema(v, found)
    elif isinstance(node, list):
        for v in node:
            _scan_schema(v, found)


def _dry_run(name: str, device: str) -> int:
    from . import feedback as F

    if not has_grasps(name):
        print(f"{name!r} has no merged grasp record — nothing to build a request from")
        return 1
    asset = load_asset(name)
    record = load_grasps(name, use_cache=False)
    print(f"[dry-run] {name}  kind={asset.kind}  extents "
          f"{np.round(asset.extents * 1000, 1).tolist()} mm  "
          f"triggered={needs_llm_retry(record)}  step={next_step(asset)}")

    views = F.render_views_for(asset, device=device)
    report = F.tried_report(record)
    content = prompt.round_a_content(asset, views, report)
    schema = prompt._schema()
    digest = state.request_digest(content)

    texts = [b["text"] for b in content if b["type"] == "text"]
    images = [v.png for v in views]
    banned: set = set()
    _scan_schema(schema, banned)

    out = state.cache_path(name).with_suffix(".dryrun.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "==== SYSTEM ====\n" + prompt.SYSTEM
        + "\n\n==== USER TEXT ====\n" + "\n\n".join(texts)
        + "\n\n==== IMAGES (in message order, before the text) ====\n"
        + "\n".join(str(p) for p in images)
        + "\n\n==== SCHEMA ====\n" + json.dumps(schema, indent=1)
        + f"\n\n==== REQUEST DIGEST ====\n{digest}\n")

    n_lines = report.count("\n") + 1
    total_png = sum(p.stat().st_size for p in images)
    print(f"  images: {len(images)} view render(s), {total_png / 1024:.0f} KiB total")
    print(f"  prompt text: {sum(len(t) for t in texts)} chars; tried-candidate report: "
          f"{n_lines} line(s)")
    print(f"  schema: structured-output safe = {not banned}"
          + (f" (BANNED KEYS PRESENT: {sorted(banned)})" if banned else
             " (no minimum/maximum/minItems/maxItems)"))
    print(f"  request digest: {digest}")
    print(f"  wrote {out}")
    print("  NO network call made — dry run stops here.")
    return 1 if banned else 0


# =================================================================================================
# status / selftest
# =================================================================================================
def cmd_status(args) -> int:
    for name in asset_names():
        if not has_grasps(name):
            continue
        record = load_grasps(name, use_cache=False)
        trig = needs_llm_retry(record)
        st = state.load_state(name, load_asset(name).mesh_sha1, prompt.PROMPT_VERSION,
                              PASS.version) if trig or state.cache_path(name).exists() else None
        rounds = sorted(st["rounds"]) if st else []
        verdict = _verdict(record)
        marker = "TRIGGERED" if trig else ("unusable" if is_unusable(record) else "-")
        print(f"  {name:20s} {marker:10s} rounds_cached={rounds or '-'} "
              f"{('verdict: ' + verdict) if verdict else ''}")
    return 0


def cmd_selftest(args) -> int:
    """The authoring-validation path end-to-end on a hand-built answer — no LLM, no shake.

    Uses the banana's canonical frame (extents ~198 x 69 x 36 mm): one plausible top-down pinch
    over the arch's highest point (guaranteed material in the column, seated 8 mm behind the
    tips), plus one of each drop class. The banana ARCS, so poses are anchored to the measured
    apex rather than the OBB centre — the centre of a curved object's box can be empty space."""
    asset = load_asset("banana")
    verts = asset.canonical_vertices()
    top = verts[int(np.argmax(verts[:, 2]))]
    x, y, z = (float(v) for v in top)
    answer = {"candidates": [
        # plausible: approach -z, TCP 8 mm below the apex, jaw across the thickness
        {"position": [x, y, z - 0.008], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [0.0, 1.0, 0.0], "width": 0.05, "label": "body",
         "rationale": "pinch across the arch apex, seated 8 mm behind the tips"},
        # gripping air: TCP 100 mm above the apex (material far in FRONT of the TCP)
        {"position": [x, y, z + 0.1], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [0.0, 1.0, 0.0], "width": 0.05, "label": "body", "rationale": "too high"},
        # buried: TCP 150 mm below the apex -> seat_depth < -0.05 -> make_candidate raises
        {"position": [x, y, z - 0.15], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [0.0, 1.0, 0.0], "width": 0.05, "label": "body", "rationale": "too deep"},
        # width outside the jaw
        {"position": [x, y, z - 0.008], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [0.0, 1.0, 0.0], "width": 0.12, "label": "body", "rationale": "too wide"},
        # jaw axis parallel to the approach
        {"position": [x, y, z - 0.008], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [0.0, 0.0, 1.0], "width": 0.05, "label": "body", "rationale": "degenerate"},
        # no material: far off the object laterally
        {"position": [0.0, 0.3, 0.0], "approach": [0.0, 0.0, -1.0],
         "jaw_axis": [1.0, 0.0, 0.0], "width": 0.05, "label": "body", "rationale": "off-object"},
    ]}
    cands, drops = derive_round(asset, answer, "a")
    print(f"banana selftest: {len(cands)} candidate(s), {len(drops)} drop(s)")
    for c in cands:
        print(f"  ok    {c.id}: seat_mode={c.seat_mode} span={c.span * 1000:.1f} mm "
              f"seat_depth={c.seat_depth * 1000:.1f} mm width={c.width * 1000:.0f} mm "
              f"labels={list(c.labels)}")
    for d in drops:
        print(f"  drop  {d['id']}: {d['reason']}")
    ok = (len(cands) == 1 and cands[0].id == "a00" and cands[0].seat_mode == "llm"
          and {d["id"] for d in drops} == {"a01", "a02", "a03", "a04", "a05"})
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="grasp_passes.llm_retry", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cycle", help="drive trigger -> rounds -> shake -> verdict per asset")
    c.add_argument("--asset", help="one catalog name (default: every triggered asset)")
    c.add_argument("--dry-run", action="store_true",
                   help=f"assemble + inspect the round-A request for one asset (default "
                        f"{DRY_RUN_DEFAULT}) with NO network call, then stop")
    c.add_argument("--device", default="cuda:0", help="render device for the view images")
    c.set_defaults(fn=cmd_cycle)

    s = sub.add_parser("status", help="per-asset trigger state, cached rounds, verdict")
    s.set_defaults(fn=cmd_status)

    t = sub.add_parser("selftest", help="authoring-validation path on hand-built poses (no LLM)")
    t.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
