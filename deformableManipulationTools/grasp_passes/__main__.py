"""SHARED — the grasp-pass CLI. Read-only for pass authors.

    .venv/bin/python -m deformableManipulationTools.grasp_passes list
    .venv/bin/python -m deformableManipulationTools.grasp_passes run fixture [--asset banana ...]
    .venv/bin/python -m deformableManipulationTools.grasp_passes run fixture --check-idempotent
    .venv/bin/python -m deformableManipulationTools.grasp_passes run-all [--force]
    .venv/bin/python -m deformableManipulationTools.grasp_passes merge [--asset banana ...]
    .venv/bin/python -m deformableManipulationTools.grasp_passes show banana
"""
from __future__ import annotations

import argparse

import numpy as np

from ..grasp_library import has_grasps, load_grasps
from . import discover_passes, get_pass, ordered_passes
from .base import PassError, check_idempotent, run_pass, sidecar_assets
from .catalog import asset_names, load_asset
from .merge import merge_all, merge_asset, merged_assets


def _targets(pass_, requested) -> list:
    if requested:
        return list(requested)
    return asset_names(pass_.kinds or None)


def cmd_list(args) -> None:
    passes = discover_passes(verbose=True)
    if not passes:
        print("no passes found (a pass is a subdirectory exporting PASS)")
        return
    order = [p.name for p in ordered_passes(list(passes.values()))]
    print(f"{len(passes)} pass(es), in run order:")
    for name in order:
        p = passes[name]
        done = sidecar_assets(name)
        print(f"  {p.name:16s} source={p.source:16s} v{p.version}"
              f"  kinds={list(p.kinds) or 'all'}"
              f"  requires={list(p.requires) or '-'}"
              f"  sidecars={len(done)}")


def cmd_run(args) -> None:
    passes = [get_pass(n) for n in args.passes] if args.passes else ordered_passes()
    for p in passes:
        targets = _targets(p, args.asset)
        print(f"[{p.name}] {len(targets)} asset(s)")
        if args.check_idempotent:
            bad, checked = [], 0
            for name in targets:
                verdict = check_idempotent(p, name)
                if verdict is None:
                    continue                       # pass does not apply to this asset
                checked += 1
                if verdict:
                    print(f"  ok    {name} idempotent")
                else:
                    bad.append(name)
                    print(f"  FAIL  {name}: two runs produced different sidecars")
            print(f"  {checked} applicable asset(s) checked")
            if bad:
                raise SystemExit(f"{p.name}: not deterministic on {bad}")
            continue
        for name in targets:
            try:
                run_pass(p, name, force=args.force)
            except PassError as exc:
                print(f"  FAIL  {name}: {exc}")
            except Exception as exc:                  # noqa: BLE001 - one asset must not stop the rest
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    if not args.no_merge:
        print("[merge]")
        merge_all()


def cmd_merge(args) -> None:
    if args.asset:
        for name in args.asset:
            merge_asset(name)
    else:
        merge_all()


def cmd_selfcheck(args) -> None:
    from .selfcheck import run_selfcheck
    run_selfcheck()


def cmd_show(args) -> None:
    name = args.asset
    asset = load_asset(name)
    print(f"{name}  kind={asset.kind}  file={asset.file}")
    print(f"  OBB extents {np.round(asset.extents, 4).tolist()}  ambiguous={asset.frame.ambiguous}"
          f"  drift={asset.frame.drift:.2e} m  rules={asset.frame.sign_rules}")
    if not has_grasps(name):
        print("  no merged record yet")
        return
    rec = load_grasps(name, use_cache=False)
    if not rec.candidates:
        print(f"  record: OUT OF REACH for this gripper — no generator found a graspable pose")
        print(f"          {rec.notes}")
        return
    print(f"  record: {len(rec.candidates)} candidate(s)  [{rec.generator}]")
    for c in rec.candidates:
        q = "-" if not c.evaluated else f"{c.quality_source}"
        print(f"    {c.id:34s} face={c.face:2s} width={c.width * 1000:5.1f} mm  "
              f"pos={np.round(c.position, 3).tolist()}  quality={q}"
              + (f"  {list(c.labels)}" if c.labels else ""))


def main() -> None:
    ap = argparse.ArgumentParser(prog="grasp_passes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show discovered passes and their run order").set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="run one or more passes, then merge")
    r.add_argument("passes", nargs="*", help="pass names (default: all, in dependency order)")
    r.add_argument("--asset", action="append", help="limit to these catalog objects (repeatable)")
    r.add_argument("--force", action="store_true", help="recompute even if the sidecar is current")
    r.add_argument("--check-idempotent", action="store_true",
                   help="run twice and diff the sidecars instead of writing once")
    r.add_argument("--no-merge", action="store_true", help="skip the merge afterwards")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("run-all", help="alias for 'run' over every discovered pass")
    m.add_argument("--asset", action="append")
    m.add_argument("--force", action="store_true")
    m.add_argument("--check-idempotent", action="store_true")
    m.add_argument("--no-merge", action="store_true")
    m.set_defaults(fn=cmd_run, passes=[])

    g = sub.add_parser("merge", help="compose sidecars into assets/objects/grasps/<name>.json")
    g.add_argument("--asset", action="append", help="limit to these catalog objects (repeatable)")
    g.set_defaults(fn=cmd_merge)

    sub.add_parser("selfcheck",
                   help="prove the pass contract (isolation, dedup, annotations) on temp sidecars"
                   ).set_defaults(fn=cmd_selfcheck)

    s = sub.add_parser("show", help="print an asset's frame and merged candidates")
    s.add_argument("asset")
    s.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
