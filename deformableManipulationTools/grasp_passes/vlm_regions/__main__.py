"""CLI for the ``vlm_regions`` pass — annotate, inspect, and (only here) bust the cache.

The harness CLI runs the pass the normal way::

    python -m deformableManipulationTools.grasp_passes run vlm_regions [--asset X]

This one adds what the harness deliberately has no vocabulary for: forcing a genuine re-annotation
(``--refresh``: new renders, new API call — the harness's ``--force`` only re-runs the pass, which
still reads the cached document), keeping the rendered views to look at, and printing a stored
annotation.

    python -m deformableManipulationTools.grasp_passes.vlm_regions list
    python -m deformableManipulationTools.grasp_passes.vlm_regions annotate --asset mug --refresh \
        --keep-renders /tmp/mug_views
    python -m deformableManipulationTools.grasp_passes.vlm_regions show mug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..base import PassContext
from ..catalog import asset_names, load_asset
from . import PASS, RIGID_KINDS, annotate_asset
from .regions import annotated_assets, load_regions, regions_path


def _targets(asset: str | None) -> list:
    if asset:
        return [asset]
    return [n for n in asset_names(RIGID_KINDS)]


def _cmd_list(args) -> int:
    done = set(annotated_assets())
    for name in _targets(None):
        doc = load_regions(name) if name in done else None
        if doc is None:
            print(f"  {name:24s} -")
        elif doc.regions:
            print(f"  {name:24s} {len(doc.regions)} region(s): "
                  + ", ".join(f"{r.label}({r.confidence:.2f})" for r in doc.regions))
        else:
            print(f"  {name:24s} none — {doc.empty_reason or 'no meaningful region'}")
    return 0


def _cmd_annotate(args) -> int:
    failed = 0
    for name in _targets(args.asset):
        print(f"[vlm_regions] {name}")
        try:
            ctx = PassContext(asset=load_asset(name))
            annotate_asset(ctx, refresh=args.refresh, model=args.model, device=args.device,
                           keep_renders=Path(args.keep_renders) / name if args.keep_renders
                           else None)
        except Exception as exc:                      # per-asset failure, keep going
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            failed += 1
    return 1 if failed else 0


def _cmd_show(args) -> int:
    doc = load_regions(args.asset)
    if doc is None:
        print(f"no regions recorded for {args.asset!r} ({regions_path(args.asset)})")
        return 1
    print(f"{doc.name} ({doc.kind})  mesh {doc.mesh_sha1[:12]}  model {doc.model} "
          f"prompt v{doc.prompt_version}")
    print(f"  extents [m]: {[round(float(e), 4) for e in doc.frame.extents]}"
          + ("  (frame AMBIGUOUS)" if doc.frame.ambiguous else ""))
    if not doc.regions:
        print(f"  no regions — {doc.empty_reason or '(no reason recorded)'}")
    for r in doc.regions:
        c = [round(float(x), 4) for x in r.center]
        print(f"  {r.id:12s} centre {c} r={r.radius * 100:.1f} cm  conf {r.confidence:.2f}  "
              f"views {list(r.views)}")
        print(f"               {r.rationale}")
    if doc.dropped:
        print(f"  dropped: {len(doc.dropped)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="grasp_passes.vlm_regions", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="every rigid catalog asset and its annotation state")

    a = sub.add_parser("annotate", help="render + ask the VLM (cached unless --refresh)")
    a.add_argument("--asset", help="one catalog name (default: every rigid asset)")
    a.add_argument("--refresh", action="store_true",
                   help="ignore the stored annotation and really re-render + re-ask")
    a.add_argument("--model", help=f"override the annotator model (default: {PASS.name}'s default)")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--keep-renders", help="directory to keep the rendered views in, for inspection")

    s = sub.add_parser("show", help="print a stored annotation")
    s.add_argument("asset")

    args = ap.parse_args(argv)
    return {"list": _cmd_list, "annotate": _cmd_annotate, "show": _cmd_show}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
