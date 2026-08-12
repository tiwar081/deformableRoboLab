"""``shake_validate``'s own entry point — the self-check, and a single trial for tuning.

    .venv/bin/python -m deformableManipulationTools.grasp_passes.shake_validate --selfcheck
    .venv/bin/python -m deformableManipulationTools.grasp_passes.shake_validate --trial banana

Running the PASS (and writing its sidecar) is the shared harness's job, not this one's:

    .venv/bin/python -m deformableManipulationTools.grasp_passes run shake_validate --asset banana
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _trial(asset_name: str, only: str | None) -> None:
    from ..base import read_sidecar
    from ..catalog import load_asset
    from .rig import run_trial

    side = read_sidecar("fixture", asset_name)
    if side is None:
        raise SystemExit(f"no upstream sidecar for {asset_name!r}; run the 'fixture' pass first")
    asset = load_asset(asset_name)
    print(f"{asset_name}  kind={asset.kind}  extents={np.round(asset.extents, 4).tolist()}")
    for candidate in side["record"].candidates:
        if only and only not in candidate.id:
            continue
        world_from_grasp = asset.frame.inverse_matrix() @ np.asarray(candidate.transform, dtype=float)
        started = time.time()
        r = run_trial(asset, world_from_grasp)

        def mm(v):
            return "   null" if v is None else f"{v * 1000:7.2f}"

        print(f"  {candidate.id:34s} width={candidate.width * 1000:5.1f} mm  face={candidate.face}")
        print(f"     in_gripper={r.object_in_gripper:.0f}  closing({mm(r.motion_closing_linear)} mm,"
              f"{mm(r.motion_closing_angular)} mrad)  shaking({mm(r.motion_shaking_linear)} mm,"
              f"{mm(r.motion_shaking_angular)} mrad)")
        print(f"     target {r.force_target:.2f} N on {r.object_mass * 1e3:.1f} g "
              f"(mu_eff {r.mu_effective:.3f}) closed in {r.close_duration:.1f} s"
              f"{'' if r.grip_converged else ' (TIMEOUT)'}, "
              f"final squeeze {r.final_squeeze:.2f} N, shake applied "
              f"at {r.shake_ratio_linear * 100:.0f}%/{r.shake_ratio_angular * 100:.0f}% of command "
              f"[{time.time() - started:.0f} s]")
        print(f"     {r.note}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="shake_validate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true",
                    help="assert the rig's geometry, schedule and metric claims (no full trial)")
    ap.add_argument("--trial", metavar="ASSET",
                    help="run this catalog object's upstream candidates and print the result")
    ap.add_argument("--candidate", default=None, help="with --trial: only ids containing this")
    args = ap.parse_args()
    if args.selfcheck:
        from .selfcheck import run_selfcheck
        run_selfcheck()
    if args.trial:
        _trial(args.trial, args.candidate)
    if not args.selfcheck and not args.trial:
        ap.error("give --selfcheck and/or --trial ASSET")


if __name__ == "__main__":
    main()
