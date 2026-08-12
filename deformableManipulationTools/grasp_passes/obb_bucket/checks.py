"""Assertions for the bucketing rule and the runtime pruner, on synthetic frames + a real record.

    .venv/bin/python -m deformableManipulationTools.grasp_passes.obb_bucket.checks [asset]

Synthetic first (the geometry is checkable by hand), then the merged record of one catalog asset if
one exists — so a failure says whether the RULE broke or only its application to real data.
"""
from __future__ import annotations

import sys

import numpy as np

from ...grasp_library import FACE_BUCKETS, ObjectFrame, body_pose, has_grasps, load_grasps
from .bucket import buckets_from_labels, face_labels, plausible_buckets
from .pruning import bucket_scores, prune_record, surviving_buckets

_FAILED = []


def ok(what: str, cond, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        _FAILED.append(what)


def _frame(rotation=None, ambiguous: bool = False) -> ObjectFrame:
    return ObjectFrame(rotation=np.eye(3) if rotation is None else np.asarray(rotation, float),
                       translation=np.zeros(3), extents=np.array([0.2, 0.1, 0.05]),
                       axis_order=(0, 1, 2), sign_rules=("skew",) * 3, ambiguous=ambiguous,
                       drift=0.0)


def check_bucketing() -> None:
    print("bucketing")
    f = _frame()
    ok("a straight-down approach enters the +z face",
       plausible_buckets(f, [0, 0, -1]) == ("+z",))
    ok("a straight-up approach enters -z", plausible_buckets(f, [0, 0, 1]) == ("-z",))
    ok("a near-axis approach is NOT borderline",
       plausible_buckets(f, [0.2, 0, -1]) == ("+z",), "11 deg off the normal")
    diag = plausible_buckets(f, [0, 1, -1])
    ok("a 45 deg edge approach is borderline in both faces",
       set(diag) == {"+z", "-y"} and len(diag) == 2, f"{diag}")
    corner = plausible_buckets(f, [-1, 1, -1])
    ok("a body-diagonal approach is borderline in three faces",
       set(corner) == {"+x", "-y", "+z"}, f"{corner}")

    labs = face_labels(f, [0, 1, -1])
    ok("an exact tie takes the lowest-axis primary, as ObjectFrame.face_of does",
       labs[0] == "face:-y", f"{labs[0]}")
    ok("borderline labels carry the primary, the flag and the alternate",
       "face_borderline" in labs and "face_alt:+z" in labs, f"{labs}")
    ok("an unambiguous frame gets no ambiguity label", "face_ambiguous" not in labs)
    ok("an ambiguous frame labels every candidate",
       "face_ambiguous" in face_labels(_frame(ambiguous=True), [0, 0, -1]))
    ok("an ambiguous frame still buckets to exactly one primary",
       face_labels(_frame(ambiguous=True), [0, 0, -1])[0] == "face:+z")
    ok("labels round-trip to (primary, *alternates)",
       buckets_from_labels(labs) == ("-y", "+z"), f"{buckets_from_labels(labs)}")
    ok("candidates the pass never saw fall back to their stored face",
       buckets_from_labels((), "+x") == ("+x",))


def check_pruning() -> None:
    print("pruning")
    f = _frame()
    up = body_pose([0.4, 0.0, 0.1], 0.0)
    keep = surviving_buckets(f, up)
    ok("an upright object keeps every face but the one on the table",
       set(keep) == set(FACE_BUCKETS) - {"-z"}, f"{keep}")
    ok("the face pointing straight up ranks first", keep[0] == "+z")
    ok("yaw does not change which buckets survive",
       set(surviving_buckets(f, body_pose([0.4, 0, 0.1], 1.1))) == set(keep))

    flip = np.eye(4)
    flip[:3, :3] = np.diag([1.0, -1.0, -1.0])          # rolled 180 deg about x
    ok("flipping the object swaps the two z buckets",
       set(surviving_buckets(f, flip)) == set(FACE_BUCKETS) - {"+z"},
       f"{surviving_buckets(f, flip)}")
    ok("a top-down-only cone keeps only the up-facing bucket",
       surviving_buckets(f, up, half_angle_deg=20.0) == ("+z",))
    ok("a placement dict reads the same as a pose matrix",
       surviving_buckets(f, {"x": 0.4, "y": 0.0, "z": 0.1, "yaw": 0.0}) == keep)
    ok("the score of the dead-on face is 1", abs(bucket_scores(f, up)["+z"] - 1.0) < 1e-12)


def check_record(asset: str) -> None:
    print(f"record: {asset}")
    if not has_grasps(asset):
        print("  --    no merged record; run the passes first")
        return
    rec = load_grasps(asset, use_cache=False)
    up = body_pose([0.4, 0.0, 0.1], 0.7)
    kept = prune_record(rec, up)
    keep = set(surviving_buckets(rec.frame, up))
    ok("every candidate carries a bucket label",
       all(any(l.startswith("face:") for l in c.labels) for c in rec.candidates),
       f"{len(rec.candidates)} candidate(s)")
    ok("the stored face agrees with the labelled primary",
       all(buckets_from_labels(c.labels, c.face)[0] == c.face for c in rec.candidates))
    ok("pruning drops exactly the candidates in ruled-out buckets",
       {c.id for c in kept} == {c.id for c in rec.candidates
                                if keep.intersection(buckets_from_labels(c.labels, c.face))},
       f"kept {len(kept)}/{len(rec.candidates)} over buckets {sorted(keep)}")
    ok("pruning never invents a candidate",
       {c.id for c in kept}.issubset({c.id for c in rec.candidates}))
    ok("an all-directions cone prunes nothing",
       len(prune_record(rec, up, half_angle_deg=180.0)) == len(rec.candidates))


def main() -> None:
    asset = sys.argv[1] if len(sys.argv) > 1 else "banana"
    check_bucketing()
    check_pruning()
    check_record(asset)
    print(f"\n{'FAILED: ' + ', '.join(_FAILED) if _FAILED else 'all checks passed'}")
    raise SystemExit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
