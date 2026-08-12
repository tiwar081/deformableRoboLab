"""Runtime pruning: rule out whole OBB face buckets from a placement, with six dot products.

This is the cheap gate that runs BEFORE anything expensive — before IK, before a reachability sweep,
before a sim rollout. An object lying on the table presents its faces in a fixed way, and the faces
pointing down into the table can never be entered by the gripper no matter what the arm does. Since
every candidate carries its face bucket, ruling out a bucket rules out all of its candidates at
once, without touching a single grasp pose.

The whole test is::

    n_world = R_world_from_body @ R_asset_from_canonical @ n_face      # the face's outward normal
    entry   = -n_world                                                 # the way the TCP travels in
    keep    = dot(entry, approach_dir) >= cos(half_angle)

with the default ``approach_dir`` pointing straight down and a 90 deg half-angle: keep every face
whose outward normal is not below horizontal. Six normals, six dots, per object per placement.

**It is a filter, never a selector.** Surviving a bucket says nothing about whether a grasp is
reachable, collision-free or good — only that it is not disqualified by which way the object landed.
And it is conservative in the one place it could be wrong: a borderline candidate (a diagonal
approach, see :mod:`.bucket`) is dropped only when every bucket it plausibly belongs to is ruled
out, so the coarse gate can never remove a grasp the expensive stage would have accepted.
"""
from __future__ import annotations

import math

import numpy as np

from ...grasp_library import FACE_BUCKETS, GraspRecord, ObjectFrame, body_pose
from .bucket import buckets_from_labels

# Default reachable-approach cone: straight down, half-angle 90 deg. The robot can enter a face from
# above or from any horizontal direction; it cannot come up from under the table. Widen the angle
# past 90 to also keep faces tilted slightly below horizontal (a settled object rarely presents an
# exactly horizontal face), narrow it for a top-down-only policy.
DOWN = (0.0, 0.0, -1.0)
DEFAULT_HALF_ANGLE_DEG = 90.0
_COS_EPS = 1.0e-9            # slack on the cone test (see surviving_buckets)

_EYE = np.eye(3)


def placement_matrix(placement) -> np.ndarray:
    """``world_from_body`` (4x4) from whatever form the caller has the placement in.

    Accepts a 4x4 matrix, a ``(pos, yaw)`` pair, a bare position, or a ``scene.json`` object entry
    (``{"x":…, "y":…, "z":…, "yaw":…}``) — the three shapes a placement actually shows up in around
    this codebase. Everything routes through :func:`grasp_library.body_pose`, so the convention
    matches how ``assets.add_ycb_mesh`` spawns a body."""
    if isinstance(placement, dict):
        pos = (float(placement.get("x", 0.0)), float(placement.get("y", 0.0)),
               float(placement.get("z", 0.0)))
        return body_pose(pos, float(placement.get("yaw", 0.0)))
    if isinstance(placement, (tuple, list)) and len(placement) == 2 \
            and np.ndim(placement[0]) == 1 and np.ndim(placement[1]) == 0:
        return body_pose(placement[0], float(placement[1]))
    m = np.asarray(placement, dtype=float)
    if m.shape == (4, 4):
        return m
    if m.shape == (3,):
        return body_pose(m, 0.0)
    raise ValueError(f"cannot read a placement from shape {m.shape}; expected a 4x4 pose, a "
                     f"(pos, yaw) pair, a position, or a scene-object dict")


def bucket_normals_world(frame: ObjectFrame, placement) -> dict:
    """``{face bucket: outward normal in world}`` for the six faces under a placement."""
    rot = placement_matrix(placement)[:3, :3] @ frame.rotation.T   # world <- canonical
    out = {}
    for k in range(3):
        for sign in (1.0, -1.0):
            out[f"{'+' if sign > 0 else '-'}{'xyz'[k]}"] = rot @ (sign * _EYE[k])
    return out


def bucket_scores(frame: ObjectFrame, placement, *, approach=DOWN) -> dict:
    """``{face bucket: dot(entry direction, approach_dir)}`` — the raw number the gate thresholds.

    Exposed because a caller that wants to ORDER buckets (most face-on first) should reuse this
    rather than recompute the geometry; 1.0 means the face is entered dead-on."""
    a = np.asarray(approach, dtype=float).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1.0e-12)
    return {face: float(-n @ a) for face, n in bucket_normals_world(frame, placement).items()}


def surviving_buckets(frame: ObjectFrame, placement, *, approach=DOWN,
                      half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> tuple:
    """The face buckets still worth considering for an object at ``placement``, best-scoring first.

    Returns a subset of :data:`grasp_library.FACE_BUCKETS`. Empty is possible in principle (a very
    narrow cone on an awkward pose) and means "no face of this object is presented to the gripper" —
    a real answer, not an error."""
    # Compared with a tolerance, and that is not cosmetic: at the default 90 deg the limit is
    # cos(pi/2) = 6e-17 rather than 0, so a face sitting EXACTLY horizontal — the common case for a
    # box on a table — would fall on the wrong side of the test and take its grasps with it. The
    # gate errs toward keeping, always: a survivor costs one expensive check, a wrong rejection
    # costs a grasp nothing downstream can get back.
    cos_lim = math.cos(math.radians(float(half_angle_deg))) - _COS_EPS
    scores = bucket_scores(frame, placement, approach=approach)
    keep = [f for f in FACE_BUCKETS if scores[f] >= cos_lim]
    return tuple(sorted(keep, key=lambda f: (-scores[f], FACE_BUCKETS.index(f))))


def prune_candidates(candidates, frame: ObjectFrame, placement, *, approach=DOWN,
                     half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> tuple:
    """Candidates whose face bucket survives ``placement`` — the pre-filter, applied.

    A candidate is kept when ANY bucket it plausibly belongs to survives: for an ordinary grasp that
    is just its stored ``face``; for one the bucket pass flagged borderline it is that plus the
    ``face_alt:`` runner-up(s) (see :mod:`.bucket`). Candidates the bucket pass never saw fall back
    to their stored ``face``, so this works on a record merged from any set of passes."""
    keep = set(surviving_buckets(frame, placement, approach=approach,
                                 half_angle_deg=half_angle_deg))
    return tuple(c for c in candidates
                 if keep.intersection(buckets_from_labels(getattr(c, "labels", ()), c.face)))


def prune_record(record: GraspRecord, placement, *, approach=DOWN,
                 half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG) -> tuple:
    """:func:`prune_candidates` for a whole record, taking the frame from the record itself."""
    return prune_candidates(record.candidates, record.frame, placement, approach=approach,
                            half_angle_deg=half_angle_deg)
