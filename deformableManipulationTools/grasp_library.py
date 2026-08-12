"""Per-asset grasp candidates: the canonical object frame, the record schema, and the loader.

Grasp candidates are PRECOMPUTED OFFLINE per catalog object and looked up at task/trajectory
generation time. This module owns the three things that must be shared by the offline generator and
every runtime consumer:

  1. **The canonical object frame** (:func:`canonical_frame`) — an oriented bounding box, reduced to
     ONE reproducible frame per mesh. Candidates are stored as poses in this frame, so a runtime
     placement (any position + yaw, and the ``rest_on_z`` lift) transforms them by composition and
     nothing has to be regenerated per scene.
  2. **The record schema** (:class:`GraspRecord`) — one JSON sidecar per catalog object at
     ``assets/objects/grasps/<catalog_name>.json``.
  3. **The loader + validation** (:func:`load_grasps`, :func:`validate_record`) — fails loudly, in
     the style of ``agentic_pipeline.load_goal_predicates``: a malformed record raises rather than
     silently yielding a grasp that points the wrong way.

Candidate GENERATION is deliberately not here — this module defines and guards the data.

**SCOPE: rigid objects and soft-bodied (FEM) objects. NOT cloth, NOT bags, and NOT cables
(2026-08-11).** The scheme rests on a canonical frame fitted to the asset's REST geometry, with
candidates stored as poses in it and composed with the runtime placement. That is sound only while
the object's runtime shape stays close to the geometry the box was fitted to — true of a rigid
mesh and a ``.tet`` FEM body. A garment or bag (``kind: "cloth"``, including ``category: "bag"``)
has no persistent rest shape: its source mesh is an authored flat shell that settles differently
every run, so its OBB describes the asset file rather than anything the gripper will meet, and
there is no rigid body transform to compose against — it is particles, not a body. Cloth grasp
targets must be resolved against the LIVE particle state (a rule such as "the +x edge, 11 cm in" —
the ``cloth_franka`` recipe), which is a different mechanism, not a parameter of this one. CABLES
(``kind: "cable"``) are excluded on the measured record rather than in principle: the rest-shape
premise failed twice — the geometry probe recovered the synthetic capsule chain's own construction
axis (108 candidates, one grasp up to translation/roll), and the 2026-08-08 full-catalog shake held
0/62 cable trials at rest-shape spans. Re-admitting cables needs a generator that reads REAL cable
geometry and measures its span UNDER LOAD; until then the pipeline is rigid + squishy only
(docs/trajPipeline/grasp-library.md).

Record layout — mirrors the ACRONYM dataset (NVIDIA, 17.7M simulated parallel-jaw grasps), whose
HDF5 files carry grasp transforms alongside per-grasp physics-simulation quality fields, which is
the structure we need. Mapping, ours ← theirs:

    object.file            ← object/file            asset path (ours: relative to assets/objects)
    object.scale           ← object/scale
    object.frame/extents   ← (none — ACRONYM stores poses in the raw mesh frame)
    gripper.type           ← gripper/type           'panda' there, 'franka_panda' here
    candidate.width        ← gripper/configuration  (per-grasp here; one value per file there)
    candidate.transform    ← grasp/transforms[i]    4x4 homogeneous, object frame ← grasp frame
    candidate.quality.*    ← grasps/qualities/flex/*  same five metric names, verbatim
    candidate.quality_source ← the 'flex' namespace  (which simulator produced the numbers)

Two deliberate departures: poses are stored in the CANONICAL frame rather than the raw mesh frame
(ACRONYM's meshes are already canonically posed by ShapeNetSem; our catalog's are not), and the
qualities start EMPTY — they are filled by a later Newton/VBD grasp-evaluation pass, not by FleX.

**Frame conventions** (both are versioned strings in the record, so a convention change is
detectable rather than silent):

``FRAME_CONVENTION`` — the canonical object frame. Origin at the OBB centre; axes are the OBB axes
sorted by extent DESCENDING (x = longest ... z = shortest, so the ±z faces are the object's two
LARGEST faces); signs resolved from mesh asymmetry. See :func:`canonical_frame`.

``POSE_CONVENTION`` (**v2**) — the grasp frame a candidate's ``transform`` places. Origin at the TCP
(``FRANKA.ee_offset``, the point ``WP.pos`` commands); ``+z`` = approach (the direction the TCP
travels into the grasp); ``+x`` = jaw closing axis; ``+y`` completes a right-handed frame. The axis
convention matches ACRONYM's panda marker (±x jaws, +z approach); the ORIGIN does not — theirs sits
at the hand base, :data:`ACRONYM_HAND_TO_TCP` metres behind ours along +z.

**v2 means the stored pose is PAD-SEATED and command-ready**: the object's material has already
been seated against the fingers by :func:`pad_seat`, so a consumer drives the TCP straight to
``candidate.transform`` with no offset math. This is not cosmetic. The TCP sits at the FINGERTIP
TIPS and the pads run from :data:`PAD_NEAR_Z` back to :data:`PAD_FAR_Z` *behind* it, so a pose
placed on the object's material — which is what v1 stored — grips with the very tips and leaves
about half the object forward of the fingers. Every generator must seat its poses through
:func:`pad_seat` before storing them; where the pads are is a property of the hand, not of how a
grasp was found, so it is applied in exactly one place.

Precisely, a stored pose carries a ``seat_mode`` (:data:`SEAT_MODES`, a first-class schema field)
naming which rule placed it, plus its measured ``span`` (the material extent along the approach).
The depth rule is ONE rule — as deep as possible subject to two caps — and the mode names which
cap bound:

* **span_flush** (the FINGERTIP cap bound) — the whole span sits flush against the fingertip end
  of the pads (far material edge at :data:`PAD_NEAR_Z`), i.e. "grip exactly ``span`` far along
  the object; the fingertips do not extend past it". This replaced the retired CENTRED rule
  (span midpoint on :data:`PAD_MID_Z`), which commanded the tips ``27.6 − span/2`` mm past the
  object's far surface — for a top-down grasp on a resting object that is INTO THE TABLE
  (9.6 mm on the 36 mm banana), which no online check could see (clearance samples only the
  corridor above the pose; the shake rig has no table).
* **clamped_deep** (the PALM cap bound) — **"jaws as deep as safe"**: the near material sits at
  :data:`SEAT_DEEPEST_Z` (palm face plus clearance) and the rest of the object protrudes forward
  past the fingertips. Centring an over-deep span would push its overhang backwards INTO the
  palm — measured 2026-08-06: on the 62.9 mm apple, midpoint seating put all 116 candidates in
  palm collision. (The centred rule also violated this cap outright for 44–54 mm spans, which
  survived only by falling through to the retreat.)
* **retreated** — the rule seat above put the measured HAND HULLS inside the object, so the pose
  backed off along the approach to the deepest collision-free depth that still leaves material
  between the pads. The pads hold only the near material (a shallow/tip grasp). Measured need
  (2026-08-06 depth sweep): on bodies wider than the 80 mm jaw stroke (mug 88–94 mm, bowl 152 mm,
  pitcher ~140 mm) every centred/clamped seat lays a finger or the 204 mm-wide palm across the
  object laterally, and the ONLY collision-free poses on the stored approach are 24–51 mm
  shallower. A pose whose approach clears at NO depth keeps its rule seat and is labelled
  ``seat_blocked`` by its generator — marked, not deleted, like every other measured failure.
  The retreat distance is BAKED INTO the stored transform (the pose is final and command-ready);
  ``seat_mode`` records how the depth was chosen, not an adjustment still to be applied.

NOTE on dependencies: reading records stays stdlib+numpy, but SEATING now depends on the ROBOT
BUILD — :func:`pad_seat`'s hand-clearance retreat tests the hand's own collision hulls
(:func:`hand_volumes`, measured off the active robot via ``robot.py``, lazily, once per process).

Runtime composition — an asset's body frame IS its baked mesh frame (``assets.add_ycb_mesh`` spawns
at ``transform(pos, quat_z(yaw))``), so::

    T_world_from_grasp = T_world_from_body @ record.asset_from_canonical() @ candidate.transform

:meth:`GraspRecord.grasp_in_world` does exactly this. Stdlib + numpy only (trimesh is imported
lazily, inside the OBB utility) so reading candidates never pulls warp/newton/trimesh.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import FRANKA
from .settings import REPO_ROOT

OBJECTS_DIR = REPO_ROOT / "assets" / "objects"
GRASPS_DIR = OBJECTS_DIR / "grasps"

# v2 (2026-08-06): candidates carry a REQUIRED ``seat_mode`` field (see SEAT_MODES). v1 records
# recorded containment only in per-generator prose, which downstream code had to string-match — the
# version bump makes every v1 record fail loudly with "regenerate" instead of loading with the
# field silently absent.
# v3 (2026-08-07): the CENTRED seat rule is retired for the tip-capped ``span_flush`` rule (a
# centred seat commands the fingertips past the object's far surface — into the table for a
# top-down grasp on a resting object), and candidates carry a REQUIRED ``span`` field (the
# measured material extent along the approach [m]) so consumers can reason about how much object
# is between the pads. v2 records fail loudly with "regenerate": their poses are up to
# 27 mm deeper than a v3 seat and nothing downstream could detect it.
# v4 (2026-08-07): candidates carry a REQUIRED ``seat_depth`` (grasp-frame z of the near material
# edge — with ``span`` it locates the material for every mode, including retreated), and the hand
# collision checks (pad_seat retreat + shake pre-check) pose the fingers at the PRE-SHAPED
# aperture ``width + PREGRASP_MARGIN`` instead of fully open, matching the trajectory that will
# execute. Retreat/blocked outcomes move, so v3 records fail loudly with "regenerate".
# v5 (2026-08-07): DEPTH-VARIANT PAIRS. For a span the palm allows, the CENTRED seat (measured
# 25% hold vs flush's 4.2% under a lift load) is stored as a SEPARATE candidate (id suffix
# "_ctr", label "centred_variant", seat_mode "centred") alongside the always-executable
# span_flush primary; both are collision-checked and shake-validated offline, and ONLINE
# selection picks between the pair by whether anything occupies the space beyond the object's
# far surface (grasp_select "depth" stage). "centred" re-enters SEAT_MODES with v2-era
# semantics; old records are still rejected by the schema_version gate.
SCHEMA_VERSION = 5
FRAME_CONVENTION = "obb_extent_desc_v1"
POSE_CONVENTION = "tcp_z_approach_x_jaw_v2"
# The convention this replaced. v1 differed ONLY in that a stored pose was wherever its generator
# put it along the approach — usually on the material centre, which seats the object on the pad TIPS
# (see PAD_NEAR_Z). v2 poses are PAD-CENTRED and command-ready. The two are indistinguishable by
# inspection, so validate_record rejects v1 by name rather than letting a half-regenerated library
# merge into one record with grasps meaning two different things.
POSE_CONVENTION_V1 = "tcp_z_approach_x_jaw_v1"
GRIPPER_TYPE = "franka_panda"

# Max PARALLEL-JAW opening [m]: twice the per-finger prismatic limit. THE definition — imported by
# agentic_pipeline.task_generator rather than re-derived there.
MAX_JAW_WIDTH = 2.0 * FRANKA.gripper_open

# Pre-grasp aperture margin [m]. The hand does NOT arrive fully open: the trajectory pre-shapes
# the jaws to the candidate's width plus this margin before the approach, so the collision check
# (:func:`pregrasp_collision`, :func:`pad_seat`'s retreat) and the validation rig both pose the
# fingers at ``min(width + PREGRASP_MARGIN, MAX_JAW_WIDTH)`` — the profile that actually executes.
# 10 mm = 5 mm of blind travel per finger during the close; checking at full 80 mm rejected poses
# whose only "collision" was an open finger sweeping space the pre-shaped hand never occupies.
PREGRASP_MARGIN = 0.010

# ACRONYM's grasp frame sits at the hand base; the pad tips (our TCP) are this far along +z, read off
# their gripper marker (finger cylinders span z = 0.066 .. 0.11217). Recorded so ACRONYM transforms
# can be converted into this schema later: T_ours = T_acronym @ translate(0, 0, ACRONYM_HAND_TO_TCP).
ACRONYM_HAND_TO_TCP = 0.11217

# =================================================================================================
# Pad geometry — where the fingers actually are, relative to the grasp frame origin
# =================================================================================================
# MEASURED off the active robot (2026-08-05): build_franka_robot at home_q, each finger collider
# transformed into the POSE_CONVENTION grasp frame. On the panda USD a finger is ONE convex mesh, so
# this is the whole finger's contact extent, not just the rubber patch.
#
# The TCP (FRANKA.ee_offset, the point every waypoint commands) sits at the FINGERTIP TIPS, and all
# pad material lies BEHIND it along the approach. That is the fact every generator has to respect:
# a grasp centre placed on the object's material leaves roughly half the object forward of the tips,
# gripped by the very end of the pads. See pad_seat().
PAD_NEAR_Z = -0.0007          # [m] fingertip tip, along +approach relative to the grasp centre
PAD_FAR_Z = -0.0545           # [m] pad root (toward the hand)
PAD_MID_Z = 0.5 * (PAD_NEAR_Z + PAD_FAR_Z)      # -0.0276 m — where a grasped object should sit
PAD_LENGTH = PAD_NEAR_Z - PAD_FAR_Z             #  0.0538 m — the most depth the pads can enclose
# Forward face of the PALM along the approach [m] (measured off the same colliders: the palm hull
# spans z = -138.7..-47.0 in the grasp frame). NOTE it sits IN FRONT of the pad root: the last
# 7.5 mm of pad length overlaps the palm's z-range, so "inside the pads" and "clear of the palm"
# are different constraints and the DEEP-SEAT CLAMP below must use this one, not PAD_FAR_Z.
PALM_NEAR_Z = -0.0470
# Clearance between the nearest object material and the palm face when deep-seating [m]. Keeps the
# seated pose out of the pre-grasp collision check's tolerance band instead of exactly on it.
PALM_CLEARANCE = 0.002
# The deepest the near edge of the object may sit: flush with the palm face plus the clearance.
SEAT_DEEPEST_Z = PALM_NEAR_Z + PALM_CLEARANCE   # -0.0450 m
# Half-width of the ray grid across the third axis when probing for material [m]: the pad footprint.
PAD_HALF_WIDTH = 0.010
_SEAT_RAYS = 5                # grid resolution per axis of the jaw column (5x5 = 25 rays)
# Column growth for the vertex fallback (see _span_by_vertices) — same figures the other local
# measurements in this repo use, because they exist for the same reason: mesh density varies ~100x.
_SEAT_MIN_SAMPLES = 8
_SEAT_MAX_GROWTH = 6
_SEAT_GROWTH = 1.6

# The ways a v2 pose can sit along its approach — a REQUIRED per-candidate schema field, in
# order of preference: fully enclosed, flush against the fingertip end; as deep as the palm
# allows; backed off to clear the hand. See pad_seat() for when each applies and the module
# docstring for what each promises. "llm" (2026-08-11) is the odd one out: the pose was placed by
# the llm_retry stage EXACTLY where the LLM put it — no seat rule ran, no retreat; span/seat_depth
# are MEASURED at that pose via measure_span_at(). See docs/trajPipeline/llm-retry.md.
SEAT_MODES = ("span_flush", "centred", "clamped_deep", "retreated", "llm")
# Label a generator stamps on a candidate whose approach clears the hand at NO depth
# (PadSeat.blocked). One shared spelling, so consumers filter one token, not one per generator.
# Since 2026-08-11 the MERGE DISCARDS these from the record (no collision-free grasp exists to
# command); the generator sidecar keeps the measurement. Generators still emit + label.
SEAT_BLOCKED_LABEL = "seat_blocked"

# ---- candidate statuses (2026-08-11, docs/trajPipeline/grasp-library.md "Candidate statuses") --
# A `retreated` candidate is a WEAK GRASP OPTION, not a legitimate candidate: the merge stamps
# this label on it; it is excluded from physics testing (shake_validate v4), from grasp_select's
# default pool, and from the "does this object have a working grasp" accounting (measured basis:
# 3% hold at n=628). Kept in the record as reachability information + llm_retry raw material.
WEAK_GRASP_LABEL = "weak_grasp_option"
# Labels shake_validate stamps on candidates it skipped (no trial ran, quality NOT filled). Named
# here because the coverage accounting below reads them; the spelling predates the constant.
SHAKE_SKIPPED_LABEL = "shake_skipped"
# Token the llm_retry pass puts in its sidecar notes once BOTH its rounds have run (the retry is
# bounded — one blind round, one visual-feedback round, ever). The merge reads it.
RETRY_EXHAUSTED_TOKEN = "llm-retry-exhausted"
# Record-level verdict, appended to record notes by the MERGE when llm_retry is exhausted and the
# record still has zero held legitimate candidates. Like OUT_OF_REACH it is a durable RESULT:
# downstream stages treat the object as having no grasp, and needs_llm_retry() goes False.
UNUSABLE_TOKEN = "UNUSABLE for this gripper"
UNUSABLE_NOTE = (
    UNUSABLE_TOKEN + ": every generator ran, physics validation covered the result, and the LLM "
    "retry stage exhausted both its rounds without a single held grasp. This is a measured "
    "verdict, not a failed run. Re-check if the gripper, the mesh, or MAX_JAW_WIDTH changes "
    "(re-arm by deleting the llm_retry sidecar)."
)
# Step of the collision-aware retreat along the approach [m]. 1 mm resolves the clear windows the
# 2026-08-06 depth sweep measured (3–16 mm wide, median 8 on the mug) without stepping over them.
# The retreat grid is anchored to the RULE seat — a function of the geometry alone — so re-seating
# a stored retreated pose lands on the same absolute depths and is idempotent.
SEAT_RETREAT_STEP = 0.001

# ---- hand clearance (shared with the shake_validate pre-check) ----------------------------------
# Material may penetrate a hand volume by this much before it counts as a collision [m]. Absorbs
# mm-scale mesh noise so a grasp that merely grazes a pad face is not rejected or retreated.
PENETRATION_TOL = 0.002
# Vertices inside a volume before it counts. One stray vertex on a scanned mesh is noise; a hand
# genuinely buried in an object puts many inside.
MIN_VERTICES = 3

# The six OBB face buckets, named by the canonical-frame outward normal they face along.
FACE_BUCKETS = ("+x", "-x", "+y", "-y", "+z", "-z")

# Catalog kinds this scheme does NOT model — see the SCOPE paragraph in the module docstring. Bags
# are built as ``kind: "cloth"`` (with ``category: "bag"``), so the one entry covers both. Cables
# joined the list 2026-08-11 on the measured record (0/62 held; the probe read the synthetic rod's
# construction axis, not geometry). Enforced in validate_record so a record can never be written
# for one by accident.
UNSUPPORTED_KINDS = ("cloth", "cable")

# Per-grasp quality metrics — ACRONYM's ``grasps/qualities/flex/*`` names verbatim. Every one starts
# None ("not measured"); a later evaluation pass fills them and names itself in ``quality_source``.
QUALITY_FIELDS = (
    "object_in_gripper",                    # binary: did the object survive the grasp
    "object_motion_during_closing_linear",  # [m]
    "object_motion_during_closing_angular", # [rad]
    "object_motion_during_shaking_linear",  # [m]
    "object_motion_during_shaking_angular", # [rad]
)

# Canonicalization tolerances (see canonical_frame).
_EXTENT_TIE_RTOL = 0.01     # extents within 1% of the largest are a tie -> the frame is ambiguous
_SKEW_TOL = 1.0e-3          # normalized third moment below this carries no sign information
# Angular precision of trimesh's OBB search. MEASURED on the YCB banana (10.7k verts), comparing the
# box found for the mesh against the box found for the same mesh randomly re-posed: the trimesh
# DEFAULT of 1 leaves up to 2.0 deg of box wobble (1.7 mm of canonical-frame drift), 2 leaves 0.4 deg,
# and 3 is exact (0.000 deg, identical extents). Cost 1.1 s -> 7.6 s per mesh, paid ONCE offline per
# asset — cheap next to the ~30 s coacd decomposition, and a wobbling frame would move every stored
# grasp. Do not lower this to speed up a batch regeneration.
_OBB_ANGLE_DIGITS = 3
# Frame drift above this [m] means the OBB minimum is near-degenerate -> the frame is ambiguous.
# 0.1 mm: well under any grasp tolerance, well over the ~1e-16 a well-conditioned mesh achieves.
_DRIFT_TOL = 1.0e-4
# Fixed probe rotations for the drift measurement — deliberately NOT random, so a record's reported
# drift is reproducible. Axis-angle pairs chosen to be unaligned with any axis or with each other.
# NOTE this is a DETECTOR, not a certified bound: a near-degenerate box can have several near-tied
# optima and finite probes only sample some of them (the YCB mug reports 1.7 mm here while wider
# random probing finds 10 cm). That is fine for its job — any value over _DRIFT_TOL means "this
# frame's orientation is not intrinsically determined", and the magnitude is indicative only.
_DRIFT_PROBES = (((0.3, 0.5, 0.81), 0.7), ((-0.6, 0.74, 0.3), 2.1), ((0.9, -0.15, 0.41), 1.3))
_ORTHO_TOL = 1.0e-6         # rotation-matrix validation tolerance
_ROUND = 9                  # JSON decimal places (diff-stable files, well inside _ORTHO_TOL)


class GraspSchemaError(ValueError):
    """A grasp record is malformed. Raised by :func:`validate_record` / :func:`load_grasps`."""


# =================================================================================================
# Mesh identity
# =================================================================================================
def mesh_digest(vertices, faces) -> str:
    """SHA1 of a mesh's geometry — the staleness guard stamped into a record.

    Same key style as ``mesh_collision._cache_key`` (float64 verts + int32 faces) so a mesh that
    hits the coacd cache also hits its grasp record. Candidates are only valid for the geometry they
    were computed from; :func:`check_mesh_current` compares this."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(np.asarray(faces).reshape(-1, 3), dtype=np.int32).tobytes())
    return h.hexdigest()


# =================================================================================================
# The canonical object frame (OBB)
# =================================================================================================
@dataclass(frozen=True)
class ObjectFrame:
    """The canonical frame of one asset mesh: ``p_canonical = rotation @ p_asset + translation``.

    ``extents`` are the full OBB side lengths [m], DESCENDING, matching the canonical x/y/z axes.
    ``axis_order`` records which trimesh OBB axis became canonical x/y/z, and ``sign_rules`` how each
    axis's direction was resolved (``"skew"`` = from mesh asymmetry, the placement-invariant case;
    ``"asset_align"`` = symmetric along that axis, so the sign was tied to the asset's authored
    orientation; ``"+handedness"`` = additionally flipped to keep the frame right-handed).

    ``drift`` is the MEASURED instability of the frame [m]: how far the canonical coordinates of the
    mesh's own vertices moved when the object was re-posed and the frame recomputed (None if not
    verified). It is not numerical noise — it is how near-degenerate the object's minimum-volume box
    is. A body of revolution (mug, bucket, bowl) has a nearly flat optimum about its symmetry axis,
    so the box yaw there is essentially arbitrary. Read it as a DETECTOR, not a bound: see
    :data:`_DRIFT_PROBES`.

    ``ambiguous`` is the summary flag — extents tie, a sign fell back, or ``drift`` exceeded
    :data:`_DRIFT_TOL`. An ambiguous frame is still bit-reproducible for THIS mesh, so every stored
    candidate stays valid; what it lacks is intrinsic MEANING in its orientation. A generator should
    cover such an object with symmetry-respecting candidates rather than assume its yaw means
    anything, and a re-export of the same object may land on a different frame."""
    rotation: np.ndarray            # (3,3) canonical_from_asset, right-handed orthonormal
    translation: np.ndarray         # (3,)
    extents: np.ndarray             # (3,) descending
    axis_order: tuple               # (3,) ints
    sign_rules: tuple               # (3,) str
    ambiguous: bool
    drift: float | None = None      # [m] measured re-pose instability; None = not verified

    def matrix(self) -> np.ndarray:
        """4x4 ``canonical_from_asset``."""
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def inverse_matrix(self) -> np.ndarray:
        """4x4 ``asset_from_canonical``."""
        m = np.eye(4)
        m[:3, :3] = self.rotation.T
        m[:3, 3] = -self.rotation.T @ self.translation
        return m

    def to_canonical(self, points) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return p @ self.rotation.T + self.translation

    def from_canonical(self, points) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return (p - self.translation) @ self.rotation

    def half_extents(self) -> np.ndarray:
        return 0.5 * self.extents

    def face_of(self, approach) -> str:
        """The OBB face an approach direction enters, as a :data:`FACE_BUCKETS` label.

        ``approach`` points INTO the object (canonical frame), so the face entered is the one whose
        OUTWARD normal most opposes it. A trajectory generator uses this to keep only the candidates
        reachable from whichever face is currently presented (e.g. facing up after settling)."""
        a = np.asarray(approach, dtype=float)
        n = np.linalg.norm(a)
        if n < 1.0e-12:
            raise ValueError("approach direction is degenerate")
        a = a / n
        k = int(np.argmax(np.abs(a)))
        return f"{'-' if a[k] > 0 else '+'}{'xyz'[k]}"

    def face_area(self, face: str) -> float:
        """Area [m²] of an OBB face — the two ``±z`` faces are the largest (extents descend)."""
        k = "xyz".index(face[1])
        return float(np.prod(np.delete(self.extents, k)))


def _rotation_about(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix (used only for the deterministic drift probes)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def canonical_frame(vertices, faces=None, *, angle_digits: int = _OBB_ANGLE_DIGITS,
                    verify: bool = True) -> ObjectFrame:
    """Compute an asset mesh's oriented bounding box and reduce it to ONE canonical frame.

    ``trimesh.bounds.oriented_bounds`` finds the minimum-volume OBB, but the frame it returns is one
    of 48 equivalent box frames (axis permutation × sign) — re-running it, or running it on a
    re-exported mesh, can return a different one. Stored poses would then silently rotate. This
    reduces the OBB to a single representative:

      1. **Origin** at the OBB centre (``oriented_bounds`` already centres the box), NOT the mesh
         pivot (arbitrary per asset) and NOT the centre of mass (needs density/watertightness).
      2. **Axis order** by extent descending — x = longest, z = shortest. Near-equal extents (within
         ``_EXTENT_TIE_RTOL``) are ordered by their alignment with the asset frame's own axes, so a
         cube resolves to the asset frame instead of an arbitrary rotation.
      3. **Axis signs** from the mesh's asymmetry about the OBB centre: the normalized third moment
         (skew) of the vertices along each axis, flipped so it is positive. Skew is computed IN the
         canonical frame, so this rule is invariant to how the asset happens to be posed. Axes too
         symmetric to carry a signal (``|skew| < _SKEW_TOL``) fall back to the asset frame's axis
         direction and mark the frame ``ambiguous``.
      4. **Right-handedness**: if the signs leave a reflection, the axis with the least skew
         information is flipped.
      5. **Stability is MEASURED, not assumed** (``verify``): the frame is recomputed on
         deterministically re-posed copies of the mesh and the worst canonical-coordinate movement
         is recorded as ``drift``, flagging the frame ambiguous past :data:`_DRIFT_TOL`. This is
         what catches a body of revolution, whose box yaw is arbitrary but whose extents do NOT tie
         (measured: the YCB mug's box rotates 1.5° about the cup axis between re-posings). Roughly
         triples the cost; pass ``verify=False`` for throwaway calls.

    Args:
        vertices: (n, 3) mesh vertices in the ASSET frame (as ``mesh_collision.load_usd_mesh``
            returns them — prim xforms already baked).
        faces: optional (m, 3) triangle indices. Given, the OBB is fit to the mesh's convex hull
            (tighter and hull-exact); omitted, it is fit to the vertices alone.
        angle_digits: angular precision of the OBB search — see :data:`_OBB_ANGLE_DIGITS` for why
            the default is raised above trimesh's.

    Returns:
        :class:`ObjectFrame`.

    The result is bit-identical on recomputation from the same vertex array, and (when every sign
    resolves from skew) invariant to the asset being re-posed. Both are asserted by ``--selfcheck``.
    """
    import trimesh

    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if len(v) < 4:
        raise ValueError(f"need at least 4 vertices for an OBB, got {len(v)}")
    f = None if faces is None else np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    # CENTRE FIRST, so translation-equivariance holds by CONSTRUCTION rather than by trimesh's
    # conditioning. (Measured on the YCB mug with matched rotations: a 1.4 m offset changes the
    # result by nothing either way — so this is insurance, not a fix for an observed failure. It
    # costs one subtraction and removes any dependence on where the asset sits in its own file.)
    centre = v.mean(axis=0)
    vc = v - centre
    obb_input = vc if f is None else trimesh.Trimesh(vertices=vc, faces=f, process=False)
    to_origin, extents = trimesh.bounds.oriented_bounds(obb_input, angle_digits=angle_digits)
    r0 = np.asarray(to_origin[:3, :3], dtype=float)   # rows = box axes in (centred) asset coords
    t0 = np.asarray(to_origin[:3, 3], dtype=float)
    extents = np.asarray(extents, dtype=float)

    # --- 2. axis order: extent descending, ties broken by alignment with the asset axes ----------
    tie = _EXTENT_TIE_RTOL * float(extents.max())
    def _order_key(i):
        bucket = -round(float(extents[i]) / tie) if tie > 0 else 0
        align = tuple(-np.round(np.abs(r0[i]), 6))     # prefer the axis most aligned with asset x, then y, z
        return (bucket, align, i)
    order = sorted(range(3), key=_order_key)
    rot = r0[order].copy()
    trans = t0[order].copy()
    ext = extents[order].copy()
    ambiguous = bool(np.any(np.diff(np.sort(extents)[::-1]) > -tie))   # any adjacent pair within tie

    # --- 3. axis signs from mesh asymmetry about the OBB centre ----------------------------------
    u = vc @ rot.T + trans                             # vertices in the (unsigned) canonical frame
    half = np.maximum(0.5 * ext, 1.0e-12)
    skew = np.mean((u / half) ** 3, axis=0)            # scale-free third moment, one per axis
    signs = np.ones(3)
    rules = []
    for k in range(3):
        if abs(skew[k]) >= _SKEW_TOL:
            signs[k] = 1.0 if skew[k] > 0 else -1.0
            rules.append("skew")
        else:
            # Symmetric along this axis: no mesh signal. Tie the sign to the asset frame's axis that
            # this canonical axis is most aligned with, which is at least reproducible for this mesh.
            j = int(np.argmax(np.abs(rot[k])))
            signs[k] = 1.0 if rot[k, j] > 0 else -1.0
            rules.append("asset_align")
            ambiguous = True

    # --- 4. right-handedness ---------------------------------------------------------------------
    if np.linalg.det((signs[:, None] * rot)) < 0:
        k = int(np.argmin(np.abs(skew)))               # flip the least informative axis
        signs[k] *= -1.0
        # Append rather than replace: which rule PROPOSED the sign still matters to a reader (an
        # "asset_align+handedness" axis had no geometric signal at all, "skew+handedness" did).
        rules[k] = f"{rules[k]}+handedness"

    rot = signs[:, None] * rot
    # Fold the centring back in: u = rot @ (p - centre) + trans_c  ==>  translation = trans_c - rot @ centre
    trans = signs * trans - rot @ centre

    # --- 5. measured stability -------------------------------------------------------------------
    drift = None
    if verify:
        ref = v @ rot.T + trans
        drift = 0.0
        for axis, angle in _DRIFT_PROBES:
            q = _rotation_about(axis, angle)
            probe = canonical_frame(v @ q.T, f, angle_digits=angle_digits, verify=False)
            drift = max(drift, float(np.abs(ref - probe.to_canonical(v @ q.T)).max()))
        if drift > _DRIFT_TOL:
            ambiguous = True

    return ObjectFrame(rotation=rot, translation=trans, extents=ext,
                       axis_order=tuple(int(i) for i in order), sign_rules=tuple(rules),
                       ambiguous=ambiguous, drift=drift)


# =================================================================================================
# Records
# =================================================================================================
@dataclass(frozen=True)
class GraspCandidate:
    """One precomputed grasp, posed in the object's canonical frame.

    ``transform`` is 4x4 homogeneous ``canonical ← grasp`` (ACRONYM ``grasps/transforms[i]``), read
    through :data:`POSE_CONVENTION`: column 0 = jaw closing axis, column 2 = approach, column 3 =
    the TCP grasp centre. ``width`` is the jaw opening at contact [m] (ACRONYM
    ``gripper/configuration``). ``face`` is the :data:`FACE_BUCKETS` label of the OBB face the
    approach enters. ``source`` tags what produced the candidate (e.g. ``"antipodal_v1"``,
    ``"manual"``, ``"acronym"``) so a later generator can supersede its own output without touching
    hand-authored entries. ``seat_mode`` (:data:`SEAT_MODES`) names which :func:`pad_seat` rule
    placed the pose along the approach — REQUIRED on disk (schema v2): how the pads hold the object
    is something consumers rank on, and before this field it lived only in per-generator prose that
    downstream code string-matched. ``span`` is the material extent :func:`pad_seat` measured along
    the approach [m] — REQUIRED on disk (schema v3): with ``seat_mode`` it tells a consumer where
    the material sits relative to the pads (``span_flush`` ⇒ ``[PAD_NEAR_Z − span, PAD_NEAR_Z]``,
    ``clamped_deep`` ⇒ ``[SEAT_DEEPEST_Z, SEAT_DEEPEST_Z + span]``) without re-measuring the mesh.
    ``quality`` holds the :data:`QUALITY_FIELDS`, all None until an
    evaluation pass fills them and names itself in ``quality_source``."""
    id: str
    transform: np.ndarray                     # (4,4)
    width: float
    face: str
    source: str
    seat_mode: str | None = None              # one of SEAT_MODES; None only on in-memory stubs
    span: float | None = None                 # material extent along the approach [m]; None only on stubs
    seat_depth: float | None = None           # grasp-frame z of the near material edge [m]; None only on stubs
    # Material protruding forward past the fingertips [m] — (seat_depth + span) − PAD_NEAR_Z,
    # floored at 0. Computed by the MERGE for clamped_deep candidates (2026-08-11): TRACKED in the
    # record because clamped-deep holds cluster on modest-overhang bodies, deliberately NOT a
    # grasp_select scoring input. None elsewhere.
    overhang: float | None = None
    quality: dict = field(default_factory=lambda: {k: None for k in QUALITY_FIELDS})
    quality_source: str | None = None
    labels: tuple = ()                        # optional semantic tags ("handle", "rim", ...)
    notes: str = ""

    @property
    def position(self) -> np.ndarray:
        """TCP grasp centre in the canonical frame."""
        return np.asarray(self.transform)[:3, 3]

    @property
    def jaw_axis(self) -> np.ndarray:
        """Jaw closing axis (canonical frame)."""
        return np.asarray(self.transform)[:3, 0]

    @property
    def approach(self) -> np.ndarray:
        """Approach direction — the way the TCP travels into the grasp (canonical frame)."""
        return np.asarray(self.transform)[:3, 2]

    @property
    def evaluated(self) -> bool:
        return self.quality_source is not None


def grasp_transform(position, approach, jaw_axis) -> np.ndarray:
    """Build a 4x4 grasp pose in :data:`POSE_CONVENTION` from a grasp centre and two directions.

    ``approach`` (+z) is taken as authoritative; ``jaw_axis`` (+x) is orthogonalized against it
    (Gram-Schmidt) and ``+y`` completes the right-handed frame. Raises if the two directions are
    parallel — a jaw axis along the approach is not a grasp."""
    p = np.asarray(position, dtype=float).reshape(3)
    z = np.asarray(approach, dtype=float).reshape(3)
    x = np.asarray(jaw_axis, dtype=float).reshape(3)
    nz = np.linalg.norm(z)
    if nz < 1.0e-12:
        raise ValueError("approach direction is degenerate")
    z = z / nz
    x = x - np.dot(x, z) * z                       # project the jaw axis into the plane normal to z
    nx = np.linalg.norm(x)
    if nx < 1.0e-9:
        raise ValueError("jaw_axis is parallel to approach — the jaws would close along the approach")
    x = x / nx
    m = np.eye(4)
    m[:3, 0] = x
    m[:3, 1] = np.cross(z, x)
    m[:3, 2] = z
    m[:3, 3] = p
    return m


# =================================================================================================
# Hand clearance — the hand's own collision volumes, in the grasp frame
# =================================================================================================
# ONE definition, shared by pad_seat's collision-aware retreat and shake_validate's pre-grasp
# collision check (which imports it from here), so the seat and the check can never disagree about
# where the hand is. Moved out of grasp_passes.shake_validate 2026-08-06 for exactly that reason.
#
# Each collider is tested as its true CONVEX HULL, not a bounding box. That distinction is not
# academic — the hand's colliders are CONVEX_MESH shapes, so the hull IS the collider and the test
# is exact, whereas the AABB of the palm hull spans 204 x 63 x 92 mm around a much smaller solid.
# With boxes the pre-check rejected 52 % of banana and 99 % of mug candidates, nearly all on the
# palm; a filter that throws away almost everything is measuring its own slack.
@dataclass(frozen=True)
class HandVolumes:
    """The hand's colliders as convex hulls, in the POSE_CONVENTION grasp frame.

    Each entry is ``(name, aabb_lo, aabb_hi, planes)`` where ``planes`` are the hull's outward face
    equations ``[nx, ny, nz, d]`` (inside iff ``n·p + d <= 0`` for all of them). The AABB is kept
    only as a cheap reject before the plane test."""
    volumes: tuple

    def at_width(self, width: float) -> tuple:
        """The volumes with the FINGER hulls translated inward to the pre-grasp aperture.

        The hulls are measured at the fully-open home pose; a finger's motion is a pure prismatic
        translation along the grasp-frame jaw axis (x), so posing the hand at
        ``min(width + PREGRASP_MARGIN, MAX_JAW_WIDTH)`` is a translation of each finger hull toward
        the centre by half the aperture difference. The palm does not move."""
        aperture = min(float(width) + PREGRASP_MARGIN, MAX_JAW_WIDTH)
        delta = 0.5 * (MAX_JAW_WIDTH - aperture)
        if delta <= 0.0:
            return self.volumes
        out = []
        for name, lo, hi, planes in self.volumes:
            if not name.startswith("finger"):
                out.append((name, lo, hi, planes))
                continue
            t = np.array([-delta if 0.5 * (lo[0] + hi[0]) > 0 else delta, 0.0, 0.0])
            pl = planes.copy()
            pl[:, 3] -= pl[:, :3] @ t
            out.append((name, lo + t, hi + t, pl))
        return tuple(out)

    def collisions(self, points: np.ndarray, tol: float = PENETRATION_TOL,
                   width: float | None = None) -> dict:
        """``{volume name: vertex count}`` for every volume the points intrude into.

        ``width`` poses the fingers at the pre-grasp aperture for that jaw width (see
        :meth:`at_width`); None tests the fully-open hand."""
        hits = {}
        for name, lo, hi, planes in (self.volumes if width is None else self.at_width(width)):
            near = np.all((points >= lo - tol) & (points <= hi + tol), axis=1)
            if not near.any():
                continue
            p = points[near]
            # Inside the hull by more than the tolerance on EVERY face.
            inside = np.all(p @ planes[:, :3].T + planes[:, 3] <= -tol, axis=1)
            n = int(inside.sum())
            if n >= MIN_VERTICES:
                hits[name] = n
        return hits


def _hull_planes(points: np.ndarray):
    """Outward face planes of a point set's convex hull, or None if it is degenerate."""
    try:
        from scipy.spatial import ConvexHull
        return np.asarray(ConvexHull(points).equations, dtype=float)
    except Exception:                          # noqa: BLE001 - scipy missing or a flat collider
        return None


def _measure_hand_volumes() -> HandVolumes:
    """Measure the hand's colliders off the ACTIVE robot at the pre-grasp opening (jaws fully open,
    the state the rig actually starts a grasp from). This is the one place grasp_library reaches
    into the robot build — lazily, because reading records must stay warp/newton-free."""
    import warp as wp
    import newton
    from .mathutils import find_body, quat_to_matrix_xyzw
    from .robot import build_franka_robot, finger_body_indices, hand_geometry

    g = hand_geometry()
    builder = build_franka_robot(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                                 table=None)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)   # home pose == jaws fully open
    body_q = state.body_q.numpy()
    labels = list(model.body_label)
    ee = find_body(labels, FRANKA.ee_link_suffix)
    fingers = finger_body_indices(model)

    t_w_ee = np.eye(4)
    t_w_ee[:3, :3] = quat_to_matrix_xyzw(body_q[ee][3:7])
    t_w_ee[:3, 3] = body_q[ee][:3]
    to_grasp = np.linalg.inv(t_w_ee @ g.ee_from_grasp)

    shape_body = model.shape_body.numpy()
    shape_xf = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    boxes = []      # (name, aabb_lo, aabb_hi, hull planes)
    for body, name in [(ee, "palm")] + list(zip(fingers, ("finger_left", "finger_right"))):
        for si in range(model.shape_count):
            if shape_body[si] != body:
                continue
            src = model.shape_source[si]
            verts = getattr(src, "vertices", None)
            if verts is None:
                continue
            v = np.asarray(verts, dtype=float) * shape_scale[si][:3]
            t_w_b = np.eye(4)
            t_w_b[:3, :3] = quat_to_matrix_xyzw(body_q[body][3:7])
            t_w_b[:3, 3] = body_q[body][:3]
            t_b_s = np.eye(4)
            t_b_s[:3, :3] = quat_to_matrix_xyzw(shape_xf[si][3:7])
            t_b_s[:3, 3] = shape_xf[si][:3]
            p = (to_grasp @ t_w_b @ t_b_s @ np.c_[v, np.ones(len(v))].T).T[:, :3]
            planes = _hull_planes(p)
            if planes is None:
                continue                      # degenerate collider: nothing meaningful to test
            boxes.append((f"{name}[{si}]", p.min(axis=0), p.max(axis=0), planes))
    return HandVolumes(volumes=tuple(boxes))


_HAND_VOLUMES: HandVolumes | None = None


def hand_volumes() -> HandVolumes:
    """Cached :class:`HandVolumes` — measuring costs a full robot load, so pay it once."""
    global _HAND_VOLUMES
    if _HAND_VOLUMES is None:
        _HAND_VOLUMES = _measure_hand_volumes()
    return _HAND_VOLUMES


def pregrasp_collision(canonical_from_grasp: np.ndarray, canonical_vertices: np.ndarray,
                       width: float) -> dict:
    """``{volume: vertex count}`` for the object material inside the hand at this pose; {} if clear.

    ``canonical_from_grasp`` is a candidate's stored 4x4 (the grasp pose in the object's canonical
    frame) and ``canonical_vertices`` the object's rest geometry in that same frame. ``width`` is
    the candidate's jaw width: the fingers are posed at the PRE-SHAPED aperture
    ``min(width + PREGRASP_MARGIN, MAX_JAW_WIDTH)`` — the state the trajectory approaches in —
    not fully open. Material *between* the pre-shaped jaws is the grasp and is fine; material
    inside the finger solids or the palm means the hand would have had to pass through the
    object."""
    inv = np.linalg.inv(np.asarray(canonical_from_grasp, dtype=float))
    pts = np.asarray(canonical_vertices, dtype=float)
    in_grasp = (inv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    return hand_volumes().collisions(in_grasp, width=width)


@dataclass(frozen=True)
class PadSeat:
    """Result of :func:`pad_seat` — where the grasp centre has to be for the pads to hold the object.

    ``position`` is the seated grasp centre (what a generator should store). ``advance`` is how far
    it moved along the approach. ``span`` is the material extent found along the approach, measured
    relative to the ORIGINAL position, so ``span[0] - advance`` and ``span[1] - advance`` are where
    that material ends up relative to the seated pose. ``contained`` is False when the object is
    deeper than :data:`PAD_LENGTH` and cannot be fully enclosed — the pose then means "jaws as deep
    as safe" (near material at :data:`SEAT_DEEPEST_Z`, overhang forward past the fingertips), not
    "object centred".
    ``pad_half_used`` is the footprint the measurement ended up needing — larger than
    :data:`PAD_HALF_WIDTH` means the vertex fallback had to grow it to find material at all, so the
    span is coarser than the pad it claims to describe.

    ``seat_mode`` (:data:`SEAT_MODES`) names which rule finally placed the pose — this is what a
    generator stores on the candidate. ``retreat`` is how far the collision-aware retreat backed
    off from the rule seat [m] (0 when the rule seat already cleared). ``blocked`` is True when the
    hand collides at EVERY depth on this approach that leaves material between the pads — the pose
    is left at the rule seat and the generator must MARK the candidate (``seat_blocked``), not
    delete it."""
    position: np.ndarray
    advance: float
    span: tuple
    contained: bool
    method: str                  # "raycast" (faces given) | "vertices" (point-set geometry)
    pad_half_used: float = PAD_HALF_WIDTH
    seat_mode: str = "span_flush"   # one of SEAT_MODES
    retreat: float = 0.0         # [m] backed off along the approach to clear the hand
    blocked: bool = False        # True: no depth on this approach clears the hand

    @property
    def seat_depth(self) -> float:
        """Grasp-frame z of the NEAR material edge at the seated pose [m] — what a generator
        stores on the candidate. With the stored ``span``, material occupies
        ``[seat_depth, seat_depth + span]`` in the grasp frame for EVERY seat mode (including
        retreated, which ``span`` + ``seat_mode`` alone cannot locate)."""
        return float(self.span[0] - self.advance)


def _column_axes(approach, jaw):
    a = np.asarray(approach, dtype=float)
    a = a / np.linalg.norm(a)
    j = np.asarray(jaw, dtype=float)
    j = j - float(j @ a) * a
    j = j / np.linalg.norm(j)
    return a, j, np.cross(a, j)


def _span_by_raycast(vertices, faces, position, a, j, t, width, pad_half):
    """Material span along the approach inside the jaw column, by casting rays through it."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3), process=False)
    us = np.linspace(-0.5 * width, 0.5 * width, _SEAT_RAYS)
    vs = np.linspace(-pad_half, pad_half, _SEAT_RAYS)
    grid = np.array([position + u * j + v * t for u in us for v in vs])
    # Start each ray well behind the object so an origin inside the mesh cannot miss the near face.
    back = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) + 1.0
    origins = grid - back * a
    dirs = np.tile(a, (len(origins), 1))
    hits, _, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    if len(hits) == 0:
        return None
    z = (hits - position) @ a
    return float(z.min()), float(z.max()), pad_half


def _span_by_vertices(vertices, position, a, j, t, width, pad_half):
    """Fallback for geometry with no faces (tet meshes, point sets): project the column's vertices.

    GROWS the pad footprint until the column holds enough samples, the same way every other local
    measurement in this codebase has to. Catalog mesh density spans two orders of magnitude — a YCB
    scan carries ~10-17k vertices, a coarse fTetWild sponge 235 — and at a fixed pad-sized footprint
    the sponge's interior contains no vertices at all, so a non-growing test reports "no material"
    for a solid object and every candidate on it gets dropped. Only the footprint grows: the jaw
    width is a physical bound, and widening it would measure material the jaws never enclose.
    Returns ``(lo, hi, pad_used)``."""
    du = np.asarray(vertices, dtype=float) - position
    j_off, t_off, a_off = du @ j, du @ t, du @ a
    within_jaw = np.abs(j_off) <= 0.5 * width
    pad = float(pad_half)
    for _ in range(_SEAT_MAX_GROWTH):
        inside = within_jaw & (np.abs(t_off) <= pad)
        if int(inside.sum()) >= _SEAT_MIN_SAMPLES:
            z = a_off[inside]
            return float(z.min()), float(z.max()), pad
        pad *= _SEAT_GROWTH
    return None


def pad_seat(position, approach, jaw_axis, width: float, vertices, faces=None, *,
             pad_half: float = PAD_HALF_WIDTH, hand_clearance: bool = True,
             rule: str = "auto") -> PadSeat | None:
    """Advance a grasp centre along its approach until the object sits between the PADS.

    THE one place gripper pad geometry is applied. Every generator must seat its poses through this
    before storing them — that is what :data:`POSE_CONVENTION` v2 promises, and it is why the offset
    is not a per-generator constant: where the pads are is a property of the hand, not of how a
    grasp was found.

    The depth rule is ONE rule — ``advance = min(tip cap, palm cap)`` — and ``seat_mode`` names
    which cap bound:

    * **span_flush** (tip cap): the far material edge lands on :data:`PAD_NEAR_Z`, so the whole
      span sits flush against the fingertip end of the pads and the tips do not extend past the
      object. The retired CENTRED rule (span midpoint on :data:`PAD_MID_Z`) seated deeper by
      ``27.6 − span/2`` mm, putting the fingertips past the object's far surface — into the
      TABLE for a top-down grasp on a resting object (9.6 mm on the 36 mm banana), which no
      online check could detect.
    * **clamped_deep** (palm cap): go as deep as the pads allow without the near material entering
      the PALM (near edge at :data:`SEAT_DEEPEST_Z`), and accept that the object protrudes
      forward past the fingertips. **A stored clamped pose therefore means "jaws as deep as
      safe", NOT "object centred".** The centring rule here would be actively wrong: a span that
      cannot fit, centred, pushes its overhang backwards into the palm — measured on the
      62.9 mm-deep apple, where midpoint seating left every one of its 116 candidates colliding
      with the palm (2 collide with the seating removed entirely).
    * **retreated**: the rule seat above put the object inside the HAND'S OWN COLLISION HULLS
      (:func:`hand_volumes` — the same test the shake_validate pre-check runs), so the pose backs
      off along the approach in :data:`SEAT_RETREAT_STEP` steps to the DEEPEST collision-free depth
      that still leaves material behind the fingertip plane. Why depth alone is worth searching:
      the 2026-08-06 sweep on the mug/bowl/pitcher skips found every clear window on the RETREAT
      side (bodies wider than the jaw stroke collide laterally at any centred/clamped depth), and
      rescuing them needs a shallower hold, not a different span accounting. If NO depth clears,
      ``blocked`` is set and the pose stays at the rule seat — the generator marks the candidate
      (``seat_blocked``) rather than deleting it, the codebase's convention for measured failures.
      ``hand_clearance=False`` skips the test (pure-geometry callers with no robot available); the
      span/rule seat is unchanged either way.

    Material is found by RAYCASTING a grid through the jaw column when ``faces`` are supplied — a
    vertex test cannot see a face whose vertices all lie outside the column, which is exactly the
    case for a large flat panel between the jaws. Point-set geometry (tet meshes, the procedural
    rod) has no faces and falls back to projecting vertices; ``method`` records which ran.

    IDEMPOTENT by construction: the rule seat depends only on the measured span (geometry), and the
    retreat grid is anchored to the rule seat, so re-seating a stored pose — including a stored
    RETREATED pose — reproduces the same absolute position (advance == 0 on the second run).

    Returns None when no material is found in the column at all — the caller should drop that
    candidate rather than store a pose seated on nothing.
    """
    p = np.asarray(position, dtype=float).reshape(3)
    a, j, t = _column_axes(approach, jaw_axis)
    if faces is None:
        span = _span_by_vertices(vertices, p, a, j, t, width, pad_half)
        method = "vertices"
    else:
        span = _span_by_raycast(vertices, faces, p, a, j, t, width, pad_half)
        method = "raycast"
        if span is None:                        # a column that misses every triangle still gets the
            span = _span_by_vertices(vertices, p, a, j, t, width, pad_half)   # cheaper test's answer
            method = "vertices"
    if span is None:
        return None
    lo, hi, pad_used = span
    contained = bool((hi - lo) <= PAD_LENGTH)
    if rule == "auto":
        # The PRIMARY depth rule: as deep as possible subject to BOTH caps. Advancing the TCP by
        # `advance` moves the material to z - advance in the seated frame.
        #   tip cap  — the fingertips must not pass the object's far surface (far material edge
        #              stays at PAD_NEAR_Z). Deeper would push the tips past the object, and for
        #              a resting object the support surface (table) is exactly there: the centred
        #              rule commanded the tips 27.6 - span/2 mm past the far surface — 9.6 mm
        #              into the table under a top-down banana grasp.
        #   palm cap — the near material edge must not pass SEAT_DEEPEST_Z (into the palm face).
        advance = min(hi - PAD_NEAR_Z, lo - SEAT_DEEPEST_Z)
        # The mode names which cap bound: span_flush = the whole span sits flush against the
        # fingertip end of the pads; clamped_deep = "jaws as deep as safe", with the remainder of
        # the object protruding forward past the fingertips.
        mode = "span_flush" if hi - PAD_NEAR_Z <= lo - SEAT_DEEPEST_Z else "clamped_deep"
    elif rule == "centred":
        # The CENTRED companion (span midpoint on PAD_MID_Z) — the measured-stronger seat under a
        # lift load (25% vs flush's 4.2%), executable only when the scene leaves room beyond the
        # object's far surface, which ONLINE selection decides. Only defined where the palm
        # allows it; None otherwise (the caller stores no variant).
        advance = 0.5 * (lo + hi) - PAD_MID_Z
        if lo - advance < SEAT_DEEPEST_Z or not contained:
            return None
        mode = "centred"
    else:
        raise ValueError(f"unknown seat rule {rule!r}")
    retreat, blocked = 0.0, False
    if hand_clearance:
        # The object's FULL geometry in the grasp frame at the ORIGINAL position (x = jaw, y = the
        # third axis, z = approach — the same frame grasp_transform builds); a seat depth d puts
        # material at z - d. The whole mesh is tested, not just the jaw column: the sweep showed
        # the binding contacts are LATERAL (rim/wall beside the fingers, the 204 mm palm), which
        # the column never sees.
        du = np.asarray(vertices, dtype=float) - p
        grasp_pts = np.column_stack((du @ j, du @ t, du @ a))
        vols = hand_volumes()

        def _collides(depth: float) -> bool:
            pts = grasp_pts.copy()
            pts[:, 2] -= depth
            # Fingers at the PRE-SHAPED aperture (width + PREGRASP_MARGIN), the state the
            # trajectory approaches in — the same width the shake pre-check tests.
            return bool(vols.collisions(pts, width=width))

        if _collides(advance):
            d = advance
            while True:
                d -= SEAT_RETREAT_STEP
                if lo - d >= PAD_NEAR_Z:      # nearest material left the pads: nothing to close on
                    blocked = True
                    break
                if not _collides(d):
                    retreat = advance - d
                    advance = d
                    mode = "retreated"
                    break
    return PadSeat(position=p + advance * a, advance=float(advance), span=(lo, hi),
                   contained=contained, method=method, pad_half_used=float(pad_used),
                   seat_mode=mode, retreat=float(retreat), blocked=blocked)


CENTRED_VARIANT_SUFFIX = "_ctr"
CENTRED_VARIANT_LABEL = "centred_variant"
# A retreated centred variant closer than this to the primary seat duplicates it — not stored.
_VARIANT_MIN_SEPARATION = 0.002


def centred_variant(primary: PadSeat, position, approach, jaw_axis, width: float, vertices,
                    faces=None, *, pad_half: float = PAD_HALF_WIDTH,
                    hand_clearance: bool = True) -> PadSeat | None:
    """The CENTRED depth companion of a primary seat, or None when it adds nothing.

    Stored as a SEPARATE candidate (id + :data:`CENTRED_VARIANT_SUFFIX`, label
    :data:`CENTRED_VARIANT_LABEL`): offline it is collision-checked (with retreat) and
    shake-validated like any pose; ONLINE ``grasp_select``'s depth stage picks between the pair
    by whether anything occupies the space beyond the object's far surface. None when the span
    does not admit a palm-safe centred seat (``pad_seat`` rule), when the hand clears at NO depth
    (``blocked`` — the flush sibling already carries that information), or when the retreat
    converged back onto the primary seat (a duplicate, not a second depth)."""
    s = pad_seat(position, approach, jaw_axis, width, vertices, faces,
                 pad_half=pad_half, hand_clearance=hand_clearance, rule="centred")
    if s is None or s.blocked:
        return None
    if abs(s.advance - primary.advance) < _VARIANT_MIN_SEPARATION:
        return None
    return s


def measure_span_at(position, approach, jaw_axis, width: float, vertices, faces=None, *,
                    pad_half: float = PAD_HALF_WIDTH) -> PadSeat | None:
    """Measure the material span at a pose WITHOUT moving it — the llm_retry authoring probe.

    The llm_retry stage stores poses exactly where the LLM put them (``seat_mode: "llm"``, no
    seat rule, no retreat — docs/trajPipeline/llm-retry.md), but ``span``/``seat_depth`` are
    required schema fields because consumers locate material with them, so they are MEASURED at
    the given pose by the same jaw-column probe :func:`pad_seat` uses (raycast with faces, grown
    vertex projection without). Returns a :class:`PadSeat` with ``advance = 0`` (``position`` is
    echoed unchanged; ``seat_depth`` is the probe's near-material edge) or None when the column
    holds no material at all. The caller judges the result: a near edge in front of the TCP
    (``seat_depth > 0``) is a pose gripping air, one behind the palm cap (``< SEAT_DEEPEST_Z``)
    is buried — both are authoring failures to DROP and report, not to auto-correct."""
    p = np.asarray(position, dtype=float).reshape(3)
    a, j, t = _column_axes(approach, jaw_axis)
    if faces is None:
        span = _span_by_vertices(vertices, p, a, j, t, width, pad_half)
        method = "vertices"
    else:
        span = _span_by_raycast(vertices, faces, p, a, j, t, width, pad_half)
        method = "raycast"
        if span is None:
            span = _span_by_vertices(vertices, p, a, j, t, width, pad_half)
            method = "vertices"
    if span is None:
        return None
    lo, hi, pad_used = span
    return PadSeat(position=p, advance=0.0, span=(lo, hi),
                   contained=bool((hi - lo) <= PAD_LENGTH), method=method,
                   pad_half_used=float(pad_used), seat_mode="llm", retreat=0.0, blocked=False)


def make_candidate(frame: ObjectFrame, cid: str, position, approach, jaw_axis, width: float,
                   source: str, *, seat_mode: str, span: float, seat_depth: float, labels=(),
                   notes: str = "") -> GraspCandidate:
    """Assemble a candidate from canonical-frame geometry, deriving the OBB face bucket from the
    approach direction so it cannot disagree with the transform (which :func:`validate_record`
    independently re-checks). This is the intended way for a generator to emit candidates.

    ``seat_mode``, ``span`` and ``seat_depth`` are required, not defaulted: a generator that
    seated through :func:`pad_seat` has all three in hand (``seat.seat_mode``,
    ``seat.span[1] - seat.span[0]``, ``seat.seat_depth``), and a silent default here would let a
    pose be stored with values nobody measured."""
    if seat_mode not in SEAT_MODES:
        raise ValueError(f"seat_mode {seat_mode!r} not in {SEAT_MODES}")
    if not (0.0 < float(span) <= 1.0):
        raise ValueError(f"span {span!r} not a plausible material extent [m]")
    if not (-0.05 <= float(seat_depth) <= 0.0):
        raise ValueError(f"seat_depth {seat_depth!r} outside the pad-reachable range [-0.05, 0]")
    t = grasp_transform(position, approach, jaw_axis)
    return GraspCandidate(id=cid, transform=t, width=float(width), face=frame.face_of(t[:3, 2]),
                          source=source, seat_mode=seat_mode, span=float(span),
                          seat_depth=float(seat_depth),
                          quality={k: None for k in QUALITY_FIELDS},
                          quality_source=None, labels=tuple(labels), notes=notes)


@dataclass(frozen=True)
class GraspRecord:
    """One catalog object's grasp sidecar: the object block, the gripper block, and the candidates.

    Keyed by CATALOG name, not asset filename — two catalog entries can share one USD (the shirts),
    and procedural kinds (``rigid_box``, ``rubiks_cube``) have no file at all."""
    name: str                       # catalog key (scene_catalog.json "name")
    kind: str                       # catalog "kind"
    frame: ObjectFrame
    candidates: tuple = ()
    file: str | None = None         # asset path relative to assets/objects (None for procedural kinds)
    scale: float = 1.0
    mesh_sha1: str | None = None
    gripper_type: str = GRIPPER_TYPE
    max_width: float = MAX_JAW_WIDTH
    generator: str = ""             # what wrote the file (tool + version)
    notes: str = ""

    def by_face(self, face: str) -> tuple:
        return tuple(c for c in self.candidates if c.face == face)

    def by_source(self, source: str) -> tuple:
        return tuple(c for c in self.candidates if c.source == source)

    def asset_from_canonical(self) -> np.ndarray:
        """4x4 mapping canonical-frame poses into the ASSET (= body) frame."""
        return self.frame.inverse_matrix()

    def grasp_in_asset(self, candidate: GraspCandidate) -> np.ndarray:
        """``candidate.transform`` expressed in the asset/body frame."""
        return self.asset_from_canonical() @ np.asarray(candidate.transform, dtype=float)

    def grasp_in_world(self, candidate: GraspCandidate, world_from_body) -> np.ndarray:
        """The candidate's world grasp pose under a runtime placement.

        ``world_from_body`` is the 4x4 pose the runtime spawned the body at — for a catalog object
        that is ``translate(pos) @ rot_z(yaw)``, with ``pos[2]`` already carrying the ``rest_on_z``
        lift (``assets.add_ycb_mesh``). Use :func:`body_pose` to build it from ``(pos, yaw)``."""
        return np.asarray(world_from_body, dtype=float) @ self.grasp_in_asset(candidate)


def body_pose(pos, yaw: float = 0.0) -> np.ndarray:
    """The 4x4 ``world_from_body`` a scene placement implies: ``translate(pos) @ rot_z(yaw)``.

    Mirrors how ``assets.add_ycb_mesh`` / ``add_soft_mesh_object`` spawn a body, so a candidate can
    be taken to world coordinates from a ``scene.json`` entry's settled ``x``/``y``/``yaw`` without
    building a sim."""
    p = np.asarray(pos, dtype=float).reshape(3)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = p
    return m


# =================================================================================================
# Validation
# =================================================================================================
def _require(cond, msg: str) -> None:
    if not cond:
        raise GraspSchemaError(msg)


def _check_rotation(r, what: str) -> None:
    r = np.asarray(r, dtype=float)
    _require(r.shape == (3, 3), f"{what}: expected a 3x3 rotation, got shape {r.shape}")
    _require(np.all(np.isfinite(r)), f"{what}: rotation has non-finite entries")
    err = float(np.abs(r @ r.T - np.eye(3)).max())
    _require(err <= _ORTHO_TOL, f"{what}: rotation is not orthonormal (max |RRᵀ−I| = {err:.2e})")
    det = float(np.linalg.det(r))
    _require(abs(det - 1.0) <= _ORTHO_TOL,
             f"{what}: rotation is not right-handed (det = {det:.6f}; a reflection would mirror "
             f"every stored grasp)")


def validate_record(data: dict, *, name: str | None = None) -> None:
    """Validate a record dict (the on-disk form). Raises :class:`GraspSchemaError` on the first
    problem, with enough context to fix it. Called by :func:`load_grasps` and :func:`write_record`,
    so a bad record can neither be read nor written."""
    where = f"grasp record {name or data.get('object', {}).get('name', '?')!r}"
    _require(isinstance(data, dict), f"{where}: not a JSON object")
    ver = data.get("schema_version")
    _require(ver == SCHEMA_VERSION,
             f"{where}: schema_version {ver!r} != {SCHEMA_VERSION} (regenerate the record)")
    for block in ("object", "gripper", "grasps"):
        _require(isinstance(data.get(block), dict), f"{where}: missing '{block}' block")

    obj, grip, grasps = data["object"], data["gripper"], data["grasps"]

    # ---- object block ---------------------------------------------------------------------------
    for key in ("name", "kind", "frame", "extents"):
        _require(key in obj, f"{where}: object.{key} is required")
    if name is not None:
        _require(obj["name"] == name,
                 f"{where}: object.name {obj['name']!r} does not match its filename ({name!r})")
    _require(obj["kind"] not in UNSUPPORTED_KINDS,
             f"{where}: kind {obj['kind']!r} is out of scope for precomputed grasp candidates. A "
             f"garment or bag has no persistent rest shape, so an OBB of its source mesh describes "
             f"the asset file rather than the settled object, and there is no rigid body to compose "
             f"a stored pose with. Cloth grasp targets must be resolved against the live particle "
             f"state instead (see the SCOPE paragraph in grasp_library.py).")
    ext = np.asarray(obj["extents"], dtype=float)
    _require(ext.shape == (3,), f"{where}: object.extents must be 3 numbers")
    _require(np.all(ext > 0) and np.all(np.isfinite(ext)),
             f"{where}: object.extents must be positive and finite, got {ext.tolist()}")
    # Descending, but only up to the tie tolerance: TIED extents are deliberately ordered by asset-
    # axis alignment instead (so a near-cube resolves to a stable frame rather than to whichever
    # axis won by a micron), which can put a marginally shorter extent first. Anything beyond the
    # tolerance is a real ordering violation.
    tie = _EXTENT_TIE_RTOL * float(ext.max())
    _require(ext[0] >= ext[1] - tie and ext[1] >= ext[2] - tie,
             f"{where}: object.extents must be DESCENDING within the {_EXTENT_TIE_RTOL:.0%} tie "
             f"tolerance (the {FRAME_CONVENTION} axis order), got {ext.tolist()}")

    frame = obj["frame"]
    _require(frame.get("convention") == FRAME_CONVENTION,
             f"{where}: frame.convention {frame.get('convention')!r} != {FRAME_CONVENTION!r}")
    _check_rotation(frame.get("rotation"), f"{where}: frame")
    _require(np.asarray(frame.get("translation"), dtype=float).shape == (3,),
             f"{where}: frame.translation must be 3 numbers")
    _require(len(frame.get("axis_order", ())) == 3 and sorted(frame["axis_order"]) == [0, 1, 2],
             f"{where}: frame.axis_order must be a permutation of (0,1,2)")
    _require(len(frame.get("sign_rules", ())) == 3,
             f"{where}: frame.sign_rules must have one entry per axis")

    # ---- gripper block --------------------------------------------------------------------------
    # v1 is called out by name because it is the dangerous case: its axes are identical to v2's, so a
    # v1 pose is READABLE and merges cleanly — it is only wrong by a ~28 mm shift along the approach
    # that nothing downstream can detect. Rejecting it is what stops a half-regenerated library from
    # producing one record whose grasps mean two different things.
    _require(grip.get("pose_convention") != POSE_CONVENTION_V1,
             f"{where}: gripper.pose_convention is {POSE_CONVENTION_V1!r}. v1 poses were stored "
             f"wherever the generator put them along the approach (usually on the material centre), "
             f"which seats the object on the pad tips; v2 poses are PAD-CENTRED via "
             f"grasp_library.pad_seat() and command-ready. The two are indistinguishable by "
             f"inspection, so this record must be REGENERATED, not relabelled.")
    _require(grip.get("pose_convention") == POSE_CONVENTION,
             f"{where}: gripper.pose_convention {grip.get('pose_convention')!r} != "
             f"{POSE_CONVENTION!r} — the transforms would be interpreted with the wrong axes")
    max_width = float(grip.get("max_width", MAX_JAW_WIDTH))
    _require(max_width > 0, f"{where}: gripper.max_width must be positive")

    # ---- candidates -----------------------------------------------------------------------------
    cands = grasps.get("candidates")
    _require(isinstance(cands, list), f"{where}: grasps.candidates must be a list")
    seen = set()
    for i, c in enumerate(cands):
        at = f"{where}: candidate[{i}]"
        _require(isinstance(c, dict), f"{at} is not a JSON object")
        cid = c.get("id")
        _require(isinstance(cid, str) and cid, f"{at}: id must be a non-empty string")
        at = f"{where}: candidate {cid!r}"
        _require(cid not in seen, f"{at}: duplicate id")
        seen.add(cid)

        t = np.asarray(c.get("transform"), dtype=float)
        _require(t.shape == (4, 4), f"{at}: transform must be 4x4, got shape {t.shape}")
        _require(np.all(np.isfinite(t)), f"{at}: transform has non-finite entries")
        _require(np.allclose(t[3], [0.0, 0.0, 0.0, 1.0], atol=_ORTHO_TOL),
                 f"{at}: transform bottom row must be [0,0,0,1], got {t[3].tolist()}")
        _check_rotation(t[:3, :3], at)

        w = c.get("width")
        _require(isinstance(w, (int, float)) and np.isfinite(w), f"{at}: width must be a number")
        _require(0.0 < float(w) <= max_width,
                 f"{at}: width {float(w):.4f} m is outside the jaw range (0, {max_width:.4f}]")

        src = c.get("source")
        _require(isinstance(src, str) and src, f"{at}: source must be a non-empty tag")

        sm = c.get("seat_mode")
        _require(sm in SEAT_MODES,
                 f"{at}: seat_mode {sm!r} not in {SEAT_MODES}. Every stored pose carries which "
                 f"pad_seat rule placed it — flush against the fingertip end, centred between "
                 f"the pads (the '_ctr' depth variant), clamped as deep as the palm allows, or "
                 f"retreated to clear the hand. A record without it predates schema v2; "
                 f"regenerate it.")

        sp = c.get("span")
        _require(isinstance(sp, (int, float)) and np.isfinite(sp) and 0.0 < float(sp) <= 1.0,
                 f"{at}: span {sp!r} must be the measured material extent along the approach in "
                 f"(0, 1] m. A record without it predates schema v3; regenerate it.")

        sd = c.get("seat_depth")
        _require(isinstance(sd, (int, float)) and np.isfinite(sd) and -0.05 <= float(sd) <= 0.0,
                 f"{at}: seat_depth {sd!r} must be the near material edge in the grasp frame, in "
                 f"[-0.05, 0] m. A record without it predates schema v4; regenerate it.")

        ov = c.get("overhang")
        _require(ov is None or (isinstance(ov, (int, float)) and np.isfinite(ov) and float(ov) >= 0),
                 f"{at}: overhang {ov!r} must be a non-negative number [m] or absent (it is "
                 f"merge-derived for clamped_deep candidates — tracked, never scored)")

        face = c.get("face")
        _require(face in FACE_BUCKETS, f"{at}: face {face!r} not in {FACE_BUCKETS}")
        # The bucket must agree with the transform: a mislabeled face silently breaks any
        # face-filtered lookup, and nothing downstream would catch it.
        approach = t[:3, 2]
        k = int(np.argmax(np.abs(approach)))
        implied = f"{'-' if approach[k] > 0 else '+'}{'xyz'[k]}"
        _require(face == implied,
                 f"{at}: face {face!r} disagrees with the transform's approach direction "
                 f"{np.round(approach, 4).tolist()} (implies {implied!r})")

        q = c.get("quality", {})
        _require(isinstance(q, dict), f"{at}: quality must be an object")
        unknown = set(q) - set(QUALITY_FIELDS)
        _require(not unknown,
                 f"{at}: unknown quality field(s) {sorted(unknown)}; known: {list(QUALITY_FIELDS)}")
        for key, val in q.items():
            _require(val is None or (isinstance(val, (int, float)) and np.isfinite(val)),
                     f"{at}: quality.{key} must be a number or null, got {val!r}")
        qsrc = c.get("quality_source")
        _require(qsrc is None or (isinstance(qsrc, str) and qsrc),
                 f"{at}: quality_source must be a non-empty string or null")
        if qsrc is None:
            _require(all(v is None for v in q.values()),
                     f"{at}: quality values are set but quality_source is null — an unattributed "
                     f"measurement cannot be reproduced")


def check_mesh_current(record: GraspRecord, vertices, faces) -> bool:
    """True if ``record`` was computed from this geometry. A record with no ``mesh_sha1`` (a
    procedural or hand-authored entry) is treated as current."""
    return record.mesh_sha1 is None or record.mesh_sha1 == mesh_digest(vertices, faces)


# =================================================================================================
# Serialization + loading
# =================================================================================================
def _r(a) -> list:
    return np.round(np.asarray(a, dtype=float), _ROUND).tolist()


def _dump_json(data: dict) -> str:
    """``json.dumps(indent=1)`` puts every number on its own line, which turns one 4x4 transform
    into 22 lines and a real record into thousands — unreviewable in a diff, which is most of why
    these are JSON at all. Collapse the innermost all-numeric arrays back onto one line so a
    transform reads as four rows. Output is ordinary JSON; only whitespace changes."""
    import re

    text = json.dumps(data, indent=1)
    # Innermost arrays only: contents must be numbers, commas and whitespace (a quoted string or a
    # nested "[" fails the class, so sign_rules and the outer 4x4 keep their line-per-row layout).
    return re.sub(r"\[\s+((?:-?\d[\d.eE+-]*(?:,\s*)?)+?)\s*\]",
                  lambda m: "[" + " ".join(m.group(1).split()) + "]", text)


def record_to_dict(record: GraspRecord) -> dict:
    return {
        "_comment": [
            "Precomputed grasp candidates for one catalog object. Poses are in the object's",
            f"CANONICAL frame ({FRAME_CONVENTION}: OBB centre, axes by descending extent);",
            f"each transform places the grasp frame ({POSE_CONVENTION}: origin at the TCP grasp",
            "centre, +z approach, +x jaw axis). Layout mirrors the ACRONYM dataset's HDF5 records.",
            "Generated offline — see deformableManipulationTools/grasp_library.py; do not hand-edit",
            "the frame block (it is derived from the mesh and guarded by object.mesh_sha1).",
        ],
        "schema_version": SCHEMA_VERSION,
        "object": {
            "name": record.name,
            "kind": record.kind,
            "file": record.file,
            "scale": float(record.scale),
            "mesh_sha1": record.mesh_sha1,
            "extents": _r(record.frame.extents),
            "frame": {
                "convention": FRAME_CONVENTION,
                "rotation": [_r(row) for row in record.frame.rotation],
                "translation": _r(record.frame.translation),
                "axis_order": [int(i) for i in record.frame.axis_order],
                "sign_rules": list(record.frame.sign_rules),
                "ambiguous": bool(record.frame.ambiguous),
                "drift": (None if record.frame.drift is None
                          else round(float(record.frame.drift), _ROUND)),
            },
        },
        "gripper": {
            "type": record.gripper_type,
            "max_width": float(record.max_width),
            "pose_convention": POSE_CONVENTION,
        },
        "grasps": {
            "generator": record.generator,
            "notes": record.notes,
            "candidates": [
                {
                    "id": c.id,
                    "transform": [_r(row) for row in np.asarray(c.transform, dtype=float)],
                    "width": round(float(c.width), _ROUND),
                    "face": c.face,
                    "source": c.source,
                    "seat_mode": c.seat_mode,
                    "span": (None if c.span is None else round(float(c.span), _ROUND)),
                    "seat_depth": (None if c.seat_depth is None
                                   else round(float(c.seat_depth), _ROUND)),
                    **({} if c.overhang is None
                       else {"overhang": round(float(c.overhang), _ROUND)}),
                    "labels": list(c.labels),
                    "notes": c.notes,
                    "quality": {k: c.quality.get(k) for k in QUALITY_FIELDS},
                    "quality_source": c.quality_source,
                }
                for c in record.candidates
            ],
        },
    }


def record_from_dict(data: dict, *, name: str | None = None, validate: bool = True) -> GraspRecord:
    if validate:
        validate_record(data, name=name)
    obj, grip, grasps = data["object"], data["gripper"], data["grasps"]
    f = obj["frame"]
    frame = ObjectFrame(
        rotation=np.asarray(f["rotation"], dtype=float),
        translation=np.asarray(f["translation"], dtype=float),
        extents=np.asarray(obj["extents"], dtype=float),
        axis_order=tuple(int(i) for i in f["axis_order"]),
        sign_rules=tuple(f["sign_rules"]),
        ambiguous=bool(f.get("ambiguous", False)),
        drift=(None if f.get("drift") is None else float(f["drift"])),
    )
    candidates = tuple(
        GraspCandidate(
            id=c["id"],
            transform=np.asarray(c["transform"], dtype=float),
            width=float(c["width"]),
            face=c["face"],
            source=c["source"],
            seat_mode=c["seat_mode"],
            span=float(c["span"]),
            seat_depth=float(c["seat_depth"]),
            overhang=(None if c.get("overhang") is None else float(c["overhang"])),
            quality={k: c.get("quality", {}).get(k) for k in QUALITY_FIELDS},
            quality_source=c.get("quality_source"),
            labels=tuple(c.get("labels", ())),
            notes=c.get("notes", ""),
        )
        for c in grasps["candidates"]
    )
    return GraspRecord(
        name=obj["name"], kind=obj["kind"], frame=frame, candidates=candidates,
        file=obj.get("file"), scale=float(obj.get("scale", 1.0)),
        mesh_sha1=obj.get("mesh_sha1"),
        gripper_type=grip.get("type", GRIPPER_TYPE),
        max_width=float(grip.get("max_width", MAX_JAW_WIDTH)),
        generator=grasps.get("generator", ""), notes=grasps.get("notes", ""),
    )


def record_path(name: str) -> Path:
    return GRASPS_DIR / f"{name}.json"


def has_grasps(name: str) -> bool:
    """True if a grasp record exists for this catalog name. A missing record is a normal state
    (nothing has been precomputed for that object yet), NOT an error."""
    return record_path(name).exists()


def available_grasps() -> list:
    """Catalog names with a grasp record, sorted."""
    if not GRASPS_DIR.exists():
        return []
    return sorted(p.stem for p in GRASPS_DIR.glob("*.json"))


_CACHE: dict = {}


def load_grasps(name: str, *, use_cache: bool = True) -> GraspRecord:
    """Load and validate one catalog object's grasp record.

    Raises ``FileNotFoundError`` if none exists (check :func:`has_grasps` first when absence is
    acceptable) and :class:`GraspSchemaError` if the record is malformed."""
    if use_cache and name in _CACHE:
        return _CACHE[name]
    path = record_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no grasp record for {name!r} at {path} "
            f"(have: {', '.join(available_grasps()) or 'none'})")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GraspSchemaError(f"grasp record {path} is not valid JSON: {exc}") from exc
    record = record_from_dict(data, name=name)
    if use_cache:
        _CACHE[name] = record
    return record


def write_record(record: GraspRecord, *, path: Path | None = None) -> Path:
    """Serialize, VALIDATE, and write a record. Validating on write means a generator cannot emit a
    record the loader would later reject."""
    data = record_to_dict(record)
    validate_record(data, name=record.name)
    out = path or record_path(record.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump_json(data) + "\n")
    _CACHE.pop(record.name, None)
    return out


def clear_cache() -> None:
    _CACHE.clear()


# =================================================================================================
# Candidate statuses + the llm_retry trigger (2026-08-11)
# =================================================================================================
# THE accounting for "does this object have a working grasp" — one definition, imported by the
# merge, grasp_select, and the llm_retry stage rather than re-derived in each. Spec:
# docs/trajPipeline/grasp-library.md "Candidate statuses" and docs/trajPipeline/llm-retry.md.
def is_weak(candidate: GraspCandidate) -> bool:
    """A weak grasp option, not a legitimate candidate. Keyed on ``seat_mode`` (the geometric
    fact) rather than :data:`WEAK_GRASP_LABEL` (which the merge derives FROM it), so the answer
    is the same for a freshly generated candidate and a loaded record."""
    return candidate.seat_mode == "retreated"


def legitimate_candidates(record: GraspRecord) -> tuple:
    """The candidates that count: not weak (retreated), not seat-blocked. Blocked candidates are
    merge-discarded and should never appear in a loaded record; the filter here is belt over
    braces for pre-merge (sidecar) records passed in by passes."""
    return tuple(c for c in record.candidates
                 if not is_weak(c) and SEAT_BLOCKED_LABEL not in c.labels)


def is_shake_covered(candidate: GraspCandidate) -> bool:
    """Physics validation has SAID something about this candidate: measured quality
    (``quality_source`` set) or an honest skip (:data:`SHAKE_SKIPPED_LABEL`). An uncovered
    candidate is "not yet known", which must never be conflated with "failed"."""
    return candidate.quality_source is not None or SHAKE_SKIPPED_LABEL in candidate.labels


def _held(candidate: GraspCandidate) -> bool:
    q = candidate.quality.get("object_in_gripper") if candidate.quality else None
    return q is not None and float(q) == 1.0


def record_holds(record: GraspRecord) -> bool:
    """True when at least one LEGITIMATE candidate held in physics validation. Holds on weak
    (retreated) candidates deliberately do not count — they are not grasps the pipeline offers."""
    return any(_held(c) for c in legitimate_candidates(record))


def is_unusable(record: GraspRecord) -> bool:
    """The merge-derived durable verdict: llm_retry exhausted both rounds, still nothing holds.
    Like out-of-reach, downstream treats the object as having no grasp."""
    return UNUSABLE_TOKEN in (record.notes or "")


def needs_llm_retry(record: GraspRecord) -> bool:
    """Whether the llm_retry stage should run for this record (docs/trajPipeline/llm-retry.md).

    True iff the kind is supported, the record is not already unusable, every legitimate candidate
    is shake-covered, and ZERO legitimate candidates hold. An empty record (out of reach) and a
    record whose only candidates — or only holds — are retreated all satisfy this vacuously:
    "no legitimate passing grasp" is exactly the population the stage exists for."""
    if record.kind in UNSUPPORTED_KINDS or is_unusable(record):
        return False
    legit = legitimate_candidates(record)
    if any(not is_shake_covered(c) for c in legit):
        return False                      # untested is "not yet known", not "failed"
    return not any(_held(c) for c in legit)


# =================================================================================================
# Self-check — the repo has no test framework, so the invariants this module PROMISES are executable:
#     .venv/bin/python -m deformableManipulationTools.grasp_library --selfcheck
#     .venv/bin/python -m deformableManipulationTools.grasp_library --selfcheck --asset ycb/banana.usd
# =================================================================================================
def _selfcheck(asset: str | None = None, verbose: bool = True) -> None:
    import tempfile
    import trimesh

    def ok(label, cond, detail=""):
        if not cond:
            raise AssertionError(f"FAILED: {label} {detail}")
        if verbose:
            print(f"  ok  {label}{('  ' + detail) if detail else ''}")

    # --- geometry: an asymmetric convex solid (no asset needed), or a real catalog mesh -----------
    if asset:
        from .mesh_collision import load_usd_mesh
        m = load_usd_mesh(OBJECTS_DIR / asset)
        v, f = np.asarray(m.vertices, dtype=float), np.asarray(m.indices).reshape(-1, 3)
        label = asset
    else:
        # Tapered box with two bumps — asymmetric about ALL THREE axes, so every sign resolves from
        # skew and the placement-invariance property below is actually exercised.
        pts = [[sx * 0.10, sy * 0.05 * (1.0 if sx < 0 else 0.45), sz * 0.025 * (1.0 if sx < 0 else 0.45)]
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        pts += [[0.10, 0.0, 0.030], [-0.02, 0.052, 0.0]]
        hull = trimesh.convex.convex_hull(np.array(pts))
        v, f = np.asarray(hull.vertices, dtype=float), np.asarray(hull.faces)
        label = "synthetic tapered box"

    print(f"[grasp_library] self-check on {label} ({len(v)} verts)")
    frame = canonical_frame(v, f)
    print(f"  OBB extents {np.round(frame.extents, 4).tolist()}  rules {frame.sign_rules}  "
          f"drift={frame.drift:.2e} m  ambiguous={frame.ambiguous}")

    ok("frame is a right-handed rotation", abs(np.linalg.det(frame.rotation) - 1.0) < _ORTHO_TOL)
    ok("frame is orthonormal", np.abs(frame.rotation @ frame.rotation.T - np.eye(3)).max() < _ORTHO_TOL)
    _tie = _EXTENT_TIE_RTOL * float(frame.extents.max())
    ok("extents descend (within the tie tolerance)",
       frame.extents[0] >= frame.extents[1] - _tie and frame.extents[1] >= frame.extents[2] - _tie)

    again = canonical_frame(v, f)
    ok("recomputation is bit-identical",
       np.array_equal(frame.rotation, again.rotation) and np.array_equal(frame.translation, again.translation))

    u = frame.to_canonical(v)
    ok("every vertex is inside the OBB",
       bool(np.all(np.abs(u) <= frame.half_extents() + 1.0e-6)),
       f"max overshoot {float((np.abs(u) - frame.half_extents()).max()):.2e} m")
    ok("to_canonical/from_canonical round-trip",
       bool(np.abs(frame.from_canonical(u) - v).max() < 1.0e-9))
    ok("matrix() and inverse_matrix() are inverses",
       bool(np.abs(frame.matrix() @ frame.inverse_matrix() - np.eye(4)).max() < 1.0e-12))

    # The property the whole design rests on: canonical coordinates of the same material points do
    # not move when the asset is re-posed. The frame REPORTS its own instability (frame.drift, from
    # two fixed probes); this re-measures it with independent RANDOM placements — rotations AND
    # metre-scale translations — and holds the record's own number to it. A frame that under-reports
    # its drift is the dangerous failure: candidates would silently sit somewhere else.
    rng = np.random.default_rng(0)
    measured = 0.0
    for _ in range(5):
        q = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        v2 = v @ q.T + rng.normal(size=3)
        probe = canonical_frame(v2, f, verify=False)
        measured = max(measured, float(np.abs(frame.to_canonical(v) - probe.to_canonical(v2)).max()))
    # THE assertion that matters: a frame claiming to be unambiguous must actually be stable under
    # placements it never probed. (The converse — that the reported magnitude bounds the true worst
    # case — is deliberately NOT asserted; drift is a detector, see _DRIFT_PROBES.)
    if not frame.ambiguous:
        ok("a frame reporting unambiguous IS placement-invariant", measured < _DRIFT_TOL,
           f"independently measured {measured:.2e} m over 5 random placements")
    elif verbose:
        print(f"  --  invariance not asserted: frame is ambiguous (reported drift "
              f"{frame.drift:.2e} m, independently measured {measured:.2e} m, rules "
              f"{frame.sign_rules}) — a near-degenerate OBB, reported not hidden")
    ok("the ambiguous flag agrees with the reported drift",
       frame.ambiguous or frame.drift <= _DRIFT_TOL)

    # --- record round-trip ------------------------------------------------------------------------
    h = frame.half_extents()
    cands = [
        make_candidate(frame, "top_across_y", [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
                       width=min(2 * h[1] + 0.01, MAX_JAW_WIDTH), source="selfcheck",
                       seat_mode="span_flush", span=float(2 * h[2]), seat_depth=-0.02,
                       labels=("synthetic",)),
        make_candidate(frame, "side_across_z", [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],
                       width=min(2 * h[2] + 0.01, MAX_JAW_WIDTH), source="selfcheck",
                       seat_mode="retreated", span=float(2 * h[1]), seat_depth=-0.01),
    ]
    ok("face buckets derive from the approach",
       [c.face for c in cands] == ["+z", "+y"], f"{[c.face for c in cands]}")
    ok("candidate accessors match the transform columns",
       bool(np.allclose(cands[0].approach, [0, 0, -1]) and np.allclose(cands[0].jaw_axis, [0, 1, 0])))
    ok("quality fields start empty",
       all(v is None for v in cands[0].quality.values()) and cands[0].quality_source is None
       and not cands[0].evaluated)

    rec = GraspRecord(name="_selfcheck", kind="ycb_mesh", frame=frame, candidates=tuple(cands),
                      file=asset, mesh_sha1=mesh_digest(v, f), generator="grasp_library._selfcheck")
    with tempfile.TemporaryDirectory() as d:
        p = write_record(rec, path=Path(d) / "_selfcheck.json")
        back = record_from_dict(json.loads(p.read_text()), name="_selfcheck")
        ok("record survives write -> read",
           back.name == rec.name and len(back.candidates) == len(rec.candidates)
           and np.abs(back.frame.rotation - rec.frame.rotation).max() < 1e-9
           and np.abs(back.candidates[0].transform - rec.candidates[0].transform).max() < 1e-9)
        ok("mesh digest detects a stale record",
           check_mesh_current(back, v, f) and not check_mesh_current(back, v * 1.001, f))

    # --- placement composition ----------------------------------------------------------------------
    world = body_pose([0.3, -0.5, 0.8], yaw=0.7)
    gw = rec.grasp_in_world(cands[0], world)
    ok("world grasp stays a rigid transform",
       abs(np.linalg.det(gw[:3, :3]) - 1.0) < 1e-9 and np.allclose(gw[3], [0, 0, 0, 1]))
    # The grasp centre must track the body exactly: same point via the mesh path and the pose path.
    p_asset = frame.from_canonical(cands[0].position)[0]
    ok("grasp centre follows the placement",
       bool(np.abs(gw[:3, 3] - (world[:3, :3] @ p_asset + world[:3, 3])).max() < 1e-9))

    # --- validation must REJECT each way a record can be wrong ---------------------------------------
    base = record_to_dict(rec)

    def rejects(what, mutate):
        import copy
        bad = copy.deepcopy(base)
        mutate(bad)
        try:
            validate_record(bad, name="_selfcheck")
        except GraspSchemaError:
            if verbose:
                print(f"  ok  rejects {what}")
            return
        raise AssertionError(f"FAILED: validation ACCEPTED {what}")

    def _t(d):
        return d["grasps"]["candidates"][0]["transform"]

    rejects("a wrong schema_version", lambda d: d.__setitem__("schema_version", 99))
    rejects("a cloth/bag record (out of scope)", lambda d: d["object"].__setitem__("kind", "cloth"))
    rejects("a filename/object.name mismatch", lambda d: d["object"].__setitem__("name", "other"))
    rejects("ascending extents", lambda d: d["object"].__setitem__("extents", [0.01, 0.02, 0.03]))
    rejects("a non-orthonormal frame",
            lambda d: d["object"]["frame"].__setitem__("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 2]]))
    rejects("a mirrored (left-handed) frame",
            lambda d: d["object"]["frame"].__setitem__("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    rejects("a changed pose convention",
            lambda d: d["gripper"].__setitem__("pose_convention", "something_else"))
    rejects("a width past the jaw limit",
            lambda d: d["grasps"]["candidates"][0].__setitem__("width", MAX_JAW_WIDTH + 0.01))
    rejects("a zero width", lambda d: d["grasps"]["candidates"][0].__setitem__("width", 0.0))
    rejects("a missing source tag", lambda d: d["grasps"]["candidates"][0].__setitem__("source", ""))
    rejects("a missing seat_mode (pre-v2 record)",
            lambda d: d["grasps"]["candidates"][0].pop("seat_mode"))
    rejects("an unknown seat_mode",
            lambda d: d["grasps"]["candidates"][0].__setitem__("seat_mode", "hovering"))
    rejects("a missing span (pre-v3 record)",
            lambda d: d["grasps"]["candidates"][0].pop("span"))
    rejects("a non-numeric span",
            lambda d: d["grasps"]["candidates"][0].__setitem__("span", "deep"))
    rejects("a missing seat_depth (pre-v4 record)",
            lambda d: d["grasps"]["candidates"][0].pop("seat_depth"))
    rejects("a seat_depth outside the pad-reachable range",
            lambda d: d["grasps"]["candidates"][0].__setitem__("seat_depth", 0.03))
    rejects("an unknown face bucket", lambda d: d["grasps"]["candidates"][0].__setitem__("face", "+w"))
    rejects("a face bucket that contradicts the transform",
            lambda d: d["grasps"]["candidates"][0].__setitem__("face", "-x"))
    rejects("a non-rigid transform", lambda d: _t(d)[0].__setitem__(0, 2.0))
    rejects("a broken homogeneous row", lambda d: _t(d)[3].__setitem__(3, 0.0))
    rejects("duplicate candidate ids",
            lambda d: d["grasps"]["candidates"].__setitem__(1, dict(d["grasps"]["candidates"][0])))
    rejects("an unknown quality field",
            lambda d: d["grasps"]["candidates"][0]["quality"].__setitem__("made_up", 1.0))
    rejects("a quality value with no quality_source",
            lambda d: d["grasps"]["candidates"][0]["quality"].__setitem__("object_in_gripper", 1.0))
    validate_record(base, name="_selfcheck")
    print("  ok  the unmutated record validates")
    print("[grasp_library] self-check PASSED")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true", help="run the invariant + validation checks")
    ap.add_argument("--asset", default=None,
                    help="run the geometry checks on a real asset (path under assets/objects, "
                         "e.g. ycb/banana.usd) instead of the synthetic solid")
    ap.add_argument("--list", action="store_true", help="list catalog names that have a grasp record")
    args = ap.parse_args()
    if args.list:
        names = available_grasps()
        print(f"{len(names)} record(s) in {GRASPS_DIR}:")
        for n in names:
            r = load_grasps(n)
            print(f"  {n:24s} {len(r.candidates):3d} candidate(s)  "
                  f"extents {np.round(r.frame.extents, 4).tolist()}")
    if args.selfcheck:
        _selfcheck(asset=args.asset)
    if not args.selfcheck and not args.list:
        ap.error("give --selfcheck and/or --list")
