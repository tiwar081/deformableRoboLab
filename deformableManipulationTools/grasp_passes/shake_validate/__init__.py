"""``shake_validate`` pass — fill a candidate's quality fields by actually grasping it.

A consumer/annotator: it produces no candidates of its own. For every upstream candidate it builds
one simulation, puts a FREE-FLOATING Franka hand on the candidate's pose, closes it through the real
force-feedback admittance controller, switches gravity on, shakes the hand the way ACRONYM shakes
its, and writes down what happened — into the five ``grasp_library.QUALITY_FIELDS``, which are
ACRONYM's ``grasps/qualities/flex/*`` names verbatim, with ``quality_source`` naming this pass.

Three design commitments, each of them load-bearing:

* **A free-floating gripper, not the arm.** The question is whether the pinch holds, not whether
  some arm can reach it, and running the arm would fold reachability and the open end-effector
  rotation projection question into the answer. See :mod:`.hand`.
* **The real closing dynamics.** The jaws are closed by the centralized :class:`grip.GripController`
  driving the real Franka finger actuators against the real harvested squeeze — never by a scripted
  width. The rig therefore takes the split MuJoCo+VBD route for every object, rigid ones included,
  because that is the only route on which the controller has a force signal to close against. See
  :mod:`.rig`.
* **Failures are recorded, not deleted.** A candidate that drops its object gets
  ``object_in_gripper = 0`` and a ``dropped`` label; a candidate whose trial diverged gets
  ``object_in_gripper = 0``, a ``diverged`` label, null motions and the reason in its notes. The
  pass never removes a candidate — it does not own any (they belong to their producer), and a
  measured failure is the most useful thing this pass can say.

**Upstream is discovered, not hard-coded** — via :class:`base.DynamicUpstreamPass`, shared with
every other discovering consumer. ``requires`` resolves to the producers that have a sidecar for the
asset being run. A frozen ``("fixture",)`` would have meant this pass validated the hand-placed
placeholders for ever and silently ignored every generator that landed later, which is the one
failure mode that looks like success.

**v4 narrows WHICH candidates get trials, not how a trial runs** (spec:
docs/trajPipeline/grasp-passes.md "Pass v4"). ``retreated`` (weak) and ``seat_blocked`` candidates
are excluded from trial selection outright — weak options are not grasps (3% hold at n=628) and
blocked ones are merge-discarded, so a trial on either is GPU spent confirming a settled verdict;
neither gets a new annotation (nothing was *chosen to run*, which is different from "unreachable"),
though a historical annotation from this pass's own previous sidecar is carried forward untouched.
Annotation is INCREMENTAL: a selectable candidate whose id + pose + width match the previous
sidecar (notes tagged ``[pose:<digest>]``) carries its annotation forward and is not re-simulated —
without this, any new producer landing would change the upstream digest and re-run an entire
asset's trial matrix to reproduce numbers already on disk. The trial procedure, pre-check, rig and
protocol are unchanged from v3, which is why v3 sidecars stay mergeable
(``merge.COMPATIBLE_VERSIONS``).
"""
from __future__ import annotations

import hashlib

import numpy as np

# pregrasp_collision moved to grasp_library (2026-08-06): pad_seat's collision-aware retreat runs
# the SAME hand-hull test at seat time, and two definitions of "is the hand inside the object"
# would let the seat and this pre-check disagree.
from ...grasp_library import (GraspSchemaError, QUALITY_FIELDS, SEAT_BLOCKED_LABEL, is_weak,
                              pregrasp_collision)
from ..base import DynamicUpstreamPass, PassContext, PassOutput, read_sidecar
from ..catalog import SUPPORTED_KINDS
from . import protocol as P
from .rig import TrialResult, run_trial

# The notes token that binds an annotation to the pose it measured. Written by v4 on every
# annotation it creates or carries; an annotation with NO token is trusted as-is (the v3
# transition case — safe because shake sidecars are deleted at every pose migration, so
# everything on disk postdates the last pose move).
_POSE_TOKEN_PREFIX = "[pose:"


def pose_digest(candidate) -> str:
    """12-hex-char identity of WHERE a candidate grasps: sha1 of the 4x4 transform rounded to
    9 decimals (the same rounding :func:`base.upstream_digest` uses) plus the jaw width. Two
    candidates sharing an id but not a digest were moved by their producer, so the stored
    measurement no longer describes them and the trial must re-run."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.round(np.asarray(candidate.transform, dtype=float), 9)).tobytes())
    h.update(f"|{candidate.width:.9f}".encode())
    return h.hexdigest()[:12]


class ShakeValidatePass(DynamicUpstreamPass):
    name = "shake_validate"
    source = "shake_validate"
    # v2 (2026-08-05): pre-grasp collision check added, so poses whose hand is inside
    # the object are skipped rather than simulated and reported as drops.
    # v3 (2026-08-07): the pre-check and the trial pose the jaws at the PRE-SHAPED aperture
    # (candidate width + PREGRASP_MARGIN) instead of fully open — the trajectory pre-shapes
    # the gripper before the approach, so this is the procedure that actually executes.
    # v4 (2026-08-11): trial selection narrows (retreated/weak and seat_blocked candidates are
    # never trialed) + INCREMENTAL annotation — unchanged id+pose+width carry their previous
    # annotation forward ([pose:<digest>] notes token); the trial procedure itself is unchanged,
    # so v3 sidecars stay valid via merge.COMPATIBLE_VERSIONS.
    version = 4
    kinds = SUPPORTED_KINDS       # every kind the scheme covers; cloth/bags are out of scope upstream

    def applies_to(self, name: str, kind: str) -> bool:
        """Only assets that actually have CANDIDATES to validate. An annotator with no upstream has
        nothing to say, and saying it as a per-asset PassError for most of the catalog would bury
        the real failures. Checked against real candidate counts, not just sidecar existence: a
        sidecar can exist and carry none, and a trial run over nothing is pure cost — and here a
        wasted trial is ~40 s, not microseconds."""
        if not super().applies_to(name, kind):
            return False
        return any(self._candidate_count(p, name) for p in self.requires)

    @staticmethod
    def _candidate_count(pass_name: str, asset: str) -> int:
        try:
            side = read_sidecar(pass_name, asset)
        except Exception:                         # noqa: BLE001 - a broken sidecar is not ours to raise on
            return 0
        return 0 if side is None else len(side["record"].candidates)

    def run(self, ctx: PassContext) -> PassOutput:
        # This pass's OWN previous sidecar is the incremental baseline. None (or unreadable — a
        # half-written file must degrade to "run everything fresh", never abort the asset) means
        # no history: every selectable candidate is trialed.
        try:
            prior = read_sidecar(self.name, ctx.name)
        except GraspSchemaError:
            prior = None
        prior_ann = prior["annotations"] if prior is not None else {}

        annotations = {}
        summary = []
        carried = trialed = precheck_skipped = weak = blocked = 0
        verts = None                              # geometry loads lazily: a zero-trial run needs none
        for candidate in ctx.upstream:
            # NOT CHOSEN TO RUN — which is different from "unreachable", so no new annotation is
            # written (not even a skip). A historical annotation from the previous sidecar (v3
            # measured these) is carried forward verbatim until it naturally ages out; carrying is
            # keyed on CURRENT upstream ids only, so nothing is ever resurrected.
            if is_weak(candidate):                # retreated = weak grasp option, 3% hold at n=628
                weak += 1
                if candidate.id in prior_ann:
                    annotations[candidate.id] = prior_ann[candidate.id]
                continue
            if SEAT_BLOCKED_LABEL in candidate.labels:   # merge-discarded; a trial proves nothing
                blocked += 1
                if candidate.id in prior_ann:
                    annotations[candidate.id] = prior_ann[candidate.id]
                continue

            # INCREMENTAL: same id + same pose/width (or a token-less v3 annotation — trusted, see
            # the module docstring) -> the stored measurement still describes this candidate.
            token = f"{_POSE_TOKEN_PREFIX}{pose_digest(candidate)}]"
            prev = prior_ann.get(candidate.id)
            if prev is not None:
                notes = prev.get("notes", "")
                if token in notes or _POSE_TOKEN_PREFIX not in notes:
                    ann = dict(prev)
                    if token not in notes:
                        ann["notes"] = f"{notes} {token}".strip()
                    annotations[candidate.id] = ann
                    carried += 1
                    continue

            if verts is None:
                verts = ctx.asset.canonical_vertices()
            # PRE-GRASP COLLISION CHECK, before spending ~40 s on a trial. A pose whose hand is
            # already inside the object cannot be reached, and simulating it would report a drop
            # that is indistinguishable from a bad pinch.
            hits = pregrasp_collision(np.asarray(candidate.transform, dtype=float), verts,
                                      width=candidate.width)
            if hits:
                ann = self._skip_annotation(candidate, hits)
                summary.append(f"{candidate.id}: skipped (hand inside object)")
                precheck_skipped += 1
            else:
                world_from_grasp = np.asarray(candidate.transform, dtype=float)
                # world_from_body is the IDENTITY by construction (rig.spawn_object puts the asset
                # frame on the world frame), so the canonical->asset step is the whole composition.
                world_from_grasp = ctx.asset.frame.inverse_matrix() @ world_from_grasp
                result = run_trial(ctx.asset, world_from_grasp, width=candidate.width)
                ann = self._annotation(candidate, result)
                summary.append(self._summary(candidate, result))
                trialed += 1
            ann["notes"] = f"{ann['notes']} {token}".strip()
            annotations[candidate.id] = ann
        note = (f"{carried} carried forward, {trialed} trialed, {precheck_skipped} pre-check "
                f"skipped, {weak} skipped weak (retreated), {blocked} skipped blocked")
        if summary:
            note += "; " + "; ".join(summary)
        return PassOutput(annotations=annotations, notes=note)

    def _skip_annotation(self, candidate, hits: dict) -> dict:
        """A pre-check rejection carries NO quality and NO quality_source.

        It is not a measurement — nothing was simulated — and writing ``object_in_gripper = 0`` here
        would put an unmeasured zero in the same field the shake protocol fills, where no consumer
        could tell them apart. Labels record it instead, so the candidate reads as *not scored*
        rather than *scored bad*, matching how the rest of the codebase treats an unevaluable
        result."""
        where = ", ".join(f"{k}:{n}" for k, n in sorted(hits.items()))
        return {
            "labels": ["shake_skipped", "pregrasp_collision"],
            "notes": f"{candidate.notes} [{self.name}: NOT simulated — at the pre-grasp opening the "
                     f"object intrudes into the hand ({where} vertices inside), so this pose is "
                     f"unreachable rather than a failed grasp]".strip(),
        }

    # ---- turning a trial into an annotation -----------------------------------------------------
    def _annotation(self, candidate, result: TrialResult) -> dict:
        # Keyed off QUALITY_FIELDS itself rather than by repeating the five names, so the schema
        # stays the single source of truth and a renamed field is an error here, not a silent
        # unfilled slot.
        measured = {
            "object_in_gripper": float(result.object_in_gripper),
            "object_motion_during_closing_linear":
                P.quantize(result.motion_closing_linear, P.ROUND_LINEAR),
            "object_motion_during_closing_angular":
                P.quantize(result.motion_closing_angular, P.ROUND_ANGULAR),
            "object_motion_during_shaking_linear":
                P.quantize(result.motion_shaking_linear, P.ROUND_LINEAR),
            "object_motion_during_shaking_angular":
                P.quantize(result.motion_shaking_angular, P.ROUND_ANGULAR),
        }
        missing = set(QUALITY_FIELDS) - set(measured)
        if missing or set(measured) - set(QUALITY_FIELDS):
            raise ValueError(f"{self.name}: quality fields do not match grasp_library.QUALITY_FIELDS "
                             f"(missing {sorted(missing)}, extra "
                             f"{sorted(set(measured) - set(QUALITY_FIELDS))})")
        quality = {k: measured[k] for k in QUALITY_FIELDS}
        labels = ["shake_validated"]
        labels.append("retained" if result.object_in_gripper >= 0.5 else "dropped")
        if not result.ok:
            labels.append("diverged")
        return {
            "quality": quality,
            "quality_source": self.quality_source,
            "labels": labels,
            # Preserve whatever the producer wrote: the merge REPLACES notes with the annotation's.
            "notes": f"{candidate.notes} [{self.quality_source}: {self._detail(result)}]".strip(),
        }

    @property
    def quality_source(self) -> str:
        """What measured the numbers, named so a reader can reproduce them: this pass at this
        version. Bumping ``version`` (which is what changing the protocol requires) therefore
        re-labels the measurements instead of silently mixing two protocols in one record."""
        return f"{self.name}@{self.version}"

    def _detail(self, result: TrialResult) -> str:
        if not result.ok:
            return result.note
        # The close outcome is stated ONCE: the rig's own note already spells out a close that timed
        # out, so only a successful one needs its duration adding here.
        gripped = f", gripped in {result.close_duration:.1f} s" if result.grip_converged else ""
        return (f"{result.note}; target {result.force_target:.1f} N on {result.object_mass * 1e3:.0f} g "
                f"(mu_eff {result.mu_effective:.2f}){gripped}, "
                f"final squeeze {result.final_squeeze:.1f} N, "
                f"shake applied at {result.shake_ratio_linear * 100:.0f}%/"
                f"{result.shake_ratio_angular * 100:.0f}% of command")

    def _summary(self, candidate, result: TrialResult) -> str:
        verdict = "diverged" if not result.ok else ("held" if result.object_in_gripper >= 0.5
                                                    else "dropped")
        if result.motion_shaking_linear is None:
            return f"{candidate.id}: {verdict}"
        return (f"{candidate.id}: {verdict} "
                f"(shake {result.motion_shaking_linear * 1e3:.1f} mm / "
                f"{np.degrees(result.motion_shaking_angular):.1f} deg)")


PASS = ShakeValidatePass()
