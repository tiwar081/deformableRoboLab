"""``obb_bucket`` pass — assign every candidate its OBB face bucket, whoever produced it.

An annotator, not a producer: it emits no grasps. For each candidate in its upstream it re-derives
the face bucket from the pose's own approach column (:mod:`.bucket`), cross-checks it against the
``face`` the record stores, and writes the bucket back as labels — ``face:<bucket>``, plus
``face_borderline`` / ``face_alt:<bucket>`` for a diagonal approach and ``face_ambiguous`` on an
ambiguous canonical frame. Those labels are what :mod:`.pruning` reads at runtime to throw out whole
buckets from an object's placement before anything expensive runs.

**Why bucket at all when the schema already stores ``face``?** Because the stored field is a single
nearest-axis answer with no notion of "and it was nearly this other face too", and pruning on it
alone would silently drop diagonal grasps that are perfectly reachable. This pass is where that
distinction is computed, once, for every producer's output at the same time — so a downstream
consumer never has to know which pass a candidate came from, or whether that pass thought about
buckets at all.

**Upstream is discovered, not hard-coded** — via :class:`base.DynamicUpstreamPass`, shared with
every other discovering consumer. ``requires`` resolves to the producers that have a sidecar for the
asset being run. Passes are written concurrently by separate agents, so naming them in a frozen
tuple would mean this pass silently skipped a producer that landed later — and declaring one that
has produced nothing for an asset would fail the run outright, since the harness treats a missing
upstream sidecar as an error. Cycle exclusion lives in ``base.discover_producers``.
"""
from __future__ import annotations

from ..base import DynamicUpstreamPass, PassContext, PassOutput
from .bucket import (AMBIGUOUS_LABEL, BORDERLINE_LABEL, BORDERLINE_RTOL, bucket_of, face_labels,
                     plausible_buckets)
from .pruning import prune_candidates, prune_record, surviving_buckets

__all__ = ["PASS", "ObbBucketPass", "bucket_of", "plausible_buckets", "face_labels",
           "surviving_buckets", "prune_candidates", "prune_record", "BORDERLINE_RTOL"]


class ObbBucketPass(DynamicUpstreamPass):
    name = "obb_bucket"
    source = "obb_bucket"        # owned but unused: this pass emits annotations, never candidates
    version = 1
    kinds = ()

    def run(self, ctx: PassContext) -> PassOutput:
        annotations, borderline = {}, 0
        for c in ctx.upstream:
            derived = bucket_of(ctx.frame, c.approach)
            if derived != c.face:
                # The record schema re-derives `face` the same way, so a disagreement means the
                # candidate reached here without validation — worth stopping for, not labelling over.
                raise ValueError(
                    f"{ctx.name}: candidate {c.id!r} stores face {c.face!r} but its approach "
                    f"{c.approach.round(4).tolist()} enters {derived!r}")
            labels = face_labels(ctx.frame, c.approach)
            borderline += BORDERLINE_LABEL in labels
            annotations[c.id] = {"labels": labels}
        srcs = ", ".join(sorted({c.source for c in ctx.upstream})) or "nothing"
        return PassOutput(
            annotations=annotations,
            notes=(f"bucketed {len(annotations)} candidate(s) from [{srcs}]; {borderline} "
                   f"borderline, frame ambiguous={ctx.frame.ambiguous}"))


PASS = ObbBucketPass()
