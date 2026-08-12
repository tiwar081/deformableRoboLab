"""SHARED — executable proof of the pass contract. Read-only for pass authors.

    .venv/bin/python -m deformableManipulationTools.grasp_passes selfcheck

Everything a parallel pass author is being asked to rely on is asserted here with throwaway
in-memory passes: two passes cannot clobber each other, re-running never accumulates candidates, a
consumer sees its producer's output, and the merge refuses the ways sidecars can disagree. It runs
against a redirected sidecar directory and merges with ``write=False``, so it never touches a real
sidecar or record.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..grasp_library import MAX_JAW_WIDTH, make_candidate
from . import base
from .base import GraspPass, PassContext, PassError, PassOutput, run_pass, sidecar_path
from .catalog import load_asset
from .merge import MergeError, merge_asset

ASSET = "sugar_box"      # small, rigid, unambiguous frame — the checks are about the machinery


class _Producer(GraspPass):
    name, source, version = "_sc_producer", "sc_a", 1

    def __init__(self, n=2):
        self.n = n

    def run(self, ctx):
        h = ctx.asset.half_extents
        return PassOutput(candidates=[
            make_candidate(ctx.frame, f"g{i}", [0.0, 0.0, 0.0], [0, -1, 0], [0, 0, 1],
                           width=min(2 * h[2] + 0.004 + 0.001 * i, MAX_JAW_WIDTH),
                           source=self.source, seat_mode="span_flush", span=0.02, seat_depth=-0.02, labels=("synthetic",))
            for i in range(self.n)])


class _Other(GraspPass):
    """A second, independent producer — the parallel-agent case."""
    name, source, version = "_sc_other", "sc_b", 1

    def run(self, ctx):
        h = ctx.asset.half_extents
        return PassOutput(candidates=[
            make_candidate(ctx.frame, "g0", [0.0, 0.0, 0.0], [0, 0, -1], [0, 1, 0],
                           width=min(2 * h[1] + 0.004, MAX_JAW_WIDTH), source=self.source,
                           seat_mode="span_flush", span=0.02, seat_depth=-0.02)])


class _Annotator(GraspPass):
    """A consumer: reads the producer's candidates and scores them."""
    name, source, version = "_sc_annotator", "sc_score", 1
    requires = ("_sc_producer",)

    def run(self, ctx):
        return PassOutput(annotations={
            c.id: {"quality": {"object_in_gripper": 1.0}, "quality_source": "selfcheck",
                   "labels": ("scored",)}
            for c in ctx.upstream})


def run_selfcheck(verbose: bool = True) -> None:
    def ok(label, cond, detail=""):
        if not cond:
            raise AssertionError(f"FAILED: {label} {detail}")
        if verbose:
            print(f"  ok  {label}{('  ' + detail) if detail else ''}")

    def rejects(label, fn, exc=(MergeError, PassError)):
        try:
            fn()
        except exc:
            if verbose:
                print(f"  ok  rejects {label}")
            return
        raise AssertionError(f"FAILED: ACCEPTED {label}")

    real, tmp = base.PASSES_DIR, Path(tempfile.mkdtemp(prefix="grasp-selfcheck-"))
    base.PASSES_DIR = tmp
    try:
        print(f"[grasp_passes] self-check on {ASSET!r} (sidecars redirected to a temp dir)")
        asset = load_asset(ASSET)
        prod, other, ann = _Producer(), _Other(), _Annotator()

        r1 = run_pass(prod, ASSET, verbose=False)
        ok("a pass writes its own sidecar", r1["status"] == "written" and
           sidecar_path(prod.name, ASSET).exists())
        ok("the sidecar lives under the pass's own directory",
           sidecar_path(prod.name, ASSET).parent.name == prod.name)

        r2 = run_pass(prod, ASSET, verbose=False)
        ok("re-running is a no-op when nothing changed", r2["status"] == "skipped")
        ok("re-running with --force is idempotent", base.check_idempotent(prod, ASSET))

        rec = merge_asset(ASSET, write=False, verbose=False)
        ok("merge composes one pass", len(rec.candidates) == 2)
        run_pass(prod, ASSET, force=True, verbose=False)
        rec = merge_asset(ASSET, write=False, verbose=False)
        ok("re-run + merge does NOT accumulate candidates", len(rec.candidates) == 2,
           "dedup by source replaces the pass's whole contribution")

        # A pass shrinking its output must shrink the record, or stale candidates would linger.
        run_pass(_Producer(n=1), ASSET, force=True, verbose=False)
        rec = merge_asset(ASSET, write=False, verbose=False)
        ok("a pass emitting fewer candidates shrinks the record", len(rec.candidates) == 1)
        run_pass(prod, ASSET, force=True, verbose=False)

        run_pass(other, ASSET, verbose=False)
        rec = merge_asset(ASSET, write=False, verbose=False)
        ok("two independent passes compose", len(rec.candidates) == 3)
        ok("ids are namespaced by source, so passes cannot collide on them",
           sorted(c.id for c in rec.candidates) == ["sc_a/g0", "sc_a/g1", "sc_b/g0"])
        ok("each pass's candidates keep its own source tag",
           {c.source for c in rec.candidates} == {"sc_a", "sc_b"})

        run_pass(ann, ASSET, verbose=False)
        rec = merge_asset(ASSET, write=False, verbose=False)
        scored = [c for c in rec.candidates if c.evaluated]
        ok("a consumer pass sees its producer's candidates and annotates them",
           len(scored) == 2 and all(c.quality["object_in_gripper"] == 1.0 for c in scored))
        ok("annotations reach only the producer's candidates",
           {c.id for c in scored} == {"sc_a/g0", "sc_a/g1"})
        ok("annotated candidates keep their producer's source",
           {c.source for c in scored} == {"sc_a"})
        ok("annotation merges labels rather than replacing them",
           all("synthetic" in c.labels and "scored" in c.labels for c in scored))
        ok("the merged record still passes the record schema", len(rec.candidates) == 3)

        # --- the ways a pass can break the contract ------------------------------------------------
        class _WrongSource(GraspPass):
            name, source = "_sc_wrong", "sc_wrong"

            def run(self, ctx):
                return [make_candidate(ctx.frame, "x", [0, 0, 0], [0, 0, -1], [0, 1, 0],
                                       width=0.02, source="sc_a", seat_mode="span_flush", span=0.02, seat_depth=-0.02)]   # not its own tag
        rejects("a pass emitting another pass's source tag",
                lambda: run_pass(_WrongSource(), ASSET, verbose=False))

        class _Dup(GraspPass):
            name, source = "_sc_dup", "sc_dup"

            def run(self, ctx):
                c = make_candidate(ctx.frame, "same", [0, 0, 0], [0, 0, -1], [0, 1, 0],
                                   width=0.02, source=self.source, seat_mode="span_flush", span=0.02, seat_depth=-0.02)
                return [c, c]
        rejects("duplicate candidate ids within one pass",
                lambda: run_pass(_Dup(), ASSET, verbose=False))

        class _Orphan(GraspPass):
            name, source = "_sc_orphan", "sc_orphan"

            def run(self, ctx):
                return PassOutput(annotations={"sc_a/g0": {"labels": ("x",)}})   # no requires
        rejects("annotating a candidate the pass never declared as upstream",
                lambda: run_pass(_Orphan(), ASSET, verbose=False))

        class _MissingDep(GraspPass):
            name, source, requires = "_sc_missing", "sc_missing", ("_sc_never_ran",)

            def run(self, ctx):
                return PassOutput()
        rejects("a consumer whose producer has not run",
                lambda: run_pass(_MissingDep(), ASSET, verbose=False))

        # Two passes claiming one source tag: the merge must refuse rather than silently blend them.
        class _Clash(GraspPass):
            name, source = "_sc_clash", "sc_a"

            def run(self, ctx):
                return [make_candidate(ctx.frame, "c", [0, 0, 0], [0, 0, -1], [0, 1, 0],
                                       width=0.02, source=self.source, seat_mode="span_flush", span=0.02, seat_depth=-0.02)]
        run_pass(_Clash(), ASSET, verbose=False)
        rejects("two passes claiming the same source tag",
                lambda: merge_asset(ASSET, write=False, verbose=False))
        shutil.rmtree(tmp / "_sc_clash")

        # A sidecar computed from different geometry must not be merged with current ones.
        import json
        p = sidecar_path(prod.name, ASSET)
        data = json.loads(p.read_text())
        data["record"]["object"]["mesh_sha1"] = "0" * 40
        p.write_text(json.dumps(data))
        rejects("a sidecar computed from a different mesh",
                lambda: merge_asset(ASSET, write=False, verbose=False))
        run_pass(prod, ASSET, force=True, verbose=False)

        data = json.loads(p.read_text())
        rot = np.asarray(data["record"]["object"]["frame"]["rotation"])
        data["record"]["object"]["frame"]["rotation"] = np.roll(rot, 1, axis=0).tolist()
        p.write_text(json.dumps(data))
        rejects("a sidecar carrying a different canonical frame",
                lambda: merge_asset(ASSET, write=False, verbose=False))

        print("[grasp_passes] self-check PASSED")
    finally:
        base.PASSES_DIR = real
        shutil.rmtree(tmp, ignore_errors=True)
