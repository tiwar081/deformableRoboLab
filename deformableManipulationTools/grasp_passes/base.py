"""SHARED — the grasp-pass interface and its per-pass sidecar. Read-only for pass authors.

A **pass** computes something about one catalog asset's grasps. Several passes are developed in
parallel by separate agents, so the contract is built around never sharing a mutable file:

  * a pass writes ONLY its own sidecar, ``assets/objects/grasps/_passes/<pass>/<asset>.json``;
  * the shared record at ``assets/objects/grasps/<asset>.json`` is a BUILD ARTIFACT produced by
    :mod:`.merge` from those sidecars — no pass ever opens it;
  * re-running a pass REPLACES its sidecar rather than appending, so running twice is a no-op.

Two kinds of pass, both expressed through :class:`PassOutput`:

  * **producers** emit candidates (tagged with the pass's ``source``);
  * **consumers/annotators** read ``ctx.upstream`` — the candidates of the passes they declare in
    ``requires`` — and emit ``annotations`` keyed by candidate id (quality numbers, extra labels).

The sidecar embeds a COMPLETE ``grasp_library`` record for the pass's own candidates, so sidecars
are validated by the same :func:`grasp_library.validate_record` as the final record and no second
schema exists. ``grasp_library`` itself is frozen — this module never modifies it.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..grasp_library import (GRASPS_DIR, GraspCandidate, GraspRecord, GraspSchemaError,
                             QUALITY_FIELDS, ObjectFrame, record_from_dict, record_to_dict,
                             validate_record)
from .catalog import CatalogAsset, load_asset

SIDECAR_VERSION = 1
# Per-pass sidecars live UNDER the grasps dir but in a "_"-prefixed subdirectory: grasp_library's
# available_grasps() globs "*.json" non-recursively, so sidecars can never be mistaken for records,
# and "_" matches the repo convention for directories asset scans skip.
PASSES_DIR = GRASPS_DIR / "_passes"
# Separator between a pass's source tag and the candidate id it chose. The harness applies it, so
# two passes developed independently cannot collide on ids.
ID_SEP = "/"


class PassError(RuntimeError):
    """A pass violated the contract (bad output, wrong source tag, non-deterministic)."""


# =================================================================================================
# What a pass receives and returns
# =================================================================================================
@dataclass(frozen=True)
class PassContext:
    """The read-only input to :meth:`GraspPass.run`.

    ``asset`` carries the rest geometry and the canonical frame (never recompute the frame — poses
    from different frames are not comparable). ``upstream`` holds the candidates produced by the
    passes named in ``GraspPass.requires``, already id-namespaced exactly as they will appear in the
    merged record, so an annotation keyed on ``candidate.id`` lands on the right one."""
    asset: CatalogAsset
    upstream: tuple = ()

    @property
    def name(self) -> str:
        return self.asset.name

    @property
    def kind(self) -> str:
        return self.asset.kind

    @property
    def frame(self) -> ObjectFrame:
        return self.asset.frame


@dataclass
class PassOutput:
    """What a pass returns. Either list may be empty — a producer fills ``candidates``, an
    annotator fills ``annotations`` (``{candidate_id: {"quality": {...}, "labels": [...]}}``)."""
    candidates: list = field(default_factory=list)
    annotations: dict = field(default_factory=dict)
    notes: str = ""


class GraspPass(ABC):
    """Base class for a grasp pass. Subclass it in your OWN directory and export ``PASS``.

    Class attributes to set:
      ``name``      unique pass id; also its sidecar directory name.
      ``source``    the source tag stamped on every candidate it emits. Must be unique across
                    passes — the merge deduplicates by it, so two passes sharing a tag would
                    silently overwrite each other.
      ``version``   bump whenever the output would change; invalidates cached sidecars.
      ``requires``  names of passes whose candidates this one consumes. Sets both the run ORDER and
                    what appears in ``ctx.upstream``.
      ``kinds``     catalog kinds this pass handles; empty = every supported kind.

    Contract (checked where it can be, trusted where it cannot):
      * **Deterministic** — the same asset must give the same output, including candidate ids.
        Seed any sampling from the asset, never from the clock or an unseeded RNG.
        ``--check-idempotent`` runs a pass twice and diffs the sidecars.
      * **Self-contained** — write nothing except through the returned :class:`PassOutput`.
      * **Frame-faithful** — emit poses in ``ctx.frame``; the harness does not transform them.
    """
    name: str = ""
    source: str = ""
    version: int = 1
    requires: tuple = ()
    kinds: tuple = ()
    # True on a pass that DISCOVERS its upstream instead of naming it (see DynamicUpstreamPass).
    # Read by discover_producers to skip such passes without touching their `requires`, which is what
    # keeps the discovery of any number of them from recursing. A plain class attribute on purpose:
    # reading it must never evaluate anything.
    dynamic_upstream: bool = False

    def applies_to(self, name: str, kind: str) -> bool:
        """Whether this pass has anything to say about this asset. Default: the ``kinds`` filter.

        Deliberately takes only the NAME and KIND, not a context: resolving an asset runs the OBB
        search three times, so a pass that skips most of the catalog must be able to say so before
        any geometry is loaded."""
        return not self.kinds or kind in self.kinds

    @abstractmethod
    def run(self, ctx: PassContext) -> PassOutput:
        """Compute this pass's contribution for one asset."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} source={self.source!r} v{self.version}>"


def _statically_requires(pass_, target: str, registry: dict, seen=None) -> bool:
    """Whether ``pass_`` requires ``target``, following only STATICALLY declared ``requires``.

    Never evaluates a dynamic pass's ``requires``. That is the whole trick: two passes that both
    compute their upstream would otherwise each ask the other for its requires and recurse forever
    (measured — it blew the stack the first time a second dynamic pass was added). A dynamic pass is
    excluded from every dynamic pass's upstream anyway, so its declarations cannot matter here."""
    if getattr(pass_, "dynamic_upstream", False):
        return False
    seen = set() if seen is None else seen
    for dep in getattr(pass_, "requires", ()):
        if dep == target:
            return True
        if dep in seen or dep not in registry:
            continue
        seen.add(dep)
        if _statically_requires(registry[dep], target, registry, seen):
            return True
    return False


def discover_producers(pass_: GraspPass, asset: str | None = None) -> tuple:
    """The passes a dynamic consumer should read: every registered CANDIDATE PRODUCER, minus cycles.

    Excluded, in order of why:

    * itself;
    * every other ``dynamic_upstream`` pass — they are consumers, their sidecars carry no candidates
      so depending on them would gain nothing, and NOT reading their ``requires`` is what makes this
      compose for any number of them;
    * anything that statically requires ``pass_``, which would be a genuine cycle.

    ``asset=None`` (ordering time — :func:`ordered_passes` has no asset in hand) claims every
    producer so the global run order is right. With an asset, only producers that actually have a
    sidecar for it are claimed: the harness treats a missing upstream sidecar as an error, and
    producers legitimately cover different subsets of the catalog.
    """
    from . import discover_passes

    registry = discover_passes()
    names = []
    for p in registry.values():
        if p.name == pass_.name or getattr(p, "dynamic_upstream", False):
            continue
        if _statically_requires(p, pass_.name, registry):
            continue
        if asset is None or sidecar_path(p.name, asset).exists():
            names.append(p.name)
    return tuple(sorted(names))


class DynamicUpstreamPass(GraspPass):
    """Base for a consumer that DISCOVERS its producers rather than naming them.

    Naming producers in a frozen tuple is wrong for a consumer that should see everything: passes
    land at different times, so a fixed tuple silently ignores whichever generator arrived later —
    the one failure mode that looks like success. Declaring them all eagerly is equally wrong,
    because the harness errors on a required pass that has no sidecar for the asset in hand.

    Subclass, and the harness sees the right ``requires`` per asset::

        class MyAnnotator(DynamicUpstreamPass):
            name = source = "my_annotator"
            def run(self, ctx): ...

    Override ``applies_to`` freely, but call ``super().applies_to(name, kind)`` — it is what records
    which asset ``requires`` should resolve for. The harness guarantees the order it needs
    (``run_pass``: ``applies_to`` -> ``load_asset`` -> ``gather_upstream``).
    """
    dynamic_upstream = True
    _asset: str | None = None

    def applies_to(self, name: str, kind: str) -> bool:
        self._asset = name
        return super().applies_to(name, kind) and bool(self.requires)

    @property
    def requires(self) -> tuple:
        return discover_producers(self, self._asset)


# =================================================================================================
# Sidecar IO
# =================================================================================================
def sidecar_path(pass_name: str, asset: str) -> Path:
    return PASSES_DIR / pass_name / f"{asset}.json"


def _dump(data: dict) -> str:
    """Same compact-numeric-array formatting grasp_library uses, so sidecars diff like records."""
    from ..grasp_library import _dump_json
    return _dump_json(data)


def namespaced_id(source: str, raw_id: str) -> str:
    return raw_id if raw_id.startswith(f"{source}{ID_SEP}") else f"{source}{ID_SEP}{raw_id}"


def source_of_id(candidate_id: str) -> str:
    return candidate_id.split(ID_SEP, 1)[0] if ID_SEP in candidate_id else ""


def upstream_digest(candidates) -> str:
    """Stable digest of the upstream a pass consumed — part of the idempotency key, so a consumer
    re-runs when its inputs change and skips when they have not."""
    h = hashlib.sha1()
    for c in sorted(candidates, key=lambda c: c.id):
        h.update(c.id.encode())
        h.update(np.ascontiguousarray(np.round(np.asarray(c.transform, dtype=float), 9)).tobytes())
        h.update(f"{c.width:.9f}|{c.face}|{c.source}".encode())
    return h.hexdigest()


def _record_for(asset: CatalogAsset, candidates, generator: str) -> GraspRecord:
    return GraspRecord(name=asset.name, kind=asset.kind, frame=asset.frame,
                       candidates=tuple(candidates), file=asset.file, scale=asset.scale,
                       mesh_sha1=asset.mesh_sha1, generator=generator)


def write_sidecar(pass_: GraspPass, asset: CatalogAsset, output: PassOutput,
                  upstream_sha1: str) -> Path:
    """Serialize, VALIDATE and write a pass's sidecar, replacing any previous one."""
    inner = record_to_dict(_record_for(asset, output.candidates,
                                       f"{pass_.name}@{pass_.version}"))
    validate_record(inner, name=asset.name)          # sidecars pass the record schema, unchanged
    data = {
        "_comment": [
            f"Sidecar written by grasp pass {pass_.name!r} for catalog object {asset.name!r}.",
            "GENERATED — re-running the pass replaces this file. Do not hand-edit.",
            "'record' is a complete grasp_library record holding ONLY this pass's candidates;",
            "'annotations' updates candidates owned by other passes (keyed by their id).",
            "assets/objects/grasps/<name>.json is composed from these by grasp_passes.merge.",
        ],
        "sidecar_version": SIDECAR_VERSION,
        "pass": {
            "name": pass_.name,
            "source": pass_.source,
            "version": int(pass_.version),
            "requires": list(pass_.requires),
            "notes": output.notes,
        },
        "upstream_sha1": upstream_sha1,
        "record": inner,
        "annotations": output.annotations,
    }
    out = sidecar_path(pass_.name, asset.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump(data) + "\n")
    return out


def read_sidecar(pass_name: str, asset: str) -> dict | None:
    """Load a sidecar as ``{"pass":…, "record": GraspRecord, "annotations":…, "upstream_sha1":…}``,
    or None if it does not exist. Raises :class:`GraspSchemaError` if it exists but is malformed."""
    p = sidecar_path(pass_name, asset)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise GraspSchemaError(f"sidecar {p} is not valid JSON: {exc}") from exc
    if data.get("sidecar_version") != SIDECAR_VERSION:
        raise GraspSchemaError(
            f"sidecar {p}: sidecar_version {data.get('sidecar_version')!r} != {SIDECAR_VERSION}; "
            f"re-run the pass")
    for key in ("pass", "record"):
        if key not in data:
            raise GraspSchemaError(f"sidecar {p}: missing {key!r}")
    record = record_from_dict(data["record"], name=asset)
    _validate_annotations(data.get("annotations", {}), p)
    return {"pass": data["pass"], "record": record,
            "annotations": data.get("annotations", {}),
            "upstream_sha1": data.get("upstream_sha1", ""), "path": p}


def _validate_annotations(ann: dict, where) -> None:
    if not isinstance(ann, dict):
        raise GraspSchemaError(f"{where}: annotations must be an object")
    for cid, entry in ann.items():
        at = f"{where}: annotation {cid!r}"
        if not isinstance(entry, dict):
            raise GraspSchemaError(f"{at}: must be an object")
        unknown = set(entry) - {"quality", "quality_source", "labels", "notes"}
        if unknown:
            raise GraspSchemaError(f"{at}: unknown field(s) {sorted(unknown)}")
        q = entry.get("quality", {})
        if not isinstance(q, dict):
            raise GraspSchemaError(f"{at}: quality must be an object")
        bad = set(q) - set(QUALITY_FIELDS)
        if bad:
            raise GraspSchemaError(f"{at}: unknown quality field(s) {sorted(bad)}; "
                                   f"known: {list(QUALITY_FIELDS)}")
        for k, v in q.items():
            if v is not None and not (isinstance(v, (int, float)) and np.isfinite(v)):
                raise GraspSchemaError(f"{at}: quality.{k} must be a number or null, got {v!r}")
        if q and not entry.get("quality_source"):
            raise GraspSchemaError(f"{at}: quality values need a quality_source naming what "
                                   f"measured them")


def sidecar_assets(pass_name: str) -> list:
    d = PASSES_DIR / pass_name
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def existing_pass_dirs() -> list:
    return sorted(p.name for p in PASSES_DIR.iterdir() if p.is_dir()) if PASSES_DIR.exists() else []


# =================================================================================================
# Running a pass
# =================================================================================================
def gather_upstream(pass_: GraspPass, asset_name: str) -> tuple:
    """Candidates from the passes this one ``requires``, id-namespaced as the merge will emit them.

    Missing upstream is an error, not an empty list: a consumer silently producing nothing because
    its producer has not run yet is the failure mode most likely to waste a parallel agent's time."""
    out = []
    for dep in pass_.requires:
        side = read_sidecar(dep, asset_name)
        if side is None:
            raise PassError(
                f"pass {pass_.name!r} requires {dep!r}, but no sidecar exists for {asset_name!r} "
                f"({sidecar_path(dep, asset_name)}). Run {dep!r} first.")
        out.extend(side["record"].candidates)
    return tuple(out)


def _normalize(pass_: GraspPass, output, ctx: PassContext) -> PassOutput:
    if output is None:
        output = PassOutput()
    elif isinstance(output, (list, tuple)):
        output = PassOutput(candidates=list(output))
    if not isinstance(output, PassOutput):
        raise PassError(f"pass {pass_.name!r} returned {type(output).__name__}, expected PassOutput "
                        f"or a list of GraspCandidate")

    stamped, seen = [], set()
    for c in output.candidates:
        if not isinstance(c, GraspCandidate):
            raise PassError(f"pass {pass_.name!r} emitted a {type(c).__name__}, expected GraspCandidate")
        if c.source != pass_.source:
            raise PassError(
                f"pass {pass_.name!r} emitted a candidate tagged source={c.source!r}, but the pass "
                f"declares source={pass_.source!r}. The merge deduplicates by source tag, so a pass "
                f"may only emit its own.")
        cid = namespaced_id(pass_.source, c.id)
        if cid in seen:
            raise PassError(f"pass {pass_.name!r} emitted duplicate candidate id {cid!r}")
        seen.add(cid)
        stamped.append(replace(c, id=cid))

    known = {c.id for c in ctx.upstream}
    for cid in output.annotations:
        if cid not in known:
            raise PassError(
                f"pass {pass_.name!r} annotated {cid!r}, which is not in its upstream "
                f"({sorted(known) if known else 'nothing — declare the producer in requires'})")
    return PassOutput(candidates=stamped, annotations=dict(output.annotations), notes=output.notes)


def run_pass(pass_: GraspPass, asset_name: str, *, force: bool = False,
             verbose: bool = True) -> dict:
    """Run one pass over one asset and write its sidecar.

    Skips the work when the existing sidecar was produced by the same pass version from the same
    mesh and the same upstream — that is what makes a re-run cheap as well as harmless. Returns
    ``{"status": "written"|"skipped"|"not-applicable", …}``."""
    from .catalog import catalog_entry
    # Gate BEFORE resolving geometry — load_asset runs the OBB search three times per asset.
    if not pass_.applies_to(asset_name, catalog_entry(asset_name)["kind"]):
        return {"status": "not-applicable", "pass": pass_.name, "asset": asset_name}

    asset = load_asset(asset_name)
    upstream = gather_upstream(pass_, asset_name)
    ctx = PassContext(asset=asset, upstream=upstream)

    up_sha = upstream_digest(upstream)
    if not force:
        prior = read_sidecar(pass_.name, asset_name)
        if (prior is not None
                and prior["pass"].get("version") == pass_.version
                and prior["record"].mesh_sha1 == asset.mesh_sha1
                and prior["upstream_sha1"] == up_sha):
            if verbose:
                print(f"  skip  {pass_.name}/{asset_name} (up to date)")
            return {"status": "skipped", "pass": pass_.name, "asset": asset_name,
                    "candidates": len(prior["record"].candidates), "path": prior["path"]}

    output = _normalize(pass_, pass_.run(ctx), ctx)
    path = write_sidecar(pass_, asset, output, up_sha)
    if verbose:
        print(f"  write {pass_.name}/{asset_name}: {len(output.candidates)} candidate(s)"
              + (f", {len(output.annotations)} annotation(s)" if output.annotations else ""))
    return {"status": "written", "pass": pass_.name, "asset": asset_name,
            "candidates": len(output.candidates), "annotations": len(output.annotations),
            "path": path}


def check_idempotent(pass_: GraspPass, asset_name: str) -> bool | None:
    """Run the pass twice and compare the sidecar bytes. The contract says a pass is deterministic;
    this is how a pass author demonstrates it before handing the pass over.

    Returns None when the pass does not apply to this asset — there is nothing to compare, and a
    vacuous "ok" would be a misleading thing to print for most of the catalog."""
    if run_pass(pass_, asset_name, force=True, verbose=False)["status"] == "not-applicable":
        return None
    first = sidecar_path(pass_.name, asset_name).read_text()
    run_pass(pass_, asset_name, force=True, verbose=False)
    return sidecar_path(pass_.name, asset_name).read_text() == first
