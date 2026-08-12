"""Grasp passes — per-asset grasp computations that compose into the shared grasp records.

Several passes are built in parallel by separate agents, so the package is arranged so that adding
a pass touches EXACTLY ONE new directory and no shared file:

    grasp_passes/
      __init__.py   base.py   catalog.py   merge.py   __main__.py    <- SHARED, read-only
      fixture/      <- one pass, one directory
      <yours>/      <- add this; nothing else

Passes are discovered by scanning for subdirectories that export a ``PASS`` object, so there is no
central registry to edit (and therefore nothing for two agents to conflict on).

    from deformableManipulationTools.grasp_passes import GraspPass, PassContext, PassOutput

Read ``grasp_passes/README.md`` for the authoring contract, and
``deformableManipulationTools/grasp_library.py`` for the record schema and the two frame
conventions. ``grasp_library`` is FROZEN — passes import from it and never modify it.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import (DynamicUpstreamPass, GraspPass, PassContext, PassError, PassOutput,
                   PASSES_DIR, check_idempotent, discover_producers, gather_upstream,
                   read_sidecar, run_pass, sidecar_path, write_sidecar)
from .catalog import AssetError, CatalogAsset, SUPPORTED_KINDS, asset_names, load_asset
from .merge import (MergeError, OUT_OF_REACH_NOTE, is_out_of_reach, merge_all,
                    merge_asset, merged_assets, sidecars_for)

__all__ = [
    "GraspPass", "DynamicUpstreamPass", "PassContext", "PassOutput", "PassError",
    "PASSES_DIR", "run_pass", "discover_producers",
    "check_idempotent", "gather_upstream", "read_sidecar", "sidecar_path", "write_sidecar",
    "CatalogAsset", "AssetError", "SUPPORTED_KINDS", "asset_names", "load_asset",
    "merge_asset", "merge_all", "merged_assets", "sidecars_for", "MergeError",
    "is_out_of_reach", "OUT_OF_REACH_NOTE",
    "discover_passes", "get_pass",
]


def discover_passes(verbose: bool = False) -> dict:
    """``{name: GraspPass}`` for every subpackage exporting a ``PASS``.

    Import failures are reported and skipped rather than raised: one agent's half-finished pass must
    not stop everyone else's from running."""
    found: dict = {}
    for mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if not mod.ispkg or mod.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as exc:                      # noqa: BLE001 - isolation is the point
            print(f"[grasp_passes] WARNING: {mod.name} failed to import: {exc}")
            continue
        obj = getattr(module, "PASS", None)
        if obj is None:
            if verbose:
                print(f"[grasp_passes] {mod.name}: no PASS export, skipped")
            continue
        if not isinstance(obj, GraspPass):
            print(f"[grasp_passes] WARNING: {mod.name}.PASS is {type(obj).__name__}, not a GraspPass")
            continue
        if not obj.name or not obj.source:
            print(f"[grasp_passes] WARNING: {mod.name}.PASS must set both 'name' and 'source'")
            continue
        if obj.name != mod.name:
            print(f"[grasp_passes] WARNING: {mod.name}.PASS.name is {obj.name!r}; the pass name must "
                  f"match its directory (it names the sidecar directory)")
            continue
        found[obj.name] = obj

    by_source: dict = {}
    for p in found.values():
        if p.source in by_source:
            raise PassError(
                f"passes {by_source[p.source]!r} and {p.name!r} both declare source tag "
                f"{p.source!r}. The merge deduplicates by source, so each pass needs its own.")
        by_source[p.source] = p.name
    return dict(sorted(found.items()))


def get_pass(name: str) -> GraspPass:
    passes = discover_passes()
    if name not in passes:
        raise PassError(f"no pass named {name!r}. Available: {', '.join(passes) or 'none'}")
    return passes[name]


def ordered_passes(passes=None) -> list:
    """Passes in dependency order (a pass runs after everything in its ``requires``)."""
    passes = passes if passes is not None else list(discover_passes().values())
    by_name = {p.name: p for p in passes}
    out, state = [], {}

    def visit(p, chain):
        if state.get(p.name) == "done":
            return
        if state.get(p.name) == "visiting":
            raise PassError(f"circular pass dependency: {' -> '.join(chain + [p.name])}")
        state[p.name] = "visiting"
        for dep in p.requires:
            if dep not in by_name:
                raise PassError(f"pass {p.name!r} requires {dep!r}, which is not registered")
            visit(by_name[dep], chain + [p.name])
        state[p.name] = "done"
        out.append(p)

    for p in sorted(passes, key=lambda p: p.name):
        visit(p, [])
    return out
