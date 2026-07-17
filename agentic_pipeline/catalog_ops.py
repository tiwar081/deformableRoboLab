"""Catalog ops — RoboLab's ``generate_catalog.py`` / ``get_usd_rigid_body_info`` for our catalog.

Ingest a directory of USD object files into ``scene_catalog.json``: scan for ``.usd``/``.usda``
files (excluding ``_``-prefixed and ``materials`` dirs, RoboLab convention), extract per-object
metadata via USD (AABB dims, mass/density, friction, authored class/description, rigid-body flag),
and append entries. Rebuilds the render catalog entry too. Used by the SKILL.md custom-object flow.

    python -m agentic_pipeline.catalog_ops --ingest <dir> [--kind ycb_mesh] [--class storage]
    python -m agentic_pipeline.catalog_ops --regen        # re-extract dims for existing entries

Runs in the venv (needs ``pxr``). Physical params NOT authored on a USD are inferred the same way
the scene catalog documents them (mass from a density estimate, friction from the material class);
each inferred field is recorded in the entry's ``inferred`` list. Concave meshes get coacd defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJECTS_DIR = REPO_ROOT / "assets" / "objects"
SCENE_CATALOG = OBJECTS_DIR / "scene_catalog.json"


def find_usd_files(root: Path) -> list[Path]:
    """RoboLab ``find_usd_files``: all USD files under ``root``, excluding any under a directory
    whose name starts with ``_`` or is ``materials``/``textures``."""
    out = []
    for ext in ("*.usd", "*.usda", "*.usdc", "*.usdz"):
        for p in root.rglob(ext):
            if any(part.startswith("_") or part in ("materials", "textures") for part in p.parts):
                continue
            out.append(p)
    return sorted(out)


def usd_object_info(usd_path: Path) -> dict:
    """RoboLab ``get_usd_rigid_body_info``: extract catalog metadata from a USD's default prim —
    AABB dims (BBoxCache local bound), authored class/description, rigid-body flag, mass/density,
    and material friction/restitution. Missing fields come back as None so the caller can infer."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetDefaultPrim() or next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)), None)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bbox.ComputeLocalBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    dims = [round(float(hi[i] - lo[i]), 4) for i in range(3)]

    def _authored(name):
        a = prim.GetAttribute(name)
        return a.Get() if a and a.HasAuthoredValue() else None

    mass = density = dyn_fric = stat_fric = rest = None
    mapi = UsdPhysics.MassAPI(prim)
    if mapi:
        mass = mapi.GetMassAttr().Get()
        density = mapi.GetDensityAttr().Get()
    rb = UsdPhysics.RigidBodyAPI(prim)
    rigid = bool(rb.GetRigidBodyEnabledAttr().Get()) if rb and rb.GetRigidBodyEnabledAttr() else True
    return {"name": prim.GetName(), "prim_path": str(prim.GetPath()), "dims": dims,
            "class": _authored("class"), "description": _authored("description"),
            "rigid_body": rigid, "mass": mass, "density": density,
            "dynamic_friction": dyn_fric, "static_friction": stat_fric, "restitution": rest}


def _catalog_entry(usd_path: Path, kind: str, cls: str | None, is_container: bool) -> dict:
    info = usd_object_info(usd_path)
    rel = usd_path.relative_to(OBJECTS_DIR).as_posix()
    inferred = []
    mass = info["mass"]
    if not mass:
        mass = round(max(0.05, 400.0 * info["dims"][0] * info["dims"][1] * info["dims"][2]), 3)
        inferred.append(f"mass {mass} kg estimated from bbox volume x 400 kg/m^3 (not authored)")
    mu = info["dynamic_friction"] or 0.45
    if info["dynamic_friction"] is None:
        inferred.append("mu 0.45 household default (no material friction authored)")
    entry = {
        "name": info["name"], "kind": kind,
        "class": cls or info["class"] or "object",
        "description": info["description"] or f"imported object {info['name']}",
        "dims": info["dims"],
        "config": {"usd_subpath": rel, "target_mass": mass, "mu": mu, "density": 400.0,
                   "color": [0.7, 0.7, 0.7], "coacd_threshold": 0.08, "coacd_max_convex_hull": 16},
        "source": f"ingested from {usd_path}",
        "inferred": inferred,
    }
    if is_container:
        entry["container"] = True
    return entry


def ingest(directory: Path, *, kind: str = "ycb_mesh", cls: str | None = None,
           container: bool = False, verbose: bool = True) -> list[str]:
    """Scan ``directory`` for USD files and append new catalog entries. Returns the names added."""
    cat = json.loads(SCENE_CATALOG.read_text())
    have = {o["name"] for o in cat["objects"]}
    added = []
    for usd in find_usd_files(Path(directory)):
        try:
            entry = _catalog_entry(usd, kind, cls, container)
        except Exception as exc:
            if verbose:
                print(f"[catalog] skip {usd} ({exc})")
            continue
        if entry["name"] in have:
            continue
        cat["objects"].append(entry)
        have.add(entry["name"])
        added.append(entry["name"])
        if verbose:
            print(f"[catalog] + {entry['name']} {entry['dims']} kind={kind}"
                  + (" CONTAINER" if container else ""))
    SCENE_CATALOG.write_text(json.dumps(cat, indent=1))
    return added


def regen_dims(verbose: bool = True) -> int:
    """Re-extract AABB dims for every catalog entry from its USD (RoboLab catalog refresh). Returns
    the count updated. Only touches ``ycb_mesh``/container entries (procedural kinds have no USD)."""
    cat = json.loads(SCENE_CATALOG.read_text())
    n = 0
    for o in cat["objects"]:
        sub = o.get("config", {}).get("usd_subpath")
        if not sub:
            continue
        try:
            dims = usd_object_info(OBJECTS_DIR / sub)["dims"]
        except Exception:
            continue
        if dims != o["dims"]:
            if verbose:
                print(f"[catalog] {o['name']}: dims {o['dims']} -> {dims}")
            o["dims"] = dims
            n += 1
    SCENE_CATALOG.write_text(json.dumps(cat, indent=1))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest", metavar="DIR", help="scan a directory of USD files into the catalog")
    ap.add_argument("--kind", default="ycb_mesh", help="builder kind for ingested objects")
    ap.add_argument("--class", dest="cls", default=None, help="override the object class")
    ap.add_argument("--container", action="store_true", help="mark ingested objects as open-top containers")
    ap.add_argument("--regen", action="store_true", help="re-extract dims for existing catalog entries")
    args = ap.parse_args()
    if args.ingest:
        names = ingest(Path(args.ingest), kind=args.kind, cls=args.cls, container=args.container)
        print(f"added {len(names)} object(s): {names}")
    if args.regen:
        print(f"updated dims on {regen_dims()} entr(ies)")
    if not args.ingest and not args.regen:
        ap.error("give --ingest DIR and/or --regen")
