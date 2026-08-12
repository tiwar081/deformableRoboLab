"""Semantic grasp REGIONS — the schema, the on-disk store, and the reader a scoring pass uses.

A region is a named, semantically meaningful place to grasp an object: a mug's handle, a bowl's
rim, a wine glass's stem, a bottle's neck. It is *not* a grasp — it has no approach direction, no
jaw axis, and no width. It says "this patch of the surface is a handle", in the canonical object
frame, and leaves it to a later pass to decide which generated candidates fall inside it.

**Regions are their own field, not a tag on candidates.** They are produced independently of any
candidate generator (which does not exist yet), and a candidate that happens to sit on a handle is a
JOIN — proximity of a candidate position to a region centre — not a property the annotator can
assert. Keeping them separate means the annotation runs once per asset and stays valid no matter how
often the generators are re-run or replaced.

Store layout (one file per catalog object, written only by this pass):

    assets/objects/grasps/_regions/<catalog_name>.json

It sits beside the pass sidecars rather than in ``_passes/vlm_regions/`` because the harness treats
every ``*.json`` in a pass directory as one asset's sidecar; a second file per asset there would be
read back as a bogus asset named ``<name>.regions``. The ``_`` prefix keeps it out of
``grasp_library.available_grasps()``, which globs ``*.json`` non-recursively.

Reading them:

    from deformableManipulationTools.grasp_passes.vlm_regions import load_regions, regions_near
    doc = load_regions("mug")                 # None if the asset was never annotated
    hits = regions_near(doc, candidate_position)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...grasp_library import FRAME_CONVENTION, GRASPS_DIR, ObjectFrame

REGIONS_DIR = GRASPS_DIR / "_regions"
# Bumped when the FILE FORMAT changes. Part of the cache key, so a format change re-annotates.
STORE_VERSION = 1

# The label vocabulary. Closed on purpose: a downstream scorer selects on these strings ("prefer a
# handle"), so free-text labels would make the join unreliable. ``other`` exists so a genuinely
# meaningful feature outside the vocabulary is still recorded (with ``detail`` naming it) instead of
# being forced into the wrong bucket.
REGION_LABELS = (
    "handle",     # a grip appendage spanning a gap (mug, jug, basket, tool)
    "rim",        # the raised boundary of an opening (bowl, cup, bucket)
    "neck",       # a narrowed section below an opening or head (bottle, vase)
    "stem",       # a slender column joining two parts (glass, fruit, funnel)
    "spout",      # a pour lip or nozzle
    "knob",       # a small protruding boss meant to be pinched (lid, drawer)
    "lip",        # a thin flanged edge that a jaw can hook or pinch
    "tab",        # a flat pull tab or strap end
    "shaft",      # the narrow elongated body of a tool (screwdriver, brush)
    "other",
)

# A region's radius is a soft neighbourhood, not a measured boundary — the VLM points at a feature,
# it does not segment it. Clamped so a proximity join is neither a point test nor the whole object.
MIN_REGION_RADIUS = 0.010
MAX_REGION_RADIUS = 0.060


class RegionSchemaError(ValueError):
    """A regions document is malformed or was written by an incompatible version."""


@dataclass(frozen=True)
class Pick:
    """One (view, image position) the VLM pointed at, and where that ray hit the surface."""
    view: str
    u: float
    v: float
    point: tuple = ()          # canonical-frame surface point; () if it missed (pick dropped)


@dataclass(frozen=True)
class GraspRegion:
    """One named region on an asset, in CANONICAL frame coordinates (``obb_extent_desc_v1``).

    ``center`` is the mean of the back-projected picks and ``radius`` their spread (clamped) — a
    ball, which is all the precision an image-space pick supports. ``normal`` is the mean outward
    surface normal at the picks, useful to a scorer that wants an approach roughly facing the
    feature; it is a hint, not a grasp axis."""
    id: str
    label: str
    center: tuple
    radius: float
    normal: tuple = ()
    confidence: float = 0.0
    detail: str = ""
    rationale: str = ""
    picks: tuple = ()
    views: tuple = ()          # names of the views the surviving picks came from

    def contains(self, point, margin: float = 0.0) -> bool:
        d = float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(self.center, dtype=float)))
        return d <= self.radius + margin


@dataclass
class RegionDoc:
    """Everything stored for one asset: the regions plus the provenance that makes them cacheable."""
    name: str
    kind: str
    frame: ObjectFrame
    regions: tuple = ()
    mesh_sha1: str = ""
    file: str | None = None
    scale: float = 1.0
    model: str = ""
    prompt_version: int = 0
    pass_version: int = 0
    views: tuple = ()
    dropped: tuple = ()        # picks/regions the back-projection rejected, kept for audit
    notes: str = ""
    empty_reason: str = ""     # why an asset has no regions (the model's own words)

    def by_label(self, label: str) -> tuple:
        return tuple(r for r in self.regions if r.label == label)


def regions_path(name: str) -> Path:
    return REGIONS_DIR / f"{name}.json"


def cache_key(mesh_sha1: str, prompt_version: int, pass_version: int) -> str:
    """What makes a stored annotation still valid. The MODEL is deliberately absent: pointing a new
    model at an unchanged mesh is not a reason to spend six renders and an API call, and a caller
    who does want that re-run has ``--refresh``."""
    return f"{STORE_VERSION}|{mesh_sha1}|{prompt_version}|{pass_version}"


# =================================================================================================
# Serialization
# =================================================================================================
def doc_to_dict(doc: RegionDoc) -> dict:
    return {
        "_comment": [
            f"Semantic grasp regions for catalog object {doc.name!r}, identified by a VLM from "
            f"rendered views of the asset's rest geometry.",
            "GENERATED by grasp_passes/vlm_regions — re-running the pass replaces this file.",
            "Positions are in the CANONICAL object frame (same convention as the grasp record); a "
            "region is a named place, NOT a grasp: no approach, no jaw axis, no width.",
            "Regions are joined to grasp candidates downstream by PROXIMITY; nothing here refers "
            "to a candidate.",
        ],
        "store_version": STORE_VERSION,
        "cache_key": cache_key(doc.mesh_sha1, doc.prompt_version, doc.pass_version),
        "object": {"name": doc.name, "kind": doc.kind, "file": doc.file, "scale": doc.scale,
                   "mesh_sha1": doc.mesh_sha1},
        # The full frame, not a summary: a consumer must be able to check that the points it is
        # joining came from the SAME canonical frame its candidates are in (the merge enforces the
        # same agreement between sidecars), and to map a region back into the asset/body frame.
        "frame": {"convention": FRAME_CONVENTION,
                  "rotation": np.asarray(doc.frame.rotation, dtype=float).tolist(),
                  "translation": np.asarray(doc.frame.translation, dtype=float).tolist(),
                  "extents": np.asarray(doc.frame.extents, dtype=float).tolist(),
                  "axis_order": list(doc.frame.axis_order),
                  "sign_rules": list(doc.frame.sign_rules),
                  "ambiguous": bool(doc.frame.ambiguous),
                  "drift": doc.frame.drift},
        "annotator": {"model": doc.model, "prompt_version": doc.prompt_version,
                      "pass_version": doc.pass_version},
        "views": list(doc.views),
        "regions": [asdict(r) for r in doc.regions],
        "dropped": list(doc.dropped),
        "empty_reason": doc.empty_reason,
        "notes": doc.notes,
    }


def doc_from_dict(data: dict, *, name: str | None = None) -> RegionDoc:
    where = f"regions document {name or data.get('object', {}).get('name', '?')!r}"
    if not isinstance(data, dict):
        raise RegionSchemaError(f"{where}: not a JSON object")
    if data.get("store_version") != STORE_VERSION:
        raise RegionSchemaError(f"{where}: store_version {data.get('store_version')!r} != "
                                f"{STORE_VERSION}; re-run the vlm_regions pass")
    for block in ("object", "frame", "regions"):
        if block not in data:
            raise RegionSchemaError(f"{where}: missing {block!r}")
    obj, fr = data["object"], data["frame"]
    if name is not None and obj.get("name") != name:
        raise RegionSchemaError(f"{where}: object.name {obj.get('name')!r} != filename ({name!r})")
    if fr.get("convention") != FRAME_CONVENTION:
        raise RegionSchemaError(f"{where}: frame.convention {fr.get('convention')!r} != "
                                f"{FRAME_CONVENTION!r} — the positions would be read in the wrong "
                                f"axes")
    frame = ObjectFrame(rotation=np.asarray(fr["rotation"], dtype=float),
                        translation=np.asarray(fr["translation"], dtype=float),
                        extents=np.asarray(fr["extents"], dtype=float),
                        axis_order=tuple(fr.get("axis_order", (0, 1, 2))),
                        sign_rules=tuple(fr.get("sign_rules", ("", "", ""))),
                        ambiguous=bool(fr.get("ambiguous", False)),
                        drift=fr.get("drift"))
    regions = []
    for i, r in enumerate(data["regions"]):
        at = f"{where}: regions[{i}]"
        for key in ("id", "label", "center", "radius"):
            if key not in r:
                raise RegionSchemaError(f"{at}: missing {key!r}")
        if r["label"] not in REGION_LABELS:
            raise RegionSchemaError(f"{at}: label {r['label']!r} not in {REGION_LABELS}")
        c = np.asarray(r["center"], dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            raise RegionSchemaError(f"{at}: center must be 3 finite numbers, got {r['center']!r}")
        if not (MIN_REGION_RADIUS - 1e-9 <= float(r["radius"]) <= MAX_REGION_RADIUS + 1e-9):
            raise RegionSchemaError(f"{at}: radius {r['radius']} outside "
                                    f"[{MIN_REGION_RADIUS}, {MAX_REGION_RADIUS}]")
        regions.append(GraspRegion(
            id=r["id"], label=r["label"], center=tuple(c.tolist()), radius=float(r["radius"]),
            normal=tuple(r.get("normal") or ()), confidence=float(r.get("confidence", 0.0)),
            detail=r.get("detail", ""), rationale=r.get("rationale", ""),
            picks=tuple(Pick(**p) if isinstance(p, dict) else p for p in r.get("picks", ())),
            views=tuple(r.get("views", ()))))
    ann = data.get("annotator", {})
    return RegionDoc(name=obj["name"], kind=obj.get("kind", ""), frame=frame,
                     regions=tuple(regions), mesh_sha1=obj.get("mesh_sha1", ""),
                     file=obj.get("file"), scale=float(obj.get("scale", 1.0)),
                     model=ann.get("model", ""), prompt_version=int(ann.get("prompt_version", 0)),
                     pass_version=int(ann.get("pass_version", 0)),
                     views=tuple(data.get("views", ())), dropped=tuple(data.get("dropped", ())),
                     notes=data.get("notes", ""), empty_reason=data.get("empty_reason", ""))


def write_doc(doc: RegionDoc) -> Path:
    from ...grasp_library import _dump_json
    out = regions_path(doc.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = doc_to_dict(doc)
    doc_from_dict(data, name=doc.name)          # never write something that cannot be read back
    out.write_text(_dump_json(data) + "\n")
    return out


def load_regions(name: str) -> RegionDoc | None:
    """The stored regions for a catalog object, or None if it was never annotated.

    None and "annotated, no regions" are different answers and stay different: an asset with an
    empty ``regions`` tuple was looked at and found to have no semantically meaningful place to
    grasp, which is a real result a scorer can rely on."""
    p = regions_path(name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise RegionSchemaError(f"{p} is not valid JSON: {exc}") from exc
    return doc_from_dict(data, name=name)


def annotated_assets() -> list:
    return sorted(p.stem for p in REGIONS_DIR.glob("*.json")) if REGIONS_DIR.exists() else []


# =================================================================================================
# The join a scoring pass performs
# =================================================================================================
def regions_near(doc: RegionDoc | None, position, margin: float = 0.0) -> list:
    """Regions whose ball contains ``position`` (a canonical-frame point, e.g. a candidate's grasp
    centre), nearest first. The whole intended consumer API: a scorer asks "what is this grasp on?"

    ``margin`` widens every region by a fixed amount — a jaw closing on a handle does not have its
    TCP exactly on the surface point the VLM picked, so a scorer typically passes something like
    half a jaw width."""
    if doc is None:
        return []
    p = np.asarray(position, dtype=float)
    hits = [(float(np.linalg.norm(p - np.asarray(r.center, dtype=float))), r) for r in doc.regions]
    return [r for d, r in sorted(hits, key=lambda t: t[0]) if d <= r.radius + margin]
