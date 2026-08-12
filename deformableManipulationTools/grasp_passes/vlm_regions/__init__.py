"""``vlm_regions`` pass — semantically meaningful grasp REGIONS, identified by a VLM.

    render 6 fixed views of the asset's rest geometry
      -> ask a VLM to point at handles / rims / necks / stems / spouts / knobs ...
      -> back-project each picked pixel onto the mesh
      -> store named regions as canonical-frame balls, once per asset.

What it is for: a grasp candidate that happens to close on a mug's handle is a better grasp than one
on its wall, and no geometric generator knows that — "handle" is semantics, not curvature. This pass
supplies the semantics; a later scoring pass joins them to candidates BY PROXIMITY.

Three properties of the design worth knowing before changing it:

* **Regions are their own artifact, not a tag on candidates.** They are computed with no candidate
  generator in the loop (none exists yet) and stay valid across generator re-runs. See
  :mod:`.regions` for the store and :func:`regions_near` for the join a consumer performs.
* **An object with no meaningful region records an EMPTY list, not a guess.** Most catalog objects
  are like that — a can, a box, a ball are grasped anywhere — and the prompt pushes hard on this,
  because a fabricated "region" would be read downstream as evidence. ``empty_reason`` keeps the
  finding auditable.
* **Rigid assets only.** Cloth, bags, and (since 2026-08-11) cables are excluded by the SCOPE
  stated in ``grasp_library.py`` (no persistent rest shape -> a frame fitted to the source mesh
  describes the asset file, not the settled object; cables additionally failed the rest-shape
  premise on the measured record — when they were still in scope, this pass excluded them anyway,
  their synthetic straight chain having no semantic features to see). This pass narrows further,
  to the RIGID kinds: a soft body's rendered rest shape is not the shape the gripper meets.

Caching: the annotation runs ONCE per asset. The stored document carries a cache key over
(store format, mesh sha1, prompt version, pass version); a run whose key matches reuses the file
without rendering or calling the model, which is also what makes this pass deterministic under
``--check-idempotent`` despite a non-deterministic annotator. ``--refresh`` forces a real re-run.

    python -m deformableManipulationTools.grasp_passes run vlm_regions --asset mug
    python -m deformableManipulationTools.grasp_passes.vlm_regions annotate --asset mug --refresh
    python -m deformableManipulationTools.grasp_passes.vlm_regions show mug
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..base import GraspPass, PassContext, PassOutput
from .regions import (GraspRegion, MAX_REGION_RADIUS, MIN_REGION_RADIUS, Pick, REGION_LABELS,
                      RegionDoc, RegionSchemaError, annotated_assets, cache_key, load_regions,
                      regions_near, regions_path, write_doc)

__all__ = ["PASS", "VlmRegionPass", "GraspRegion", "RegionDoc", "load_regions", "regions_near",
           "annotated_assets", "regions_path", "annotate_asset", "REGION_LABELS", "RIGID_KINDS"]

# Rigid catalog kinds only — see the module docstring.
RIGID_KINDS = ("ycb_mesh", "rubiks_cube", "rigid_box")

# A region whose confidence is below this is dropped. The prompt asks for a self-assessment and a
# low one usually accompanies a feature the model could not actually see in the renders.
MIN_CONFIDENCE = 0.35

# Picks farther apart than this are treated as DIFFERENT places on the same named feature and become
# separate regions (``rim_0``, ``rim_1``, ...) instead of being averaged into one.
#
# Averaging is wrong for any feature that wraps around the object, and a rim is the common case:
# picks on opposite sides of a mug's mouth average to the CENTRE OF THE HOLE — measured, the first
# working annotation put the mug's rim 8 mm from its axis, in mid-air, instead of on the wall. A
# region is a ball, so an annular feature is honestly represented as several balls on the ring.
# 3.5 cm is a little under half a jaw opening: two grasp places that far apart are not the same
# grasp. The cost of the rule is that a long feature (a tool shaft) picked at both ends is split in
# two, which is a fair description of it rather than an error.
PICK_LINK_DISTANCE = 0.035


class VlmRegionPass(GraspPass):
    name = "vlm_regions"
    source = "vlm_regions"
    version = 2                 # v2: pick clustering (see PICK_LINK_DISTANCE)
    kinds = RIGID_KINDS
    requires = ()

    def run(self, ctx: PassContext) -> PassOutput:
        doc = annotate_asset(ctx, refresh=False)
        n = len(doc.regions)
        summary = ", ".join(sorted({r.label for r in doc.regions})) if n else (
            doc.empty_reason or "no semantically distinguished region")
        # No candidates: a region is a place, not a grasp. The pass's OUTPUT is the regions
        # document; the sidecar exists so the pass is a first-class member of the harness (run
        # order, per-asset error reporting, cache invalidation by mesh sha1).
        return PassOutput(notes=f"{n} region(s) [{summary}] -> {regions_path(ctx.name).name}")


PASS = VlmRegionPass()


# =================================================================================================
# The annotation itself
# =================================================================================================
def annotate_asset(ctx: PassContext, *, refresh: bool = False, model: str | None = None,
                   device: str = "cuda:0", keep_renders: Path | None = None,
                   verbose: bool = True) -> RegionDoc:
    """Annotate one asset, reusing the stored document unless ``refresh``.

    Returns the :class:`RegionDoc` that is now on disk."""
    key = cache_key(ctx.asset.mesh_sha1, _prompt_version(), PASS.version)
    if not refresh:
        try:
            cached = load_regions(ctx.name)
        except RegionSchemaError as exc:
            if verbose:
                print(f"  stored regions for {ctx.name!r} unusable ({exc}); re-annotating")
            cached = None
        if cached is not None and cache_key(cached.mesh_sha1, cached.prompt_version,
                                            cached.pass_version) == key:
            if verbose:
                print(f"  cached {ctx.name}: {len(cached.regions)} region(s)")
            return cached

    if ctx.asset.faces is None:
        raise ValueError(f"{ctx.name!r} has no surface triangles; vlm_regions renders and "
                         f"ray-casts a surface mesh, so it cannot annotate this kind")

    verts = ctx.asset.canonical_vertices()
    faces = ctx.asset.faces
    work = Path(keep_renders) if keep_renders else Path(tempfile.mkdtemp(prefix="vlm_regions_"))
    try:
        doc = _annotate(ctx, verts, faces, work, model=model, device=device, verbose=verbose)
    finally:
        if keep_renders is None:
            shutil.rmtree(work, ignore_errors=True)
    write_doc(doc)
    return doc


def _cluster(points: np.ndarray, link: float) -> list:
    """Single-linkage clusters of ``points``, as lists of indices in input order.

    Order-stable (so the pass stays deterministic) and brute force — a region has at most one pick
    per view."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(points[i] - points[j])) <= link:
                parent[find(j)] = find(i)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def _prompt_version() -> int:
    from .prompt import PROMPT_VERSION
    return PROMPT_VERSION


def _annotate(ctx: PassContext, verts: np.ndarray, faces: np.ndarray, work: Path, *,
              model: str | None, device: str, verbose: bool) -> RegionDoc:
    from .prompt import PROMPT_VERSION, ask_regions
    from .views import backproject, plan_views, render_views

    views = render_views(verts, faces, plan_views(verts), work, device=device, verbose=verbose)
    answer, used_model = ask_regions(ctx.name, ctx.kind, ctx.frame.extents, ctx.asset.entry,
                                     views, model=model, verbose=verbose)
    by_name = {v.name: v for v in views}

    regions, dropped = [], []
    counts: dict[str, int] = {}
    for i, raw in enumerate(answer.get("regions") or []):
        label = raw.get("label", "other")
        conf = float(raw.get("confidence", 0.0))
        if conf < MIN_CONFIDENCE:
            dropped.append({"region": i, "label": label, "reason": f"confidence {conf:.2f} < "
                                                                   f"{MIN_CONFIDENCE}"})
            continue
        picks, points, normals = [], [], []
        for p in raw.get("picks") or []:
            view = by_name.get(p.get("view", ""))
            if view is None:
                dropped.append({"region": i, "label": label,
                                "reason": f"unknown view {p.get('view')!r}"})
                continue
            hit = backproject(view, float(p["u"]), float(p["v"]), verts, faces)
            if hit is None:
                # The pick landed on background (or off-image). Dropping it is the point: a pick
                # that misses the object carries no information about where the feature is, and
                # snapping it to the nearest surface would invent a position the model never chose.
                dropped.append({"region": i, "label": label, "view": view.name,
                                "u": float(p["u"]), "v": float(p["v"]),
                                "reason": "ray missed the mesh"})
                picks.append(Pick(view=view.name, u=float(p["u"]), v=float(p["v"])))
                continue
            point, normal = hit
            points.append(point)
            normals.append(normal)
            picks.append(Pick(view=view.name, u=float(p["u"]), v=float(p["v"]),
                              point=tuple(point.tolist())))
        if not points:
            dropped.append({"region": i, "label": label, "reason": "no pick hit the mesh"})
            continue

        hit_picks = [p for p in picks if p.point]
        for members in _cluster(np.asarray(points, dtype=float), PICK_LINK_DISTANCE):
            pts = np.asarray([points[m] for m in members], dtype=float)
            centre = pts.mean(axis=0)
            spread = float(np.linalg.norm(pts - centre, axis=1).max()) if len(pts) > 1 else 0.0
            radius = float(np.clip(max(spread, MIN_REGION_RADIUS), MIN_REGION_RADIUS,
                                   MAX_REGION_RADIUS))
            n = np.asarray([normals[m] for m in members], dtype=float).mean(axis=0)
            n_norm = float(np.linalg.norm(n))
            counts[label] = counts.get(label, 0) + 1
            cluster_picks = tuple(hit_picks[m] for m in members)
            regions.append(GraspRegion(
                id=f"{label}_{counts[label] - 1}", label=label,
                center=tuple(centre.tolist()), radius=radius,
                # Picks from opposite sides of a thin feature cancel; an ill-conditioned mean is
                # reported as "no normal" rather than as a direction that is really numerical noise.
                normal=tuple((n / n_norm).tolist()) if n_norm > 0.2 else (),
                confidence=conf, detail=str(raw.get("detail", "")),
                rationale=str(raw.get("rationale", "")), picks=cluster_picks,
                views=tuple(dict.fromkeys(p.view for p in cluster_picks))))

    if verbose and dropped:
        print(f"  dropped {len(dropped)} pick/region(s): "
              + "; ".join(sorted({d['reason'] for d in dropped})))
    return RegionDoc(
        name=ctx.name, kind=ctx.kind, frame=ctx.frame, regions=tuple(regions),
        mesh_sha1=ctx.asset.mesh_sha1, file=ctx.asset.file, scale=ctx.asset.scale,
        model=used_model, prompt_version=PROMPT_VERSION, pass_version=PASS.version,
        views=tuple(v.to_dict() for v in views), dropped=tuple(dropped),
        empty_reason=str(answer.get("empty_reason", "")) if not regions else "",
        notes="Regions are canonical-frame balls; join to candidates by proximity.")
