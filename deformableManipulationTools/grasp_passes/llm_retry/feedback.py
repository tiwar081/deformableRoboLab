"""What the LLM is told about the grasps that were already tried — text report + round-B images.

Two consumers:

* **Round A** sends :func:`tried_report` — every merged-record candidate serialized compactly with
  its measured outcome ("the corresponding json file in distilled form", docs/trajPipeline/
  llm-retry.md), weak retreated poses in their own section flagged as reusable raw material.
* **Round B** additionally sends one labelled RENDER per round-A candidate (including the ones
  dropped at authoring) — drawn with ``grasp_passes.geometric.viz``'s marker tooling, one image per
  candidate so the model can see exactly where each failed pose sat — plus a per-candidate failure
  report (:func:`round_b_feedback`).

Rendering reuses the shared viz/view machinery rather than re-implementing it: the six canonical
views come from ``vlm_regions.views`` (same renderer, same lit-rotation scheme, same view names —
that module's docstring explains why the object turns and the camera does not), and the grasp
markers from ``geometric.viz.write_png``. The one geometry wrinkle this module owns is
:func:`render_geometry`: a ``soft_mesh`` asset has no surface triangles in its catalog resolution
(``ctx.asset.faces is None``), so its BOUNDARY faces are derived from the tet file for RENDERING
only — measurement (``measure_span_at``) still runs on ``ctx.asset.faces`` exactly as every other
consumer measures.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...grasp_library import (GraspCandidate, SEAT_BLOCKED_LABEL, SHAKE_SKIPPED_LABEL,
                              grasp_transform, is_weak)
from ...settings import REPO_ROOT

# All llm_retry artifacts a human might want to look at land here (task-specified location):
# feedback marker images at the top level, the six canonical views under views/.
OUTPUT_DIR = REPO_ROOT / "outputs" / "grasp_viz" / "llm_retry"


# =================================================================================================
# Geometry for rendering
# =================================================================================================
def render_geometry(asset) -> tuple[np.ndarray, np.ndarray]:
    """``(canonical_vertices, faces)`` suitable for the raycast renderer.

    Rigid kinds carry surface triangles already. A ``soft_mesh`` resolves to tet-mesh POINTS
    (``faces is None``) — un-renderable — so its boundary triangles are extracted from the tet
    file: every tet face that belongs to exactly one tet is on the surface. The vertex array and
    ordering are identical to the catalog's resolution (same file, same scale), so the canonical
    frame maps the derived faces exactly. Raises for a kind with neither faces nor tets — a loud
    per-asset failure beats a blind prompt (README rule 8)."""
    verts = asset.canonical_vertices()
    if asset.faces is not None:
        return verts, np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    if asset.kind == "soft_mesh":
        from ...assets import load_tet_mesh
        from ..catalog import OBJECTS_DIR
        raw, tets = load_tet_mesh(OBJECTS_DIR / asset.entry["config"]["tet_subpath"])
        if len(raw) != len(verts):
            raise ValueError(f"{asset.name!r}: tet file vertex count {len(raw)} != catalog "
                             f"resolution {len(verts)} — cannot map boundary faces")
        return verts, _tet_boundary_faces(np.asarray(tets, dtype=np.int64))
    raise ValueError(f"{asset.name!r} (kind {asset.kind!r}) has no surface triangles and no tet "
                     f"source; llm_retry cannot render it")


def _tet_boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Surface triangles of a tet mesh: the faces that appear in exactly one tetrahedron."""
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    _, index, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return np.ascontiguousarray(faces[index[counts == 1]])


def render_views_for(asset, *, device: str = "cuda:0", verbose: bool = True) -> list:
    """The six canonical annotation views of this asset, kept under ``outputs/grasp_viz/llm_retry``
    for audit. Same renderer, plan, and view names as ``vlm_regions`` — imported, not copied."""
    from ..vlm_regions.views import plan_views, render_views

    verts, faces = render_geometry(asset)
    out_dir = OUTPUT_DIR / asset.name / "views"
    return render_views(verts, faces, plan_views(verts), out_dir, device=device, verbose=verbose)


# =================================================================================================
# Outcome accounting — one candidate, one honest line
# =================================================================================================
def _fmt_vec(v, digits: int = 3) -> str:
    return "(" + ",".join(f"{float(x):.{digits}f}" for x in np.asarray(v, dtype=float)) + ")"


def _mm(x) -> str:
    return "?" if x is None else f"{float(x) * 1000:.0f}mm"


def outcome_of(candidate: GraspCandidate) -> str:
    """The candidate's measured outcome as one short phrase. Honest about the unknowns: an
    untested candidate reads "not yet tested", never "failed"."""
    q = candidate.quality or {}
    held = q.get("object_in_gripper")
    if "diverged" in candidate.labels:
        return "trial DIVERGED (solver ejection — no verdict)"
    if held is not None:
        if float(held) == 1.0:
            return (f"HELD (close {_mm(q.get('object_motion_during_closing_linear'))}, "
                    f"shake {_mm(q.get('object_motion_during_shaking_linear'))})")
        return (f"DROPPED (close {_mm(q.get('object_motion_during_closing_linear'))}"
                f"/{_deg(q.get('object_motion_during_closing_angular'))}, "
                f"shake {_mm(q.get('object_motion_during_shaking_linear'))}"
                f"/{_deg(q.get('object_motion_during_shaking_angular'))})")
    if "pregrasp_collision" in candidate.labels:
        return "SKIPPED: pre-grasp collision (hand inside the object at this pose)"
    if SHAKE_SKIPPED_LABEL in candidate.labels:
        return "SKIPPED by physics validation (see notes)"
    if is_weak(candidate):
        return "never physics-tested (weak retreated pose)"
    return "not yet tested"


def _deg(x) -> str:
    return "?" if x is None else f"{np.degrees(float(x)):.0f}deg"


def candidate_line(candidate: GraspCandidate) -> str:
    """One compact line: id, source, seat_mode, pose, width, span, seat_depth, outcome."""
    return (f"{candidate.id} [{candidate.source}, seat={candidate.seat_mode}] "
            f"pos={_fmt_vec(candidate.position)} appr={_fmt_vec(candidate.approach, 2)} "
            f"jaw={_fmt_vec(candidate.jaw_axis, 2)} width={_mm(candidate.width)} "
            f"span={_mm(candidate.span)} seat_depth={_mm(candidate.seat_depth)}"
            f" -> {outcome_of(candidate)}")


def tried_report(record, *, exclude_sources: tuple = ()) -> str:
    """Everything the merged record says was tried, serialized for the prompt.

    Legitimate candidates first (each with its measured outcome), then the WEAK retreated poses in
    their own explicitly-flagged section — collision-free by construction, never physics-tested,
    offered to the LLM as reusable/modifiable raw material (docs/trajPipeline/llm-retry.md).
    ``seat_blocked`` candidates are excluded ("these never reach this stage"): they only appear in
    a record that predates the merge derivation layer."""
    legit, weak = [], []
    for c in record.candidates:
        if c.source in exclude_sources or SEAT_BLOCKED_LABEL in c.labels:
            continue
        (weak if is_weak(c) else legit).append(c)

    lines = []
    if legit:
        lines.append(f"TRIED CANDIDATES ({len(legit)}) — every legitimate grasp already generated "
                     f"and its physics-validation outcome:")
        lines += [f"  - {candidate_line(c)}" for c in sorted(legit, key=lambda c: c.id)]
    else:
        lines.append("TRIED CANDIDATES: none — no generator found a legitimate grasp within the "
                     "jaw for this object (it is 'out of reach' so far).")
    if weak:
        lines.append("")
        lines.append(f"WEAK RETREATED POSES ({len(weak)}) — REUSABLE RAW MATERIAL. These poses "
                     f"had to back away along their approach to clear the hand; they are "
                     f"collision-free by construction but hold almost nothing (measured: ~3% at "
                     f"n=628), so they were never physics-tested and do not count as grasps. You "
                     f"MAY adopt or modify one (different width, jaw axis, or depth) if you see "
                     f"how to turn its reachability into a real hold:")
        lines += [f"  - {candidate_line(c)}" for c in sorted(weak, key=lambda c: c.id)]
    return "\n".join(lines)


# =================================================================================================
# Round-B feedback: one labelled marker render per round-A candidate + a failure report
# =================================================================================================
def _stub_candidate(rid: str, raw: dict, frame) -> GraspCandidate | None:
    """A drawable stand-in for an authoring-dropped pose, or None when even its transform is
    degenerate (jaw parallel to approach — there is nothing geometric to draw)."""
    try:
        t = grasp_transform(np.asarray(raw["position"], dtype=float),
                            np.asarray(raw["approach"], dtype=float),
                            np.asarray(raw["jaw_axis"], dtype=float))
        width = float(raw["width"])
        if not (0.0 < width <= 0.2):
            width = 0.04                       # draw SOMETHING sensible for an absurd width
        return GraspCandidate(id=rid, transform=t, width=width,
                              face=frame.face_of(t[:3, 2]), source="llm_retry")
    except Exception:                          # noqa: BLE001 - degenerate pose, image impossible
        return None


def round_b_feedback(asset, record, cands_a: list, drops_a: list,
                     *, verbose: bool = True) -> tuple[list, str, str]:
    """Build the round-B feedback: ``(image_paths, report_text, feedback_digest)``.

    One image per round-A candidate id (a00..a09), authoring-dropped ones included, each rendered
    via ``geometric.viz.write_png`` with JUST that candidate and a title of id + outcome; images
    land in ``outputs/grasp_viz/llm_retry/<asset>/``. The report pairs each id with what happened:
    the authoring drop reason, the pre-check skip, or the shake outcome with motions and notes."""
    import trimesh
    from ..base import namespaced_id
    from ..geometric.viz import write_png

    verts, faces = render_geometry(asset)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    merged = {c.id: c for c in record.candidates}
    drops = {d["id"]: d for d in drops_a}
    out_dir = OUTPUT_DIR / asset.name

    items = []                                  # (rid, outcome_short, detail, candidate_or_None)
    for c in cands_a:
        rid = c.id                              # raw id (a00…) — run() hands us pre-namespace ones
        mc = merged.get(namespaced_id("llm_retry", rid))
        shown = mc if mc is not None else c
        detail = (mc.notes if mc is not None and mc.notes else c.notes) or ""
        items.append((rid, outcome_of(shown), detail, shown))
    for rid, d in sorted(drops.items()):
        stub = _stub_candidate(rid, d.get("raw", {}), asset.frame)
        items.append((rid, f"dropped at authoring: {d['reason']}", "", stub))
    items.sort(key=lambda it: it[0])

    paths, lines = [], ["WHAT HAPPENED TO EACH OF YOUR ROUND-A CANDIDATES:"]
    for rid, outcome, detail, cand in items:
        line = f"  - {rid}: {outcome}"
        if cand is not None and cand.seat_mode is not None:
            line = f"  - {candidate_line(cand).replace(cand.id, rid, 1)}"
        elif cand is None:
            line += " (pose too degenerate to draw)"
        lines.append(line)
        if detail:
            lines.append(f"      notes: {detail}")
        if cand is not None:
            png = out_dir / f"{rid}.png"
            write_png(mesh, [cand], png, f"{asset.name} {rid}: {outcome}")
            paths.append(png)
    report = "\n".join(lines)
    if verbose:
        print(f"  feedback: {len(paths)} marker image(s) -> {out_dir}")

    h = hashlib.sha1(report.encode())
    for p in paths:
        h.update(Path(p).read_bytes())
    return paths, report, h.hexdigest()
