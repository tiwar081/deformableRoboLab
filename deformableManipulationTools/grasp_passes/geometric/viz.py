"""Visual inspection of one asset's grasp candidates — the way to actually look at this pass's output.

    .venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz banana
    .venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz mug --method cross_section
    .venv/bin/python -m deformableManipulationTools.grasp_passes.geometric.viz wood_block --verify

Writes two artifacts per asset into ``outputs/grasp_viz/<asset>/``, because neither alone is enough:

* **``<asset>_grasps.glb``** — the object with a gripper marker at every candidate pose, coloured by
  the method that produced it (blue = ``medial_axis``, orange = ``cross_section``). Open it in any
  glTF viewer to inspect the grasps interactively, from any angle. This is the artifact to trust:
  it is the real geometry, not a projection of it.
* **``<asset>_grasps.png``** — three orthographic panels of the canonical frame, with each grasp
  drawn as its jaw chord plus a stub along the approach. This exists because the repo's environment
  has no GL context (``trimesh``'s own ``save_image`` needs a display), so a still has to be drawn
  some other way — and because three flat views next to each other are the fastest way to see
  whether coverage is even or has a hole in it.

``--verify`` adds a numeric check that does not depend on looking at anything: it re-measures each
candidate against the mesh and reports how far the pad points sit from the surface and whether the
grasp centre is inside the object. Poses are stored in the CANONICAL frame, so a systematic frame or
convention error shows up as pad points floating millimetres off the surface — the failure a picture
is worst at catching, since a wrong-but-consistent frame still looks tidy.

The viewer never writes a sidecar or a record; it only reads them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...grasp_library import (MAX_JAW_WIDTH, PAD_FAR_Z, PAD_NEAR_Z, load_grasps, pad_seat)
from ...settings import REPO_ROOT
from ..base import read_sidecar
from ..catalog import load_asset
from . import METHOD_LABELS
from .config import DEFAULT
from .marker import METHOD_COLORS, color_for, gripper_marker, method_of
from .meshprep import prepare

OUTPUT_DIR = REPO_ROOT / "outputs" / "grasp_viz"
PASS_NAME = "geometric"

# Length [m] of the approach stub drawn behind each grasp in the PNG — long enough to read the
# direction the hand comes from, short enough not to bury the object in lines.
_APPROACH_STUB = 0.02


def _candidates(asset_name: str, merged: bool):
    """This pass's candidates for an asset, or the whole merged record with ``merged``."""
    if merged:
        record = load_grasps(asset_name, use_cache=False)
        return record.candidates, f"merged record ({record.generator or 'multi-pass'})"
    side = read_sidecar(PASS_NAME, asset_name)
    if side is None:
        raise SystemExit(
            f"no '{PASS_NAME}' sidecar for {asset_name!r}. Generate it first:\n"
            f"  .venv/bin/python -m deformableManipulationTools.grasp_passes run {PASS_NAME} "
            f"--asset {asset_name}")
    return side["record"].candidates, f"{PASS_NAME} sidecar: {side['pass'].get('notes', '')}"


def _filter(candidates, method, limit, id_substring):
    out = list(candidates)
    if method:
        out = [c for c in out if method_of(c) == method]
    if id_substring:
        out = [c for c in out if id_substring in c.id]
    if limit and len(out) > limit:
        # Even stride rather than a prefix: a prefix of an id-sorted list is one end of the object.
        out = [out[i] for i in np.linspace(0, len(out) - 1, limit).round().astype(int)]
    return out


def _pads(candidate):
    """The two pad contact points implied by a candidate's pose and width."""
    t = np.asarray(candidate.transform, dtype=float)
    centre, jaw = t[:3, 3], t[:3, 0]
    half = 0.5 * float(candidate.width)
    return centre - half * jaw, centre + half * jaw


# =================================================================================================
# Artifacts
# =================================================================================================
def write_glb(mesh, candidates, path: Path) -> Path:
    """Object + a gripper marker per candidate, as a single glTF scene."""
    import trimesh

    body = mesh.copy()
    body.visual.face_colors = np.array([190, 190, 195, 210], dtype=np.uint8)
    scene = trimesh.Scene([body])
    for c in candidates:
        marker = gripper_marker(c.width, color=color_for(c))
        marker.apply_transform(np.asarray(c.transform, dtype=float))
        scene.add_geometry(marker, node_name=c.id.replace("/", "_"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    return path


def write_png(mesh, candidates, path: Path, title: str) -> Path:
    """Three orthographic panels of the canonical frame with the grasps drawn in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) > 4000:                       # even stride keeps the silhouette, not a corner of it
        verts = verts[:: int(np.ceil(len(verts) / 4000))]

    # One limit for all three panels: the canonical frame is centred on the OBB, so a shared range
    # keeps the views to the same scale and a grasp's size can be compared between them.
    half = 1.1 * float(np.abs(verts).max()) * 1000.0

    panels = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (i, j, li, lj) in zip(axes, panels):
        ax.scatter(verts[:, i] * 1000, verts[:, j] * 1000, s=1, c="#c8c8cc", alpha=0.35,
                   linewidths=0, zorder=1)
        for c in candidates:
            col = np.asarray(color_for(c)[:3]) / 255.0
            p0, p1 = _pads(c)
            ax.plot([p0[i] * 1000, p1[i] * 1000], [p0[j] * 1000, p1[j] * 1000],
                    color=col, lw=1.4, alpha=0.85, zorder=3)
            centre = np.asarray(c.transform)[:3, 3]
            back = centre - _APPROACH_STUB * np.asarray(c.approach)
            ax.plot([centre[i] * 1000, back[i] * 1000], [centre[j] * 1000, back[j] * 1000],
                    color=col, lw=0.7, alpha=0.45, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_xlabel(f"{li} [mm]")
        ax.set_ylabel(f"{lj} [mm]")
        ax.set_title(f"{li}{lj}", fontsize=10, color="0.35")
        ax.grid(alpha=0.2, lw=0.5)

    counts = {m: sum(1 for c in candidates if method_of(c) == m) for m in METHOD_LABELS}
    handles = [Line2D([], [], color=np.asarray(METHOD_COLORS[m][:3]) / 255.0, lw=2,
                      label=f"{m} ({counts[m]})") for m in METHOD_LABELS]
    other = len(candidates) - sum(counts.values())
    if other:
        handles.append(Line2D([], [], color="0.5", lw=2, label=f"other source ({other})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =================================================================================================
# Numeric verification
# =================================================================================================
def pad_containment(candidates, vertices, faces):
    """Per candidate, the fraction of the local material span that lies BETWEEN the pads.

    The check on rule 2. ``pad_seat`` is re-run on each STORED pose, which is already seated, so the
    span it reports is the material's position relative to the pose as saved. Intersecting that with
    the pad extent ``[PAD_FAR_Z, PAD_NEAR_Z]`` answers the question that matters — how much of what
    the jaws close on is actually held by pad rather than sticking out past the fingertips.

    Returns ``(fractions, advances)``; ``advances`` should be ~0 for a correctly seated pose, so a
    non-zero value here means the stored pose was NOT seated (a v1 pose, or a skipped call).
    Candidates whose column has no material at all are reported as fraction 0."""
    fractions, advances = [], []
    for c in candidates:
        seat = pad_seat(c.position, c.approach, c.jaw_axis, c.width, vertices, faces)
        if seat is None:
            fractions.append(0.0)
            advances.append(float("nan"))
            continue
        lo, hi = seat.span
        depth = hi - lo
        overlap = max(0.0, min(hi, PAD_NEAR_Z) - max(lo, PAD_FAR_Z))
        fractions.append(1.0 if depth <= 0 else overlap / depth)
        advances.append(seat.advance)
    return np.asarray(fractions), np.asarray(advances)


def verify(mesh, candidates, watertight: bool) -> dict:
    """Re-measure candidates against the mesh: pad-to-surface distance and centre containment.

    A candidate claims the jaws close on the object at ``width``; that means both pad points lie ON
    the surface. Any systematic offset here is a frame or convention error, which is exactly what a
    rendering will not show you."""
    import trimesh

    if not candidates:
        return {"n": 0}
    pads = np.array([p for c in candidates for p in _pads(c)])
    _close, dist, _tri = trimesh.proximity.closest_point(mesh, pads)
    dist = np.asarray(dist, dtype=float).reshape(len(candidates), 2)
    worst = dist.max(axis=1)

    centres = np.array([np.asarray(c.transform)[:3, 3] for c in candidates])
    inside = mesh.contains(centres) if watertight else None
    widths = np.array([c.width for c in candidates])
    return {
        "n": len(candidates),
        "pad_dist_med": float(np.median(worst)),
        "pad_dist_p95": float(np.percentile(worst, 95)),
        "pad_dist_max": float(worst.max()),
        "centre_inside": None if inside is None else int(np.sum(inside)),
        "width_ok": int(np.sum((widths > 0) & (widths <= MAX_JAW_WIDTH))),
    }


# =================================================================================================
# CLI
# =================================================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="grasp_passes.geometric.viz", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", help="catalog object name (e.g. banana)")
    ap.add_argument("--method", choices=list(METHOD_LABELS), default=None,
                    help="show only candidates from one method")
    ap.add_argument("--id", dest="id_substring", default=None,
                    help="show only candidates whose id contains this")
    ap.add_argument("--limit", type=int, default=40,
                    help="max candidates to draw, spread evenly (default 40; 0 = all)")
    ap.add_argument("--merged", action="store_true",
                    help="read the merged record (every pass) instead of this pass's sidecar")
    ap.add_argument("--verify", action="store_true",
                    help="also re-measure the candidates against the mesh")
    ap.add_argument("--out", type=Path, default=None, help=f"output directory (default {OUTPUT_DIR})")
    args = ap.parse_args(argv)

    asset = load_asset(args.asset)
    prepared = prepare(asset.name, asset.canonical_vertices(), asset.faces, DEFAULT)
    everything, provenance = _candidates(args.asset, args.merged)
    shown = _filter(everything, args.method, args.limit, args.id_substring)

    print(f"{args.asset}  kind={asset.kind}  extents "
          f"{np.round(asset.extents * 1000, 1).tolist()} mm  "
          f"{'watertight' if prepared.watertight else 'OPEN surface'}")
    print(f"  source: {provenance}")
    counts = {m: sum(1 for c in everything if method_of(c) == m) for m in METHOD_LABELS}
    print(f"  {len(everything)} candidate(s): " + ", ".join(f"{m}={n}" for m, n in counts.items())
          + f"  -> drawing {len(shown)}")
    for c in shown[:12]:
        print(f"    {c.id:32s} {method_of(c):14s} face={c.face:2s} "
              f"width={c.width * 1000:5.1f} mm")
    if len(shown) > 12:
        print(f"    ... and {len(shown) - 12} more")

    out_dir = (args.out or OUTPUT_DIR) / args.asset
    stem = f"{args.asset}_grasps" + (f"_{args.method}" if args.method else "")
    glb = write_glb(prepared.mesh, shown, out_dir / f"{stem}.glb")
    png = write_png(prepared.mesh, shown, out_dir / f"{stem}.png",
                    f"{args.asset} — grasp candidates in the canonical frame")
    print(f"  wrote {glb.relative_to(REPO_ROOT)}")
    print(f"  wrote {png.relative_to(REPO_ROOT)}")

    if args.verify:
        stats = verify(prepared.mesh, shown, prepared.watertight)
        if not stats["n"]:
            print("  verify: nothing to check")
            return
        print(f"  verify ({stats['n']} shown): pad-to-surface distance "
              f"median {stats['pad_dist_med'] * 1000:.2f} mm, "
              f"p95 {stats['pad_dist_p95'] * 1000:.2f} mm, "
              f"max {stats['pad_dist_max'] * 1000:.2f} mm")
        print(f"           width within the jaw: {stats['width_ok']}/{stats['n']}")
        if stats["centre_inside"] is not None:
            outside = stats["n"] - stats["centre_inside"]
            # Not a failure count. A grasp that SPANS a hollow object — across a mug, over a handle
            # gap — has material at both pads and void between them, so its centre is legitimately
            # in air. The number worth watching is the pad distance above.
            print(f"           grasp centre inside the mesh: {stats['centre_inside']}/{stats['n']}"
                  + (f" ({outside} span a hollow — expected, not a fault)" if outside else ""))
        else:
            print("           containment not checked (open surface)")


if __name__ == "__main__":
    main()
