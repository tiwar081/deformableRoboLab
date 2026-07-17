"""Footprint geometry for the scene solver: oriented-bounding-box (OBB) overlap via the separating
axis theorem, plus the RoboLab container/stacking fit models. Used to make collision resolution
ELONGATED-AWARE — RoboLab's spatial solver reduces every object to a max(w,d) bounding circle (its
"convex hull" docstring is dead code), which is wasteful for long thin objects (a 0.18 m banana
gets a 0.18 m circle in both axes). Here a rotated rectangle is tested as a true OBB, so a yawed
banana only conflicts along its actual footprint.

Conventions: a footprint is (cx, cy, yaw_rad, w, d) — center, rotation about +z, full width (local
x) and depth (local y). All lengths in metres.
"""
from __future__ import annotations

import math


def obb_corners(cx: float, cy: float, yaw: float, w: float, d: float) -> list[tuple[float, float]]:
    """The 4 world-frame corners of a (w x d) rectangle centered at (cx, cy), rotated by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = 0.5 * w, 0.5 * d
    return [(cx + c * sx * hx - s * sy * hy, cy + s * sx * hx + c * sy * hy)
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def _axes(yaw: float) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    return [(c, s), (-s, c)]                 # local x and y unit axes in world


def obb_penetration(a: tuple, b: tuple, margin: float = 0.0) -> tuple[float, float, float]:
    """SAT overlap between two OBBs ``a``/``b`` = (cx, cy, yaw, w, d), each inflated by ``margin``
    on every side. Returns ``(depth, nx, ny)``: the minimum translation to separate them along the
    unit axis (nx, ny) that points from ``a`` toward ``b``. ``depth <= 0`` means no overlap.
    Tests the 4 candidate axes (both rectangles' local x/y — sufficient for rectangles)."""
    (ax, ay, ayaw, aw, ad) = a
    (bx, by, byaw, bw, bd) = b
    ca = obb_corners(ax, ay, ayaw, aw + 2 * margin, ad + 2 * margin)
    cb = obb_corners(bx, by, byaw, bw + 2 * margin, bd + 2 * margin)
    best_depth, best_axis = math.inf, (1.0, 0.0)
    for nx, ny in _axes(ayaw) + _axes(byaw):
        amin = min(px * nx + py * ny for px, py in ca)
        amax = max(px * nx + py * ny for px, py in ca)
        bmin = min(px * nx + py * ny for px, py in cb)
        bmax = max(px * nx + py * ny for px, py in cb)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap <= 0.0:
            return 0.0, 0.0, 0.0                       # a separating axis exists -> no collision
        if overlap < best_depth:
            best_depth = overlap
            # orient the axis from a's center toward b's center
            if (bx - ax) * nx + (by - ay) * ny < 0:
                nx, ny = -nx, -ny
            best_axis = (nx, ny)
    return best_depth, best_axis[0], best_axis[1]


def rotated_extent(w: float, d: float, yaw: float) -> tuple[float, float]:
    """AABB extent of a (w x d) rectangle rotated by ``yaw`` (RoboLab ``_rotated_footprint``)."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return w * c + d * s, d * c + w * s


def fits_support_rectangle(local_x: float, local_y: float, fx: float, fy: float,
                           support_w: float, support_d: float, eps: float = 1e-9) -> bool:
    """Footprint (AABB extent fx x fy at support-local offset) fully inside the support top
    rectangle (RoboLab ``_fits_support_rectangle``)."""
    return (abs(local_x) + fx / 2 <= support_w / 2 + eps
            and abs(local_y) + fy / 2 <= support_d / 2 + eps)


def support_ratio(obj_w: float, obj_d: float, obj_yaw: float,
                  support_w: float, support_d: float) -> float:
    """Fraction of the object's footprint AABB that lies over the support top (centred stack).
    1.0 = fully supported; RoboLab flags stacks below ~0.5-0.8 as unstable. Uses the rotated AABB
    extent, so a yawed elongated object's true overhang is captured."""
    fx, fy = rotated_extent(obj_w, obj_d, obj_yaw)
    over_x = max(0.0, min(fx, support_w) )
    over_y = max(0.0, min(fy, support_d))
    return (over_x * over_y) / max(fx * fy, 1e-9)


def fits_container_ellipse(local_x: float, local_y: float, fx: float, fy: float,
                           radius_x: float, radius_y: float, *, partial: float = 0.0) -> bool:
    """Footprint fit in the container-mouth ellipse (RoboLab ``_fits_container_ellipse``, mouth
    semi-axes = 0.43 x dims). With ``partial == 0`` this is RoboLab-strict: the far AABB corner
    must lie inside the ellipse. With ``partial`` in (0, 1] it is a PARTIAL-CONTAINMENT model for
    elongated objects (a banana in a bowl) that physically settle in even though a flat corner
    pokes past the rim: the object's CENTER must be inside the mouth AND at least ``1 - partial``
    of its footprint AABB (sampled) must lie within the ellipse — i.e. ``partial`` is the maximum
    fraction allowed to protrude."""
    if partial <= 0.0:
        nx = (abs(local_x) + fx / 2) / radius_x
        ny = (abs(local_y) + fy / 2) / radius_y
        return nx * nx + ny * ny <= 1.0
    if (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2 > 1.0:
        return False                               # center must be over the mouth
    # Partial containment for an elongated object (banana in a bowl): it settles/drapes in if its
    # SHORTER footprint axis fits the mouth (the long axis may protrude by up to ``partial`` of the
    # mouth) AND at least (1 - partial) of the footprint AABB samples lie inside the ellipse.
    short, long = sorted((fx, fy))
    short_r = radius_y if fx >= fy else radius_x   # the mouth radius across the object's short axis
    if short / 2 > short_r * (1.0 + partial):
        return False
    n, inside = 5, 0
    for i in range(n):
        for j in range(n):
            px = local_x + (i / (n - 1) - 0.5) * fx
            py = local_y + (j / (n - 1) - 0.5) * fy
            if (px / radius_x) ** 2 + (py / radius_y) ** 2 <= 1.0:
                inside += 1
    return inside / (n * n) >= 1.0 - partial


def container_mouth(container_dims: list[float]) -> tuple[float, float, float]:
    """(radius_x, radius_y, top_z_offset) for a container's opening — semi-axes 0.43 x footprint,
    the RoboLab ellipse-mouth model. top_z_offset = half the container height (mouth at the rim)."""
    return 0.43 * container_dims[0], 0.43 * container_dims[1], 0.5 * container_dims[2]


def pack_into_container(obj_dims_list: list[list[float]], container_dims: list[float], *,
                        partial: float = 0.25, layer_gap: float = 0.025):
    """Multi-object upward-layered ellipse packing (RoboLab ``_solve_place_in``): pack the objects
    into the container mouth, overflowing UPWARD in layers rather than through the wall, so physics
    settling drops the pile in. Returns ``(placements, ok, reason)`` where placements is a list of
    ``(local_x, local_y, local_yaw, z_above_mouth)`` and ``ok`` says all fit. Objects are packed
    largest-footprint-first; each tries existing layers then a fresh layer stacked above."""
    rx, ry, _ = container_mouth(container_dims)
    order = sorted(range(len(obj_dims_list)),
                   key=lambda i: -(obj_dims_list[i][0] * obj_dims_list[i][1]))
    yaw_choices = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    offsets = [(0.0, 0.0)] + [(rx * f * math.cos(2 * math.pi * k / n), ry * f * math.sin(2 * math.pi * k / n))
                              for f, n in ((0.35, 8), (0.55, 12)) for k in range(n)]
    layers: list[dict] = []
    placements: dict[int, tuple] = {}
    unplaced = []
    for idx in order:
        w, d, h = obj_dims_list[idx]
        placed = False
        for layer in layers + [None]:
            if layer is None:                          # open a new layer stacked above
                bottom = 0.02 if not layers else layers[-1]["bottom"] + layers[-1]["height"] + layer_gap
                layer = {"bottom": bottom, "height": 0.0, "rects": []}
                new_layer = True
            else:
                new_layer = False
            for yaw in yaw_choices:
                fx, fy = rotated_extent(w, d, yaw)
                for lx, ly in offsets:
                    if not fits_container_ellipse(lx, ly, fx, fy, rx, ry, partial=partial):
                        continue
                    if any(abs(lx - ox) < (fx + ofx) / 2 + 0.005 and abs(ly - oy) < (fy + ofy) / 2 + 0.005
                           for ox, oy, ofx, ofy in layer["rects"]):
                        continue
                    if new_layer:
                        layers.append(layer)
                    layer["rects"].append((lx, ly, fx, fy))
                    layer["height"] = max(layer["height"], h)
                    placements[idx] = (lx, ly, yaw, layer["bottom"] + h / 2)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            unplaced.append(idx)
    ok = not unplaced
    reason = ("all fit" if ok else
              f"{len(unplaced)} object(s) do not fit the container even stacked in layers")
    return [placements.get(i) for i in range(len(obj_dims_list))], ok, reason
