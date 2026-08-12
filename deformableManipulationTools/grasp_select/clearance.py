"""Approach clearance: is there room to drive the TCP into the grasp along its own approach?

The corridor is the straight segment the hand travels down before the pads close — from
``p - standoff·a`` to the grasp centre ``p``, where ``a`` is the approach. A candidate whose
corridor passes through the tabletop, or through another object, cannot be executed however good
the grasp itself is, and finding that out here costs a handful of dot products instead of an IK
solve and a rollout.

Two things are checked, and deliberately only two:

* **the table**, always — it is under every scene and it is what makes half the buckets that survive
  face pruning still unreachable (pruning asks which way a face POINTS, this asks whether the hand
  can get there);
* **other objects**, when the caller passes them. They are boxes, not meshes: an OBB test is exact
  for the bins and boxes in the catalog and conservative for everything else, and a mesh-accurate
  swept-volume check belongs to a collision-checking stage that does not exist yet.

The TARGET object is never an obstacle to its own grasp. For an uncontained pose the object legally
protrudes past the fingertips (``grasp_library``: "jaws as deep as safe"), so testing the target
would reject exactly the grasps the seating policy deliberately produced.

The corridor is inflated by half the jaw width, so the check covers where the open fingers sweep
rather than a bare centre line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# How much straight-line run-up the hand needs before the grasp [m]. 10 cm: the demos' approach legs
# are 8-15 cm of vertical descent, and anything shorter leaves no room to blend into the pose.
DEFAULT_STANDOFF = 0.10
# Samples along the corridor. 12 over 10 cm = 8 mm apart, finer than the 25 mm the smallest catalog
# obstacle is thick, so a thin obstacle cannot slip between samples.
_SAMPLES = 12
# Clearance below the tabletop that still counts as "through the table" [m]. The pads may legally
# reach the surface (the cloth grasp presses 5 mm INTO it), so this only catches a corridor that
# genuinely arrives from underneath.
TABLE_TOLERANCE = 0.006


@dataclass(frozen=True)
class Obstacle:
    """A scene object other than the grasp target, as a world-frame box.

    ``center``/``half`` are the box centre and half-extents [m]; ``yaw`` its rotation about world z
    (catalog objects spawn as ``translate(pos) @ rot_z(yaw)``, so this is the whole placement)."""
    name: str
    center: tuple
    half: tuple
    yaw: float = 0.0

    def contains(self, point, margin: float = 0.0) -> bool:
        d = np.asarray(point, dtype=float) - np.asarray(self.center, dtype=float)
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        return bool(np.all(np.abs(local) <= np.asarray(self.half, dtype=float) + margin))


@dataclass(frozen=True)
class ClearanceCheck:
    """Result for one candidate: how much of the corridor is free, and what ended it."""
    free_length: float             # [m] along -approach from the grasp centre
    required: float
    blocker: str = ""              # "table", an obstacle name, or "" when nothing blocks
    blocked_at: float = float("inf")

    @property
    def ok(self) -> bool:
        return self.free_length >= self.required

    def describe(self) -> str:
        if self.ok:
            return f"clear {self.free_length * 100:.0f} cm"
        return (f"blocked by {self.blocker or 'unknown'} at "
                f"{self.blocked_at * 100:.1f} cm (need {self.required * 100:.0f} cm)")


def approach_clearance(position, approach, *, table_z: float, jaw_width: float = 0.0,
                       obstacles=(), standoff: float = DEFAULT_STANDOFF) -> ClearanceCheck:
    """Free run-up along the approach before the table or an obstacle gets in the way.

    ``position``/``approach`` are the WORLD grasp centre and approach direction (the way the TCP
    travels IN, so the corridor runs the other way)."""
    p = np.asarray(position, dtype=float).reshape(3)
    a = np.asarray(approach, dtype=float).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1.0e-12)
    margin = 0.5 * float(jaw_width)
    floor = float(table_z) - TABLE_TOLERANCE

    for i in range(1, _SAMPLES + 1):
        d = standoff * i / _SAMPLES
        q = p - d * a
        if q[2] < floor:
            return ClearanceCheck(free_length=standoff * (i - 1) / _SAMPLES, required=standoff,
                                  blocker="table", blocked_at=d)
        for ob in obstacles:
            if ob.contains(q, margin):
                return ClearanceCheck(free_length=standoff * (i - 1) / _SAMPLES, required=standoff,
                                      blocker=ob.name, blocked_at=d)
    return ClearanceCheck(free_length=standoff, required=standoff)


def obstacles_from_scene(scene: dict, catalog_by_name: dict, exclude: str = "") -> list:
    """Convenience: build :class:`Obstacle` boxes from a pipeline ``scene.json`` and the catalog.

    Uses each entry's catalog ``dims`` as the box, centred on the settled placement. Objects with no
    dims (cables carry nodes instead) are skipped rather than guessed at."""
    out = []
    for o in scene.get("objects", ()):
        if o.get("name") == exclude:
            continue
        dims = catalog_by_name.get(o.get("name"), {}).get("dims")
        if not dims:
            continue
        z = float(o.get("z", 0.0))
        out.append(Obstacle(name=o["name"],
                            center=(float(o.get("x", 0.0)), float(o.get("y", 0.0)),
                                    z + 0.5 * float(dims[2])),
                            half=(0.5 * float(dims[0]), 0.5 * float(dims[1]),
                                  0.5 * float(dims[2])),
                            yaw=float(o.get("yaw", 0.0))))
    return out


# Obstacle-test margin for the protruded fingertip lines [m] — half a finger's thickness.
_FINGER_MARGIN = 0.010
_BEYOND_SAMPLES = 6


def beyond_clearance(pose, seat_depth: float, span: float, width: float, *, table_z: float,
                     obstacles=()) -> ClearanceCheck:
    """Is the space the FINGERTIPS occupy BEYOND the object's far surface actually free?

    The online half of the depth-variant pair (``grasp_library`` schema v5): a ``_ctr`` centred
    candidate pushes the fingertips ``-(seat_depth + span)`` past the object's far surface — on a
    table-resting top-down grasp the table is exactly there, and on a cluttered shelf another
    object may be. Samples the two finger centerlines over that protruded segment (far-surface
    plane -> fingertip plane) against the tabletop and the scene's obstacle boxes. The flush
    sibling needs no such check (its tips sit ~0.7 mm past the surface, inside the table
    tolerance), which is why selection only calls this for centred variants."""
    pose = np.asarray(pose, dtype=float)
    p, jaw, a = pose[:3, 3], pose[:3, 0], pose[:3, 2]
    protrusion = max(0.0, -(float(seat_depth) + float(span)))
    if protrusion <= 0.002:
        return ClearanceCheck(free_length=protrusion, required=protrusion)
    floor = float(table_z) - TABLE_TOLERANCE
    half = 0.5 * float(width)
    for i in range(1, _BEYOND_SAMPLES + 1):
        d = protrusion * i / _BEYOND_SAMPLES
        # z in the grasp frame runs from the far surface (seat_depth+span) forward to the tips
        # (0); the world point sits d short of the fingertip plane along the approach.
        for side in (-1.0, 1.0):
            q = p - d * a + side * half * jaw
            free = protrusion * (i - 1) / _BEYOND_SAMPLES
            if q[2] < floor:
                return ClearanceCheck(free_length=free, required=protrusion,
                                      blocker="table", blocked_at=d)
            for ob in obstacles:
                if ob.contains(q, _FINGER_MARGIN):
                    return ClearanceCheck(free_length=free, required=protrusion,
                                          blocker=ob.name, blocked_at=d)
    return ClearanceCheck(free_length=protrusion, required=protrusion)
