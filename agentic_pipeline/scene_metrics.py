"""Scene quality metrics — RoboLab's ``FeedbackSystem.generate_scene_evaluation`` /
``generate_success_feedback`` formulas (feedback_system.py), applied to our scene dict. These are
diagnostic scores stored on the scene manifest; they do not gate acceptance (the solver + settle
check do that), but a low score can prompt a re-generation and is surfaced to the user.
"""
from __future__ import annotations

import math

from . import scene_generator as sg   # catalog helpers

_CONTAINER_TOKENS = ("bowl", "bin", "box", "container", "bucket", "cabinet", "crate", "tote", "mug",
                     "pitcher")


def scene_metrics(scene: dict) -> dict:
    """Compactness, diversity, has_container, coverage — RoboLab's exact formulas.
    - compactness = 1 / (1 + mean(||pos_i - centroid||))   (0 if <2 objects; higher = clustered)
    - diversity   = |unique object classes| / num_objects
    - has_container = any object name/class reads as an open-top container
    - coverage    = num_objects / (table_area * 100)       (RoboLab's rough estimate)
    """
    x0, x1, y0, y1 = sg.workspace_bounds(margin=0.0)
    table_area = (x1 - x0) * (y1 - y0)
    by_name = sg.catalog_by_name()
    objs = scene.get("objects", [])
    n = len(objs)
    pts = [(float(o["x"]), float(o["y"])) for o in objs if "x" in o and "y" in o]

    compactness = 0.0
    if len(pts) > 1:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        mean_d = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts)
        compactness = 1.0 / (1.0 + mean_d)

    classes = {by_name[o["name"]]["class"] for o in objs}
    diversity = len(classes) / max(n, 1)
    has_container = any(any(tok in o["name"].lower() or tok in by_name[o["name"]]["class"].lower()
                            for tok in _CONTAINER_TOKENS) for o in objs)
    coverage = n / max(table_area * 100.0, 1.0)

    return {"num_objects": n, "compactness": round(compactness, 3),
            "diversity": round(diversity, 3), "has_container": has_container,
            "coverage": round(coverage, 3)}


def quality_feedback(m: dict) -> list[str]:
    """RoboLab's qualitative thresholds, as human-readable notes (empty = nothing notable)."""
    notes = []
    if m["num_objects"] < 3:
        notes.append("scene is sparse (fewer than 3 objects)")
    elif m["num_objects"] > 15:
        notes.append("scene is very dense (more than 15 objects)")
    if not m["has_container"]:
        notes.append("no container present")
    if m["compactness"] < 0.3:
        notes.append("objects are well-distributed (low compactness)")
    elif m["compactness"] > 0.7:
        notes.append("objects are tightly clustered (high compactness)")
    return notes
