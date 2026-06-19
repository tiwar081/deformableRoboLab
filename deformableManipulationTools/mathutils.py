"""Small math helpers shared across the physics package and the example policies."""
from __future__ import annotations

import numpy as np
import warp as wp


def quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def find_body(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.endswith(suffix):
            return i
    raise ValueError(f"Could not find body ending with {suffix!r}.")


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.func
def wp_smoothstep(x: float) -> float:
    x = wp.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])
