"""Uncollapsed FK mirror of the simulated robot for visual puppeteering.

The physics robot is imported with ``collapse_fixed_joints=True``, so
fixed-mounted links (``fr3_hand``, ``fr3_link8``, ...) have no body of their
own in the simulation model — but the rendered robot USD keeps every URDF link
as a prim. This class builds a second Newton model from the same URDF with
``collapse_fixed_joints=False`` (identical generalized coordinates: fixed
joints add no DOFs) and runs ``eval_fk`` on the simulated ``joint_q`` each
rendered frame, yielding a world transform for *every* URDF link by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton


class RobotVisualFK:
    def __init__(self, urdf_path: Path | str, base_xform: wp.transform, device, expected_joint_coords: int):
        builder = newton.ModelBuilder()
        builder.add_urdf(
            urdf_path,
            xform=base_xform,
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=False,
            force_show_colliders=False,
        )
        self.model = builder.finalize(device=device)
        if self.model.joint_coord_count != expected_joint_coords:
            raise ValueError(
                f"FK mirror has {self.model.joint_coord_count} joint coords, "
                f"but the simulated robot has {expected_joint_coords}; the URDFs do not match."
            )
        self.state = self.model.state()
        # Newton prefixes body labels with the import root (e.g. "fr3/fr3_hand");
        # USD link prims use the bare URDF link names.
        self.link_names: list[str] = [label.split("/")[-1] for label in self.model.body_label]
        if len(set(self.link_names)) != len(self.link_names):
            raise ValueError(f"Duplicate link names after prefix strip: {self.link_names}")

    def link_world_transforms(self, joint_q: wp.array) -> dict[str, np.ndarray]:
        """FK from the simulated joint coordinates -> {link_name: (px,py,pz,qx,qy,qz,qw)} in sim frame."""
        wp.copy(self.model.joint_q, joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        body_q = self.state.body_q.numpy()
        return {name: body_q[i] for i, name in enumerate(self.link_names)}
