"""Arm feasibility: can the ARM actually hold a planned TCP pose? (IK-verified, FK-measured.)

``grasp_select``'s reach stage is a radius test and its projection proves the ORIENTATION
vocabulary covers SO(3) — but neither knows the arm's joint limits or its inner workspace
boundary. Measured on the first live rollout (2026-08-12, tool_sorting_station): a side grasp
0.30 m from the base passed both and the executor's IK then missed it by 21 cm — the rollout burnt
five minutes to learn what one IK solve knows in milliseconds. So the trajectory stage verifies
every drawn candidate's pre-grasp + grasp poses (and the plan's place pose) through the SAME
solver the executor uses (``robot.solve_gripper_ik``, seed ladder included), measuring the
solution's TCP error by FK; a pose the arm cannot hold within tolerance rejects the candidate
BEFORE a rollout is spent.

One robot build per checker (~seconds); each pose check is an IK solve + one FK (milliseconds).
"""
from __future__ import annotations

import numpy as np

# A solution further than this from the commanded TCP means the arm cannot hold the pose.
# 15 mm: half the fingertip pad's usable depth — beyond it the jaws close on different geometry
# than the candidate measured (same scale as grasp_select's MAX_DISTORTION rationale).
TCP_TOL = 0.015


class ArmChecker:
    """Lazy one-robot IK context bound to a placement's base transform."""

    def __init__(self, robot_base_xform, robot_table=None):
        import warp as wp
        import newton
        from ..params import FRANKA, TABLE
        from ..mathutils import find_body
        from ..robot import build_franka_robot

        builder = build_franka_robot(xform=robot_base_xform,
                                     table=robot_table if robot_table is not None else TABLE)
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.ee_body = find_body(list(self.model.body_label), FRANKA.ee_link_suffix)
        self.ee_offset = np.array(FRANKA.ee_offset, dtype=np.float32)
        self.gripper_open = float(FRANKA.gripper_open)
        self._wp = wp
        self._newton = newton

    def tcp_error(self, pos, yaw: float = 0.0, tilt: float = 0.0,
                  tilt_axis=(1.0, 0.0, 0.0)) -> float:
        """IK the pose, FK the solution, return |achieved TCP - commanded TCP| [m].

        The state is FK-RESET TO HOME first: the solver seeds from the entry state, so without
        the reset a check's verdict depends on whatever pose the PREVIOUS check left behind —
        measured 2026-08-13: the same candidate passed the gate in one task and 'missed by
        85 mm' in another, purely from check order."""
        from ..mathutils import quat_rotate_xyzw
        from ..robot import solve_gripper_ik

        self._newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        q = solve_gripper_ik(self.model, self.state, self.ee_body, self.ee_offset,
                             np.asarray(pos, dtype=np.float32), self.gripper_open,
                             yaw=float(yaw), tilt=float(tilt), tilt_axis=tuple(tilt_axis))
        q_arr = self._wp.array(np.asarray(q, dtype=np.float32), dtype=self._wp.float32,
                               device=self.model.device)
        self._newton.eval_fk(self.model, q_arr, self.model.joint_qd, self.state)
        body_q = self.state.body_q.numpy()[self.ee_body]
        tcp = body_q[:3] + quat_rotate_xyzw(body_q[3:7], self.ee_offset)
        return float(np.linalg.norm(tcp - np.asarray(pos, dtype=np.float32)))

    def pose_ok(self, pos, yaw: float = 0.0, tilt: float = 0.0, tilt_axis=(1.0, 0.0, 0.0),
                tol: float = TCP_TOL) -> tuple:
        err = self.tcp_error(pos, yaw, tilt, tilt_axis)
        return err <= tol, err

    def path_errors(self, waypoints: list, grasp_window: dict) -> list:
        """FK-measured TCP error of THE EXECUTOR'S OWN path solve, per waypoint.

        Byte-identical inputs to ``demo_runner.plan``: the plan's waypoint dicts become the same
        ``(pos, yaw, tilt, tilt_axis)`` specs, gripped segments get the same 8.0 branch-jump
        weight, and the state is FK-reset to home first (the executor solves from a fresh home
        state too). This is the gate the per-pose ladder cannot be: pass-1 chained seeding and the
        global branch pass can land on a different branch than a single laddered solve — measured
        2026-08-12 (a side grasp my per-pose check accepted at 4 mm was executed 208 mm off).
        Returns ``[{t, err, via}]`` in waypoint order."""
        from ..mathutils import quat_rotate_xyzw
        from ..robot import solve_gripper_ik_path

        self._newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        specs = [(None if w.get("pos") is None else np.asarray(w["pos"], dtype=np.float32),
                  float(w.get("yaw", 0.0)), float(w.get("tilt", 0.0)),
                  tuple(w.get("tilt_axis", (1.0, 0.0, 0.0)))) for w in waypoints]
        wins = grasp_window if isinstance(grasp_window, list) else [grasp_window]

        def gripped(t0: float, t1: float) -> bool:
            return any(float(w["close_start"]) < t1 and t0 < float(w["release_end"])
                       for w in wins)

        edge_w = [8.0 if gripped(a["t"], b["t"]) else 1.0
                  for a, b in zip(waypoints, waypoints[1:])]
        qs = solve_gripper_ik_path(self.model, self.state, self.ee_body, self.ee_offset,
                                   specs, self.gripper_open, edge_weights=edge_w)
        out = []
        for w, q in zip(waypoints, qs):
            q_arr = self._wp.array(np.asarray(q, dtype=np.float32), dtype=self._wp.float32,
                                   device=self.model.device)
            self._newton.eval_fk(self.model, q_arr, self.model.joint_qd, self.state)
            bq = self.state.body_q.numpy()[self.ee_body]
            tcp = bq[:3] + quat_rotate_xyzw(bq[3:7], self.ee_offset)
            err = float(np.linalg.norm(tcp - np.asarray(w["pos"], dtype=np.float32))) \
                if w.get("pos") is not None else 0.0
            out.append({"t": float(w["t"]), "err": err, "via": bool(w.get("via", False))})
        return out


def checker_for_placement(placement: dict) -> ArmChecker:
    """An :class:`ArmChecker` at the run's robot placement (same transform the demo builds)."""
    import math

    import warp as wp

    base = placement["base"]
    xform = wp.transform(
        wp.vec3(float(base[0]), float(base[1]), float(base[2])),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(placement["yaw_deg"])))
    return ArmChecker(xform)
