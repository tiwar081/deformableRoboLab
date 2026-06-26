"""EXPERIMENT: Franka grasps a T-shirt (Newton's `example_cloth_franka` shirt) under THIS framework's
existing grip regime — the split MuJoCo-robot + VBD-cloth path with the force-target proxy GripController.

Only the SHIRT placement (the vendored `unisex_shirt.usd` mesh, dropped onto the table) and an
open-loop grasp policy (descend -> close -> lift -> drag, after Newton's corner-grasp sequence) are
copied; everything else (solver split, proxy grip, coupling) is the framework's. The cloth is a thin
SHELL, NOT a compressible FEM volume, and nothing here is tuned for it (metre scale; see ClothConfig) —
this is a "does the existing regime even hold a shirt" probe. Grasp mode below is `compressible=False`
(a shirt is a thin shell, closer to the incompressible bucket than to the soft-FEM block — see the
project discussion); no re-tuning attempted.

Run: python -m examples cloth_franka --device cuda:0
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from deformableManipulationTools.helper import install_terminal_log

_terminal_log = install_terminal_log()

import warp as wp

import examples
from robolabViz.scenic import ScenicGraspExample
from deformableManipulationTools import FRANKA, TABLE, PARTICLE_SOLVER_KWARGS, GraspWindow, add_table, solve_gripper_ik, wp_smoothstep
from deformableManipulationTools.assets import ClothConfig, add_cloth

CLOTH = ClothConfig()


@wp.func
def _blend(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, i: int) -> float:
    return (1.0 - s) * a[i] + s * b[i]


@wp.kernel
def _set_arm_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home_q: wp.array(dtype=float),
    pregrasp_q: wp.array(dtype=float),
    pickup_q: wp.array(dtype=float),
    lift_q: wp.array(dtype=float),
    drag_q: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
):
    # ARM DOFs only; the fingers are owned by the centralized force-target GripController.
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    settle_start = 2.5     # let the shirt drop + settle on the table first
    descend_start = 3.2
    close_start = 4.2      # (grasp window opens here; see grasp_windows)
    lift_start = 4.6
    drag_start = 6.2

    if t < settle_start:
        q = home_q[i]
    elif t < descend_start:
        q = _blend(home_q, pregrasp_q, wp_smoothstep((t - settle_start) / (descend_start - settle_start)), i)
    elif t < close_start:
        q = _blend(pregrasp_q, pickup_q, wp_smoothstep((t - descend_start) / (close_start - descend_start)), i)
    elif t < lift_start:
        q = pickup_q[i]
    elif t < drag_start:
        q = _blend(pickup_q, lift_q, wp_smoothstep((t - lift_start) / (drag_start - lift_start)), i)
    else:
        q = _blend(lift_q, drag_q, wp_smoothstep((t - drag_start) / 2.0), i)

    joint_target_q[i] = q


class Example(ScenicGraspExample):
    has_particles = True
    soft_block = CLOTH                       # framework reads soft_contact_* off this
    coupling_soft_ke = CLOTH.soft_contact_ke  # enable proxy<->particle grip-force harvest

    def configure(self, args):
        self.table_top_z = TABLE.top_z
        # Lay the shirt FLAT on the table (clear of the gripper), then grasp near its near edge. The
        # shirt is laid flat (ClothConfig.flatten_z) and dropped from just above the table so it never
        # touches the gripper proxies at init (which would otherwise drag the whole arm down).
        self.cloth_center_xy = np.array([0.10, -0.45], dtype=np.float32)
        self.grasp_xy = np.array([0.10, -0.25], dtype=np.float32)   # near edge (toward the robot)
        self.drag_xy = np.array([0.34, -0.30], dtype=np.float32)
        self.cloth_drop_pos = np.array([self.cloth_center_xy[0], self.cloth_center_xy[1],
                                        self.table_top_z + 0.04], dtype=np.float32)
        self.grasp_tcp_height = self.table_top_z + 0.015   # descend to just above the table to pinch the cloth
        self.lift_height = self.table_top_z + 0.22
        # Unified admittance grasp: regulate to GRIP.force_target (default, no per-demo override). Held to
        # the run's end (no release). EXPERIMENTAL probe — see header; not tuned, may not hold the shell.
        self.grasp_windows = [GraspWindow(close_start=4.2, close_end=4.6)]
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 2048,
                                     "rigid_body_particle_contact_buffer_size": 16384, **PARTICLE_SOLVER_KWARGS}
        self.object_pipeline_kwargs = {"soft_contact_margin": CLOTH.contact_margin}

    def plan(self, ik_model, ik_state):
        def ik_at(xy, z):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array([xy[0], xy[1], z], dtype=np.float32), self.gripper_open)

        self.pregrasp_q = ik_at(self.grasp_xy, self.table_top_z + 0.15)
        self.pickup_q = ik_at(self.grasp_xy, self.grasp_tcp_height)
        self.lift_q = ik_at(self.grasp_xy, self.lift_height)
        self.drag_q = ik_at(self.drag_xy, self.lift_height)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = kf(self.home_q)
        self._pregrasp_q_wp = kf(self.pregrasp_q)
        self._pickup_q_wp = kf(self.pickup_q)
        self._lift_q_wp = kf(self.lift_q)
        self._drag_q_wp = kf(self.drag_q)

    def build_scene(self, builder, robot_builder):
        from deformableManipulationTools import build_gripper_proxies
        self._obj_table_shape = add_table(builder, TABLE)
        self.cloth_particle_start, self.cloth_particle_count = add_cloth(builder, CLOTH, self.cloth_drop_pos)
        # Proxies WITH particle collision so the pads grip the cloth.
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape)

    def set_arm_targets(self, substep):
        wp.launch(_set_arm_targets_kernel, dim=FRANKA.n_arm_dof, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp,
            self._pickup_q_wp, self._lift_q_wp, self._drag_q_wp,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        # EXPERIMENT: assert only that the sim stayed sane (finite, no blow-up). Whether the shirt is
        # actually lifted is the OUTCOME we report, not a pass/fail gate.
        pq = self.object_state_0.particle_q.numpy()
        if not np.all(np.isfinite(pq)):
            raise ValueError("Non-finite cloth particle position detected (sim blew up).")
        qd = self.object_state_0.particle_qd.numpy()
        if np.max(np.abs(qd)) > 200.0:
            raise ValueError(f"Cloth particle velocity exploded (max |v|={np.max(np.abs(qd)):.1f} m/s).")

    @staticmethod
    def create_parser():
        # Capped at the stable/informative window: with the init fixed, the gripper descends + grasps +
        # lifts cleanly to ~6 s; beyond that the UNTUNED VBD cloth destabilizes under the lift (part of
        # the shell tunnels the table, then NaNs ~6.2 s) — an untuned-cloth issue, reported, not hidden.
        parser = examples.create_parser()
        parser.set_defaults(num_frames=360)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=6, help="VBD iterations for the cloth.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="cloth_franka")
    examples.run(Example(viewer, args), args)
