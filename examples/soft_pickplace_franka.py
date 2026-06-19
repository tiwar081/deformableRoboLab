"""Franka picks a small soft FEM block off the table and places it at a target location. 16
substeps. The block is FEM particles, so the proxies carry particle collision and the coupling
harvests the proxy<->particle reaction. Pure SCENE + POLICY; all physics/robot/asset detail comes
from :mod:`deformableManipulationTools`, all rendering from :class:`robolabViz.scenic.ScenicGraspExample`.

Run: python -m examples soft_pickplace_franka --device cuda:0
  (scenic by default -> outputs/soft_pickplace_franka/{frames/, simulation.mp4};
   --output-style basic writes outputs/soft_pickplace_franka.usd instead.)
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
from deformableManipulationTools import (
    GRIP, SOFT_BLOCK, TABLE, PARTICLE_SOLVER_KWARGS,
    add_table, add_soft_block, build_gripper_proxies, solve_gripper_ik, wp_smoothstep,
)


@wp.func
def _blend(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, i: int) -> float:
    return (1.0 - s) * a[i] + s * b[i]


@wp.kernel
def _set_robot_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home_q: wp.array(dtype=float),
    pregrasp_q: wp.array(dtype=float),
    pickup_q: wp.array(dtype=float),
    lift_q: wp.array(dtype=float),
    place_high_q: wp.array(dtype=float),
    place_q: wp.array(dtype=float),
    gripper_open: float,
    gripper_closed: float,
    joint_target_q: wp.array(dtype=float),
):
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    descend_start = 2.2
    close_start = 3.2
    lift_start = 4.4
    over_start = 5.2
    lower_start = 6.4
    hold_start = 7.4
    release_start = 8.0
    retreat_start = 8.6
    home_start = 9.2

    if t < descend_start:
        q = _blend(home_q, pregrasp_q, wp_smoothstep(t / descend_start), i)
    elif t < close_start:
        q = _blend(pregrasp_q, pickup_q, wp_smoothstep((t - descend_start) / (close_start - descend_start)), i)
    elif t < lift_start:
        q = pickup_q[i]
    elif t < over_start:
        q = _blend(pickup_q, lift_q, wp_smoothstep((t - lift_start) / (over_start - lift_start)), i)
    elif t < lower_start:
        q = _blend(lift_q, place_high_q, wp_smoothstep((t - over_start) / (lower_start - over_start)), i)
    elif t < hold_start:
        q = _blend(place_high_q, place_q, wp_smoothstep((t - lower_start) / (hold_start - lower_start)), i)
    elif t < retreat_start:
        q = place_q[i]
    elif t < home_start:
        q = _blend(place_q, place_high_q, wp_smoothstep((t - retreat_start) / (home_start - retreat_start)), i)
    else:
        q = _blend(place_high_q, home_q, wp_smoothstep((t - home_start) / 2.0), i)

    if i >= 7:
        if t < close_start:
            q = gripper_open
        elif t < lift_start:
            alpha = wp_smoothstep((t - close_start) / (lift_start - close_start))
            q = (1.0 - alpha) * gripper_open + alpha * gripper_closed
        elif t < release_start:
            q = gripper_closed
        elif t < retreat_start:
            alpha = wp_smoothstep((t - release_start) / (retreat_start - release_start))
            q = (1.0 - alpha) * gripper_closed + alpha * gripper_open
        else:
            q = gripper_open

    joint_target_q[i] = q


class Example(ScenicGraspExample):
    has_particles = True
    soft_block = SOFT_BLOCK
    coupling_soft_ke = SOFT_BLOCK.soft_contact_ke

    def configure(self, args):
        self.table_top_z = TABLE.top_z
        soft = SOFT_BLOCK
        self.block_half = 0.5 * soft.dim[0] * soft.cell
        self.pick_xy = np.array([0.10, -0.50], dtype=np.float32)
        self.place_xy = np.array([0.34, -0.28], dtype=np.float32)
        self.soft_start_pos = np.array([self.pick_xy[0], self.pick_xy[1], self.table_top_z], dtype=np.float32)
        self.grasp_tcp_height = self.table_top_z + self.block_half
        self.lift_height = self.table_top_z + 0.16
        # Squeeze the compliant block: pads ~5 mm inside the block half-width.
        self.gripper_closed = self.block_half - 0.005
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 2048,
                                     "rigid_body_particle_contact_buffer_size": 4096, **PARTICLE_SOLVER_KWARGS}
        self.object_pipeline_kwargs = {"soft_contact_margin": soft.contact_margin}

    def plan(self, ik_model, ik_state):
        def ik_at(xy, z):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array([xy[0], xy[1], z], dtype=np.float32), self.gripper_open)

        self.pregrasp_q = ik_at(self.pick_xy, self.table_top_z + 0.12)
        self.pickup_q = ik_at(self.pick_xy, self.grasp_tcp_height)
        self.lift_q = ik_at(self.pick_xy, self.lift_height)
        self.place_high_q = ik_at(self.place_xy, self.lift_height)
        self.place_q = ik_at(self.place_xy, self.grasp_tcp_height)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = kf(self.home_q)
        self._pregrasp_q_wp = kf(self.pregrasp_q)
        self._pickup_q_wp = kf(self.pickup_q)
        self._lift_q_wp = kf(self.lift_q)
        self._place_high_q_wp = kf(self.place_high_q)
        self._place_q_wp = kf(self.place_q)

    def build_scene(self, builder, robot_builder):
        self._obj_table_shape = add_table(builder, TABLE)
        add_soft_block(builder, SOFT_BLOCK, self.soft_start_pos)
        # Proxies WITH particle collision so the pads grip the soft block.
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape,
            gap=GRIP.proxy_margin * 8, has_particle_collision=True)

    def set_robot_targets(self, substep):
        wp.launch(_set_robot_targets_kernel, dim=9, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp,
            self._pickup_q_wp, self._lift_q_wp, self._place_high_q_wp, self._place_q_wp,
            self.gripper_open, self.gripper_closed,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        pq = self.object_state_0.particle_q.numpy()
        if not np.all(np.isfinite(pq)):
            raise ValueError("Non-finite soft-body particle position detected.")
        if np.min(pq[:, 2]) < self.table_top_z - 0.03:
            raise ValueError("The soft block fell through the table.")
        if self.sim_time >= 9.2:
            c = pq.mean(axis=0)
            if np.linalg.norm(c[:2] - self.place_xy) > 0.08:
                raise ValueError(f"The block was not placed near the target (centroid xy={c[:2]}).")
            if c[2] > self.table_top_z + 2.0 * self.block_half:
                raise ValueError("The block did not settle on the table at the place location.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the soft block.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="soft_pickplace_franka")
    examples.run(Example(viewer, args), args)
