"""Franka grasps a cable off the table, lifts it, and sweeps it side to side over a soft FEM
block; the swept cable dents and nudges the block. 16 substeps. Pure SCENE + POLICY; all
physics/robot/asset detail comes from :mod:`deformableManipulationTools`, all rendering from
:class:`robolabViz.scenic.ScenicGraspExample`.

Run: python -m examples cable_soft_franka --device cuda:0
  (scenic by default -> outputs/cable_soft_franka/{frames/, simulation.mp4};
   --output-style basic writes outputs/cable_soft_franka.usd instead.)
"""
from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from deformableManipulationTools.helper import install_terminal_log

_terminal_log = install_terminal_log()

import warp as wp

import examples
from robolabViz.scenic import ScenicGraspExample
from deformableManipulationTools import (
    CABLE, FRANKA, SOFT_BLOCK, TABLE, PARTICLE_SOLVER_KWARGS, GraspWindow,
    add_table, add_cable, add_soft_block, build_gripper_proxies, solve_gripper_ik, wp_smoothstep,
)


@wp.kernel
def _set_arm_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home_q: wp.array(dtype=float),
    pickup_q: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
):
    # ARM DOFs only (0..n_arm_dof-1); the fingers are owned by the centralized force-stop GripController.
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    close_start = 2.8
    lift_start = 4.8
    sweep_start = 6.8

    if t < close_start:
        alpha = wp_smoothstep(t / close_start)
        q = (1.0 - alpha) * home_q[i] + alpha * pickup_q[i]
    elif t < lift_start:
        q = pickup_q[i]
    elif t < sweep_start:
        alpha = wp_smoothstep((t - lift_start) / (sweep_start - lift_start))
        q = (1.0 - alpha) * pickup_q[i] + alpha * home_q[i]
    else:
        q = home_q[i]
        phase = 2.0 * 3.141592653589793 * 0.18 * (t - sweep_start)
        ramp = wp_smoothstep((t - sweep_start) / 1.5)
        if i == 0:
            q += 0.55 * ramp * wp.sin(phase)
        elif i == 3:
            q += 0.18 * ramp * wp.sin(phase + 0.35)
        elif i == 5:
            q -= 0.20 * ramp * wp.sin(phase)

    joint_target_q[i] = q


class Example(ScenicGraspExample):
    has_particles = True
    soft_block = SOFT_BLOCK

    def configure(self, args):
        self.table_top_z = TABLE.top_z
        self.soft_start_pos = np.array([0.28, -0.30, self.table_top_z], dtype=np.float32)
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 2048,
                                     "rigid_body_particle_contact_buffer_size": 512, **PARTICLE_SOLVER_KWARGS}
        self.object_pipeline_kwargs = {"soft_contact_margin": SOFT_BLOCK.contact_margin}
        # Force-target grasp on the cable (incompressible → close until the squeeze reaches
        # GRIP.force_target_rigid, then freeze; geometry emerges, no per-object knob). Held to the run's end.
        self.grasp_windows = [GraspWindow(close_start=2.8, close_end=4.6)]

    def plan(self, ik_model, ik_state):
        start = self.tcp_position(ik_state)
        direction = np.array([1.0, 0.05, 0.0], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
        extent = direction * CABLE.segment_length * (CABLE.node_count - 1)
        m = 0.04
        for ax in range(2):
            start[ax] = np.clip(start[ax], TABLE.pos[ax] - TABLE.half[ax] + m,
                                TABLE.pos[ax] + TABLE.half[ax] - m - max(extent[ax], 0.0))
        start[2] = self.table_top_z + CABLE.radius
        self.cable_node_positions = [
            (start + direction * CABLE.segment_length * i
             + normal * CABLE.bow * math.sin(math.pi * i / (CABLE.node_count - 1))).astype(np.float32)
            for i in range(CABLE.node_count)
        ]
        grasp = 0.5 * (self.cable_node_positions[3] + self.cable_node_positions[4])
        grasp[2] = self.table_top_z
        self.pickup_q = solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset, grasp, self.gripper_open)
        self._home_q_wp = wp.array(self.home_q, dtype=wp.float32, device=ik_model.device)
        self._pickup_q_wp = wp.array(self.pickup_q, dtype=wp.float32, device=ik_model.device)

    def build_scene(self, builder, robot_builder):
        self._obj_table_shape = add_table(builder, TABLE)
        add_soft_block(builder, SOFT_BLOCK, self.soft_start_pos)
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape)
        self.cable_body_start = builder.body_count
        self.cable_bodies, _, _ = add_cable(builder, self.cable_node_positions)
        self.cable_body_count = len(self.cable_bodies)

    def set_arm_targets(self, substep):
        wp.launch(_set_arm_targets_kernel, dim=FRANKA.n_arm_dof, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pickup_q_wp,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        body_q = self.object_state_0.body_q.numpy()
        cable_q = body_q[self.cable_body_start : self.cable_body_start + self.cable_body_count]
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(cable_q)):
            raise ValueError("Non-finite body transform detected.")
        particle_q = self.object_state_0.particle_q.numpy()
        if particle_q.size and not np.all(np.isfinite(particle_q)):
            raise ValueError("Non-finite soft-body particle position detected.")
        if particle_q.size and np.min(particle_q[:, 2]) < self.table_top_z - 0.03:
            raise ValueError("The soft body fell through the table.")
        if self.sim_time >= 7.0:
            ee = self.tcp_position(self.robot_state_0)
            d = np.linalg.norm(cable_q[:, :3] - ee[None, :], axis=1)
            if d.min() > 0.05:
                raise ValueError("The cable is no longer grasped by the gripper.")
            if cable_q[d.argmin(), 2] < self.table_top_z + 0.05:
                raise ValueError("The grasped cable node is not lifted above the table.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the cable.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="cable_soft_franka")
    examples.run(Example(viewer, args), args)
