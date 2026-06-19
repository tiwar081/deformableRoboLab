"""Franka picks a flat metal sheet by its handle, carries it over a soft FEM block, and
presses/drops it so the sheet compresses the block. 16 substeps. Pure SCENE + POLICY; all
physics/robot/asset detail comes from :mod:`deformableManipulationTools`, all rendering from
:class:`robolabViz.scenic.ScenicGraspExample`.

Run: python -m examples soft_compression_franka --device cuda:0
  (scenic by default -> outputs/soft_compression_franka/{frames/, simulation.mp4};
   --output-style basic writes outputs/soft_compression_franka.usd instead.)
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from deformableManipulationTools.helper import install_terminal_log

_terminal_log = install_terminal_log()

import warp as wp

import newton

import examples
from robolabViz.scenic import ScenicGraspExample
from deformableManipulationTools import (
    GRIP, SOFT_BLOCK_COMPRESS, STEEL_CUBE, TABLE, PARTICLE_SOLVER_KWARGS,
    add_table, add_soft_block, build_gripper_proxies, solve_gripper_ik, wp_smoothstep,
)


@wp.kernel
def _set_robot_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home_q: wp.array(dtype=float),
    pregrasp_q: wp.array(dtype=float),
    pickup_q: wp.array(dtype=float),
    drop_q: wp.array(dtype=float),
    gripper_open: float,
    gripper_closed: float,
    joint_target_q: wp.array(dtype=float),
):
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    descend_start = 2.2
    close_start = 3.2
    hold_start = 4.4
    carry_start = 5.0
    settle_start = 7.0
    release_start = 8.0
    retreat_start = 8.6

    if t < descend_start:
        alpha = wp_smoothstep(t / descend_start)
        q = (1.0 - alpha) * home_q[i] + alpha * pregrasp_q[i]
    elif t < close_start:
        alpha = wp_smoothstep((t - descend_start) / (close_start - descend_start))
        q = (1.0 - alpha) * pregrasp_q[i] + alpha * pickup_q[i]
    elif t < carry_start:
        q = pickup_q[i]
    elif t < settle_start:
        alpha = wp_smoothstep((t - carry_start) / (settle_start - carry_start))
        q = (1.0 - alpha) * pickup_q[i] + alpha * drop_q[i]
    elif t < retreat_start:
        q = drop_q[i]
    else:
        alpha = wp_smoothstep((t - retreat_start) / 2.0)
        q = (1.0 - alpha) * drop_q[i] + alpha * home_q[i]

    if i >= 7:
        if t < close_start:
            q = gripper_open
        elif t < hold_start:
            alpha = wp_smoothstep((t - close_start) / (hold_start - close_start))
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
    soft_block = SOFT_BLOCK_COMPRESS

    def configure(self, args):
        self.table_top_z = TABLE.top_z
        # A thin plate with a graspable handle on top (an example-specific tool).
        self.sheet_half = np.array([0.09, 0.06, 0.004], dtype=np.float32)
        self.handle_half = np.array([0.016, 0.012, 0.024], dtype=np.float32)
        self.sheet_contact_margin = 0.001
        self.sheet_start_pos = np.array([0.10, -0.55, self.table_top_z + self.sheet_half[2]], dtype=np.float32)
        self.handle_local_pos = np.array([0.0, 0.0, self.sheet_half[2] + self.handle_half[2]], dtype=np.float32)
        self.handle_start_pos = self.sheet_start_pos + self.handle_local_pos
        # ~2x the steel-cube mass, spread over the plate+handle, so the press is firm.
        cube_vol = (2.0 * STEEL_CUBE.half_extent) ** 3
        sheet_vol = float(np.prod(2.0 * self.sheet_half))
        handle_vol = float(np.prod(2.0 * self.handle_half))
        self.sheet_density = 2.0 * STEEL_CUBE.density * cube_vol / (sheet_vol + handle_vol)
        self.soft_start_pos = np.array([0.28, -0.30, self.table_top_z], dtype=np.float32)
        self.soft_drop_offset = np.array([0.5 * SOFT_BLOCK_COMPRESS.dim[0] * SOFT_BLOCK_COMPRESS.cell, 0.0, 0.0],
                                         dtype=np.float32)
        self.drop_tcp_height = self.table_top_z + 0.19
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 2048,
                                     "rigid_body_particle_contact_buffer_size": 4096, **PARTICLE_SOLVER_KWARGS}
        self.object_pipeline_kwargs = {"soft_contact_margin": SOFT_BLOCK_COMPRESS.contact_margin}

    def plan(self, ik_model, ik_state):
        def ik_at(pos):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array(pos, dtype=np.float32), self.gripper_open)

        pre = self.handle_start_pos.copy(); pre[2] = self.table_top_z + 0.16
        pick = self.handle_start_pos.copy(); pick[2] = float(self.handle_start_pos[2])
        drop = self.soft_start_pos + self.soft_drop_offset; drop[2] = self.drop_tcp_height
        self.pregrasp_q = ik_at(pre)
        self.pickup_q = ik_at(pick)
        self.drop_q = ik_at(drop)
        self.gripper_closed = (self.handle_half[1] + self.sheet_contact_margin
                               + GRIP.proxy_margin - GRIP.grasp_interference)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = kf(self.home_q)
        self._pregrasp_q_wp = kf(self.pregrasp_q)
        self._pickup_q_wp = kf(self.pickup_q)
        self._drop_q_wp = kf(self.drop_q)

    def build_scene(self, builder, robot_builder):
        self._obj_table_shape = add_table(builder, TABLE)
        add_soft_block(builder, SOFT_BLOCK_COMPRESS, self.soft_start_pos)
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape, gap=GRIP.proxy_margin * 8)
        # The sheet: thin plate + handle (one rigid body, two boxes).
        self.sheet_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.sheet_start_pos), wp.quat_identity()), label="compression_sheet")
        sheet_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.sheet_density, margin=self.sheet_contact_margin,
            ke=GRIP.proxy_ke, kd=GRIP.object_contact_kd, mu=0.8)
        self.sheet_shapes = [
            builder.add_shape_box(body=self.sheet_body, hx=float(self.sheet_half[0]), hy=float(self.sheet_half[1]),
                                  hz=float(self.sheet_half[2]), cfg=sheet_cfg,
                                  color=wp.vec3(0.62, 0.65, 0.68), label="metal_sheet"),
            builder.add_shape_box(body=self.sheet_body, xform=wp.transform(wp.vec3(*self.handle_local_pos), wp.quat_identity()),
                                  hx=float(self.handle_half[0]), hy=float(self.handle_half[1]), hz=float(self.handle_half[2]),
                                  cfg=sheet_cfg, color=wp.vec3(0.40, 0.42, 0.45), label="grasp_handle"),
        ]
        # Firm sheet<->block contact for a clean compression (match the soft-contact ke).
        self.material_overrides.append(
            {"shapes": self.sheet_shapes, "ke": SOFT_BLOCK_COMPRESS.soft_contact_ke,
             "kd": SOFT_BLOCK_COMPRESS.soft_contact_kd})

    def set_robot_targets(self, substep):
        wp.launch(_set_robot_targets_kernel, dim=9, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp,
            self._pickup_q_wp, self._drop_q_wp, self.gripper_open, self.gripper_closed,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        body_q = self.object_state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Non-finite body transform detected.")
        particle_q = self.object_state_0.particle_q.numpy()
        if particle_q.size and not np.all(np.isfinite(particle_q)):
            raise ValueError("Non-finite soft-body particle position detected.")
        if particle_q.size and np.min(particle_q[:, 2]) < self.table_top_z - 0.03:
            raise ValueError("The soft body fell through the table.")
        sheet = body_q[self.sheet_body, :3]
        if sheet[2] < self.table_top_z - float(self.sheet_half[2]):
            raise ValueError("The metal sheet fell through the table.")
        if self.sim_time >= 10.0:
            if sheet[2] > self.table_top_z + 0.15:
                raise ValueError("The metal sheet did not drop after the gripper opened.")
            if np.linalg.norm(sheet[:2] - self.soft_start_pos[:2]) > 0.25:
                raise ValueError("The metal sheet did not land near the soft block.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the objects.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="soft_compression_franka")
    examples.run(Example(viewer, args), args)
