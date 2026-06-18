"""Franka picks a flat metal sheet by its handle, carries it over a soft FEM block,
and presses/drops it so the sheet compresses the block. 16 substeps.

Robot, grip (dynamic finite-mass proxy, physical bounded force, no cap), and shared loop
come from examples.franka_common + examples.grip_coupling; physics parameters from
assets.params. Only the sheet/soft-block objects and the pick-press motion are example-specific.

Run: python -m examples soft_compression_franka --viewer usd --device cuda:0
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


from examples.helper import install_terminal_log

_terminal_log = install_terminal_log()

import warp as wp

import examples
try:
    import newton
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This example requires the Newton Python package. Run it from an environment where `import newton` works."
    ) from exc

from examples.franka_common import (
    GraspExample, build_franka_robot, build_gripper_proxies, build_viz_model,
    find_body, finger_body_indices, make_robot_solver, quat_rotate_xyzw,
    restore_proxy_materials, solve_gripper_ik, wp_smoothstep,
)
from examples.grip_coupling import TwoWayProxyCoupling
from assets.params import FRANKA, GRIP, SOFT_BLOCK_COMPRESS, STEEL_CUBE, TABLE


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
    # Device-side keyframe schedule (CUDA-graph capturable): pick the sheet by its handle, carry
    # it over the block, press/release. The dynamic proxy provides the grip.
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


class Example(GraspExample):
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.has_particles = True

        self.table_top_z = TABLE.top_z
        self.gripper_open = FRANKA.gripper_open
        # Compression sheet: a thin plate with a graspable handle on top (example-specific tool).
        self.sheet_half = np.array([0.09, 0.06, 0.004], dtype=np.float32)
        self.handle_half = np.array([0.016, 0.012, 0.024], dtype=np.float32)
        self.sheet_contact_margin = 0.001
        self.sheet_start_pos = np.array([0.10, -0.55, self.table_top_z + self.sheet_half[2]], dtype=np.float32)
        self.handle_local_pos = np.array([0.0, 0.0, self.sheet_half[2] + self.handle_half[2]], dtype=np.float32)
        self.handle_start_pos = self.sheet_start_pos + self.handle_local_pos
        # ~2x the steel-cube mass, spread over the plate+handle, so the press is firm.
        cube_volume = (2.0 * STEEL_CUBE.half_extent) ** 3
        sheet_volume = float(np.prod(2.0 * self.sheet_half))
        handle_volume = float(np.prod(2.0 * self.handle_half))
        self.sheet_density = 2.0 * STEEL_CUBE.density * cube_volume / (sheet_volume + handle_volume)

        self.soft_start_pos = np.array([0.28, -0.30, self.table_top_z], dtype=np.float32)
        self.soft_drop_offset = np.array([0.5 * SOFT_BLOCK_COMPRESS.dim[0] * SOFT_BLOCK_COMPRESS.cell, 0.0, 0.0],
                                         dtype=np.float32)
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005
        self.grasp_tcp_height = float(self.handle_start_pos[2])
        self.drop_tcp_height = self.table_top_z + 0.19
        # Pad face at the handle half-width (gripper closes along y) + margins - interference.
        self.gripper_closed = (
            self.handle_half[1] + self.sheet_contact_margin + GRIP.proxy_margin - GRIP.grasp_interference
        )
        self.home_q = np.array(FRANKA.home_q, dtype=np.float32)

        device = wp.get_device(args.device) if args.device else None
        robot_builder = build_franka_robot(
            xform=wp.transform((-0.45, -0.45, self.table_top_z), wp.quat_identity()))

        ik_model = robot_builder.finalize(device=device)
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)

        self.ee_body = find_body(list(ik_model.body_label), FRANKA.ee_link_suffix)
        self.ee_offset = np.array(FRANKA.ee_offset, dtype=np.float32)

        pregrasp_pos = self.handle_start_pos.copy()
        pregrasp_pos[2] = self.table_top_z + 0.16
        self.pregrasp_q = solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset, pregrasp_pos, self.gripper_open)
        pickup_pos = self.handle_start_pos.copy()
        pickup_pos[2] = self.grasp_tcp_height
        self.pickup_q = solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset, pickup_pos, self.gripper_open)
        drop_pos = self.soft_start_pos + self.soft_drop_offset
        drop_pos[2] = self.drop_tcp_height
        self.drop_q = solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset, drop_pos, self.gripper_open)

        self.robot_finger_bodies = finger_body_indices(ik_model)

        self.robot_model = robot_builder.finalize(device=ik_model.device)
        self.robot_state_0 = self.robot_model.state()
        self.robot_state_1 = self.robot_model.state()
        self.robot_control = self.robot_model.control()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_0)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_1)
        wp.copy(self.robot_control.joint_target_q, self.robot_model.joint_q)
        robot_contact_max = 2048
        self.robot_model.rigid_contact_max = robot_contact_max
        self.robot_collision_pipeline = newton.CollisionPipeline(
            self.robot_model, reduce_contacts=True, rigid_contact_max=robot_contact_max, broad_phase="nxn")
        self.robot_contacts = self.robot_collision_pipeline.contacts()
        self.robot_solver = make_robot_solver(self.robot_model, robot_contact_max)

        object_builder = self._build_object_builder(robot_builder)
        object_builder.color(balance_colors=False)

        self.object_model = object_builder.finalize(device=ik_model.device)
        self.object_model.soft_contact_ke = SOFT_BLOCK_COMPRESS.soft_contact_ke
        self.object_model.soft_contact_kd = SOFT_BLOCK_COMPRESS.soft_contact_kd
        self.object_model.soft_contact_kf = SOFT_BLOCK_COMPRESS.soft_contact_kf
        self.object_model.soft_contact_mu = SOFT_BLOCK_COMPRESS.soft_contact_mu
        self.object_model.shape_material_ke.fill_(GRIP.proxy_ke)
        self.object_model.shape_material_kd.fill_(GRIP.object_contact_kd)
        self.object_model.shape_material_mu.fill_(0.8)
        restore_proxy_materials(self.object_model, self.gripper_proxy_shapes)
        self._restore_sheet_materials()
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        self.object_collision_pipeline = newton.CollisionPipeline(
            self.object_model, contact_matching="latest", soft_contact_margin=SOFT_BLOCK_COMPRESS.contact_margin)
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)

        self.object_solver = newton.solvers.SolverVBD(
            self.object_model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=2048,  # headroom for the wrench harvest
            rigid_body_particle_contact_buffer_size=4096,  # the flat sheet face → many block contacts
            rigid_contact_stick_motion_eps=0.0,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )

        self._t_frame = wp.zeros(1, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = wp.array(self.home_q, dtype=wp.float32, device=ik_model.device)
        self._pregrasp_q_wp = wp.array(self.pregrasp_q, dtype=wp.float32, device=ik_model.device)
        self._pickup_q_wp = wp.array(self.pickup_q, dtype=wp.float32, device=ik_model.device)
        self._drop_q_wp = wp.array(self.drop_q, dtype=wp.float32, device=ik_model.device)
        self.coupling = TwoWayProxyCoupling(
            self.robot_model, self.object_model, self.object_solver, self.object_contacts,
            self.object_state_0, self.robot_finger_bodies, self.gripper_proxy_bodies,
            self.ee_body, self.sim_dt)
        self.graph = None
        self._frames_simulated = 0
        self._capture_enabled = (
            wp.get_device(str(ik_model.device)).is_cuda
            and self.sim_substeps % 2 == 0
            and not os.environ.get("CABLE_NO_CAPTURE")
        )

        self._sync_gripper_proxies()

        self.viz_model, self.viz_object_body_start = build_viz_model(robot_builder, object_builder, ik_model.device)
        self.viz_state = self.viz_model.state()
        self._sync_viz_state()

        self.viewer.set_model(self.viz_model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.85, 0.30, 0.55), pitch=-22.0, yaw=-130.0)

    def _tcp_position(self, state) -> np.ndarray:
        body_q = state.body_q.numpy()[self.ee_body]
        return body_q[:3] + quat_rotate_xyzw(body_q[3:7], self.ee_offset)

    def _build_object_builder(self, robot_builder: newton.ModelBuilder) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = GRIP.proxy_ke
        builder.default_shape_cfg.kd = GRIP.object_contact_kd
        builder.default_shape_cfg.mu = 0.8

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=TABLE.object_ke, kd=TABLE.object_kd, mu=TABLE.object_mu)
        self._obj_table_shape = builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(*TABLE.pos), wp.quat_identity()),
            hx=float(TABLE.half[0]), hy=float(TABLE.half[1]), hz=float(TABLE.half[2]),
            cfg=table_cfg, color=wp.vec3(*TABLE.color), label="table")

        soft = SOFT_BLOCK_COMPRESS
        builder.default_particle_radius = soft.particle_radius
        builder.particle_max_velocity = 50.0
        dim_x, dim_y, dim_z = soft.dim
        builder.add_soft_grid(
            pos=wp.vec3(
                float(self.soft_start_pos[0] - 0.5 * dim_x * soft.cell),
                float(self.soft_start_pos[1] - 0.5 * dim_y * soft.cell),
                float(self.soft_start_pos[2]),
            ),
            rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x, dim_y=dim_y, dim_z=dim_z,
            cell_x=soft.cell, cell_y=soft.cell, cell_z=soft.cell,
            density=soft.density, k_mu=soft.k_mu, k_lambda=soft.k_lambda, k_damp=soft.k_damp,
        )

        # Dynamic finger proxies (the grip contact bridge).
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape, gap=GRIP.proxy_margin * 8)

        # The sheet: thin plate + handle (one rigid body).
        self.sheet_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.sheet_start_pos), wp.quat_identity()), label="compression_sheet")
        sheet_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.sheet_density, margin=self.sheet_contact_margin,
            ke=GRIP.proxy_ke, kd=GRIP.object_contact_kd, mu=0.8)
        self.sheet_shapes = [
            builder.add_shape_box(
                body=self.sheet_body, hx=float(self.sheet_half[0]), hy=float(self.sheet_half[1]),
                hz=float(self.sheet_half[2]), cfg=sheet_cfg, color=wp.vec3(0.62, 0.65, 0.68), label="metal_sheet"),
            builder.add_shape_box(
                body=self.sheet_body, xform=wp.transform(wp.vec3(*self.handle_local_pos), wp.quat_identity()),
                hx=float(self.handle_half[0]), hy=float(self.handle_half[1]), hz=float(self.handle_half[2]),
                cfg=sheet_cfg, color=wp.vec3(0.40, 0.42, 0.45), label="grasp_handle"),
        ]
        return builder

    def _restore_sheet_materials(self) -> None:
        # Firm sheet↔block contact for a clean compression (match the soft-contact ke).
        ke = self.object_model.shape_material_ke.numpy()
        kd = self.object_model.shape_material_kd.numpy()
        for shape in self.sheet_shapes:
            ke[shape] = SOFT_BLOCK_COMPRESS.soft_contact_ke
            kd[shape] = SOFT_BLOCK_COMPRESS.soft_contact_kd
        self.object_model.shape_material_ke.assign(ke)
        self.object_model.shape_material_kd.assign(kd)

    def _set_robot_targets(self, substep: int) -> None:
        wp.launch(
            _set_robot_targets_kernel, dim=9,
            inputs=[self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp,
                    self._pickup_q_wp, self._drop_q_wp, self.gripper_open, self.gripper_closed],
            outputs=[self.robot_control.joint_target_q],
            device=self.robot_control.joint_target_q.device,
        )

    def test_final(self) -> None:
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
        parser.set_defaults(output_path=str(Path("outputs") / "soft_compression_franka.usd"), num_frames=720)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the objects.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="soft_compression_franka")
    examples.run(Example(viewer, args), args)
