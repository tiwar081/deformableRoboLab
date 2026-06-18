"""Franka picks a small soft FEM block off the table and places it at a target
location. 16 substeps.

Robot, grip (dynamic finite-mass proxy, physical bounded force, no cap), and shared loop
come from examples.franka_common + examples.grip_coupling; physics parameters from
assets.params. The block is FEM particles, so the proxies carry particle collision and the
coupling harvests the proxy↔particle reaction (recomputed from public soft-contact geometry).

Run: python -m examples soft_pickplace_franka --viewer usd --device cuda:0
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
from assets.params import FRANKA, GRIP, SOFT_BLOCK_PICK, TABLE


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
    # Pick-and-place keyframe schedule (CUDA-graph capturable). The dynamic proxy provides the
    # grip; the gripper just position-controls open → closed → open.
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
        soft = SOFT_BLOCK_PICK
        self.block_half = 0.5 * soft.dim[0] * soft.cell
        self.pick_xy = np.array([0.10, -0.50], dtype=np.float32)
        self.place_xy = np.array([0.34, -0.28], dtype=np.float32)
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005
        self.grasp_tcp_height = self.table_top_z + self.block_half
        self.lift_height = self.table_top_z + 0.16
        # Squeeze the compliant block: pads ~5 mm inside the block half-width. The dynamic proxy's
        # particle contact (soft harvest) provides the physical grip; mu=0.8 holds it during lift.
        self.gripper_closed = self.block_half - 0.005
        self.home_q = np.array(FRANKA.home_q, dtype=np.float32)

        device = wp.get_device(args.device) if args.device else None
        robot_builder = build_franka_robot(
            xform=wp.transform((-0.45, -0.45, self.table_top_z), wp.quat_identity()))

        ik_model = robot_builder.finalize(device=device)
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)
        self.ee_body = find_body(list(ik_model.body_label), FRANKA.ee_link_suffix)
        self.ee_offset = np.array(FRANKA.ee_offset, dtype=np.float32)

        def ik_at(xy, z):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array([xy[0], xy[1], z], dtype=np.float32), self.gripper_open)

        self.pregrasp_q = ik_at(self.pick_xy, self.table_top_z + 0.12)
        self.pickup_q = ik_at(self.pick_xy, self.grasp_tcp_height)
        self.lift_q = ik_at(self.pick_xy, self.lift_height)
        self.place_high_q = ik_at(self.place_xy, self.lift_height)
        self.place_q = ik_at(self.place_xy, self.grasp_tcp_height)

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
        self.object_model.soft_contact_ke = soft.soft_contact_ke
        self.object_model.soft_contact_kd = soft.soft_contact_kd
        self.object_model.soft_contact_kf = soft.soft_contact_kf
        self.object_model.soft_contact_mu = soft.soft_contact_mu
        self.object_model.shape_material_ke.fill_(GRIP.proxy_ke)
        self.object_model.shape_material_kd.fill_(GRIP.object_contact_kd)
        self.object_model.shape_material_mu.fill_(0.8)
        restore_proxy_materials(self.object_model, self.gripper_proxy_shapes)
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        self.object_collision_pipeline = newton.CollisionPipeline(
            self.object_model, contact_matching="latest", soft_contact_margin=soft.contact_margin)
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)

        self.object_solver = newton.solvers.SolverVBD(
            self.object_model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=2048,  # headroom for the wrench harvest
            rigid_body_particle_contact_buffer_size=4096,  # the pads grip many block particles
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
        self._lift_q_wp = wp.array(self.lift_q, dtype=wp.float32, device=ik_model.device)
        self._place_high_q_wp = wp.array(self.place_high_q, dtype=wp.float32, device=ik_model.device)
        self._place_q_wp = wp.array(self.place_q, dtype=wp.float32, device=ik_model.device)
        # soft_contact_ke enables the proxy↔particle reaction harvest (the soft block's squeeze).
        self.coupling = TwoWayProxyCoupling(
            self.robot_model, self.object_model, self.object_solver, self.object_contacts,
            self.object_state_0, self.robot_finger_bodies, self.gripper_proxy_bodies,
            self.ee_body, self.sim_dt, soft_contact_ke=soft.soft_contact_ke)
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

        soft = SOFT_BLOCK_PICK
        builder.default_particle_radius = soft.particle_radius
        builder.particle_max_velocity = 50.0
        dx, dy, dz = soft.dim
        builder.add_soft_grid(
            pos=wp.vec3(float(self.pick_xy[0] - 0.5 * dx * soft.cell),
                        float(self.pick_xy[1] - 0.5 * dy * soft.cell),
                        float(self.table_top_z)),
            rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dx, dim_y=dy, dim_z=dz,
            cell_x=soft.cell, cell_y=soft.cell, cell_z=soft.cell,
            density=soft.density, k_mu=soft.k_mu, k_lambda=soft.k_lambda, k_damp=soft.k_damp,
        )

        # Dynamic finger proxies WITH particle collision so the pads grip the soft block.
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape,
            gap=GRIP.proxy_margin * 8, has_particle_collision=True)
        return builder

    def _set_robot_targets(self, substep: int) -> None:
        wp.launch(
            _set_robot_targets_kernel, dim=9,
            inputs=[self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp,
                    self._pickup_q_wp, self._lift_q_wp, self._place_high_q_wp, self._place_q_wp,
                    self.gripper_open, self.gripper_closed],
            outputs=[self.robot_control.joint_target_q],
            device=self.robot_control.joint_target_q.device,
        )

    def test_final(self):
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
        parser.set_defaults(output_path=str(Path("outputs") / "soft_pickplace_franka.usd"), num_frames=720)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the soft block.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="soft_pickplace_franka")
    examples.run(Example(viewer, args), args)
