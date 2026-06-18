"""Franka grasps a cable off the table, lifts it, and sweeps it side to side past
a rigid cube. (Same cable motion as cable_soft_franka, with a rigid cube instead of
a soft block.) 8 substeps.

Robot, grip (dynamic finite-mass proxy, physical bounded force, no cap), and shared loop
come from examples.franka_common + examples.grip_coupling; physics parameters from
assets.params. Only the cable/cube objects and the sweep motion are example-specific.

Run: python -m examples cable_rigidCube_franka --viewer usd --device cuda:0
"""
from __future__ import annotations

import math
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
from assets.params import CABLE, FRANKA, GRIP, RIGID_CUBE, TABLE


@wp.kernel
def _set_robot_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home_q: wp.array(dtype=float),
    pickup_q: wp.array(dtype=float),
    gripper_open: float,
    gripper_closed: float,
    joint_target_q: wp.array(dtype=float),
):
    # Device-side mirror of the keyframe schedule (CUDA-graph capturable).
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    close_start = 2.8
    hold_start = 4.0
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

    if i >= 7:
        if t < close_start:
            q = gripper_open
        elif t < hold_start:
            alpha = wp_smoothstep((t - close_start) / (hold_start - close_start))
            q = (1.0 - alpha) * gripper_open + alpha * gripper_closed
        else:
            q = gripper_closed

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
        self.has_particles = False

        self.table_top_z = TABLE.top_z
        self.gripper_open = FRANKA.gripper_open
        self.cube_half = RIGID_CUBE.half_extent
        self.cube_start_pos = np.array([0.28, -0.30, self.table_top_z + self.cube_half], dtype=np.float32)
        self.cable_direction = np.array([1.0, 0.05, 0.0], dtype=np.float32)
        self.cable_direction /= np.linalg.norm(self.cable_direction)
        self.grasp_tcp_height = self.table_top_z
        self.gripper_closed = (
            CABLE.radius + CABLE.contact_margin + GRIP.proxy_margin - GRIP.grasp_interference
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
        initial_ee_pos = self._tcp_position(ik_state)
        self.cable_start_pos = initial_ee_pos.copy()
        cable_extent = self.cable_direction * CABLE.segment_length * (CABLE.node_count - 1)
        table_margin = 0.04
        for axis in range(2):
            self.cable_start_pos[axis] = np.clip(
                self.cable_start_pos[axis],
                float(TABLE.pos[axis] - TABLE.half[axis] + table_margin),
                float(TABLE.pos[axis] + TABLE.half[axis] - table_margin - max(cable_extent[axis], 0.0)),
            )
        self.cable_start_pos[2] = self.table_top_z + CABLE.radius
        self.cable_node_positions = self._cable_layout_positions()
        grasp_pos = 0.5 * (self.cable_node_positions[3] + self.cable_node_positions[4])
        grasp_pos[2] = self.grasp_tcp_height
        self.pickup_q = solve_gripper_ik(
            ik_model, ik_state, self.ee_body, self.ee_offset, grasp_pos, self.gripper_open)

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
        self.object_model.shape_material_ke.fill_(GRIP.proxy_ke)
        self.object_model.shape_material_kd.fill_(GRIP.object_contact_kd)
        self.object_model.shape_material_mu.fill_(0.8)
        self._restore_cable_materials()
        restore_proxy_materials(self.object_model, self.gripper_proxy_shapes)
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        self.object_collision_pipeline = newton.CollisionPipeline(self.object_model, contact_matching="latest")
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)

        self.object_solver = newton.solvers.SolverVBD(
            self.object_model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=2048,  # headroom so wrench-harvest buffers don't grow mid-capture
            # NVIDIA default-hard contacts (no alpha=0 / cross-step history): bounds the grip force
            # so the dynamic proxy stays stable; the cable is held by honest squeeze friction.
            rigid_contact_stick_motion_eps=0.0,
        )

        self._t_frame = wp.zeros(1, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = wp.array(self.home_q, dtype=wp.float32, device=ik_model.device)
        self._pickup_q_wp = wp.array(self.pickup_q, dtype=wp.float32, device=ik_model.device)
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

    def _cable_layout_positions(self) -> list[np.ndarray]:
        normal = np.array([-self.cable_direction[1], self.cable_direction[0], 0.0], dtype=np.float32)
        positions = []
        for i in range(CABLE.node_count):
            s = i / (CABLE.node_count - 1)
            p = self.cable_start_pos + self.cable_direction * CABLE.segment_length * i
            positions.append((p + normal * CABLE.bow * math.sin(math.pi * s)).astype(np.float32))
        return positions

    def _build_object_builder(self, robot_builder: newton.ModelBuilder) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = CABLE.contact_ke
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.mu = 0.8

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=TABLE.object_ke, kd=TABLE.object_kd, mu=TABLE.object_mu)
        self._obj_table_shape = builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(*TABLE.pos), wp.quat_identity()),
            hx=float(TABLE.half[0]), hy=float(TABLE.half[1]), hz=float(TABLE.half[2]),
            cfg=table_cfg, color=wp.vec3(*TABLE.color), label="table")

        self.cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.cube_start_pos), wp.quat_identity()), mass=0.2, label="cube")
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=RIGID_CUBE.density, margin=RIGID_CUBE.contact_margin,
            ke=RIGID_CUBE.contact_ke, kd=RIGID_CUBE.contact_kd, mu=RIGID_CUBE.contact_mu)
        builder.add_shape_box(
            body=self.cube_body, hx=self.cube_half, hy=self.cube_half, hz=self.cube_half,
            cfg=cube_cfg, color=wp.vec3(0.18, 0.42, 0.95), label="cube_shape")

        # Dynamic finger proxies (the grip contact bridge).
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape, gap=CABLE.radius)

        positions = [wp.vec3(*p) for p in self.cable_node_positions]
        self.cable_body_start = builder.body_count
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=CABLE.density, margin=CABLE.contact_margin,
            ke=CABLE.contact_ke, kd=20.0, mu=CABLE.friction)
        self.cable_bodies, self.cable_joints = builder.add_rod(
            positions=positions, radius=CABLE.radius, cfg=cable_cfg,
            stretch_stiffness=CABLE.stretch_stiffness, stretch_damping=CABLE.stretch_damping,
            bend_stiffness=CABLE.bend_stiffness, bend_damping=CABLE.bend_damping,
            label="vbd_cable", wrap_in_articulation=True, body_frame_origin="start",
        )
        self.cable_body_count = len(self.cable_bodies)
        return builder

    def _restore_cable_materials(self) -> None:
        body_set = set(self.cable_bodies)
        mu = self.object_model.shape_material_mu.numpy()
        ke = self.object_model.shape_material_ke.numpy()
        kd = self.object_model.shape_material_kd.numpy()
        for shape, body in enumerate(self.object_model.shape_body.numpy()):
            if int(body) in body_set:
                mu[shape] = CABLE.friction
                ke[shape] = CABLE.contact_ke
                kd[shape] = CABLE.contact_kd
        self.object_model.shape_material_mu.assign(mu)
        self.object_model.shape_material_ke.assign(ke)
        self.object_model.shape_material_kd.assign(kd)

    def _set_robot_targets(self, substep: int) -> None:
        wp.launch(
            _set_robot_targets_kernel, dim=9,
            inputs=[self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pickup_q_wp,
                    self.gripper_open, self.gripper_closed],
            outputs=[self.robot_control.joint_target_q],
            device=self.robot_control.joint_target_q.device,
        )

    def test_final(self) -> None:
        body_q = self.object_state_0.body_q.numpy()
        cable_q = body_q[self.cable_body_start : self.cable_body_start + self.cable_body_count]
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(cable_q)):
            raise ValueError("Non-finite body transform detected.")

        cube = body_q[self.cube_body, :3]
        if cube[2] - self.cube_half > self.table_top_z + 0.03:
            raise ValueError("The cube is still floating above the table.")
        if cube[2] < self.table_top_z:
            raise ValueError("The cube fell through the table.")

        if self.sim_time >= 7.0:
            ee = self._tcp_position(self.robot_state_0)
            d = np.linalg.norm(cable_q[:, :3] - ee[None, :], axis=1)
            if d.min() > 0.05:
                raise ValueError("The cable is no longer grasped by the gripper.")
            if cable_q[d.argmin(), 2] < self.table_top_z + 0.05:
                raise ValueError("The grasped cable node is not lifted above the table.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "cable_rigidCube_franka.usd"), num_frames=720)
        parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the cable.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="cable_rigidCube_franka")
    examples.run(Example(viewer, args), args)
