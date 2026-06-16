"""Franka picks up a soft FEM block, carries it across the table, and places it.

This is the soft-body counterpart to ``rigidCube_soft_franka``'s force-limited
gripper. The gripper is force-controlled by a latch: it creeps closed until the
contact reaction reaches a threshold, then holds that width. For a soft body the
reaction grows gradually with compression, so the gripper keeps squeezing until
the threshold and stops — squeezing the block by a bounded amount rather than
crushing it. The reaction is read from the PUBLIC soft-contact geometry
(``soft_contact_*`` + ``particle_q`` + ``soft_contact_ke``) since Newton exposes
no force readback for body-particle contacts; it is the same penalty law VBD uses
internally. Nothing in ``_external`` is modified or imported privately.

Run: python -m examples soft_pickplace_franka --viewer usd --device cuda:0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


class _TerminalTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _install_terminal_log():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = (output_dir / "terminal").open("w", buffering=1)
    sys.stdout = _TerminalTee(sys.__stdout__, log)
    sys.stderr = _TerminalTee(sys.__stderr__, log)
    return log


_terminal_log = _install_terminal_log()

import warp as wp

import examples
try:
    import newton
    import newton.ik as ik
    import newton.utils
    from newton import JointTargetMode
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This example requires the Newton Python package. Run it from an environment where `import newton` works."
    ) from exc


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _find_body(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.endswith(suffix):
            return i
    raise ValueError(f"Could not find body ending with {suffix!r}.")


def _quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.func
def _wp_smoothstep(x: float) -> float:
    x = wp.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


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
    grip_target: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
):
    # Pick-and-place keyframe schedule (arm); the gripper DOFs (i>=7) follow the
    # force-triggered latch target. Device-resident for CUDA-graph capture.
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
        q = _blend(home_q, pregrasp_q, _wp_smoothstep(t / descend_start), i)
    elif t < close_start:
        q = _blend(pregrasp_q, pickup_q, _wp_smoothstep((t - descend_start) / (close_start - descend_start)), i)
    elif t < lift_start:
        q = pickup_q[i]
    elif t < over_start:
        q = _blend(pickup_q, lift_q, _wp_smoothstep((t - lift_start) / (over_start - lift_start)), i)
    elif t < lower_start:
        q = _blend(lift_q, place_high_q, _wp_smoothstep((t - over_start) / (lower_start - over_start)), i)
    elif t < hold_start:
        q = _blend(place_high_q, place_q, _wp_smoothstep((t - lower_start) / (hold_start - lower_start)), i)
    elif t < retreat_start:
        q = place_q[i]
    elif t < home_start:
        q = _blend(place_q, place_high_q, _wp_smoothstep((t - retreat_start) / (home_start - retreat_start)), i)
    else:
        q = _blend(place_high_q, home_q, _wp_smoothstep((t - home_start) / 2.0), i)

    if i >= 7:
        q = grip_target[0]

    joint_target_q[i] = q


@wp.kernel
def _update_grip_target_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    reaction: wp.array(dtype=float),
    threshold: float,
    close_rate: float,
    gripper_open: float,
    grip_target: wp.array(dtype=float),
    latched: wp.array(dtype=wp.int32),
):
    # Creep closed until the squeeze reaction reaches the threshold, then latch and
    # hold. Soft contact ramps gradually, so the latch fires at a bounded squeeze.
    t = t_frame[0] + float(substep) * sim_dt
    close_start = 3.2
    release_start = 8.0

    if t < close_start:
        grip_target[0] = gripper_open
        latched[0] = 0
    elif t < release_start:
        if latched[0] == 0:
            if reaction[0] >= threshold or reaction[1] >= threshold:
                latched[0] = 1
            else:
                grip_target[0] = wp.max(grip_target[0] - close_rate * sim_dt, 0.0)
    else:
        grip_target[0] = gripper_open
        latched[0] = 0


@wp.kernel
def _sync_gripper_proxies_kernel(
    robot_body_q: wp.array(dtype=wp.transform),
    finger_bodies: wp.array(dtype=wp.int32),
    proxy_bodies: wp.array(dtype=wp.int32),
    object_body_q_0: wp.array(dtype=wp.transform),
    object_body_q_1: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    tf = robot_body_q[finger_bodies[i]]
    object_body_q_0[proxy_bodies[i]] = tf
    object_body_q_1[proxy_bodies[i]] = tf


@wp.kernel
def _soft_grip_reaction_kernel(
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=wp.int32),
    soft_contact_shape: wp.array(dtype=wp.int32),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    object_body_q: wp.array(dtype=wp.transform),
    soft_contact_ke: float,
    left_proxy: int,
    right_proxy: int,
    grip_reaction: wp.array(dtype=float),
):
    # Sum the body-particle penalty force (ke * penetration) the soft block exerts
    # on each gripper proxy. Newton exposes no soft-contact force readback, so we
    # recompute it from the PUBLIC soft-contact geometry using VBD's own penalty law.
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    pid = soft_contact_particle[i]
    shape = soft_contact_shape[i]
    if pid < 0 or shape < 0:
        return
    body = shape_body[shape]
    if body != left_proxy and body != right_proxy:
        return

    bx = wp.transform_point(object_body_q[body], soft_contact_body_pos[i])
    n = soft_contact_normal[i]
    pen = -(wp.dot(n, particle_q[pid] - bx) - particle_radius[pid])
    if pen <= 0.0:
        return
    fmag = soft_contact_ke * pen
    if body == left_proxy:
        wp.atomic_add(grip_reaction, 0, fmag)
    else:
        wp.atomic_add(grip_reaction, 1, fmag)


@wp.kernel
def _blend_grip_kernel(raw: wp.array(dtype=float), beta: float, filtered: wp.array(dtype=float)):
    i = wp.tid()
    filtered[i] = beta * filtered[i] + (1.0 - beta) * raw[i]


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.table_pos = wp.vec3(0.12, -0.45, 0.035)
        self.table_half = wp.vec3(0.45, 0.35, 0.035)
        self.table_top_z = float(self.table_pos[2] + self.table_half[2])
        self.gripper_proxy_margin = 0.001
        self.gripper_proxy_gap = 0.008
        self.gripper_open = 0.04

        # Small graspable soft block (~33 mm across, fits the 40 mm gripper).
        self.soft_grid_dims = (3, 3, 3)
        self.soft_grid_cell = 0.011
        self.block_half = 0.5 * self.soft_grid_dims[0] * self.soft_grid_cell
        self.pick_xy = np.array([0.10, -0.50], dtype=np.float32)
        self.place_xy = np.array([0.34, -0.28], dtype=np.float32)
        self.soft_k_mu = 5.0e2  # 4x softened (Newton 1.4 re-tune): more visible compression
        self.soft_k_lambda = 2.5e3
        self.soft_density = 200.0
        self.soft_k_damp = 10.0
        self.soft_particle_radius = 0.0035
        self.soft_body_contact_margin = 0.01
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005
        self.soft_contact_ke = 1.0e5
        self.soft_contact_kd = 1.0e-4
        self.soft_contact_kf = 1.0e3
        self.soft_contact_mu = 0.8  # high friction so the gripper can lift the block

        self.grasp_tcp_height = self.table_top_z + self.block_half
        self.lift_height = self.table_top_z + 0.16

        # Force-controlled gripper latch (see module docstring).
        self.grip_force_threshold = float(getattr(args, "grip_threshold", 8.0))
        self.grip_close_rate = 0.02
        self.grip_filter = 0.5

        self.home_q = np.array(
            [-0.0036802115, 0.023901723, 0.003680411, -2.3683236, -0.00012918962,
             2.3922248, 0.785492, self.gripper_open, self.gripper_open],
            dtype=np.float32,
        )
        robot_builder = self._build_robot_builder()

        ik_model = robot_builder.finalize(device=self._device_from_args(args))
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)
        self.ee_body = _find_body(list(ik_model.body_label), "fr3_link7")
        self.ee_offset = np.array([0.0, 0.0, 0.22], dtype=np.float32)

        def ik_at(xy, z):
            return self._solve_gripper_ik(ik_model, ik_state, np.array([xy[0], xy[1], z], dtype=np.float32))

        self.pregrasp_q = ik_at(self.pick_xy, self.table_top_z + 0.12)
        self.pickup_q = ik_at(self.pick_xy, self.grasp_tcp_height)
        self.lift_q = ik_at(self.pick_xy, self.lift_height)
        self.place_high_q = ik_at(self.place_xy, self.lift_height)
        self.place_q = ik_at(self.place_xy, self.grasp_tcp_height)

        self.robot_finger_bodies = [
            _find_body(list(ik_model.body_label), "fr3_leftfinger"),
            _find_body(list(ik_model.body_label), "fr3_rightfinger"),
        ]

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
            self.robot_model, reduce_contacts=True, rigid_contact_max=robot_contact_max, broad_phase="nxn"
        )
        self.robot_contacts = self.robot_collision_pipeline.contacts()
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.robot_model, solver="newton", integrator="implicitfast", iterations=15, ls_iterations=100,
            nconmax=robot_contact_max, njmax=robot_contact_max * 2, cone="elliptic", impratio=50.0,
            use_mujoco_contacts=False,
        )

        object_builder = self._build_object_builder(robot_builder)
        object_builder.color(balance_colors=False)
        self.object_model = object_builder.finalize(device=ik_model.device)
        self.object_model.soft_contact_ke = self.soft_contact_ke
        self.object_model.soft_contact_kd = self.soft_contact_kd
        self.object_model.soft_contact_kf = self.soft_contact_kf
        self.object_model.soft_contact_mu = self.soft_contact_mu
        self.object_model.shape_material_ke.fill_(5.0e4)
        self.object_model.shape_material_kd.fill_(1.0e2)
        self.object_model.shape_material_mu.fill_(0.8)
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        self.object_collision_pipeline = newton.CollisionPipeline(
            self.object_model, contact_matching="latest", soft_contact_margin=self.soft_body_contact_margin
        )
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)

        self.object_solver = newton.solvers.SolverVBD(
            self.object_model, iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=4096,
            rigid_contact_history=True, rigid_contact_stick_motion_eps=0.0, rigid_avbd_contact_alpha=0.0,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False, particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=32, particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )

        device = self.object_state_0.body_q.device
        self._t_frame = wp.zeros(1, dtype=wp.float32, device=device)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=device)
        self._home_q_wp = kf(self.home_q)
        self._pregrasp_q_wp = kf(self.pregrasp_q)
        self._pickup_q_wp = kf(self.pickup_q)
        self._lift_q_wp = kf(self.lift_q)
        self._place_high_q_wp = kf(self.place_high_q)
        self._place_q_wp = kf(self.place_q)
        self._finger_bodies_wp = wp.array(self.robot_finger_bodies, dtype=wp.int32, device=device)
        self._proxy_bodies_wp = wp.array(self.gripper_proxy_bodies, dtype=wp.int32, device=device)
        self.graph = None
        self._frames_simulated = 0
        self._capture_enabled = wp.get_device(str(ik_model.device)).is_cuda and self.sim_substeps % 2 == 0

        self.left_proxy, self.right_proxy = (int(b) for b in self.gripper_proxy_bodies)
        self._grip_reaction_raw = wp.zeros(2, dtype=wp.float32, device=device)
        self._grip_reaction = wp.zeros(2, dtype=wp.float32, device=device)
        self._grip_target = wp.array([self.gripper_open], dtype=wp.float32, device=device)
        self._grip_latched = wp.zeros(1, dtype=wp.int32, device=device)

        self._sync_gripper_proxies()

        viz_builder = newton.ModelBuilder()
        viz_builder.add_builder(robot_builder)
        self.viz_object_body_start = robot_builder.body_count
        viz_builder.add_builder(object_builder)
        self.viz_model = viz_builder.finalize(device=ik_model.device)
        self.viz_state = self.viz_model.state()
        self._sync_viz_state()

        self.viewer.set_model(self.viz_model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.85, 0.30, 0.55), pitch=-22.0, yaw=-130.0)

    @staticmethod
    def _device_from_args(args):
        return wp.get_device(args.device) if args.device else None

    def _build_robot_builder(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.005
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            Path(asset_path) / "urdf" / "fr3_franka_hand.urdf",
            xform=wp.transform((-0.45, -0.45, self.table_top_z), wp.quat_identity()),
            floating=False, enable_self_collisions=False, parse_visuals_as_colliders=False,
            collapse_fixed_joints=True, force_show_colliders=False,
        )
        builder.joint_q[:9] = self.home_q.tolist()
        builder.joint_target_q[:9] = self.home_q.tolist()
        for dof in range(9):
            builder.joint_target_ke[dof] = 420.0 if dof < 7 else 300.0
            builder.joint_target_kd[dof] = 42.0 if dof < 7 else 30.0
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            builder.joint_effort_limit[dof] = 87.0 if dof < 7 else 20.0
            builder.joint_armature[dof] = 0.1

        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof in range(7):
            gravcomp_attr.values[dof] = True
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body in range(2, builder.body_count):
            gravcomp_body.values[body] = 1.0

        table_cfg = newton.ModelBuilder.ShapeConfig(
            margin=1.0e-3, density=1000.0, is_visible=False, ke=5.0e4, kd=5.0e2, mu=1.0
        )
        builder.add_shape_box(
            body=-1, xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half[0]), hy=float(self.table_half[1]), hz=float(self.table_half[2]),
            cfg=table_cfg, label="robot_contact_table",
        )
        return builder

    def _build_object_builder(self, robot_builder: newton.ModelBuilder) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.mu = 0.8

        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=1.0e2, mu=0.8)
        builder.add_shape_box(
            body=-1, xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half[0]), hy=float(self.table_half[1]), hz=float(self.table_half[2]),
            cfg=table_cfg, color=wp.vec3(0.52, 0.52, 0.48), label="table",
        )

        builder.default_particle_radius = self.soft_particle_radius
        builder.particle_max_velocity = 50.0
        dx, dy, dz = self.soft_grid_dims
        builder.add_soft_grid(
            pos=wp.vec3(float(self.pick_xy[0] - 0.5 * dx * self.soft_grid_cell),
                        float(self.pick_xy[1] - 0.5 * dy * self.soft_grid_cell),
                        float(self.table_top_z)),
            rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dx, dim_y=dy, dim_z=dz,
            cell_x=self.soft_grid_cell, cell_y=self.soft_grid_cell, cell_z=self.soft_grid_cell,
            density=self.soft_density, k_mu=self.soft_k_mu, k_lambda=self.soft_k_lambda, k_damp=self.soft_k_damp,
        )

        # Gripper proxies WITH particle collision so the pads contact the soft block.
        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, is_visible=False, has_shape_collision=True, has_particle_collision=True,
            margin=self.gripper_proxy_margin, gap=self.gripper_proxy_gap, ke=5.0e4, kd=1.0e2, mu=1.0,
        )
        self.gripper_proxy_bodies = []
        self.gripper_proxy_shapes = []
        for label, finger_body in zip(
            ("left_gripper_contact_proxy", "right_gripper_contact_proxy"), self.robot_finger_bodies, strict=True
        ):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                                    is_kinematic=True, label=label)
            self.gripper_proxy_bodies.append(body)
            self._copy_robot_finger_collision_shapes(robot_builder, builder, finger_body, body, proxy_cfg, label)
        return builder

    def _copy_robot_finger_collision_shapes(self, robot_builder, object_builder, finger_body, proxy_body, cfg, prefix):
        n = 0
        for idx, sb in enumerate(robot_builder.shape_body):
            if sb != finger_body:
                continue
            if not (robot_builder.shape_flags[idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            shape = object_builder.add_shape(
                body=proxy_body, type=robot_builder.shape_type[idx], xform=robot_builder.shape_transform[idx],
                cfg=cfg, scale=robot_builder.shape_scale[idx], src=robot_builder.shape_source[idx],
                label=f"{prefix}_shape_{n}",
            )
            self.gripper_proxy_shapes.append(shape)
            n += 1
        if n == 0:
            raise RuntimeError(f"No colliding shapes on finger body {finger_body!r}.")

    def _body_offset_position(self, state, body, offset):
        bq = state.body_q.numpy()[body]
        return bq[:3] + _quat_rotate_xyzw(bq[3:7], offset)

    def _solve_gripper_ik(self, model, state, target_pos):
        bq = state.body_q.numpy()[self.ee_body]
        ee_tf = wp.transform(*bq)
        target_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
        objs = [
            ik.IKObjectivePosition(link_index=self.ee_body, link_offset=wp.vec3(*self.ee_offset),
                                   target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3)),
            ik.IKObjectiveRotation(link_index=self.ee_body, link_offset_rotation=wp.quat_identity(),
                                   target_rotations=wp.array([_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4)),
            ik.IKObjectiveJointLimit(joint_limit_lower=model.joint_limit_lower,
                                     joint_limit_upper=model.joint_limit_upper, weight=10.0),
        ]
        solver = ik.IKSolver(model=model, n_problems=1, objectives=objs, lambda_initial=0.1,
                             jacobian_mode=ik.IKJacobianType.ANALYTIC)
        solver.step(target_q, target_q, iterations=64)
        q = np.array(target_q.numpy()[0, :9], dtype=np.float32)
        q[7:9] = self.gripper_open
        return q

    def _set_robot_targets(self, substep):
        wp.launch(_set_robot_targets_kernel, dim=9, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_q_wp, self._pregrasp_q_wp, self._pickup_q_wp,
            self._lift_q_wp, self._place_high_q_wp, self._place_q_wp, self._grip_target,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def _update_grip_target(self, substep):
        wp.launch(_update_grip_target_kernel, dim=1, inputs=[
            self._t_frame, substep, self.sim_dt, self._grip_reaction, self.grip_force_threshold,
            self.grip_close_rate, self.gripper_open,
        ], outputs=[self._grip_target, self._grip_latched], device=self._grip_target.device)

    def _sync_gripper_proxies(self):
        wp.launch(_sync_gripper_proxies_kernel, dim=2,
                  inputs=[self.robot_state_0.body_q, self._finger_bodies_wp, self._proxy_bodies_wp],
                  outputs=[self.object_state_0.body_q, self.object_state_1.body_q],
                  device=self.object_state_0.body_q.device)

    def _update_grip_reaction(self):
        c = self.object_contacts
        self._grip_reaction_raw.zero_()
        wp.launch(_soft_grip_reaction_kernel, dim=c.soft_contact_particle.shape[0], inputs=[
            c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape, c.soft_contact_body_pos,
            c.soft_contact_normal, self.object_state_0.particle_q, self.object_model.particle_radius,
            self.object_model.shape_body, self.object_state_0.body_q, self.soft_contact_ke,
            self.left_proxy, self.right_proxy,
        ], outputs=[self._grip_reaction_raw], device=self._grip_reaction_raw.device)
        wp.launch(_blend_grip_kernel, dim=2, inputs=[self._grip_reaction_raw, self.grip_filter],
                  outputs=[self._grip_reaction], device=self._grip_reaction.device)

    def grip_reaction_norms(self):
        r = self._grip_reaction.numpy()
        return float(r[0]), float(r[1])

    def block_centroid(self):
        return self.object_state_0.particle_q.numpy().mean(axis=0)

    def _sync_viz_state(self):
        bq = self.viz_state.body_q.numpy()
        bqd = self.viz_state.body_qd.numpy()
        rc = self.robot_model.body_count
        bq[:rc] = self.robot_state_0.body_q.numpy()
        bqd[:rc] = self.robot_state_0.body_qd.numpy()
        s = self.viz_object_body_start
        e = s + self.object_model.body_count
        bq[s:e] = self.object_state_0.body_q.numpy()
        bqd[s:e] = self.object_state_0.body_qd.numpy()
        self.viz_state.body_q.assign(bq)
        self.viz_state.body_qd.assign(bqd)
        wp.copy(self.viz_state.particle_q, self.object_state_0.particle_q)
        wp.copy(self.viz_state.particle_qd, self.object_state_0.particle_qd)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self._update_grip_target(substep)
            self._set_robot_targets(substep)

            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(self.robot_state_0, self.robot_state_1, self.robot_control, self.robot_contacts, self.sim_dt)
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

            self._sync_gripper_proxies()

            self.object_state_0.clear_forces()
            self.object_model.collide(self.object_state_0, self.object_contacts)
            self.object_solver.step(self.object_state_0, self.object_state_1, self.object_control, self.object_contacts, self.sim_dt)
            self.object_state_0, self.object_state_1 = self.object_state_1, self.object_state_0

            self._update_grip_reaction()

    def step(self):
        self._t_frame.assign(np.array([self.sim_time], dtype=np.float32))
        if self.graph is None and self._capture_enabled and self._frames_simulated >= 1:
            saved = (self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1)
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as exc:
                self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1 = saved
                print(f"CUDA graph capture failed ({exc}); continuing without capture.")
                self._capture_enabled = False
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._frames_simulated += 1
        self.sim_time += self.frame_dt

    def render(self):
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

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
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations for the objects.")
        parser.add_argument("--grip-threshold", type=float, default=8.0, help="Gripper stop force per finger [N].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="soft_pickplace_franka")
    examples.run(Example(viewer, args), args)
