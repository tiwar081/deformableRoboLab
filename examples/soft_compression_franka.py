from __future__ import annotations

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


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.func
def _wp_smoothstep(x: float) -> float:
    x = wp.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


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
    # Device-side mirror of the keyframe schedule so the substep loop is free
    # of host round-trips and can be captured into a CUDA graph.
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
        alpha = _wp_smoothstep(t / descend_start)
        q = (1.0 - alpha) * home_q[i] + alpha * pregrasp_q[i]
    elif t < close_start:
        alpha = _wp_smoothstep((t - descend_start) / (close_start - descend_start))
        q = (1.0 - alpha) * pregrasp_q[i] + alpha * pickup_q[i]
    elif t < carry_start:
        q = pickup_q[i]
    elif t < settle_start:
        alpha = _wp_smoothstep((t - carry_start) / (settle_start - carry_start))
        q = (1.0 - alpha) * pickup_q[i] + alpha * drop_q[i]
    elif t < retreat_start:
        q = drop_q[i]
    else:
        alpha = _wp_smoothstep((t - retreat_start) / 2.0)
        q = (1.0 - alpha) * drop_q[i] + alpha * home_q[i]

    if i >= 7:
        if t < close_start:
            q = gripper_open
        elif t < hold_start:
            alpha = _wp_smoothstep((t - close_start) / (hold_start - close_start))
            q = (1.0 - alpha) * gripper_open + alpha * gripper_closed
        elif t < release_start:
            q = gripper_closed
        elif t < retreat_start:
            alpha = _wp_smoothstep((t - release_start) / (retreat_start - release_start))
            q = (1.0 - alpha) * gripper_closed + alpha * gripper_open
        else:
            q = gripper_open

    joint_target_q[i] = q


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
        self.original_cube_half = 0.025
        self.original_cube_density = 7800.0
        self.sheet_half = np.array([0.09, 0.06, 0.004], dtype=np.float32)
        self.handle_half = np.array([0.016, 0.012, 0.024], dtype=np.float32)
        self.sheet_contact_margin = 0.001
        self.gripper_proxy_margin = 0.001
        self.gripper_proxy_gap = 0.008
        self.sheet_start_pos = np.array(
            [0.10, -0.55, self.table_top_z + self.sheet_half[2]],
            dtype=np.float32,
        )
        self.handle_local_pos = np.array(
            [0.0, 0.0, self.sheet_half[2] + self.handle_half[2]],
            dtype=np.float32,
        )
        self.handle_start_pos = self.sheet_start_pos + self.handle_local_pos
        cube_volume = (2.0 * self.original_cube_half) ** 3
        sheet_volume = float(np.prod(2.0 * self.sheet_half))
        handle_volume = float(np.prod(2.0 * self.handle_half))
        self.sheet_density = 2.0 * self.original_cube_density * cube_volume / (sheet_volume + handle_volume)
        # Soft body: the FEM block from Newton's rigid_soft_contact example
        # (the only upstream two-way VBD rigid+soft scene), scaled to the
        # table. soft_start_pos is the block center on the table.
        self.soft_start_pos = np.array(
            [0.28, -0.30, self.table_top_z],
            dtype=np.float32,
        )
        # Same size as the rigid cube in the companion example (5 cm sides).
        self.soft_grid_dims = (4, 4, 4)
        self.soft_grid_cell = 0.0125
        # Drop one half-block-width off center so the sheet lands partly over
        # the soft body rather than centered on it.
        self.soft_drop_offset = np.array(
            [0.5 * self.soft_grid_dims[0] * self.soft_grid_cell, 0.0, 0.0],
            dtype=np.float32,
        )
        # Very soft FEM response: about 5% of the upstream stiffness, so the
        # sheet visibly sinks into the block like a small pillow.
        self.soft_k_mu = 1.0e4
        self.soft_k_lambda = 5.0e4
        # Contact boundary sits one particle radius above the rendered surface;
        # 3.5 mm keeps it visually tight while leaving >2 substeps of contact
        # engagement at the impact speed (~1.5 mm/substep).
        self.soft_particle_radius = 0.0035
        self.soft_body_contact_margin = 0.01
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005
        self.soft_contact_ke = 1.0e5
        self.soft_contact_kd = 1.0e-4
        self.soft_contact_kf = 1.0e3
        self.soft_contact_mu = 0.3
        self.grasp_tcp_height = float(self.handle_start_pos[2])
        self.drop_tcp_height = self.table_top_z + 0.19
        self.gripper_open = 0.04
        # The sheet exists only in the VBD object model, so no contact force can
        # stop the fingers in the robot model. The close target must itself stop
        # at the handle: pad face at handle half-width plus the summed contact
        # margins, minus a small interference whose contact stiffness sets the
        # grip force.
        grasp_interference = 0.001
        self.gripper_closed = (
            self.handle_half[1] + self.sheet_contact_margin + self.gripper_proxy_margin - grasp_interference
        )
        self.home_q = np.array(
            [
                -0.0036802115,
                0.023901723,
                0.003680411,
                -2.3683236,
                -0.00012918962,
                2.3922248,
                0.785492,
                self.gripper_open,
                self.gripper_open,
            ],
            dtype=np.float32,
        )
        robot_builder = self._build_robot_builder()

        ik_model = robot_builder.finalize(device=self._device_from_args(args))
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)

        self.ee_body = _find_body(list(ik_model.body_label), "fr3_link7")
        self.ee_offset = np.array([0.0, 0.0, 0.22], dtype=np.float32)

        # Pre-grasp waypoint straight above the handle: the joint-space descent
        # from home traces an arc, and without the waypoint the open pads clip
        # the handle sideways before the grasp.
        pregrasp_pos = self.handle_start_pos.copy()
        pregrasp_pos[2] = self.table_top_z + 0.12
        self.pregrasp_q = self._solve_gripper_ik(ik_model, ik_state, pregrasp_pos)

        pickup_pos = self.handle_start_pos.copy()
        pickup_pos[2] = self.grasp_tcp_height
        self.pickup_q = self._solve_gripper_ik(ik_model, ik_state, pickup_pos)

        drop_pos = self.soft_start_pos + self.soft_drop_offset
        drop_pos[2] = self.drop_tcp_height
        self.drop_q = self._solve_gripper_ik(ik_model, ik_state, drop_pos)

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
            self.robot_model,
            reduce_contacts=True,
            rigid_contact_max=robot_contact_max,
            broad_phase="nxn",
        )
        self.robot_contacts = self.robot_collision_pipeline.contacts()
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.robot_model,
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=robot_contact_max,
            njmax=robot_contact_max * 2,
            cone="elliptic",
            impratio=50.0,
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
        self._restore_gripper_proxy_materials()
        self._restore_sheet_materials()
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        self.object_collision_pipeline = newton.CollisionPipeline(
            self.object_model,
            contact_matching="latest",
            soft_contact_margin=self.soft_body_contact_margin,
        )
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)

        self.object_solver = newton.solvers.SolverVBD(
            self.object_model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=512,
            # The flat sheet face pressing into the soft block produces hundreds of
            # contacts; an overflowing buffer drops contacts frame-to-frame and
            # destabilizes the impact.
            rigid_body_particle_contact_buffer_size=4096,
            rigid_contact_history=True,
            # Same contact configuration as the cable examples: hard contacts
            # with sticky replay disabled and full per-step penetration
            # correction, so the kinematically-driven gripper pads do not kick
            # the grasped object.
            rigid_contact_stick_motion_eps=0.0,
            rigid_avbd_contact_alpha=0.0,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )

        # Device-side trajectory/sync state so the substep loop has no host
        # round-trips and can be captured into a CUDA graph (one graph launch
        # per frame instead of hundreds of individual kernel launches).
        self._t_frame = wp.zeros(1, dtype=wp.float32, device=ik_model.device)
        self._home_q_wp = wp.array(self.home_q, dtype=wp.float32, device=ik_model.device)
        self._pregrasp_q_wp = wp.array(self.pregrasp_q, dtype=wp.float32, device=ik_model.device)
        self._pickup_q_wp = wp.array(self.pickup_q, dtype=wp.float32, device=ik_model.device)
        self._drop_q_wp = wp.array(self.drop_q, dtype=wp.float32, device=ik_model.device)
        self._finger_bodies_wp = wp.array(self.robot_finger_bodies, dtype=wp.int32, device=ik_model.device)
        self._proxy_bodies_wp = wp.array(self.gripper_proxy_bodies, dtype=wp.int32, device=ik_model.device)
        self.graph = None
        self._frames_simulated = 0
        # Capture needs an even substep count so the state buffer swap returns
        # to its starting binding at the end of each captured frame.
        self._capture_enabled = wp.get_device(str(ik_model.device)).is_cuda and self.sim_substeps % 2 == 0

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
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
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
            margin=1.0e-3,
            density=1000.0,
            is_visible=False,
            ke=5.0e4,
            kd=5.0e2,
            mu=1.0,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half[0]),
            hy=float(self.table_half[1]),
            hz=float(self.table_half[2]),
            cfg=table_cfg,
            label="robot_contact_table",
        )

        return builder

    def _build_object_builder(self, robot_builder: newton.ModelBuilder) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.mu = 0.8

        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=1.0e2, mu=0.8)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half[0]),
            hy=float(self.table_half[1]),
            hz=float(self.table_half[2]),
            cfg=table_cfg,
            color=wp.vec3(0.52, 0.52, 0.48),
            label="table",
        )

        # Soft FEM block, centered at soft_start_pos on the table.
        builder.default_particle_radius = self.soft_particle_radius
        builder.particle_max_velocity = 50.0
        dim_x, dim_y, dim_z = self.soft_grid_dims
        builder.add_soft_grid(
            pos=wp.vec3(
                float(self.soft_start_pos[0] - 0.5 * dim_x * self.soft_grid_cell),
                float(self.soft_start_pos[1] - 0.5 * dim_y * self.soft_grid_cell),
                float(self.soft_start_pos[2]),
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=self.soft_grid_cell,
            cell_y=self.soft_grid_cell,
            cell_z=self.soft_grid_cell,
            density=100.0,
            k_mu=self.soft_k_mu,
            k_lambda=self.soft_k_lambda,
            k_damp=1.0,
        )

        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=False,
            has_shape_collision=True,
            has_particle_collision=False,
            margin=self.gripper_proxy_margin,
            gap=self.gripper_proxy_gap,
            ke=5.0e4,
            kd=1.0e2,
            mu=1.0,
        )
        self.gripper_proxy_bodies = []
        self.gripper_proxy_shapes = []
        for label, robot_finger_body in zip(
            ("left_gripper_contact_proxy", "right_gripper_contact_proxy"),
            self.robot_finger_bodies,
            strict=True,
        ):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                is_kinematic=True,
                label=label,
            )
            self.gripper_proxy_bodies.append(body)
            self._copy_robot_finger_collision_shapes(
                robot_builder=robot_builder,
                object_builder=builder,
                robot_finger_body=robot_finger_body,
                proxy_body=body,
                cfg=proxy_cfg,
                label_prefix=label,
            )

        # Compound metal sheet: a broad thin plate with a small top handle. Its
        # density is computed so the combined rigid body is exactly twice the
        # mass of the old 5 cm steel cube.
        self.sheet_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.sheet_start_pos), wp.quat_identity()),
            label="compression_sheet",
        )
        sheet_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.sheet_density,
            margin=self.sheet_contact_margin,
            ke=5.0e4,
            kd=1.0e2,
            mu=0.8,
        )
        builder.add_shape_box(
            body=self.sheet_body,
            hx=float(self.sheet_half[0]),
            hy=float(self.sheet_half[1]),
            hz=float(self.sheet_half[2]),
            cfg=sheet_cfg,
            color=wp.vec3(0.62, 0.65, 0.68),
            label="metal_sheet",
        )
        builder.add_shape_box(
            body=self.sheet_body,
            xform=wp.transform(wp.vec3(*self.handle_local_pos), wp.quat_identity()),
            hx=float(self.handle_half[0]),
            hy=float(self.handle_half[1]),
            hz=float(self.handle_half[2]),
            cfg=sheet_cfg,
            color=wp.vec3(0.40, 0.42, 0.45),
            label="grasp_handle",
        )

        return builder

    def _copy_robot_finger_collision_shapes(
        self,
        robot_builder: newton.ModelBuilder,
        object_builder: newton.ModelBuilder,
        robot_finger_body: int,
        proxy_body: int,
        cfg: newton.ModelBuilder.ShapeConfig,
        label_prefix: str,
    ) -> None:
        shape_count = 0
        for shape_idx, shape_body in enumerate(robot_builder.shape_body):
            if shape_body != robot_finger_body:
                continue
            if not (robot_builder.shape_flags[shape_idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue

            shape = object_builder.add_shape(
                body=proxy_body,
                type=robot_builder.shape_type[shape_idx],
                xform=robot_builder.shape_transform[shape_idx],
                cfg=cfg,
                scale=robot_builder.shape_scale[shape_idx],
                src=robot_builder.shape_source[shape_idx],
                label=f"{label_prefix}_shape_{shape_count}",
            )
            self.gripper_proxy_shapes.append(shape)
            shape_count += 1

        if shape_count == 0:
            finger_label = robot_builder.body_label[robot_finger_body]
            raise RuntimeError(f"No colliding shapes found on Franka finger body {finger_label!r}.")

    def _restore_gripper_proxy_materials(self) -> None:
        if not self.gripper_proxy_shapes:
            return

        mu = self.object_model.shape_material_mu.numpy()
        ke = self.object_model.shape_material_ke.numpy()
        kd = self.object_model.shape_material_kd.numpy()
        for shape in self.gripper_proxy_shapes:
            mu[shape] = 1.0
            ke[shape] = 5.0e4
            kd[shape] = 1.0e2
        self.object_model.shape_material_mu.assign(mu)
        self.object_model.shape_material_ke.assign(ke)
        self.object_model.shape_material_kd.assign(kd)

    def _restore_sheet_materials(self) -> None:
        # Keep the sheet-block pair firm enough that the rigid sheet compresses
        # the pillow-like soft body instead of sinking through its contact skin.
        mu = self.object_model.shape_material_mu.numpy()
        ke = self.object_model.shape_material_ke.numpy()
        kd = self.object_model.shape_material_kd.numpy()
        for shape, body in enumerate(self.object_model.shape_body.numpy()):
            if int(body) == self.sheet_body:
                mu[shape] = 0.8
                ke[shape] = 1.0e5
                kd[shape] = 1.0e-4
        self.object_model.shape_material_mu.assign(mu)
        self.object_model.shape_material_ke.assign(ke)
        self.object_model.shape_material_kd.assign(kd)

    def _body_offset_position(self, state, body: int, offset: np.ndarray) -> np.ndarray:
        body_q = state.body_q.numpy()[body]
        pos = body_q[:3]
        quat = body_q[3:7]
        return pos + _quat_rotate_xyzw(quat, offset)

    def _end_effector_position(self) -> np.ndarray:
        return self._body_offset_position(self.robot_state_0, self.ee_body, self.ee_offset)

    def _solve_gripper_ik(self, model, state, target_pos: np.ndarray) -> np.ndarray:
        body_q = state.body_q.numpy()[self.ee_body]
        ee_tf = wp.transform(*body_q)
        target_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
        pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_body,
            link_offset=wp.vec3(*self.ee_offset),
            target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3),
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=10.0,
        )
        solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[pos_obj, rot_obj, joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        solver.step(target_q, target_q, iterations=64)

        q = np.array(target_q.numpy()[0, :9], dtype=np.float32)
        q[7:9] = self.gripper_open
        return q

    def _set_robot_targets(self, substep: int) -> None:
        wp.launch(
            _set_robot_targets_kernel,
            dim=9,
            inputs=[
                self._t_frame,
                substep,
                self.sim_dt,
                self._home_q_wp,
                self._pregrasp_q_wp,
                self._pickup_q_wp,
                self._drop_q_wp,
                self.gripper_open,
                self.gripper_closed,
            ],
            outputs=[self.robot_control.joint_target_q],
            device=self.robot_control.joint_target_q.device,
        )

    def _sync_gripper_proxies(self) -> None:
        wp.launch(
            _sync_gripper_proxies_kernel,
            dim=2,
            inputs=[self.robot_state_0.body_q, self._finger_bodies_wp, self._proxy_bodies_wp],
            outputs=[self.object_state_0.body_q, self.object_state_1.body_q],
            device=self.object_state_0.body_q.device,
        )

    def _sync_viz_state(self) -> None:
        body_q = self.viz_state.body_q.numpy()
        body_qd = self.viz_state.body_qd.numpy()

        robot_body_count = self.robot_model.body_count
        body_q[:robot_body_count] = self.robot_state_0.body_q.numpy()
        body_qd[:robot_body_count] = self.robot_state_0.body_qd.numpy()

        object_start = self.viz_object_body_start
        object_end = object_start + self.object_model.body_count
        body_q[object_start:object_end] = self.object_state_0.body_q.numpy()
        body_qd[object_start:object_end] = self.object_state_0.body_qd.numpy()

        self.viz_state.body_q.assign(body_q)
        self.viz_state.body_qd.assign(body_qd)

        # The soft body's particles are part of the viz model too; without this
        # copy the viewer draws the block frozen at its rest shape and rigid
        # objects appear to penetrate it.
        wp.copy(self.viz_state.particle_q, self.object_state_0.particle_q)
        wp.copy(self.viz_state.particle_qd, self.object_state_0.particle_qd)

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            self._set_robot_targets(substep)

            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(
                self.robot_state_0,
                self.robot_state_1,
                self.robot_control,
                self.robot_contacts,
                self.sim_dt,
            )
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

            self._sync_gripper_proxies()

            self.object_state_0.clear_forces()
            self.object_model.collide(self.object_state_0, self.object_contacts)
            self.object_solver.step(
                self.object_state_0,
                self.object_state_1,
                self.object_control,
                self.object_contacts,
                self.sim_dt,
            )
            self.object_state_0, self.object_state_1 = self.object_state_1, self.object_state_0

    def step(self) -> None:
        self._t_frame.assign(np.array([self.sim_time], dtype=np.float32))

        # Capture after one uncaptured warm-up frame so all lazy solver/pipeline
        # allocations have happened (allocation inside capture raises).
        if self.graph is None and self._capture_enabled and self._frames_simulated >= 1:
            saved_states = (self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1)
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as exc:
                self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1 = saved_states
                print(f"CUDA graph capture failed ({exc}); continuing without capture.")
                self._capture_enabled = False

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._frames_simulated += 1
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

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

        # After the release the sheet must have dropped out of the gripper and
        # landed near the soft block (on it, or bounced off next to it).
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
