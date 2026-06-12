from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np
from pxr import Usd

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
        self.cable_radius = 0.008
        self.cable_friction = 1.5
        self.gripper_body_contact_margin = self.cable_radius
        self.soft_start_pos = np.array(
            [0.28, -0.30, self.table_top_z + 0.03],
            dtype=np.float32,
        )
        self.soft_particle_radius = 0.005
        self.soft_body_contact_margin = 0.01
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005
        self.soft_contact_ke = 2.0e6
        self.soft_contact_kd = 1.0e-7
        self.soft_contact_mu = 0.5
        self.cable_direction = np.array([1.0, 0.05, 0.0], dtype=np.float32)
        self.cable_direction /= np.linalg.norm(self.cable_direction)
        self.cable_segment_length = 0.035
        self.cable_node_count = 15
        self.grasp_tcp_height = self.table_top_z
        self.gripper_open = 0.045
        self.gripper_closed = 0.0
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
        initial_ee_pos = self._body_offset_position(ik_state, self.ee_body, self.ee_offset)
        self.cable_start_pos = initial_ee_pos.copy()
        table_margin = 0.04
        self.cable_start_pos[0] = np.clip(
            self.cable_start_pos[0],
            float(self.table_pos[0] - self.table_half[0] + table_margin),
            float(self.table_pos[0] + self.table_half[0] - table_margin),
        )
        self.cable_start_pos[1] = np.clip(
            self.cable_start_pos[1],
            float(self.table_pos[1] - self.table_half[1] + table_margin),
            float(self.table_pos[1] + self.table_half[1] - table_margin),
        )
        self.cable_start_pos[2] = self.table_top_z + self.cable_radius
        grasp_pos = self.cable_start_pos + self.cable_direction * self.cable_segment_length * (
            self.cable_node_count - 1
        ) * 0.25
        grasp_pos[2] = self.grasp_tcp_height
        self.pickup_q = self._solve_gripper_ik(ik_model, ik_state, grasp_pos)

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

        object_builder = self._build_object_builder(self.cable_start_pos, robot_builder)
        object_builder.color(balance_colors=False)

        self.object_model = object_builder.finalize(device=ik_model.device)
        self.object_model.soft_contact_ke = self.soft_contact_ke
        self.object_model.soft_contact_kd = self.soft_contact_kd
        self.object_model.soft_contact_mu = self.soft_contact_mu
        self.object_model.shape_material_ke.fill_(5.0e4)
        self.object_model.shape_material_kd.fill_(1.0e2)
        self.object_model.shape_material_mu.fill_(0.8)
        self._restore_cable_materials()
        self._restore_gripper_proxy_materials()
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
            rigid_contact_history=True,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )
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

    def _build_object_builder(self, cable_start_pos: np.ndarray, robot_builder: newton.ModelBuilder) -> newton.ModelBuilder:
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

        duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
        usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
        prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh = newton.TetMesh.create_from_usd(prim)
        builder.add_soft_mesh(
            pos=wp.vec3(*self.soft_start_pos),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=100.0,
            k_mu=1.0e6,
            k_lambda=1.0e6,
            k_damp=1.0e-6,
            particle_radius=self.soft_particle_radius,
            label="soft_duck",
        )

        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=False,
            has_shape_collision=True,
            has_particle_collision=False,
            margin=self.gripper_body_contact_margin,
            gap=self.gripper_body_contact_margin,
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

        positions = [
            wp.vec3(*(cable_start_pos + self.cable_direction * self.cable_segment_length * i))
            for i in range(self.cable_node_count)
        ]

        self.cable_body_start = builder.body_count
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=80.0,
            margin=0.001,
            ke=2.0e4,
            kd=20.0,
            mu=self.cable_friction,
        )
        self.cable_bodies, self.cable_joints = builder.add_rod(
            positions=positions,
            radius=self.cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=2.5e4,
            stretch_damping=0.05,
            bend_stiffness=1.5e1,
            bend_damping=0.02,
            label="vbd_cable",
            wrap_in_articulation=True,
        )
        self.cable_body_count = len(self.cable_bodies)

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

    def _restore_cable_materials(self) -> None:
        body_set = set(self.cable_bodies)
        mu = self.object_model.shape_material_mu.numpy()
        for shape, body in enumerate(self.object_model.shape_body.numpy()):
            if int(body) in body_set:
                mu[shape] = self.cable_friction
        self.object_model.shape_material_mu.assign(mu)

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

    def _set_robot_targets(self, t: float) -> None:
        lower_start = 0.0
        close_start = 2.8
        hold_start = 4.0
        lift_start = 4.8
        sweep_start = 6.8

        home_open = self.home_q.copy()
        pickup_open = self.pickup_q.copy()
        pickup_closed = self.pickup_q.copy()
        home_closed = self.home_q.copy()
        pickup_open[7:9] = self.gripper_open
        pickup_closed[7:9] = self.gripper_closed
        home_closed[7:9] = self.gripper_closed

        if t < lower_start:
            q = home_open
        elif t < close_start:
            alpha = _smoothstep((t - lower_start) / (close_start - lower_start))
            q = (1.0 - alpha) * home_open + alpha * pickup_open
        elif t < hold_start:
            alpha = _smoothstep((t - close_start) / (hold_start - close_start))
            q = (1.0 - alpha) * pickup_open + alpha * pickup_closed
        elif t < lift_start:
            q = pickup_closed
        elif t < sweep_start:
            alpha = _smoothstep((t - lift_start) / (sweep_start - lift_start))
            q = (1.0 - alpha) * pickup_closed + alpha * home_closed
        else:
            phase = 2.0 * math.pi * 0.18 * (t - sweep_start)
            q = home_closed.copy()
            q[0] += 0.55 * math.sin(phase)
            q[3] += 0.18 * math.sin(phase + 0.35)
            q[5] -= 0.20 * math.sin(phase)

        target = self.robot_control.joint_target_q.numpy()
        target[:9] = q
        self.robot_control.joint_target_q.assign(target)

    def _sync_gripper_proxies(self) -> None:
        robot_q = self.robot_state_0.body_q.numpy()
        object_q0 = self.object_state_0.body_q.numpy()
        object_q1 = self.object_state_1.body_q.numpy()

        for proxy_body, finger_body in zip(self.gripper_proxy_bodies, self.robot_finger_bodies, strict=True):
            object_q0[proxy_body] = robot_q[finger_body]
            object_q1[proxy_body] = robot_q[finger_body]

        self.object_state_0.body_q.assign(object_q0)
        self.object_state_1.body_q.assign(object_q1)

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

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._set_robot_targets(t)

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
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.object_state_0.body_q.numpy()
        cable_q = body_q[self.cable_body_start : self.cable_body_start + self.cable_body_count]
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(cable_q)):
            raise ValueError("Non-finite body transform detected.")

        particle_q = self.object_state_0.particle_q.numpy()
        if particle_q.size and not np.all(np.isfinite(particle_q)):
            raise ValueError("Non-finite soft-body particle position detected.")
        if particle_q.size and np.min(particle_q[:, 2]) < self.table_top_z - 0.03:
            raise ValueError("The soft body fell through the table.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "cable_soft_franka.usd"), num_frames=720)
        parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=8, help="VBD iterations for the cable.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="cable_soft_franka")
    examples.run(Example(viewer, args), args)
