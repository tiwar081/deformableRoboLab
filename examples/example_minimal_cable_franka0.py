from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

import warp as wp

import examples
try:
    import newton
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
        self.cube_half = 0.035
        self.cube_start_pos = np.array(
            [0.28, -0.30, self.table_top_z + self.cube_half],
            dtype=np.float32,
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
                0.018,
                0.018,
            ],
            dtype=np.float32,
        )

        rigid_builder = self._build_rigid_builder()
        rigid_builder_viz = copy.deepcopy(rigid_builder)

        self.rigid_model = rigid_builder.finalize(device=self._device_from_args(args))
        self.rigid_state_0 = self.rigid_model.state()
        self.rigid_state_1 = self.rigid_model.state()
        self.rigid_control = self.rigid_model.control()
        newton.eval_fk(self.rigid_model, self.rigid_model.joint_q, self.rigid_model.joint_qd, self.rigid_state_0)
        wp.copy(self.rigid_control.joint_target_q, self.rigid_model.joint_q)

        self.ee_body = _find_body(list(self.rigid_model.body_label), "fr3_link7")
        self.ee_offset = np.array([0.0, 0.0, 0.22], dtype=np.float32)
        handle_pos = self._end_effector_position()

        cable_builder = self._build_cable_builder(handle_pos)
        cable_builder_viz = copy.deepcopy(cable_builder)

        self.cable_model = cable_builder.finalize(device=self.rigid_model.device)
        self.cable_state_0 = self.cable_model.state()
        self.cable_state_1 = self.cable_model.state()
        self.cable_control = self.cable_model.control()
        self.cable_contacts = self.cable_model.contacts()
        self.cable_solver = newton.solvers.SolverVBD(
            self.cable_model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=64,
            rigid_contact_history=False,
        )

        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.rigid_model,
            solver="newton",
            integrator="implicitfast",
            iterations=12,
            ls_iterations=50,
            disable_contacts=False,
            use_mujoco_cpu=False,
        )

        viz_builder = newton.ModelBuilder()
        viz_builder.add_builder(rigid_builder_viz)
        viz_builder.add_builder(cable_builder_viz)
        self.rigid_body_count = self.rigid_model.body_count
        self.cable_body_count = self.cable_model.body_count
        self.viz_model = viz_builder.finalize(device=self.rigid_model.device)
        self.viz_state = self.viz_model.state()
        self._sync_viz_state()

        self.viewer.set_model(self.viz_model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.85, 0.30, 0.55), pitch=-22.0, yaw=-130.0)

    @staticmethod
    def _device_from_args(args):
        return wp.get_device(args.device) if args.device else None

    def _build_rigid_builder(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            Path(asset_path) / "urdf" / "fr3_franka_hand.urdf",
            xform=wp.transform((-0.45, -0.45, self.table_top_z), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=True,
        )

        builder.joint_q[:9] = self.home_q.tolist()
        builder.joint_target_q[:9] = self.home_q.tolist()
        for dof in range(9):
            builder.joint_target_ke[dof] = 420.0 if dof < 7 else 300.0
            builder.joint_target_kd[dof] = 42.0 if dof < 7 else 30.0
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            builder.joint_effort_limit[dof] = 87.0 if dof < 7 else 20.0
            builder.joint_armature[dof] = 0.1

        gravcomp_joint = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        gravcomp_joint.values = gravcomp_joint.values or {}
        for dof in range(7):
            gravcomp_joint.values[dof] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp_body.values = gravcomp_body.values or {}
        for body in range(builder.body_count):
            gravcomp_body.values[body] = 1.0

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

        self.cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.cube_start_pos), wp.quat_identity()),
            mass=0.2,
            label="non_touching_object",
        )
        object_cfg = newton.ModelBuilder.ShapeConfig(density=250.0, margin=0.0, ke=5.0e4, kd=1.0e2, mu=0.6)
        builder.add_shape_box(
            body=self.cube_body,
            hx=self.cube_half,
            hy=self.cube_half,
            hz=self.cube_half,
            cfg=object_cfg,
            color=wp.vec3(0.18, 0.42, 0.95),
            label="non_touching_object_shape",
        )

        return builder

    def _build_cable_builder(self, handle_pos: np.ndarray) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.mu = 0.8

        hidden_table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=False,
            has_shape_collision=True,
            has_particle_collision=False,
            ke=5.0e4,
            kd=1.0e2,
            mu=0.8,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half[0]),
            hy=float(self.table_half[1]),
            hz=float(self.table_half[2]),
            cfg=hidden_table_cfg,
            label="hidden_cable_table_collider",
        )

        direction = np.array([1.0, 0.05, -0.03], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        segment_length = 0.035
        node_count = 15
        positions = [wp.vec3(*(handle_pos + direction * segment_length * i)) for i in range(node_count)]

        cable_cfg = newton.ModelBuilder.ShapeConfig(density=80.0, margin=0.001, ke=2.0e4, kd=20.0, mu=0.7)
        self.cable_bodies, self.cable_joints = builder.add_rod(
            positions=positions,
            radius=0.008,
            cfg=cable_cfg,
            stretch_stiffness=2.5e4,
            stretch_damping=0.05,
            bend_stiffness=1.5e1,
            bend_damping=0.02,
            label="vbd_cable",
            wrap_in_articulation=True,
        )

        builder.color(balance_colors=False)
        return builder

    def _end_effector_position(self) -> np.ndarray:
        body_q = self.rigid_state_0.body_q.numpy()[self.ee_body]
        pos = body_q[:3]
        quat = body_q[3:7]
        return pos + _quat_rotate_xyzw(quat, self.ee_offset)

    def _set_robot_targets(self, t: float) -> None:
        phase = 2.0 * math.pi * 0.25 * t
        q = self.home_q.copy()
        q[0] += 0.55 * math.sin(phase)
        q[3] += 0.18 * math.sin(phase + 0.35)
        q[5] -= 0.20 * math.sin(phase)
        q[7] = 0.0
        q[8] = 0.0

        target = self.rigid_control.joint_target_q.numpy()
        target[:9] = q
        self.rigid_control.joint_target_q.assign(target)

    def _sync_viz_state(self) -> None:
        body_q = self.viz_state.body_q.numpy()
        body_qd = self.viz_state.body_qd.numpy()
        body_q[: self.rigid_body_count] = self.rigid_state_0.body_q.numpy()
        body_qd[: self.rigid_body_count] = self.rigid_state_0.body_qd.numpy()
        start = self.rigid_body_count
        end = start + self.cable_body_count
        body_q[start:end] = self.cable_state_0.body_q.numpy()
        body_qd[start:end] = self.cable_state_0.body_qd.numpy()
        self.viz_state.body_q.assign(body_q)
        self.viz_state.body_qd.assign(body_qd)

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._set_robot_targets(t)

            self.rigid_state_0.clear_forces()
            self.rigid_solver.step(self.rigid_state_0, self.rigid_state_1, self.rigid_control, None, self.sim_dt)
            self.rigid_state_0, self.rigid_state_1 = self.rigid_state_1, self.rigid_state_0

            self.cable_state_0.clear_forces()
            self.cable_model.collide(self.cable_state_0, self.cable_contacts)
            self.cable_solver.step(
                self.cable_state_0,
                self.cable_state_1,
                self.cable_control,
                self.cable_contacts,
                self.sim_dt,
            )
            self.cable_state_0, self.cable_state_1 = self.cable_state_1, self.cable_state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.rigid_state_0.body_q.numpy()
        cable_q = self.cable_state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(cable_q)):
            raise ValueError("Non-finite body transform detected.")

        cube = body_q[self.cube_body, :3]
        if cube[2] - self.cube_half > self.table_top_z + 0.03:
            raise ValueError("The cube is still floating above the table.")
        if cube[2] < self.table_top_z:
            raise ValueError("The cube fell through the table.")
        if np.min(np.linalg.norm(cable_q[:, :3] - cube[None, :], axis=1)) < 0.12:
            raise ValueError("The non-touching rigid object came too close to the cable.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "minimal_cable_franka.usd"), num_frames=240)
        parser.add_argument("--substeps", type=int, default=8, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=8, help="VBD iterations for the cable.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="minimal_cable_franka")
    examples.run(Example(viewer, args), args)
