"""Shared building blocks for the Franka grasp examples.

Centralizes the code that was copy-pasted across every example: small math helpers, the Franka
robot builder + MuJoCo solver, gripper IK, the dynamic finger-proxy construction, the viz model
build/sync, and the per-frame CUDA-graph capture + substep loop (``GraspExample``).

All physics constants come from :mod:`assets.params` so the robot and grip behave identically
across demos. Per-example code only provides the OBJECT (``_build_object_model``-style setup,
done in the subclass ``__init__``) and the MOTION (``_set_robot_targets``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils
from newton import JointTargetMode

from assets.params import FRANKA, GRIP, TABLE, RobotConfig, GripConfig, TableConfig


# ---------------------------------------------------------------------------------------------
# Small math helpers (were duplicated verbatim in every example).
# ---------------------------------------------------------------------------------------------
def quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def find_body(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.endswith(suffix):
            return i
    raise ValueError(f"Could not find body ending with {suffix!r}.")


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.func
def wp_smoothstep(x: float) -> float:
    x = wp.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quat_to_vec4(q) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


# ---------------------------------------------------------------------------------------------
# Robot.
# ---------------------------------------------------------------------------------------------
def build_franka_robot(
    xform: wp.transform,
    table: TableConfig | None = TABLE,
    robot: RobotConfig = FRANKA,
) -> newton.ModelBuilder:
    """Franka FR3 + hand builder with the shared actuator/gravcomp config and an optional hidden
    robot-side table collider (stops the gripper at the table surface)."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.005
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    asset_path = newton.utils.download_asset(robot.asset_name)
    builder.add_urdf(
        Path(asset_path) / robot.urdf_subpath,
        xform=xform,
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )

    home_q = list(robot.home_q)
    builder.joint_q[: robot.n_dof] = home_q
    builder.joint_target_q[: robot.n_dof] = home_q
    for dof in range(robot.n_dof):
        is_arm = dof < robot.n_arm_dof
        builder.joint_target_ke[dof] = robot.arm_target_ke if is_arm else robot.finger_target_ke
        builder.joint_target_kd[dof] = robot.arm_target_kd if is_arm else robot.finger_target_kd
        builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        builder.joint_effort_limit[dof] = robot.arm_effort if is_arm else robot.finger_effort
        builder.joint_armature[dof] = robot.armature

    gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_attr.values is None:
        gravcomp_attr.values = {}
    for dof in range(robot.n_arm_dof):
        gravcomp_attr.values[dof] = True
    gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body in range(2, builder.body_count):
        gravcomp_body.values[body] = 1.0

    if table is not None:
        table_cfg = newton.ModelBuilder.ShapeConfig(
            margin=table.robot_margin, density=table.robot_density, is_visible=False,
            ke=table.robot_ke, kd=table.robot_kd, mu=table.robot_mu,
        )
        builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(*table.pos), wp.quat_identity()),
            hx=float(table.half[0]), hy=float(table.half[1]), hz=float(table.half[2]),
            cfg=table_cfg, label="robot_contact_table",
        )
    return builder


def finger_body_indices(model_or_labels, robot: RobotConfig = FRANKA) -> list[int]:
    labels = list(model_or_labels.body_label) if hasattr(model_or_labels, "body_label") else list(model_or_labels)
    return [find_body(labels, robot.left_finger_suffix), find_body(labels, robot.right_finger_suffix)]


def make_robot_solver(model, contact_max: int, robot: RobotConfig = FRANKA):
    return newton.solvers.SolverMuJoCo(
        model, solver=robot.solver, integrator=robot.integrator,
        iterations=robot.solver_iterations, ls_iterations=robot.solver_ls_iterations,
        nconmax=contact_max, njmax=contact_max * 2, cone=robot.cone,
        impratio=robot.solver_impratio, use_mujoco_contacts=False,
    )


def solve_gripper_ik(model, state, ee_body: int, ee_offset: np.ndarray, target_pos: np.ndarray,
                     gripper_open: float, robot: RobotConfig = FRANKA) -> np.ndarray:
    """IK the TCP (link7 + ee_offset) to ``target_pos`` keeping the home orientation; returns the
    9-vector joint target with the fingers held open."""
    body_q = state.body_q.numpy()[ee_body]
    ee_tf = wp.transform(*body_q)
    target_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
    pos_obj = ik.IKObjectivePosition(
        link_index=ee_body, link_offset=wp.vec3(*ee_offset),
        target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=ee_body, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
    )
    joint_limits_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0,
    )
    solver = ik.IKSolver(
        model=model, n_problems=1, objectives=[pos_obj, rot_obj, joint_limits_obj],
        lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(target_q, target_q, iterations=64)
    q = np.array(target_q.numpy()[0, : robot.n_dof], dtype=np.float32)
    q[robot.n_arm_dof : robot.n_dof] = gripper_open
    return q


# ---------------------------------------------------------------------------------------------
# Dynamic finger proxies (the grip contact bridge — see grip_coupling.TwoWayProxyCoupling).
# ---------------------------------------------------------------------------------------------
def build_gripper_proxies(object_builder, robot_builder, finger_bodies: list[int],
                          object_table_shape: int | None, gap: float, grip: GripConfig = GRIP,
                          has_particle_collision: bool = False):
    """Add the two dynamic finite-mass finger proxies to ``object_builder`` (collision shapes
    copied from the robot fingers), filtered against the object-model table. Set
    ``has_particle_collision=True`` to also grip FEM-particle objects (soft blocks). Returns
    (proxy_bodies, proxy_shapes)."""
    proxy_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, is_visible=False, has_shape_collision=True,
        has_particle_collision=has_particle_collision,
        margin=grip.proxy_margin, gap=gap, ke=grip.proxy_ke, kd=grip.proxy_kd, mu=grip.proxy_mu,
    )
    proxy_bodies, proxy_shapes = [], []
    inertia = wp.mat33(grip.proxy_inertia, 0.0, 0.0, 0.0, grip.proxy_inertia, 0.0, 0.0, 0.0, grip.proxy_inertia)
    for label, finger_body in zip(("left_gripper_contact_proxy", "right_gripper_contact_proxy"),
                                  finger_bodies, strict=True):
        body = object_builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            is_kinematic=False, mass=grip.proxy_mass, inertia=inertia,
            com=wp.vec3(0.0, 0.0, 0.0), lock_inertia=True, label=label,
        )
        proxy_bodies.append(body)
        n = 0
        for shape_idx, shape_body in enumerate(robot_builder.shape_body):
            if shape_body != finger_body:
                continue
            if not (robot_builder.shape_flags[shape_idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            shape = object_builder.add_shape(
                body=body, type=robot_builder.shape_type[shape_idx],
                xform=robot_builder.shape_transform[shape_idx], cfg=proxy_cfg,
                scale=robot_builder.shape_scale[shape_idx], src=robot_builder.shape_source[shape_idx],
                label=f"{label}_shape_{n}",
            )
            proxy_shapes.append(shape)
            n += 1
        if n == 0:
            raise RuntimeError(f"No colliding shapes on Franka finger body {finger_body}.")
    # A dynamic proxy re-pinned against the static table resolves explosively; the robot-side
    # table already stops the real fingers, so this object-model contact is redundant.
    if object_table_shape is not None:
        for shape in proxy_shapes:
            object_builder.add_shape_collision_filter_pair(object_table_shape, shape)
    return proxy_bodies, proxy_shapes


def restore_proxy_materials(object_model, proxy_shapes: list[int], grip: GripConfig = GRIP):
    """Re-apply proxy contact material after any blanket ``shape_material_*.fill_`` (which would
    clobber it). Must run after finalize."""
    if not proxy_shapes:
        return
    mu = object_model.shape_material_mu.numpy()
    ke = object_model.shape_material_ke.numpy()
    kd = object_model.shape_material_kd.numpy()
    for shape in proxy_shapes:
        mu[shape] = grip.proxy_mu
        ke[shape] = grip.proxy_ke
        kd[shape] = grip.proxy_kd
    object_model.shape_material_mu.assign(mu)
    object_model.shape_material_ke.assign(ke)
    object_model.shape_material_kd.assign(kd)


# ---------------------------------------------------------------------------------------------
# Viz model (robot + object combined for rendering).
# ---------------------------------------------------------------------------------------------
def build_viz_model(robot_builder, object_builder, device):
    viz_builder = newton.ModelBuilder()
    viz_builder.add_builder(robot_builder)
    object_body_start = robot_builder.body_count
    viz_builder.add_builder(object_builder)
    return viz_builder.finalize(device=device), object_body_start


# ---------------------------------------------------------------------------------------------
# Base example: identical per-frame loop, CUDA-graph capture, viz sync.
# ---------------------------------------------------------------------------------------------
class GraspExample:
    """Per-frame backbone shared by every grasp example. The subclass ``__init__`` must build and
    set: the robot (``robot_model``, ``robot_state_0/1``, ``robot_control``, ``robot_solver``,
    ``robot_collision_pipeline``, ``robot_contacts``), the object (``object_model``,
    ``object_state_0/1``, ``object_control``, ``object_solver``, ``object_contacts``), the
    ``coupling`` (grip_coupling.TwoWayProxyCoupling), the viz (``viz_model``, ``viz_state``,
    ``viz_object_body_start``), ``robot_finger_bodies``, ``sim_substeps``, ``sim_dt``,
    ``frame_dt``, ``_t_frame``, ``_capture_enabled`` and ``has_particles``. The subclass
    implements ``_set_robot_targets(substep)`` and ``_sync_gripper_proxies()``.
    """

    # These are set by the subclass; declared for clarity.
    coupling = None
    has_particles: bool = True

    def _sync_gripper_proxies(self) -> None:
        self.coupling.sync_proxies(self.robot_state_0, self.object_state_0, self.object_state_1)

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            self._set_robot_targets(substep)

            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.coupling.apply_to_robot(self.robot_state_0)
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(self.robot_state_0, self.robot_state_1, self.robot_control,
                                   self.robot_contacts, self.sim_dt)
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

            self._sync_gripper_proxies()
            self.coupling.snapshot(self.object_state_0)

            self.object_state_0.clear_forces()
            self.object_model.collide(self.object_state_0, self.object_contacts)
            self.object_solver.step(self.object_state_0, self.object_state_1, self.object_control,
                                    self.object_contacts, self.sim_dt)
            self.object_state_0, self.object_state_1 = self.object_state_1, self.object_state_0

            self.coupling.harvest(self.object_state_0)

    def step(self) -> None:
        self._t_frame.assign(np.array([self.sim_time], dtype=np.float32))

        # Capture after one uncaptured warm-up frame so lazy allocations have happened.
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

    def _sync_viz_state(self) -> None:
        body_q = self.viz_state.body_q.numpy()
        body_qd = self.viz_state.body_qd.numpy()
        n = self.robot_model.body_count
        body_q[:n] = self.robot_state_0.body_q.numpy()
        body_qd[:n] = self.robot_state_0.body_qd.numpy()
        start = self.viz_object_body_start
        end = start + self.object_model.body_count
        body_q[start:end] = self.object_state_0.body_q.numpy()
        body_qd[start:end] = self.object_state_0.body_qd.numpy()
        self.viz_state.body_q.assign(body_q)
        self.viz_state.body_qd.assign(body_qd)
        # Soft bodies: the particles are part of the viz model; without this copy the viewer draws
        # the block frozen at its rest shape (#1 recurring soft-body bug).
        if self.has_particles:
            wp.copy(self.viz_state.particle_q, self.object_state_0.particle_q)
            wp.copy(self.viz_state.particle_qd, self.object_state_0.particle_qd)

    def render(self) -> None:
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()
