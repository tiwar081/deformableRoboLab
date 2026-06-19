"""The Franka FR3 + hand robot: builder, MuJoCo solver, gripper IK. All physical properties come
from :mod:`deformableManipulationTools.params` so the robot is identical across every demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils
from newton import JointTargetMode

from .params import FRANKA, RobotConfig, TableConfig, TABLE
from .mathutils import find_body, quat_to_vec4


def build_franka_robot(
    xform: wp.transform,
    table: TableConfig | None = TABLE,
    robot: RobotConfig = FRANKA,
) -> newton.ModelBuilder:
    """Franka FR3 + hand builder with the shared actuator/gravcomp config and an optional hidden
    robot-side table collider (stops the gripper at the table surface). Pass ``table=None`` for an
    example that places its own table."""
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


def add_robot_table_box(builder: newton.ModelBuilder, center_xy, top_z: float,
                        half=(0.35, 0.5, 0.025), robot: RobotConfig = FRANKA) -> int:
    """Add a hidden robot-side table collider at an arbitrary placement (for examples that don't
    use the shared :data:`TABLE`, e.g. the YCB demo's (0.45, 0) table)."""
    cfg = newton.ModelBuilder.ShapeConfig(margin=1e-3, density=1000.0, is_visible=False,
                                          ke=5e4, kd=5e2, mu=1.0)
    return builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(center_xy[0], center_xy[1], top_z - half[2]), wp.quat_identity()),
        hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label="robot_contact_table")


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
                     gripper_open: float, yaw: float = 0.0, robot: RobotConfig = FRANKA) -> np.ndarray:
    """IK the TCP (link7 + ee_offset) to ``target_pos`` keeping the home orientation (optionally
    yawed about world z); returns the 9-vector joint target with the fingers held open."""
    body_q = state.body_q.numpy()[ee_body]
    ee_tf = wp.transform(*body_q)
    rot = wp.transform_get_rotation(ee_tf)
    if yaw != 0.0:
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw) * rot
    target_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
    pos_obj = ik.IKObjectivePosition(
        link_index=ee_body, link_offset=wp.vec3(*ee_offset),
        target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=ee_body, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([quat_to_vec4(rot)], dtype=wp.vec4),
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
