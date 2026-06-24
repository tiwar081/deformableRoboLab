"""deformableManipulationTools — the centralized physics for the Franka manipulation demos.

ALL shared physics lives here so the examples only describe a *scene* and a *policy*:

  * physics framework — :class:`framework.GraspExample` (build + per-substep loop + capture + viz)
  * robot             — :mod:`robot` (FR3 + hand builder, MuJoCo solver, gripper IK)
  * grip              — :mod:`grip` (dynamic finite-mass proxies + ``TwoWayProxyCoupling``)
  * asset properties  — :mod:`params` (frozen configs) + :mod:`assets` (object builders) +
                        :mod:`mesh_collision` (coacd convex-decomposition collision)

Centralization is a STANDING REQUIREMENT (see CLAUDE.md): new physics/robot/asset behavior goes
here, never inline in an example. Import the public API from the package root.
"""
from __future__ import annotations

# Parameters (single source of truth).
from .params import (
    RobotConfig, GripConfig, MujocoGripConfig, GraspWindow, TableConfig, CableConfig, SoftBlockConfig,
    RigidBoxConfig, PlateConfig, YcbMeshConfig,
    FRANKA, GRIP, MUJOCO_GRIP, TABLE, TABLE_YCB, CABLE,
    SOFT_BLOCK, RIGID_CUBE, PLATE, RUBIKS_CUBE, BOWL_YCB, BANANA_YCB,
)
# Math helpers.
from .mathutils import quat_rotate_xyzw, find_body, smoothstep, wp_smoothstep, quat_to_vec4
# Robot.
from .robot import (
    build_franka_robot, add_robot_table_box, make_robot_solver, solve_gripper_ik, finger_body_indices,
    set_mujoco_grip_controller,
)
# Grip (proxies + coupling).
from .grip import build_gripper_proxies, restore_proxy_materials, TwoWayProxyCoupling, GripController
# Asset builders.
from .assets import (
    add_table, add_soft_block, add_cable, add_rigid_box, add_plate, add_rubiks_cube, add_ycb_mesh,
    PARTICLE_SOLVER_KWARGS, OBJECTS_DIR,
)
# Mesh collision (coacd).
from .mesh_collision import (
    load_usd_mesh, decimate_mesh, convex_decompose, add_collision_pieces, add_visual_mesh, rescale_body_mass,
)
# Framework.
from .framework import GraspExample, build_viz_model

__all__ = [
    "RobotConfig", "GripConfig", "MujocoGripConfig", "GraspWindow", "TableConfig", "CableConfig",
    "SoftBlockConfig",
    "RigidBoxConfig", "PlateConfig", "YcbMeshConfig", "FRANKA", "GRIP", "MUJOCO_GRIP", "TABLE",
    "TABLE_YCB", "CABLE", "SOFT_BLOCK", "RIGID_CUBE", "PLATE", "RUBIKS_CUBE", "BOWL_YCB", "BANANA_YCB",
    "quat_rotate_xyzw", "find_body", "smoothstep", "wp_smoothstep", "quat_to_vec4",
    "build_franka_robot", "add_robot_table_box", "make_robot_solver", "solve_gripper_ik",
    "finger_body_indices", "set_mujoco_grip_controller",
    "build_gripper_proxies", "restore_proxy_materials", "TwoWayProxyCoupling", "GripController",
    "add_table", "add_soft_block", "add_cable", "add_rigid_box", "add_plate", "add_rubiks_cube",
    "add_ycb_mesh", "PARTICLE_SOLVER_KWARGS",
    "OBJECTS_DIR", "load_usd_mesh", "decimate_mesh", "convex_decompose", "add_collision_pieces",
    "add_visual_mesh", "rescale_body_mass", "GraspExample", "build_viz_model",
]
