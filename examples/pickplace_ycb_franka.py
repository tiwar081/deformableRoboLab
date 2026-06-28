"""DATA FILE — pickplace_ycb_franka: rigid-only workspace (objects merged into ONE SolverMuJoCo, true
two-way grasp). Pick a rubik's cube then a banana into a bowl; the gripper follows an explicit
open/close schedule and MuJoCo contact holds each object (no force controller / proxies)."""
import numpy as np
import warp as wp

from deformableManipulationTools import (FRANKA, MUJOCO_GRIP, TABLE_YCB, RUBIKS_CUBE, BOWL_YCB,
                                          BANANA_YCB, OBJECTS_DIR, load_usd_mesh)
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

TT = TABLE_YCB.top_z
CH = RUBIKS_CUBE.half_extent
CUBE_XY = (0.431, -0.097)
BANANA_XY = (0.539, -0.076)
TRAY_XY = (0.443, 0.127)
DROP_Z = TT + 0.16
YAW = np.pi / 2
OPEN, CLOSED = FRANKA.gripper_open, MUJOCO_GRIP.close_target

_bv = np.asarray(load_usd_mesh(OBJECTS_DIR / BANANA_YCB.usd_subpath).vertices)
_gy = _bv[:, 1].min() + 0.25 * (_bv[:, 1].max() - _bv[:, 1].min())
_band = np.abs(_bv[:, 1] - _gy) < 0.012
BGRASP = (BANANA_XY[0] + float(_bv[_band, 0].mean()), BANANA_XY[1] + _gy)
_TRAY32 = np.array([TRAY_XY[0], TRAY_XY[1]], np.float32)   # match the demo's float32 tray_pos rounding
CUBE_DROP = (_TRAY32[0] + 0.06, _TRAY32[1])

DEMO = DemoSpec(
    scene=[   # rigid-only: NO object-side table (objects rest on the robot-side TABLE_YCB) and NO proxies
        Obj("ycb_mesh", BOWL_YCB, pos=(TRAY_XY[0], TRAY_XY[1], 0.0), rest_on_z=True, mass=BOWL_YCB.target_mass),
        Obj("rubiks_cube", RUBIKS_CUBE, pos=(CUBE_XY[0], CUBE_XY[1], TT + CH)),
        Obj("ycb_mesh", BANANA_YCB, pos=(BANANA_XY[0], BANANA_XY[1], TT + 0.02), mass=BANANA_YCB.target_mass),
    ],
    waypoints=[
        WP(0.0),
        WP(1.6, (CUBE_XY[0], CUBE_XY[1], TT + 0.14)),
        WP(2.6, (CUBE_XY[0], CUBE_XY[1], TT + CH)),
        WP(4.0, (CUBE_XY[0], CUBE_XY[1], TT + CH)),
        WP(4.8, (CUBE_XY[0], CUBE_XY[1], DROP_Z)),
        WP(6.0, (CUBE_DROP[0], CUBE_DROP[1], DROP_Z)),
        WP(7.0, (CUBE_DROP[0], CUBE_DROP[1], DROP_Z)),
        WP(8.6, (BGRASP[0], BGRASP[1], TT + 0.14), yaw=YAW),
        WP(10.2, (BGRASP[0], BGRASP[1], TT + 0.01), yaw=YAW),
        WP(12.2, (BGRASP[0], BGRASP[1], TT + 0.01), yaw=YAW),
        WP(16.2, (BGRASP[0], BGRASP[1], DROP_Z), yaw=YAW),
        WP(17.8, (TRAY_XY[0], TRAY_XY[1], DROP_Z), yaw=YAW),
        WP(18.6, (TRAY_XY[0], TRAY_XY[1], DROP_Z), yaw=YAW),
        WP(20.6, None),
    ],
    finger_schedule=[   # explicit open/close; MuJoCo contact holds the object, opening the pads drops it
        (0.0, OPEN), (2.6, OPEN), (3.0, CLOSED), (6.2, CLOSED), (6.8, OPEN),
        (10.2, OPEN), (10.8, CLOSED), (17.8, CLOSED), (18.4, OPEN),
    ],
    blanket_fill=False,
    robot_base_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
    robot_table=TABLE_YCB,
    camera=(wp.vec3(1.1, 0.55, 0.6), -25.0, -130.0),
    scenic_check_table=False,
    robot_contact_max=32768,
    substeps=16, vbd_iterations=12, num_frames=1320,
)
