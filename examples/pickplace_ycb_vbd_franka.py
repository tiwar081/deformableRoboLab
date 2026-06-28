"""DATA FILE — pickplace_ycb_vbd_franka: pick a rubik's cube then a banana into a bowl. A token soft
cube routes the workspace to the VBD path (force-grip via proxies). A/B twin of pickplace_ycb_franka."""
import numpy as np
import warp as wp

from deformableManipulationTools import (TABLE_YCB, RUBIKS_CUBE, BOWL_YCB, BANANA_YCB, SOFT_BLOCK,
                                          GraspWindow, OBJECTS_DIR, load_usd_mesh)
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

TT = TABLE_YCB.top_z
CH = RUBIKS_CUBE.half_extent
CUBE_XY = (0.431, -0.097)
BANANA_XY = (0.539, -0.076)
TRAY_XY = (0.443, 0.127)
SOFT_XY = (0.55, -0.28)
DROP_Z = TT + 0.16
YAW = np.pi / 2

# Banana grasp: a quarter down from one end, at the centreline; yaw 90 closes across its narrow width.
_bv = np.asarray(load_usd_mesh(OBJECTS_DIR / BANANA_YCB.usd_subpath).vertices)
_gy = _bv[:, 1].min() + 0.25 * (_bv[:, 1].max() - _bv[:, 1].min())
_band = np.abs(_bv[:, 1] - _gy) < 0.012
_gx = float(_bv[_band, 0].mean())
BGRASP = (BANANA_XY[0] + _gx, BANANA_XY[1] + _gy)
_TRAY32 = np.array([TRAY_XY[0], TRAY_XY[1]], np.float32)   # match the demo's float32 tray_pos rounding
CUBE_DROP = (_TRAY32[0] + 0.06, _TRAY32[1])

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE_YCB),
        Obj("ycb_mesh", BOWL_YCB, pos=(TRAY_XY[0], TRAY_XY[1], 0.0), rest_on_z=True, mass=BOWL_YCB.target_mass),
        Obj("proxies"),
        Obj("rubiks_cube", RUBIKS_CUBE, pos=(CUBE_XY[0], CUBE_XY[1], TT + CH)),
        Obj("ycb_mesh", BANANA_YCB, pos=(BANANA_XY[0], BANANA_XY[1], TT + 0.02), mass=BANANA_YCB.target_mass),
        Obj("soft_block", SOFT_BLOCK, pos=(SOFT_XY[0], SOFT_XY[1], TT)),     # token deformable -> VBD routing
    ],
    waypoints=[
        WP(0.0),
        WP(1.6, (CUBE_XY[0], CUBE_XY[1], TT + 0.14)),     # a_pre
        WP(2.6, (CUBE_XY[0], CUBE_XY[1], TT + CH)),        # a_grasp
        WP(4.0, (CUBE_XY[0], CUBE_XY[1], TT + CH)),        # hold
        WP(4.8, (CUBE_XY[0], CUBE_XY[1], DROP_Z)),         # a_lift
        WP(6.0, (CUBE_DROP[0], CUBE_DROP[1], DROP_Z)),     # a_drop (over bowl)
        WP(7.0, (CUBE_DROP[0], CUBE_DROP[1], DROP_Z)),     # hold (release)
        WP(8.6, (BGRASP[0], BGRASP[1], TT + 0.14), yaw=YAW),   # b_pre
        WP(10.2, (BGRASP[0], BGRASP[1], TT + 0.01), yaw=YAW),  # b_grasp
        WP(12.2, (BGRASP[0], BGRASP[1], TT + 0.01), yaw=YAW),  # hold
        WP(16.2, (BGRASP[0], BGRASP[1], DROP_Z), yaw=YAW),     # b_lift (slow)
        WP(17.8, (TRAY_XY[0], TRAY_XY[1], DROP_Z), yaw=YAW),   # b_drop (over bowl)
        WP(18.6, (TRAY_XY[0], TRAY_XY[1], DROP_Z), yaw=YAW),   # hold (release)
        WP(20.6, None),
    ],
    grasp_windows=[
        GraspWindow(close_start=2.6, close_end=3.8, release_start=6.2, release_end=6.8),
        GraspWindow(close_start=10.2, close_end=11.6, release_start=17.8, release_end=18.4, force_target=80.0),
    ],
    blanket_fill=False,
    robot_base_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
    robot_table=TABLE_YCB,
    camera=(wp.vec3(1.1, 0.55, 0.6), -25.0, -130.0),
    scenic_check_table=False,
    object_solver_kwargs={"rigid_body_contact_buffer_size": 16384,
                          "rigid_body_particle_contact_buffer_size": 4096},
    object_pipeline_kwargs={"rigid_contact_max": 100000, "max_triangle_pairs": 1_048_576,
                            "reduce_contacts": True, "soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=1320,
)
