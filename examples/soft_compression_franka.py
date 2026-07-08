"""DATA FILE — soft_compression_franka: grasp the plate handle, carry it over a soft block, drop to press."""
from deformableManipulationTools import SOFT_BLOCK, PLATE, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

TT = TABLE.top_z
SHEET_Z = TT + PLATE.sheet_half[2]
HANDLE_Z = SHEET_Z + PLATE.handle_local_pos[2]   # grasp the handle top
SHEET = (0.10, -0.55)
SOFT = (0.28, -0.30)
DROP = (SOFT[0] + 0.5 * SOFT_BLOCK.dim[0] * SOFT_BLOCK.cell, SOFT[1])

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("soft_block", SOFT_BLOCK, pos=(SOFT[0], SOFT[1], TT)),
        Obj("proxies"),
        Obj("plate", PLATE, pos=(SHEET[0], SHEET[1], SHEET_Z)),
    ],
    waypoints=[
        WP(0.0),
        WP(2.2, (SHEET[0], SHEET[1], TT + 0.16)),    # pregrasp above the handle
        WP(3.2, (SHEET[0], SHEET[1], HANDLE_Z)),     # grasp the handle
        WP(5.0, (SHEET[0], SHEET[1], HANDLE_Z)),     # hold
        WP(7.0, (DROP[0], DROP[1], TT + 0.19)),      # over the block
        WP(8.6, (DROP[0], DROP[1], TT + 0.19)),      # hold (release)
        WP(10.6, None),
    ],
    grasp_windows=[GraspWindow(close_start=3.2, close_end=4.4, release_start=8.0,
                               release_end=8.6, force_target=30.0)],
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 4096},
    object_pipeline_kwargs={"soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=720,
)
