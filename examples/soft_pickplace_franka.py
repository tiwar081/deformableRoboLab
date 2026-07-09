"""DATA FILE — soft_pickplace_franka: Franka picks a soft FEM block off the table and places it at a
target. Declares only the scene + policy; example.py + deformableManipulationTools/demo_runner.py
supply everything else (robot, solver, force grip, contact materials, the per-substep executor)."""
from deformableManipulationTools import SOFT_BLOCK, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

TT = TABLE.top_z
BLOCK_HALF = 0.5 * SOFT_BLOCK.dim[0] * SOFT_BLOCK.cell
PICK = (0.10, -0.50)
PLACE = (0.34, -0.28)
GRASP_Z = TT + BLOCK_HALF
LIFT_Z = TT + 0.16

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("soft_block", SOFT_BLOCK, pos=(PICK[0], PICK[1], TT)),
        Obj("proxies"),
    ],
    waypoints=[
        WP(0.0),                                          # home
        WP(2.2, (PICK[0], PICK[1], TT + 0.12)),           # pregrasp (above pick)
        WP(3.2, (PICK[0], PICK[1], GRASP_Z)),             # descend to pick
        WP(4.4, (PICK[0], PICK[1], GRASP_Z)),             # hold (close)
        WP(5.2, (PICK[0], PICK[1], LIFT_Z)),              # lift
        WP(6.4, (PLACE[0], PLACE[1], LIFT_Z)),            # over place
        WP(7.4, (PLACE[0], PLACE[1], GRASP_Z)),           # lower to place
        WP(8.6, (PLACE[0], PLACE[1], GRASP_Z)),           # hold (release)
        WP(9.2, (PLACE[0], PLACE[1], LIFT_Z)),            # retreat up
        WP(11.2, None),                                   # home
    ],
    grasp_windows=[GraspWindow(close_start=3.2, close_end=4.4, release_start=8.0,
                               release_end=8.6, force_target=3.0)],
    coupling_soft_ke=SOFT_BLOCK.soft_contact_ke,
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 4096},
    object_pipeline_kwargs={"soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=720,
)
