"""DATA FILE — soft_pickplace_franka: Franka picks a soft FEM block off the table and places it at a
target. Declares only the scene + policy; example.py + deformableManipulationTools/demo_runner.py
supply everything else (robot, solver, force grip, contact materials, the per-substep executor)."""
from deformableManipulationTools import SOFT_BLOCK, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

TT = TABLE.top_z
BLOCK_HALF = 0.5 * SOFT_BLOCK.dim[0] * SOFT_BLOCK.cell
PICK = (0.10, -0.50)
PLACE = (0.34, -0.28)
GRASP_Z = TT + BLOCK_HALF - 0.01   # pads 1 cm BELOW the berry's equator: the squeeze dents a
                                   # pocket that cups the fruit (form closure) — friction alone is
                                   # marginal at delicate forces (capacity ~2*0.63*squeeze vs 1.1 N
                                   # weight)
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
        WP(8.0, (PICK[0], PICK[1], GRASP_Z)),             # hold (close) — a 5.7 kPa berry yields
                                                          # slowly (~54 N/m secant through the
                                                          # tissue, jaw at ~3.7 mm/s): the squeeze
                                                          # needs ~4.4 s of dwell to build real
                                                          # hold margin before the lift (measured)
        WP(9.6, (PICK[0], PICK[1], LIFT_Z)),              # SLOW lift (1.6 s): the delicate-force
                                                          # hold margin is fractions of a newton —
                                                          # fast-lift dynamic load alone sheds it
        WP(10.8, (PLACE[0], PLACE[1], LIFT_Z)),           # over place
        WP(11.8, (PLACE[0], PLACE[1], GRASP_Z)),          # lower to place
        WP(13.0, (PLACE[0], PLACE[1], GRASP_Z)),          # hold (release)
        WP(13.6, (PLACE[0], PLACE[1], LIFT_Z)),           # retreat up
        WP(15.6, None),                                   # home
    ],
    grasp_windows=[GraspWindow(close_start=3.2, close_end=7.8, release_start=12.4,
                               release_end=13.0, force_target=3.5)],  # delicate berry (81 g, 0.80 N).
                                                                      # MEASURED sweep: sub-2 N targets
                                                                      # plateau near ~1-1.4 N of squeeze
                                                                      # (tissue pushback ~54 N/m secant)
                                                                      # — enough to LIFT, but the shallow
                                                                      # dent sheds at the first lateral
                                                                      # acceleration of the carry. 3.5 N
                                                                      # closes a deeper pocket that cups
                                                                      # the fruit through the whole
                                                                      # pick->place (peak force on the
                                                                      # fruit ~3.2 N, well under the
                                                                      # 5-10 N raspberry crush range).
    coupling_soft_ke=SOFT_BLOCK.soft_contact_ke,
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 4096},
    object_pipeline_kwargs={"soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=990,
)
