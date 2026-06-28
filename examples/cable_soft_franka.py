"""DATA FILE — cable_soft_franka: grasp a cable, lift it, sweep it side to side past a soft block."""
from deformableManipulationTools import SOFT_BLOCK, CABLE, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP, Sweep, cable_layout

TT = TABLE.top_z
SOFT = (0.28, -0.30)
nodes = lambda ctx: cable_layout(ctx.home_tcp)[0]    # cable laid from the home TCP (clamped to table)
grasp = lambda ctx: cable_layout(ctx.home_tcp)[1]    # midpoint of nodes 3-4

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("soft_block", SOFT_BLOCK, pos=(SOFT[0], SOFT[1], TT)),
        Obj("proxies"),
        Obj("cable", CABLE, pos=nodes),
    ],
    waypoints=[
        WP(0.0),
        WP(2.8, grasp),     # close on the cable
        WP(4.8, grasp),     # hold (lift)
        WP(6.8, None),      # back to home, then sweep
    ],
    grasp_windows=[GraspWindow(close_start=2.8, close_end=4.6)],   # default 30 N
    sweep=Sweep(start=6.8, freq=0.18, ramp=1.5, terms=[(0, 0.55, 0.0), (3, 0.18, 0.35), (5, -0.20, 0.0)]),
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 512},
    object_pipeline_kwargs={"soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=720,
)
