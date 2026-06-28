"""DATA FILE — cable_rigidCube_franka: grasp a cable, lift it, sweep it side to side past a rigid cube."""
from deformableManipulationTools import RIGID_CUBE, CABLE, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP, Sweep, cable_layout

TT = TABLE.top_z
CH = RIGID_CUBE.half_extent
CUBE = (0.28, -0.30)
nodes = lambda ctx: cable_layout(ctx.home_tcp)[0]
grasp = lambda ctx: cable_layout(ctx.home_tcp)[1]

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("rigid_box", RIGID_CUBE, pos=(CUBE[0], CUBE[1], TT + CH), half=CH),
        Obj("proxies"),
        Obj("cable", CABLE, pos=nodes),
    ],
    waypoints=[
        WP(0.0),
        WP(2.8, grasp),
        WP(4.8, grasp),
        WP(6.8, None),
    ],
    grasp_windows=[GraspWindow(close_start=2.8, close_end=4.6, force_target=30.0)],
    sweep=Sweep(start=6.8, freq=0.18, ramp=1.5, terms=[(0, 0.55, 0.0), (3, 0.18, 0.35), (5, -0.20, 0.0)]),
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048},
    substeps=8, vbd_iterations=12, num_frames=720,
)
