"""DATA FILE — rigidCube_soft_franka: pick a rigid cube, carry it over a soft block, drop it to squash."""
from deformableManipulationTools import SOFT_BLOCK, RIGID_CUBE, TABLE, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP
from robolabViz import ObjectStyle, RenderSpec

TT = TABLE.top_z
CH = RIGID_CUBE.half_extent
CUBE = (0.10, -0.55)
SOFT = (0.28, -0.30)
DROP = (SOFT[0] + 0.5 * SOFT_BLOCK.dim[0] * SOFT_BLOCK.cell, SOFT[1])   # soft_start + drop offset

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("soft_block", SOFT_BLOCK, pos=(SOFT[0], SOFT[1], TT)),
        Obj("proxies"),
        Obj("rigid_box", RIGID_CUBE, pos=(CUBE[0], CUBE[1], TT + CH), half=CH),
    ],
    waypoints=[
        WP(0.0),
        WP(2.2, (CUBE[0], CUBE[1], TT + 0.12)),     # pregrasp
        WP(3.2, (CUBE[0], CUBE[1], TT + CH)),        # pickup
        WP(5.0, (CUBE[0], CUBE[1], TT + CH)),        # hold (carry begins)
        WP(7.0, (DROP[0], DROP[1], TT + 0.19)),      # over the block
        WP(8.6, (DROP[0], DROP[1], TT + 0.19)),      # hold (release)
        WP(10.6, None),                              # home
    ],
    grasp_windows=[GraspWindow(close_start=3.2, close_end=4.4, release_start=8.0, release_end=8.6)],
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 4096},
    object_pipeline_kwargs={"soft_contact_margin": SOFT_BLOCK.contact_margin},
    substeps=16, vbd_iterations=12, num_frames=720,
    # Per-demo render look (exemplar): matte black work table in a garage, terracotta
    # soft block, glossy blue cube. CLI --table/--background still override.
    render=RenderSpec(
        table="black",
        background="garage",
        soft_body_style=ObjectStyle(color=(0.83, 0.35, 0.20), roughness=0.85),
        object_styles={"cube": ObjectStyle(color=(0.16, 0.42, 0.85), roughness=0.35)},
    ),
)
