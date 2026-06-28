"""DATA FILE — cloth_franka: Franka runs Newton's example_cloth_franka corner scoop on a flat T-shirt
(45 deg tilt; approach -> descend-to-surface -> close -> lift -> drag -> release), explicit fingers.
The cloth self-contact / damping is centralized (ClothConfig + framework); see docs/cloths.md."""
from deformableManipulationTools import FRANKA, TABLE
from deformableManipulationTools.assets import ClothConfig
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

CLOTH = ClothConfig()
TT = TABLE.top_z
OPEN = FRANKA.gripper_open
TILT, AXIS = 0.7854, (1.0, 0.0, 0.0)        # Newton's ~45 deg grasp tilt
CORNER = (0.10, -0.78)
ACROSS = (0.30, -0.55)
T = lambda xyz: WP(xyz[0], (xyz[1], xyz[2], xyz[3]), tilt=TILT, tilt_axis=AXIS)

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("cloth", CLOTH, pos=(0.10, -0.45, TT + 0.04)),
        Obj("proxies"),
    ],
    waypoints=[
        WP(0.0),
        T((2.8, CORNER[0], CORNER[1], TT + 0.15)),     # approach above the corner
        T((3.9, CORNER[0], CORNER[1], TT + 0.008)),    # descend to the table surface
        T((4.7, CORNER[0], CORNER[1], TT + 0.008)),    # close (scoop)
        T((6.2, CORNER[0], CORNER[1] + 0.10, TT + 0.12)),  # lift + translate
        T((7.8, ACROSS[0], ACROSS[1], TT + 0.12)),     # drag across
        T((8.6, ACROSS[0], ACROSS[1], TT + 0.12)),     # release
        T((9.8, ACROSS[0], ACROSS[1], TT + 0.18)),     # retract
    ],
    finger_schedule=[(0.0, OPEN), (2.8, OPEN), (3.9, OPEN), (4.7, 0.0),
                     (6.2, 0.0), (7.8, 0.0), (8.6, OPEN), (9.8, OPEN)],
    coupling_soft_ke=CLOTH.soft_contact_ke,
    object_solver_kwargs={"rigid_body_contact_buffer_size": 2048,
                          "rigid_body_particle_contact_buffer_size": 16384},
    object_pipeline_kwargs={"soft_contact_margin": CLOTH.contact_margin},
    substeps=16, vbd_iterations=6, num_frames=600,
)
