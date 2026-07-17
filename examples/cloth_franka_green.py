"""DATA FILE — cloth_franka_green: the ``cloth_franka`` hot-dog-fold policy
applied to RGBench's measured green knit T-shirt, with the grasp shifted down
onto the main torso.  The passive garment's geometry and measured material
values are the centralized ``GREEN_TSHIRT`` asset; this file declares only its
scene pose and policy.
"""
import math

from deformableManipulationTools import TABLE, GREEN_TSHIRT, GraspWindow
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

CLOTH = GREEN_TSHIRT
CLOTH_YAW = math.pi
CLOTH_POS = (0.125, -0.527, TABLE.top_z + 0.20)
TT = TABLE.top_z
P0 = (0.10, -0.45, TT + 0.48)  # frame 0: start high and clear, never at home_q inside the shirt
P1 = (0.10, -0.45, TT + 0.32)  # settling hover stays clear of the authored shirt geometry

GRAB = (0.31, -0.45)  # 4 cm down the torso from the collar-side grasp (yaw=pi => decreasing world x)
DROP = (-0.11, -0.45)
SURF = TT - 0.005
UP = TT + 0.11
HI = TT + 0.15

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("cloth", CLOTH, pos=CLOTH_POS, yaw=CLOTH_YAW),
        Obj("proxies"),
    ],
    waypoints=[
        WP(0.0, P0),
        WP(0.5, P1),
        WP(4.0, P1),
        WP(6.0, (GRAB[0], GRAB[1], HI)),
        WP(8.5, (GRAB[0], GRAB[1], SURF)),
        WP(14.0, (GRAB[0], GRAB[1], SURF)),
        WP(15.5, (GRAB[0], GRAB[1], UP)),
        WP(16.25, (0.235, GRAB[1], UP), via=True),
        WP(17.0, (0.12, GRAB[1], UP), via=True),
        WP(17.75, (0.005, GRAB[1], UP), via=True),
        WP(18.5, (DROP[0], DROP[1], UP)),
        WP(19.5, (DROP[0], DROP[1], UP)),
        WP(21.0, (-0.015, -0.45, TT + 0.25), via=True),
        WP(22.0, P1),
    ],
    grasp_windows=[GraspWindow(close_start=8.5, close_end=10.5,
                               release_start=19.5, release_end=20.5,
                               # 4 N, NOT the Newton shirt's 2 N: this mid-torso grasp wads the knit
                               # between the pads, and at 2 N the wad's ~0.3 N brush satisfies the
                               # engage threshold at a 5 cm jaw — the regulator then crawls and the
                               # lift leaves the shirt behind (measured 2026-07-16). At 4 N the
                               # approach pushes through the wad and converges to a stable ~5.6 mm
                               # pinch at ~4.5 N; at 6 N the jaw grinds to ~1 mm (zero-gap expulsion
                               # risk on a thin shell — docs/cloths.md).
                               force_target=4.0)],
    start_at_first_waypoint=True,
    coupling_soft_ke=CLOTH.soft_contact_ke,
    object_solver_kwargs={"rigid_body_contact_buffer_size": 4096,
                          "rigid_body_particle_contact_buffer_size": 65536},
    object_pipeline_kwargs={"soft_contact_margin": CLOTH.contact_margin},
    substeps=10, vbd_iterations=5, num_frames=int(23.0 * 60),
)
