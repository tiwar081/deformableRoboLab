"""DATA FILE — cloth_franka: ONE hot-dog fold of the shirt, force-grip. The shirt drops inflated,
settles flat and centred on the shared table; the Franka pinches the middle of its +x edge region,
lifts, and folds it in half across the fold line along y (the +x edge lands on the -x edge), then
releases and retracts. Standard dynamic gripper-PROXY bridge (two-way coupling, default pads).

GRIP — force windows (GraspWindow): the centralized admittance GripController drives the fingers
from the harvested squeeze. ``force_target=2 N`` sits INSIDE the shell's achievable squeeze range
(~0.3–3 N), so under the target-relative law (GripConfig.window_params: gain ∝ 1/target, deadband
∝ target) the regulator latches at the 0.3 N engage floor, tightens briskly (~2.4 mm/s at full-scale
error), and CONVERGES to a true force equilibrium near the proven ~8–9 mm jaw — the regulator itself
stops the close and stays live to re-tighten a shedding pinch. (History: under the old fixed-constant
law a ≤2 N target was literally dead — the 2 N absolute deadband exceeded the whole error range — and
the workaround was an unreachable 8 N target creeping shut forever; see docs/gripper.md.) The controller must stop the close itself
(a commanded zero-gap pinch expels a thin shell; docs/cloths.md).

GRASP RECIPE (Newton's, docs/cloths.md): descend the fingertip COMMANDED 5 mm BELOW the table top
(the stopper converts the excess into pressing normal force — the mu*N anchor that makes the pinch
work at PHYSICAL per-object friction), straight-down pinch, lift, drag at ~TT+0.11. (Tilt 0 everywhere: Newton's right-side folds grasp untilted, and
the tilted orientation is q6-limit-bound for the panda past x≈0 — a mid-drag tilt blend twisted the
marginal pinch and helped shed it.) Grasping is pure frictional contact; the cloth is the central
SI-converted Newton cloth (ClothConfig defaults) — the demo declares NO physics."""
import math

from deformableManipulationTools import TABLE, GraspWindow
from deformableManipulationTools.assets import ClothConfig
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

CLOTH = ClothConfig(flatten_z=1.0)               # the canonical SI-Newton cloth, dropped inflated
CLOTH_YAW = math.pi                              # Newton's shirt orientation (180° about z)
CLOTH_POS = (0.125, -0.527, TABLE.top_z + 0.20)  # settled footprint centres on the table; RAISED so the
                                                 # inflated shell starts clear of table AND parked arm
TT = TABLE.top_z
P0 = (0.10, -0.45, TT + 0.48)                  # frame 0: START up & clear of the inflated shirt
P1 = (0.10, -0.45, TT + 0.25)                  # hover while the shirt drops + settles

GRAB = (0.35, -0.45)                           # middle of the shirt's +x edge region (~11 cm inside;
                                               # x=0.37 untilted is q6-limit-bound: 20 mm IK miss)
DROP = (-0.11, -0.45)                          # GRAB mirrored across the shirt centre (x=0.12): the
                                               # +x edge lands on the -x edge — a fold in half
SURF = TT - 0.005                              # fingertip COMMANDED 5 mm below the table top
                                               # (Newton's recipe does the same): the robot-side
                                               # stopper holds the fingers AT the surface and the
                                               # excess becomes pressing normal force — anchoring
                                               # is mu*N, so N compensates the physical (lower)
                                               # cloth<->table friction of the realistic material
                                               # set (no shape-mu stamp anymore)
UP = TT + 0.11                                 # lift/drag height (Newton's)
HI = TT + 0.15                                 # approach height

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("cloth", CLOTH, pos=CLOTH_POS, yaw=CLOTH_YAW),
        Obj("proxies"),                                    # standard dynamic-proxy grip (two-way)
    ],
    waypoints=[
        WP(0.0, P0),                                   # start up & clear
        WP(0.5, P1),                                   # come down to a hover
        WP(4.0, P1),                                   # hold while the shirt drops + settles
        WP(6.0, (GRAB[0], GRAB[1], HI)),               # approach above the +x edge middle
        WP(8.5, (GRAB[0], GRAB[1], SURF)),             # descend fingertip to the table top
        WP(14.0, (GRAB[0], GRAB[1], SURF)),            # dwell while the force grip closes, latches,
                                                       # and the regulator tightens toward target
        WP(15.5, (GRAB[0], GRAB[1], UP)),              # lift
        # drag across — the fold. Via-points (via=True) keep the TCP on the straight Cartesian
        # line at CONSTANT velocity: our executor blends joint keyframes, so one 50 cm segment
        # would bow the arm ~10 cm downward mid-drag (measured), while easing at every via-point
        # would pulse/pause the arm at each one.
        WP(16.25, (0.235, GRAB[1], UP), via=True),
        WP(17.0, (0.12, GRAB[1], UP), via=True),
        WP(17.75, (0.005, GRAB[1], UP), via=True),
        WP(18.5, (DROP[0], DROP[1], UP)),
        WP(19.5, (DROP[0], DROP[1], UP)),              # hold; release window opens
        WP(21.0, (-0.015, -0.45, TT + 0.25), via=True),  # retract via-point (same reason)
        WP(22.0, P1),                                  # retract
    ],
    grasp_windows=[GraspWindow(close_start=8.5, close_end=10.5,
                               release_start=19.5, release_end=20.5,
                               force_target=2.0)],     # inside the shell's achievable squeeze -> the
                                                       # regulator converges to a stable ~8-9 mm pinch
    start_at_first_waypoint=True,                      # frame 0 = P0 (up), not home_q inside the shirt
    coupling_soft_ke=CLOTH.soft_contact_ke,            # enable the harvest; framework derives the
                                                       # effective ke centrally (harmonic, 30 N/m)
    object_solver_kwargs={"rigid_body_contact_buffer_size": 4096,
                          "rigid_body_particle_contact_buffer_size": 65536},
    object_pipeline_kwargs={"soft_contact_margin": CLOTH.contact_margin},
    substeps=10, vbd_iterations=5, num_frames=int(23.0 * 60),  # Newton's dt (10 @ 60 fps) — part of
                                                       # the dimensionless contact parity (docs/cloths.md)
)
