"""DATA FILE — cloth_franka: Newton's `example_cloth_franka` shirt-folding sequence (same timeline,
same four folds, same shirt regions grasped/released) through the standard dynamic gripper-PROXY
bridge (two-way coupling, EE feedback ON — no hacks). The scene runs on the ONE table every demo
shares, so Newton's pose offsets are applied about the shirt centre at SCALE=0.8 (in-spirit mapping;
the exact world points would put the far grasps out of the arm's reach envelope from the shared
base placement). The kinematic-finger twin is cloth_franka_hack; the box-slice-pad
limitation twin is cloth_franka_sliceProxies.

MAPPING (verified numerically; see docs/cloths.md):
  * Poses: p = shirt_centre + 0.8 * (p_newton - newton_shirt_centre) in xy (same shirt regions),
    z -> TT + (z_newton - 20 cm) (table-relative, fingertip to the table top on grasps).
  * Our home EE orientation IS Newton's grasp quat (0.9239,-0.3827,0,0) — measured identical, closing
    axis pure world-y — so Newton's two orientations map to tilt=0 and tilt=-45° about world y
    (its "left" grasps: q_left = rot(y,-45°) ∘ q_right, decomposed exactly).
  * The shirt drops so its SETTLED footprint centres on the shared table, RAISED (TT+0.20) so it
    starts fully clear of the table and the parked arm (Newton's own init spawns 10 cm through
    the table), then falls and settles into the same flat state.
  * Fingers: Newton's activations — open 0.8 -> 3.2 cm/finger, close 0.1 -> 0.4 cm/finger (8 mm
    total jaw gap, NEVER 0 — a zero-gap penalty pinch expels the shell).

SEQUENCE (Newton's `robot_key_poses`, 70.5 s): descend → fold the shirt's four regions in turn
(top-left, bottom-left, top-right, bottom-right — each: descend to the table top, close to 8 mm,
lift, drag, release) → final bottom grasp dragged toward the robot → retreat. Grasping is pure
frictional contact (μ_eff ≈ 0.61); the cloth is the central SI-converted Newton cloth (ClothConfig
defaults) — the demo declares NO physics."""
import math

from deformableManipulationTools import FRANKA, TABLE
from deformableManipulationTools.assets import ClothConfig
from deformableManipulationTools.demo_runner import DemoSpec, Obj, WP

CLOTH = ClothConfig(flatten_z=1.0)             # the canonical SI-Newton cloth, dropped inflated
CLOTH_YAW = math.pi                            # Newton rotates the shirt 180° about z
CLOTH_POS = (0.125, -0.527, TABLE.top_z + 0.20)  # drop so the SETTLED footprint centres on the shared
                                                 # table (the mesh mean sits +7.7 cm in y of the footprint
                                                 # centre), RAISED so the inflated shell starts fully clear
                                                 # of the table AND the parked arm (hem 5.5 cm above the top)
TT = TABLE.top_z
OPEN = 0.032                                   # Newton's open activation: 0.8 -> 3.2 cm per finger
SHUT = 0.004                                   # Newton's close: 0.4 cm per finger -> 8 mm jaw gap
P0 = (0.10, -0.45, TT + 0.48)                  # frame 0: START up & clear of the inflated shirt
TILT_L = -0.7854                               # Newton's "left" grasp orientation: 45° about world -y
SCALE = 0.8                                    # in-spirit pose mapping: Newton's offsets about its shirt
                                               # centre, scaled 0.8 so every pose stays reachable with the
                                               # shirt centred on the SHARED default table (same shirt
                                               # regions grasped/released, not the exact world points)

# Newton's robot_key_poses, verbatim: (duration [s], x, y, z [cm], tilted?, closed?).
ROWS = [
    (4.0, 31, -60, 40.0, True, False),         # descend to working height
    (2.0, 31, -60, 20.0, True, False),         # --- top left: approach at the table top
    (2.0, 31, -60, 20.0, True, True),          #     close to 8 mm
    (2.0, 26, -60, 26.0, True, True),          #     lift
    (2.0, 12, -60, 31.0, True, True),          #     drag
    (3.0, -6, -60, 31.0, True, True),          #     drag
    (1.0, -6, -60, 31.0, True, False),         #     release
    (2.0, 15, -33, 31.0, True, False),         # --- bottom left
    (3.0, 15, -33, 21.0, True, False),
    (3.0, 15, -33, 21.0, True, True),
    (2.0, 15, -33, 28.0, True, True),
    (3.0, -2, -33, 28.0, True, True),
    (1.0, -2, -33, 28.0, True, False),
    (2.0, -28, -60, 28.0, False, False),       # --- top right (straight-down orientation)
    (2.0, -28, -60, 20.0, False, False),
    (2.0, -28, -60, 20.0, False, True),
    (2.0, -18, -60, 31.0, False, True),
    (3.0, 5, -60, 31.0, False, True),
    (1.0, 5, -60, 31.0, False, False),
    (3.0, -18, -30, 20.5, False, False),       # --- bottom right (low sweep approach)
    (3.0, -18, -30, 20.5, False, True),
    (2.0, -3, -30, 31.0, False, True),
    (3.0, -3, -30, 31.0, False, True),
    (2.0, -3, -30, 31.0, False, False),
    (2.0, 0, -20, 30.0, False, False),         # --- bottom: drag toward the robot
    (2.0, 0, -20, 19.5, False, False),
    (2.0, 0, -20, 19.5, False, True),
    (2.0, 0, -20, 35.0, False, True),
    (1.0, 0, -30, 35.0, False, True),
    (1.5, 0, -30, 35.0, False, True),
    (1.5, 0, -40, 35.0, False, True),
    (1.5, 0, -40, 35.0, False, False),         #     release
    (2.0, -28, -60, 28.0, False, False),       # retreat
]


def _newton_policy():
    """Newton's key poses -> (waypoints, finger_schedule): cm -> m, +5 cm base offset, z table-relative;
    the finger keyframes blend each open/close over its row like Newton's per-row activation target."""
    wps, fingers, t = [WP(0.0, P0)], [(0.0, OPEN)], 0.0
    closed = False
    for dur, x, y, z, tilted, close in ROWS:
        pos = (0.12 + SCALE * 0.01 * x, -0.05 + SCALE * 0.01 * y, TT + 0.01 * (z - 20.0))
        if close != closed:
            fingers += [(t, SHUT if closed else OPEN), (t + dur, SHUT if close else OPEN)]
            closed = close
        t += dur
        wps.append(WP(t, pos, tilt=(TILT_L if tilted else 0.0), tilt_axis=(0.0, 1.0, 0.0)))
    return wps, fingers


_WAYPOINTS, _FINGERS = _newton_policy()

DEMO = DemoSpec(
    scene=[
        Obj("table", TABLE),
        Obj("cloth", CLOTH, pos=CLOTH_POS, yaw=CLOTH_YAW),
        Obj("proxies"),                                    # standard dynamic-proxy grip (two-way)
    ],
    waypoints=_WAYPOINTS,
    finger_schedule=_FINGERS,
    start_at_first_waypoint=True,                      # frame 0 = P0 (up), not home_q inside the shirt
    coupling_soft_ke=CLOTH.soft_contact_ke,            # enable the harvest; framework averages in the
                                                       # shape side centrally (ke_eff = 30 N/m)
    object_solver_kwargs={"rigid_body_contact_buffer_size": 4096,
                          "rigid_body_particle_contact_buffer_size": 65536},
    object_pipeline_kwargs={"soft_contact_margin": CLOTH.contact_margin},
    substeps=10, vbd_iterations=5, num_frames=int(71.0 * 60),  # Newton's dt (10 @ 60 fps) — part of
                                                       # the dimensionless contact parity (docs/cloths.md)
)
