"""DATA FILE — cloth_franka_full: Newton's `example_cloth_franka` FULL shirt-folding sequence (same
four folds, similar shirt regions grasped/released) through the standard dynamic gripper-PROXY
bridge. The long companion of `cloth_franka` (which does ONE force-grip hot-dog fold); this one
follows Newton IN SPIRIT, not exactly: the scene runs on the ONE table every demo shares, so
Newton's pose offsets are applied about the shirt centre at SCALE=0.8 (the exact world points would
put the far grasps out of the arm's reach envelope from the shared base placement), every grasp is
TOP-DOWN (Newton's 45°-tilted "left" wrist is replaced with the straight-down orientation), and the
transits are restructured so the gripper never plows the cloth (see SEQUENCE).

MAPPING (verified numerically; see docs/physicsEngine/cloths.md):
  * Poses: p = shirt_centre + 0.8 * (p_newton - newton_shirt_centre) in xy (same shirt regions),
    z -> TT + (z_newton - 20 cm) (table-relative, fingertip to the table top on grasps).
  * Our home EE orientation IS Newton's straight-down grasp quat (0.9239,-0.3827,0,0) — measured
    identical, closing axis pure world-y — used for EVERY pose (tilt=0 throughout).
  * The shirt drops so its SETTLED footprint centres on the shared table, RAISED (TT+0.20) so it
    starts fully clear of the table and the parked arm, then falls and settles flat.
  * Fingers: Newton's activations via an explicit schedule — open 0.8 -> 3.2 cm/finger, close
    0.1 -> 0.4 cm/finger (8 mm total jaw gap, NEVER 0 — a zero-gap penalty pinch expels the shell).

SEQUENCE (Newton's four folds — top-left, bottom-left, top-right, bottom-right — then the final
bottom drag, 76.0 s — the bottom hem is carried all the way to the shirt's TOP edge, a full half
fold). Every fold is the same clean cycle: GLIDE empty at cruise height Z_HI (15 cm above the
table, clear of the folded shirt) to directly above the grasp point → VERTICAL descent to the table
top (the first two grasps press 5 mm BELOW it — cloth_franka's μ·N anchor recipe; the stopper
converts the excess into pressing normal force) → close to 8 mm → VERTICAL lift back to carry
height → carry to the release point → release → rise clear. Grasping is pure frictional contact
(μ_eff ≈ 0.61); the cloth is the central SI-converted Newton cloth (ClothConfig defaults) — the
demo declares NO physics."""
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
Z_HI = 35.0                                    # open-glide cruise height [Newton cm]: 15 cm above the
                                               # table top, clear of the settled/folded shirt
SCALE = 0.8                                    # in-spirit pose mapping: Newton's offsets about its shirt
                                               # centre, scaled 0.8 so every pose stays reachable with the
                                               # shirt centred on the SHARED default table (same shirt
                                               # regions grasped/released, not the exact world points)

# Newton's grasp/release points, restructured into the clean per-fold cycle (glide high → vertical
# descent → close → vertical lift → carry → release → rise clear): (duration [s], x, y, z [cm], closed?).
ROWS = [
    (4.0, 31, -60, Z_HI, False),               # --- top left: glide to above the grasp point
    (2.0, 31, -60, 19.5, False),               #     vertical descent, pressing 5 mm below the table
                                               #     top (cloth_franka's μ·N anchor recipe)
    (2.0, 31, -60, 19.5, True),                #     close to 8 mm
    (2.0, 31, -60, 31.0, True),                #     vertical lift to carry height
    (2.0, 12, -60, 31.0, True),                #     carry
    (3.0, -6, -60, 31.0, True),                #     carry
    (1.0, -6, -60, 31.0, False),               #     release
    (1.0, -6, -60, Z_HI, False),               #     rise clear
    (2.0, 15, -33, Z_HI, False),               # --- bottom left: glide to above the grasp point
    (2.0, 15, -33, 19.5, False),               #     vertical descent, pressing 5 mm below the top
    (3.0, 15, -33, 19.5, True),                #     close
    (2.0, 15, -33, 28.0, True),                #     vertical lift
    (3.0, -2, -33, 28.0, True),                #     carry
    (1.0, -2, -33, 28.0, False),               #     release
    (1.0, -2, -33, Z_HI, False),               #     rise clear
    (2.0, -28, -60, Z_HI, False),              # --- top right: glide to above the grasp point
    (2.0, -28, -60, 20.0, False),              #     vertical descent
    (2.0, -28, -60, 20.0, True),               #     close
    (2.0, -28, -60, 31.0, True),               #     vertical lift
    (2.0, -18, -60, 31.0, True),               #     carry
    (3.0, 5, -60, 31.0, True),                 #     carry
    (1.0, 5, -60, 31.0, False),                #     release
    (1.0, 5, -60, Z_HI, False),                #     rise clear
    (2.0, -18, -30, Z_HI, False),              # --- bottom right: glide to above the grasp point
    (2.0, -18, -30, 20.5, False),              #     vertical descent
    (3.0, -18, -30, 20.5, True),               #     close
    (2.0, -18, -30, 31.0, True),               #     vertical lift
    (2.0, -3, -30, 31.0, True),                #     carry
    (2.0, -3, -30, 31.0, False),               #     release
    (1.0, -3, -30, Z_HI, False),               #     rise clear
    (2.0, 0, -20, Z_HI, False),                # --- bottom: glide to above the grasp point
    (2.0, 0, -20, 19.5, False),                #     vertical descent
    (2.0, 0, -20, 19.5, True),                 #     close
    (2.0, 0, -20, 35.0, True),                 #     vertical lift
    (1.5, 0, -32, 35.0, True),                 #     carry the bottom hem up the shirt
    (1.5, 0, -44, 35.0, True),                 #     carry
    (1.5, 0, -58, 35.0, True),                 #     carry — all the way to the shirt's TOP edge
    (1.5, 0, -58, 35.0, False),                #     release
    (2.0, -28, -60, 40.0, False),              # retreat up & away
]


def _newton_policy():
    """Newton's key poses -> (waypoints, finger_schedule): in-spirit xy mapping (scaled about the
    shirt centre), z table-relative; the finger keyframes blend each open/close over its row like
    Newton's per-row activation target."""
    wps, fingers, t = [WP(0.0, P0)], [(0.0, OPEN)], 0.0
    closed = False
    for dur, x, y, z, close in ROWS:
        pos = (0.12 + SCALE * 0.01 * x, -0.05 + SCALE * 0.01 * y, TT + 0.01 * (z - 20.0))
        if close != closed:
            fingers += [(t, SHUT if closed else OPEN), (t + dur, SHUT if close else OPEN)]
            closed = close
        t += dur
        wps.append(WP(t, pos))
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
    coupling_soft_ke=CLOTH.soft_contact_ke,            # enable the harvest; framework derives the
                                                       # effective avg(particle, pad) ke centrally
    object_solver_kwargs={"rigid_body_contact_buffer_size": 4096,
                          "rigid_body_particle_contact_buffer_size": 65536},
    object_pipeline_kwargs={"soft_contact_margin": CLOTH.contact_margin},
    substeps=10, vbd_iterations=5, num_frames=int(76.5 * 60),  # Newton's dt (10 @ 60 fps) — part of
                                                       # the dimensionless contact parity (docs/physicsEngine/cloths.md)
)
