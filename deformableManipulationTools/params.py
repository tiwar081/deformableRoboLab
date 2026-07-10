"""Single source of truth for robot, grip, and object physics parameters.

Every Franka example imports these so the **robot** and a given **object type** have identical
properties across all demos. Centralizing here also makes the Newton-version-sensitive values
(absolute VBD contact damping, re-derived for the pinned `_external/newton`) live in one place —
see CLAUDE.md "Newton version (environment gotcha)".

Convention: frozen dataclasses with one canonical instance per type (`FRANKA`, `GRIP`, `CABLE`,
`TABLE`, …). Object variants that are genuinely different physical objects (e.g. a pillow-soft
block for drop-impact vs. a firm block for pick-and-place) are separate *named* instances, still
defined here so the parameters are never re-derived ad hoc in an example.

All damping coefficients are **absolute** VBD units (Newton ≥1.4): contact `kd` [N·s/m], etc.
"""
from __future__ import annotations

from dataclasses import dataclass

from .settings import SETTINGS, REPO_ROOT


# ---------------------------------------------------------------------------------------------
# Robot (Franka FR3 + hand) — identical across every example.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotConfig:
    # ---- asset source: a downloaded URDF (loader="urdf") OR a local USD (loader="usd") ----
    # The two supported robots are kinematically identical (same arm, same TCP at the same home pose)
    # and differ only in the end-effector geometry + the load path, so EVERYTHING below the source
    # fields (home pose, TCP offset, gains, limits) is shared verbatim — see params.ROBOTS.
    loader: str = "urdf"                              # "urdf" -> download_asset + add_urdf; "usd" -> add_usd
    asset_name: str = "franka_emika_panda"           # newton.utils.download_asset key (urdf loader)
    urdf_subpath: str = "urdf/fr3_franka_hand.urdf"  # under the downloaded asset (urdf loader)
    usd_path: str | None = None                      # absolute path to the robot USD (usd loader)
    ee_link_suffix: str = "fr3_link7"
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.22)  # TCP offset in link7 frame [m]
    left_finger_suffix: str = "fr3_leftfinger"
    right_finger_suffix: str = "fr3_rightfinger"
    hand_link_suffix: str = "fr3_hand"                # the hand/palm link (wrist-camera mount, render only)
    # Optional scenic-render tint for the gripper (hand + fingers), to tell robots apart at a glance.
    # None -> default gray. Render-only; physics is unaffected.
    viz_gripper_color: tuple[float, float, float] | None = None
    # Render the robot from the SIMULATED (physics) model's collision/visual meshes rather than the USD's
    # detailed visual meshes, so the scenic render shows exactly the geometry being simulated (e.g. the
    # panda's per-link convex hulls). Render-only; physics is unaffected.
    render_from_physics: bool = False
    short_name: str = "fr3_franka"                    # output-folder name for this robot (outputs/<short_name>/<demo>)
    n_dof: int = 9                                    # 7 arm + 2 finger
    n_arm_dof: int = 7
    # Resting/home joint configuration (7 arm + 2 finger; fingers set to gripper_open).
    home_q: tuple[float, ...] = (
        -0.0036802115, 0.023901723, 0.003680411, -2.3683236,
        -0.00012918962, 2.3922248, 0.785492, 0.04, 0.04,
    )
    gripper_open: float = 0.04                        # URDF prismatic upper limit [m]
    # PD actuator gains (position mode) — arm DOFs 0-6, finger DOFs 7-8.
    # arm_target_ke/kd are the arm's joint impedance — they SET how much the arm sags/jolts under
    # the harvested EE load (docs/gripper.md "net-to-EE"). 400/80 EXACTLY matches RoboLab's DROID
    # Franka (_external/RoboLab robolab/robots/droid.py, ImplicitActuatorCfg stiffness=400,
    # damping=80 on all 7 arm joints) — RoboLab is the sim2real reference, so the arm must feel
    # loads the same way. The old 420/42 had matching stiffness but HALF the damping (transients
    # under-damped → visible jolt on grasp/load). Known, accepted differences: a real stiff
    # position-controlled Franka is ~5-10x stiffer still (libfranka joint impedance 2000-3000,
    # menagerie kp 2000-4500 — a 20 N payload sags our TCP ~1-2.5 cm, which RoboLab would too);
    # Newton's own Franka demos drive the arm KINEMATICALLY (infinitely stiff, no force feedback),
    # so ours is softer than Newton's by construction. RoboLab also splits arm effort 87 (j1-4) /
    # 12 (j5-7) N*m where we use 87 uniformly — noted, not adopted (single-scalar schema).
    arm_target_ke: float = 400.0
    arm_target_kd: float = 80.0
    arm_effort: float = 87.0                          # Franka arm joint torque limit [N·m]
    finger_target_ke: float = 300.0
    finger_target_kd: float = 30.0
    finger_effort: float = 20.0                       # finger actuator force [N] (physical, NOT a grip cap)
    armature: float = 0.1
    # SolverMuJoCo config — NVIDIA cable/cube two-way-coupling settings (stiff contacts).
    solver: str = "newton"
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    solver_iterations: int = 20
    solver_ls_iterations: int = 100
    solver_impratio: float = 1000.0
    # Continuous collision detection — robust contacts for fast grasp/impact (off by MuJoCo default
    # unless passed). Beneficial for every demo's robot contacts; essential when MuJoCo also hosts the
    # objects (rigid-only path). enable_multiccd gives up to 4 contacts/pair (single if either margin>0).
    ccd_iterations: int = 50
    ccd_tolerance: float = 1.0e-6
    enable_multiccd: bool = True


# ---------------------------------------------------------------------------------------------
# Grip — dynamic finite-mass proxy bridge (NVIDIA recipe). Physical bounded force, NO cap.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GripConfig:
    """Dynamic finite-mass gripper-proxy contact bridge (deformableManipulationTools/grip.py).

    The proxies mirror the fingers as dynamic bodies in the object's VBD model; the object's
    contact reaction is harvested and the net external load fed to the arm/EE. Grip force is the
    position-controlled squeeze against bounded contact — finite and physical, never capped.
    """
    proxy_mass: float = 10.0          # kg per proxy (reflected articulated-chain inertia, scaled for stability)
    proxy_inertia: float = 0.1        # kg·m², isotropic
    proxy_ke: float = 5.0e4           # proxy contact stiffness [N/m]
    proxy_kd: float = 1.0e2           # proxy contact damping [N·s/m] (absolute; re-derived, was 5e6 ~1e4x critical)
    proxy_mu: float = 1.5             # proxy/pad friction
    proxy_margin: float = 0.001       # contact margin [m]
    # Centralized proxy contact gap [m]: force builds over this distance (f≈ke·(gap−separation)), so it
    # sets how far the pad sits off the object at a given squeeze. SMALL on purpose — a large gap (the
    # old 8 mm) holds a stiff object firmly only at a deep, NON-physical squeeze (the pad floats far off
    # at any physical force); ~2 mm lets a physical force_target (~30 N) engage the pad near the true
    # surface. Still ≫ the swept cable's ~0.9 mm/substep, so no tunneling (verify on Newton bumps).
    proxy_gap: float = 0.002
    # Contact damping applied to the *gripped object's* shapes (cable/box) — physical, re-derived.
    object_contact_kd: float = 1.0e2
    # ---- Force-target grasp — ONE UNIFIED controller for EVERY object (rigid, cable, AND soft) ----
    # A single BIDIRECTIONAL admittance regulator (grip.py _grip_force_stop_kernel); the grasp no longer
    # depends on whether the demo's object is soft or incompressible — only the TARGET FORCE differs (the
    # one per-demo knob, GraspWindow.force_target). It reads the closing-axis PROJECTED squeeze
    # (TwoWayProxyCoupling.grip_squeeze_signal), not the raw magnitude: the lift/sweep LOAD is tangential to
    # the jaw axis, so it projects out and the regulator holds steady under load instead of mis-reading the
    # load as over-squeeze and opening. Velocity form (integral → zero steady-state error).
    #
    # It is ASYMMETRIC — close responsively, open reluctantly — which is what makes a force regulator
    # stable on a frictional grasp whose contact force is spiky + load-dominated: F_filt =
    # EMA(grip_squeeze_signal, force_filter_tau); close at grip_rate_max to first contact (F_filt >=
    # engage_force), then regulate with DIRECTION-DEPENDENT GAINS: w_dot = k_close*(err + db) when
    # UNDER target (chase the target / restore a DECAYING grip) and w_dot = k_open*(err - db) when
    # OVER (k_close = k_open_ratio * k_open), so the lift/sweep LOAD and transient SPIKES (which
    # transiently dwarf the target) back the jaw off only slowly, in proportion to the over-force,
    # and the EMA + deadband eat the short ones entirely. The RATE CAP is symmetric and PHYSICAL
    # (grip_rate_max both ways — a real jaw opens as fast as it closes; the old design's separate
    # 200x-slower open CAP was a controller choice disguised as a limit and lived in the wrong knob).
    # "Grab firmly, release reluctantly" — bidirectional but load-robust. The target is in PROJECTED
    # (closing-axis) N; pick it per demo near the achievable steady squeeze (a thin cable tops out ~4 N at
    # any width so its target sets the cage tightness; a rigid box reaches ~30 N; a soft block ~8 N gentle).
    force_target: float = 30.0        # DEFAULT target [N] if a GraspWindow gives no per-demo override.
    #
    # ---- TARGET-RELATIVE controller parameters (resolved per grasp window by window_params()) ----
    # The regulator must work from ~2 N (a cloth shell) to ~80 N (the slip-prone banana) with the ONE
    # demo knob being force_target — so the parameters that are fractions-of-setpoint BY FUNCTION are
    # derived from the window's target, anchored so ft == adm_ref_force reproduces the long-tested
    # 30 N constants bit-for-bit. Everything that is a FIXED PHYSICAL PROPERTY of a real robot
    # (max finger speed grip_rate_max, the sensing-noise engage_floor, the sensor-set filter tau,
    # actuator gains/effort in RobotConfig) stays absolute — sim2real: those cannot be retuned on
    # hardware per task, so the sim must not either.
    #
    # k_adm scales ∝ 1/target (relative-error admittance): the CLOSE gain is
    # k_close = k_adm*adm_ref_force/ft — the full-scale-error close speed k_close*ft = 3e-3 m/s is
    # target-independent, so a 2 N cloth grasp tightens as briskly (in fraction-of-target units) as a
    # 30 N cube grasp instead of being ~15x slower (the old fixed gain made low-target regulation
    # glacial; with the old 2 N ABSOLUTE deadband it was literally dead — w_dot = 0 for any ft <= 2).
    # The OPEN gain is k_close/k_open_ratio: the anti-drop asymmetry lives HERE (a software gain),
    # not in the rate caps. Jaw retreat under a spike of size a*ft lasting T is then
    # (3e-3/k_open_ratio)*a*T — at ratio 20, a 2x-target 0.5 s spike retreats ~0.15 mm.
    k_adm: float = 1.0e-4             # CLOSE admittance gain [m/s per N] AT ft = adm_ref_force
    k_open_ratio: float = 20.0        # close/open gain ratio (>1): the asymmetry that keeps a spiky,
                                       # load-dominated grasp from opening; modest by design (the old
                                       # mechanism was a 200x-slower open RATE CAP). 10 lost the cable
                                       # cage in cable_soft (its long high-spike sweep); 20 holds.
    adm_ref_force: float = 30.0       # [N] anchor target: window_params(30) close-side == legacy constants
    k_adm_cap: float = 2.0e-3         # [m/s per N] close-gain guard for very low targets — keeps the
                                       # discrete contact loop stable (k*ke*dt << 1 even on the stiffest
                                       # proxy contact ke=5e4 at dt~1e-3); inactive for ft >= 1.5 N
    force_filter_tau: float = 0.05    # [s] EMA time constant on the squeeze signal (swallows contact
                                       # spikes). FIXED: set by the signal's noise spectrum (a sensor
                                       # property, not a task property).
    grip_rate_max: float = 0.04       # [m/s] jaw rate limit, BOTH directions + the blind approach speed.
                                       # FIXED (PHYSICAL) and SYMMETRIC: just under the real Franka Hand's
                                       # max finger speed (~0.05 m/s), which is direction-independent —
                                       # identical for every grasp (NOT per-demo, NOT target-scaled).
    # Engage threshold (projected squeeze) to switch from blind approach to force regulation. RELATIVE to
    # the target (clamped) so it scales: a low-target SOFT grip engages at a light touch — BEFORE the fast
    # approach over-compresses the compliant block — while a high-target rigid/cable grip engages firmly (a
    # firmer seed is what gives the cable its cage). engage = clamp(engage_frac*target, engage_floor, engage_cap).
    engage_frac: float = 0.15         # engage threshold as a fraction of the target force
    engage_floor: float = 0.3         # [N] min engage threshold. FIXED (PHYSICAL): the contact-DETECTION
                                       # floor below which a real force estimate is indistinguishable from
                                       # noise/tare drift — scaling it down would fire engage on noise.
    engage_cap: float = 2.0           # [N] max engage threshold. FIXED: bounds the approach-phase ram-in
                                       # (overshoot ~ contact_ke*v_close*lag — target-independent).
    # Deadband scales ∝ target (a limit-cycle killer is a fraction-of-setpoint quantity, 1/15 of ft —
    # the old 2 N ABSOLUTE band exceeded low targets entirely and disabled their regulation), floored at
    # an absolute noise band (physical: don't chatter on measurement noise).
    grip_force_deadband: float = 2.0  # [N] deadband around the target AT ft = adm_ref_force
    grip_force_deadband_floor: float = 0.1  # [N] absolute noise floor for the scaled deadband

    def window_params(self, force_target: float) -> tuple[float, float, float]:
        """THE centralized per-grasp-window controller derivation:
        ``(k_close_w, k_open_w, deadband_w)`` for a window's ``force_target``. Consumed by
        GripController (packed per window into the kernel's windows array) and by instrumentation
        (plot reference bands) — never derived anywhere else. Anchored: the close side of
        ``window_params(adm_ref_force)`` equals the legacy constants ``(k_adm, grip_force_deadband)``
        exactly, so the long-tested 30 N closes (cable cage, rigid cube, rubik's) are bit-identical;
        the open side is ``k_close/k_open_ratio`` (the anti-drop gain asymmetry)."""
        s = float(force_target) / self.adm_ref_force
        k_close = min(self.k_adm / s, self.k_adm_cap)
        return (k_close, k_close / self.k_open_ratio,
                max(self.grip_force_deadband * s, self.grip_force_deadband_floor))
    # Hand/palm "blocker" proxy (CENTRALIZED — every VBD demo's gripper gets it; not demo-tweakable).
    # The two finger proxies already carry the finger collision geometry, but the Franka hand is
    # collapsed into link7 and its collider lives only in MuJoCo, so the VBD world has NO collider for
    # the palm/wrist — a swept cable passes straight through the gripper PALM and the EE. A single box
    # mirroring the EE (link7) blocks it. It is a BLOCKER: re-pinned to the EE each substep, harvested
    # two-way to the EE (momentum-consistent, so it never shoves an object one-way) but NOT in the grip
    # signal. Box dims are in the link7/EE frame (z grows wrist→fingertip):
    # it spans the wrist→hand region up to just below the finger origins (~z=0.165) and stays 6 cm clear
    # of the fingertip grasp zone (TCP at z=0.22). Cheap: rigid-shape collision only (no particle).
    palm_box_half: tuple[float, float, float] = (0.05, 0.05, 0.08)    # half-extents [m], EE(link7) frame
    palm_box_offset: tuple[float, float, float] = (0.0, 0.0, 0.08)    # box center in the EE(link7) frame [m]


@dataclass(frozen=True)
class GraspWindow:
    """One grasp episode declared by a demo's POLICY for the force-stop GripController (grip.py):
    close the fingers over ``[close_start, close_end]``, hold until ``release_start``, reopen by
    ``release_end``. For a grasp held to the end of the demo (no release, e.g. the cable sweep),
    leave the +inf defaults. All times are in seconds of sim time, on the same clock the arm-target
    kernel uses.

    The ONLY physics knob is ``force_target`` — the per-demo TARGET GRASP FORCE [N] (``None`` → the
    centralized default ``GRIP.force_target``). The SAME bidirectional admittance controller regulates the
    closing-axis projected squeeze to that target for EVERY object (rigid, cable, AND soft) — the grasp does
    not depend on the object type, only on the target. Geometry-independent: the cable, a flat box and a soft
    block each end at whatever width gives the requested force ("specify force, get emergent geometry")."""
    close_start: float
    close_end: float
    release_start: float = float("inf")
    release_end: float = float("inf")
    force_target: float | None = None    # per-demo target grasp force [N]; None → GRIP.force_target default


@dataclass(frozen=True)
class MujocoGripConfig:
    """Franka finger control for a TRUE two-way MuJoCo grasp (the rigid-only path, where robot AND
    objects share one SolverMuJoCo). The gripper closes to a FIXED target and contact + the actuator
    effort hold the object — NO object-size preset width (cf. _external/RoboLab). Stiffer/stronger than
    the default finger gains, which in the VBD-proxy path only POSITION the fingers (the proxies grip)."""
    finger_target_ke: float = 2.0e3   # finger position-actuator gain [N/m]
    finger_target_kd: float = 1.0e2
    finger_effort: float = 200.0      # finger grip force cap [N]
    close_target: float = 0.0         # fully-closed finger target [m]; contact stops the pads at the object


# ---------------------------------------------------------------------------------------------
# Table — shared static collider (object-model visible + robot-model hidden stopper).
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TableConfig:
    pos: tuple[float, float, float] = (0.12, -0.45, 0.035)
    half: tuple[float, float, float] = (0.45, 0.35, 0.035)
    robot_ke: float = 5.0e4
    robot_kd: float = 5.0e2
    robot_mu: float = 1.0
    robot_margin: float = 1.0e-3
    robot_density: float = 1000.0
    object_ke: float = 5.0e4
    object_kd: float = 1.0e2
    object_mu: float = 0.8
    color: tuple[float, float, float] = (0.52, 0.52, 0.48)

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + self.half[2])


# ---------------------------------------------------------------------------------------------
# Objects.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CableConfig:
    """VBD rod (add_rod). Damping is absolute VBD units (Newton ≥1.4)."""
    radius: float = 0.008
    segment_length: float = 0.035
    node_count: int = 15
    density: float = 1200.0           # jacketed-cable density; light cables turn pinch residuals into ejection
    stretch_stiffness: float = 2.5e4
    stretch_damping: float = 1.25e3   # absolute: 0.05·stretch_stiffness
    bend_stiffness: float = 1.5e1
    bend_damping: float = 0.3         # absolute: 0.02·bend_stiffness
    friction: float = 1.5
    contact_ke: float = 2.0e4
    contact_kd: float = 1.0e2         # absolute; re-derived (was 20·ke=4e5, ~1e4x critical)
    contact_margin: float = 0.001
    bow: float = 0.02                 # geometric layout bow that locks the free rolling mode


@dataclass(frozen=True)
class SoftBlockConfig:
    """FEM block (add_soft_grid). ONE canonical block shared by every soft demo (cable sweep,
    cube-drop squash, plate compression, pick-and-place) so they are cross-comparable. Medium
    stiffness (k_mu≈5e2): soft enough to visibly dent/squash, firm enough to grasp and lift
    without squishing out of the pads. Damping/contact in the absolute objective-C=FᵀF metric."""
    dim: tuple[int, int, int] = (4, 4, 4)
    cell: float = 0.0125              # 4 cells × 0.0125 = 0.05 m cube
    density: float = 150.0
    k_mu: float = 5.0e2
    k_lambda: float = 2.5e3
    k_damp: float = 1.0e1
    particle_radius: float = 0.0035
    contact_margin: float = 0.01
    soft_contact_ke: float = 1.0e5
    soft_contact_kd: float = 1.0e-4
    soft_contact_kf: float = 1.0e3
    soft_contact_mu: float = 0.8


@dataclass(frozen=True)
class RigidBoxConfig:
    """Rigid box object. ``mass`` (when not None) is the exact body mass [kg]; otherwise the mass
    is density-derived from the shape volume (used by the distinct YCB rubik's cube)."""
    half_extent: float = 0.025
    mass: float | None = 1.0          # kg — the canonical rigid cube is 1 kg in every demo
    density: float = 250.0            # pre-mass seed / used only when ``mass`` is None
    contact_ke: float = 5.0e4
    contact_kd: float = 1.0e2
    contact_mu: float = 0.8
    contact_margin: float = 1.0e-3


@dataclass(frozen=True)
class PlateConfig:
    """A flat rigid plate with a graspable handle on top (sheet + handle, ONE rigid body, two
    boxes). A distinct centralized presser tool — built by ``add_plate``, never inline in a demo."""
    sheet_half: tuple[float, float, float] = (0.09, 0.06, 0.004)
    handle_half: tuple[float, float, float] = (0.016, 0.012, 0.024)
    density: float = 9300.0           # ~2 kg over the sheet+handle volume (≈2× the rigid cube)
    contact_ke: float = 1.0e5         # firm plate↔block press
    contact_kd: float = 1.0e-4
    contact_mu: float = 0.8
    contact_margin: float = 1.0e-3
    sheet_color: tuple[float, float, float] = (0.62, 0.65, 0.68)
    handle_color: tuple[float, float, float] = (0.40, 0.42, 0.45)

    @property
    def handle_local_pos(self) -> tuple[float, float, float]:
        return (0.0, 0.0, self.sheet_half[2] + self.handle_half[2])


@dataclass(frozen=True)
class YcbMeshConfig:
    """A YCB mesh object loaded for Newton collision (deformableManipulationTools/mesh_collision.py).

    The full mesh renders; coacd convex-hull pieces collide (consistent normals, cavity
    preserved — docs/SOLVERS.md §4: a raw concave mesh ejects the solve). ``target_mass`` is the
    realistic YCB mass applied after finalize so the body isn't flung (SOLVERS.md §5); ``density``
    only seeds the pre-rescale inertia shape. ``ke``/``kd`` are the absolute VBD contact material."""
    usd_subpath: str                  # under assets/objects, e.g. "ycb/bowl.usd"
    target_mass: float                # kg — exact mass applied post-finalize (realistic YCB object)
    density: float = 400.0            # pre-rescale seed only
    ke: float = 5.0e4
    kd: float = 1.0e2
    mu: float = 1.0
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    # coacd convex decomposition (preprocess_mode='on' is forced in the worker — 'auto' segfaults
    # on raw non-watertight YCB scans).
    coacd_threshold: float = 0.08
    coacd_max_convex_hull: int = 12
    coacd_preprocess_resolution: int = 50
    coacd_max_ch_vertex: int = 32
    coacd_seed: int = 0
    piece_maxhullvert: int = 32       # re-hull cap per piece (convex-preserving small BVH)


# ---- Robot registry: two kinematically-identical Frankas, selected centrally by settings.yaml ----
# Only the end-effector geometry + the load path differ, so the panda overrides ONLY the loader, the
# USD path, and the three link-name suffixes; the home pose, TCP offset, gains, and limits are inherited
# from RobotConfig's defaults (verified drop-in: same link7/finger/TCP world pose at the shared home_q).
FR3_FRANKA_HAND = RobotConfig()                                  # Franka FR3 + hand (Newton URDF pack)
FRANKA_PANDA_ISAACSIM = RobotConfig(                             # Isaac Sim Franka Panda (native USD)
    loader="usd",
    asset_name="franka_panda_isaacsim",
    usd_path=str(REPO_ROOT / "assets" / "robots" / "franka_panda_isaacsim" / "franka.usd"),
    ee_link_suffix="panda_link7",
    left_finger_suffix="panda_leftfinger",
    right_finger_suffix="panda_rightfinger",
    hand_link_suffix="panda_hand",
    viz_gripper_color=None,    # off for now; set to (0.05, 0.75, 0.82) teal to distinguish from fr3
    render_from_physics=True,  # demos render the simulated convex-hull gripper (matches the proxy viz)
    short_name="panda_franka",
)
ROBOTS = {"fr3_franka_hand": FR3_FRANKA_HAND, "franka_panda_isaacsim": FRANKA_PANDA_ISAACSIM}
if SETTINGS.robot not in ROBOTS:
    raise ValueError(f"settings.yaml robot={SETTINGS.robot!r} is not one of {sorted(ROBOTS)}.")

# ---- Canonical instances (single source of truth) -------------------------------------------
FRANKA = ROBOTS[SETTINGS.robot]    # the ACTIVE robot — everything imports this; settings.yaml picks it
GRIP = GripConfig()
MUJOCO_GRIP = MujocoGripConfig()   # finger control for the rigid-only single-MuJoCo grasp
TABLE = TableConfig()
CABLE = CableConfig()

# ONE soft block + ONE rigid cube + ONE plate, shared identically by every non-YCB demo so the
# demos are cross-comparable (see CLAUDE.md "Code layout"). No per-demo variants.
SOFT_BLOCK = SoftBlockConfig()
RIGID_CUBE = RigidBoxConfig()                                             # 1 kg cube, every demo
PLATE = PlateConfig()

# The YCB rubik's cube is a DISTINCT object (its own size/mass/material) — no relationship to the
# canonical RIGID_CUBE. Density-derived mass (mass=None).
RUBIKS_CUBE = RigidBoxConfig(half_extent=0.029, mass=None, density=1025.0, contact_ke=5.0e4,
                             contact_kd=1.0e2, contact_mu=1.2, contact_margin=0.0)

# YCB mesh objects (pickplace_ycb). coacd-decomposed for collision; realistic YCB masses.
BOWL_YCB = YcbMeshConfig(
    usd_subpath="ycb/bowl.usd", target_mass=0.147, density=400.0, mu=1.0,
    color=(0.75, 0.22, 0.18), coacd_threshold=0.10, coacd_max_convex_hull=12)   # YCB 024_bowl
BANANA_YCB = YcbMeshConfig(
    usd_subpath="ycb/banana.usd", target_mass=0.066, density=300.0, mu=2.0,
    color=(0.93, 0.82, 0.12), coacd_threshold=0.08, coacd_max_convex_hull=8)    # YCB 011_banana
