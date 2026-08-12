"""THE one place a stored SE(3) grasp pose meets what the arm can be COMMANDED to hold.

A record stores a full rotation (``POSE_CONVENTION`` v2: ``+z`` approach, ``+x`` jaw). The executor
does not take a rotation: every waypoint is ``(pos, yaw, tilt, tilt_axis)`` and
:func:`robot._ik_target_rot` turns that into

    R_cmd = Rot(tilt_axis, tilt) · Rot(z, yaw) · R_home                                       (1)

— the home hand orientation, yawed about WORLD z, then tilted about a caller-chosen axis. Somebody
has to convert, and if two callers convert differently the same stored grasp becomes two different
robot poses, so the conversion lives here and nowhere else.

**What the vocabulary actually restricts.** It is tempting to read (1) as "straight down, or yawed,
or slightly tilted" and conclude that most stored rotations are unreachable. That is not what the
algebra says. With ``tilt_axis`` free in the horizontal plane, (1) is a Z-X-Z Euler decomposition of
``R_cmd · R_home⁻¹``, and Z-X-Z covers all of SO(3): EVERY rotation has an exact (yaw, tilt, axis).
The measured constraint is therefore not coverage but MAGNITUDE — a 100 deg tilt is a pose the arm
should not be asked for, whatever the algebra permits. So the policy is:

  1. Decompose exactly. ``Δ = R_des · R_homeᵀ = Rz(α)·Rx(β)·Rz(γ)`` gives
     ``tilt_axis = (cos α, sin α, 0)``, ``tilt = β``, ``yaw = γ + α``. No approximation, no search.
  2. **Use the jaw's 180 deg symmetry.** A parallel-jaw grasp and the same grasp with the jaw axis
     reversed are the same physical grasp, so both are decomposed and the cheaper wrist commanded.
     Measured (and asserted in the self-test): the flip CANNOT change the tilt — it is a rotation
     about the approach itself, and ``(Δ·Rz(π))[2,2] = Δ[2,2]``, so β is a property of the approach
     direction alone. What it changes is the yaw, by exactly 180 deg. That is still worth taking:
     arm joint 7 has a finite range and the executor blends yaw linearly between waypoints, so the
     flip nearer zero is the one to command. Tilt-based rejection is therefore never something the
     flip can rescue — an approach the arm cannot tilt to is unreachable, full stop.
  3. Clamp the tilt to :data:`TILT_MAX` and CHARGE for it. Because the clamp moves only β, the
     geodesic error is exactly ``β − TILT_MAX`` — the distortion is an exact number, not an estimate
     (:func:`geodesic_angle` verifies it; the self-test asserts the identity).
  4. Reject past :data:`MAX_DISTORTION`. A pose needing more than that is not "slightly adjusted",
     it is a different grasp, and silently commanding it would grip somewhere the record never
     described.

Every projection returns its distortion whether or not it was rejected, so a caller can report what
the vocabulary cost rather than discovering later that its poses were quietly bent.

The geodesic is the standard SO(3) metric, ``θ = arccos((tr(R₁ᵀR₂) − 1) / 2)`` — the angle of the
relative rotation, the bi-invariant Riemannian distance. NOTE: the task named PRISM Appendix B.1 as
the source; that paper is not in this repo or its docs, so this is the standard definition, which
is what B.1 is expected to state. If B.1 uses a variant (e.g. the chordal/Frobenius distance
``‖R₁ − R₂‖_F``, which is monotone in the same angle but not equal to it), only this function
changes — nothing else in the package computes a rotation distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The home hand pose in the GRASP frame convention (+z approach, +x jaw), world <- grasp.
# MEASURED, not assumed: FK at FRANKA.home_q puts link7's +z along world -z and the finger
# separation along link (x+y)/sqrt(2), i.e. world -y; both robots in params.ROBOTS agree to 3e-6
# (they are kinematically identical at home_q by construction). Reproduce with:
#   build_franka_robot(...); newton.eval_fk(...); read body_q[ee_body] and the two finger bodies.
# Columns are the grasp frame's x (jaw), y, z (approach) axes in world coordinates.
HOME_GRASP_ROT = np.array([[0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0]])

# Largest tilt off straight-down this policy will command [rad]. 90 deg = a horizontal hand: the
# span from straight-down to horizontal is the whole range of approaches an object on a table
# presents, and a side grasp is an ordinary Franka pose, not an extreme one. Past 90 deg the
# approach points UPWARD — the hand is under-slung, reaching beneath the object — which is a
# different proposition and is what the clamp is for.
# MEASURED 2026-08-06, and the reason this is not 75 deg: at a 75 deg cap, 733 of 922 catalog
# survivors were clamped, every one of them by exactly 15.0 deg. They were the pure side grasps
# (tilt exactly 90 deg), all sitting precisely on the rejection boundary — i.e. the cap was
# silently bending a whole legitimate class of grasp by the maximum allowed, and any tightening of
# MAX_DISTORTION would have deleted them all at once. A cap that distorts the common case is
# mis-calibrated, whatever it says about wrist comfort.
TILT_MAX = math.radians(90.0)
# Largest orientation error this policy will accept after clamping [rad]. 15 deg is about the
# angular width of a fingertip pad across a 3 cm object — beyond it the jaws no longer close on the
# geometry the candidate measured.
MAX_DISTORTION = math.radians(15.0)

_DEGENERATE_SIN = 1.0e-9


def geodesic_angle(r_a, r_b) -> float:
    """Geodesic SO(3) distance between two rotation matrices [rad], in ``[0, π]``."""
    a = np.asarray(r_a, dtype=float).reshape(3, 3)
    b = np.asarray(r_b, dtype=float).reshape(3, 3)
    c = 0.5 * (float(np.trace(a.T @ b)) - 1.0)
    return float(math.acos(max(-1.0, min(1.0, c))))


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < _DEGENERATE_SIN:
        raise ValueError("rotation axis is degenerate")
    a = a / n
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def wrap_angle(angle: float) -> float:
    """To ``(-π, π]`` — the executor blends yaw linearly, so the short way round matters."""
    a = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def command_rotation(yaw: float, tilt: float, tilt_axis, home=HOME_GRASP_ROT) -> np.ndarray:
    """Equation (1): what the arm actually holds for a ``(yaw, tilt, tilt_axis)`` waypoint.

    Mirrors :func:`robot._ik_target_rot` (yaw about world z first, then the tilt), expressed in the
    GRASP frame rather than the link frame. The two differ by a constant 45 deg roll about the
    shared approach axis — link7's x is 45 deg off the jaw axis — which cancels out of (1), so a
    grasp-frame decomposition and a link-frame one give the SAME yaw/tilt/axis."""
    r = rot_z(yaw) @ np.asarray(home, dtype=float).reshape(3, 3)
    if tilt != 0.0:
        r = rot_axis(tilt_axis, tilt) @ r
    return r


@dataclass(frozen=True)
class HandCommand:
    """A stored rotation expressed in the executor's vocabulary, with what that cost.

    ``rotation`` is what the arm will hold — equal to the request when ``distortion == 0``.
    ``jaw_flipped`` records that the 180 deg jaw symmetry was used (the same physical grasp, the
    cheaper wrist). ``accepted`` is False when the distortion exceeded :data:`MAX_DISTORTION`; the
    command is still returned so the caller can report how far off it was."""
    yaw: float
    tilt: float
    tilt_axis: tuple
    rotation: np.ndarray
    distortion: float          # [rad] geodesic error introduced by the tilt clamp
    clamped: bool
    jaw_flipped: bool
    accepted: bool

    @property
    def waypoint_kwargs(self) -> dict:
        """The three fields a ``demo_runner.WP`` needs to reproduce this orientation."""
        return {"yaw": self.yaw, "tilt": self.tilt, "tilt_axis": self.tilt_axis}

    def describe(self) -> str:
        bits = [f"yaw={math.degrees(self.yaw):+6.1f}deg", f"tilt={math.degrees(self.tilt):5.1f}deg"]
        if self.jaw_flipped:
            bits.append("jaw-flipped")
        if self.distortion > 0.0:
            bits.append(f"DISTORTED {math.degrees(self.distortion):.1f}deg"
                        + ("" if self.accepted else " -> REJECTED"))
        return " ".join(bits)


def _zxz(delta: np.ndarray) -> tuple:
    """``Δ = Rz(α)·Rx(β)·Rz(γ)`` -> ``(α, β, γ)``, handling the two degenerate ``sin β = 0`` cases."""
    beta = math.acos(max(-1.0, min(1.0, float(delta[2, 2]))))
    if math.sin(beta) <= _DEGENERATE_SIN:
        # Gimbal: only α ± γ is determined. Put it all in γ (yaw) and leave the axis at +x, which
        # is what a straight-down or straight-up pose means anyway — there is nothing to tilt about.
        sign = 1.0 if delta[2, 2] > 0.0 else -1.0
        return 0.0, beta, math.atan2(sign * delta[1, 0], delta[0, 0])
    return (math.atan2(float(delta[0, 2]), -float(delta[1, 2])), beta,
            math.atan2(float(delta[2, 0]), float(delta[2, 1])))


def _project_one(rot_des: np.ndarray, home: np.ndarray, tilt_max: float, flipped: bool,
                 max_distortion: float) -> HandCommand:
    alpha, beta, gamma = _zxz(rot_des @ home.T)
    tilt = min(beta, tilt_max)
    distortion = beta - tilt                    # exact: the clamp moves only β (see module docs)
    axis = (math.cos(alpha), math.sin(alpha), 0.0)
    yaw = wrap_angle(gamma + alpha)
    return HandCommand(yaw=yaw, tilt=tilt, tilt_axis=axis,
                       rotation=command_rotation(yaw, tilt, axis, home),
                       distortion=float(distortion), clamped=distortion > 0.0,
                       jaw_flipped=flipped, accepted=distortion <= max_distortion)


# The jaw's own symmetry: rotating the grasp frame 180 deg about its approach swaps which pad is
# which. Same contact, same width, same face bucket — a different wrist angle.
_JAW_FLIP = np.diag([-1.0, -1.0, 1.0])


def project_pose(rot_des, *, home=HOME_GRASP_ROT, tilt_max: float = TILT_MAX,
                 max_distortion: float = MAX_DISTORTION) -> HandCommand:
    """Express a world-frame grasp rotation as ``(yaw, tilt, tilt_axis)``. See the module docstring.

    Exact whenever the required tilt is within ``tilt_max``; otherwise clamped, with the exact
    geodesic error reported in ``distortion`` and ``accepted`` False past ``max_distortion``."""
    r = np.asarray(rot_des, dtype=float).reshape(3, 3)
    h = np.asarray(home, dtype=float).reshape(3, 3)
    options = [_project_one(r, h, tilt_max, False, max_distortion),
               _project_one(r @ _JAW_FLIP, h, tilt_max, True, max_distortion)]
    # The two differ ONLY in yaw (see the module docstring), so choose the wrist nearer zero. The
    # unflipped option wins exact ties, so the choice is deterministic and a caller that ignores
    # jaw_flipped is never surprised by a 180 deg wrist swing it did not ask for.
    return min(options, key=lambda c: (round(abs(c.yaw), 12), c.jaw_flipped))
