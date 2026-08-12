"""THE PROTOCOL: what "test this grasp" means here, as data rather than as scattered constants.

Structure follows ACRONYM's (NVIDIA, 17.7M simulated parallel-jaw grasps), because our record schema
already borrows its five ``grasps/qualities/flex/*`` metric names and a number recorded under one of
those names should mean what the name says:

    "The simulation is initialized with the gripper in the pre-grasp state ... fingers are closed
     using a velocity-based controller until a force threshold is reached or the hand is fully
     closed. Finally, a shaking motion is executed: The hand first moves up and down along its
     approach direction, then rotates around a line parallel to the prismatic joint axes of the
     fingers. Afterwards, we record grasp success by testing whether the object is still in contact
     with both fingers."

The two deliberate departures, both because this is our simulator and not FleX:

1. **Their numbers are not imported.** ACRONYM never publishes shake amplitudes or thresholds, and
   the ones it used were tuned against a different contact model; a constant carried across would be
   cargo. Every number below was picked against OUR solver and its provenance is in its comment.
2. **Gravity is applied**, which ACRONYM deliberately does not do ("gravity is not present"; they
   shake instead, precisely so no gravity DIRECTION is assumed). We want both, so the timeline puts
   gravity where it does not corrupt the closing measurement: OFF while the jaws close (an object in
   free fall would make ``object_motion_during_closing_*`` a measurement of falling, not of the
   grasp), then ON for the whole disturbance phase. Its direction is the candidate's own APPROACH
   axis, which is the same as testing every candidate with its object hanging out of the jaws —
   the standard pick load, and identical in gripper-relative terms for every candidate, so two
   candidates' numbers are comparable instead of encoding which way the object happened to face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# =================================================================================================
# Timeline + disturbance
# =================================================================================================
# Sine-with-Hann-envelope, not a bare sine: a bare sine has its PEAK VELOCITY at t=0, so the shake
# would start with a velocity step the position-controlled base cannot follow — the grasp would then
# be tested by a start transient whose size depends on base gains rather than by the shake we asked
# for. The Hann envelope leaves displacement AND velocity zero at both ends and puts the peak
# acceleration in the middle of the window.
def envelope(t: float, duration: float) -> float:
    """Hann window over ``[0, duration]`` (0 outside)."""
    if not 0.0 <= t <= duration or duration <= 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t / duration))


@dataclass(frozen=True)
class Windows:
    """When the disturbance happens, once the close has finished. All absolute sim times [s]."""
    linear_start: float
    linear_end: float
    angular_start: float
    angular_end: float
    end: float


@dataclass(frozen=True)
class ShakeProtocol:
    """One complete trial specification. Every duration is in seconds of sim time."""

    # ---- integration: the demo path's own numbers (demo_runner.DemoSpec defaults), so a candidate
    # is validated at the fidelity the demos run at, not at a cheaper one chosen for this pass.
    fps: int = 60
    substeps: int = 16
    vbd_iterations: int = 12

    # ---- phase durations ----
    # settle: let the VBD solve relax its initialisation before anything is measured. 0.2 s is ~12
    # frames, which is where the sponge's tet mesh stops twitching (measured); a rigid body needs none.
    settle: float = 0.2
    # The close is not a fixed window. ACRONYM closes "until a force threshold is reached or the hand
    # is fully closed", and a fixed window here was actively wrong: the admittance regulator's close
    # rate is proportional to the force ERROR, so a light object with a 1 N target closes its last
    # millimetre roughly twenty times slower than a 20 N one. A 1.8 s window was ample for the 514 g
    # sugar box and cut the 6 g sponge off mid-close, at 0.5 N of a 1 N target with the pads still
    # 1.6 mm short of its surface — which then read as "dropped" when it was really "never gripped".
    # So: close until the FILTERED squeeze has sat inside the controller's own deadband for
    # ``close_settle``, or until ``close_max`` runs out, and record which happened.
    close_max: float = 6.0        # [s] timeout: the 1.0 s worst-case blind approach (the full 40 mm
                                  # half-aperture at GRIP.grip_rate_max) plus 5 s of regulation.
                                  # Calibrated against the SLOWEST close in the catalog — the 6 g
                                  # sponge at a 1 N target, whose squeeze is still rising at 4 s — so
                                  # that a slow grasp is measured after its transient and not during
                                  # it. A candidate still regulating at 6 s is reported unconverged
                                  # rather than waited on indefinitely.
    close_settle: float = 0.3     # [s] dwell inside the deadband that counts as converged. 6x the
                                  # controller's own force_filter_tau, so the filter has caught up.
    # gravity_hold: gravity switches on at the end of the close and the grasp is left to take the
    # load before it is shaken. Long enough for the admittance loop to re-converge (force_filter_tau
    # is 0.05 s, so ~10 tau).
    gravity_hold: float = 0.5
    # recover: quiescent window at the end. Retention is judged over THIS window rather than at one
    # instant, so a single flickering contact frame cannot decide a candidate.
    recover: float = 0.5

    # ---- the shake ----
    # 2 Hz: fast enough that peak acceleration is comparable to g at centimetre amplitude, slow
    # enough to sit an order of magnitude below the base actuator's own corner frequency (~50 Hz
    # linear) so the commanded motion is what the object actually receives. MEASURED and reported per
    # trial as TrialResult.shake_ratio_*: the base reaches 96% of both commanded amplitudes.
    frequency: float = 2.0
    cycles: int = 4
    # Amplitudes: TUNED AGAINST THIS SOLVER (see README "Tuning"), not taken from ACRONYM, which
    # publishes none. 40 mm at 2 Hz peaks at 6.3 m/s^2 (0.64 g) and 0.5 rad peaks at 79 rad/s^2 —
    # a disturbance comparable to the load rather than dominated by it, which is what makes the shake
    # add information instead of re-measuring the load phase. No trial has diverged UNDER THE SHAKE;
    # the one divergence on the fixture set (the mug wall pinch) happens at the pre-grasp, where the
    # pads start inside the mug, and is reported as such.
    linear_amplitude: float = 0.04         # [m] along the approach axis
    angular_amplitude: float = 0.5         # [rad] about the jaw closing axis

    gravity: float = 9.81                  # [m/s^2], applied along the candidate's approach axis

    # ---- phase boundaries (derived; the rig and the kernel both read these) ----
    @property
    def close_start(self) -> float:
        return self.settle

    @property
    def close_deadline(self) -> float:
        """Latest time the close may run to before the trial gives up and loads it anyway."""
        return self.close_start + self.close_max

    @property
    def shake_duration(self) -> float:
        return self.cycles / self.frequency

    def windows(self, t_gravity: float) -> "Windows":
        """The disturbance schedule, ANCHORED on the moment gravity came on.

        Everything after the close is a fixed sequence; only where it starts depends on how long the
        candidate took to grip. Both the rig's kernel launch and :meth:`base_command` read these, so
        there is one definition of when the shake happens."""
        lin0 = t_gravity + self.gravity_hold
        lin1 = lin0 + self.shake_duration
        ang1 = lin1 + self.shake_duration
        return Windows(linear_start=lin0, linear_end=lin1, angular_start=lin1, angular_end=ang1,
                       end=ang1 + self.recover)

    @property
    def max_duration(self) -> float:
        """Longest a trial can run: a close that uses its whole timeout, then the full disturbance."""
        return self.windows(self.close_deadline).end

    @property
    def max_frames(self) -> int:
        return int(round(self.max_duration * self.fps))

    @property
    def peak_linear_acceleration(self) -> float:
        """[m/s^2] at the middle of the linear shake window (envelope at 1)."""
        return self.linear_amplitude * (2.0 * math.pi * self.frequency) ** 2

    @property
    def peak_angular_acceleration(self) -> float:
        """[rad/s^2] at the middle of the angular shake window."""
        return self.angular_amplitude * (2.0 * math.pi * self.frequency) ** 2

    def base_command(self, t: float, t_gravity: float) -> tuple[float, float]:
        """``(approach translation [m], jaw-axis rotation [rad])`` commanded at sim time ``t``.

        The host-side twin of the rig's warp kernel; the self-check proves the two agree, so the
        schedule can be reasoned about here without reading kernel code."""
        win = self.windows(t_gravity)
        lin = ang = 0.0
        w = 2.0 * math.pi * self.frequency
        if win.linear_start <= t <= win.linear_end:
            u = t - win.linear_start
            lin = self.linear_amplitude * math.sin(w * u) * envelope(u, self.shake_duration)
        if win.angular_start <= t <= win.angular_end:
            u = t - win.angular_start
            ang = self.angular_amplitude * math.sin(w * u) * envelope(u, self.shake_duration)
        return lin, ang


PROTOCOL = ShakeProtocol()

# The anchor the rig uses while the close is still running: every window sits in the far future, so
# the base holds still without the kernel needing a branch. Named here so the self-check can probe
# the same value the rig passes.
_NEVER_PROBE = 1.0e9


# =================================================================================================
# The grasp force this pass asks for
# =================================================================================================
# ``GraspWindow.force_target`` is the one grasp knob a policy owns, and this pass is a policy: it has
# to choose one per trial. Hand-picking per object would make candidates incomparable and would be
# exactly the per-object tuning the codebase forbids, so the target is DERIVED from physics that the
# built model already knows:
#
#     F = safety * m * (g + a_peak) / (2 * mu_eff)
#
# the Coulomb force needed by two pads to carry the object's peak inertial load, times a margin.
# Every input is read off the finalized model (mass after the realistic-mass override, mu after the
# central material coupling), so nothing here is a tuned constant except the three below.
# NOT a reliability knob, and NOT tuned to a result — MEASURED (README "Tuning"): raising it makes a
# smooth cylinder WORSE (the can holds at 12 N and is ejected at 31 N, the curved-surface ejection of
# SOLVERS.md section 6) while it makes the banana better, so no value makes every sound grasp hold.
# It therefore stays at what physics says — twice the Coulomb minimum for the load this protocol
# applies — which is a policy a controller could choose without knowing the answer. 1.0 would put
# every grasp exactly on the slip boundary and the pass would measure the margin, not the pose.
FORCE_SAFETY = 2.0
FORCE_TARGET_MIN = 1.0    # [N] floor. Below ~0.7 N the regulator cannot engage at all: engage is
                          # clamp(0.15*ft, 0.3, 2.0) and the 0.3 N floor is a PHYSICAL sensing floor.
FORCE_TARGET_MAX = 40.0   # [N] ceiling. Above this the pass would be testing whether our contact
                          # model survives a crush, not whether the pose holds.


def force_target(mass: float, mu_effective: float, lever: float,
                 protocol: ShakeProtocol = PROTOCOL) -> float:
    """The target squeeze [N] for one trial. See the block comment above for the derivation.

    ``lever`` is the object's largest half-extent — it converts the angular shake into the tangential
    acceleration the outermost material sees, so a long object is gripped for the load its ends
    actually generate."""
    a_peak = protocol.peak_linear_acceleration + protocol.peak_angular_acceleration * float(lever)
    mu = max(float(mu_effective), 1.0e-3)
    required = float(mass) * (protocol.gravity + a_peak) / (2.0 * mu)
    return float(min(max(FORCE_SAFETY * required, FORCE_TARGET_MIN), FORCE_TARGET_MAX))


# =================================================================================================
# Retention + reporting resolution
# =================================================================================================
# ACRONYM's success test is "whether the object is still in contact with both fingers". The literal
# reading — count generated contact constraints — does NOT work here: Newton's narrow phase emits
# contact candidates out to the shape margin, so a pad reports "contacts" with an object it is
# 17 mm away from (measured, at the pre-grasp). The signal that means what ACRONYM's sentence means
# is the one the grip controller already regulates: ``TwoWayProxyCoupling.grip_squeeze_signal``, the
# MIN over the two pads of the harvested reaction projected onto the closing axis. It is positive
# exactly when BOTH pads are pressing on the object, which is the condition being tested.
RETENTION_SQUEEZE_MIN = 0.05   # [N] above harvest noise, far below the 1 N minimum force target
# Judged over the whole quiescent recover window rather than at one instant, so a single frame of
# contact flicker cannot decide a candidate.
RETENTION_CONTACT_FRACTION = 0.8

# An object DISPLACED this far from where it started is out of the jaws — a 40 mm shake cannot
# produce it and no amount of slip inside a <= 80 mm gripper can either. (Displacement, not distance
# from the grasp centre: a candidate may legitimately grip an elongated object near one end —
# historically the 0.5 m cable — which puts its
# centroid further from the jaws at rest than any threshold could allow.) Reaching it ends the trial
# early: the remaining frames would only measure free fall, and stopping keeps the reported motion a
# number about the grasp rather than a number about how long the clock ran.
ESCAPE_RADIUS = 0.15      # [m]
# ... unless it left FAST. Compared against the AVERAGE rate since gravity came on, which for a
# genuine drop is well under 1 m/s (free fall covers ESCAPE_RADIUS in 0.17 s, and a real drop happens
# part-way through a 4.5 s disturbance). Deliberately loose: it should catch a solver ejection, which
# is orders out, without ever second-guessing a physical drop.
DIVERGENCE_SPEED = 5.0    # [m/s]

# Reported precision. The pass contract wants the same asset to give the same sidecar, and a GPU
# contact solve is not guaranteed bit-reproducible (atomics reorder). MEASURED here: repeated trials
# of the same candidate agreed bit-for-bit (README "Repeatability"), so these quanta are insurance
# rather than a fix — set two orders below anything a consumer would act on (10 um, 10 urad) so they
# cost no signal. They make a re-run agree in the ordinary case; they cannot rescue a value that
# lands exactly on a quantum boundary, and the README says so.
ROUND_LINEAR = 1.0e-5     # [m]
ROUND_ANGULAR = 1.0e-5    # [rad]

def quantize(value: float | None, quantum: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(round(float(value) / quantum) * quantum)
