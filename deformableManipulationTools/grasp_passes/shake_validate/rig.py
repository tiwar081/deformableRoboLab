"""One trial: put the free-floating hand on a candidate pose, close it, load it, shake it, measure.

The rig is a :class:`framework.GraspExample` subclass on purpose. Everything that decides what the
physics IS — finalize ordering, the blanket-fill/authored-material restore, the central harmonic
particle-contact coupling, the per-deformable VBD particle config, the realistic-mass override, the
proxy build and the two-way harvest, the substep loop — lives in ``_build_split_mujoco_vbd`` and
``_simulate_split`` and is INHERITED, not copied. This module supplies only the two things a
subclass is supposed to supply: the SCENE (one catalog object, alone, at its own asset frame) and
the POLICY (hold the base still, shake it on schedule, and let the centralized ``GripController``
own the fingers).

It always takes the split MuJoCo+VBD route, even for a rigid object that ``framework``'s central
routing would send to the single-MuJoCo path. That is not a shortcut around the routing rule — it is
the requirement that closing go through the real admittance controller. The rigid-only path has no
gripper proxies, therefore no harvested squeeze signal, therefore ``force_stop_enabled == 0``, and
the controller degrades to a scripted smoothstep to a fixed width. A candidate "validated" under a
scripted jaw width would say nothing about how it behaves under the controller that actually runs.

Nothing here writes to disk; the pass turns a :class:`TrialResult` into an annotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton

from ...assets import (add_rigid_box, add_rubiks_cube, add_soft_block,
                       add_soft_mesh_object, add_ycb_mesh, load_tet_mesh)
from ...framework import GraspExample
from ...grasp_library import MAX_JAW_WIDTH, PREGRASP_MARGIN
from ...grip import GripController, build_gripper_proxies
from ...mathutils import quat_rotate_xyzw
from ...params import (FRANKA, GRIP, RIGID_CUBE, RUBIKS_CUBE, SOFT_BLOCK, GraspWindow,
                       SoftMeshConfig, YcbMeshConfig)
from ...settings import SETTINGS
from ..catalog import OBJECTS_DIR, CatalogAsset
from . import protocol as P
from ...mathutils import quat_to_matrix_xyzw
from .hand import DOF_ANG_JAW, DOF_LIN_APPROACH, FINGER_DOFS, build_floating_hand

TWO_PI = 6.283185307179586
# Anchor used before the close has finished: every shake window then sits in the far future, so the
# base is commanded to hold still without needing a branch in the kernel.
_NEVER = P._NEVER_PROBE


# =================================================================================================
# Policy kernel: the base pose command
# =================================================================================================
@wp.kernel
def _base_target_kernel(
    t_frame: wp.array(dtype=wp.float32),
    substep: int,
    sim_dt: float,
    lin_start: float, lin_end: float,
    ang_start: float, ang_end: float,
    omega: float, lin_amp: float, ang_amp: float,
    joint_target_q: wp.array(dtype=wp.float32),
):
    # dim=1. Writes the six base coordinates only; the two finger DOFs after them belong to the
    # GripController and are never touched here. Coordinates are in the GRASP frame (hand.py), so
    # "translate along the approach" and "rotate about the jaw axis" — ACRONYM's two shake motions —
    # are one coordinate each and need no world-frame trigonometry.
    t = t_frame[0] + float(substep) * sim_dt
    lin = float(0.0)
    ang = float(0.0)
    if t >= lin_start and t <= lin_end:
        u = t - lin_start
        d = lin_end - lin_start
        lin = lin_amp * wp.sin(omega * u) * 0.5 * (1.0 - wp.cos(TWO_PI * u / d))
    if t >= ang_start and t <= ang_end:
        u = t - ang_start
        d = ang_end - ang_start
        ang = ang_amp * wp.sin(omega * u) * 0.5 * (1.0 - wp.cos(TWO_PI * u / d))
    joint_target_q[0] = 0.0
    joint_target_q[1] = 0.0
    joint_target_q[2] = lin
    joint_target_q[3] = ang
    joint_target_q[4] = 0.0
    joint_target_q[5] = 0.0


# =================================================================================================
# Tracking the object's rigid motion (works for a body and a particle cloud alike)
# =================================================================================================
def _kabsch(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Best-fit rotation taking centred ``reference`` onto centred ``current`` (3x3, right-handed).

    A deformable has no pose, so "how far did it rotate" has to mean the RIGID COMPONENT of its
    motion. Kabsch is the least-squares answer and degrades gracefully: for a body that happens not
    to deform it returns exactly the body's rotation."""
    h = reference.T @ current
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


@dataclass
class _Pose:
    center: np.ndarray          # (3,)
    rotation: np.ndarray        # (3,3)


def _pose_delta(a: _Pose, b: _Pose) -> tuple[float, float]:
    """``(linear [m], angular [rad])`` motion from pose ``a`` to pose ``b``."""
    linear = float(np.linalg.norm(b.center - a.center))
    r = b.rotation @ a.rotation.T
    angular = float(np.arccos(np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0)))
    return linear, angular


class _ObjectTracker:
    """Samples the object's rigid pose from the live object state, whatever the object is made of.

    ``bodies`` are the object's rigid bodies (one for a mesh, none for an FEM
    body); ``particles`` is the particle slice. A single rigid body reports its own COM and
    orientation exactly; anything else reports the centroid and the Kabsch rotation of its point
    cloud against the configuration captured at ``reference()``."""

    def __init__(self, model, bodies, particle_range):
        self.bodies = np.asarray(bodies, dtype=np.int64)
        self.p0, self.p1 = particle_range
        self.single_body = len(self.bodies) == 1 and self.p1 <= self.p0
        self._com = model.body_com.numpy()[self.bodies] if len(self.bodies) else None
        self._reference: np.ndarray | None = None

    def points(self, state) -> np.ndarray:
        out = []
        if len(self.bodies):
            out.append(state.body_q.numpy()[self.bodies][:, :3])
        if self.p1 > self.p0:
            out.append(state.particle_q.numpy()[self.p0:self.p1])
        return np.concatenate(out, axis=0)

    def reference(self, state) -> None:
        """Freeze the shape the Kabsch rotation is measured against (called once, at t=0)."""
        self._reference = self.points(state)

    def sample(self, state) -> _Pose:
        if self.single_body:
            q = state.body_q.numpy()[self.bodies[0]]
            return _Pose(center=np.asarray(q[:3]) + quat_rotate_xyzw(q[3:7], self._com[0]),
                         rotation=quat_to_matrix_xyzw(q[3:7]))
        pts = self.points(state)
        ref = self._reference if self._reference is not None else pts
        c, rc = pts.mean(axis=0), ref.mean(axis=0)
        return _Pose(center=c, rotation=_kabsch(ref - rc, pts - c))

    def finite(self, state) -> bool:
        return bool(np.all(np.isfinite(self.points(state))))


# =================================================================================================
# Scene: one catalog object, alone, spawned at its own asset frame
# =================================================================================================
@dataclass
class _Spawned:
    bodies: list = field(default_factory=list)
    particle_range: tuple = (0, 0)
    mass_overrides: list = field(default_factory=list)
    deformable_cfg: object = None        # SoftMeshConfig/SoftBlockConfig -> model.soft_contact_*
    contact_margin: float | None = None
    # Where the spawn believes the object's MATERIAL now is, in world coordinates, for the kinds
    # whose builder needed pre-compensation. None for a single body, whose whole claim is that its
    # transform is the identity. The self-check holds this against the geometry the canonical frame
    # was fitted to — the one way a systematic placement error could stay silent.
    material_points: np.ndarray | None = None


def _config_for(asset: CatalogAsset):
    """The physics config a catalog entry names, built exactly as ``grasp_passes.catalog`` does."""
    cfg = dict(asset.entry.get("config", {}))
    kind = asset.kind
    if kind == "ycb_mesh":
        return YcbMeshConfig(**cfg)
    if kind == "soft_mesh":
        return SoftMeshConfig(**cfg)
    if kind == "rubiks_cube":
        return replace(RUBIKS_CUBE, **cfg) if cfg else RUBIKS_CUBE
    if kind == "rigid_box":
        return replace(RIGID_CUBE, **cfg) if cfg else RIGID_CUBE
    if kind == "soft_block":
        return replace(SOFT_BLOCK, **cfg) if cfg else SOFT_BLOCK
    # kind == "cable" had a rod-spawn branch here until 2026-08-11, when cables left the grasp
    # pipeline's scope entirely (docs/trajPipeline/grasp-library.md; 0/62 held at rest-shape spans).
    raise ValueError(f"{asset.name!r}: shake_validate has no spawn for catalog kind {kind!r}")


def spawn_object(builder: newton.ModelBuilder, asset: CatalogAsset) -> _Spawned:
    """Add the catalog object to ``builder`` AT ITS ASSET FRAME, i.e. ``world_from_body`` = identity.

    That identity is the whole point: a stored candidate is a pose in the canonical frame, and
    ``GraspRecord.grasp_in_world`` composes it with ``world_from_body``. Spawning the object so that
    its asset frame IS the world frame makes the candidate's world pose exactly
    ``record.grasp_in_asset(candidate)``, with no placement to get wrong. The shared builders in
    ``assets.py`` are used verbatim (a pass must not re-author an object's physics); several of them
    re-centre or rest what they are given, so the position handed to them is pre-compensated here to
    land the asset frame back on the origin — the SAME geometry ``grasp_passes.catalog`` fitted the
    canonical frame to."""
    cfg = _config_for(asset)
    out = _Spawned()

    if asset.kind == "ycb_mesh":
        body, _ = add_ycb_mesh(builder, cfg, (0.0, 0.0, 0.0), rest_on_z=None, yaw=0.0)
        out.bodies = [body]
        out.mass_overrides = [(body, cfg.target_mass)]

    elif asset.kind == "rubiks_cube":
        out.bodies = [add_rubiks_cube(builder, (0.0, 0.0, 0.0), cfg)]

    elif asset.kind == "rigid_box":
        body, _ = add_rigid_box(builder, (0.0, 0.0, 0.0), cfg.half_extent, cfg)
        out.bodies = [body]

    elif asset.kind == "soft_mesh":
        # add_soft_mesh_object centres x/y and rests the lowest vertex on center_pos[2]; undo that
        # by handing it the raw mesh's own centre/floor, so the particles land on the raw vertices.
        verts, _tets = load_tet_mesh(OBJECTS_DIR / cfg.tet_subpath)
        verts = np.asarray(verts, dtype=float) * float(cfg.scale)
        p0 = builder.particle_count
        add_soft_mesh_object(builder, cfg, (float(verts[:, 0].mean()), float(verts[:, 1].mean()),
                                            float(verts[:, 2].min())), yaw=0.0)
        out.particle_range = (p0, builder.particle_count)
        out.deformable_cfg = cfg
        out.contact_margin = cfg.contact_margin
        out.material_points = np.asarray(
            [[float(v) for v in q] for q in builder.particle_q[p0:builder.particle_count]])

    elif asset.kind == "soft_block":
        p0 = builder.particle_count
        add_soft_block(builder, cfg, (0.0, 0.0, -0.5 * cfg.dim[2] * cfg.cell))
        out.particle_range = (p0, builder.particle_count)
        out.deformable_cfg = cfg
        out.contact_margin = cfg.contact_margin
        out.material_points = np.asarray(
            [[float(v) for v in q] for q in builder.particle_q[p0:builder.particle_count]])

    else:
        raise ValueError(f"{asset.name!r}: shake_validate has no spawn for kind {asset.kind!r}")

    return out


# =================================================================================================
# The trial
# =================================================================================================
@dataclass
class TrialResult:
    """What one candidate's simulation produced. ``ok`` False means the trial itself failed (the
    solve diverged or the object left the workspace) — reported, never silently dropped."""
    object_in_gripper: float
    motion_closing_linear: float | None
    motion_closing_angular: float | None
    motion_shaking_linear: float | None
    motion_shaking_angular: float | None
    ok: bool = True
    note: str = ""
    force_target: float = 0.0
    object_mass: float = 0.0
    mu_effective: float = 0.0
    final_squeeze: float = 0.0
    pad_contact_fraction: float = 0.0
    # Was the shake actually applied? Achieved peak / commanded amplitude, per axis. This rather
    # than a raw |commanded - achieved|: at 2 Hz that difference is dominated by the base's ~7 deg
    # phase lag, which delays the motion without shrinking it, and would read as an error when
    # nothing is wrong. The ratio is the claim worth checking.
    shake_ratio_linear: float = 0.0
    shake_ratio_angular: float = 0.0
    grip_converged: bool = False       # did the close reach its force target before the timeout
    close_duration: float = 0.0        # [s] how long the close actually took


class GraspTrial(GraspExample):
    """One candidate, one sim. Construct, then :meth:`run`."""

    # framework hooks that a data-driven demo would fill; this rig sets everything in __init__.
    robot_table = None
    blanket_fill = False          # every content shape authors + registers its own material, as in
                                  # the ycb VBD demo; the fill has nothing to add and the restore
                                  # would only re-apply what is already there.
    robot_contact_max = 4096      # the hand alone, with no table and self-collision filtered

    def __init__(self, asset: CatalogAsset, world_from_grasp, protocol: P.ShakeProtocol = P.PROTOCOL,
                 device=None, width: float | None = None):
        newton.use_coord_layout_targets = True
        self.asset = asset
        self.protocol = protocol
        self.world_from_grasp = np.asarray(world_from_grasp, dtype=float)
        self.fps = protocol.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = protocol.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        # Pre-shaped aperture: the trajectory opens to the candidate's width + PREGRASP_MARGIN
        # before the approach, so the trial starts there too (per-finger = half the aperture).
        # width=None (fixture tools) keeps the full-open ACRONYM pre-grasp.
        self.gripper_open = (FRANKA.gripper_open if width is None
                             else 0.5 * min(float(width) + PREGRASP_MARGIN, MAX_JAW_WIDTH))
        self.mass_overrides: list[tuple[int, float]] = []
        self.viewer = None

        device = wp.get_device(device or SETTINGS.device or None)

        # ---- the free-floating hand, placed on the candidate ----
        robot_builder, hand_info = build_floating_hand(self.world_from_grasp, self.gripper_open)
        self.ee_body = hand_info["ee_body"]
        self.robot_finger_bodies = list(hand_info["finger_bodies"])

        # ---- the object, alone, plus the gripper proxies ----
        object_builder = newton.ModelBuilder()
        object_builder.default_shape_cfg.ke = GRIP.proxy_ke
        object_builder.default_shape_cfg.kd = GRIP.object_contact_kd
        object_builder.default_shape_cfg.mu = GRIP.proxy_mu
        spawned = spawn_object(object_builder, asset)
        self.mass_overrides.extend(spawned.mass_overrides)
        self._spawned = spawned
        self.has_particles = spawned.particle_range[1] > spawned.particle_range[0]
        self.soft_block = spawned.deformable_cfg if self.has_particles else None
        # coupling_soft_ke is only the OPT-IN for the proxy<->particle harvest; the framework
        # replaces the value with the true effective contact stiffness. Without it a soft object
        # produces NO squeeze signal at all and the admittance controller could never engage.
        self.coupling_soft_ke = (spawned.deformable_cfg.soft_contact_ke if self.has_particles
                                 else None)
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 8192,
                                     "rigid_body_particle_contact_buffer_size": 4096}
        self.object_pipeline_kwargs = ({"soft_contact_margin": spawned.contact_margin}
                                       if self.has_particles and spawned.contact_margin else {})
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            object_builder, robot_builder, self.robot_finger_bodies, object_table_shape=None)
        object_builder.color(balance_colors=False)

        # ---- the CENTRALIZED build: materials, coupling, particle solver config, masses ----
        args = SimpleNamespace(vbd_iterations=protocol.vbd_iterations)
        self._build_split_mujoco_vbd(robot_builder, object_builder, device, self.robot_contact_max,
                                     args)

        # The hand is weightless and position-controlled (ACRONYM's "unconstrained
        # position-controlled object"): with no gravity on the robot model the base tracks the shake
        # identically whichever way the candidate faces, so orientation cannot leak into the result.
        self.robot_model.set_gravity((0.0, 0.0, 0.0))
        self._set_object_gravity(0.0)          # gravity arrives after the close (protocol.py)

        # ---- what to grip with, derived from the model that was just built ----
        self.object_mass, self.mu_effective = self._object_properties()
        self.force_target = P.force_target(self.object_mass, self.mu_effective,
                                           float(np.max(asset.half_extents)), protocol)
        # ONE grasp window, held open to the end of the trial (no release). close_end is the field's
        # rigid-only-path meaning (when a scripted smoothstep would finish) and is unused on the
        # admittance path this rig always takes; the deadline is the honest value to put there.
        self.grasp_windows = [GraspWindow(close_start=protocol.close_start,
                                          close_end=protocol.close_deadline,
                                          force_target=self.force_target)]
        self._t_frame = wp.zeros(1, dtype=wp.float32, device=device)
        self.grip_controller = GripController(
            self.robot_control, self._t_frame, self.sim_dt, finger_dofs=FINGER_DOFS,
            gripper_open=self.gripper_open, grasp_windows=self.grasp_windows,
            grip_force_signal=self.coupling.grip_force_signal,
            grip_squeeze_signal=self.coupling.grip_squeeze_signal, close_target=0.0)

        # CUDA-graph capture is armed only once gravity is live (run()): the coupling's gravity is a
        # kernel INPUT, so a graph captured before the switch would replay the old value forever and
        # the proxies' momentum-consistent undo would be wrong for the whole shake.
        self.graph = None
        self._frames_simulated = 0
        self._capture_enabled = False
        self._device = device
        self._tracker = _ObjectTracker(self.object_model, spawned.bodies, spawned.particle_range)
        # The disturbance schedule is anchored on whenever the close finishes; until then there is
        # no anchor and the base is commanded to hold still.
        self._t_gravity: float | None = None
        self._close_duration = 0.0
        self._deadband = GRIP.window_params(self.force_target)[2]
        self._sync_gripper_proxies()

    # ---- framework hooks -----------------------------------------------------------------------
    def configure(self, args) -> None:            # everything is set in __init__
        pass

    def plan(self, ik_model, ik_state) -> None:   # no arm, therefore no IK
        pass

    def build_scene(self, object_builder, robot_builder) -> None:
        pass

    def set_arm_targets(self, substep: int) -> None:
        """Hold the base still, then shake it. The window times are launch scalars computed from the
        anchor, so the schedule follows however long this candidate took to grip. Before the anchor
        is set they sit in the far future and the command is a constant zero — and CUDA-graph capture
        is only armed after the anchor is fixed, so a replayed graph never carries stale windows."""
        p = self.protocol
        w = p.windows(self._t_gravity if self._t_gravity is not None else _NEVER)
        wp.launch(_base_target_kernel, dim=1, inputs=[
            self._t_frame, int(substep), self.sim_dt,
            w.linear_start, w.linear_end, w.angular_start, w.angular_end,
            TWO_PI * p.frequency, p.linear_amplitude, p.angular_amplitude,
        ], outputs=[self.robot_control.joint_target_q],
            device=self.robot_control.joint_target_q.device)

    def _grip_converged(self) -> bool:
        """Has the admittance controller reached the force it was asked for?

        Read off the controller's OWN state — the filtered squeeze it regulates and the deadband it
        derives from the target — rather than a threshold invented here, so "converged" means what
        the controller means by it."""
        _width, f_filt, engaged = self.grip_controller.grip_widths()[0]
        if engaged < 0.5:
            return False
        return abs(float(f_filt) - self.force_target) <= self._deadband

    # ---- model queries -------------------------------------------------------------------------
    def _set_object_gravity(self, magnitude: float) -> None:
        """Gravity along the candidate's APPROACH axis — the object hangs out of the jaws.

        The coupling's copy must move with it: it subtracts ``dt * gravity`` when re-pinning the
        dynamic proxies, so a stale value would leave the pads drifting against the object."""
        vec = tuple(float(magnitude) * self.world_from_grasp[:3, 2])
        self.object_model.set_gravity(vec)
        self.coupling.gravity = wp.vec3(*vec)

    def _object_bodies_mask(self) -> np.ndarray:
        """Object bodies only — the three gripper proxies are part of the rig, not the payload."""
        mask = np.ones(self.object_model.body_count, dtype=bool)
        mask[np.asarray(self.gripper_proxy_bodies, dtype=np.int64)] = False
        return mask

    def _object_properties(self) -> tuple[float, float]:
        """``(mass [kg], effective pad<->object friction)`` read off the FINALIZED model.

        Reading the built model rather than the catalog is what keeps the derived force target
        honest: the mass is the realistic post-override one, and the friction is the value the
        contact will actually use — Newton mixes shape friction geometrically, so the pad's
        ``GRIP.proxy_mu`` against the object's own authored ``mu``."""
        mask = self._object_bodies_mask()
        mass = float(self.object_model.body_mass.numpy()[mask].sum())
        mu_values = []
        if self.has_particles:
            mass += float(self.object_model.particle_mass.numpy()
                          [self._spawned.particle_range[0]:self._spawned.particle_range[1]].sum())
            mu_values.append(float(self.object_model.soft_contact_mu))
        shape_mu = self.object_model.shape_material_mu.numpy()
        proxy_set = set(int(s) for s in self.gripper_proxy_shapes)
        flags = self.object_model.shape_flags.numpy()
        for shape in range(self.object_model.shape_count):
            if shape in proxy_set or not (int(flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            mu_values.append(float(shape_mu[shape]))
        mu_object = float(np.mean(mu_values)) if mu_values else GRIP.proxy_mu
        return mass, float(np.sqrt(max(mu_object, 0.0) * GRIP.proxy_mu))

    def squeeze(self) -> float:
        """The harvested closing-axis squeeze [N] — MIN over the two pads, so it is positive only
        while BOTH pads press. This is ACRONYM's "in contact with both fingers" expressed in the
        signal the grip controller itself regulates; see protocol.RETENTION_SQUEEZE_MIN for why the
        literal contact-constraint count is not usable."""
        return float(self.coupling.grip_squeeze_signal.numpy()[0])

    def _base_excursion(self) -> tuple[float, float]:
        """``(|approach translation| [m], |jaw-axis rotation| [rad])`` the base has ACHIEVED right
        now — the rig reporting on itself, so "the shake was applied" is measured, not assumed."""
        q = self.robot_state_0.joint_q.numpy()
        return abs(float(q[DOF_LIN_APPROACH])), abs(float(q[DOF_ANG_JAW]))

    # ---- run -----------------------------------------------------------------------------------
    def run(self) -> TrialResult:
        p = self.protocol
        self._tracker.reference(self.object_state_0)
        pose_ref = self._tracker.sample(self.object_state_0)
        pose_close = None
        held_frames, hold_frames = 0, 0
        peak = (0.0, 0.0)
        settled_since: float | None = None
        converged = False
        window_end = float("inf")

        for frame in range(p.max_frames):
            t = self.sim_time
            if self._t_gravity is None and t >= p.close_start:
                # Close until the controller has held its target for close_settle, or until the
                # timeout. A fixed close window would cut a light object off mid-close (the regulator
                # closes at a rate proportional to the force error) and report "dropped" for a grasp
                # that was never made.
                if self._grip_converged():
                    settled_since = t if settled_since is None else settled_since
                    converged = (t - settled_since) >= p.close_settle
                else:
                    settled_since = None
                if converged or t >= p.close_deadline:
                    # The closing measurement is taken the instant before the load arrives, so
                    # object_motion_during_closing_* is motion caused by the JAWS, not by falling.
                    pose_close = self._tracker.sample(self.object_state_0)
                    self._t_gravity = t
                    self._close_duration = t - p.close_start
                    window_end = p.windows(t).end
                    self._set_object_gravity(p.gravity)
                    # Only NOW is CUDA-graph capture safe: the coupling's gravity AND the shake
                    # windows are kernel inputs, and a graph captured earlier would replay the old
                    # values for the whole disturbance.
                    self.graph = None
                    self._capture_enabled = (self._device.is_cuda and self.sim_substeps % 2 == 0)

            self.step()

            if not self._tracker.finite(self.object_state_0):
                return self._diverged("non-finite object state", frame)
            pose = self._tracker.sample(self.object_state_0)
            # DISPLACEMENT from where the object started, not distance from the grasp centre: a
            # candidate may legitimately grip an elongated object near one end (historically the
            # 0.5 m cable), putting its centroid
            # further from the jaws at rest than any escape threshold could allow.
            escape = float(np.linalg.norm(pose.center - pose_ref.center))
            if escape > P.ESCAPE_RADIUS:
                # Out of the jaws. Stop here: everything after this is free fall, and letting it run
                # would replace "how far did it slip" with "how long was the clock left running".
                speed = (escape / max(t - self._t_gravity, self.frame_dt)
                         if self._t_gravity is not None else 0.0)
                if speed > P.DIVERGENCE_SPEED:
                    return self._diverged(f"object ejected at {speed:.1f} m/s", frame)
                if self._t_gravity is None:
                    # Punted out before the jaws ever closed on it — a mis-centred grasp centre, so
                    # say that rather than reporting it as a drop under load.
                    pose_close = pose
                    why = f"pushed out of the jaws during the close, {t - p.close_start:.2f} s in"
                else:
                    why = f"object left the jaws {t - self._t_gravity:.2f} s after gravity came on"
                return self._result(pose_ref, pose_close, pose, retained=False, peak=peak,
                                    converged=converged, note=f"dropped: {why}")

            if self._t_gravity is not None:
                peak = tuple(map(max, peak, self._base_excursion()))
                if t >= p.windows(self._t_gravity).angular_end:   # the quiescent recover window
                    hold_frames += 1
                    held_frames += int(self.squeeze() > P.RETENTION_SQUEEZE_MIN)
                if self.sim_time >= window_end:
                    break

        fraction = held_frames / max(hold_frames, 1)
        retained = fraction >= P.RETENTION_CONTACT_FRACTION
        return self._result(
            pose_ref, pose_close, self._tracker.sample(self.object_state_0),
            retained=retained, peak=peak, contact_fraction=fraction, converged=converged,
            note=("retained" if retained else "dropped")
                 + f"; both pads pressing {fraction * 100:.0f}% of the hold")

    def _result(self, pose_ref, pose_close, pose_end, *, retained: bool, peak: tuple,
                contact_fraction: float = 0.0, converged: bool = False,
                note: str = "") -> TrialResult:
        if not converged and self._t_gravity is not None:
            note = (f"{note}; grip did NOT reach its {self.force_target:.1f} N target in "
                    f"{self._close_duration:.1f} s of closing")
        close_lin, close_ang = _pose_delta(pose_ref, pose_close)
        shake_lin, shake_ang = _pose_delta(pose_close, pose_end)
        return TrialResult(
            object_in_gripper=1.0 if retained else 0.0,
            motion_closing_linear=close_lin, motion_closing_angular=close_ang,
            motion_shaking_linear=shake_lin, motion_shaking_angular=shake_ang,
            ok=True, note=note, force_target=self.force_target, object_mass=self.object_mass,
            mu_effective=self.mu_effective, final_squeeze=self.squeeze(),
            pad_contact_fraction=contact_fraction, grip_converged=converged,
            close_duration=self._close_duration,
            shake_ratio_linear=peak[0] / self.protocol.linear_amplitude,
            shake_ratio_angular=peak[1] / self.protocol.angular_amplitude)

    def _diverged(self, why: str, frame: int) -> TrialResult:
        """A trial that failed AS A TRIAL. Reported with null motions and a ``diverged`` label rather
        than folded in as a drop: the numbers past a solver failure describe the solver."""
        return TrialResult(
            object_in_gripper=0.0, motion_closing_linear=None, motion_closing_angular=None,
            motion_shaking_linear=None, motion_shaking_angular=None, ok=False,
            note=f"trial diverged at frame {frame}/{self.protocol.max_frames}: {why}",
            force_target=self.force_target, object_mass=self.object_mass,
            mu_effective=self.mu_effective)


def run_trial(asset: CatalogAsset, world_from_grasp, protocol: P.ShakeProtocol = P.PROTOCOL,
              device=None, width: float | None = None) -> TrialResult:
    """Build, run and tear down one trial. One model per candidate: the base joint's frame IS the
    candidate pose, and that is baked at finalize."""
    trial = GraspTrial(asset, world_from_grasp, protocol, device=device, width=width)
    try:
        return trial.run()
    finally:
        trial.graph = None
