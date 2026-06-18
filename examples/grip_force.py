"""Centralized force-limited gripper clamp (explicit applied grasp force).

Background — why this exists
----------------------------
In the split-solver framework (``SolverMuJoCo`` robot + ``SolverVBD`` objects with
kinematic finger proxies) the grasp force used to be *emergent*: the proxy pads were
driven into the object and the VBD penalty/ALM contact produced the squeeze. Measured,
that squeeze is uncontrollable — it slams to ~4900 N on a rigid cube (2500x the ~2 N
actually needed) and, worse, when the kinematic pads accelerate (lift/carry) the stiff
hard-ALM contact (``rigid_avbd_contact_alpha=0``) converts the stored penetration into a
multi-m/s phantom kick that *ejects* the object. Neither a touch-latch nor a width-servo
can cap that force: the force-vs-width curve is near-vertical and the ALM multiplier
accumulates regardless of width.

What this does instead
----------------------
The grip force is produced *explicitly* and is identical for rigid and soft objects:

  * the proxy<->object penalty contact is **suppressed** (the example sets the proxy
    shapes ``has_shape_collision=False``), so it can no longer eject anything;
  * a **commanded clamp force** is applied to the grasped body each substep — the normal
    magnitude ramps linearly ``0 -> max_force`` over ``ramp_time`` on first contact and
    **caps at max_force**; a capped Coulomb friction force (<= ``mu * 2 * max_force``,
    two pads) holds the object to the gripper as it carries.

The pads stay kinematic (the bridge is still one-way: objects cannot push the arm). The
clamp writes only the public ``State.body_f`` / ``State.particle_f`` and reads public
state, so nothing in ``_external`` is touched and it captures cleanly into the CUDA graph.

Rigid bodies are supported here; soft-body particle support is added in a follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class GraspTarget:
    """One grasped rigid body and the window over which the gripper holds it.

    Attributes:
        body: Object-model body index of the grasped body.
        window: ``(t_start, t_end)`` seconds — the clamp is active only in this window
            (mirrors the example's grasp/close window); outside it the clamp releases.
            The window only disambiguates which body a detected contact belongs to; the
            ramp itself is triggered by live contact, not by the window or any dimension.
    """

    body: int
    window: tuple[float, float]


@wp.kernel
def _grip_clamp_rigid_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    proxy_prev: wp.array(dtype=wp.transform),  # [left_prev, right_prev]
    left_proxy: int,
    right_proxy: int,
    tgt_body: wp.array(dtype=wp.int32),
    win0: wp.array(dtype=float),
    win1: wp.array(dtype=float),
    reaction: wp.array(dtype=float),
    touch_eps: float,
    max_force: float,
    ramp_time: float,
    mu: float,
    kp: float,
    kd: float,
    # state (per target)
    engaged: wp.array(dtype=wp.int32),
    engage_t: wp.array(dtype=float),
    anchor_l: wp.array(dtype=wp.vec3),
    anchor_r: wp.array(dtype=wp.vec3),
    # output
    body_f: wp.array(dtype=wp.spatial_vector),
    grip_force: wp.array(dtype=float),
):
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt
    b = tgt_body[i]

    pL = wp.transform_get_translation(body_q[left_proxy])
    pR = wp.transform_get_translation(body_q[right_proxy])
    pLp = wp.transform_get_translation(proxy_prev[0])
    pRp = wp.transform_get_translation(proxy_prev[1])
    vpadL = (pL - pLp) / sim_dt
    vpadR = (pR - pRp) / sim_dt

    Xo = body_q[b]

    in_win = t >= win0[i] and t < win1[i]
    if not in_win:
        engaged[i] = 0
        grip_force[i] = 0.0
        return

    if engaged[i] == 0:
        # Engagement is purely contact-driven (no baked object dimensions): the ramp
        # starts the moment a live pad<->object reaction is detected. The grasp window
        # disambiguates which target body the contact belongs to. Capture each pad's
        # position in the object frame — the body-frame points the two pads grip.
        if wp.max(reaction[0], reaction[1]) > touch_eps:
            engaged[i] = 1
            engage_t[i] = t
            Xo_inv = wp.transform_inverse(Xo)
            anchor_l[i] = wp.transform_point(Xo_inv, pL)
            anchor_r[i] = wp.transform_point(Xo_inv, pR)
        else:
            grip_force[i] = 0.0
            return

    # Linear force ramp 0 -> max_force over ramp_time, then capped.
    ramp = wp.clamp((t - engage_t[i]) / ramp_time, 0.0, 1.0)
    n = max_force * ramp
    grip_force[i] = n
    cap_per = mu * n  # Coulomb friction budget per pad

    # Two-point capped-friction hold: pull the two captured grip points on the body back
    # to the (moving) pad positions, each force capped at mu*n. Two points naturally
    # constrain both translation and rotation — a regularized stick-slip grasp, not a
    # rigid attachment (it slips if the load exceeds the friction budget).
    com = wp.transform_point(Xo, body_com[b])
    gL = wp.transform_point(Xo, anchor_l[i])
    gR = wp.transform_point(Xo, anchor_r[i])
    v = wp.spatial_top(body_qd[b])
    w = wp.spatial_bottom(body_qd[b])
    voL = v + wp.cross(w, gL - com)
    voR = v + wp.cross(w, gR - com)

    fL = kp * (pL - gL) - kd * (voL - vpadL)
    flL = wp.length(fL)
    if flL > cap_per and flL > 0.0:
        fL = fL * (cap_per / flL)
    fR = kp * (pR - gR) - kd * (voR - vpadR)
    flR = wp.length(fR)
    if flR > cap_per and flR > 0.0:
        fR = fR * (cap_per / flR)

    force = fL + fR
    torque = wp.cross(gL - com, fL) + wp.cross(gR - com, fR)
    body_f[b] = body_f[b] + wp.spatial_vector(force, torque)


@wp.kernel
def _update_proxy_prev_kernel(
    body_q: wp.array(dtype=wp.transform),
    left_proxy: int,
    right_proxy: int,
    proxy_prev: wp.array(dtype=wp.transform),
):
    proxy_prev[0] = body_q[left_proxy]
    proxy_prev[1] = body_q[right_proxy]


@wp.kernel
def _grip_reaction_kernel(
    contact_count: wp.array(dtype=wp.int32),
    body0: wp.array(dtype=wp.int32),
    body1: wp.array(dtype=wp.int32),
    force_on_body1: wp.array(dtype=wp.vec3),
    object_body_q: wp.array(dtype=wp.transform),
    left_proxy: int,
    right_proxy: int,
    grip_reaction: wp.array(dtype=float),
):
    # Reduce the per-contact proxy<->object forces to a scalar squeeze per pad along the
    # closing axis (the live contact-detection / force-feedback signal).
    i = wp.tid()
    if i >= contact_count[0]:
        return
    pl = wp.transform_get_translation(object_body_q[left_proxy])
    pr = wp.transform_get_translation(object_body_q[right_proxy])
    axis = wp.normalize(pr - pl)
    b0 = body0[i]
    b1 = body1[i]
    if b1 == left_proxy:
        r = wp.dot(force_on_body1[i], -axis)
        if r > 0.0:
            wp.atomic_add(grip_reaction, 0, r)
    elif b1 == right_proxy:
        r = wp.dot(force_on_body1[i], axis)
        if r > 0.0:
            wp.atomic_add(grip_reaction, 1, r)
    elif b0 == left_proxy:
        r = wp.dot(-force_on_body1[i], -axis)
        if r > 0.0:
            wp.atomic_add(grip_reaction, 0, r)
    elif b0 == right_proxy:
        r = wp.dot(-force_on_body1[i], axis)
        if r > 0.0:
            wp.atomic_add(grip_reaction, 1, r)


@wp.kernel
def _blend_grip_kernel(raw: wp.array(dtype=float), beta: float, filtered: wp.array(dtype=float)):
    i = wp.tid()
    filtered[i] = beta * filtered[i] + (1.0 - beta) * raw[i]


class GripReaction:
    """Per-pad rigid proxy<->object squeeze force [N] from the PUBLIC VBD contact API.

    Snapshot ``body_q`` before the solver step, then call :meth:`update` after it. The
    EMA-filtered per-pad reaction (the contact-detection / force-feedback signal that
    drives the width controllers and the clamp) is exposed as :attr:`reaction`.
    """

    def __init__(self, object_solver, object_contacts, object_state, left_proxy: int,
                 right_proxy: int, filter_beta: float = 0.5):
        device = object_state.body_q.device
        self.object_solver = object_solver
        self.object_contacts = object_contacts
        self.left_proxy = int(left_proxy)
        self.right_proxy = int(right_proxy)
        self.filter_beta = float(filter_beta)
        self._prev = wp.zeros(object_state.body_q.shape[0], dtype=wp.transform, device=device)
        self._raw = wp.zeros(2, dtype=wp.float32, device=device)
        self.reaction = wp.zeros(2, dtype=wp.float32, device=device)

    def snapshot(self, object_state):
        wp.copy(self._prev, object_state.body_q)

    def update(self, object_state, sim_dt: float):
        body0, body1, _p0, _p1, f1, cc = self.object_solver.collect_rigid_contact_forces(
            object_state.body_q, self._prev, self.object_contacts, sim_dt)
        self._raw.zero_()
        wp.launch(_grip_reaction_kernel, dim=body0.shape[0], inputs=[
            cc, body0, body1, f1, object_state.body_q, self.left_proxy, self.right_proxy,
        ], outputs=[self._raw], device=self._raw.device)
        wp.launch(_blend_grip_kernel, dim=2, inputs=[self._raw, self.filter_beta],
                  outputs=[self.reaction], device=self.reaction.device)

    def norms(self):
        r = self.reaction.numpy()
        return float(r[0]), float(r[1])


class GripForceClamp:
    """Applies the explicit capped-Coulomb grasp force to rigid grasped bodies.

    Call :meth:`apply` once per substep after ``clear_forces`` and before the object
    solver step; call :meth:`update_prev` once per substep after it (both go inside the
    captured loop).
    """

    def __init__(
        self,
        object_model,
        object_state,
        left_proxy: int,
        right_proxy: int,
        targets: list[GraspTarget],
        reaction: wp.array,
        max_force: float = 15.0,
        ramp_time: float = 0.5,
        mu: float = 0.8,
        kp: float = 1200.0,
        kd: float = 40.0,
        touch_eps: float = 0.5,
    ):
        device = object_state.body_q.device
        self.left_proxy = int(left_proxy)
        self.right_proxy = int(right_proxy)
        self.max_force = float(max_force)
        self.ramp_time = float(ramp_time)
        self.mu = float(mu)
        self.kp = float(kp)
        self.kd = float(kd)
        self.touch_eps = float(touch_eps)
        self._reaction = reaction
        self._body_com = object_model.body_com

        n = len(targets)
        self._tgt_body = wp.array([t.body for t in targets], dtype=wp.int32, device=device)
        self._win0 = wp.array([t.window[0] for t in targets], dtype=wp.float32, device=device)
        self._win1 = wp.array([t.window[1] for t in targets], dtype=wp.float32, device=device)
        self._engaged = wp.zeros(n, dtype=wp.int32, device=device)
        self._engage_t = wp.zeros(n, dtype=wp.float32, device=device)
        self._anchor_l = wp.zeros(n, dtype=wp.vec3, device=device)
        self._anchor_r = wp.zeros(n, dtype=wp.vec3, device=device)
        self._grip_force = wp.zeros(n, dtype=wp.float32, device=device)
        self._proxy_prev = wp.zeros(2, dtype=wp.transform, device=device)
        self._n = n
        # Seed prev proxy poses so the first substep's pad velocity is finite.
        self.update_prev(object_state)

    def apply(self, object_state, t_frame: wp.array, substep: int, sim_dt: float):
        wp.launch(
            _grip_clamp_rigid_kernel,
            dim=self._n,
            inputs=[
                t_frame, substep, sim_dt,
                object_state.body_q, object_state.body_qd, self._body_com,
                self._proxy_prev, self.left_proxy, self.right_proxy,
                self._tgt_body, self._win0, self._win1,
                self._reaction, self.touch_eps,
                self.max_force, self.ramp_time, self.mu, self.kp, self.kd,
                self._engaged, self._engage_t, self._anchor_l, self._anchor_r,
            ],
            outputs=[object_state.body_f, self._grip_force],
            device=object_state.body_f.device,
        )

    def update_prev(self, object_state):
        wp.launch(
            _update_proxy_prev_kernel,
            dim=1,
            inputs=[object_state.body_q, self.left_proxy, self.right_proxy],
            outputs=[self._proxy_prev],
            device=object_state.body_q.device,
        )

    def grip_forces(self) -> np.ndarray:
        """Per-target commanded clamp force [N] (the ramped normal magnitude)."""
        return self._grip_force.numpy()


@wp.kernel
def _rigid_width_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    reaction: wp.array(dtype=float),
    touch_eps: float,
    release_eps: float,
    close_rate: float,
    gripper_open: float,
    win0: wp.array(dtype=float),
    win1: wp.array(dtype=float),
    n_win: int,
    grip_target: wp.array(dtype=float),
    contacted: wp.array(dtype=wp.int32),
):
    # RIGID width control, contact-driven (no baked dimensions): in a grasp window creep
    # the pads closed until a live reaction is read, then STOP closing and ease OUTward
    # only, just enough to relax the stiff penalty squeeze (the grasp force is supplied
    # by GripForceClamp). Outside every window the pads open.
    t = t_frame[0] + float(substep) * sim_dt
    in_win = int(0)
    for k in range(n_win):
        if t >= win0[k] and t < win1[k]:
            in_win = 1
    if in_win == 0:
        grip_target[0] = gripper_open
        contacted[0] = 0
        return
    r = wp.max(reaction[0], reaction[1])
    if r > touch_eps:
        contacted[0] = 1
    if contacted[0] == 0:
        grip_target[0] = wp.max(grip_target[0] - close_rate * sim_dt, 0.0)
    elif r > release_eps:
        grip_target[0] = grip_target[0] + close_rate * sim_dt


@wp.kernel
def _soft_width_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    reaction: wp.array(dtype=float),
    touch_eps: float,
    max_force: float,
    ramp_time: float,
    close_rate: float,
    gripper_open: float,
    win0: wp.array(dtype=float),
    win1: wp.array(dtype=float),
    n_win: int,
    grip_target: wp.array(dtype=float),
    engaged: wp.array(dtype=wp.int32),
    engage_t: wp.array(dtype=float),
):
    # SOFT (compressible) width control: keep squeezing until the MEASURED per-pad soft
    # reaction reaches the ramped setpoint (0 -> max_force over ramp_time), capped at
    # max_force. The real soft-contact force is the grip; a bidirectional width servo
    # tracks the setpoint (works because the soft force is a gradual function of
    # compression, unlike the near-vertical rigid curve).
    t = t_frame[0] + float(substep) * sim_dt
    in_win = int(0)
    for k in range(n_win):
        if t >= win0[k] and t < win1[k]:
            in_win = 1
    if in_win == 0:
        grip_target[0] = gripper_open
        engaged[0] = 0
        return
    r = wp.max(reaction[0], reaction[1])
    if engaged[0] == 0:
        if r > touch_eps:
            engaged[0] = 1
            engage_t[0] = t
        else:
            grip_target[0] = wp.max(grip_target[0] - close_rate * sim_dt, 0.0)
        return
    setpoint = max_force * wp.clamp((t - engage_t[0]) / ramp_time, 0.0, 1.0)
    if r < setpoint:
        grip_target[0] = wp.max(grip_target[0] - close_rate * sim_dt, 0.0)
    elif r > setpoint:
        grip_target[0] = grip_target[0] + close_rate * sim_dt


class _WidthController:
    """Base: holds the shared grip-target width array and the grasp windows."""

    def __init__(self, grip_target: wp.array, windows: list[tuple[float, float]],
                 close_rate: float, gripper_open: float, touch_eps: float = 0.5):
        device = grip_target.device
        self.grip_target = grip_target
        self.close_rate = float(close_rate)
        self.gripper_open = float(gripper_open)
        self.touch_eps = float(touch_eps)
        self._win0 = wp.array([w[0] for w in windows], dtype=wp.float32, device=device)
        self._win1 = wp.array([w[1] for w in windows], dtype=wp.float32, device=device)
        self._n_win = len(windows)


class RigidGripWidth(_WidthController):
    """Contact-driven width control for rigid grasps (stop at first contact, ease out)."""

    def __init__(self, grip_target, windows, reaction, close_rate=0.02, gripper_open=0.04,
                 touch_eps=0.5, release_eps=2.0):
        super().__init__(grip_target, windows, close_rate, gripper_open, touch_eps)
        self.release_eps = float(release_eps)
        self._reaction = reaction
        self._contacted = wp.zeros(1, dtype=wp.int32, device=grip_target.device)

    def update(self, t_frame: wp.array, substep: int, sim_dt: float):
        wp.launch(_rigid_width_kernel, dim=1, inputs=[
            t_frame, substep, sim_dt, self._reaction, self.touch_eps, self.release_eps,
            self.close_rate, self.gripper_open, self._win0, self._win1, self._n_win,
        ], outputs=[self.grip_target, self._contacted], device=self.grip_target.device)


class SoftGripWidth(_WidthController):
    """Squeeze-to-force width control for compressible grasps (close until measured
    per-pad reaction reaches the ramped 0->max_force setpoint)."""

    def __init__(self, grip_target, windows, reaction, max_force=15.0, ramp_time=0.5,
                 close_rate=0.02, gripper_open=0.04, touch_eps=0.5):
        super().__init__(grip_target, windows, close_rate, gripper_open, touch_eps)
        self.max_force = float(max_force)
        self.ramp_time = float(ramp_time)
        self._reaction = reaction
        self._engaged = wp.zeros(1, dtype=wp.int32, device=grip_target.device)
        self._engage_t = wp.zeros(1, dtype=wp.float32, device=grip_target.device)

    def update(self, t_frame: wp.array, substep: int, sim_dt: float):
        wp.launch(_soft_width_kernel, dim=1, inputs=[
            t_frame, substep, sim_dt, self._reaction, self.touch_eps, self.max_force,
            self.ramp_time, self.close_rate, self.gripper_open, self._win0, self._win1, self._n_win,
        ], outputs=[self.grip_target, self._engaged, self._engage_t], device=self.grip_target.device)
