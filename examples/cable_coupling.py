"""Two-way dynamic-proxy coupling between the MuJoCo robot and the VBD cable.

Background
----------
The split-solver bridge in these demos was historically ONE-WAY (robot -> objects):
kinematic zero-mass finger proxies mirrored the finger poses into the VBD model, so the
arm never felt the object and grips had to be faked with a commanded interference. For a
light flexible rod that is fragile.

This module implements NVIDIA's two-way cable-manipulation recipe (developer.nvidia.com
"Newton adds contact-rich manipulation"): the finger proxies are DYNAMIC finite-mass
bodies, the cable's contact wrench on them is harvested and fed back onto the robot
(one-step lag), and the proxies are re-pinned to the finger pose each substep with the
gravity + coupling-force velocity deltas pre-subtracted so they stay slaved to the robot
while still participating as finite-mass contact bodies in the VBD solve. The grasp is then
emergent two-way contact — the arm feels the cable.

Per substep the host loop calls, in order:
    coupling.apply_to_robot(robot_state)   # after robot.clear_forces(), before robot.step()
    ... robot.step(), swap ...
    coupling.sync_proxies(robot_state, obj_state_0, obj_state_1)
    coupling.snapshot(obj_state_0)
    ... object.collide(), object.step(), swap ...
    coupling.harvest(obj_state_0)          # collect cable->proxy wrench for next step

All public Newton API (``State.body_f``, ``collect_rigid_contact_forces``); nothing in
``_external`` is modified. Everything is CUDA-graph capturable.
"""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _apply_coupling_to_fingers_kernel(
    force_lag: wp.array(dtype=wp.vec3),
    torque_lag: wp.array(dtype=wp.vec3),
    proxy_bodies: wp.array(dtype=wp.int32),
    finger_bodies: wp.array(dtype=wp.int32),
    robot_body_f: wp.array(dtype=wp.spatial_vector),
):
    # Force-limited mode: route each pad's lagged reaction to its OWN finger DOF so the
    # effort-limited gripper feels the cable push-back and backs off — this is what lets
    # the penetration (and the ALM multiplier) resolve at the actuator-force equilibrium
    # instead of ramping. Distinct fingers per proxy → no write race.
    i = wp.tid()
    pb = proxy_bodies[i]
    fb = finger_bodies[i]
    robot_body_f[fb] = robot_body_f[fb] + wp.spatial_vector(force_lag[pb], torque_lag[pb])


@wp.kernel
def _apply_coupling_to_ee_kernel(
    force_lag: wp.array(dtype=wp.vec3),
    torque_lag: wp.array(dtype=wp.vec3),
    proxy_bodies: wp.array(dtype=wp.int32),
    finger_bodies: wp.array(dtype=wp.int32),
    n_proxy: int,
    ee_body: int,
    robot_body_q: wp.array(dtype=wp.transform),
    robot_body_com: wp.array(dtype=wp.vec3),
    robot_body_f: wp.array(dtype=wp.spatial_vector),
):
    # Apply the previous step's harvested cable->proxy reaction onto the ARM (EE body),
    # not the finger DOFs: the SUM of the two pad wrenches cancels the internal squeeze
    # and leaves the real external cable load (weight + sweep reaction), so the arm feels
    # the cable while the position-controlled fingers keep their grip. Each pad wrench
    # (about the finger origin) is transferred to the EE COM. dim=1 (single EE body).
    c = wp.transform_point(robot_body_q[ee_body], robot_body_com[ee_body])
    f_tot = wp.vec3(0.0, 0.0, 0.0)
    t_tot = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n_proxy):
        pb = proxy_bodies[k]
        p = wp.transform_get_translation(robot_body_q[finger_bodies[k]])
        f = force_lag[pb]
        f_tot = f_tot + f
        t_tot = t_tot + torque_lag[pb] + wp.cross(p - c, f)
    robot_body_f[ee_body] = robot_body_f[ee_body] + wp.spatial_vector(f_tot, t_tot)


@wp.kernel
def _sync_proxy_pose_kernel(
    robot_body_q: wp.array(dtype=wp.transform),
    finger_bodies: wp.array(dtype=wp.int32),
    proxy_bodies: wp.array(dtype=wp.int32),
    object_body_q_0: wp.array(dtype=wp.transform),
    object_body_q_1: wp.array(dtype=wp.transform),
):
    # Kinematic proxy: mirror the finger pose only (VBD doesn't integrate kinematic
    # bodies, so no velocity / gravity / coupling undo is needed). The contact reaction
    # is still harvestable — collect_rigid_contact_forces returns the penalty/ALM force
    # k*C+lambda from penetration, independent of body mass.
    i = wp.tid()
    tf = robot_body_q[finger_bodies[i]]
    object_body_q_0[proxy_bodies[i]] = tf
    object_body_q_1[proxy_bodies[i]] = tf


@wp.kernel
def _sync_proxy_state_kernel(
    robot_body_q: wp.array(dtype=wp.transform),
    robot_body_qd: wp.array(dtype=wp.spatial_vector),
    finger_bodies: wp.array(dtype=wp.int32),
    proxy_bodies: wp.array(dtype=wp.int32),
    force_lag: wp.array(dtype=wp.vec3),
    torque_lag: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
    object_body_q_0: wp.array(dtype=wp.transform),
    object_body_qd_0: wp.array(dtype=wp.spatial_vector),
    object_body_q_1: wp.array(dtype=wp.transform),
    object_body_qd_1: wp.array(dtype=wp.spatial_vector),
):
    # Re-pin the dynamic proxy to the finger pose+velocity, then subtract the velocity
    # change that gravity + the (lagged) contact wrench WILL impart during the VBD step,
    # so after VBD integrates them the proxy ends at the finger's velocity. Mirrors
    # Newton's integrate_rigid_body (solver.py): v += (f*inv_m + g*nonzero(inv_m))*dt,
    # angular delta = R*(inv_I*(R^-1*tau))*dt.
    i = wp.tid()
    fb = finger_bodies[i]
    pb = proxy_bodies[i]
    q = robot_body_q[fb]
    qd = robot_body_qd[fb]
    invm = body_inv_mass[pb]
    r = wp.transform_get_rotation(q)
    dv = dt * (invm * force_lag[pb] + gravity * wp.nonzero(invm))
    dw = dt * wp.quat_rotate(r, body_inv_inertia[pb] * wp.quat_rotate_inv(r, torque_lag[pb]))
    qd_new = qd - wp.spatial_vector(dv, dw)
    object_body_q_0[pb] = q
    object_body_qd_0[pb] = qd_new
    object_body_q_1[pb] = q
    object_body_qd_1[pb] = qd_new


@wp.kernel
def _filter_clamp_wrench_kernel(
    proxy_bodies: wp.array(dtype=wp.int32),
    proxy_force: wp.array(dtype=wp.vec3),
    proxy_torque: wp.array(dtype=wp.vec3),
    beta: float,          # EMA retention on the lagged wrench
    scale: float,         # coupling gain (<1 under-couples for stability)
    force_clamp: float,   # max |force| fed to the robot [N]
    torque_clamp: float,  # max |torque| fed to the robot [N·m]
    force_lag: wp.array(dtype=wp.vec3),
    torque_lag: wp.array(dtype=wp.vec3),
):
    # Smooth + bound the harvested wrench before it is fed (lagged) to the robot. The
    # raw penalty/ALM force is spiky during the dynamic lift; an unfiltered/unbounded
    # feedback resonates against the position-controlled finger (one-step lag + stiff
    # contact) and ejects the arm. EMA + clamp + gain keep the exchange stable.
    i = wp.tid()
    pb = proxy_bodies[i]
    f = scale * proxy_force[pb]
    fl = wp.length(f)
    if fl > force_clamp and fl > 0.0:
        f = f * (force_clamp / fl)
    tq = scale * proxy_torque[pb]
    tl = wp.length(tq)
    if tl > torque_clamp and tl > 0.0:
        tq = tq * (torque_clamp / tl)
    force_lag[pb] = beta * force_lag[pb] + (1.0 - beta) * f
    torque_lag[pb] = beta * torque_lag[pb] + (1.0 - beta) * tq


@wp.kernel
def _harvest_proxy_wrench_kernel(
    contact_count: wp.array(dtype=wp.int32),
    body0: wp.array(dtype=wp.int32),
    body1: wp.array(dtype=wp.int32),
    point0_world: wp.array(dtype=wp.vec3),
    point1_world: wp.array(dtype=wp.vec3),
    force_on_body1: wp.array(dtype=wp.vec3),
    object_body_q: wp.array(dtype=wp.transform),
    left_proxy: int,
    right_proxy: int,
    proxy_force: wp.array(dtype=wp.vec3),
    proxy_torque: wp.array(dtype=wp.vec3),
):
    # Sum, per proxy, the contact force the other body (cable) exerts on it and the
    # torque about the proxy COM. force_on_body1 is the force on body1; body0 gets -it.
    i = wp.tid()
    if i >= contact_count[0]:
        return
    b0 = body0[i]
    b1 = body1[i]
    if b1 == left_proxy or b1 == right_proxy:
        f = force_on_body1[i]
        com = wp.transform_get_translation(object_body_q[b1])
        wp.atomic_add(proxy_force, b1, f)
        wp.atomic_add(proxy_torque, b1, wp.cross(point1_world[i] - com, f))
    elif b0 == left_proxy or b0 == right_proxy:
        f = -force_on_body1[i]
        com = wp.transform_get_translation(object_body_q[b0])
        wp.atomic_add(proxy_force, b0, f)
        wp.atomic_add(proxy_torque, b0, wp.cross(point0_world[i] - com, f))


class TwoWayProxyCoupling:
    """Dynamic-proxy two-way bridge between a MuJoCo robot and a VBD object model."""

    def __init__(self, robot_model, object_model, object_solver, object_contacts,
                 object_state, robot_finger_bodies, proxy_bodies, ee_body, sim_dt,
                 gravity=(0.0, 0.0, -9.81), proxy_dynamic=False, feedback_to_fingers=False,
                 wrench_filter=0.5, coupling_gain=1.0, force_clamp=20.0, torque_clamp=1.0):
        device = object_state.body_q.device
        self.ee_body = int(ee_body)
        self.feedback_to_fingers = bool(feedback_to_fingers)
        self._robot_body_com = robot_model.body_com
        self.object_solver = object_solver
        self.object_contacts = object_contacts
        self.sim_dt = float(sim_dt)
        self.wrench_filter = float(wrench_filter)
        self.coupling_gain = float(coupling_gain)
        self.force_clamp = float(force_clamp)
        self.torque_clamp = float(torque_clamp)
        # proxy_dynamic=True follows the article literally (finite-mass proxies with the
        # gravity/coupling velocity undo) but free-body dynamic proxies NaN in this Newton
        # build's VBD contact/articulation solve. proxy_dynamic=False keeps the proven
        # KINEMATIC pose-mirror proxy and still harvests its contact reaction (the penalty
        # force is mass-independent) — a stable two-way bridge.
        self.proxy_dynamic = bool(proxy_dynamic)
        self.gravity = wp.vec3(*gravity)
        self._body_inv_mass = object_model.body_inv_mass
        self._body_inv_inertia = object_model.body_inv_inertia
        self._finger_bodies = wp.array(robot_finger_bodies, dtype=wp.int32, device=device)
        self._proxy_bodies = wp.array(proxy_bodies, dtype=wp.int32, device=device)
        self.left_proxy, self.right_proxy = (int(b) for b in proxy_bodies)
        self._n = len(proxy_bodies)

        nb = object_model.body_count
        self._proxy_force = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._proxy_torque = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._force_lag = wp.zeros(nb, dtype=wp.vec3, device=device)      # arm feedback (clamped/filtered)
        self._torque_lag = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._raw_force_lag = wp.zeros(nb, dtype=wp.vec3, device=device)  # raw (for momentum-consistent undo)
        self._raw_torque_lag = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._obj_body_q_prev = wp.zeros(nb, dtype=wp.transform, device=device)

    def apply_to_robot(self, robot_state):
        if self.feedback_to_fingers:
            # Per-pad reaction onto the fingers (force-limited mode): lets the effort-
            # limited gripper back off so the grip equilibrates at the actuator force.
            wp.launch(_apply_coupling_to_fingers_kernel, dim=self._n, inputs=[
                self._force_lag, self._torque_lag, self._proxy_bodies, self._finger_bodies,
            ], outputs=[robot_state.body_f], device=robot_state.body_f.device)
            return
        # M1 default: net reaction onto the EE/arm (squeeze cancels) — arm feels the cable
        # without opening the position-held grip.
        wp.launch(_apply_coupling_to_ee_kernel, dim=1, inputs=[
            self._force_lag, self._torque_lag, self._proxy_bodies, self._finger_bodies,
            self._n, self.ee_body, robot_state.body_q, self._robot_body_com,
        ], outputs=[robot_state.body_f], device=robot_state.body_f.device)

    def sync_proxies(self, robot_state, object_state_0, object_state_1):
        if not self.proxy_dynamic:
            wp.launch(_sync_proxy_pose_kernel, dim=self._n,
                      inputs=[robot_state.body_q, self._finger_bodies, self._proxy_bodies],
                      outputs=[object_state_0.body_q, object_state_1.body_q],
                      device=object_state_0.body_q.device)
            return
        # The undo MUST use the RAW lagged contact force (what VBD actually applies to the
        # proxy), not the clamped/filtered arm-feedback copy — else it can't cancel the
        # proxy's contact velocity change and the body diverges. Momentum-consistent.
        wp.launch(_sync_proxy_state_kernel, dim=self._n, inputs=[
            robot_state.body_q, robot_state.body_qd, self._finger_bodies, self._proxy_bodies,
            self._raw_force_lag, self._raw_torque_lag, self._body_inv_mass, self._body_inv_inertia,
            self.gravity, self.sim_dt,
        ], outputs=[
            object_state_0.body_q, object_state_0.body_qd,
            object_state_1.body_q, object_state_1.body_qd,
        ], device=object_state_0.body_q.device)

    def snapshot(self, object_state_0):
        wp.copy(self._obj_body_q_prev, object_state_0.body_q)

    def harvest(self, object_state_0):
        body0, body1, p0, p1, f1, cc = self.object_solver.collect_rigid_contact_forces(
            object_state_0.body_q, self._obj_body_q_prev, self.object_contacts, self.sim_dt)
        self._proxy_force.zero_()
        self._proxy_torque.zero_()
        wp.launch(_harvest_proxy_wrench_kernel, dim=body0.shape[0], inputs=[
            cc, body0, body1, p0, p1, f1, object_state_0.body_q, self.left_proxy, self.right_proxy,
        ], outputs=[self._proxy_force, self._proxy_torque], device=self._proxy_force.device)
        # Raw lagged wrench (un-clamped) — drives the momentum-consistent sync undo.
        wp.copy(self._raw_force_lag, self._proxy_force)
        wp.copy(self._raw_torque_lag, self._proxy_torque)
        # Smooth + bound + gain → the lagged wrench fed to the ARM (fallback safety bounds).
        wp.launch(_filter_clamp_wrench_kernel, dim=self._n, inputs=[
            self._proxy_bodies, self._proxy_force, self._proxy_torque, self.wrench_filter,
            self.coupling_gain, self.force_clamp, self.torque_clamp,
        ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)

    def wrench_norms(self):
        """Per-proxy (|force| [N], |torque| [N·m]) of the clamped arm-feedback wrench (host)."""
        f = self._force_lag.numpy()
        t = self._torque_lag.numpy()
        return [(float(np.linalg.norm(f[p])), float(np.linalg.norm(t[p])))
                for p in (self.left_proxy, self.right_proxy)]

    def raw_force_norms(self):
        """Per-proxy |force| [N] of the RAW contact reaction — the actual force the
        grasped object feels (un-clamped). This is what a 15 N limit must bring down."""
        f = self._proxy_force.numpy()
        return [float(np.linalg.norm(f[p])) for p in (self.left_proxy, self.right_proxy)]
