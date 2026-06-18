"""Two-way dynamic-proxy coupling between the MuJoCo robot and the VBD cable.

Implements NVIDIA's two-way cable-manipulation recipe (developer.nvidia.com "Newton adds
contact-rich manipulation"): the finger proxies are DYNAMIC finite-mass bodies living in the
VBD object model and mirroring the robot finger poses. Each substep the proxy is re-pinned to
the finger pose+velocity with the gravity + (lagged) coupling-force velocity deltas
pre-subtracted (momentum-consistent "undo"), so it stays slaved to the robot while still
participating as a finite-mass contact body in the VBD solve. The cable's contact wrench on the
proxies is harvested and the NET (both pads summed → internal squeeze cancels, external load
remains) is fed back onto the robot ARM/EE one substep later. The grasp is emergent two-way
contact: the arm genuinely feels the cable while the position-held fingers keep their grip, and
the grip force is whatever the squeeze produces against the bounded contact — finite and physical,
no force cap. Per-pad reaction is available via ``raw_force_norms()`` (tactile), but is NOT fed to
the finger DOFs (that pushes the pads open and loses the grasp).

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
    # Apply the previous step's harvested cable->proxy reaction onto the ARM (EE body), NOT the
    # finger bodies: the SUM of the two pad wrenches cancels the internal squeeze and leaves the
    # real external cable load (weight + sweep reaction), so the arm feels the cable while the
    # position-controlled fingers keep their grip. Feeding the per-pad squeeze to the fingers
    # instead pushes them open and loses the grasp (the "no continuous feedback into the gripper
    # DOF" invariant). Each pad wrench (about the finger origin) is transferred to the EE COM.
    # dim=1 (single EE body). body_f is [linear; angular] (see solver.integrate_rigid_body).
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
    # Re-pin the dynamic proxy to the finger pose+velocity, then subtract the velocity change
    # that gravity + the (lagged) contact wrench WILL impart during the VBD step, so after VBD
    # integrates them the proxy ends at the finger's velocity. Mirrors solver.integrate_rigid_body:
    # v += (f*inv_m + g*nonzero(inv_m))*dt, angular delta = R*(inv_I*(R^-1*tau))*dt. The undo uses
    # the SAME lagged wrench that apply_to_robot fed the robot this substep (momentum-consistent);
    # the residual (current - lagged) is small once the grip force is bounded.
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
    # Sum, per proxy, the contact force the other body (cable) exerts on it and the torque about
    # the proxy COM. force_on_body1 is the force on body1; body0 gets -it.
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

    def __init__(self, robot_model, object_model, object_solver, object_contacts, object_state,
                 robot_finger_bodies, proxy_bodies, ee_body, sim_dt, gravity=(0.0, 0.0, -9.81)):
        device = object_state.body_q.device
        self.object_solver = object_solver
        self.object_contacts = object_contacts
        self.sim_dt = float(sim_dt)
        self.gravity = wp.vec3(*gravity)
        self.ee_body = int(ee_body)
        self._robot_body_com = robot_model.body_com
        self._body_inv_mass = object_model.body_inv_mass
        self._body_inv_inertia = object_model.body_inv_inertia
        self._finger_bodies = wp.array(robot_finger_bodies, dtype=wp.int32, device=device)
        self._proxy_bodies = wp.array(proxy_bodies, dtype=wp.int32, device=device)
        self.left_proxy, self.right_proxy = (int(b) for b in proxy_bodies)
        self._n = len(proxy_bodies)

        nb = object_model.body_count
        # ONE raw lagged-wrench pair, fed to BOTH the momentum-consistent undo (sync_proxies) and
        # the robot feedback (apply_to_robot, net-to-EE) — they must use the identical wrench for
        # the one-step-lag bookkeeping to be consistent. No clamp/EMA: the grip force is bounded by
        # physics (bounded contact), so the raw reaction is fed straight through.
        self._force_lag = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._torque_lag = wp.zeros(nb, dtype=wp.vec3, device=device)
        self._obj_body_q_prev = wp.zeros(nb, dtype=wp.transform, device=device)

    def apply_to_robot(self, robot_state):
        wp.launch(_apply_coupling_to_ee_kernel, dim=1, inputs=[
            self._force_lag, self._torque_lag, self._proxy_bodies, self._finger_bodies,
            self._n, self.ee_body, robot_state.body_q, self._robot_body_com,
        ], outputs=[robot_state.body_f], device=robot_state.body_f.device)

    def sync_proxies(self, robot_state, object_state_0, object_state_1):
        wp.launch(_sync_proxy_state_kernel, dim=self._n, inputs=[
            robot_state.body_q, robot_state.body_qd, self._finger_bodies, self._proxy_bodies,
            self._force_lag, self._torque_lag, self._body_inv_mass, self._body_inv_inertia,
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
        self._force_lag.zero_()
        self._torque_lag.zero_()
        wp.launch(_harvest_proxy_wrench_kernel, dim=body0.shape[0], inputs=[
            cc, body0, body1, p0, p1, f1, object_state_0.body_q, self.left_proxy, self.right_proxy,
        ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)

    def raw_force_norms(self):
        """Per-proxy |force| [N] of the harvested cable reaction — the actual force the grasped
        cable exerts on each pad (raw, un-clamped). With the dynamic two-way grip this is the
        physical grip force; it should be finite and bounded (not the kinematic 1e4-1e6 N)."""
        f = self._force_lag.numpy()
        return [float(np.linalg.norm(f[p])) for p in (self.left_proxy, self.right_proxy)]
