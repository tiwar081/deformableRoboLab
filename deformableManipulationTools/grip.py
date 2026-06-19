"""Generalized two-way dynamic-proxy grip coupling between the MuJoCo robot and a VBD object.

NVIDIA's two-way recipe, generalized to any gripped VBD object (cable rod, rigid box, FEM block):
the finger proxies are DYNAMIC finite-mass bodies in the object model mirroring the fingers; each
substep they are re-pinned to the finger pose+velocity with the gravity + (lagged) contact-wrench
velocity deltas pre-subtracted (momentum-consistent "undo"), so they stay slaved to the robot
while still participating as finite-mass contact bodies in the VBD solve. The object's contact
reaction on the proxies is harvested and the NET (both pads summed → internal squeeze cancels,
external load remains) is fed back onto the robot ARM/EE one substep later. Grip force is the
position-controlled squeeze against bounded contact — finite and physical, NO force cap.

Two harvest paths, both accumulated into the same lagged wrench:
  * rigid objects  → ``SolverVBD.collect_rigid_contact_forces`` (rigid↔rigid contact).
  * soft particles → re-computed ``ke·penetration`` over the public ``Contacts.soft_contact_*``
    geometry (Newton exposes no body↔particle force API), enabled when the proxies carry
    ``has_particle_collision=True``.

Per substep the host loop calls, in order:
    coupling.apply_to_robot(robot_state)   # after robot.clear_forces(), before robot.step()
    ... robot.step(), swap ...
    coupling.sync_proxies(robot_state, obj_state_0, obj_state_1)
    coupling.snapshot(obj_state_0)
    ... object.collide(), object.step(), swap ...
    coupling.harvest(obj_state_0)          # collect object->proxy wrench for next step

All public Newton API; nothing in ``_external`` is modified. Everything is CUDA-graph capturable.
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .params import GRIP, GripConfig


# ---------------------------------------------------------------------------------------------
# Dynamic finger proxies (the grip contact bridge consumed by TwoWayProxyCoupling below).
# ---------------------------------------------------------------------------------------------
def build_gripper_proxies(object_builder, robot_builder, finger_bodies: list[int],
                          object_table_shape: int | None, gap: float, grip: GripConfig = GRIP,
                          has_particle_collision: bool = False):
    """Add the two dynamic finite-mass finger proxies to ``object_builder`` (collision shapes
    copied from the robot fingers), filtered against the object-model table. Set
    ``has_particle_collision=True`` to also grip FEM-particle objects (soft blocks). Returns
    (proxy_bodies, proxy_shapes)."""
    proxy_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, is_visible=False, has_shape_collision=True,
        has_particle_collision=has_particle_collision,
        margin=grip.proxy_margin, gap=gap, ke=grip.proxy_ke, kd=grip.proxy_kd, mu=grip.proxy_mu,
    )
    proxy_bodies, proxy_shapes = [], []
    inertia = wp.mat33(grip.proxy_inertia, 0.0, 0.0, 0.0, grip.proxy_inertia, 0.0, 0.0, 0.0, grip.proxy_inertia)
    for label, finger_body in zip(("left_gripper_contact_proxy", "right_gripper_contact_proxy"),
                                  finger_bodies, strict=True):
        body = object_builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            is_kinematic=False, mass=grip.proxy_mass, inertia=inertia,
            com=wp.vec3(0.0, 0.0, 0.0), lock_inertia=True, label=label,
        )
        proxy_bodies.append(body)
        n = 0
        for shape_idx, shape_body in enumerate(robot_builder.shape_body):
            if shape_body != finger_body:
                continue
            if not (robot_builder.shape_flags[shape_idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            shape = object_builder.add_shape(
                body=body, type=robot_builder.shape_type[shape_idx],
                xform=robot_builder.shape_transform[shape_idx], cfg=proxy_cfg,
                scale=robot_builder.shape_scale[shape_idx], src=robot_builder.shape_source[shape_idx],
                label=f"{label}_shape_{n}",
            )
            proxy_shapes.append(shape)
            n += 1
        if n == 0:
            raise RuntimeError(f"No colliding shapes on Franka finger body {finger_body}.")
    # A dynamic proxy re-pinned against the static table resolves explosively; the robot-side
    # table already stops the real fingers, so this object-model contact is redundant.
    if object_table_shape is not None:
        for shape in proxy_shapes:
            object_builder.add_shape_collision_filter_pair(object_table_shape, shape)
    return proxy_bodies, proxy_shapes


def restore_proxy_materials(object_model, proxy_shapes: list[int], grip: GripConfig = GRIP):
    """Re-apply proxy contact material after any blanket ``shape_material_*.fill_`` (which would
    clobber it). Must run after finalize."""
    if not proxy_shapes:
        return
    mu = object_model.shape_material_mu.numpy()
    ke = object_model.shape_material_ke.numpy()
    kd = object_model.shape_material_kd.numpy()
    for shape in proxy_shapes:
        mu[shape] = grip.proxy_mu
        ke[shape] = grip.proxy_ke
        kd[shape] = grip.proxy_kd
    object_model.shape_material_mu.assign(mu)
    object_model.shape_material_ke.assign(ke)
    object_model.shape_material_kd.assign(kd)


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
    # Apply the previous step's harvested object->proxy reaction onto the ARM (EE body), NOT the
    # finger bodies: the SUM of the two pad wrenches cancels the internal squeeze and leaves the
    # real external load (weight + motion reaction), so the arm feels the object while the
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
    # Re-pin the dynamic proxy to the finger pose+velocity, then subtract the velocity change that
    # gravity + the (lagged) contact wrench WILL impart during the VBD step, so after VBD integrates
    # them the proxy ends at the finger's velocity. Mirrors solver.integrate_rigid_body. The undo
    # uses the SAME lagged wrench apply_to_robot fed the robot this substep (momentum-consistent);
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
    # Rigid harvest: sum, per proxy, the rigid contact force the other body exerts on it and the
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


@wp.kernel
def _harvest_soft_wrench_kernel(
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=wp.int32),
    soft_contact_shape: wp.array(dtype=wp.int32),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    object_body_q: wp.array(dtype=wp.transform),
    soft_contact_ke: float,
    left_proxy: int,
    right_proxy: int,
    proxy_force: wp.array(dtype=wp.vec3),
    proxy_torque: wp.array(dtype=wp.vec3),
):
    # Soft harvest: Newton exposes no body-particle force readback, so recompute the penalty force
    # (n·ke·penetration, matching VBD's own body-particle law) the soft block exerts on each proxy
    # pad, from the PUBLIC soft-contact geometry. Accumulate force + torque about the proxy COM.
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    pid = soft_contact_particle[i]
    shape = soft_contact_shape[i]
    if pid < 0 or shape < 0:
        return
    body = shape_body[shape]
    if body != left_proxy and body != right_proxy:
        return
    bx = wp.transform_point(object_body_q[body], soft_contact_body_pos[i])
    n = soft_contact_normal[i]
    pen = -(wp.dot(n, particle_q[pid] - bx) - particle_radius[pid])
    if pen <= 0.0:
        return
    f = n * (soft_contact_ke * pen)
    com = wp.transform_get_translation(object_body_q[body])
    wp.atomic_add(proxy_force, body, f)
    wp.atomic_add(proxy_torque, body, wp.cross(bx - com, f))


class TwoWayProxyCoupling:
    """Dynamic-proxy two-way bridge between a MuJoCo robot and a VBD object model. Works for any
    gripped object; set ``soft_contact_ke`` (and use particle-colliding proxies) to also harvest
    the proxy↔FEM-particle reaction."""

    def __init__(self, robot_model, object_model, object_solver, object_contacts, object_state,
                 robot_finger_bodies, proxy_bodies, ee_body, sim_dt,
                 gravity=(0.0, 0.0, -9.81), soft_contact_ke=None):
        device = object_state.body_q.device
        self.object_model = object_model
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
        # When set, the proxies carry has_particle_collision=True and we also harvest the
        # body↔particle reaction (recomputed) so the soft block's squeeze is fed back too.
        self.soft_contact_ke = None if soft_contact_ke is None else float(soft_contact_ke)

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
        self._force_lag.zero_()
        self._torque_lag.zero_()
        body0, body1, p0, p1, f1, cc = self.object_solver.collect_rigid_contact_forces(
            object_state_0.body_q, self._obj_body_q_prev, self.object_contacts, self.sim_dt)
        wp.launch(_harvest_proxy_wrench_kernel, dim=body0.shape[0], inputs=[
            cc, body0, body1, p0, p1, f1, object_state_0.body_q, self.left_proxy, self.right_proxy,
        ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)
        if self.soft_contact_ke is not None:
            c = self.object_contacts
            wp.launch(_harvest_soft_wrench_kernel, dim=c.soft_contact_particle.shape[0], inputs=[
                c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape,
                c.soft_contact_body_pos, c.soft_contact_normal, object_state_0.particle_q,
                self.object_model.particle_radius, self.object_model.shape_body,
                object_state_0.body_q, self.soft_contact_ke, self.left_proxy, self.right_proxy,
            ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)

    def raw_force_norms(self):
        """Per-proxy |force| [N] of the harvested object reaction — the physical grip force on each
        pad (raw, un-clamped). Finite and bounded (not the old kinematic 1e4-1e6 N)."""
        f = self._force_lag.numpy()
        return [float(np.linalg.norm(f[p])) for p in (self.left_proxy, self.right_proxy)]
