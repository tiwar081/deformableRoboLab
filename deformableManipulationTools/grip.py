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
from .mathutils import wp_smoothstep, quat_rotate_xyzw


def _finger_collider_aabb(robot_builder, shape_idx):
    """Axis-aligned bounds ``(lo, hi)`` in the finger BODY frame over the finger's collider shapes
    (boxes and/or convex meshes). Used to synthesize a single box finger proxy for robots whose
    finger collider is a mesh (see :func:`build_gripper_proxies`)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in shape_idx:
        xf = robot_builder.shape_transform[s]
        off = np.array([float(x) for x in (xf.p if hasattr(xf, "p") else xf[:3])])
        q = np.array([float(x) for x in (xf.q if hasattr(xf, "q") else xf[3:7])])
        scale = np.array([float(x) for x in robot_builder.shape_scale[s]])
        if int(robot_builder.shape_type[s]) == int(newton.GeoType.BOX):
            pts = np.array([[sx * scale[0], sy * scale[1], sz * scale[2]]
                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        else:
            pts = np.asarray(robot_builder.shape_source[s].vertices, dtype=np.float64) * scale[None, :]
        pts = quat_rotate_xyzw(q, pts) + off
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    return lo, hi


# ---------------------------------------------------------------------------------------------
# Dynamic finger proxies (the grip contact bridge consumed by TwoWayProxyCoupling below).
# ---------------------------------------------------------------------------------------------
def build_gripper_proxies(object_builder, robot_builder, finger_bodies: list[int],
                          object_table_shape: int | None, grip: GripConfig = GRIP):
    """Add the CENTRALIZED dynamic-proxy gripper collision to ``object_builder`` — IDENTICAL for every
    VBD demo, with NO per-demo knobs. The geometry is fixed in :data:`GRIP`:

      * two finger proxies, each carrying the Franka finger's TRUE collision shapes (the sparse URDF
        boxes — pad + edges + knuckle — copied one-for-one). These two are the GRIP pads
        (``proxy_bodies[:2]`` — harvested, in the grip signal) and grip FEM particles too. (The gaps
        between the sparse boxes let a swept cable clip ~1 mm into the fingers; the palm blocker still
        stops the larger palm/wrist penetration.)
      * a THIRD palm/EE "blocker" proxy: one synthetic box (``grip.palm_box_*``, EE/link7 frame) that
        stops a swept cable passing through the gripper palm / wrist (the hand collider is collapsed
        into link7 and lives only in MuJoCo). The caller pins it to the EE via the coupling's mirror
        list. Its contact reaction IS harvested (momentum-consistent: it flows to the EE and is undone
        in sync_proxies, so it never shoves an object one-way), but it is NEVER in the grip signal.

    Returns ``(proxy_bodies, proxy_shapes)`` with the two finger proxies FIRST and the palm proxy LAST.
    """
    proxy_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, is_visible=False, has_shape_collision=True, has_particle_collision=True,
        margin=grip.proxy_margin, gap=grip.proxy_gap, ke=grip.proxy_ke, kd=grip.proxy_kd, mu=grip.proxy_mu,
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
        # The finger proxy is built from the robot's OWN finger colliders, by one of two paths chosen
        # from the collider TYPE (so any Franka-family robot gets a correct, well-behaved pad with no
        # per-demo knob):
        #   * BOX colliders (FR3 URDF: 4 sparse boxes — pad + edges + knuckle) → copy each one-for-one.
        #     This is the TRUE collider (faithful); reverted here once from a gap-filled AABB box.
        #   * a CONVEX_MESH collider (Isaac Sim Panda USD: one mesh per finger) → synthesize a single
        #     box = the finger's collider AABB. A convex-mesh proxy is contacted LATE then explosively by
        #     the VBD penalty solver (a 50 mm cube spiked the grip to ~2 kN at a ~2 mm latch); the AABB
        #     box restores box-on-box contact like every other VBD demo (physical ~65 N latch at the true
        #     pad width). The AABB is per-finger, so each pad faces the grasp center via its own mesh
        #     (the panda's two finger bodies share one orientation — the right pad is a MIRRORED mesh, not
        #     a flipped body), and its pad face matches the FR3 boxes to <0.5 mm.
        finger_shape_idx = [s for s, fb in enumerate(robot_builder.shape_body)
                            if fb == finger_body
                            and (robot_builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES))]
        if not finger_shape_idx:
            raise RuntimeError(f"No colliding shapes on Franka finger body {finger_body}.")
        if all(int(robot_builder.shape_type[s]) == int(newton.GeoType.BOX) for s in finger_shape_idx):
            for n, s in enumerate(finger_shape_idx):
                shape = object_builder.add_shape(
                    body=body, type=robot_builder.shape_type[s], xform=robot_builder.shape_transform[s],
                    cfg=proxy_cfg, scale=robot_builder.shape_scale[s], src=robot_builder.shape_source[s],
                    label=f"{label}_shape_{n}")
                proxy_shapes.append(shape)
        else:
            lo, hi = _finger_collider_aabb(robot_builder, finger_shape_idx)
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            shape = object_builder.add_shape_box(
                body=body, xform=wp.transform(wp.vec3(*center.astype(float)), wp.quat_identity()),
                hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=proxy_cfg,
                label=f"{label}_aabb_box")
            proxy_shapes.append(shape)
    finger_shapes = list(proxy_shapes)
    # Palm/EE blocker proxy: one box (EE/link7 frame) stopping the swept cable through the palm/wrist
    # (rigid-shape collision only — cheap). Pinned to the EE by the coupling; harvested two-way to the EE.
    palm_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, is_visible=False, has_shape_collision=True, has_particle_collision=False,
        margin=grip.proxy_margin, ke=grip.proxy_ke, kd=grip.proxy_kd, mu=grip.proxy_mu)
    palm_body = object_builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=False, mass=grip.proxy_mass, inertia=inertia,
        com=wp.vec3(0.0, 0.0, 0.0), lock_inertia=True, label="palm_ee_blocker_proxy")
    hx, hy, hz = grip.palm_box_half
    palm_shape = object_builder.add_shape_box(
        body=palm_body, xform=wp.transform(wp.vec3(*grip.palm_box_offset), wp.quat_identity()),
        hx=hx, hy=hy, hz=hz, cfg=palm_cfg, label="palm_ee_blocker_shape")
    proxy_bodies.append(palm_body)
    proxy_shapes.append(palm_shape)
    # The palm box sits just behind the fingers; filter it against the finger proxy shapes.
    for fs in finger_shapes:
        object_builder.add_shape_collision_filter_pair(palm_shape, fs)
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
    proxy_bodies: wp.array(dtype=wp.int32),
    n_proxy: int,
    proxy_force: wp.array(dtype=wp.vec3),
    proxy_torque: wp.array(dtype=wp.vec3),
):
    # Rigid harvest: sum, per proxy, the rigid contact force the other body exerts on it and the torque
    # about the proxy COM. force_on_body1 is the force on body1; body0 gets -it. Harvests ALL proxies
    # (the two finger pads AND the palm/EE blocker), so the palm's contact reaction is momentum-
    # consistent: it flows to the EE via _apply_coupling_to_ee_kernel and is undone in sync_proxies,
    # instead of being a free one-way shove. The grip SIGNAL stays on the two pads (_reduce_grip_signal).
    i = wp.tid()
    if i >= contact_count[0]:
        return
    b0 = body0[i]
    b1 = body1[i]
    for k in range(n_proxy):
        pb = proxy_bodies[k]
        if b1 == pb:
            f = force_on_body1[i]
            com = wp.transform_get_translation(object_body_q[b1])
            wp.atomic_add(proxy_force, b1, f)
            wp.atomic_add(proxy_torque, b1, wp.cross(point1_world[i] - com, f))
        elif b0 == pb:
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
    # ``soft_contact_normal`` points OUT of the shape toward the particle, so n·ke·pen is the force on
    # the PARTICLE; the REACTION on the proxy/pad (what feeds the arm) is the opposite. Largely cancels
    # in the two-pad net + the arm is position-controlled, so this corrects the EE force-feedback sign
    # without changing the visible grasp.
    f = -n * (soft_contact_ke * pen)
    com = wp.transform_get_translation(object_body_q[body])
    wp.atomic_add(proxy_force, body, f)
    wp.atomic_add(proxy_torque, body, wp.cross(bx - com, f))


@wp.kernel
def _reduce_grip_signal_kernel(
    force_lag: wp.array(dtype=wp.vec3),
    left_proxy: int,
    right_proxy: int,
    grip_force_signal: wp.array(dtype=wp.float32),
):
    # dim=1. The both-pads-engaged grip force = min of the two per-pad reaction magnitudes (min, not
    # sum, so the latch needs BOTH pads pressing). Read by GripController one substep later.
    fl = wp.length(force_lag[left_proxy])
    fr = wp.length(force_lag[right_proxy])
    grip_force_signal[0] = wp.min(fl, fr)


class TwoWayProxyCoupling:
    """Dynamic-proxy two-way bridge between a MuJoCo robot and a VBD object model. Works for any
    gripped object; set ``soft_contact_ke`` (and use particle-colliding proxies) to also harvest
    the proxy↔FEM-particle reaction."""

    def __init__(self, robot_model, object_model, object_solver, object_contacts, object_state,
                 mirror_bodies, proxy_bodies, ee_body, sim_dt,
                 gravity=(0.0, 0.0, -9.81), soft_contact_ke=None):
        # ``mirror_bodies[i]`` is the ROBOT body that ``proxy_bodies[i]`` is re-pinned to each substep.
        # The first TWO proxies are the grip PADS (fingers); any extra proxies (e.g. the palm/EE blocker,
        # mirrored to the EE) are synced + summed-to-EE generically. ALL proxies are harvested (so every
        # proxy contact is momentum-consistent — no one-way shove), but only the two pads feed the grip
        # SIGNAL (_reduce_grip_signal_kernel), so the palm blocker never perturbs the force-target latch.
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
        self._finger_bodies = wp.array(mirror_bodies, dtype=wp.int32, device=device)
        self._proxy_bodies = wp.array(proxy_bodies, dtype=wp.int32, device=device)
        self.left_proxy, self.right_proxy = int(proxy_bodies[0]), int(proxy_bodies[1])
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
        # Device-side grip-force signal the force-stop GripController reads next substep (one-step
        # stale, like the wrench feedback): the both-pads-engaged squeeze magnitude
        # min(|f_left|, |f_right|) [N]. min (not sum) requires BOTH pads pressing, so a single-pad
        # graze can't trip the latch. Recomputed at the end of harvest(), so it is graph-capturable.
        self.grip_force_signal = wp.zeros(1, dtype=wp.float32, device=device)

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
            cc, body0, body1, p0, p1, f1, object_state_0.body_q, self._proxy_bodies, self._n,
        ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)
        if self.soft_contact_ke is not None:
            c = self.object_contacts
            wp.launch(_harvest_soft_wrench_kernel, dim=c.soft_contact_particle.shape[0], inputs=[
                c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape,
                c.soft_contact_body_pos, c.soft_contact_normal, object_state_0.particle_q,
                self.object_model.particle_radius, self.object_model.shape_body,
                object_state_0.body_q, self.soft_contact_ke, self.left_proxy, self.right_proxy,
            ], outputs=[self._force_lag, self._torque_lag], device=self._force_lag.device)
        wp.launch(_reduce_grip_signal_kernel, dim=1, inputs=[
            self._force_lag, self.left_proxy, self.right_proxy,
        ], outputs=[self.grip_force_signal], device=self._force_lag.device)

    def raw_force_norms(self):
        """Per-proxy |force| [N] of the harvested object reaction — the physical grip force on each
        pad (raw, un-clamped). Finite and bounded (not the old kinematic 1e4-1e6 N)."""
        f = self._force_lag.numpy()
        return [float(np.linalg.norm(f[p])) for p in (self.left_proxy, self.right_proxy)]


# ---------------------------------------------------------------------------------------------
# Force-feedback finger controller (replaces the per-demo geometric preset close width).
# ---------------------------------------------------------------------------------------------
@wp.kernel
def _grip_force_stop_kernel(
    t_frame: wp.array(dtype=wp.float32),
    substep: int,
    sim_dt: float,
    windows: wp.array(dtype=wp.float32),        # flat [close_start, close_end, release_start, release_end, ft, servo_mode]*n
    n_windows: int,
    gripper_open: float,
    close_target: float,                        # ramp/fully-closed target (min_close_width or mujoco close_target)
    force_stop_enabled: int,                    # 1 = VBD force-stop; 0 = rigid-only smoothstep, no latch
    latch_arm_margin: float,
    latch_debounce: int,
    grip_servo_rate: float,                     # close-only hold-servo slew [m/s] (tighten until steady ft)
    grip_force_deadband: float,                 # [N] deadband around the target force
    max_overclose: float,                       # cap [m] on servo tightening past the first-contact width
    grip_signal: wp.array(dtype=wp.float32),
    finger_q: wp.array(dtype=wp.float32),       # measured robot joint positions (read the finger DOFs)
    finger_dof0: int,
    finger_dof1: int,
    latch_state: wp.array(dtype=wp.float32),    # flat [latched, latched_w, current_w, debounce, seed_w]*n
    joint_target_q: wp.array(dtype=wp.float32),
):
    # dim=1, single thread owns the shared per-window latch state (no intra-launch race). The latch
    # state persists across substeps AND frames inside the replayed CUDA graph; runtime branches on
    # the device t_frame / grip_signal work on replay (capture bakes launches, not branch outcomes).
    t = t_frame[0] + float(substep) * sim_dt
    sig = grip_signal[0]
    out = gripper_open                          # default (before/between/after all windows): open
    for w in range(n_windows):
        wb = 6 * w
        cs = windows[wb + 0]
        ce = windows[wb + 1]
        rs = windows[wb + 2]
        re = windows[wb + 3]
        ft = windows[wb + 4]                     # pre-resolved force TARGET [N] (rigid ~30 N, soft ~8 N)
        servo_mode = windows[wb + 5]             # 1 = incompressible (first-contact latch + servo); 0 = compressible (latch at ft, freeze)
        sb = 5 * w
        if t < cs:
            # before this grasp: open, (re)arm the latch state for a clean cycle
            latch_state[sb + 0] = 0.0           # latched flag
            latch_state[sb + 1] = close_target  # latched width
            latch_state[sb + 2] = gripper_open  # current commanded width
            latch_state[sb + 3] = 0.0           # debounce counter
            latch_state[sb + 4] = gripper_open  # first-contact seed width (for the over-close cap)
        elif t < re:
            latched = latch_state[sb + 0]
            latched_w = latch_state[sb + 1]
            current_w = latch_state[sb + 2]
            debounce = latch_state[sb + 3]
            seed_w = latch_state[sb + 4]
            width = current_w
            if t >= rs:                         # RELEASE: smoothstep reopen FROM the held width to open
                # KEEP `latched` set for the whole ramp so `base` stays latched_w every substep (clearing
                # it made later substeps fall back to close_target → a one-step inward crush). The latch is
                # re-armed for the next grasp in the `t < cs` branch.
                base = close_target
                if latched > 0.5:
                    base = latched_w
                alpha = wp_smoothstep((t - rs) / (re - rs))
                width = (1.0 - alpha) * base + alpha * gripper_open
            elif latched > 0.5:                 # FORCE-HOLD
                # Incompressible (servo_mode=1): close-only servo to the STEADY target force. Tighten
                # (never open) while the steady min-of-pads squeeze is below target — over-closes a
                # SETTLING contact (the cable, whose resting force decays at any fixed width) just enough
                # to maintain it; barely moves a stiff object already at target. NEVER opening means the
                # lift/sweep LOAD (sig ≫ ft) can't loosen the grip. The force-derived replacement for the
                # old per-object bite. Compressible (servo_mode=0): pure FREEZE — the soft block was
                # latched AT the gentle target during close; any extra tightening would crush/eject it.
                if force_stop_enabled == 1 and servo_mode > 0.5:
                    if sig < ft - grip_force_deadband:
                        # tighten, but NEVER past seed_w - max_overclose (so a settling cable, whose
                        # steady force never reaches ft, can't run the servo down to the crush floor)
                        floor = wp.max(seed_w - max_overclose, close_target)
                        latched_w = wp.max(latched_w - grip_servo_rate * sim_dt, floor)
                width = latched_w
            elif t < ce:                        # CLOSING (not yet latched): velocity-limited smoothstep ramp
                alpha = wp_smoothstep((t - cs) / (ce - cs))
                current_w = (1.0 - alpha) * gripper_open + alpha * close_target
                width = current_w
                if force_stop_enabled == 1:
                    hit = float(0.0)            # arming guard + threshold (no boolean `and` in Warp)
                    if current_w < (gripper_open - latch_arm_margin):
                        if servo_mode > 0.5:
                            # Incompressible: latch at FIRST CONTACT (sig>0) — its only job is to HALT the
                            # fast smoothstep so it can't race to the floor and crush. The force target is
                            # then reached by the servo, NOT here: the rigid target is only a brief
                            # transient during the fast close, so latching on it is unreliable.
                            if sig > 0.0:
                                hit = 1.0
                        else:
                            # Compressible: latch AT the gentle force target — the soft block builds force
                            # smoothly (reliable threshold) and must be compressed to the target before the
                            # freeze, NOT just first-contact (which would be too loose to lift it).
                            if sig >= ft:
                                hit = 1.0
                    if hit > 0.5:
                        debounce = debounce + 1.0
                    else:
                        debounce = 0.0
                    if debounce >= float(latch_debounce):
                        latched = 1.0
                        # Seed the hold at the MEASURED finger position (NOT the open-loop command, which
                        # races ahead of the effort-limited fingers → inward spikes); the servo refines it.
                        # seed_w records this first-contact width for the over-close cap.
                        measured = 0.5 * (finger_q[finger_dof0] + finger_q[finger_dof1])
                        latched_w = wp.max(measured, close_target)
                        seed_w = latched_w
                        width = latched_w
            else:                               # HOLD but the target was never reached: best-effort closed
                width = close_target
            latch_state[sb + 0] = latched
            latch_state[sb + 1] = latched_w
            latch_state[sb + 2] = current_w
            latch_state[sb + 3] = debounce
            latch_state[sb + 4] = seed_w
            out = width
    joint_target_q[finger_dof0] = out
    joint_target_q[finger_dof1] = out


class GripController:
    """Centralized force-feedback finger controller — the legal place to "close until force" given
    that the position-controlled fingers feel nothing in their own (MuJoCo) solver. It DERIVES a
    finger POSITION command from the harvested grip-force READING (``coupling.grip_force_signal``),
    never injecting force on the finger DOF (the no-per-finger-feedback invariant is untouched).

    Per grasp window it closes the fingers (velocity-limited smoothstep) toward ``close_target`` and
    FREEZES the position the moment the min-of-both-pads squeeze first reaches the window's force TARGET
    (debounced). The target is derived CENTRALLY from the window's ``compressible`` flag (incompressible
    → ``GRIP.force_target_rigid`` ~30 N; compressible → ``GRIP.force_target`` ~8 N) — "specify force, get
    emergent geometry". The freeze (not a continuous servo) is intentional: under the lift/sweep load the
    harvested force is the object's LOAD reaction, not over-squeeze, so a servo would open and drop it;
    the frozen width lets the squeeze rise under load while holding ~target at rest. On the rigid-only
    MuJoCo path (no coupling) ``force_stop_enabled=0``: it degrades to a plain smoothstep close to
    ``close_target`` (= MUJOCO_GRIP.close_target) and lets true two-way contact stop the pads."""

    def __init__(self, robot_control, t_frame, sim_dt, finger_dofs, gripper_open, grasp_windows,
                 grip_force_signal=None, close_target=None, grip: GripConfig = GRIP):
        device = robot_control.joint_target_q.device
        self.joint_target_q = robot_control.joint_target_q
        self._t_frame = t_frame
        self.sim_dt = float(sim_dt)
        self.finger_dof0, self.finger_dof1 = int(finger_dofs[0]), int(finger_dofs[1])
        self.gripper_open = float(gripper_open)
        self.force_stop_enabled = 1 if grip_force_signal is not None else 0
        self.grip_signal = (grip_force_signal if grip_force_signal is not None
                            else wp.zeros(1, dtype=wp.float32, device=device))
        self.close_target = float(close_target if close_target is not None else grip.min_close_width)
        self.latch_arm_margin = float(grip.latch_arm_margin)
        self.latch_debounce = int(grip.latch_debounce)
        self.grip_servo_rate = float(grip.grip_servo_rate)
        self.grip_force_deadband = float(grip.grip_force_deadband)
        self.max_overclose = float(grip.max_overclose)
        self.n_windows = len(grasp_windows)
        flat: list[float] = []
        for win in grasp_windows:
            # CENTRALIZED force target + mode selected by win.compressible (no per-object knobs):
            #   compressible  → GRIP.force_target (gentle); latch AT that force, then FREEZE — a soft
            #                   block builds force smoothly, so the threshold is reliable, and any extra
            #                   tightening would crush/eject it. servo_mode = 0.
            #   incompressible→ GRIP.force_target_rigid (firm); latch at FIRST contact, then close-only
            #                   servo up to that force (a stiff object reaches it fast; a settling cable
            #                   over-closes to the centralized cap). servo_mode = 1.
            ft = float(grip.force_target) if win.compressible else float(grip.force_target_rigid)
            servo_mode = 0.0 if win.compressible else 1.0
            flat += [float(win.close_start), float(win.close_end),
                     float(win.release_start), float(win.release_end), ft, servo_mode]
        self._windows = wp.array(flat, dtype=wp.float32, device=device)
        self._latch_state = wp.zeros(self.n_windows * 5, dtype=wp.float32, device=device)

    def step(self, substep: int, finger_q) -> None:
        wp.launch(_grip_force_stop_kernel, dim=1, inputs=[
            self._t_frame, int(substep), self.sim_dt, self._windows, self.n_windows,
            self.gripper_open, self.close_target,
            self.force_stop_enabled, self.latch_arm_margin, self.latch_debounce,
            self.grip_servo_rate, self.grip_force_deadband, self.max_overclose,
            self.grip_signal, finger_q, self.finger_dof0, self.finger_dof1,
        ], outputs=[self._latch_state, self.joint_target_q], device=self.joint_target_q.device)

    def latch_widths(self):
        """Host readout (debug): per-window [latched_flag, latched_width, current_width, debounce, seed_width]."""
        s = self._latch_state.numpy().reshape(self.n_windows, 5)
        return s
