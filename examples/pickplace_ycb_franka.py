"""Franka picks a rubik's cube and a banana and drops them into a box tray.

Friction test modeled on RoboLab's RubiksCubeAndBananaTask
(`_external/RoboLab/examples/run_recorded.py`): the same objects (rubik's cube,
YCB banana, bowl) at the demo's recorded start positions. The recorded
Panda+Robotiq trajectory does not transfer to our FR3+Franka-Hand (different
mount/gripper/kinematics), so we reuse the demo's grasp/release LOCATIONS (the
recorded object and bowl positions) and re-plan the arm with IK + our
force-limited gripper. The point is to watch whether the gripper holds each
object by FRICTION (a Coulomb body-body contact between the object and the
kinematic gripper pads in the VBD object model) through the carry.

Objects are real meshes loaded from the vendored USD (`Mesh.create_from_usd`),
decimated for stable collision: the banana and bowl collide as their meshes, the
cube as a box (it is one) rendered as a 3x3 colored rubik's cube. Public Newton
API only; nothing in `_external` is modified or imported privately.

Run: python -m examples pickplace_ycb_franka --viewer usd --device cuda:0
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")


class _TerminalTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def _install_terminal_log():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "terminal").open("w", buffering=1)
    sys.stdout = _TerminalTee(sys.__stdout__, log)
    sys.stderr = _TerminalTee(sys.__stderr__, log)
    return log


_terminal_log = _install_terminal_log()

import warp as wp

import examples
try:
    import newton
    import newton.ik as ik
    import newton.utils
    from newton import JointTargetMode
    from newton import Mesh
except ModuleNotFoundError as exc:
    raise SystemExit("This example requires the Newton Python package.") from exc

from pxr import Usd, UsdGeom

from examples.grip_force import GraspTarget, GripForceClamp, RigidGripWidth


# Recorded start poses from RoboLab's RubiksCubeAndBananaTask (world frame, robot
# at origin). z adjusted so each object rests on our table top.
TABLE_TOP_Z = 0.05
ASSET = Path(__file__).resolve().parent.parent / "assets" / "objects"
BANANA_USD = ASSET / "ycb" / "banana.usd"
BOWL_USD = ASSET / "ycb" / "bowl.usd"
CUBE_XY = (0.431, -0.097)
BANANA_XY = (0.539, -0.076)
TRAY_XY = (0.443, 0.127)


def _quat_rotate_xyzw(q, v):
    qx = q[:3]
    t = 2.0 * np.cross(qx, v)
    return v + q[3] * t + np.cross(qx, t)


def _find_body(labels, suffix):
    for i, l in enumerate(labels):
        if l.endswith(suffix):
            return i
    raise ValueError(f"no body ending {suffix!r}")


def _quat_to_vec4(q):
    return wp.vec4(q[0], q[1], q[2], q[3])


def _load_usd_mesh(path: Path) -> Mesh:
    stage = Usd.Stage.Open(str(path))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    return Mesh.create_from_usd(prim)


def _decimate_mesh(mesh: Mesh, target_faces: int) -> Mesh:
    # A low-poly copy for COLLISION: the full-res banana BVH (16k tris) segfaults
    # the mesh midphase in a multi-shape scene; ~400 tris is still banana-shaped
    # but a small, stable BVH. The full mesh is kept for rendering.
    import trimesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.indices).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    try:
        d = tm.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:
        d = tm.simplify_quadric_decimation(target_faces)
    return Mesh(d.vertices.astype(np.float32), d.faces.flatten().astype(np.int32))


@wp.func
def _wp_smoothstep(x: float) -> float:
    x = wp.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.func
def _lerp(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, i: int) -> float:
    return (1.0 - s) * a[i] + s * b[i]


# Two pick-and-place cycles. Each cycle: approach above object, descend, [close],
# lift, move over tray, [open]. Waypoint arrays are 9-vectors (arm + 2 fingers);
# the gripper DOFs follow the force-triggered latch (grip_target).
@wp.kernel
def _set_robot_targets_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    home: wp.array(dtype=float),
    a_pre: wp.array(dtype=float),
    a_grasp: wp.array(dtype=float),
    a_lift: wp.array(dtype=float),
    a_drop: wp.array(dtype=float),
    b_pre: wp.array(dtype=float),
    b_grasp: wp.array(dtype=float),
    b_lift: wp.array(dtype=float),
    b_drop: wp.array(dtype=float),
    grip_target: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
):
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    # cycle A (cube): 0..9, cycle B (banana): 9..18, home after 18
    if t < 1.6:
        q = _lerp(home, a_pre, _wp_smoothstep(t / 1.6), i)
    elif t < 2.6:
        q = _lerp(a_pre, a_grasp, _wp_smoothstep((t - 1.6) / 1.0), i)
    elif t < 4.0:
        q = a_grasp[i]  # close window 2.6..3.8
    elif t < 4.8:
        q = _lerp(a_grasp, a_lift, _wp_smoothstep((t - 4.0) / 0.8), i)
    elif t < 6.0:
        q = _lerp(a_lift, a_drop, _wp_smoothstep((t - 4.8) / 1.2), i)
    elif t < 7.0:
        q = a_drop[i]  # open window 6.2..6.8
    elif t < 8.6:
        q = _lerp(a_drop, b_pre, _wp_smoothstep((t - 7.0) / 1.6), i)    # move above banana
    elif t < 10.2:
        q = _lerp(b_pre, b_grasp, _wp_smoothstep((t - 8.6) / 1.6), i)   # descend slowly to the table
    elif t < 12.2:
        q = b_grasp[i]                                                  # STOP, then grasp (gripper closes)
    elif t < 16.2:
        q = _lerp(b_grasp, b_lift, _wp_smoothstep((t - 12.2) / 4.0), i)  # slow ascent — don't rush
    elif t < 17.8:
        q = _lerp(b_lift, b_drop, _wp_smoothstep((t - 16.2) / 1.6), i)   # carry to the bowl
    elif t < 18.6:
        q = b_drop[i]                                                   # hold (release)
    else:
        q = _lerp(b_drop, home, _wp_smoothstep((t - 18.6) / 2.0), i)

    if i >= 7:
        q = grip_target[0]
    joint_target_q[i] = q


@wp.kernel
def _sync_gripper_proxies_kernel(
    robot_body_q: wp.array(dtype=wp.transform),
    finger_bodies: wp.array(dtype=wp.int32),
    proxy_bodies: wp.array(dtype=wp.int32),
    object_body_q_0: wp.array(dtype=wp.transform),
    object_body_q_1: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    tf = robot_body_q[finger_bodies[i]]
    object_body_q_0[proxy_bodies[i]] = tf
    object_body_q_1[proxy_bodies[i]] = tf


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


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.table_top_z = TABLE_TOP_Z
        self.gripper_proxy_margin = 0.001
        self.gripper_proxy_gap = 0.008
        self.gripper_open = 0.04
        self.grip_force_threshold = float(getattr(args, "grip_threshold", 15.0))
        self.grip_close_rate = 0.02
        self.grip_filter = 0.5
        self.grip_touch_eps = 0.5  # reaction [N] that counts as first contact (live detection)
        self.grip_release_eps = 2.0  # back off the pads until the penalty squeeze drops below this [N]
        self.grip_ramp_time = float(getattr(args, "grip_ramp_time", 0.5))

        self.cube_half = 0.029
        self.cube_pos = np.array([CUBE_XY[0], CUBE_XY[1], self.table_top_z + self.cube_half], dtype=np.float32)
        self.banana_z = self.table_top_z + 0.02
        self.banana_pos = np.array([BANANA_XY[0], BANANA_XY[1], self.banana_z], dtype=np.float32)
        self.tray_pos = np.array([TRAY_XY[0], TRAY_XY[1], self.table_top_z], dtype=np.float32)
        self.drop_height = self.table_top_z + 0.16

        self.home_q = np.array(
            [-0.0036802115, 0.023901723, 0.003680411, -2.3683236, -0.00012918962,
             2.3922248, 0.785492, self.gripper_open, self.gripper_open], dtype=np.float32)

        robot_builder = self._build_robot_builder()
        ik_model = robot_builder.finalize(device=self._device_from_args(args))
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)
        self.ee_body = _find_body(list(ik_model.body_label), "fr3_link7")
        self.ee_offset = np.array([0.0, 0.0, 0.22], dtype=np.float32)

        def grasp_set(xy, grasp_z, yaw=0.0, drop_xy=None):
            dxy = drop_xy if drop_xy is not None else (self.tray_pos[0], self.tray_pos[1])
            pre = self._solve_ik(ik_model, ik_state, np.array([xy[0], xy[1], self.table_top_z + 0.14], np.float32), yaw)
            grasp = self._solve_ik(ik_model, ik_state, np.array([xy[0], xy[1], grasp_z], np.float32), yaw)
            lift = self._solve_ik(ik_model, ik_state, np.array([xy[0], xy[1], self.drop_height], np.float32), yaw)
            drop = self._solve_ik(ik_model, ik_state, np.array([dxy[0], dxy[1], self.drop_height], np.float32), yaw)
            return pre, grasp, lift, drop

        # Drop the cube off-center, over the bowl's rim, so it strikes the bowl at
        # an angle (not its base center) and knocks the bowl — an off-center impact
        # demo. The bowl is a dynamic body (see _build_object_builder) so it moves.
        cube_drop_xy = (self.tray_pos[0] + 0.06, self.tray_pos[1])
        self.a_pre, self.a_grasp, self.a_lift, self.a_drop = grasp_set(
            CUBE_XY, self.table_top_z + self.cube_half, drop_xy=cube_drop_xy)
        # Banana: grasp ~a quarter down from one end. The banana is curved, so its
        # centerline x at that slice is offset from the body center — grasp the
        # actual centerline point (mean vertex x in a band around the slice y) so
        # the pads straddle the banana, not miss it. Yaw 90 deg closes the pads
        # across the banana's narrow width rather than along its long axis.
        bv = np.asarray(_load_usd_mesh(BANANA_USD).vertices)
        gy = bv[:, 1].min() + 0.25 * (bv[:, 1].max() - bv[:, 1].min())
        band = np.abs(bv[:, 1] - gy) < 0.012
        gx = float(bv[band, 0].mean())
        self.banana_grasp_xy = (BANANA_XY[0] + gx, BANANA_XY[1] + gy)
        # Descend until the fingertips reach the table: command the TCP slightly
        # below the surface so the fingers are stopped by the robot-side table
        # collider, grasping the banana low against the table.
        banana_grasp_z = self.table_top_z - 0.01
        self.b_pre, self.b_grasp, self.b_lift, self.b_drop = grasp_set(self.banana_grasp_xy, banana_grasp_z, yaw=np.pi / 2)

        self.robot_finger_bodies = [
            _find_body(list(ik_model.body_label), "fr3_leftfinger"),
            _find_body(list(ik_model.body_label), "fr3_rightfinger"),
        ]
        self.robot_model = robot_builder.finalize(device=ik_model.device)
        self.robot_state_0 = self.robot_model.state()
        self.robot_state_1 = self.robot_model.state()
        self.robot_control = self.robot_model.control()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_0)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_1)
        wp.copy(self.robot_control.joint_target_q, self.robot_model.joint_q)
        rcm = 2048
        self.robot_model.rigid_contact_max = rcm
        self.robot_collision_pipeline = newton.CollisionPipeline(
            self.robot_model, reduce_contacts=True, rigid_contact_max=rcm, broad_phase="nxn")
        self.robot_contacts = self.robot_collision_pipeline.contacts()
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.robot_model, solver="newton", integrator="implicitfast", iterations=15, ls_iterations=100,
            nconmax=rcm, njmax=rcm * 2, cone="elliptic", impratio=50.0, use_mujoco_contacts=False)

        object_builder = self._build_object_builder(robot_builder)
        object_builder.color(balance_colors=False)

        # Build the combined viz model FIRST, before the object model is finalized.
        # The viz model re-uses the object builder's mesh shapes (the same `Mesh`
        # objects); finalizing a model rebuilds those meshes' BVHs. If the viz
        # model is finalized AFTER the object model, it rebuilds the shared BVH and
        # leaves the object model/pipeline pointing at freed BVH memory → segfault
        # in the narrow phase. The viz model is render-only (never collides), so
        # its now-stale BVH is harmless; the object model, finalized last, owns the
        # live BVH used for collision.
        viz_builder = newton.ModelBuilder()
        viz_builder.add_builder(robot_builder)
        self.viz_object_body_start = robot_builder.body_count
        viz_builder.add_builder(object_builder)
        self.viz_model = viz_builder.finalize(device=ik_model.device)
        self.viz_state = self.viz_model.state()

        self.object_model = object_builder.finalize(device=ik_model.device)
        self.object_model.shape_material_ke.fill_(5.0e4)
        self.object_model.shape_material_kd.fill_(1.0e2)
        # NB: do not fill mu here — it would overwrite the per-shape friction
        # (banana/pads are set high so the grip holds). The pair friction between
        # the banana and the kinematic gripper pads is what carries it.
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()

        ocm = 100000
        self.object_model.rigid_contact_max = ocm
        self.object_collision_pipeline = newton.CollisionPipeline(
            self.object_model, contact_matching="latest", rigid_contact_max=ocm,
            max_triangle_pairs=1_048_576, reduce_contacts=True)
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)
        self.object_solver = newton.solvers.SolverVBD(
            self.object_model, iterations=args.vbd_iterations, rigid_body_contact_buffer_size=16384,
            rigid_body_particle_contact_buffer_size=512, rigid_contact_history=True,
            rigid_contact_stick_motion_eps=0.0, rigid_avbd_contact_alpha=0.0)

        device = self.object_state_0.body_q.device
        self._t_frame = wp.zeros(1, dtype=wp.float32, device=device)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=device)
        self._home_wp = kf(self.home_q)
        self._a = [kf(x) for x in (self.a_pre, self.a_grasp, self.a_lift, self.a_drop)]
        self._b = [kf(x) for x in (self.b_pre, self.b_grasp, self.b_lift, self.b_drop)]
        self._finger_bodies_wp = wp.array(self.robot_finger_bodies, dtype=wp.int32, device=device)
        self._proxy_bodies_wp = wp.array(self.gripper_proxy_bodies, dtype=wp.int32, device=device)
        self.left_proxy, self.right_proxy = (int(b) for b in self.gripper_proxy_bodies)
        self._grip_reaction_raw = wp.zeros(2, dtype=wp.float32, device=device)
        self._grip_reaction = wp.zeros(2, dtype=wp.float32, device=device)
        self._grip_target = wp.array([self.gripper_open], dtype=wp.float32, device=device)
        self._obj_body_q_prev = wp.zeros(self.object_model.body_count, dtype=wp.transform, device=device)

        # Force-limited grasp (rigid): contact-driven width control closes the pads to
        # first contact (then eases out to relax the penalty), and GripForceClamp ramps
        # an explicit 0->15 N grip force and holds the body to the pads.
        self._grasp_windows = [(2.6, 6.2), (10.2, 17.8)]  # cube, banana
        self.grip_width = RigidGripWidth(
            self._grip_target, self._grasp_windows, self._grip_reaction,
            close_rate=self.grip_close_rate, gripper_open=self.gripper_open,
            touch_eps=self.grip_touch_eps, release_eps=self.grip_release_eps)
        self.grip_clamp = GripForceClamp(
            self.object_model, self.object_state_0, self.left_proxy, self.right_proxy,
            targets=[GraspTarget(self.cube_body, (2.6, 6.2)),
                     GraspTarget(self.banana_body, (10.2, 17.8))],
            reaction=self._grip_reaction, max_force=self.grip_force_threshold,
            ramp_time=self.grip_ramp_time, mu=0.8, touch_eps=self.grip_touch_eps)

        self.graph = None
        self._frames_simulated = 0
        self._capture_enabled = wp.get_device(str(ik_model.device)).is_cuda and self.sim_substeps % 2 == 0

        self._sync_gripper_proxies()

        self._sync_viz_state()
        self.viewer.set_model(self.viz_model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.1, 0.55, 0.6), pitch=-25.0, yaw=-130.0)

    @staticmethod
    def _device_from_args(args):
        return wp.get_device(args.device) if args.device else None

    def _build_robot_builder(self):
        b = newton.ModelBuilder()
        b.rigid_gap = 0.005
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        ap = newton.utils.download_asset("franka_emika_panda")
        b.add_urdf(Path(ap) / "urdf" / "fr3_franka_hand.urdf", xform=wp.transform((0, 0, 0), wp.quat_identity()),
                   floating=False, enable_self_collisions=False, parse_visuals_as_colliders=False,
                   collapse_fixed_joints=True, force_show_colliders=False)
        b.joint_q[:9] = self.home_q.tolist()
        b.joint_target_q[:9] = self.home_q.tolist()
        for dof in range(9):
            b.joint_target_ke[dof] = 420.0 if dof < 7 else 300.0
            b.joint_target_kd[dof] = 42.0 if dof < 7 else 30.0
            b.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            b.joint_effort_limit[dof] = 87.0 if dof < 7 else 20.0
            b.joint_armature[dof] = 0.1
        ga = b.custom_attributes["mujoco:jnt_actgravcomp"]
        if ga.values is None:
            ga.values = {}
        for dof in range(7):
            ga.values[dof] = True
        gb = b.custom_attributes["mujoco:gravcomp"]
        if gb.values is None:
            gb.values = {}
        for body in range(2, b.body_count):
            gb.values[body] = 1.0
        table_cfg = newton.ModelBuilder.ShapeConfig(margin=1e-3, density=1000.0, is_visible=False, ke=5e4, kd=5e2, mu=1.0)
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.45, 0.0, self.table_top_z - 0.025), wp.quat_identity()),
                        hx=0.35, hy=0.5, hz=0.025, cfg=table_cfg, label="robot_contact_table")
        return b

    def _build_object_builder(self, robot_builder):
        b = newton.ModelBuilder()
        b.default_shape_cfg.ke = 5.0e4
        b.default_shape_cfg.kd = 1.0e2
        b.default_shape_cfg.mu = 1.0

        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5e4, kd=1e2, mu=1.0)
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.45, 0.0, self.table_top_z - 0.025), wp.quat_identity()),
                        hx=0.35, hy=0.5, hz=0.025, cfg=table_cfg, color=wp.vec3(0.55, 0.45, 0.32), label="table")

        # Circular bowl: the real YCB bowl mesh (decimated for a stable midphase
        # BVH), static (body=-1), as both the collider the objects drop into and
        # the rendered shape. Placed so the bowl bottom rests on the table.
        tx, ty = float(self.tray_pos[0]), float(self.tray_pos[1])
        bowl_mesh = _decimate_mesh(_load_usd_mesh(BOWL_USD), 1800)
        bw = np.asarray(bowl_mesh.vertices)
        bowl_z = self.table_top_z - float(bw[:, 2].min())  # lift so its lowest point sits on the table
        self.bowl_top_z = self.table_top_z + float(bw[:, 2].max() - bw[:, 2].min())
        # Dynamic bowl (a rigid body, not a static collider) so an off-center cube
        # impact can push/tip it. Light-ish density so the hit visibly moves it.
        self.bowl_body = b.add_body(xform=wp.transform(wp.vec3(tx, ty, bowl_z), wp.quat_identity()), label="bowl")
        bowl_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, ke=5e4, kd=1e2, mu=1.0)
        b.add_shape(body=self.bowl_body, type=int(newton.GeoType.MESH), xform=wp.transform_identity(),
                    cfg=bowl_cfg, scale=(1.0, 1.0, 1.0), src=bowl_mesh, color=wp.vec3(0.75, 0.22, 0.18), label="bowl")

        # Gripper proxies: collision is kept ON so the proxy<->object contact provides
        # LIVE contact detection (the width controller closes until it reads a reaction)
        # and a non-penetration backstop. For rigid objects the grasp FORCE itself is
        # applied explicitly by GripForceClamp (see grip_force.py) the moment contact is
        # detected — the pads stop at first contact and the penalty stays quiet because
        # the clamp holds the object to the pads (no further penetration).
        proxy_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=False, has_shape_collision=True,
                                                    has_particle_collision=False, margin=self.gripper_proxy_margin,
                                                    gap=self.gripper_proxy_gap, ke=5e4, kd=1e2, mu=2.0)
        self.gripper_proxy_bodies = []
        self.gripper_proxy_shapes = []
        for label, fb in zip(("left_gripper_contact_proxy", "right_gripper_contact_proxy"), self.robot_finger_bodies, strict=True):
            body = b.add_body(xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), is_kinematic=True, label=label)
            self.gripper_proxy_bodies.append(body)
            self._copy_finger_shapes(robot_builder, b, fb, body, proxy_cfg, label)

        # Rubik's cube: box collision (it is a cube). Rendered as a black body with
        # six colored face stickers (one per side) to look like a real cube.
        self.cube_body = b.add_body(xform=wp.transform(wp.vec3(*self.cube_pos), wp.quat_identity()), label="rubiks_cube")
        cube_cfg = newton.ModelBuilder.ShapeConfig(density=1025.0, ke=5e4, kd=1e2, mu=1.2, is_visible=False)
        b.add_shape_box(body=self.cube_body, hx=self.cube_half, hy=self.cube_half, hz=self.cube_half,
                        cfg=cube_cfg, label="cube_collision")
        h = self.cube_half
        vis_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        b.add_shape_box(body=self.cube_body, hx=h, hy=h, hz=h, cfg=vis_cfg,
                        color=wp.vec3(0.03, 0.03, 0.03), label="cube_body")  # black body
        # Each face is a 3x3 grid of colored cubie stickers with black gaps so the
        # individual pieces (the ridges) are visible. Opposite faces use a standard
        # rubik's pairing (white/yellow, red/orange, blue/green).
        st, off = 0.024, h + 0.0008      # face half-extent and proud offset
        spacing = st * 2.0 / 3.0          # cubie center spacing
        chs = st / 3.0 - 0.0013           # cubie half-size (the gap = the black ridge)
        thin = 0.0006
        faces = [
            ((1, 0, 0), (0.80, 0.10, 0.10)),   # +x red
            ((-1, 0, 0), (0.90, 0.45, 0.05)),  # -x orange
            ((0, 1, 0), (0.10, 0.20, 0.80)),   # +y blue
            ((0, -1, 0), (0.05, 0.55, 0.20)),  # -y green
            ((0, 0, 1), (0.95, 0.95, 0.95)),   # +z white
            ((0, 0, -1), (0.92, 0.85, 0.05)),  # -z yellow
        ]
        for (nx, ny, nz), col in faces:
            for u in (-1, 0, 1):
                for v in (-1, 0, 1):
                    if nx:      # normal x -> tangents y, z
                        pos, hxyz = wp.vec3(off * nx, u * spacing, v * spacing), (thin, chs, chs)
                    elif ny:    # normal y -> tangents x, z
                        pos, hxyz = wp.vec3(u * spacing, off * ny, v * spacing), (chs, thin, chs)
                    else:       # normal z -> tangents x, y
                        pos, hxyz = wp.vec3(u * spacing, v * spacing, off * nz), (chs, chs, thin)
                    b.add_shape_box(body=self.cube_body, xform=wp.transform(pos, wp.quat_identity()),
                                    hx=hxyz[0], hy=hxyz[1], hz=hxyz[2], cfg=vis_cfg,
                                    color=wp.vec3(*col), label="cube_cubie")

        # Banana: the real YCB mesh as the COLLISION + render shape (mesh-vs-box
        # against the table and gripper pads). Decimated to ~1200 tris: the full
        # 16k-tri BVH is heavy/unstable in the mesh midphase, and ~1200 still reads
        # clearly as a banana. (The segfault that also affected this was the shared
        # Mesh BVH between the object and viz models — fixed by finalizing the viz
        # model first; see __init__.)
        self.banana_mesh = _decimate_mesh(_load_usd_mesh(BANANA_USD), 1200)
        banana_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, ke=5e4, kd=1e2, mu=2.0)
        self.banana_body = b.add_body(xform=wp.transform(wp.vec3(*self.banana_pos), wp.quat_identity()), label="banana")
        b.add_shape(body=self.banana_body, type=int(newton.GeoType.MESH), xform=wp.transform_identity(),
                    cfg=banana_cfg, scale=(1.0, 1.0, 1.0), src=self.banana_mesh, color=wp.vec3(0.93, 0.82, 0.12), label="banana_shape")
        return b

    def _copy_finger_shapes(self, rb, ob, fb, proxy_body, cfg, prefix):
        n = 0
        for idx, sb in enumerate(rb.shape_body):
            if sb != fb or not (rb.shape_flags[idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            sh = ob.add_shape(body=proxy_body, type=rb.shape_type[idx], xform=rb.shape_transform[idx], cfg=cfg,
                              scale=rb.shape_scale[idx], src=rb.shape_source[idx], label=f"{prefix}_{n}")
            self.gripper_proxy_shapes.append(sh)
            n += 1
        if n == 0:
            raise RuntimeError("no finger shapes")

    def _solve_ik(self, model, state, target_pos, yaw=0.0):
        bq = state.body_q.numpy()[self.ee_body]
        base_rot = bq[3:7]
        # apply yaw about world z to the gripper orientation
        cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
        yq = np.array([0, 0, sz, cz], dtype=np.float32)  # xyzw
        rot = self._quat_mul(yq, base_rot)
        tq = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
        objs = [
            ik.IKObjectivePosition(link_index=self.ee_body, link_offset=wp.vec3(*self.ee_offset),
                                   target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3)),
            ik.IKObjectiveRotation(link_index=self.ee_body, link_offset_rotation=wp.quat_identity(),
                                   target_rotations=wp.array([_quat_to_vec4(rot)], dtype=wp.vec4)),
            ik.IKObjectiveJointLimit(joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0),
        ]
        solver = ik.IKSolver(model=model, n_problems=1, objectives=objs, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        solver.step(tq, tq, iterations=64)
        q = np.array(tq.numpy()[0, :9], dtype=np.float32)
        q[7:9] = self.gripper_open
        return q

    @staticmethod
    def _quat_mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw,
                         aw * bw - ax * bx - ay * by - az * bz], dtype=np.float32)

    def _set_robot_targets(self, substep):
        wp.launch(_set_robot_targets_kernel, dim=9, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_wp, self._a[0], self._a[1], self._a[2], self._a[3],
            self._b[0], self._b[1], self._b[2], self._b[3], self._grip_target,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def _update_grip_target(self, substep):
        self.grip_width.update(self._t_frame, substep, self.sim_dt)

    def _sync_gripper_proxies(self):
        wp.launch(_sync_gripper_proxies_kernel, dim=2,
                  inputs=[self.robot_state_0.body_q, self._finger_bodies_wp, self._proxy_bodies_wp],
                  outputs=[self.object_state_0.body_q, self.object_state_1.body_q], device=self.object_state_0.body_q.device)

    def _snapshot_object_body_q(self):
        wp.copy(self._obj_body_q_prev, self.object_state_0.body_q)

    def _update_grip_reaction(self):
        body0, body1, _p0, _p1, f1, cc = self.object_solver.collect_rigid_contact_forces(
            self.object_state_0.body_q, self._obj_body_q_prev, self.object_contacts, self.sim_dt)
        self._grip_reaction_raw.zero_()
        wp.launch(_grip_reaction_kernel, dim=body0.shape[0], inputs=[
            cc, body0, body1, f1, self.object_state_0.body_q, self.left_proxy, self.right_proxy,
        ], outputs=[self._grip_reaction_raw], device=self._grip_reaction_raw.device)
        wp.launch(_blend_grip_kernel, dim=2, inputs=[self._grip_reaction_raw, self.grip_filter],
                  outputs=[self._grip_reaction], device=self._grip_reaction.device)

    def grip_reaction_norms(self):
        r = self._grip_reaction.numpy()
        return float(r[0]), float(r[1])

    def obj_pos(self, body):
        return self.object_state_0.body_q.numpy()[body][:3]

    def _sync_viz_state(self):
        bq = self.viz_state.body_q.numpy()
        bqd = self.viz_state.body_qd.numpy()
        rc = self.robot_model.body_count
        bq[:rc] = self.robot_state_0.body_q.numpy()
        bqd[:rc] = self.robot_state_0.body_qd.numpy()
        s = self.viz_object_body_start
        e = s + self.object_model.body_count
        bq[s:e] = self.object_state_0.body_q.numpy()
        bqd[s:e] = self.object_state_0.body_qd.numpy()
        self.viz_state.body_q.assign(bq)
        self.viz_state.body_qd.assign(bqd)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self._update_grip_target(substep)
            self._set_robot_targets(substep)
            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(self.robot_state_0, self.robot_state_1, self.robot_control, self.robot_contacts, self.sim_dt)
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0
            self._sync_gripper_proxies()
            self._snapshot_object_body_q()
            self.object_state_0.clear_forces()
            self.object_model.collide(self.object_state_0, self.object_contacts)
            self.grip_clamp.apply(self.object_state_0, self._t_frame, substep, self.sim_dt)
            self.object_solver.step(self.object_state_0, self.object_state_1, self.object_control, self.object_contacts, self.sim_dt)
            self.object_state_0, self.object_state_1 = self.object_state_1, self.object_state_0
            self.grip_clamp.update_prev(self.object_state_0)
            self._update_grip_reaction()

    def step(self):
        self._t_frame.assign(np.array([self.sim_time], dtype=np.float32))
        if self.graph is None and self._capture_enabled and self._frames_simulated >= 1:
            saved = (self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1)
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as exc:
                self.robot_state_0, self.robot_state_1, self.object_state_0, self.object_state_1 = saved
                print(f"CUDA graph capture failed ({exc}); continuing without capture.")
                self._capture_enabled = False
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._frames_simulated += 1
        self.sim_time += self.frame_dt

    def render(self):
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

    def test_final(self):
        bq = self.object_state_0.body_q.numpy()
        if not np.all(np.isfinite(bq)):
            raise ValueError("Non-finite body transform.")
        for name, body in [("cube", self.cube_body), ("banana", self.banana_body)]:
            if bq[body][2] < self.table_top_z - 0.05:
                raise ValueError(f"The {name} fell through the table/world.")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "pickplace_ycb_franka.usd"), num_frames=1320)
        parser.add_argument("--substeps", type=int, default=16)
        parser.add_argument("--vbd-iterations", type=int, default=12)
        parser.add_argument("--grip-threshold", type=float, default=15.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="pickplace_ycb_franka")
    examples.run(Example(viewer, args), args)
