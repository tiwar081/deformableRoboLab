"""The FREE-FLOATING GRIPPER: the Franka hand alone, on a position-controlled 6-DOF base.

Why a hand and not the arm — a grasp candidate is a property of the JAWS and the object, not of
whether some arm can reach it. Running the arm would fold reachability (and the still-open
end-effector rotation projection question) into a number that is supposed to say "does this pinch
hold". ACRONYM makes the same split: "the gripper itself is simulated as an unconstrained
position-controlled object".

What this module does NOT do is invent a gripper. It lifts the REAL hand out of the real robot
:func:`robot.build_franka_robot` builds — the same finger bodies, the same collision geometry, the
same prismatic joints with ``FRANKA.finger_target_ke/kd``, ``finger_effort`` and ``armature`` — and
re-parents it to the world through a stiff 6-DOF joint. Everything downstream (the proxies copied
off these finger colliders, the harvest, the admittance controller driving these finger DOFs) is
then bit-identical to the demo path; only the thing holding the hand up changed. It works for
whichever robot ``settings.yaml`` selects, because the subtree is found by ``FRANKA``'s own link
suffixes rather than by hard-coded indices.

The base joint's frame IS THE GRASP FRAME (``grasp_library.POSE_CONVENTION``): the joint is placed
so that at zero coordinates the TCP sits exactly on the candidate's grasp centre with the jaws on
its jaw axis. That buys two things:

  * placing a candidate is one transform, with no IK and no seed-posture ambiguity;
  * the shake is expressible in ACRONYM's own words — "moves up and down along its approach
    direction" is base coordinate 2, "rotates around a line parallel to the prismatic joint axes of
    the fingers" is base coordinate 3 — instead of world-frame trigonometry that would differ per
    candidate.
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ...params import FRANKA
from ...robot import HandGeometry, build_franka_robot, finger_body_indices, hand_geometry

__all__ = ["HandGeometry", "hand_geometry", "build_floating_hand"]

# ---- the 6-DOF base actuator -----------------------------------------------------------------
# STIFF on purpose: the base is the test fixture, not the subject. ACRONYM's hand is an
# "unconstrained position-controlled object" — it imposes the shake and does not itself yield to
# the payload, so the number that comes out describes the JAWS. Sized about 25x above the shake
# frequency, which leaves the applied motion within ~1% of the commanded amplitude (the residual is
# a ~7 deg phase lag, which delays the shake without shrinking it). MEASURED per trial and reported,
# not assumed — see TrialResult.shake_ratio_linear/angular.
BASE_LINEAR_KE = 1.0e5            # [N/m]   -> ~50 Hz corner for a 1 kg hand, 25x the shake
BASE_LINEAR_KD = 1.0e3            # [N·s/m] ~1.6x critical: overdamped, so it never rings into the object
BASE_ANGULAR_KE = 3.0e3           # [N·m/rad]
BASE_ANGULAR_KD = 3.0e1           # [N·m·s/rad]
BASE_EFFORT = 1.0e4               # far above the protocol's demand; the cap must not shape the shake
BASE_ARMATURE = 0.0

# Base DOF layout of add_joint_d6(linear_axes=3, angular_axes=3): the three linear coordinates come
# first, then the three angular ones, each about the JOINT frame's axis. With the joint frame being
# the grasp frame, that is (x = jaw, y, z = approach) twice over.
BASE_DOF = 6
DOF_LIN_APPROACH = 2              # translation along the approach axis  (ACRONYM's linear shake)
DOF_ANG_JAW = 3                   # rotation about the jaw closing axis  (ACRONYM's angular shake)
FINGER_DOFS = (BASE_DOF, BASE_DOF + 1)


def _quat(r: np.ndarray) -> np.ndarray:
    """xyzw quaternion from a 3x3 rotation (Shepperd's branchless-enough form)."""
    t = float(np.trace(r))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                         (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2.0
    q = np.zeros(4)
    q[3] = (r[k, j] - r[j, k]) / s
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return q


def _xform(m: np.ndarray) -> wp.transform:
    return wp.transform(wp.vec3(*[float(v) for v in m[:3, 3]]), wp.quat(*[float(v) for v in _quat(m[:3, :3])]))


# HandGeometry / hand_geometry() — where the grasp frame sits in the EE body, measured off the real
# robot — MOVED to robot.py (2026-08-06): grasp_library's hand-clearance retreat needs the same
# measurement, and two copies of "where is the hand" is how the rig and the seat drift apart.
# Re-exported above for the rig's own callers.


# =================================================================================================
# Building the free-floating hand
# =================================================================================================
def _shape_config(builder: newton.ModelBuilder, shape: int) -> newton.ModelBuilder.ShapeConfig:
    """Reproduce a source shape's ShapeConfig from the builder's per-shape arrays.

    ``density=0`` because the copied body carries the source's EXACT mass/inertia/COM with
    ``lock_inertia`` — re-deriving inertia from geometry would quietly change the hand."""
    flags = int(builder.shape_flags[shape])
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=float(builder.shape_material_ke[shape]), kd=float(builder.shape_material_kd[shape]),
        kf=float(builder.shape_material_kf[shape]), ka=float(builder.shape_material_ka[shape]),
        mu=float(builder.shape_material_mu[shape]),
        restitution=float(builder.shape_material_restitution[shape]),
        margin=float(builder.shape_margin[shape]), gap=float(builder.shape_gap[shape]),
        is_solid=bool(builder.shape_is_solid[shape]),
        has_shape_collision=bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)),
        has_particle_collision=bool(flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)),
        is_visible=bool(flags & int(newton.ShapeFlags.VISIBLE)))


def _copy_body(dst: newton.ModelBuilder, src: newton.ModelBuilder, body: int) -> int:
    """Copy one body and every shape on it, mass properties included, into a fresh builder.

    ``add_link``, not ``add_body``: the latter is shorthand for "link + free joint + articulation",
    and the hand's three links are joined by the base and finger joints added below — a second,
    automatic free joint per link makes the articulation ill-formed ("multiple joints lead to body")."""
    new_body = dst.add_link(
        xform=src.body_q[body], mass=float(src.body_mass[body]),
        inertia=src.body_inertia[body], com=src.body_com[body],
        lock_inertia=True, label=str(src.body_label[body]))
    for shape in range(src.shape_count):
        if int(src.shape_body[shape]) != body:
            continue
        source = src.shape_source[shape]
        if source is not None and hasattr(source, "vertices"):
            # DEEP-COPY a mesh collider. Sharing a Mesh between two models makes their BVHs alias one
            # pool, and whichever model finalizes last frees it out from under the other's narrow
            # phase (the documented shared-mesh-BVH fault, CLAUDE.md / SOLVERS section 4). grip.py
            # does the same when it copies these very colliders into the object model.
            source = source.copy() if hasattr(source, "copy") else newton.Mesh(
                np.array(source.vertices, dtype=np.float32).copy(),
                np.array(source.indices, dtype=np.int32).copy())
        dst.add_shape(
            body=new_body, type=int(src.shape_type[shape]), xform=src.shape_transform[shape],
            cfg=_shape_config(src, shape), scale=src.shape_scale[shape], src=source,
            color=src.shape_color[shape], label=str(src.shape_label[shape]))
    return new_body


def _copy_finger_joint(dst: newton.ModelBuilder, src: newton.ModelBuilder, joint: int,
                       parent: int, child: int) -> int:
    """Copy a finger's prismatic joint with its real actuator: the same axis, limits, target gains,
    effort limit and armature the demo path drives. These ARE the closing dynamics — a jaw that
    reached its width some other way would make the recorded quality meaningless."""
    dof = src.joint_qd_start[joint]
    return dst.add_joint_prismatic(
        parent=parent, child=child,
        parent_xform=src.joint_X_p[joint], child_xform=src.joint_X_c[joint],
        axis=wp.vec3(*[float(v) for v in src.joint_axis[dof]]),
        target_ke=float(src.joint_target_ke[dof]), target_kd=float(src.joint_target_kd[dof]),
        limit_lower=float(src.joint_limit_lower[dof]), limit_upper=float(src.joint_limit_upper[dof]),
        limit_ke=float(src.joint_limit_ke[dof]), limit_kd=float(src.joint_limit_kd[dof]),
        armature=float(src.joint_armature[dof]), effort_limit=float(src.joint_effort_limit[dof]),
        velocity_limit=float(src.joint_velocity_limit[dof]),
        actuator_mode=newton.JointTargetMode.POSITION,
        label=str(src.joint_label[joint]))


def _base_axis(axis, ke: float, kd: float) -> newton.ModelBuilder.JointDofConfig:
    return newton.ModelBuilder.JointDofConfig(
        axis=axis, target_ke=ke, target_kd=kd, effort_limit=BASE_EFFORT, armature=BASE_ARMATURE,
        actuator_mode=newton.JointTargetMode.POSITION)


def build_floating_hand(world_from_grasp: np.ndarray,
                        gripper_open: float | None = None) -> tuple[newton.ModelBuilder, dict]:
    """The Franka hand alone, its base joint frame placed ON a world grasp pose.

    ``world_from_grasp`` is a 4x4 in ``grasp_library.POSE_CONVENTION`` (origin = TCP grasp centre,
    ``+z`` approach, ``+x`` jaw axis). At zero base coordinates the hand's TCP lands exactly there.

    Returns ``(builder, info)`` where ``info`` carries ``ee_body`` and ``finger_bodies`` in the NEW
    model's indexing, ready for ``grip.build_gripper_proxies`` and ``TwoWayProxyCoupling``."""
    geom = hand_geometry()
    src = build_franka_robot(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                             table=None)
    ee_src = geom.ee_body_src
    fingers_src = geom.finger_bodies_src

    dst = newton.ModelBuilder()
    dst.rigid_gap = src.rigid_gap
    newton.solvers.SolverMuJoCo.register_custom_attributes(dst)

    ee = _copy_body(dst, src, ee_src)
    fingers = [_copy_body(dst, src, f) for f in fingers_src]

    # The base: world -> hand through a 6-DOF position-controlled joint whose frame is the grasp
    # frame. parent_xform puts that frame on the candidate pose in the world; child_xform says where
    # the same frame sits inside the EE body. Zero coordinates therefore ARE the candidate pose.
    base_joint = dst.add_joint_d6(
        parent=-1, child=ee,
        parent_xform=_xform(np.asarray(world_from_grasp, dtype=float)),
        child_xform=_xform(geom.ee_from_grasp),
        linear_axes=[_base_axis(a, BASE_LINEAR_KE, BASE_LINEAR_KD)
                     for a in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
        angular_axes=[_base_axis(a, BASE_ANGULAR_KE, BASE_ANGULAR_KD)
                      for a in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
        label="floating_grasp_base")
    finger_joints = [_copy_finger_joint(dst, src, j, ee, child)
                     for j, child in zip(_finger_joints(src, ee_src, fingers_src), fingers,
                                         strict=True)]
    dst.add_articulation([base_joint] + finger_joints, label="floating_gripper")

    # Open the jaws to the PRE-SHAPED aperture (per-finger). Default = the Panda's full aperture
    # (ACRONYM's pre-grasp state); the rig passes the candidate's width + PREGRASP_MARGIN, the
    # state the trajectory actually approaches in — see grasp_library.PREGRASP_MARGIN.
    q_open = FRANKA.gripper_open if gripper_open is None else float(gripper_open)
    for dof in FINGER_DOFS:
        dst.joint_q[dof] = q_open
        dst.joint_target_q[dof] = q_open

    # The two jaws must never collide each other — they close on an object BETWEEN them. Harmless
    # for box pads, ESSENTIAL for the panda's convex-mesh fingers, whose mesh<->mesh pair faults
    # Newton's GJK-MPR narrow phase.
    for left in dst.body_shapes[fingers[0]]:
        for right in dst.body_shapes[fingers[1]]:
            dst.add_shape_collision_filter_pair(left, right)
    return dst, {"ee_body": ee, "finger_bodies": fingers, "base_joint": base_joint}


def _finger_joints(src: newton.ModelBuilder, ee_src: int, fingers_src: list) -> list:
    """The source joints driving each finger body, in the same order as ``fingers_src``."""
    by_child = {int(src.joint_child[j]): j for j in range(src.joint_count)}
    out = []
    for finger in fingers_src:
        joint = by_child.get(int(finger))
        if joint is None or int(src.joint_parent[joint]) != int(ee_src):
            raise RuntimeError(
                f"finger body {finger} is not a direct child of the end effector {ee_src} in the "
                f"{FRANKA.short_name} robot — the free-floating hand cannot be lifted out as a "
                f"subtree. Check RobotConfig's link suffixes.")
        out.append(joint)
    return out
