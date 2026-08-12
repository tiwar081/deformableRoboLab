"""Executable claims — the things this pass ASSERTS but that no full trial would catch.

    .venv/bin/python -m deformableManipulationTools.grasp_passes.shake_validate --selfcheck

A trial takes tens of seconds and its verdict is a physics result, so it can never tell you whether
the rig was pointed at the right place to begin with. These checks are the cheap half: the hand
lands on the candidate pose, the shake the kernel applies is the shake the protocol describes, the
object is spawned where the canonical frame says it is, and the rigid-motion metric measures
rotation. Every one of them is a silent-wrong-answer bug if it fails — the sim would run happily and
report a number about the wrong configuration.

Mirrors ``grasp_library --selfcheck`` in style, because the repo has no test framework and this is
where a claim gets to be executable.
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ...mathutils import quat_rotate_xyzw, quat_to_matrix_xyzw
from ...params import FRANKA
from . import protocol as P
from .hand import BASE_DOF, FINGER_DOFS, build_floating_hand, hand_geometry
from .rig import TWO_PI, _base_target_kernel, _kabsch, _pose_delta, _Pose, spawn_object

_TOL = 1.0e-6


def _ok(label: str, cond: bool, detail: str = "", verbose: bool = True) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {label} {detail}")
    if verbose:
        print(f"  ok  {label}{('  ' + detail) if detail else ''}")


def _grasp_pose(position, approach, jaw) -> np.ndarray:
    z = np.asarray(approach, dtype=float)
    z /= np.linalg.norm(z)
    x = np.asarray(jaw, dtype=float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, np.asarray(position, dtype=float)
    return m


def check_hand(verbose: bool = True) -> None:
    """The free-floating hand lands EXACTLY on the candidate pose."""
    geom = hand_geometry()
    r = geom.ee_from_grasp[:3, :3]
    _ok("the measured grasp frame is a right-handed rotation",
        abs(np.linalg.det(r) - 1.0) < _TOL and np.abs(r @ r.T - np.eye(3)).max() < _TOL, verbose=verbose)
    _ok("approach is the EE's +z (the axis FRANKA.ee_offset is measured along)",
        bool(np.allclose(geom.ee_from_grasp[:3, 2], [0.0, 0.0, 1.0], atol=_TOL)), verbose=verbose)
    _ok("the jaw axis is perpendicular to the approach",
        abs(float(geom.ee_from_grasp[:3, 0] @ geom.ee_from_grasp[:3, 2])) < _TOL, verbose=verbose)

    # An awkward pose on purpose: nothing axis-aligned, so a transposed or half-applied rotation
    # cannot pass by coincidence.
    target = _grasp_pose([0.31, -0.17, 0.52], [0.3, -0.5, -0.81], [0.7, 0.6, 0.1])
    builder, info = build_floating_hand(target)
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    ee = body_q[info["ee_body"]]
    tcp = ee[:3] + quat_rotate_xyzw(ee[3:7], np.asarray(FRANKA.ee_offset, dtype=float))
    _ok("the TCP lands on the candidate's grasp centre",
        float(np.abs(tcp - target[:3, 3]).max()) < 1.0e-5,
        f"error {float(np.abs(tcp - target[:3, 3]).max()):.2e} m", verbose=verbose)
    r_ee = quat_to_matrix_xyzw(ee[3:7])
    _ok("the approach axis points where the candidate says",
        float(np.abs(r_ee @ geom.ee_from_grasp[:3, 2] - target[:3, 2]).max()) < 1.0e-5, verbose=verbose)
    _ok("the jaw closing axis points where the candidate says",
        float(np.abs(r_ee @ geom.ee_from_grasp[:3, 0] - target[:3, 0]).max()) < 1.0e-5, verbose=verbose)
    lf, rf = info["finger_bodies"]
    _ok("the jaws start at the Panda's full 80 mm aperture (ACRONYM's pre-grasp)",
        abs(float(np.linalg.norm(body_q[lf][:3] - body_q[rf][:3])) - 2 * FRANKA.gripper_open) < 1.0e-4,
        verbose=verbose)
    _ok("the finger DOFs are where the GripController is told they are",
        model.joint_coord_count == BASE_DOF + 2 and FINGER_DOFS == (BASE_DOF, BASE_DOF + 1),
        f"{model.joint_coord_count} coords", verbose=verbose)
    _ok("the real finger actuator came across with the finger",
        all(abs(float(builder.joint_target_ke[d]) - FRANKA.finger_target_ke) < 1e-6
            and abs(float(builder.joint_effort_limit[d]) - FRANKA.finger_effort) < 1e-6
            for d in FINGER_DOFS),
        f"ke={FRANKA.finger_target_ke}, effort={FRANKA.finger_effort} N", verbose=verbose)


def check_shake(verbose: bool = True) -> None:
    """The shake the KERNEL applies is the shake protocol.py documents.

    Checked at TWO anchors, because the close now ends on convergence rather than on a clock and the
    disturbance schedule slides with it: a window time baked in one place and recomputed in another
    would drift silently."""
    p = P.PROTOCOL
    for anchor in (p.close_start, p.close_deadline):
        w = p.windows(anchor)
        _ok(f"the schedule anchored at t={anchor:.1f} s is monotonic",
            anchor < w.linear_start < w.linear_end == w.angular_start < w.angular_end < w.end,
            verbose=verbose)
    _ok("a trial's frame budget covers the slowest close plus the full disturbance",
        p.max_frames >= p.windows(p.close_deadline).end * p.fps - 1, verbose=verbose)

    anchor = p.close_deadline
    win = p.windows(anchor)
    target = wp.zeros(BASE_DOF + 2, dtype=wp.float32)
    t_frame = wp.zeros(1, dtype=wp.float32)
    worst_lin = worst_ang = 0.0
    for i in range(400):
        t = win.end * i / 399.0
        t_frame.assign(np.array([t], dtype=np.float32))
        wp.launch(_base_target_kernel, dim=1, inputs=[
            t_frame, 0, 0.0, win.linear_start, win.linear_end, win.angular_start, win.angular_end,
            TWO_PI * p.frequency, p.linear_amplitude, p.angular_amplitude], outputs=[target])
        got = target.numpy()
        want_lin, want_ang = p.base_command(t, anchor)
        worst_lin = max(worst_lin, abs(float(got[2]) - want_lin))
        worst_ang = max(worst_ang, abs(float(got[3]) - want_ang))
        if float(np.abs(got[[0, 1, 4, 5]]).max()) > 0.0:
            raise AssertionError("FAILED: the base kernel wrote an axis the protocol does not use")
        if float(np.abs(got[BASE_DOF:]).max()) > 0.0:
            raise AssertionError("FAILED: the base kernel wrote the finger DOFs, which the "
                                 "GripController owns")
    # 2e-5 rather than machine epsilon: the kernel is float32 and the host formula float64, so on a
    # 0.5 rad amplitude the two agree to ~2.5e-6 relative and no tighter. Five orders below the
    # reported quantum, which is what the check is for.
    _ok("the kernel reproduces protocol.base_command", max(worst_lin, worst_ang) < 2.0e-5,
        f"worst {max(worst_lin, worst_ang):.2e}", verbose=verbose)
    _ok("the shake starts and ends at rest (Hann envelope, whole cycles)",
        max(abs(p.base_command(t, anchor)[0]) for t in (win.linear_start, win.linear_end)) < 1.0e-9
        and max(abs(p.base_command(t, anchor)[1])
                for t in (win.angular_start, win.angular_end)) < 1.0e-9, verbose=verbose)
    _ok("the base is commanded to hold still before the close finishes",
        p.base_command(p.close_start, P._NEVER_PROBE) == (0.0, 0.0), verbose=verbose)
    _ok("the shake alone cannot trip the escape detector",
        p.linear_amplitude < P.ESCAPE_RADIUS, verbose=verbose)


def check_metrics(verbose: bool = True) -> None:
    """The rigid-motion metric measures rigid motion."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    pts -= pts.mean(axis=0)      # centred, so the centroid delta is exactly the translation applied
    angle, axis = 0.37, np.array([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    moved = pts @ r.T + np.array([0.01, -0.02, 0.03])
    fit = _kabsch(pts - pts.mean(0), moved - moved.mean(0))
    _ok("Kabsch recovers a known rotation of a point cloud",
        float(np.abs(fit - r).max()) < 1.0e-9, verbose=verbose)
    lin, ang = _pose_delta(_Pose(pts.mean(0), np.eye(3)), _Pose(moved.mean(0), fit))
    _ok("the pose delta reports that rotation and translation",
        abs(ang - angle) < 1e-9 and abs(lin - np.linalg.norm([0.01, -0.02, 0.03])) < 1e-9,
        f"{np.degrees(ang):.2f} deg, {lin * 1000:.2f} mm", verbose=verbose)
    # A deformed cloud must still report the RIGID component, not blow up.
    noisy = moved + rng.normal(scale=0.002, size=moved.shape)
    fit2 = _kabsch(pts - pts.mean(0), noisy - noisy.mean(0))
    _ok("a deforming cloud still reports its rigid component",
        abs(_pose_delta(_Pose(pts.mean(0), np.eye(3)), _Pose(noisy.mean(0), fit2))[1] - angle) < 0.02,
        verbose=verbose)

    _ok("the force target scales with load and clamps at both ends",
        P.force_target(0.0, 0.5, 0.05) == P.FORCE_TARGET_MIN
        and P.force_target(1e6, 0.5, 0.05) == P.FORCE_TARGET_MAX
        and P.force_target(0.5, 0.5, 0.05) > P.force_target(0.1, 0.5, 0.05)
        and P.force_target(0.5, 0.9, 0.05) < P.force_target(0.5, 0.3, 0.05), verbose=verbose)
    # Stated as a property, not a hard-coded expected value, so changing a quantum does not silently
    # turn this into a check of the old one.
    value = 0.01234567
    for quantum in (P.ROUND_LINEAR, P.ROUND_ANGULAR):
        got = P.quantize(value, quantum)
        _ok(f"quantize({quantum:g}) lands on the grid and stays within half a quantum",
            abs(got / quantum - round(got / quantum)) < 1e-6 and abs(got - value) <= 0.5 * quantum,
            verbose=verbose)
    _ok("quantize passes None (an unmeasured field) through",
        P.quantize(None, P.ROUND_LINEAR) is None
        and P.quantize(float("nan"), P.ROUND_LINEAR) is None, verbose=verbose)


def check_spawn(asset_names=("banana", "sponge"), verbose: bool = True) -> None:
    """The object is spawned where the canonical frame says it is.

    THE check a silent wrong answer would hide behind: the canonical frame is fitted to the geometry
    ``grasp_passes.catalog`` resolves, and a candidate's world pose is composed assuming
    ``world_from_body`` is the identity. Several ``assets.py`` builders re-centre or rest what they
    are handed, so if ``spawn_object``'s pre-compensation were wrong, every trial would close on thin
    air a few centimetres off the object — and would still run, and would still report numbers.

    Compared per kind, because "where the material is" is a different array per kind: a single body
    claims only that its transform is the identity (its mesh rides along unchanged), while an FEM
    body is held against the very vertices the frame was fitted to. (The rod case left with cables
    leaving the pipeline's scope, 2026-08-11 — see docs/trajPipeline/grasp-library.md.)"""
    from ..catalog import load_asset

    for name in asset_names:
        asset = load_asset(name)
        builder = newton.ModelBuilder()
        spawned = spawn_object(builder, asset)
        want = np.asarray(asset.vertices, dtype=float)
        if spawned.material_points is None:
            xf = builder.body_q[spawned.bodies[0]]
            pose = np.asarray([float(v) for v in (list(xf.p) + list(xf.q))])
            _ok(f"{name}: single body spawned at the identity (world_from_body == I)",
                float(np.abs(pose - np.array([0, 0, 0, 0, 0, 0, 1.0])).max()) < 1.0e-9,
                verbose=verbose)
            continue
        got = spawned.material_points
        # Centroid AND extent: the centroid alone would miss a mirrored or rescaled spawn, the
        # extent alone would miss a uniform offset. Tolerance is one capsule radius, because a rod's
        # nodes are the chain centreline while the fitted geometry is its outer surface.
        tol = 0.5 * float(np.min(asset.extents))
        err = max(float(np.abs(got.mean(0) - want.mean(0)).max()),
                  float(np.abs(0.5 * (got.min(0) + got.max(0)) - 0.5 * (want.min(0) + want.max(0))).max()))
        _ok(f"{name}: material spawned on its asset frame (world_from_body == I)", err < tol,
            f"worst axis error {err * 1000:.3f} mm (tol {tol * 1000:.1f} mm)", verbose=verbose)
        _ok(f"{name}: spawned material fits the geometry the frame was fitted to",
            bool(np.all(got.min(0) >= want.min(0) - tol) and np.all(got.max(0) <= want.max(0) + tol)),
            verbose=verbose)


def run_selfcheck(verbose: bool = True) -> None:
    print("[shake_validate] self-check")
    check_hand(verbose)
    check_shake(verbose)
    check_metrics(verbose)
    check_spawn(verbose=verbose)
    print("[shake_validate] self-check PASSED")
