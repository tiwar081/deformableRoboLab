"""The Franka FR3 + hand robot: builder, MuJoCo solver, gripper IK. All physical properties come
from :mod:`deformableManipulationTools.params` so the robot is identical across every demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils
from newton import JointTargetMode

from .params import FRANKA, RobotConfig, TableConfig, TABLE, MUJOCO_GRIP, MujocoGripConfig
from .mathutils import find_body, quat_rotate_xyzw, quat_to_vec4


def build_franka_robot(
    xform: wp.transform,
    table: TableConfig | None = TABLE,
    robot: RobotConfig = FRANKA,
) -> newton.ModelBuilder:
    """Franka FR3 + hand builder with the shared actuator/gravcomp config and an optional hidden
    robot-side table collider (stops the gripper at the table surface). Pass ``table=None`` for an
    example that places its own table."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.005
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # Two interchangeable load paths producing the SAME articulation (D6/fixed base + 7 arm revolute +
    # 2 prismatic finger DOFs, arm 0-6 then fingers 7-8). The post-load actuator/gravcomp setup below
    # is identical for both, so switching robots never touches the controller. collapse_fixed_joints
    # folds the hand into link7 (URDF) / panda_link7 (USD) so the EE/finger suffixes resolve the same way.
    if robot.loader == "usd":
        builder.add_usd(
            robot.usd_path,
            xform=xform,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
    else:
        asset_path = newton.utils.download_asset(robot.asset_name)
        builder.add_urdf(
            Path(asset_path) / robot.urdf_subpath,
            xform=xform,
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

    home_q = list(robot.home_q)
    builder.joint_q[: robot.n_dof] = home_q
    builder.joint_target_q[: robot.n_dof] = home_q
    for dof in range(robot.n_dof):
        is_arm = dof < robot.n_arm_dof
        builder.joint_target_ke[dof] = robot.arm_target_ke if is_arm else robot.finger_target_ke
        builder.joint_target_kd[dof] = robot.arm_target_kd if is_arm else robot.finger_target_kd
        builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        builder.joint_effort_limit[dof] = robot.arm_effort if is_arm else robot.finger_effort
        builder.joint_armature[dof] = robot.armature

    gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_attr.values is None:
        gravcomp_attr.values = {}
    for dof in range(robot.n_arm_dof):
        gravcomp_attr.values[dof] = True
    gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body in range(2, builder.body_count):
        gravcomp_body.values[body] = 1.0

    if table is not None:
        table_cfg = newton.ModelBuilder.ShapeConfig(
            margin=table.robot_margin, density=table.robot_density, is_visible=False,
            ke=table.robot_ke, kd=table.robot_kd, mu=table.robot_mu,
        )
        builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(*table.pos), wp.quat_identity()),
            hx=float(table.half[0]), hy=float(table.half[1]), hz=float(table.half[2]),
            cfg=table_cfg, label="robot_contact_table",
        )
    return builder


def set_mujoco_grip_controller(builder: newton.ModelBuilder, cfg: MujocoGripConfig = MUJOCO_GRIP,
                               robot: RobotConfig = FRANKA) -> None:
    """Stiffen the finger position actuators for a TRUE two-way MuJoCo grasp (rigid-only path), so the
    fingers closing to a fixed target develop a real holding force against contact. Call on the robot
    builder BEFORE finalize. Not used by the VBD-proxy path (there the proxies grip)."""
    for dof in range(robot.n_arm_dof, robot.n_dof):          # finger DOFs (7, 8)
        builder.joint_target_ke[dof] = cfg.finger_target_ke
        builder.joint_target_kd[dof] = cfg.finger_target_kd
        builder.joint_effort_limit[dof] = cfg.finger_effort


def finger_body_indices(model_or_labels, robot: RobotConfig = FRANKA) -> list[int]:
    labels = list(model_or_labels.body_label) if hasattr(model_or_labels, "body_label") else list(model_or_labels)
    return [find_body(labels, robot.left_finger_suffix), find_body(labels, robot.right_finger_suffix)]


def make_robot_solver(model, contact_max: int, robot: RobotConfig = FRANKA):
    return newton.solvers.SolverMuJoCo(
        model, solver=robot.solver, integrator=robot.integrator,
        iterations=robot.solver_iterations, ls_iterations=robot.solver_ls_iterations,
        nconmax=contact_max, njmax=contact_max * 2, cone=robot.cone,
        impratio=robot.solver_impratio, use_mujoco_contacts=False,
        ccd_iterations=robot.ccd_iterations, ccd_tolerance=robot.ccd_tolerance,
        enable_multiccd=robot.enable_multiccd,
    )


def _ik_solve_once(model, rot, ee_body: int, ee_offset: np.ndarray, target_pos: np.ndarray,
                   robot: RobotConfig) -> np.ndarray:
    """One LM IK solve toward (target_pos, rot), seeded from the CURRENT ``model.joint_q``."""
    target_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
    pos_obj = ik.IKObjectivePosition(
        link_index=ee_body, link_offset=wp.vec3(*ee_offset),
        target_positions=wp.array([wp.vec3(*target_pos)], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=ee_body, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([quat_to_vec4(rot)], dtype=wp.vec4),
    )
    joint_limits_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0,
    )
    solver = ik.IKSolver(
        model=model, n_problems=1, objectives=[pos_obj, rot_obj, joint_limits_obj],
        lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(target_q, target_q, iterations=64)
    return np.array(target_q.numpy()[0, : robot.n_dof], dtype=np.float32)


def _ik_tcp_residual(model, ee_body: int, ee_offset: np.ndarray, q: np.ndarray,
                     target_pos: np.ndarray, robot: RobotConfig) -> float:
    """|FK(q) TCP − target| [m], via a scratch state cached on the model. ``model.joint_q`` is
    saved/restored EXACTLY, so the verification never perturbs the solver's seed state."""
    scratch = getattr(model, "_robolab_ik_scratch_state", None)
    if scratch is None:
        scratch = model._robolab_ik_scratch_state = model.state()
    saved = model.joint_q.numpy()
    jq = saved.copy()
    jq[: robot.n_dof] = q[: robot.n_dof]
    model.joint_q.assign(jq)
    newton.eval_fk(model, model.joint_q, model.joint_qd, scratch)
    body_q = scratch.body_q.numpy()[ee_body]
    tcp = body_q[:3] + quat_rotate_xyzw(body_q[3:7], np.asarray(ee_offset, dtype=np.float64))
    model.joint_q.assign(saved)
    return float(np.linalg.norm(tcp - np.asarray(target_pos, dtype=np.float64)))


# A keyframe IK solve must actually REACH its target: beyond this TCP miss the solve is retried
# from a seed ladder (the LM solver can stall in a bad branch when a long keyframe sequence, e.g.
# the cloth-folding demo's 33 poses with orientation flips, walks the seed far from the target).
IK_RETRY_TOL = 5.0e-3   # [m]
# An alternate-branch solution is accepted only within this arm-joint L2 distance of the PREVIOUS
# keyframe's solution: consecutive keyframes are SMOOTHSTEPPED IN JOINT SPACE, so an exact but
# far-branch keyframe (e.g. wrist flipped, shoulder swung) would spin/bow the TCP wildly
# mid-segment — worse than a few-cm best-effort miss on a joint-limit-bound pose. A same-branch
# best-effort is preferred over a far-branch exact solve up to IK_FAR_BRANCH_ERR of TCP miss.
IK_BRANCH_DQ_MAX = 2.0    # [rad]
IK_FAR_BRANCH_ERR = 8.0e-2  # [m]


def _ik_target_rot(state, ee_body: int, yaw: float, tilt: float, tilt_axis):
    """Target EE orientation: the state's (home) EE orientation, yawed about world z, then tilted."""
    body_q = state.body_q.numpy()[ee_body]
    rot = wp.transform_get_rotation(wp.transform(*body_q))
    if yaw != 0.0:
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw) * rot
    if tilt != 0.0:
        rot = wp.quat_from_axis_angle(wp.vec3(*tilt_axis), tilt) * rot
    return rot


def solve_gripper_ik_path(model, state, ee_body: int, ee_offset: np.ndarray, waypoints,
                          gripper_open: float, robot: RobotConfig = FRANKA,
                          edge_weights=None) -> list:
    """Branch-consistent IK for a keyframe SEQUENCE (used by ``demo_runner.plan``).

    ``waypoints``: list of ``(pos_or_None, yaw, tilt, tilt_axis)``; ``None`` means the home pose.
    Returns one 9-vector joint target per waypoint (fingers at ``gripper_open``).

    Pass 1 is the status-quo per-pose solve (``solve_gripper_ik`` once per UNIQUE pose in
    first-appearance order, reused for holds) — if every keyframe hits ``IK_RETRY_TOL`` this is
    returned unchanged, so ordinary demos are untouched. When some keyframe MISSES (e.g. the
    cloth-folding demo's 33-pose sequence with orientation flips and joint-limit-bound drag ends),
    a GLOBAL pass re-chooses arm branches jointly: a deterministic seed ladder builds a candidate
    solution set per unique pose, and a shortest path (Viterbi) over the time sequence minimizes
    ``IK_PATH_ERR_WEIGHT·(TCP miss) + Σ w_k·|Δq_k|`` — consecutive keyframes are smoothstepped in
    JOINT space, so branch hops between them bow/spin the TCP mid-segment, and only a joint choice
    over the whole sequence can trade a small same-branch miss against a far-branch exact solve.
    ``edge_weights[k]`` scales the jump cost of segment k→k+1: the caller marks GRIPPED segments
    (fingers closed on an object) heavy so any unavoidable branch hop lands on an open transit."""
    na = robot.n_arm_dof
    home = np.asarray(robot.home_q, dtype=np.float32)

    def key_of(wpt):
        pos, yaw, tilt, axis = wpt
        if pos is None:
            return None
        return (tuple(np.round(np.asarray(pos, dtype=np.float64), 9)),
                round(float(yaw), 9), round(float(tilt), 9), tuple(axis))

    # ---- pass 1: status-quo sequential solve (dedup preserved) ----
    seq_cache: dict = {}
    seq_qs, seq_errs = [], []
    for wpt in waypoints:
        pos, yaw, tilt, axis = wpt
        k = key_of(wpt)
        if k is None:
            seq_qs.append(home.copy())
            seq_errs.append(0.0)
            continue
        if k not in seq_cache:
            q = solve_gripper_ik(model, state, ee_body, ee_offset,
                                 np.asarray(pos, dtype=np.float32), gripper_open,
                                 yaw=yaw, tilt=tilt, tilt_axis=axis, robot=robot)
            seq_cache[k] = (q, _ik_tcp_residual(model, ee_body, ee_offset, q,
                                                np.asarray(pos, dtype=np.float32), robot))
        seq_qs.append(seq_cache[k][0].copy())
        seq_errs.append(seq_cache[k][1])
    if max(seq_errs) <= IK_RETRY_TOL:
        return seq_qs

    # ---- pass 2: global branch choice (only for sequences with missed keyframes) ----
    lo = model.joint_limit_lower.numpy()[:na]
    hi = model.joint_limit_upper.numpy()[:na]
    wrist_flip = home.copy()
    wrist_flip[5] -= 1.5
    wrist_flip[6] -= 1.5
    cand_cache: dict = {}
    for wpt, q_seq, e_seq in zip(waypoints, seq_qs, seq_errs):
        k = key_of(wpt)
        if k is None or k in cand_cache:
            continue
        pos, yaw, tilt, axis = wpt
        pos = np.asarray(pos, dtype=np.float32)
        rot = _ik_target_rot(state, ee_body, yaw, tilt, axis)
        rng = np.random.default_rng(abs(hash(k)))
        seeds = [home, wrist_flip] + [lo + rng.random(na).astype(np.float32) * (hi - lo)
                                      for _ in range(8)]
        cands = [(q_seq, e_seq)]
        for seed in seeds:
            jq = model.joint_q.numpy()
            jq[:na] = seed[:na]
            model.joint_q.assign(jq)
            q_i = _ik_solve_once(model, rot, ee_body, ee_offset, pos, robot)
            e_i = _ik_tcp_residual(model, ee_body, ee_offset, q_i, pos, robot)
            if all(np.linalg.norm(q_i[:na] - c[0][:na]) > 0.05 for c in cands):
                cands.append((q_i, e_i))
        cand_cache[k] = cands

    node_cands = [[(home.copy(), 0.0)] if key_of(w) is None else cand_cache[key_of(w)]
                  for w in waypoints]

    weights = ([1.0] * (len(waypoints) - 1)) if edge_weights is None else list(edge_weights)

    def viterbi(cands_per_node):
        # cost = per-node scaled TCP miss + weighted joint-space edge lengths; anchored at home.
        prev_cost = [IK_PATH_ERR_WEIGHT * max(0.0, e - IK_RETRY_TOL)
                     + float(np.linalg.norm(c[:na] - home[:na]))
                     for c, e in cands_per_node[0]]
        back = []
        for knode in range(1, len(cands_per_node)):
            costs, bptr = [], []
            w_edge = weights[knode - 1]
            for c, e in cands_per_node[knode]:
                best_j, best = 0, np.inf
                for j, (cj, _) in enumerate(cands_per_node[knode - 1]):
                    v = prev_cost[j] + w_edge * float(np.linalg.norm(c[:na] - cj[:na]))
                    if v < best:
                        best, best_j = v, j
                costs.append(best + IK_PATH_ERR_WEIGHT * max(0.0, e - IK_RETRY_TOL))
                bptr.append(best_j)
            prev_cost = costs
            back.append(bptr)
        idx = int(np.argmin(prev_cost))
        choice = [0] * len(cands_per_node)
        for knode in range(len(cands_per_node) - 1, 0, -1):
            choice[knode] = idx
            idx = back[knode - 1][idx]
        choice[0] = idx
        return choice

    choice = viterbi(node_cands)
    # One refinement sweep: re-solve each keyframe seeded from its CHOSEN neighbours' solutions —
    # the ladder above cannot produce "the same branch as the neighbour" candidates (e.g. a
    # joint-limit-pinned near-solution reachable only from the adjacent keyframe's config) — then
    # re-run the branch choice with the enriched candidate sets.
    for knode, w in enumerate(waypoints):
        k = key_of(w)
        if k is None:
            continue
        pos, yaw, tilt, axis = w
        pos = np.asarray(pos, dtype=np.float32)
        rot = _ik_target_rot(state, ee_body, yaw, tilt, axis)
        for jnode in (knode - 1, knode + 1):
            if not (0 <= jnode < len(waypoints)):
                continue
            seed = node_cands[jnode][choice[jnode]][0]
            jq = model.joint_q.numpy()
            jq[:na] = seed[:na]
            model.joint_q.assign(jq)
            q_i = _ik_solve_once(model, rot, ee_body, ee_offset, pos, robot)
            e_i = _ik_tcp_residual(model, ee_body, ee_offset, q_i, pos, robot)
            if all(np.linalg.norm(q_i[:na] - c[0][:na]) > 0.05 for c in cand_cache[k]):
                cand_cache[k].append((q_i, e_i))
    node_cands = [[(home.copy(), 0.0)] if key_of(w) is None else cand_cache[key_of(w)]
                  for w in waypoints]
    choice = viterbi(node_cands)

    out = []
    worst = 0.0
    for knode, w in enumerate(waypoints):
        q, e = node_cands[knode][choice[knode]]
        q = q.copy()
        if key_of(w) is not None:
            q[robot.n_arm_dof: robot.n_dof] = gripper_open
        out.append(q)
        worst = max(worst, e)
    if worst > IK_RETRY_TOL:
        print(f"[robot.ik] keyframe path: globally-consistent branches chosen; worst best-effort "
              f"TCP miss {worst * 1e3:.1f} mm (joint-limit-bound pose(s))")
    return out


# Node-miss weight of the global keyframe-path choice: [rad of joint travel] per [m of TCP miss].
# 60 means a 43 mm limit-bound miss (2.6) is preferred over a 5.5 rad branch hop, while a genuine
# exact solution wins over one 0.1 rad closer that misses by centimetres.
IK_PATH_ERR_WEIGHT = 60.0


def solve_gripper_ik(model, state, ee_body: int, ee_offset: np.ndarray, target_pos: np.ndarray,
                     gripper_open: float, yaw: float = 0.0, tilt: float = 0.0,
                     tilt_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
                     robot: RobotConfig = FRANKA) -> np.ndarray:
    """IK the TCP (link7 + ee_offset) to ``target_pos`` keeping the home orientation, optionally
    yawed about world z then TILTED about ``tilt_axis`` (world frame) by ``tilt`` rad. A non-zero
    tilt turns the straight-down approach into an angled one — needed to SCOOP/pinch a thin object
    (e.g. a cloth corner) whose flat top a vertical jaw can only press, not grasp. Returns the
    9-vector joint target with the fingers held open.

    Every solve is VERIFIED by FK. A solve within ``IK_RETRY_TOL`` returns immediately (status-quo
    seeding preserved). Otherwise a deterministic seed LADDER is tried — home, a wrist-flipped
    home, and 8 target-hashed random seeds — and the winner is the within-tolerance candidate
    CLOSEST in arm-joint space to the entry-seeded solution (branch continuity, see
    ``IK_BRANCH_DQ_MAX``), else the smallest-error candidate. Measured: without this, the
    cloth-folding keyframe sequence accumulated 130–240 mm misses on its later poses; with it only
    genuinely joint-limit-bound poses miss (warned)."""
    body_q = state.body_q.numpy()[ee_body]
    ee_tf = wp.transform(*body_q)
    rot = wp.transform_get_rotation(ee_tf)
    if yaw != 0.0:
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw) * rot
    if tilt != 0.0:
        rot = wp.quat_from_axis_angle(wp.vec3(*tilt_axis), tilt) * rot

    na = robot.n_arm_dof
    q_prev = model.joint_q.numpy()[:na].copy()    # the PREVIOUS accepted keyframe (continuity anchor:
                                                  # consecutive keyframes are smoothstepped in joint space)
    q = _ik_solve_once(model, rot, ee_body, ee_offset, target_pos, robot)
    err = _ik_tcp_residual(model, ee_body, ee_offset, q, target_pos, robot)

    def dq(c):
        return float(np.linalg.norm(c[0][:na] - q_prev))

    if err > IK_RETRY_TOL or dq((q, err)) > IK_BRANCH_DQ_MAX:
        home = np.asarray(robot.home_q, dtype=np.float32)
        wrist_flip = home.copy()
        wrist_flip[5] -= 1.5                      # the panda's q6 range is asymmetric; this seed
        wrist_flip[6] -= 1.5                      # reaches the wrist branch the home seed cannot
        lo = model.joint_limit_lower.numpy()[:na]
        hi = model.joint_limit_upper.numpy()[:na]
        rng = np.random.default_rng(abs(hash((tuple(np.round(np.asarray(target_pos, dtype=np.float64), 6)),
                                              round(float(yaw), 6), round(float(tilt), 6)))))
        jitter = [np.clip(q_prev + rng.normal(0.0, 0.3, na).astype(np.float32), lo, hi)
                  for _ in range(4)]              # near-anchor first: a same-branch solution wins
        seeds = [q_prev, *jitter, home, wrist_flip] + [lo + rng.random(na).astype(np.float32) * (hi - lo)
                                                       for _ in range(8)]
        cands = [(q, err)]
        for seed in seeds:
            jq = model.joint_q.numpy()
            jq[:na] = seed[:na]
            model.joint_q.assign(jq)
            q_i = _ik_solve_once(model, rot, ee_body, ee_offset, target_pos, robot)
            e_i = _ik_tcp_residual(model, ee_body, ee_offset, q_i, target_pos, robot)
            cands.append((q_i, e_i))
            if e_i <= IK_RETRY_TOL and dq(cands[-1]) <= 1.0:
                break                              # exact AND same-branch: no need to search further
        # Selection, best first: (1) exact same-branch; (2) same-branch within the far-branch error
        # bound — a smooth few-cm best-effort (e.g. a joint-limit-bound drag end) beats an exact
        # keyframe on a far branch, whose joint-space blend would spin/bow the TCP mid-segment;
        # (3) exact far-branch; (4) global best-effort.
        close = [c for c in cands if dq(c) <= IK_BRANCH_DQ_MAX]
        exact_close = [c for c in close if c[1] <= IK_RETRY_TOL]
        exact_any = [c for c in cands if c[1] <= IK_RETRY_TOL]
        if exact_close:
            q, err = min(exact_close, key=dq)
        elif close and min(c[1] for c in close) <= IK_FAR_BRANCH_ERR:
            q, err = min(close, key=lambda c: c[1])
        elif exact_any:
            q, err = min(exact_any, key=dq)
        else:
            q, err = min(cands, key=lambda c: c[1])
        if err > IK_RETRY_TOL:
            print(f"[robot.ik] keyframe IK best-effort: target {tuple(np.round(target_pos, 3))} "
                  f"missed by {err * 1e3:.1f} mm (joint-limit-bound; same-branch solution preferred)")
    # Explicit chaining: the next keyframe's solve (and its retry-ladder continuity anchor) seeds
    # from THIS accepted solution, so consecutive keyframes stay on the same arm branch and the
    # joint-space smoothstep between them cannot swing the TCP through a wild arc.
    jq = model.joint_q.numpy()
    jq[: robot.n_dof] = q
    model.joint_q.assign(jq)
    q[robot.n_arm_dof : robot.n_dof] = gripper_open
    return q
