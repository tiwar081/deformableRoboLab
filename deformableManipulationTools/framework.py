"""The simulation framework: the ``GraspExample`` base that owns EVERYTHING shared across demos —
robot+solver build, object-model assembly (finalize ordering, contact materials, masses, VBD
solver), the dynamic-proxy coupling, the per-substep loop, CUDA-graph capture, and viz sync.

A concrete demo subclass supplies only the two things that make it a distinct demo:
  1. the SCENE  — ``configure`` (positions/params) + ``build_scene`` (which objects, where);
  2. the POLICY — ``plan`` (IK keyframes) + ``set_robot_targets`` (the per-substep command kernel).

Everything else (the physics framework, the robot, the asset assembly nuances) is centralized
here and in the sibling modules, so a new demo cannot reintroduce a solver/contact/collision bug.
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton

from .params import FRANKA, GRIP, MUJOCO_GRIP, TABLE
from .robot import build_franka_robot, finger_body_indices, make_robot_solver, set_mujoco_grip_controller
from .grip import TwoWayProxyCoupling, restore_proxy_materials, GripController
from .mathutils import find_body, quat_rotate_xyzw
from .mesh_collision import rescale_body_mass


def build_viz_model(robot_builder, object_builder, device):
    """Combined robot+object model for rendering only. Returns (viz_model, object_body_start)."""
    viz_builder = newton.ModelBuilder()
    viz_builder.add_builder(robot_builder)
    object_body_start = robot_builder.body_count
    viz_builder.add_builder(object_builder)
    return viz_builder.finalize(device=device), object_body_start


class GraspExample:
    """Shared per-frame backbone AND construction. The subclass implements:

      * ``configure(args)``  — set ``table_top_z``, object positions, ``gripper_closed``, and any
        config attributes (``robot_base_xform``, ``robot_table``, ``has_particles``, ``camera``,
        ``soft_block``, ``coupling_soft_ke``, ``object_solver_kwargs``, ``object_pipeline_kwargs``,
        ``blanket_fill``).
      * ``plan(ik_model, ik_state)`` — solve IK keyframes; stash device arrays for the policy.
      * ``build_scene(object_builder, robot_builder)`` — add the table/objects/proxies via the
        :mod:`assets` builders; set ``gripper_proxy_bodies``/``gripper_proxy_shapes`` and record
        body indices; optionally append to ``self.material_overrides`` / ``self.mass_overrides``.
      * ``set_robot_targets(substep)`` — launch the policy kernel onto ``robot_control.joint_target_q``.
      * ``test_final()``.
    """

    # ---- subclass configuration defaults (override in configure() or as class attrs) ----
    has_particles: bool = False
    robot_table = TABLE                                  # build_franka_robot table arg
    robot_base_xform = None                              # None -> table-relative default
    blanket_fill: bool = True                            # fill proxy material then restore overrides
    soft_block = None                                    # SoftBlockConfig -> sets model.soft_contact_*
    coupling_soft_ke = None                              # enable proxy<->particle harvest
    object_solver_kwargs: dict = {}                      # extra SolverVBD kwargs (per scene)
    object_pipeline_kwargs: dict = {}                    # extra CollisionPipeline kwargs (per scene)
    camera = (wp.vec3(0.85, 0.30, 0.55), -22.0, -130.0)  # (eye, pitch, yaw)
    coupling: TwoWayProxyCoupling = None
    has_deformable: bool = False                         # set by the centralized routing in __init__
    object_model = None                                  # None on the rigid-only (single-MuJoCo) path
    robot_contact_max: int = 8192                        # MuJoCo contact buffer (raise for mesh scenes)
    gripper_proxy_bodies = ()                            # set by build_scene on the VBD path (proxies)
    gripper_proxy_shapes = ()
    grasp_windows = None                                 # list[GraspWindow]: enables the centralized
    grip_controller: GripController = None               # force-stop finger controller (set in __init__)

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.gripper_open = FRANKA.gripper_open
        self.home_q = np.array(FRANKA.home_q, dtype=np.float32)
        self.material_overrides: list[dict] = []
        self.mass_overrides: list[tuple[int, float]] = []

        self.configure(args)
        if self.robot_base_xform is None:
            self.robot_base_xform = wp.transform((-0.45, -0.45, self.table_top_z), wp.quat_identity())
        # Convenience copies of the table placement (the robolab object-view cameras read these).
        if self.robot_table is not None:
            self.table_pos = np.array(self.robot_table.pos, dtype=np.float32)
            self.table_half = np.array(self.robot_table.half, dtype=np.float32)

        device = wp.get_device(args.device) if args.device else None
        robot_builder = build_franka_robot(xform=self.robot_base_xform, table=self.robot_table)

        ik_model = robot_builder.finalize(device=device)
        ik_state = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)
        self.ee_body = find_body(list(ik_model.body_label), FRANKA.ee_link_suffix)
        self.ee_offset = np.array(FRANKA.ee_offset, dtype=np.float32)
        self.plan(ik_model, ik_state)
        self.robot_finger_bodies = finger_body_indices(ik_model)
        device = ik_model.device

        # ---- scene declared into a neutral builder, BEFORE the MuJoCo-vs-VBD routing decision ----
        object_builder = newton.ModelBuilder()
        object_builder.default_shape_cfg.ke = GRIP.proxy_ke
        object_builder.default_shape_cfg.kd = GRIP.object_contact_kd
        object_builder.default_shape_cfg.mu = 0.8
        self.build_scene(object_builder, robot_builder)
        object_builder.color(balance_colors=False)

        # CENTRALIZED ROUTING: any deformable present routes ALL objects to VBD + dynamic gripper
        # proxies (the original two-solver bridge); a rigid-only workspace routes robot AND objects
        # into ONE SolverMuJoCo (true two-way grasp, no preset gripper widths, CCD on). FEM/cloth
        # deformables are particles; a ROD (add_rod) is capsule bodies coupled by CABLE joints and
        # carries NO particles, so detect it explicitly — else the cable misroutes to MuJoCo, whose
        # forward kinematics skips CABLE joints and fails to build the articulation.
        has_cable_joint = any(int(t) == int(newton.JointType.CABLE) for t in object_builder.joint_type)
        self.has_deformable = object_builder.particle_count > 0 or has_cable_joint
        rcm = self.robot_contact_max
        if self.has_deformable:
            self._build_split_mujoco_vbd(robot_builder, object_builder, device, rcm, args)
        else:
            self._build_rigid_only_mujoco(robot_builder, object_builder, device, rcm)

        # ---- capture + initial viz ----
        self._t_frame = wp.zeros(1, dtype=wp.float32, device=device)
        # Centralized force-stop finger controller (replaces per-demo preset close widths). Built only
        # when the demo declares grasp_windows; legacy demos that override set_robot_targets are
        # untouched. VBD path reads the harvested grip force; rigid-only closes to MUJOCO_GRIP.close_target.
        if self.grasp_windows is not None:
            self.grip_controller = GripController(
                self.robot_control, self._t_frame, self.sim_dt,
                finger_dofs=range(FRANKA.n_arm_dof, FRANKA.n_dof), gripper_open=self.gripper_open,
                grasp_windows=self.grasp_windows,
                grip_force_signal=(self.coupling.grip_force_signal if self.coupling is not None else None),
                close_target=(GRIP.min_close_width if self.coupling is not None else MUJOCO_GRIP.close_target))
        self.graph = None
        self._frames_simulated = 0
        self._capture_enabled = (
            wp.get_device(str(device)).is_cuda and self.sim_substeps % 2 == 0
            and not (os.environ.get("GRASP_NO_CAPTURE") or os.environ.get("CABLE_NO_CAPTURE")))
        if self.coupling is not None:
            self._sync_gripper_proxies()
        self._sync_viz_state()
        self.viewer.set_model(self.viz_model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_camera"):
            eye, pitch, yaw = self.camera
            self.viewer.set_camera(eye, pitch=pitch, yaw=yaw)

    # ---- build paths (chosen centrally by has_deformable) ----
    def _init_robot_states(self, rcm: int) -> None:
        """Robot states/control/FK + collision pipeline + MuJoCo solver (CCD on). Shared by both
        routing paths; called after ``robot_model`` is finalized."""
        self.robot_state_0 = self.robot_model.state()
        self.robot_state_1 = self.robot_model.state()
        self.robot_control = self.robot_model.control()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_0)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.robot_state_1)
        wp.copy(self.robot_control.joint_target_q, self.robot_model.joint_q)
        self.robot_model.rigid_contact_max = rcm
        self.robot_collision_pipeline = newton.CollisionPipeline(
            self.robot_model, reduce_contacts=True, rigid_contact_max=rcm, broad_phase="nxn")
        self.robot_contacts = self.robot_collision_pipeline.contacts()
        self.robot_solver = make_robot_solver(self.robot_model, rcm)

    def _build_rigid_only_mujoco(self, robot_builder, object_builder, device, rcm: int) -> None:
        """Rigid-only workspace: merge the objects into the robot's builder and run robot+objects in
        ONE SolverMuJoCo — true two-way grasp (fingers close to a fixed target, contact holds the
        object), CCD on. No VBD model / proxies / coupling; the one MuJoCo model is also the viz model."""
        set_mujoco_grip_controller(robot_builder)               # stiffen fingers for a real grasp
        self.object_body_start = robot_builder.body_count       # objects are appended after the robot
        robot_builder.add_builder(object_builder)
        self.robot_model = robot_builder.finalize(device=device)
        self._init_robot_states(rcm)
        # Object masses (e.g. realistic YCB masses) are applied post-finalize on the robot model, with
        # the merge offset — the demo records scene-builder body indices, oblivious to the routing.
        for body, mass in self.mass_overrides:
            rescale_body_mass(self.robot_model, self.object_body_start + int(body), mass)
        self.object_model = None
        self.object_state_0 = None
        self.object_state_1 = None
        self.coupling = None
        self.viz_model = self.robot_model                       # robot+objects already in one model
        self.viz_state = None
        self.viz_object_body_start = self.object_body_start

    def _build_split_mujoco_vbd(self, robot_builder, object_builder, device, rcm: int, args) -> None:
        """Deformable present: MuJoCo robot + VBD object model + dynamic gripper proxies (the original
        two-solver path with the one-way proxy bridge)."""
        self.object_body_start = None
        # Build the viz model FIRST (SOLVERS §4). It shares the robot's and objects' Mesh sources but
        # NEVER collides, so its stale BVHs are harmless — while the robot and object models, finalized
        # AFTER it, each rebuild and OWN the live BVH their collision pipeline reads. Finalizing viz LAST
        # instead frees those shared BVHs and corrupts the narrow-phase. This used to bite only object
        # meshes (the old object-only `mesh_first` ordering); the panda robot's per-link CONVEX_MESH
        # colliders made it bite the ROBOT meshes too, so viz-first — which protects both — replaces it.
        self.viz_model, self.viz_object_body_start = build_viz_model(robot_builder, object_builder, device)
        self.viz_state = self.viz_model.state()
        self.robot_model = robot_builder.finalize(device=device)
        self._init_robot_states(rcm)
        self.object_model = object_builder.finalize(device=device)

        # ---- contact materials ----
        if self.soft_block is not None:
            self.object_model.soft_contact_ke = self.soft_block.soft_contact_ke
            self.object_model.soft_contact_kd = self.soft_block.soft_contact_kd
            self.object_model.soft_contact_kf = self.soft_block.soft_contact_kf
            self.object_model.soft_contact_mu = self.soft_block.soft_contact_mu
        if self.blanket_fill:
            self.object_model.shape_material_ke.fill_(GRIP.proxy_ke)
            self.object_model.shape_material_kd.fill_(GRIP.object_contact_kd)
            self.object_model.shape_material_mu.fill_(0.8)
        for ov in getattr(object_builder, "_robolab_material_restores", []):
            self._apply_material_override(ov)
        for ov in self.material_overrides:
            self._apply_material_override(ov)
        restore_proxy_materials(self.object_model, self.gripper_proxy_shapes)
        for body, mass in self.mass_overrides:
            rescale_body_mass(self.object_model, body, mass)

        # ---- object states / pipeline / solver ----
        self.object_state_0 = self.object_model.state()
        self.object_state_1 = self.object_model.state()
        self.object_control = self.object_model.control()
        pipeline_kwargs = dict(contact_matching="latest")
        pipeline_kwargs.update(self.object_pipeline_kwargs)
        if "rigid_contact_max" in pipeline_kwargs:
            self.object_model.rigid_contact_max = pipeline_kwargs["rigid_contact_max"]
        self.object_collision_pipeline = newton.CollisionPipeline(self.object_model, **pipeline_kwargs)
        self.object_contacts = self.object_model.contacts(collision_pipeline=self.object_collision_pipeline)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_0)
        newton.eval_fk(self.object_model, self.object_model.joint_q, self.object_model.joint_qd, self.object_state_1)
        wp.copy(self.object_control.joint_target_q, self.object_model.joint_q)
        solver_kwargs = dict(iterations=args.vbd_iterations, rigid_contact_stick_motion_eps=0.0)
        solver_kwargs.update(self.object_solver_kwargs)
        self.object_solver = newton.solvers.SolverVBD(self.object_model, **solver_kwargs)
        self.coupling = self._make_coupling()

    # ---- subclass hooks (must override) ----
    def configure(self, args) -> None:
        raise NotImplementedError

    def plan(self, ik_model, ik_state) -> None:
        raise NotImplementedError

    def build_scene(self, object_builder, robot_builder) -> None:
        raise NotImplementedError

    def set_robot_targets(self, substep: int) -> None:
        """Per-substep command. Default (force-stop demos): the subclass writes the ARM DOFs in
        ``set_arm_targets`` and the centralized ``GripController`` writes the finger DOFs from the
        harvested grip force. A legacy demo may instead override this method and drive all 9 DOFs
        itself (then it must not declare ``grasp_windows``)."""
        self.set_arm_targets(substep)
        self.grip_controller.step(substep, self.robot_state_0.joint_q)

    def set_arm_targets(self, substep: int) -> None:
        """Force-stop demos: launch the policy kernel writing ONLY the arm DOFs (0..n_arm_dof-1) of
        ``robot_control.joint_target_q``; the finger DOFs are owned by the GripController."""
        raise NotImplementedError

    # ---- seam (VBD path only) ----
    def _make_coupling(self):
        """Factory for the dynamic-proxy coupling (VBD path). Overridable; default = the gripper-only
        ``TwoWayProxyCoupling``. Any proxy beyond the two finger proxies (the palm/EE blocker added by
        ``build_gripper_proxies``) is pinned to the EE body."""
        mirror_bodies = list(self.robot_finger_bodies)
        mirror_bodies += [self.ee_body] * (len(self.gripper_proxy_bodies) - len(mirror_bodies))
        return TwoWayProxyCoupling(
            self.robot_model, self.object_model, self.object_solver, self.object_contacts,
            self.object_state_0, mirror_bodies, self.gripper_proxy_bodies,
            self.ee_body, self.sim_dt, soft_contact_ke=self.coupling_soft_ke)

    # ---- shared helpers ----
    def tcp_position(self, state) -> np.ndarray:
        body_q = state.body_q.numpy()[self.ee_body]
        return body_q[:3] + quat_rotate_xyzw(body_q[3:7], self.ee_offset)

    def object_body_q(self) -> np.ndarray:
        """Object body transforms [n, 7] indexed by the SCENE-builder body index, regardless of routing
        (objects are in object_model on the VBD path; in robot_model after object_body_start on the
        MuJoCo path). Demos use this in check_physics so they need not track which solver hosts them."""
        if self.object_model is None:
            return self.robot_state_0.body_q.numpy()[self.object_body_start:]
        return self.object_state_0.body_q.numpy()

    def grip_force_norms(self):
        """Per-pad physical grip force [N] (raw harvested object reaction). Empty on the rigid-only
        path, where the grasp is true two-way MuJoCo contact (read it from the robot solver instead)."""
        return [] if self.coupling is None else self.coupling.raw_force_norms()

    def _apply_material_override(self, ov: dict) -> None:
        ke = self.object_model.shape_material_ke.numpy()
        kd = self.object_model.shape_material_kd.numpy()
        mu = self.object_model.shape_material_mu.numpy()
        if "bodies" in ov:
            body_set = set(int(b) for b in ov["bodies"])
            shapes = [s for s, b in enumerate(self.object_model.shape_body.numpy()) if int(b) in body_set]
        else:
            shapes = list(ov["shapes"])
        for s in shapes:
            if ov.get("ke") is not None:
                ke[s] = ov["ke"]
            if ov.get("kd") is not None:
                kd[s] = ov["kd"]
            if ov.get("mu") is not None:
                mu[s] = ov["mu"]
        self.object_model.shape_material_ke.assign(ke)
        self.object_model.shape_material_kd.assign(kd)
        self.object_model.shape_material_mu.assign(mu)

    # ---- per-frame loop (identical for every demo) ----
    def _sync_gripper_proxies(self) -> None:
        self.coupling.sync_proxies(self.robot_state_0, self.object_state_0, self.object_state_1)

    def simulate(self) -> None:
        if self.object_model is None:
            self._simulate_rigid_only()
        else:
            self._simulate_split()

    def _simulate_rigid_only(self) -> None:
        """Rigid-only: robot + objects integrated together by the single MuJoCo solver."""
        for substep in range(self.sim_substeps):
            self.set_robot_targets(substep)
            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(self.robot_state_0, self.robot_state_1, self.robot_control,
                                   self.robot_contacts, self.sim_dt)
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

    def _simulate_split(self) -> None:
        """Deformable present: staggered MuJoCo-robot + VBD-objects step with the one-way proxy bridge."""
        for substep in range(self.sim_substeps):
            self.set_robot_targets(substep)

            self.robot_state_0.clear_forces()
            self.robot_state_1.clear_forces()
            self.coupling.apply_to_robot(self.robot_state_0)
            self.robot_collision_pipeline.collide(self.robot_state_0, self.robot_contacts)
            self.robot_solver.step(self.robot_state_0, self.robot_state_1, self.robot_control,
                                   self.robot_contacts, self.sim_dt)
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

            self._sync_gripper_proxies()
            self.coupling.snapshot(self.object_state_0)

            self.object_state_0.clear_forces()
            self.object_model.collide(self.object_state_0, self.object_contacts)
            self.object_solver.step(self.object_state_0, self.object_state_1, self.object_control,
                                    self.object_contacts, self.sim_dt)
            self.object_state_0, self.object_state_1 = self.object_state_1, self.object_state_0

            self.coupling.harvest(self.object_state_0)

    def step(self) -> None:
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

    def _sync_viz_state(self) -> None:
        if self.object_model is None:
            return  # rigid-only: viz_model IS robot_model; render() logs robot_state_0 directly
        body_q = self.viz_state.body_q.numpy()
        body_qd = self.viz_state.body_qd.numpy()
        n = self.robot_model.body_count
        body_q[:n] = self.robot_state_0.body_q.numpy()
        body_qd[:n] = self.robot_state_0.body_qd.numpy()
        start = self.viz_object_body_start
        end = start + self.object_model.body_count
        body_q[start:end] = self.object_state_0.body_q.numpy()
        body_qd[start:end] = self.object_state_0.body_qd.numpy()
        self.viz_state.body_q.assign(body_q)
        self.viz_state.body_qd.assign(body_qd)
        if self.has_particles:
            wp.copy(self.viz_state.particle_q, self.object_state_0.particle_q)
            wp.copy(self.viz_state.particle_qd, self.object_state_0.particle_qd)

    def render(self) -> None:
        self._sync_viz_state()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.robot_state_0 if self.object_model is None else self.viz_state)
        self.viewer.end_frame()
