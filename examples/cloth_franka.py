"""Franka manipulates a T-shirt (Newton's `example_cloth_franka` shirt) on THIS framework's split
MuJoCo-robot + VBD-cloth path.

The POLICY here REPLICATES Newton's `example_cloth_franka` corner grasp: the gripper is tilted ~45deg
(Newton's grasp tilt) and runs the same per-corner SCOOP sequence — approach -> descend to the table
SURFACE open -> close -> lift+translate -> drag -> release -> retract — driving the fingers by
per-waypoint open/close (Newton's position activation), not the force GripController. Cloth friction is
matched to Newton (ClothConfig.soft_contact_mu=0.25). The cloth physics (a thin SHELL) is centralized
in `ClothConfig` + `cloth_solver_kwargs` (≈critically-damped particle↔body contact + self-contact).

The sim is STABLE (no blow-up). OUTCOME: the shirt is still NOT lifted. Instrumented runs show the
gripper PROXY (the box that mirrors each finger in the VBD world — our one-way contact bridge) JAMS the
table (~177 N) instead of scooping UNDER the cloth edge, and a sliding pad does not drag the cloth.
This is an architectural limit of the proxy bridge: Newton's robot fingers contact the cloth DIRECTLY
in one VBD solver, whereas ours couples a MuJoCo robot to VBD cloth via mirrored proxies, which cannot
perform a delicate cloth scoop. Left visible (not hidden). See docs/cloths.md.

Run: python -m examples cloth_franka --device cuda:0
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

from deformableManipulationTools.helper import install_terminal_log

_terminal_log = install_terminal_log()

import warp as wp

import examples
from robolabViz.scenic import ScenicGraspExample
from deformableManipulationTools import FRANKA, TABLE, add_table, solve_gripper_ik
from deformableManipulationTools.assets import ClothConfig, add_cloth, cloth_solver_kwargs

CLOTH = ClothConfig()


@wp.kernel
def _scoop_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    kf_t: wp.array(dtype=float),     # keyframe times [n_kf]
    kf_q: wp.array2d(dtype=float),   # keyframe joint targets [n_kf, n_dof] (7 arm + 2 finger)
    n_kf: int,
    joint_target_q: wp.array(dtype=float),
):
    # One thread per DOF. Replicates Newton's example_cloth_franka corner grasp: a TILTED, multi-
    # waypoint SCOOP (approach -> descend to the table surface OPEN -> close -> lift+translate ->
    # drag -> release), driving ALL 9 DOFs (arm from IK keyframes, fingers from per-waypoint
    # open/close — Newton's position-activation gripper), not the force-target GripController.
    d = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt
    seg = int(0)
    for i in range(n_kf):
        if t >= kf_t[i]:
            seg = i
    if seg >= n_kf - 1:
        joint_target_q[d] = kf_q[n_kf - 1, d]
        return
    a = wp.clamp((t - kf_t[seg]) / wp.max(kf_t[seg + 1] - kf_t[seg], 1.0e-6), 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    joint_target_q[d] = (1.0 - a) * kf_q[seg, d] + a * kf_q[seg + 1, d]


class Example(ScenicGraspExample):
    has_particles = True
    soft_block = CLOTH                       # framework reads soft_contact_* off this
    coupling_soft_ke = CLOTH.soft_contact_ke  # enable proxy<->particle grip-force harvest

    # Newton corner-grasp orientation: example_cloth_franka tilts the gripper ~45deg (its clean
    # grasps use quat (w,x,y,z)=(0.9239,-0.3827,0,0) = 45deg about -X) and approaches a corner. We
    # apply the SAME tilt magnitude from our top-down home via solve_gripper_ik(tilt=).
    SCOOP_TILT = 0.7854          # 45 deg (Newton's grasp tilt)
    SCOOP_AXIS = (1.0, 0.0, 0.0)

    def configure(self, args):
        self.table_top_z = TABLE.top_z
        tt = self.table_top_z
        self.cloth_center_xy = np.array([0.10, -0.45], dtype=np.float32)
        self.cloth_drop_pos = np.array([self.cloth_center_xy[0], self.cloth_center_xy[1],
                                        tt + 0.04], dtype=np.float32)
        # ---- Newton example_cloth_franka MOTION, ported to our scene ----
        # Newton runs in cm; its per-corner grasp is: descend the tilted gripper to the table TOP,
        # close, then lift while translating (drag the corner). We replicate that sequence here on the
        # shirt's near (-y) edge — the closest our flattened-shirt scene has to a graspable corner.
        # The fingers are driven by per-waypoint open/close (Newton's position activation), NOT the
        # force GripController, so we leave grasp_windows unset and drive all 9 DOFs in set_robot_targets.
        corner = (0.10, -0.78)        # near (-y) edge of the shirt
        across = (0.30, -0.55)        # drag the grasped corner inward/across (Newton drags +x toward centre)
        gopen = self.gripper_open
        # (time_s, tcp_xyz | None->home, finger_width): approach -> descend to surface OPEN -> close
        # (scoop) -> lift+translate -> drag across -> release -> retract.
        self.scoop_waypoints = [
            (0.0,  None,                             gopen),  # hold home while the shirt settles
            (2.8,  (corner[0], corner[1], tt+0.15),  gopen),  # approach above the corner (tilted)
            (3.9,  (corner[0], corner[1], tt+0.008), gopen),  # DESCEND to the table surface, open
            (4.7,  (corner[0], corner[1], tt+0.008), 0.0),    # CLOSE (scoop the corner)
            (6.2,  (corner[0], corner[1]+0.10, tt+0.12), 0.0),# LIFT + translate
            (7.8,  (across[0], across[1], tt+0.12),  0.0),    # DRAG across
            (8.6,  (across[0], across[1], tt+0.12),  gopen),  # RELEASE
            (9.8,  (across[0], across[1], tt+0.18),  gopen),  # retract
        ]
        self.object_solver_kwargs = cloth_solver_kwargs(CLOTH)
        self.object_pipeline_kwargs = {"soft_contact_margin": CLOTH.contact_margin}

    def plan(self, ik_model, ik_state):
        n_dof = FRANKA.n_dof
        qs = []
        for _, pos, fng in self.scoop_waypoints:
            if pos is None:
                q = np.array(self.home_q, dtype=np.float32)
            else:
                q = np.array(solve_gripper_ik(
                    ik_model, ik_state, self.ee_body, self.ee_offset,
                    np.array(pos, dtype=np.float32), self.gripper_open,
                    tilt=self.SCOOP_TILT, tilt_axis=self.SCOOP_AXIS), dtype=np.float32)
            q[FRANKA.n_arm_dof:n_dof] = fng
            qs.append(q)
        self._kf_t = wp.array(np.array([w[0] for w in self.scoop_waypoints], dtype=np.float32),
                              dtype=wp.float32, device=ik_model.device)
        self._kf_q = wp.array(np.stack(qs).astype(np.float32), dtype=wp.float32, device=ik_model.device)
        self._n_kf = len(self.scoop_waypoints)

    def build_scene(self, builder, robot_builder):
        from deformableManipulationTools import build_gripper_proxies
        self._obj_table_shape = add_table(builder, TABLE)
        self.cloth_particle_start, self.cloth_particle_count = add_cloth(builder, CLOTH, self.cloth_drop_pos)
        # Proxies WITH particle collision so the pads grip the cloth.
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape)

    def set_robot_targets(self, substep):
        # Drive ALL 9 DOFs from the Newton scoop keyframes (arm + fingers); no GripController.
        wp.launch(_scoop_kernel, dim=FRANKA.n_dof, inputs=[
            self._t_frame, substep, self.sim_dt, self._kf_t, self._kf_q, self._n_kf,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        # EXPERIMENT: assert only that the sim stayed sane (finite, no blow-up). Whether the shirt is
        # actually lifted is the OUTCOME we report, not a pass/fail gate.
        pq = self.object_state_0.particle_q.numpy()
        if not np.all(np.isfinite(pq)):
            raise ValueError("Non-finite cloth particle position detected (sim blew up).")
        qd = self.object_state_0.particle_qd.numpy()
        if np.max(np.abs(qd)) > 200.0:
            raise ValueError(f"Cloth particle velocity exploded (max |v|={np.max(np.abs(qd)):.1f} m/s).")

    @staticmethod
    def create_parser():
        # 600 frames (10 s) covers Newton's full per-corner sequence: approach -> descend -> close ->
        # lift -> drag -> release -> retract. The sim is stable throughout (cloth contact ≈critically
        # damped + self-colliding). See the header for the lift OUTCOME (proxy-bridge limitation).
        parser = examples.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument("--substeps", type=int, default=16, help="Simulation substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=6, help="VBD iterations for the cloth.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="cloth_franka")
    examples.run(Example(viewer, args), args)
