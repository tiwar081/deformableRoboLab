"""The pickplace_ycb scene, but with a token soft FEM cube parked in a far table corner — so the
CENTRALIZED routing sees a deformable (particle_count > 0) and runs the workspace on the VBD path
(all objects in SolverVBD + dynamic gripper proxies, preset interference-fit gripper widths) instead
of the single-MuJoCo path that pickplace_ycb_franka (rigid-only) takes. The soft cube is never
touched; its mere presence flips the solver. This is the A/B twin that demonstrates the routing
decision: identical rigid manipulation, deformable-driven solver choice.

Pure SCENE + POLICY (cf. pickplace_ycb_franka). The gripper holds each object by the genuine
dynamic finite-mass proxy grip (preset closed widths give the squeeze); opening the pads drops it.

Run: python -m examples pickplace_ycb_vbd_franka --device cuda:0
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
from deformableManipulationTools import (
    FRANKA, SOFT_BLOCK, TABLE_YCB, RUBIKS_CUBE, BOWL_YCB, BANANA_YCB, PARTICLE_SOLVER_KWARGS, GraspWindow,
    add_table, add_soft_block, add_ycb_mesh, add_rubiks_cube, build_gripper_proxies, load_usd_mesh,
    solve_gripper_ik, wp_smoothstep, OBJECTS_DIR,
)

CUBE_XY = (0.431, -0.097)
BANANA_XY = (0.539, -0.076)
TRAY_XY = (0.443, 0.127)
SOFT_XY = (0.55, -0.28)       # token deformable, ON the visible work table & clear of the objects —
                              # forces the VBD routing, never touched (the DROID table spans y∈[-0.35,0.35])
BANANA_USD = OBJECTS_DIR / BANANA_YCB.usd_subpath


@wp.func
def _lerp(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, i: int) -> float:
    return (1.0 - s) * a[i] + s * b[i]


@wp.kernel
def _set_arm_targets_kernel(
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
    joint_target_q: wp.array(dtype=float),
):
    # ARM DOFs only (0..n_arm_dof-1); the fingers are owned by the centralized force-stop GripController
    # (TWO grasp windows: rubik's cube then banana).
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    if t < 1.6:
        q = _lerp(home, a_pre, wp_smoothstep(t / 1.6), i)
    elif t < 2.6:
        q = _lerp(a_pre, a_grasp, wp_smoothstep((t - 1.6) / 1.0), i)
    elif t < 4.0:
        q = a_grasp[i]
    elif t < 4.8:
        q = _lerp(a_grasp, a_lift, wp_smoothstep((t - 4.0) / 0.8), i)
    elif t < 6.0:
        q = _lerp(a_lift, a_drop, wp_smoothstep((t - 4.8) / 1.2), i)
    elif t < 7.0:
        q = a_drop[i]
    elif t < 8.6:
        q = _lerp(a_drop, b_pre, wp_smoothstep((t - 7.0) / 1.6), i)
    elif t < 10.2:
        q = _lerp(b_pre, b_grasp, wp_smoothstep((t - 8.6) / 1.6), i)
    elif t < 12.2:
        q = b_grasp[i]
    elif t < 16.2:
        q = _lerp(b_grasp, b_lift, wp_smoothstep((t - 12.2) / 4.0), i)
    elif t < 17.8:
        q = _lerp(b_lift, b_drop, wp_smoothstep((t - 16.2) / 1.6), i)
    elif t < 18.6:
        q = b_drop[i]
    else:
        q = _lerp(b_drop, home, wp_smoothstep((t - 18.6) / 2.0), i)

    joint_target_q[i] = q


class Example(ScenicGraspExample):
    scenic_check_table = False
    has_particles = True          # the token soft cube -> particles -> VBD routing
    soft_block = SOFT_BLOCK

    def configure(self, args):
        self.table_top_z = TABLE_YCB.top_z
        self.robot_base_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
        self.robot_table = TABLE_YCB
        self.blanket_fill = False
        self.camera = (wp.vec3(1.1, 0.55, 0.6), -25.0, -130.0)
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 16384,
                                     "rigid_body_particle_contact_buffer_size": 4096, **PARTICLE_SOLVER_KWARGS}
        self.object_pipeline_kwargs = {"rigid_contact_max": 100000, "max_triangle_pairs": 1_048_576,
                                       "reduce_contacts": True, "soft_contact_margin": SOFT_BLOCK.contact_margin}
        self.cube_half = RUBIKS_CUBE.half_extent
        self.cube_pos = np.array([CUBE_XY[0], CUBE_XY[1], self.table_top_z + self.cube_half], np.float32)
        self.banana_pos = np.array([BANANA_XY[0], BANANA_XY[1], self.table_top_z + 0.02], np.float32)
        self.tray_pos = np.array([TRAY_XY[0], TRAY_XY[1], self.table_top_z], np.float32)
        self.soft_start_pos = np.array([SOFT_XY[0], SOFT_XY[1], self.table_top_z], np.float32)
        self.drop_height = self.table_top_z + 0.16
        self.bowl_mass = args.bowl_mass
        self.banana_mass = args.banana_mass
        # Force-stop grasp, TWO windows (both incompressible → latch at FIRST contact + a fixed
        # GRIP.grasp_interference bite, the geometry-free equivalent of the old preset interference-fit
        # widths): rubik's cube close [2.6, 3.8] / release [6.2, 6.8], then banana close [10.2, 11.6] /
        # release [17.8, 18.4]. The banana's grasp pose was raised to table + 1 cm (see plan()) so the
        # pads clear the table and can close onto it.
        self.grasp_windows = [
            GraspWindow(close_start=2.6, close_end=3.8, release_start=6.2, release_end=6.8),
            GraspWindow(close_start=10.2, close_end=11.6, release_start=17.8, release_end=18.4),
        ]

    def plan(self, ik_model, ik_state):
        def ik_at(xy, z, yaw=0.0):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array([xy[0], xy[1], z], np.float32), self.gripper_open, yaw=yaw)

        cube_drop_xy = (self.tray_pos[0] + 0.06, self.tray_pos[1])
        self.a_pre = ik_at(CUBE_XY, self.table_top_z + 0.14)
        self.a_grasp = ik_at(CUBE_XY, self.table_top_z + self.cube_half)
        self.a_lift = ik_at(CUBE_XY, self.drop_height)
        self.a_drop = ik_at(cube_drop_xy, self.drop_height)
        bv = np.asarray(load_usd_mesh(BANANA_USD).vertices)
        gy = bv[:, 1].min() + 0.25 * (bv[:, 1].max() - bv[:, 1].min())
        band = np.abs(bv[:, 1] - gy) < 0.012
        gx = float(bv[band, 0].mean())
        grasp_xy = (BANANA_XY[0] + gx, BANANA_XY[1] + gy)
        self.banana_half = float(np.max(np.abs(bv[band, 0] - gx)))
        yaw = np.pi / 2
        self.b_pre = ik_at(grasp_xy, self.table_top_z + 0.14, yaw=yaw)
        # Raised from table − 1 cm: the sub-table pose jammed the finger pads against the robot-side
        # table collider so they could not close to reach the banana (the force-stop found no contact).
        # At table + 1 cm the pads clear the table and close onto the banana.
        self.b_grasp = ik_at(grasp_xy, self.table_top_z + 0.01, yaw=yaw)
        self.b_lift = ik_at(grasp_xy, self.drop_height, yaw=yaw)
        self.b_drop = ik_at((self.tray_pos[0], self.tray_pos[1]), self.drop_height, yaw=yaw)
        kf = lambda a: wp.array(a, dtype=wp.float32, device=ik_model.device)
        self._home_wp = kf(self.home_q)
        self._a = [kf(x) for x in (self.a_pre, self.a_grasp, self.a_lift, self.a_drop)]
        self._b = [kf(x) for x in (self.b_pre, self.b_grasp, self.b_lift, self.b_drop)]

    def build_scene(self, builder, robot_builder):
        self._obj_table_shape = add_table(builder, TABLE_YCB)
        self.bowl_body, _ = add_ycb_mesh(builder, BOWL_YCB,
                                         (self.tray_pos[0], self.tray_pos[1], 0.0), rest_on_z=self.table_top_z)
        self.mass_overrides.append((self.bowl_body, self.bowl_mass))
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape)
        self.cube_body = add_rubiks_cube(builder, self.cube_pos, RUBIKS_CUBE)
        self.banana_body, self.banana_mesh = add_ycb_mesh(builder, BANANA_YCB, self.banana_pos)
        self.mass_overrides.append((self.banana_body, self.banana_mass))
        # The token deformable — its particles route the whole workspace to VBD. Never manipulated.
        add_soft_block(builder, SOFT_BLOCK, self.soft_start_pos)

    def set_arm_targets(self, substep):
        wp.launch(_set_arm_targets_kernel, dim=FRANKA.n_arm_dof, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_wp,
            self._a[0], self._a[1], self._a[2], self._a[3],
            self._b[0], self._b[1], self._b[2], self._b[3],
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        bq = self.object_body_q()
        if not np.all(np.isfinite(bq)):
            raise ValueError("Non-finite body transform.")
        pq = self.object_state_0.particle_q.numpy()
        if pq.size and not np.all(np.isfinite(pq)):
            raise ValueError("Non-finite soft-body particle position detected.")
        # Gravity guardrail (at rest, end of run): nothing hangs in mid-air. The soft block is never
        # grasped, so it must rest on the table; the rigid objects must end settled near the table/bowl.
        if pq.size and pq[:, 2].min() > self.table_top_z + 0.08:
            raise ValueError("The soft block is floating above the table — gravity/contact not settling it.")
        for name, body in (("cube", self.cube_body), ("banana", self.banana_body), ("bowl", self.bowl_body)):
            if bq[body][2] < self.table_top_z - 0.05:
                raise ValueError(f"The {name} fell through the table/world.")
            if bq[body][2] > self.table_top_z + 0.20:
                raise ValueError(f"The {name} is suspended above the table at rest (gravity not applied?).")

    @staticmethod
    def create_parser():
        parser = examples.create_parser()
        parser.set_defaults(num_frames=1320)
        parser.add_argument("--substeps", type=int, default=16)
        parser.add_argument("--vbd-iterations", type=int, default=12)
        parser.add_argument("--bowl-mass", type=float, default=BOWL_YCB.target_mass)
        parser.add_argument("--banana-mass", type=float, default=BANANA_YCB.target_mass)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="pickplace_ycb_vbd_franka")
    examples.run(Example(viewer, args), args)
