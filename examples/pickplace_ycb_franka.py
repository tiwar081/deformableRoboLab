"""Franka picks a rubik's cube and a banana and drops them into a bowl tray.

A FRICTION test: does the gripper hold each object by genuine contact friction (the centralized
dynamic finite-mass proxy grip) through the carry, and drop it physically when the pads open? The
bowl and banana collide as coacd convex-hull pieces (a raw concave bowl mesh ejects the VBD solve,
docs/SOLVERS.md §4) while their full meshes render; the cube is a box rendered as a rubik's cube.

This file is pure SCENE + POLICY: every physics/robot/asset detail comes from
:mod:`deformableManipulationTools`, all rendering from :class:`robolabViz.scenic.ScenicGraspExample`.
The robot is at the origin with its own table (centered at (0.45, 0), top z=0.05) — the scenic
view reads that placement straight off the physics example.

Run: python -m examples pickplace_ycb_franka --device cuda:0
  (scenic by default -> outputs/pickplace_ycb_franka/{frames/, simulation.mp4};
   --output-style basic writes outputs/pickplace_ycb_franka.usd instead.)
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
    GRIP, TABLE_YCB, RUBIKS_CUBE, BOWL_YCB, BANANA_YCB,
    add_table, add_ycb_mesh, add_rubiks_cube, build_gripper_proxies, load_usd_mesh,
    solve_gripper_ik, wp_smoothstep, OBJECTS_DIR,
)

# Recorded start poses from RoboLab's RubiksCubeAndBananaTask (world frame, robot at origin).
CUBE_XY = (0.431, -0.097)
BANANA_XY = (0.539, -0.076)
TRAY_XY = (0.443, 0.127)
BANANA_USD = OBJECTS_DIR / BANANA_YCB.usd_subpath


@wp.func
def _lerp(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, i: int) -> float:
    return (1.0 - s) * a[i] + s * b[i]


@wp.func
def _ss_lerp(a: float, b: float, s: float) -> float:
    w = wp_smoothstep(s)
    return (1.0 - w) * a + w * b


# Two pick-and-place cycles. Each: approach above object, descend, [close], lift, move over the
# tray, [open]. Arm waypoints are 9-vectors; the gripper DOFs follow an explicit open/closed
# schedule (genuine contact holds the object; opening the pads drops it — no spring/latch).
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
    gripper_open: float,
    cube_closed: float,
    banana_closed: float,
    joint_target_q: wp.array(dtype=float),
):
    i = wp.tid()
    t = t_frame[0] + float(substep) * sim_dt

    if t < 1.6:
        q = _lerp(home, a_pre, wp_smoothstep(t / 1.6), i)
    elif t < 2.6:
        q = _lerp(a_pre, a_grasp, wp_smoothstep((t - 1.6) / 1.0), i)
    elif t < 4.0:
        q = a_grasp[i]                                                   # grasp + hold (gripper closes)
    elif t < 4.8:
        q = _lerp(a_grasp, a_lift, wp_smoothstep((t - 4.0) / 0.8), i)
    elif t < 6.0:
        q = _lerp(a_lift, a_drop, wp_smoothstep((t - 4.8) / 1.2), i)
    elif t < 7.0:
        q = a_drop[i]                                                   # hold over tray (gripper opens)
    elif t < 8.6:
        q = _lerp(a_drop, b_pre, wp_smoothstep((t - 7.0) / 1.6), i)     # move above banana
    elif t < 10.2:
        q = _lerp(b_pre, b_grasp, wp_smoothstep((t - 8.6) / 1.6), i)    # descend slowly to the table
    elif t < 12.2:
        q = b_grasp[i]                                                  # stop, grasp + hold
    elif t < 16.2:
        q = _lerp(b_grasp, b_lift, wp_smoothstep((t - 12.2) / 4.0), i)  # slow ascent — don't rush
    elif t < 17.8:
        q = _lerp(b_lift, b_drop, wp_smoothstep((t - 16.2) / 1.6), i)   # carry to the bowl
    elif t < 18.6:
        q = b_drop[i]                                                   # hold (release)
    else:
        q = _lerp(b_drop, home, wp_smoothstep((t - 18.6) / 2.0), i)

    if i >= 7:
        if t < 2.6:
            q = gripper_open
        elif t < 3.0:
            q = _ss_lerp(gripper_open, cube_closed, (t - 2.6) / 0.4)
        elif t < 6.2:
            q = cube_closed
        elif t < 6.8:
            q = _ss_lerp(cube_closed, gripper_open, (t - 6.2) / 0.6)
        elif t < 10.2:
            q = gripper_open
        elif t < 10.8:
            q = _ss_lerp(gripper_open, banana_closed, (t - 10.2) / 0.6)
        elif t < 17.8:
            q = banana_closed
        elif t < 18.4:
            q = _ss_lerp(banana_closed, gripper_open, (t - 17.8) / 0.6)
        else:
            q = gripper_open

    joint_target_q[i] = q


class Example(ScenicGraspExample):
    # TABLE_YCB is an oversized contact slab (1.0 m deep); the DROID maple table
    # need not span it — every object rests near center, on the visible surface.
    scenic_check_table = False

    def configure(self, args):
        self.table_top_z = TABLE_YCB.top_z
        self.robot_base_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())  # FR3 at the origin
        self.robot_table = TABLE_YCB
        self.blanket_fill = False                # per-shape materials from the YCB/box configs
        self.has_particles = False
        self.camera = (wp.vec3(1.1, 0.55, 0.6), -25.0, -130.0)
        self.object_solver_kwargs = {"rigid_body_contact_buffer_size": 16384}
        self.object_pipeline_kwargs = {"rigid_contact_max": 100000, "max_triangle_pairs": 1_048_576,
                                       "reduce_contacts": True}
        self.gripper_proxy_gap = 0.008
        self.cube_half = RUBIKS_CUBE.half_extent
        self.cube_pos = np.array([CUBE_XY[0], CUBE_XY[1], self.table_top_z + self.cube_half], np.float32)
        self.banana_pos = np.array([BANANA_XY[0], BANANA_XY[1], self.table_top_z + 0.02], np.float32)
        self.tray_pos = np.array([TRAY_XY[0], TRAY_XY[1], self.table_top_z], np.float32)
        self.drop_height = self.table_top_z + 0.16
        self.bowl_mass = args.bowl_mass
        self.banana_mass = args.banana_mass

    def plan(self, ik_model, ik_state):
        def ik_at(xy, z, yaw=0.0):
            return solve_gripper_ik(ik_model, ik_state, self.ee_body, self.ee_offset,
                                    np.array([xy[0], xy[1], z], np.float32), self.gripper_open, yaw=yaw)

        # Cube: drop off-center over the bowl rim so it strikes at an angle and knocks the (dynamic) bowl.
        cube_drop_xy = (self.tray_pos[0] + 0.06, self.tray_pos[1])
        self.a_pre = ik_at(CUBE_XY, self.table_top_z + 0.14)
        self.a_grasp = ik_at(CUBE_XY, self.table_top_z + self.cube_half)
        self.a_lift = ik_at(CUBE_XY, self.drop_height)
        self.a_drop = ik_at(cube_drop_xy, self.drop_height)
        # Banana: grasp a quarter down from one end at the centerline (curved); yaw 90 closes the pads
        # across its narrow width.
        bv = np.asarray(load_usd_mesh(BANANA_USD).vertices)
        gy = bv[:, 1].min() + 0.25 * (bv[:, 1].max() - bv[:, 1].min())
        band = np.abs(bv[:, 1] - gy) < 0.012
        gx = float(bv[band, 0].mean())
        grasp_xy = (BANANA_XY[0] + gx, BANANA_XY[1] + gy)
        self.banana_half = float(np.max(np.abs(bv[band, 0] - gx)))
        yaw = np.pi / 2
        self.b_pre = ik_at(grasp_xy, self.table_top_z + 0.14, yaw=yaw)
        self.b_grasp = ik_at(grasp_xy, self.table_top_z - 0.01, yaw=yaw)
        self.b_lift = ik_at(grasp_xy, self.drop_height, yaw=yaw)
        self.b_drop = ik_at((self.tray_pos[0], self.tray_pos[1]), self.drop_height, yaw=yaw)
        # Interference-fit closed widths (the genuine proxy contact gives the squeeze).
        self.cube_closed = self.cube_half + GRIP.proxy_margin - GRIP.grasp_interference
        self.banana_closed = self.banana_half + GRIP.proxy_margin - GRIP.grasp_interference

        kf = lambda a: wp.array(a, dtype=wp.float32, device=ik_model.device)
        self._home_wp = kf(self.home_q)
        self._a = [kf(x) for x in (self.a_pre, self.a_grasp, self.a_lift, self.a_drop)]
        self._b = [kf(x) for x in (self.b_pre, self.b_grasp, self.b_lift, self.b_drop)]

    def build_scene(self, builder, robot_builder):
        self._obj_table_shape = add_table(builder, TABLE_YCB)
        # Bowl: dynamic, rests on the table; coacd convex pieces + realistic mass keep it from ejecting.
        self.bowl_body, _ = add_ycb_mesh(builder, BOWL_YCB,
                                         (self.tray_pos[0], self.tray_pos[1], 0.0), rest_on_z=self.table_top_z)
        self.mass_overrides.append((self.bowl_body, self.bowl_mass))
        # Dynamic finger proxies (the grip contact bridge).
        self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
            builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape, gap=self.gripper_proxy_gap)
        self.cube_body = add_rubiks_cube(builder, self.cube_pos, RUBIKS_CUBE)
        self.banana_body, self.banana_mesh = add_ycb_mesh(builder, BANANA_YCB, self.banana_pos)
        self.mass_overrides.append((self.banana_body, self.banana_mass))

    def set_robot_targets(self, substep):
        wp.launch(_set_robot_targets_kernel, dim=9, inputs=[
            self._t_frame, substep, self.sim_dt, self._home_wp,
            self._a[0], self._a[1], self._a[2], self._a[3],
            self._b[0], self._b[1], self._b[2], self._b[3],
            self.gripper_open, self.cube_closed, self.banana_closed,
        ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

    def check_physics(self):
        bq = self.object_state_0.body_q.numpy()
        if not np.all(np.isfinite(bq)):
            raise ValueError("Non-finite body transform.")
        for name, body in (("cube", self.cube_body), ("banana", self.banana_body), ("bowl", self.bowl_body)):
            if bq[body][2] < self.table_top_z - 0.05:
                raise ValueError(f"The {name} fell through the table/world.")

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
    viewer, args = examples.init(parser, example_name="pickplace_ycb_franka")
    examples.run(Example(viewer, args), args)
