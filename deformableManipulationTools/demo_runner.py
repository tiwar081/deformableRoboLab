"""Data-driven demo runner: ONE generic example that plays any demo declared as a DATA FILE.

A demo file in ``examples/`` declares a :class:`DemoSpec` — only the SCENE (which objects, where) and
the POLICY (gripper waypoints, when it grasps + ``force_target``, when it releases). Everything else
(robot, solver, grip, contact materials, the per-substep policy executor) is here and shared, so adding
a demo means writing one data file and changing nothing in :mod:`example` or this module.

The policy executor reproduces every motion the hand-written demos used, exactly:
  * waypoint blend  — smoothstep between IK keyframes (``WP``);
  * grasp           — ``GraspWindow`` list → the centralized force GripController (arm-only kernel), OR
                      an explicit ``finger_schedule`` (smoothstep finger keyframes, all-9-DOF kernel);
  * sweep           — additive ``amp·ramp·sin(2π·f·(t−t₀)+φ)`` per joint (:class:`Sweep`).
IK waypoints accept ``yaw`` and ``tilt`` (so the YCB banana and the cloth scoop transfer unchanged).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import warp as wp

from .params import (FRANKA, TABLE, RIGID_CUBE, CABLE, SOFT_BLOCK, PLATE, RUBIKS_CUBE, GraspWindow)
from .assets import (add_table, add_rigid_box, add_cable, add_soft_block, add_plate, add_ycb_mesh,
                     add_rubiks_cube, ClothConfig, add_cloth)
from .robot import solve_gripper_ik
from .grip import build_gripper_proxies
from .mathutils import wp_smoothstep

# Particle-deformable kinds whose presence routes a scene to the VBD path + needs gripper proxies.
_DEFORMABLE_KINDS = {"cable", "soft_block", "cloth", "plate_soft"}


# ---------------------------------------------------------------------------------------------
# Data schema — what a demo file declares.
# ---------------------------------------------------------------------------------------------
@dataclass
class Obj:
    """One scene object. ``pos`` is an xyz (or, for ``cable``, a list of node xyz), or a
    ``callable(ctx)`` returning the same (for placements computed from the robot pose, e.g. the cable
    laid from the home TCP). ``config`` defaults to the canonical config for the kind."""
    kind: str
    config: Any = None
    pos: Any = None
    half: float | None = None         # rigid_box half-extent
    rest_on_z: bool = False           # ycb_mesh: drop so its lowest point rests on the table top
    yaw: float = 0.0                  # cloth yaw
    mass: float | None = None         # ycb post-finalize mass override


@dataclass
class WP:
    """One arm waypoint: at sim time ``t`` the TCP is at ``pos`` (xyz, or callable(ctx)->xyz, or
    None→home pose) with optional ``yaw``/``tilt``. The executor smoothsteps between consecutive WPs."""
    t: float
    pos: Any = None
    yaw: float = 0.0
    tilt: float = 0.0
    tilt_axis: tuple = (1.0, 0.0, 0.0)


@dataclass
class Sweep:
    """Procedural side-to-side sweep added to selected arm joints after ``start`` (the cable demos):
    ``q[j] += amp·smoothstep((t−start)/ramp)·sin(2π·freq·(t−start)+phase)`` for each (j, amp, phase)."""
    start: float
    freq: float
    ramp: float
    terms: list   # [(joint_index, amplitude, phase), ...]


@dataclass
class DemoSpec:
    scene: list                                  # list[Obj]
    waypoints: list                              # list[WP] (arm)
    grasp_windows: list | None = None            # list[GraspWindow] -> GripController; None -> explicit fingers
    finger_schedule: list | None = None          # explicit mode: [(t, width), ...] (smoothstep keyframes)
    sweep: Sweep | None = None
    # grasped particle deformable -> harvest its reaction (proxy<->particle); else None.
    coupling_soft_ke: float | None = None
    # scene-specific solver/pipeline buffers (the per-deformable PARTICLE config is added centrally).
    object_solver_kwargs: dict = field(default_factory=dict)
    object_pipeline_kwargs: dict = field(default_factory=dict)
    # robot / workspace
    robot_base_xform: Any = None                 # None -> table-relative default; or wp.transform
    robot_table: Any = TABLE
    blanket_fill: bool = True
    robot_contact_max: int = 8192
    camera: Any = None                           # None -> framework default
    scenic_check_table: bool = True
    # run knobs (argparse defaults)
    substeps: int = 16
    vbd_iterations: int = 12
    num_frames: int = 720
    extra_args: dict = field(default_factory=dict)   # name -> default (e.g. bowl_mass); exposed as --name
    check: Callable | None = None                # optional post-run validation: check(example)


def _resolve(v, ctx):
    return v(ctx) if callable(v) else v


def cable_layout(home_tcp, table=TABLE, cable=CABLE):
    """Bowed cable laid on the table from the home TCP (clamped to the table footprint), reproducing the
    layout the cable demos build. Returns ``(node_positions, grasp_pos)`` — pass node_positions to a
    ``cable`` scene Obj and grasp_pos to the grasp waypoint (both as callables of ``ctx.home_tcp``)."""
    start = np.array(home_tcp).copy()
    direction = np.array([1.0, 0.05, 0.0], dtype=np.float32); direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
    extent = direction * cable.segment_length * (cable.node_count - 1)
    m = 0.04
    for ax in range(2):
        start[ax] = np.clip(start[ax], table.pos[ax] - table.half[ax] + m,
                            table.pos[ax] + table.half[ax] - m - max(extent[ax], 0.0))
    start[2] = table.top_z + cable.radius
    nodes = [(start + direction * cable.segment_length * i
              + normal * cable.bow * math.sin(math.pi * i / (cable.node_count - 1))).astype(np.float32)
             for i in range(cable.node_count)]
    grasp = (0.5 * (nodes[3] + nodes[4])).copy(); grasp[2] = table.top_z
    return nodes, grasp


# ---------------------------------------------------------------------------------------------
# Policy executor kernel — reproduces waypoint-blend + sweep (arm) and finger keyframes exactly.
# ---------------------------------------------------------------------------------------------
@wp.func
def _seg(kf_t: wp.array(dtype=float), n: int, t: float) -> int:
    s = int(0)
    for i in range(n):
        if t >= kf_t[i]:
            s = i
    return s


@wp.kernel
def _policy_kernel(
    t_frame: wp.array(dtype=float),
    substep: int,
    sim_dt: float,
    akf_t: wp.array(dtype=float),            # arm keyframe times [na]
    akf_q: wp.array2d(dtype=float),          # arm keyframe joint targets [na, 7]
    n_akf: int,
    fkf_t: wp.array(dtype=float),            # finger keyframe times [nf]
    fkf_w: wp.array(dtype=float),            # finger keyframe widths [nf]
    n_fkf: int,
    has_sweep: int,
    sweep_start: float,
    sweep_freq: float,
    sweep_ramp: float,
    sweep_amp: wp.array(dtype=float),        # [7]
    sweep_phase: wp.array(dtype=float),      # [7]
    joint_target_q: wp.array(dtype=float),
):
    d = wp.tid()                              # launched dim = 7 (arm-only) or 9 (explicit fingers)
    t = t_frame[0] + float(substep) * sim_dt
    if d < 7:
        s = _seg(akf_t, n_akf, t)
        if s >= n_akf - 1:
            q = akf_q[n_akf - 1, d]
        else:
            a = wp.clamp((t - akf_t[s]) / wp.max(akf_t[s + 1] - akf_t[s], 1.0e-6), 0.0, 1.0)
            a = a * a * (3.0 - 2.0 * a)
            q = (1.0 - a) * akf_q[s, d] + a * akf_q[s + 1, d]
        if has_sweep == 1 and t >= sweep_start:
            ramp = wp_smoothstep((t - sweep_start) / sweep_ramp)
            q += sweep_amp[d] * ramp * wp.sin(2.0 * 3.141592653589793 * sweep_freq * (t - sweep_start) + sweep_phase[d])
        joint_target_q[d] = q
    else:                                     # finger DOFs 7,8 — independent smoothstep keyframe timeline
        s = _seg(fkf_t, n_fkf, t)
        if s >= n_fkf - 1:
            joint_target_q[d] = fkf_w[n_fkf - 1]
        else:
            a = wp.clamp((t - fkf_t[s]) / wp.max(fkf_t[s + 1] - fkf_t[s], 1.0e-6), 0.0, 1.0)
            a = a * a * (3.0 - 2.0 * a)
            joint_target_q[d] = (1.0 - a) * fkf_w[s] + a * fkf_w[s + 1]


# Imported lazily by example.py; subclasses ScenicGraspExample so demos get the RoboLab render for free.
def make_data_driven_example(base_cls):
    class DataDrivenExample(base_cls):
        spec: DemoSpec = None  # set by the factory below

        def configure(self, args):
            s = self.spec
            self.table_top_z = s.robot_table.top_z if s.robot_table is not None else TABLE.top_z
            self.robot_table = s.robot_table
            if s.robot_base_xform is not None:
                self.robot_base_xform = s.robot_base_xform
            self.blanket_fill = s.blanket_fill
            self.robot_contact_max = s.robot_contact_max
            if s.camera is not None:
                self.camera = s.camera
            self.scenic_check_table = s.scenic_check_table
            self.grasp_windows = s.grasp_windows
            self.coupling_soft_ke = s.coupling_soft_ke
            self.object_solver_kwargs = dict(s.object_solver_kwargs)
            self.object_pipeline_kwargs = dict(s.object_pipeline_kwargs)
            # has_particles + soft_block are DERIVED from the scene (no demo override needed): the
            # grasped/contacting particle deformable's config sets model.soft_contact_*.
            self._deformable_cfg = self._scene_deformable_cfg()
            self.has_particles = self._deformable_cfg is not None and self._scene_has_particles()
            self.soft_block = self._deformable_cfg if self._scene_has_particles() else None

        def _scene_has_particles(self):
            return any(o.kind in ("soft_block", "cloth", "plate_soft") for o in self.spec.scene)

        def _scene_deformable_cfg(self):
            for o in self.spec.scene:                 # the particle deformable whose soft_contact_* apply
                if o.kind == "cloth":
                    return o.config or ClothConfig()
                if o.kind == "soft_block":
                    return o.config or SOFT_BLOCK
            return None

        def _ctx(self, ik_model, ik_state):
            from types import SimpleNamespace
            home_tcp = self.tcp_position(ik_state)
            return SimpleNamespace(ik_model=ik_model, ik_state=ik_state, ee_body=self.ee_body,
                                   ee_offset=self.ee_offset, gripper_open=self.gripper_open,
                                   home_q=self.home_q, home_tcp=home_tcp, table=self.robot_table,
                                   table_top_z=self.table_top_z, tcp_position=self.tcp_position)

        def plan(self, ik_model, ik_state):
            ctx = self._ctx(ik_model, ik_state)
            self._ctx_cache = ctx
            dev = ik_model.device
            # ---- arm keyframes (IK each UNIQUE waypoint pose ONCE, in first-appearance order; reuse for
            # holds). solve_gripper_ik seeds from model.joint_q and chains, so a hand-written demo IKs each
            # distinct pose once in sequence — deduping here reproduces that exact chain (an extra IK call
            # for a repeated hold would mutate the seed and shift later keyframes). None pos -> home. ----
            akf_t, akf_q, ik_cache = [], [], {}
            for w in self.spec.waypoints:
                pos = _resolve(w.pos, ctx)
                if pos is None:
                    q = np.array(self.home_q, dtype=np.float32)
                else:
                    key = (tuple(np.round(np.asarray(pos, dtype=np.float64), 9)),
                           round(float(w.yaw), 9), round(float(w.tilt), 9), tuple(w.tilt_axis))
                    if key not in ik_cache:
                        ik_cache[key] = np.array(solve_gripper_ik(
                            ik_model, ik_state, self.ee_body, self.ee_offset,
                            np.array(pos, dtype=np.float32), self.gripper_open,
                            yaw=w.yaw, tilt=w.tilt, tilt_axis=w.tilt_axis), dtype=np.float32)
                    q = ik_cache[key]
                akf_t.append(float(w.t)); akf_q.append(q[:FRANKA.n_arm_dof])
            self._akf_t = wp.array(np.array(akf_t, dtype=np.float32), dtype=wp.float32, device=dev)
            self._akf_q = wp.array(np.stack(akf_q).astype(np.float32), dtype=wp.float32, device=dev)
            self._n_akf = len(akf_t)
            # ---- finger keyframes (explicit mode only) ----
            fs = self.spec.finger_schedule or [(0.0, self.gripper_open)]
            self._fkf_t = wp.array(np.array([f[0] for f in fs], dtype=np.float32), dtype=wp.float32, device=dev)
            self._fkf_w = wp.array(np.array([f[1] for f in fs], dtype=np.float32), dtype=wp.float32, device=dev)
            self._n_fkf = len(fs)
            # ---- sweep ----
            sw = self.spec.sweep
            amp = np.zeros(FRANKA.n_arm_dof, dtype=np.float32); phase = np.zeros(FRANKA.n_arm_dof, dtype=np.float32)
            if sw is not None:
                for j, a, p in sw.terms:
                    amp[j] = a; phase[j] = p
            self._sweep_amp = wp.array(amp, dtype=wp.float32, device=dev)
            self._sweep_phase = wp.array(phase, dtype=wp.float32, device=dev)
            self._has_sweep = 1 if sw is not None else 0
            self._sweep = sw

        def build_scene(self, builder, robot_builder):
            ctx = self._ctx_cache
            self._obj_table_shape = None
            self._obj_bodies = {}
            for o in self.spec.scene:
                pos = _resolve(o.pos, ctx)
                if o.kind == "table":
                    self._obj_table_shape = add_table(builder, o.config or TABLE)
                elif o.kind == "rigid_box":
                    body, _ = add_rigid_box(builder, np.array(pos, np.float32), o.half, o.config or RIGID_CUBE)
                    self._obj_bodies.setdefault("rigid_box", []).append(body)
                elif o.kind == "cable":
                    self._cable_body_start = builder.body_count
                    bodies, _, _ = add_cable(builder, pos, o.config or CABLE)
                    self._cable_bodies = bodies
                elif o.kind == "soft_block":
                    add_soft_block(builder, o.config or SOFT_BLOCK, np.array(pos, np.float32))
                elif o.kind == "plate":
                    body, _ = add_plate(builder, np.array(pos, np.float32), o.config or PLATE)
                    self._obj_bodies.setdefault("plate", []).append(body)
                elif o.kind == "ycb_mesh":
                    rest = self.table_top_z if o.rest_on_z else None
                    body, mesh = add_ycb_mesh(builder, o.config, np.array(pos, np.float32), rest_on_z=rest)
                    self._obj_bodies.setdefault("ycb", []).append((body, mesh))
                    if o.mass is not None:
                        self.mass_overrides.append((body, o.mass))
                elif o.kind == "rubiks_cube":
                    body = add_rubiks_cube(builder, np.array(pos, np.float32), o.config or RUBIKS_CUBE)
                    self._obj_bodies.setdefault("rubiks", []).append(body)
                elif o.kind == "cloth":
                    add_cloth(builder, o.config or ClothConfig(), np.array(pos, np.float32), yaw=o.yaw)
                elif o.kind == "proxies":
                    # Build the gripper proxies HERE (exact builder position preserves the original body
                    # order -> byte-identical object model). VBD demos place a single Obj("proxies").
                    self.gripper_proxy_bodies, self.gripper_proxy_shapes = build_gripper_proxies(
                        builder, robot_builder, self.robot_finger_bodies, self._obj_table_shape)
                else:
                    raise ValueError(f"unknown scene object kind {o.kind!r}")

        def _launch_policy(self, substep, n_out):
            wp.launch(_policy_kernel, dim=n_out, inputs=[
                self._t_frame, substep, self.sim_dt, self._akf_t, self._akf_q, self._n_akf,
                self._fkf_t, self._fkf_w, self._n_fkf, self._has_sweep,
                (self._sweep.start if self._sweep else 0.0), (self._sweep.freq if self._sweep else 0.0),
                (self._sweep.ramp if self._sweep else 1.0), self._sweep_amp, self._sweep_phase,
            ], outputs=[self.robot_control.joint_target_q], device=self.robot_control.joint_target_q.device)

        def set_arm_targets(self, substep):                 # grasp_windows mode: arm only; GripController -> fingers
            self._launch_policy(substep, FRANKA.n_arm_dof)

        def set_robot_targets(self, substep):
            if self.grip_controller is not None:            # framework default: arm kernel + GripController
                super().set_robot_targets(substep)
            else:                                            # explicit-finger mode: drive all 9 DOFs
                self._launch_policy(substep, FRANKA.n_dof)

        def check_physics(self):
            # Centralized sanity (every demo): nothing blew up. The hand-written demos' detailed
            # success-assertions are intentionally NOT in the data files (scene + policy only); a demo
            # may attach one via DemoSpec.check if it wants a stricter --test gate.
            if self.object_model is not None and self.object_model.particle_count:
                pq = self.object_state_0.particle_q.numpy()
                if not np.all(np.isfinite(pq)):
                    raise ValueError("Non-finite particle position (sim blew up).")
            bq = self.object_body_q()
            if bq.size and not np.all(np.isfinite(bq)):
                raise ValueError("Non-finite body transform (sim blew up).")
            if self.spec.check is not None:
                self.spec.check(self)

    return DataDrivenExample
