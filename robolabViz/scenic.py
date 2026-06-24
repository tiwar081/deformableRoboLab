"""Scenic-rendering base that bolts robolabViz onto a Newton grasp demo (glue).

Every demo's ``Example`` subclasses :class:`ScenicGraspExample` instead of
:class:`~deformableManipulationTools.GraspExample` and implements the usual
``configure`` / ``plan`` / ``build_scene`` / ``set_robot_targets`` plus
``check_physics`` (the per-demo finite/grasp assertions — formerly the body of
``test_final``). The ``--output-style`` flag then selects how the run is rendered:

- ``basic``  — the Newton USD viewer writes ``outputs/<name>.usd`` (no scene look,
  the original demo behaviour). Selected when ``args.output_style == "basic"``.
- ``scenic`` (default) — robolabViz renders ``outputs/<name>/`` with a
  ``frames/`` folder (a per-camera still every ``--frames-per-image`` frames) and
  ``simulation.mp4`` containing BOTH the over-shoulder-left and the gripper wrist
  camera side by side. Produced on any CUDA GPU via the warp-raycast preview.

All the RoboLab wiring (DROID scene placed from the physics table, the FK mirror,
the raycast preview, optional ``--usd`` / ``--npz`` outputs, and the viz/sim
parity + table-footprint + wrist-occlusion checks) lives here so every demo —
and every future demo — inherits it identically and never repeats it. The robot
base pose, table placement, and (optional) soft-object position are read from the
physics example, so this base needs no per-demo configuration.

This module is the one place robolabViz depends on the physics package
(``deformableManipulationTools.GraspExample``); the import is kept here, not in
``robolabViz/__init__.py``, so ``import robolabViz`` stays free of Newton/Warp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import newton.utils

from deformableManipulationTools import GraspExample

from .config import CameraConfig, droid_scene_config, fixture_world_bbox, look_at_quat_wxyz
from .raycast import RaycastPreviewRenderer
from .robot_fk import RobotVisualFK
from .robot_usd import ensure_robot_usd
from .stage import RoboLabStageWriter

# The two policy cameras RoboLab records and the scenic simulation.mp4 shows.
SCENIC_CAMERAS = ["over_shoulder_left_camera", "wrist_camera"]


class ScenicGraspExample(GraspExample):
    """``GraspExample`` + optional RoboLab scenic rendering, driven by
    ``--output-style``. Subclasses implement ``check_physics`` for their asserts."""

    # Assert the visible work table covers the whole physics contact footprint and
    # its top aligns. True for the corner-mounted-robot demos (where the visual and
    # contact tables are meant to coincide). Set False when the physics table is an
    # intentionally oversized contact slab that the DROID table need not span.
    scenic_check_table: bool = True

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self._scenic = getattr(args, "output_style", "scenic") == "scenic"
        if self._scenic:
            self._setup_scenic(args)

    # ---- scenic build (only when --output-style scenic) ----
    def _setup_scenic(self, args) -> None:
        # The robot import mirrors the simulated robot: same URDF asset, same base
        # pose (read from the physics example's robot_base_xform, so origin-mounted
        # demos like pickplace_ycb and table-corner demos both work unchanged). The
        # FK/sim parity assertion in the checks guards this against drift.
        self.robot_urdf_path = (
            Path(newton.utils.download_asset("franka_emika_panda")) / "urdf" / "fr3_franka_hand.urdf"
        )
        self.robot_base_pos = tuple(float(x) for x in self.robot_base_xform.p)
        robot_usd = ensure_robot_usd(self.robot_urdf_path, name="fr3_franka_hand")

        # RoboLab's DROID scene, with the visual work-table placed from THIS demo's
        # physics table so the rendered surface coincides with the invisible contact
        # table (same top height, same footprint).
        sim_to_viz = tuple(-c for c in self.robot_base_pos)
        table_top_viz = self.table_top_z + sim_to_viz[2]
        table_center_viz = (
            float(self.table_pos[0] + sim_to_viz[0]),
            float(self.table_pos[1] + sim_to_viz[1]),
        )
        scene = droid_scene_config(
            table_top_z=table_top_viz,
            table_center_xy=table_center_viz,
            table=args.table,
            background=args.background,
        )
        scene.robot_usd = robot_usd
        scene.fps = self.fps
        scene.sim_to_viz_translation = sim_to_viz
        if args.wrist_eye is not None:
            scene.wrist_camera.eye = tuple(args.wrist_eye)
        if args.wrist_target is not None:
            scene.wrist_camera.target = tuple(args.wrist_target)

        # Optional fixed "object view" camera: a still, elevated 3/4 view framed on
        # the soft object, posed from its viz-frame position so it tracks the
        # layout. Rendered + dumped as PNG frames only (in_combined_video=False
        # keeps it out of simulation.mp4). Only available when the demo exposes a
        # soft_start_pos to aim at.
        preview_cameras = list(SCENIC_CAMERAS)
        if args.objectview:
            if not hasattr(self, "soft_start_pos"):
                print("[robolabViz] --objectview ignored: this demo has no soft_start_pos to frame.")
            else:
                soft_viz = np.asarray(self.soft_start_pos, dtype=float) + np.asarray(sim_to_viz, dtype=float)
                target = (float(soft_viz[0]), float(soft_viz[1]), float(soft_viz[2]) + 0.025)
                eye = (target[0] + 0.30, target[1] - 0.30, target[2] + 0.35)
                scene.exterior_cameras.append(
                    CameraConfig(
                        name="object_view_camera",
                        position=eye,
                        orientation_wxyz=look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)),
                        focal_length=6.0,  # narrower FOV: tight on the object + surroundings
                        in_combined_video=False,
                    )
                )
                preview_cameras.append("object_view_camera")

        self.scene_cfg = scene
        self.viz_fk = RobotVisualFK(
            self.robot_urdf_path,
            base_xform=self.robot_base_xform,
            device=self.robot_model.device,
            expected_joint_coords=self.robot_model.joint_coord_count,
        )
        # The output directory examples.init resolved for scenic runs (outputs/<name>/).
        self._scenic_dir = Path(getattr(args, "output_dir", None) or Path(args.output_path).parent)

        # Rigid-only routing: there is no VBD object model — the objects are merged into the robot's
        # MuJoCo model (from object_body_start on). Render them from there (posed by the robot state),
        # skipping the robot's own links (those render from the URDF/USD via FK).
        self._rigid_only = self.object_model is None
        viz_obj_model = self.robot_model if self._rigid_only else self.object_model
        object_body_min = self.object_body_start if self._rigid_only else None

        # The full time-sampled USD scene is an opt-in output (--usd).
        self.stage_writer: RoboLabStageWriter | None = None
        if args.usd:
            self.stage_writer = RoboLabStageWriter(
                output_path=args.output_path,
                scene=scene,
                object_model=viz_obj_model,
                num_frames=args.num_frames,
            )
        self.preview = RaycastPreviewRenderer(
            scene=scene,
            object_model=viz_obj_model,
            robot_link_names=self.viz_fk.link_names,
            output_dir=self._scenic_dir,
            device=self.robot_model.device,
            camera_names=preview_cameras,
            frames_per_image=args.frames_per_image,
            object_body_min=object_body_min,
        )
        # geometry.pkl is only useful alongside the state cache (rerender needs both).
        if args.npz:
            self.preview.export_instances(self._scenic_dir / "geometry.pkl")

        self._viz_frames_written = 0
        self._viz_num_frames = args.num_frames
        self._viz_finalized = False
        self.preview_summary: dict = {}
        # Per-frame state cache (opt-in, --npz): lets robolabViz.rerender re-render
        # any camera (e.g. while tuning the wrist mount) without re-simulating.
        self._state_cache: dict[str, list] = {"link_tfs": [], "body_q": [], "particle_q": []}

    # ---- per-frame loop ----
    def render(self) -> None:
        super().render()
        if not self._scenic:
            return

        link_tfs = self.viz_fk.link_world_transforms(self.robot_state_0.joint_q)
        # Object body poses: the VBD object state, or the robot state when objects are merged into the
        # MuJoCo model (rigid-only). The instance keys are body indices into this same body_q.
        obj_state = self.robot_state_0 if self._rigid_only else self.object_state_0
        object_body_q = obj_state.body_q.numpy()
        # particle_q is None for rigid-only scenes (no deformables).
        particle_q = None if self._rigid_only else self.object_state_0.particle_q
        particle_q_np = particle_q.numpy() if particle_q is not None else np.zeros((0, 3), dtype=np.float32)

        if self.stage_writer is not None:
            self.stage_writer.write_frame(self._viz_frames_written, link_tfs, object_body_q, particle_q_np)
        self.preview.render_frame(link_tfs, object_body_q, particle_q)
        if self.args.npz:
            self._state_cache["link_tfs"].append(np.stack([link_tfs[n] for n in self.viz_fk.link_names]))
            self._state_cache["body_q"].append(object_body_q.copy())
            self._state_cache["particle_q"].append(particle_q_np.copy())

        self._viz_frames_written += 1
        if self._viz_frames_written >= self._viz_num_frames:
            self._finalize_viz()

    def _finalize_viz(self) -> None:
        if self._viz_finalized:
            return
        self._viz_finalized = True
        if self.stage_writer is not None:
            self.stage_writer.save()
        if self.args.npz and self._state_cache["link_tfs"]:
            cache_path = self._scenic_dir / f"{Path(self.args.output_path).stem}.state.npz"
            np.savez_compressed(
                cache_path,
                link_names=np.array(self.viz_fk.link_names),
                link_tfs=np.stack(self._state_cache["link_tfs"]),
                body_q=np.stack(self._state_cache["body_q"]),
                particle_q=np.stack(self._state_cache["particle_q"]),
                fps=self.fps,
            )
            print(f"[robolabViz] State cache saved to {cache_path}")
        self.preview_summary = self.preview.close()
        if self.preview_summary:
            cov = self.preview_summary["wrist_mean_coverage"]
            print(
                "[robolabViz] wrist camera mean coverage: "
                + ", ".join(f"{k}={v:.3f}" for k, v in cov.items())
            )

    # ---- verification ----
    def check_physics(self) -> None:
        """Per-demo finite/grasp assertions. Subclasses override (was test_final)."""

    def test_final(self) -> None:
        self.check_physics()
        if not self._scenic:
            return
        self._finalize_viz()

        # Viz/sim parity: the FK mirror must agree with the simulated robot on the
        # bodies they share (the collapsed model's fingers/links).
        link_tfs = self.viz_fk.link_world_transforms(self.robot_state_0.joint_q)
        sim_body_q = self.robot_state_0.body_q.numpy()
        sim_labels = list(self.robot_model.body_label)
        checked = 0
        for name, tf in link_tfs.items():
            for i, label in enumerate(sim_labels):
                if label.endswith(name):
                    err = float(np.linalg.norm(sim_body_q[i][:3] - tf[:3]))
                    if err > 1.0e-4:
                        raise ValueError(f"FK mirror disagrees with simulated body {name!r} by {err:.2e} m.")
                    checked += 1
        if checked < 5:
            raise ValueError(f"FK parity check matched only {checked} links; label mapping is broken.")

        if self._viz_frames_written < min(self._viz_num_frames, 1):
            raise ValueError("No viz frames were written.")

        # Contact/visual consistency: the visible work table must cover the whole
        # physics contact footprint and its top surface must coincide with the
        # physics table top (where objects actually rest). Skipped when the demo's
        # physics table is an oversized contact slab (scenic_check_table=False).
        if self.scenic_check_table:
            work_table = next((f for f in self.scene_cfg.fixtures if f.name == "work_table"), None)
            if work_table is None:
                raise ValueError("work_table fixture missing from the scene config.")
            tmn, tmx = fixture_world_bbox(work_table)
            off = np.asarray(self.scene_cfg.sim_to_viz_translation)
            half = np.array([float(self.table_half[i]) for i in range(3)])
            center = np.array([float(self.table_pos[i]) for i in range(3)]) + off
            pmn, pmx = center - half, center + half
            if not (tmn[0] <= pmn[0] + 1e-3 and tmx[0] >= pmx[0] - 1e-3
                    and tmn[1] <= pmn[1] + 1e-3 and tmx[1] >= pmx[1] - 1e-3):
                raise ValueError(
                    f"Visual table x/y {[round(v,3) for v in tmn[:2]]}..{[round(v,3) for v in tmx[:2]]} "
                    f"does not cover the physics contact footprint "
                    f"{[round(v,3) for v in pmn[:2]]}..{[round(v,3) for v in pmx[:2]]}."
                )
            if abs(tmx[2] - pmx[2]) > 5e-3:
                raise ValueError(
                    f"Visual table top z={tmx[2]:.3f} not aligned with physics contact surface z={pmx[2]:.3f}."
                )

        # The wrist camera must not be buried inside the gripper: in a sane mount
        # the robot occupies a minority of the frame on average.
        if self.preview_summary:
            robot_cov = self.preview_summary["wrist_mean_coverage"]["robot"]
            if robot_cov > 0.55:
                raise ValueError(
                    f"Wrist camera sees {robot_cov:.0%} robot pixels on average — view is blocked by the gripper."
                )

        # simulation.mp4 is the primary scenic output: it must exist and be non-empty.
        mp4 = self.preview.output_dir / "simulation.mp4"
        if not mp4.exists() or mp4.stat().st_size < 1024:
            raise ValueError(f"Expected scenic video {mp4} was not written.")
