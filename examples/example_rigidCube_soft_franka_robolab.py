"""``rigidCube_soft_franka`` with RoboLab's graphics instead of the Newton viewer.

The physics is *identical* to ``examples.rigidCube_soft_franka`` — this module
subclasses its ``Example`` and only swaps the visualization layer for the
RoboLab (NVIDIA DROID rig) look reproduced by ``robolab_viz``. Unlike the cable
demo, the Franka here grasps a heavy steel-density cube (~1 kg), carries it over
a pillow-soft FEM block (``k_mu=5e2, k_lambda=2.5e3``), and drops it half-offset
onto the block so the cube visibly squashes the block edge and rolls off — the
deformation reads clearly even in the flat-shaded preview.

- the scene from RoboLab's ``run_recorded.py``: home-office dome background,
  sphere key light, maple work table, Franka pedestal, robot base at the
  viz-frame origin;
- both of run_recorded's policy camera perspectives: the over-shoulder-left
  exterior camera and the gripper-mounted wrist camera (plus the egocentric
  viewport camera prim for video);
- the robot rendered from an Isaac-converted USD of the *same* URDF that is
  simulated, puppeteered per frame from the solver state.

Outputs per run (in ``outputs/robolab_preview/``):

- ``combined.mp4`` — the warp-raycast preview video (the configured cameras
  side-by-side, flat-shaded but geometrically exact), produced on any CUDA GPU.
  This is the only video kept.
- ``frames/<camera>_NNNN.png`` — full-resolution per-camera still frames.
- ``wrist_coverage.json`` — wrist-camera occlusion metrics.

Opt-in outputs (off by default):

- ``--usd`` writes ``outputs/rigidCube_soft_franka_robolab.usd`` — the full
  RoboLab-styled scene with one time sample per frame. Open it in
  usdview/Omniverse, or RTX-render with ``python -m robolab_viz.isaac_render``
  on an RTX-capable GPU (H100/H200 have no RT cores and cannot RTX-render).
- ``--npz`` writes the per-frame state cache (+ ``robolab_preview/geometry.pkl``)
  that ``python -m robolab_viz.rerender`` replays to re-render cameras (e.g.
  while tuning the wrist mount) without re-simulating.

Run:

    python -m examples rigidCube_soft_franka_robolab --viewer null --device cuda:0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import examples
import newton.utils
from examples.rigidCube_soft_franka import Example as RigidCubeSoftPhysicsExample
from robolab_viz import droid_scene_config
from robolab_viz.config import CameraConfig, available_tables, fixture_world_bbox, look_at_quat_wxyz
from robolab_viz.raycast import RaycastPreviewRenderer
from robolab_viz.robot_fk import RobotVisualFK
from robolab_viz.robot_usd import ensure_robot_usd
from robolab_viz.stage import RoboLabStageWriter


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t", "on")


class Example(RigidCubeSoftPhysicsExample):
    def __init__(self, viewer, args):
        if args.viewer != "null":
            raise SystemExit(
                "rigidCube_soft_franka_robolab renders through robolab_viz (USD + raycast preview); "
                "run it with --viewer null."
            )
        super().__init__(viewer, args)

        # Mirror the base example's robot import (same URDF asset, same base
        # pose as its _build_robot_builder). The FK/sim parity assertion in
        # test_final() guards these against drifting from the base example.
        self.robot_urdf_path = (
            Path(newton.utils.download_asset("franka_emika_panda")) / "urdf" / "fr3_franka_hand.urdf"
        )
        self.robot_base_pos = (-0.45, -0.45, self.table_top_z)

        robot_usd = ensure_robot_usd(self.robot_urdf_path, name="fr3_franka_hand")

        # RoboLab's DROID scene, with the visual work-table placed from THIS
        # demo's physics table so the rendered surface coincides with the
        # invisible contact table (same top height, same footprint). Customize
        # here: swap fixtures/lights/cameras, restyle objects, etc.
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

        # Optional fixed "object view" camera: a still, elevated 3/4 view framed
        # on the soft body and a bit of surrounding table, posed from the block's
        # viz-frame position so it tracks the physics layout. It is rendered and
        # dumped as PNG frames only (in_combined_video=False keeps it out of
        # combined.mp4, so enabling --objectview does not change that video).
        preview_cameras = list(args.preview_cameras)
        if args.objectview:
            soft_viz = np.asarray(self.soft_start_pos, dtype=float) + np.asarray(sim_to_viz, dtype=float)
            target = (float(soft_viz[0]), float(soft_viz[1]), float(soft_viz[2]) + 0.025)
            eye = (target[0] + 0.30, target[1] - 0.30, target[2] + 0.35)
            scene.exterior_cameras.append(
                CameraConfig(
                    name="object_view_camera",
                    position=eye,
                    orientation_wxyz=look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)),
                    focal_length=6.0,  # narrower FOV than the exterior cams: tight on the block + surroundings
                    in_combined_video=False,
                )
            )
            preview_cameras.append("object_view_camera")

        self.scene_cfg = scene
        self.viz_fk = RobotVisualFK(
            self.robot_urdf_path,
            base_xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            device=self.robot_model.device,
            expected_joint_coords=self.robot_model.joint_coord_count,
        )
        # The USD scene is an opt-in output (``--usd``); when off, skip building
        # and writing it entirely.
        self.stage_writer: RoboLabStageWriter | None = None
        if args.usd:
            self.stage_writer = RoboLabStageWriter(
                output_path=args.output_path,
                scene=scene,
                object_model=self.object_model,
                num_frames=args.num_frames,
            )
        self.preview: RaycastPreviewRenderer | None = None
        if args.preview:
            self.preview = RaycastPreviewRenderer(
                scene=scene,
                object_model=self.object_model,
                robot_link_names=self.viz_fk.link_names,
                output_dir=Path(args.output_path).parent / "robolab_preview",
                device=self.robot_model.device,
                camera_names=preview_cameras,
                png_every=args.png_every,
            )
            # geometry.pkl is only useful with the state cache (rerender needs both).
            if args.npz:
                self.preview.export_instances(self.preview.output_dir / "geometry.pkl")
        self._viz_frames_written = 0
        self._viz_num_frames = args.num_frames
        self._viz_finalized = False
        self.preview_summary: dict = {}
        # Per-frame state cache (opt-in, ``--npz``): lets robolab_viz.rerender
        # re-render any camera (e.g. while tuning the wrist mount) without
        # re-simulating.
        self._state_cache: dict[str, list] = {"link_tfs": [], "body_q": [], "particle_q": []}

    def render(self) -> None:
        super().render()

        link_tfs = self.viz_fk.link_world_transforms(self.robot_state_0.joint_q)
        object_body_q = self.object_state_0.body_q.numpy()
        particle_q_np = self.object_state_0.particle_q.numpy()

        if self.stage_writer is not None:
            self.stage_writer.write_frame(self._viz_frames_written, link_tfs, object_body_q, particle_q_np)
        if self.preview is not None:
            self.preview.render_frame(link_tfs, object_body_q, self.object_state_0.particle_q)
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
            cache_path = Path(self.args.output_path).with_suffix(".state.npz")
            np.savez_compressed(
                cache_path,
                link_names=np.array(self.viz_fk.link_names),
                link_tfs=np.stack(self._state_cache["link_tfs"]),
                body_q=np.stack(self._state_cache["body_q"]),
                particle_q=np.stack(self._state_cache["particle_q"]),
                fps=self.fps,
            )
            print(f"[robolab_viz] State cache saved to {cache_path}")
        if self.preview is not None:
            self.preview_summary = self.preview.close()
            if self.preview_summary:
                cov = self.preview_summary["wrist_mean_coverage"]
                print(
                    "[robolab_viz] wrist camera mean coverage: "
                    + ", ".join(f"{k}={v:.3f}" for k, v in cov.items())
                )

    def test_final(self) -> None:
        super().test_final()
        self._finalize_viz()

        # Viz/sim parity: the FK mirror must agree with the simulated robot on
        # the bodies they share (the collapsed model's fingers/links).
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

        # Contact/visual consistency: the visible work table must cover the
        # whole physics contact footprint and its top surface must coincide
        # with the physics table top (where objects actually rest). Computed
        # from the fixture config so it holds regardless of --usd.
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

        # The wrist camera must not be buried inside the gripper: in a sane
        # mount the robot occupies a minority of the frame on average.
        if self.preview_summary:
            robot_cov = self.preview_summary["wrist_mean_coverage"]["robot"]
            if robot_cov > 0.55:
                raise ValueError(
                    f"Wrist camera sees {robot_cov:.0%} robot pixels on average — view is blocked by the gripper."
                )

        # combined.mp4 is the primary output: it must exist and be non-empty.
        if self.preview is not None:
            mp4 = self.preview.output_dir / "combined.mp4"
            if not mp4.exists() or mp4.stat().st_size < 1024:
                raise ValueError(f"Expected preview video {mp4} was not written.")

    @staticmethod
    def create_parser():
        parser = RigidCubeSoftPhysicsExample.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "rigidCube_soft_franka_robolab.usd"), viewer="null")
        parser.add_argument("--table", default="maple", choices=available_tables(),
                            help="Vendored work-table texture (default: maple).")
        parser.add_argument("--background", default="home_office",
                            help="Vendored dome background by name (e.g. home_office, garage_2k, "
                                 "machine_shop_01_2k; see robolab_viz.config.available_backgrounds).")
        parser.add_argument("--usd", type=_str2bool, nargs="?", const=True, default=False,
                            help="Write the full time-sampled USD scene (default: off).")
        parser.add_argument("--npz", type=_str2bool, nargs="?", const=True, default=False,
                            help="Write the per-frame state cache + geometry.pkl for rerender (default: off).")
        parser.add_argument("--preview", action="store_true", default=True,
                            help="Render the warp-raycast preview (combined.mp4 + PNG frames).")
        parser.add_argument("--no-preview", dest="preview", action="store_false")
        parser.add_argument("--preview-cameras", nargs="+",
                            default=["over_shoulder_left_camera", "wrist_camera"],
                            help="Camera names to preview-render (see robolab_viz.config).")
        parser.add_argument("--objectview", type=_str2bool, nargs="?", const=True, default=False,
                            help="Add a fixed object-inspection camera framed on the soft body; its PNG "
                                 "frames are saved to frames/ but it is kept out of combined.mp4 (default: off).")
        parser.add_argument("--png-every", type=int, default=30,
                            help="Dump a per-camera PNG frame every N frames into frames/ (0 = off).")
        parser.add_argument("--wrist-eye", type=float, nargs=3, default=None,
                            help="Override wrist camera eye in the hand frame (x y z).")
        parser.add_argument("--wrist-target", type=float, nargs=3, default=None,
                            help="Override wrist camera look-at target in the hand frame (x y z).")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="rigidCube_soft_franka_robolab")
    examples.run(Example(viewer, args), args)
