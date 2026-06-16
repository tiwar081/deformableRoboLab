"""``pickplace_ycb_franka`` rendered through robolab_viz (RoboLab DROID look).

Same physics as ``pickplace_ycb_franka`` (Franka picks a rubik's cube and a banana
and drops them into a box tray, force-limited gripper), rendered to
``outputs/robolab_preview/combined.mp4`` via the warp-raycast preview (forces
``--viewer null``).

Run: python -m examples pickplace_ycb_robolab --device cuda:0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import examples
import newton.utils
from examples.pickplace_ycb_franka import Example as PickPlacePhysicsExample
from robolab_viz import droid_scene_config
from robolab_viz.config import available_tables
from robolab_viz.raycast import RaycastPreviewRenderer
from robolab_viz.robot_fk import RobotVisualFK
from robolab_viz.robot_usd import ensure_robot_usd


# Physics table from pickplace_ycb_franka._build_robot_builder (box at (0.45,0,*)).
TABLE_CENTER_XY = (0.45, 0.0)


class Example(PickPlacePhysicsExample):
    def __init__(self, viewer, args):
        if args.viewer != "null":
            raise SystemExit("pickplace_ycb_robolab renders through robolab_viz; run with --viewer null.")
        super().__init__(viewer, args)

        self.robot_urdf_path = Path(newton.utils.download_asset("franka_emika_panda")) / "urdf" / "fr3_franka_hand.urdf"
        self.robot_base_pos = (0.0, 0.0, 0.0)  # this example mounts the FR3 at the origin
        robot_usd = ensure_robot_usd(self.robot_urdf_path, name="fr3_franka_hand")

        scene = droid_scene_config(
            table_top_z=self.table_top_z,
            table_center_xy=TABLE_CENTER_XY,
            table=args.table,
            background=args.background,
        )
        scene.robot_usd = robot_usd
        scene.fps = self.fps
        scene.sim_to_viz_translation = (0.0, 0.0, 0.0)
        self.scene_cfg = scene

        self.viz_fk = RobotVisualFK(
            self.robot_urdf_path,
            base_xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            device=self.robot_model.device,
            expected_joint_coords=self.robot_model.joint_coord_count,
        )
        self.preview = RaycastPreviewRenderer(
            scene=scene,
            object_model=self.object_model,
            robot_link_names=self.viz_fk.link_names,
            output_dir=Path(args.output_path).parent / "robolab_preview",
            device=self.robot_model.device,
            camera_names=list(args.preview_cameras),
            png_every=args.png_every,
        )
        self._viz_frames_written = 0
        self._viz_num_frames = args.num_frames
        self._viz_finalized = False
        self.preview_summary: dict = {}

    def render(self) -> None:
        super().render()
        link_tfs = self.viz_fk.link_world_transforms(self.robot_state_0.joint_q)
        object_body_q = self.object_state_0.body_q.numpy()
        self.preview.render_frame(link_tfs, object_body_q, self.object_state_0.particle_q)
        self._viz_frames_written += 1
        if self._viz_frames_written >= self._viz_num_frames:
            self._finalize_viz()

    def _finalize_viz(self) -> None:
        if self._viz_finalized:
            return
        self._viz_finalized = True
        self.preview_summary = self.preview.close()

    def test_final(self) -> None:
        super().test_final()
        self._finalize_viz()
        mp4 = self.preview.output_dir / "combined.mp4"
        if not mp4.exists() or mp4.stat().st_size < 1024:
            raise ValueError(f"Expected preview video {mp4} was not written.")

    @staticmethod
    def create_parser():
        parser = PickPlacePhysicsExample.create_parser()
        parser.set_defaults(output_path=str(Path("outputs") / "pickplace_ycb_robolab.usd"), viewer="null")
        parser.add_argument("--table", default="maple", choices=available_tables())
        parser.add_argument("--background", default="home_office")
        parser.add_argument("--preview-cameras", nargs="+", default=["over_shoulder_left_camera"])
        parser.add_argument("--png-every", type=int, default=60)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = examples.init(parser, example_name="pickplace_ycb_robolab")
    examples.run(Example(viewer, args), args)
