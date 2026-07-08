"""Visualize the VBD gripper + palm contact proxies overlaid on the Franka robot.

The proxies are invisible finite-mass bodies that live only in the VBD *object* model
(``deformableManipulationTools.grip.build_gripper_proxies``); at runtime each substep they are
re-pinned to a robot body (``TwoWayProxyCoupling.sync_proxies``):

  * two FINGER proxies  -> the left/right Franka finger bodies, carrying the finger's OWN collider
    copied one-for-one by default (the panda USD's CONVEX_MESH per finger, rendered here as the
    actual ghosted mesh + its tight bounds wireframe; the FR3 URDF's sparse boxes, rendered as the
    boxes), or optional contained box slices for the panda — the GRIP pads that are harvested into
    the grip signal;
  * one PALM/EE blocker -> the EE (link7) body (a synthetic box spanning wrist->hand), a blocker
    only (never harvested), stopping a swept cable passing through the gripper palm.

This script reconstructs exactly that: it builds the real Franka model, runs FK at the home pose,
asks ``build_gripper_proxies`` for the proxy geometry, pins each proxy to its mirror body, and
ghost-renders the boxes over the robot with a small self-contained Warp raycaster (headless, no
display / RT cores needed). Output PNGs land in ``outputs/visualizations/``.

Run:  python visualizations/franka_vbd_proxies.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

# Make the repo importable when run directly (visualizations/<this> -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUTPUT_DIR = REPO_ROOT / "outputs" / "visualizations"

import newton  # noqa: E402

from deformableManipulationTools.robot import build_franka_robot, finger_body_indices  # noqa: E402
from deformableManipulationTools.grip import build_gripper_proxies  # noqa: E402
from deformableManipulationTools.params import ROBOTS, RobotConfig, TABLE  # noqa: E402
from deformableManipulationTools.mathutils import find_body  # noqa: E402

# Proxy overlay colours (RGB 0..1): finger pads warm orange, palm/EE blocker cyan.
FINGER_COLOR = (0.95, 0.45, 0.10)
PALM_COLOR = (0.15, 0.70, 0.95)
ROBOT_LINK_COLOR = (0.82, 0.82, 0.85)
ROBOT_FINGER_COLOR = (0.28, 0.28, 0.31)
PROXY_ALPHA = 0.45          # fill opacity of the proxy boxes (tint; the outline guarantees the extent)
ROBOT_ALPHA = 0.55          # opacity of the robot itself (low: ghosted so proxies show through it fully)
OUTLINE_THICKNESS = 2        # proxy bounding-box wireframe thickness [px] (always drawn on top)
SSAA = 2                     # supersample factor (rendered at SSAA x, box-filtered down)

# 12 edges of a box, indexing the 8 corners as 4*ix+2*iy+iz (ix/iy/iz in {0,1}) — matches _box_tris.
_BOX_EDGES = ((0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7),
              (0, 1), (2, 3), (4, 5), (6, 7))


# ---------------------------------------------------------------------------------------------
# Small numpy transform helpers (pos + quat xyzw, the Newton/warp convention).
# ---------------------------------------------------------------------------------------------
def quat_rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def tf_compose(parent, local):
    """Compose two (pos, quat_xyzw) transforms: world = parent o local."""
    pp, pq = np.asarray(parent[:3], float), np.asarray(parent[3:7], float)
    lp, lq = np.asarray(local[:3], float), np.asarray(local[3:7], float)
    return np.concatenate([pp + quat_rotate(pq, lp), quat_mul(pq, lq)])


# ---------------------------------------------------------------------------------------------
# Geometry extraction: world-frame triangle soup for the robot, world-frame boxes for proxies.
# ---------------------------------------------------------------------------------------------
def _box_tris(center, half, quat, color):
    """8 world-frame corners + 12 triangles + per-face colour for an oriented box."""
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(np.asarray(center) + quat_rotate(quat, [sx * half[0], sy * half[1], sz * half[2]]))
    verts = np.asarray(corners, dtype=np.float32)
    # corner index = (sx,sy,sz) -> 4*ix+2*iy+iz, ix/iy/iz in {0,1}
    tris = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype=np.int32)
    face_color = np.tile(np.asarray(color, dtype=np.float32), (len(tris), 1))
    return verts, tris, face_color


def build_robot_meshes(model, body_q):
    """World-frame (verts, tris, face_color) for every VISIBLE robot shape at the given FK pose."""
    st = model.shape_type.numpy()
    sf = model.shape_flags.numpy()
    sb = model.shape_body.numpy()
    s_tf = model.shape_transform.numpy()
    s_scale = model.shape_scale.numpy()
    labels = list(model.shape_label)
    blabels = list(model.body_label)
    VIS = int(newton.ShapeFlags.VISIBLE)

    all_v, all_t, all_c = [], [], []
    offset = 0
    mesh_types = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    for s in range(model.shape_count):
        if not (int(sf[s]) & VIS):
            continue
        if int(st[s]) not in mesh_types:                # MESH (fr3 URDF) or CONVEX_MESH (panda USD)
            continue
        src = model.shape_source[s]
        if src is None:
            continue
        v_local = np.asarray(src.vertices, dtype=np.float64) * np.asarray(s_scale[s], dtype=np.float64)[None, :]
        tris = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
        body = int(sb[s])
        world_tf = tf_compose(body_q[body], s_tf[s]) if body >= 0 else s_tf[s]
        wp_pos, wp_q = world_tf[:3], world_tf[3:7]
        v_world = (quat_rotate(wp_q, v_local) + wp_pos).astype(np.float32)
        bname = blabels[body] if body >= 0 else ""
        color = ROBOT_FINGER_COLOR if "finger" in bname else ROBOT_LINK_COLOR
        all_v.append(v_world)
        all_t.append(tris + offset)
        all_c.append(np.tile(np.asarray(color, dtype=np.float32), (len(tris), 1)))
        offset += len(v_world)
    return (np.concatenate(all_v), np.concatenate(all_t).astype(np.int32), np.concatenate(all_c))


def current_proxy_geometry(robot_builder, robot_model, body_q, robot: RobotConfig, box_slice_proxy: bool = False):
    """World-frame proxy geometry for the CURRENT grip: ``(tri_geoms, outline_boxes)``.

    Builds the real proxies via ``build_gripper_proxies`` (the same call the framework makes), then
    pins each proxy body to its mirror robot body exactly as ``TwoWayProxyCoupling.sync_proxies``
    does at runtime: finger proxies -> finger bodies, palm proxy -> EE (link7).

    The default finger pad is the finger's own collider copied one-for-one — a CONVEX_MESH on the
    panda USD (rendered as the actual ghosted mesh, outlined by its tight bounds), sparse BOXes on the
    FR3 URDF (rendered as the boxes themselves). With ``box_slice_proxy=True``, mesh fingers use
    contained box slices instead. The palm/EE blocker is always a synthetic box.
    ``tri_geoms`` is a list of ``(verts_world, tris, color)``; ``outline_boxes`` of
    ``(center, half, quat, color)`` for the depth-independent wireframes."""
    finger_bodies = finger_body_indices(robot_model, robot=robot)
    ee_body = find_body(list(robot_model.body_label), robot.ee_link_suffix)

    ob = newton.ModelBuilder()
    proxy_bodies, proxy_shapes = build_gripper_proxies(
        ob, robot_builder, finger_bodies, object_table_shape=None, box_slice_proxy=box_slice_proxy)
    # proxy_bodies order: [left finger, right finger, palm/EE]; mirror to robot bodies accordingly.
    mirror = {proxy_bodies[0]: finger_bodies[0], proxy_bodies[1]: finger_bodies[1],
              proxy_bodies[2]: ee_body}
    palm_pb = proxy_bodies[2]
    mesh_types = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))

    tri_geoms, boxes = [], []
    for s in proxy_shapes:
        pb = ob.shape_body[s]
        local_tf = np.asarray(ob.shape_transform[s], dtype=np.float64)
        world_tf = tf_compose(body_q[mirror[pb]], local_tf)
        color = PALM_COLOR if pb == palm_pb else FINGER_COLOR
        src = ob.shape_source[s]
        scale = np.asarray(ob.shape_scale[s], dtype=np.float64)[:3]
        if int(ob.shape_type[s]) in mesh_types and src is not None:
            v_local = np.asarray(src.vertices, dtype=np.float64) * scale[None, :]
            tris = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
            v_world = quat_rotate(world_tf[3:7], v_local) + world_tf[:3]
            tri_geoms.append((v_world.astype(np.float32), tris, color))
            # tight local bounds of the pad mesh -> outlined oriented box
            lo, hi = v_local.min(axis=0), v_local.max(axis=0)
            c_local = 0.5 * (lo + hi)
            center = world_tf[:3] + quat_rotate(world_tf[3:7], c_local)
            boxes.append((center, 0.5 * (hi - lo), world_tf[3:7], color))
        else:
            v, t, c = _box_tris(world_tf[:3], scale, world_tf[3:7], color)
            tri_geoms.append((v, t, color))
            boxes.append((world_tf[:3], scale, world_tf[3:7], color))
    return tri_geoms, boxes


# ---------------------------------------------------------------------------------------------
# Self-contained Warp raycaster with a ghosted (alpha) proxy overlay layer.
# ---------------------------------------------------------------------------------------------
@wp.func
def _shade(n: wp.vec3, d: wp.vec3) -> float:
    key = wp.normalize(wp.vec3(0.35, -0.45, -0.82))
    return 0.40 + 0.45 * wp.max(wp.dot(n, -key), 0.0) + 0.15 * wp.max(wp.dot(n, -d), 0.0)


@wp.func
def _bg(d: wp.vec3) -> wp.vec3:
    g = 0.60 + 0.30 * wp.max(d[2], 0.0)
    return wp.vec3(g * 0.93, g * 0.95, g)


@wp.kernel
def _render_kernel(
    robot_mesh: wp.uint64,
    prox_mesh: wp.uint64,
    has_prox: int,
    cam_pos: wp.vec3,
    cam_quat: wp.quat,
    tan_w: float,
    tan_h: float,
    width: int,
    height: int,
    robot_fcolor: wp.array(dtype=wp.vec3),
    prox_fcolor: wp.array(dtype=wp.vec3),
    prox_alpha: float,
    robot_alpha: float,
    img: wp.array(dtype=wp.uint8, ndim=3),
):
    px, py = wp.tid()
    u = (float(px) + 0.5) / float(width) * 2.0 - 1.0
    v = 1.0 - (float(py) + 0.5) / float(height) * 2.0
    d_cam = wp.normalize(wp.vec3(u * tan_w, v * tan_h, -1.0))
    d = wp.quat_rotate(cam_quat, d_cam)
    bg = _bg(d)

    # nearest robot surface (translucent shell) ...
    col_r = wp.vec3(0.0, 0.0, 0.0)
    t_r = 1.0e6
    qr = wp.mesh_query_ray(robot_mesh, cam_pos, d, 100.0)
    has_r = qr.result
    if has_r:
        n = qr.normal
        if wp.dot(n, d) > 0.0:
            n = -n
        col_r = robot_fcolor[qr.face] * _shade(n, d)
        t_r = qr.t

    # ... and nearest proxy surface.
    col_p = wp.vec3(0.0, 0.0, 0.0)
    t_p = 1.0e6
    has_p = False
    if has_prox == 1:
        qp = wp.mesh_query_ray(prox_mesh, cam_pos, d, 100.0)
        if qp.result:
            has_p = True
            n = qp.normal
            if wp.dot(n, d) > 0.0:
                n = -n
            col_p = prox_fcolor[qp.face] * _shade(n, d)
            t_p = qp.t

    # Back-to-front alpha composite of the (up to two) surfaces over the background, so a proxy box
    # BEHIND a robot surface still shows through the ghosted robot (robot_alpha < 1).
    c = bg
    if has_r and has_p:
        if t_r >= t_p:                       # robot farther -> robot first, then proxy
            c = col_r * robot_alpha + bg * (1.0 - robot_alpha)
            c = col_p * prox_alpha + c * (1.0 - prox_alpha)
        else:                                # proxy farther -> proxy first, then robot
            c = col_p * prox_alpha + bg * (1.0 - prox_alpha)
            c = col_r * robot_alpha + c * (1.0 - robot_alpha)
    elif has_r:
        c = col_r * robot_alpha + bg * (1.0 - robot_alpha)
    elif has_p:
        c = col_p * prox_alpha + bg * (1.0 - prox_alpha)

    img[py, px, 0] = wp.uint8(wp.clamp(c[0], 0.0, 1.0) * 255.0)
    img[py, px, 1] = wp.uint8(wp.clamp(c[1], 0.0, 1.0) * 255.0)
    img[py, px, 2] = wp.uint8(wp.clamp(c[2], 0.0, 1.0) * 255.0)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (xyzw) for a camera at ``eye`` looking at ``target`` (camera looks down -z, +y up)."""
    eye, target, up = map(lambda a: np.asarray(a, dtype=np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, f)
    R = np.column_stack([right, true_up, -f])  # camera x,y,z axes in world
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class Raycaster:
    def __init__(self, robot, proxy_geoms, proxy_boxes, device):
        self.device = device
        self._boxes = list(proxy_boxes)
        rv, rt, rc = robot
        with wp.ScopedDevice(device):
            self._robot_pts = wp.array(rv.astype(np.float32), dtype=wp.vec3)
            self._robot_tris = wp.array(rt.flatten().astype(np.int32), dtype=wp.int32)
            self._robot_mesh = wp.Mesh(points=self._robot_pts, indices=self._robot_tris)
            self._robot_fcolor = wp.array(rc.astype(np.float32), dtype=wp.vec3)

            if proxy_geoms:
                pv, pt, pc, off = [], [], [], 0
                for verts, tris, color in proxy_geoms:
                    pv.append(np.asarray(verts, dtype=np.float32))
                    pt.append(np.asarray(tris, dtype=np.int32) + off)
                    pc.append(np.tile(np.asarray(color, dtype=np.float32), (len(tris), 1)))
                    off += len(verts)
                self._prox_pts = wp.array(np.concatenate(pv).astype(np.float32), dtype=wp.vec3)
                self._prox_tris = wp.array(np.concatenate(pt).flatten().astype(np.int32), dtype=wp.int32)
                self._prox_mesh = wp.Mesh(points=self._prox_pts, indices=self._prox_tris)
                self._prox_fcolor = wp.array(np.concatenate(pc).astype(np.float32), dtype=wp.vec3)
                self._has_prox = 1
            else:
                self._prox_mesh = self._robot_mesh
                self._prox_fcolor = self._robot_fcolor
                self._has_prox = 0

    def render(self, eye, target, out_path, width=1400, height=1050, fov_deg=38.0):
        W, H = width * SSAA, height * SSAA
        cam_quat = look_at_quat(eye, target)
        tan_h = np.tan(np.radians(fov_deg) * 0.5)
        tan_w = tan_h * (W / H)
        with wp.ScopedDevice(self.device):
            img = wp.zeros((H, W, 3), dtype=wp.uint8)
            wp.launch(_render_kernel, dim=(W, H), inputs=[
                self._robot_mesh.id, self._prox_mesh.id, self._has_prox,
                wp.vec3(*np.asarray(eye, np.float32)), wp.quat(*cam_quat.astype(np.float32)),
                float(tan_w), float(tan_h), W, H,
                self._robot_fcolor, self._prox_fcolor, float(PROXY_ALPHA), float(ROBOT_ALPHA),
            ], outputs=[img])
        rgb = img.numpy()
        # box-filter supersample down to the requested resolution
        rgb = rgb.reshape(height, SSAA, width, SSAA, 3).mean(axis=(1, 3)).astype(np.uint8)
        # crisp proxy bounding-box wireframe ON TOP (always visible, depth-independent)
        _draw_box_outlines(rgb, self._boxes, eye, cam_quat, tan_w, tan_h, width, height)
        _save_png(rgb, out_path)
        print(f"[franka_vbd_proxies] wrote {out_path}")


def _project(points_world, eye, cam_quat, tan_w, tan_h, width, height):
    """Project world points to pixel coords using the same pinhole model as _render_kernel.
    Returns a list of (px, py) or None (point behind the camera)."""
    conj = np.array([-cam_quat[0], -cam_quat[1], -cam_quat[2], cam_quat[3]])  # inverse of the unit quat
    eye = np.asarray(eye, dtype=np.float64)
    out = []
    for P in points_world:
        pc = quat_rotate(conj, np.asarray(P, dtype=np.float64) - eye)         # world -> camera frame
        if pc[2] >= -1e-6:                                                    # at/behind the image plane
            out.append(None)
            continue
        u = -pc[0] / (pc[2] * tan_w)
        v = -pc[1] / (pc[2] * tan_h)
        out.append(((u + 1.0) * 0.5 * width, (1.0 - v) * 0.5 * height))
    return out


def _draw_box_outlines(rgb, boxes, eye, cam_quat, tan_w, tan_h, width, height):
    """Draw each proxy box's 12-edge wireframe over the rendered image (depth-independent, so the
    full bounding box is visible no matter what occludes it)."""
    try:
        import cv2
    except Exception:
        return
    for center, half, quat, color in boxes:
        corners = [np.asarray(center) + quat_rotate(quat, [sx * half[0], sy * half[1], sz * half[2]])
                   for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        pts = _project(corners, eye, cam_quat, tan_w, tan_h, width, height)
        col = tuple(int(round(c * 255)) for c in color)                       # RGB (rgb array is RGB)
        for a, b in _BOX_EDGES:
            if pts[a] is None or pts[b] is None:
                continue
            p0 = (int(round(pts[a][0])), int(round(pts[a][1])))
            p1 = (int(round(pts[b][0])), int(round(pts[b][1])))
            cv2.line(rgb, p0, p1, col, OUTLINE_THICKNESS, cv2.LINE_AA)


def _save_png(rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception:
        from PIL import Image
        Image.fromarray(rgb).save(str(path))


# ---------------------------------------------------------------------------------------------
def render_robot(gripper_name: str, robot_cfg: RobotConfig, box_slice_proxy: bool, views: list[str], device):
    print(f"[franka_vbd_proxies] rendering {gripper_name}")

    robot_xform = wp.transform((-0.45, -0.45, TABLE.top_z), wp.quat_identity())
    robot_builder = build_franka_robot(xform=robot_xform, table=TABLE, robot=robot_cfg)
    model = robot_builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()

    robot = build_robot_meshes(model, body_q)
    geoms, boxes = current_proxy_geometry(
        robot_builder, model, body_q, robot=robot_cfg, box_slice_proxy=box_slice_proxy)

    # Frame the gripper: target = midpoint of the two finger bodies.
    fb = finger_body_indices(model, robot=robot_cfg)
    grip_center = 0.5 * (body_q[fb[0]][:3] + body_q[fb[1]][:3])
    base = body_q[0][:3]
    robot_mid = 0.5 * (base + grip_center) + np.array([0.0, 0.0, 0.10])

    rc = Raycaster(robot, geoms, boxes, device)
    if "full" in views:
        rc.render(eye=robot_mid + np.array([0.85, -0.75, 0.45]), target=robot_mid,
                  out_path=OUTPUT_DIR / f"{gripper_name}_full.png", fov_deg=40.0)
    if "gripper" in views:
        # Side close-up: shows the finger proxies and the palm/EE blocker depth behind them.
        rc.render(eye=grip_center + np.array([0.34, 0.02, 0.06]), target=grip_center,
                  out_path=OUTPUT_DIR / f"{gripper_name}_gripper.png", fov_deg=34.0)


GRIPPERS = {
    "fr3": ("fr3_franka_hand", False),
    "panda_full_mesh": ("franka_panda_isaacsim", False),
    "panda_box_slices": ("franka_panda_isaacsim", True),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Render the whole-robot view.")
    parser.add_argument("--gripper", action="store_true", help="Render the side gripper close-up view.")
    parser.add_argument("--fr3", action="store_true", help="Render the FR3 sparse-box finger proxies.")
    parser.add_argument("--panda_full_mesh", action="store_true", help="Render Panda full-mesh finger proxies.")
    parser.add_argument("--panda_box_slices", action="store_true", help="Render Panda box-slice finger proxies.")
    parser.add_argument("--all", action="store_true", help="Render all views for all gripper proxy types.")
    return parser.parse_args()


def main():
    args = parse_args()
    wp.init()
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    views = []
    if args.all or args.full:
        views.append("full")
    if args.all or args.gripper:
        views.append("gripper")
    if not views:
        views = ["full", "gripper"]

    selected = []
    for name in GRIPPERS:
        if args.all or getattr(args, name):
            selected.append(name)
    if not selected:
        selected = list(GRIPPERS)

    for name in selected:
        robot_key, box_slice_proxy = GRIPPERS[name]
        render_robot(name, ROBOTS[robot_key], box_slice_proxy, views, device)


if __name__ == "__main__":
    main()
