"""Multi-view rendering of one asset's rest geometry, and back-projection of a picked pixel.

Both halves share ONE camera model, which is the raycast renderer's own (``robolabViz.raycast``
``_render_kernel``: -Z forward, +Y up, ``tan_half = aperture / (2 * focal)``). That is the point of
keeping them in one file — a pixel the VLM picks in a rendered image is turned back into a canonical
3D point by re-casting the very ray the renderer shaded, so an error in the convention shows up as a
missed mesh rather than as a plausible-but-wrong region.

**The camera does not orbit; the object turns.** One fixed camera sits on the direction the flat
render kernel's key light comes from, and each of the six views rotates the object to present one
canonical face to it. This is not a stylistic choice — it is what makes CAVITIES legible, and a
cavity is exactly what distinguishes a rim from a closed base:

    the renderer's two lights are fixed WORLD directions (deliberately, so the wrist camera has no
    headlamp). An orbiting camera therefore looks into an unlit interior half the time. Rendered
    that way the YCB mug's open end is a flat dark disc indistinguishable from its base, and a VLM
    shown those six views labelled the SOLID BASE as the rim — a confidently wrong annotation, in
    the frame that matters. Presenting each face toward the light instead shows the mug's interior
    walls plainly (measured: open end reads as concentric rings, base end as a flat disc).

Six views, one per canonical axis direction, at a distance derived from the bounding sphere so every
asset is framed alike. Nothing here depends on the clock, the catalog order, or an RNG, so the same
mesh always produces the same images and the same rays.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Camera intrinsics for the annotation views. A narrower FOV than the scene cameras (~44 deg) keeps
# perspective distortion low, so a pixel the VLM picks near the silhouette still back-projects onto
# the surface it meant rather than onto a foreshortened neighbour.
IMAGE_SIZE = 640
APERTURE = 3.024
FOCAL_LENGTH = 3.8
# Bounding-sphere multiple the camera sits at, on top of the distance that exactly fills the frame.
# 1.18 leaves a visible margin of background on all sides — a region touching the image border is
# ambiguous to a VLM (it cannot tell whether the feature continues out of frame).
FRAME_MARGIN = 1.18

# Direction the flat kernel's KEY LIGHT comes from: it shades with max(dot(n, -key), 0) for
# key = normalize(0.35, -0.45, -0.82), so surfaces facing this direction are the brightest. The
# camera sits here, which is what lights the inside of an opening. Mirrored from _render_kernel in
# robolabViz/raycast.py — if that light ever moves, this must follow (the symptom is dark cavities).
LIT_DIRECTION = np.array([-0.35, 0.45, 0.82])
# World up hint for the camera. Not parallel to LIT_DIRECTION, so the look-at is well conditioned.
CAMERA_UP = np.array([0.0, 0.0, 1.0])

# The six views: (name, canonical axis presented to the camera, canonical direction to put at the
# TOP of the image). The names are what the VLM answers with, so they are part of the prompt
# contract: keep them stable. "front" means the object's +x face is turned toward you.
VIEWPOINTS: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("front",  (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("back",   (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("left",   (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("right",  (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("top",    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ("bottom", (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
)
VIEW_NAMES = tuple(v[0] for v in VIEWPOINTS)


@dataclass(frozen=True)
class View:
    """One rendered viewpoint plus everything needed to undo its projection.

    ``rotation`` is the 3x3 taking CANONICAL coordinates to the RENDER frame this view's geometry
    was posed in, about ``pivot``. Rays are transformed back through it, so every point this module
    returns is canonical — no caller ever sees the render frame."""
    name: str
    eye: np.ndarray             # render-frame camera position (the same for every view)
    target: np.ndarray
    up: np.ndarray
    rotation: np.ndarray        # 3x3 canonical -> render
    pivot: np.ndarray           # rotation centre, shared by both frames
    width: int
    height: int
    tan_half_w: float
    tan_half_h: float
    basis: np.ndarray           # 3x3, columns = camera X (right), Y (up), Z (backward)
    png: Path | None = None

    def render_to_canonical(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=float) - self.pivot) @ self.rotation + self.pivot

    def ray(self, u_frac: float, v_frac: float) -> tuple[np.ndarray, np.ndarray]:
        """CANONICAL-frame ray for a point given in IMAGE coordinates: ``(0,0)`` top-left,
        ``(1,1)`` bottom-right.

        The direction mirrors ``_render_kernel`` exactly, but continuously — the kernel evaluates it
        at pixel centres, and a VLM's pick is a fraction, not a pixel index. Ray and geometry must
        meet in one frame; taking the ray back to canonical (rather than posing a second copy of the
        mesh) keeps the geometry the single copy everything is measured against."""
        x = float(u_frac) * 2.0 - 1.0
        y = 1.0 - float(v_frac) * 2.0
        d_cam = np.array([x * self.tan_half_w, y * self.tan_half_h, -1.0], dtype=float)
        d = self.basis @ d_cam
        d = d / np.linalg.norm(d)
        return self.render_to_canonical(self.eye), d @ self.rotation

    def to_dict(self) -> dict:
        return {"name": self.name, "eye": self.eye.tolist(), "target": self.target.tolist(),
                "up": self.up.tolist(), "rotation": self.rotation.tolist(),
                "pivot": self.pivot.tolist(), "width": self.width, "height": self.height,
                "focal_length": FOCAL_LENGTH, "aperture": APERTURE}


def _look_at_basis(eye, target, up) -> np.ndarray:
    """Columns = camera basis in the parent frame (X right, Y up, Z backward).

    Same construction as ``robolabViz.config.look_at_quat_wxyz``, kept as a matrix because the
    back-projection needs the basis rather than the quaternion the renderer wants."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    n = np.linalg.norm(right)
    if n < 1.0e-6:
        raise ValueError("view up vector is parallel to the view direction")
    right = right / n
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, -forward], axis=1)


def _orthonormal(primary, secondary) -> np.ndarray:
    """Right-handed 3x3 with the given primary axis and the secondary Gram-Schmidted against it."""
    a = np.asarray(primary, dtype=float)
    a = a / np.linalg.norm(a)
    b = np.asarray(secondary, dtype=float)
    b = b - (b @ a) * a
    if np.linalg.norm(b) < 1.0e-9:
        raise ValueError("view axes are parallel")
    b = b / np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def plan_views(canonical_vertices: np.ndarray) -> list[View]:
    """The six annotation views for one asset's canonical geometry.

    The pivot / look-at target is the geometry's bounding-sphere centre rather than the frame
    origin: the canonical frame centres the OBB, and for a hollow or lopsided asset that is not
    where the material is, which would push the object off-centre in half the views."""
    v = np.asarray(canonical_vertices, dtype=float)
    lo, hi = v.min(axis=0), v.max(axis=0)
    pivot = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(v - pivot, axis=1).max())
    tan_half = APERTURE / (2.0 * FOCAL_LENGTH)
    distance = FRAME_MARGIN * radius / tan_half

    lit = LIT_DIRECTION / np.linalg.norm(LIT_DIRECTION)
    eye = pivot + distance * lit
    basis = _look_at_basis(eye, pivot, CAMERA_UP)
    # Where the presented face and the image-up hint must LAND in the render frame: the face turns
    # to look at the camera, the up hint to the camera's up.
    render_frame = _orthonormal(lit, basis[:, 1])

    views = []
    for name, face, up_hint in VIEWPOINTS:
        # rotation maps the canonical (face, up) pair onto the render (toward-camera, up) pair.
        rotation = render_frame @ _orthonormal(face, up_hint).T
        views.append(View(name=name, eye=eye, target=pivot, up=CAMERA_UP.copy(),
                          rotation=rotation, pivot=pivot,
                          width=IMAGE_SIZE, height=IMAGE_SIZE,
                          tan_half_w=tan_half, tan_half_h=tan_half, basis=basis))
    return views


def pose_for_render(canonical_vertices: np.ndarray, view: View) -> np.ndarray:
    """The view's geometry: canonical vertices rotated into the render frame."""
    v = np.asarray(canonical_vertices, dtype=float)
    return (v - view.pivot) @ view.rotation.T + view.pivot


# =================================================================================================
# Rendering
# =================================================================================================
def render_views(canonical_vertices: np.ndarray, faces: np.ndarray, views: list[View],
                 out_dir: Path, *, device: str = "cuda:0", verbose: bool = True) -> list[View]:
    """Render each view to ``out_dir`` and return the views with their ``png`` filled in.

    Uses the same ``RaycastPreviewRenderer`` the scene-generation verification stage renders with —
    driven directly from geometry (no Newton model, no robot, no table) since an asset annotation
    has no scene. The lightweight flat tier is deliberate: the HDRI/PBR tier costs seconds per image
    and adds specular highlights and cast shadows a VLM can mistake for surface features, while
    shape reads identically under flat shading. One renderer per view because each view poses the
    geometry differently (see the module docstring); at ~60 ms a view that is not worth avoiding.
    """
    from robolabViz.config import CameraConfig, RenderQuality, RoboLabSceneConfig, look_at_quat_wxyz
    from robolabViz.raycast import CATEGORIES, RaycastPreviewRenderer, _Instance

    out_dir.mkdir(parents=True, exist_ok=True)
    quality = RenderQuality(use_hdri=False, use_textures=False)
    out = []
    for view in views:
        cam = CameraConfig(name=view.name, position=tuple(view.eye), width=view.width,
                           height=view.height,
                           orientation_wxyz=look_at_quat_wxyz(view.eye, view.target, view.up),
                           focal_length=FOCAL_LENGTH, horizontal_aperture=APERTURE,
                           vertical_aperture=APERTURE,
                           # Out of the combined video, which here means no video is written at
                           # all: these are stills, and an mp4 of one frame is a stray artifact.
                           in_combined_video=False)
        inst = _Instance(name="asset", category=CATEGORIES.index("objects"), kind="static",
                         key=None,
                         verts=np.ascontiguousarray(pose_for_render(canonical_vertices, view),
                                                    dtype=np.float32),
                         tris=np.ascontiguousarray(faces, dtype=np.int32),
                         color=(0.78, 0.75, 0.71), roughness=0.6, smooth=True)
        work = out_dir / view.name
        renderer = RaycastPreviewRenderer(
            scene=RoboLabSceneConfig(exterior_cameras=[cam]), object_model=None,
            robot_link_names=None, output_dir=work, device=device, instances=[inst],
            frames_per_image=1, write_coverage_json=False, quality=quality)
        renderer.render_frame({}, np.zeros((0, 7), dtype=np.float32), None)
        renderer.close()
        raw = work / "frames" / f"{view.name}_0000.png"
        if not raw.exists():
            raise RuntimeError(f"view {view.name!r} did not render to {raw}")
        final = out_dir / f"{view.name}.png"
        _annotate_grid(raw, final, view.name)
        out.append(replace(view, png=final))
    if verbose:
        print(f"  rendered {len(out)} views -> {out_dir}")
    return out


_GRID_STEPS = 10


def _annotate_grid(src: Path, dst: Path, label: str) -> None:
    """Draw a faint labelled 10x10 grid over the render and write it to ``dst``.

    The grid is a READING AID for the VLM, not data: it is asked for a position as a fraction of
    the image, and reading "just past the 0.3 line" off ruled gridlines is far more accurate than
    estimating a fraction of a blank frame. Back-projection uses the fraction, never the drawn
    pixels, so the overlay cannot bias the geometry.
    """
    from PIL import Image, ImageDraw

    img = Image.open(src).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(1, _GRID_STEPS):
        f = i / _GRID_STEPS
        draw.line([(f * w, 0), (f * w, h)], fill=(255, 255, 255, 60), width=1)
        draw.line([(0, f * h), (w, f * h)], fill=(255, 255, 255, 60), width=1)
        draw.text((f * w + 3, 3), f"{f:.1f}", fill=(255, 255, 255, 190))
        draw.text((3, f * h + 3), f"{f:.1f}", fill=(255, 255, 255, 190))
    draw.text((6, h - 18), f"view: {label}", fill=(255, 255, 255, 230))
    img.save(dst)


# =================================================================================================
# Back-projection
# =================================================================================================
def _ray_mesh_hit(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray,
                  faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Nearest front hit of one ray on the mesh -> (point, outward normal), or None if it misses.

    Vectorized Moller-Trumbore over every triangle. Deliberately brute force and dependency-free:
    a handful of rays per asset makes an acceleration structure pointless, and the alternative
    (trimesh's ray backends) picks a different implementation depending on whether embree is
    installed, which is exactly the kind of environment-dependent answer a cached annotation must
    not have.
    """
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    pv = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1.0e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tv = origin - v0
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ direction) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1.0 + 1.0e-9) & (t > 1.0e-6)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)[np.argmin(t[hit])]
    point = origin + t[idx] * direction
    normal = np.cross(e1[idx], e2[idx])
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
    if normal @ direction > 0.0:            # face the camera, whatever the winding
        normal = -normal
    return point, normal


def backproject(view: View, u_frac: float, v_frac: float, vertices: np.ndarray,
                faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Canonical-frame surface point (and its outward normal) under a picked image position.

    Returns None when the pick misses the object — the honest outcome for a VLM that pointed at
    background, and the reason a pick is dropped rather than snapped to the nearest surface."""
    if not (0.0 <= u_frac <= 1.0 and 0.0 <= v_frac <= 1.0):
        return None
    origin, direction = view.ray(u_frac, v_frac)
    return _ray_mesh_hit(origin, direction, np.asarray(vertices, dtype=float),
                         np.asarray(faces, dtype=np.int64))
