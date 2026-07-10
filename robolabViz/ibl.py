"""Image-based-lighting precompute for the raycast renderer's advanced tier.

Everything here runs ONCE at renderer init (numpy + cv2, no warp): decode the
dome HDR/EXR to a linear, auto-exposed environment map, then derive from it

- ``extract_key_lights`` — the K brightest solid-angle-weighted regions as
  directional "suns" (shadow-tested in the kernel), with those texels zeroed
  out of a residual map so their energy is not double counted,
- ``build_irradiance``  — a small cosine-convolved equirect (``E(n)/pi``)
  sampled by surface normal for diffuse IBL,
- ``prefilter_env``     — Gaussian-blurred equirect levels sampled along the
  reflection vector for roughness-dependent environment speculars.

Direction convention matches the render kernels' equirect lookup (Z up):
``lon = atan2(d.y, d.x)/(2*pi) + 0.5`` maps to the column, ``lat = acos(d.z)/pi``
to the row.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Same auto-exposure target as the flat preview's backdrop tone map: mean scene
# luminance lands at mid-gray, so the two tiers read consistently bright.
_MEAN_LUM_TARGET = 0.45


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _texel_dirs(width: int, height: int) -> np.ndarray:
    """Unit direction of every equirect texel center as an ``(H, W, 3)`` array."""
    lat = (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi          # 0..pi from +Z
    lon = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 * math.pi - math.pi
    sin_lat = np.sin(lat)[:, None]
    d = np.empty((height, width, 3), dtype=np.float64)
    d[..., 0] = sin_lat * np.cos(lon)[None, :]
    d[..., 1] = sin_lat * np.sin(lon)[None, :]
    d[..., 2] = np.cos(lat)[:, None]
    return d


def _texel_solid_angles(width: int, height: int) -> np.ndarray:
    """Solid angle of every equirect texel as an ``(H, W)`` array (sums to 4*pi)."""
    lat = (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi
    return np.broadcast_to(
        ((2.0 * math.pi / width) * (math.pi / height) * np.sin(lat))[:, None], (height, width)
    ).copy()


def load_linear_env(path: Path | str, out_w: int = 2048) -> np.ndarray:
    """Decode an HDR/EXR equirect to LINEAR float32 RGB, auto-exposed so the mean
    luminance is mid-gray (the kernel tone-maps at the end, so unlike the flat
    preview's backdrop nothing is clamped here). Returns ``(out_w//2, out_w, 3)``."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read environment map {path}")
    lin = cv2.cvtColor(raw[:, :, :3].astype(np.float32), cv2.COLOR_BGR2RGB)
    lin = np.clip(lin, 0.0, None)
    mean_lum = float(np.mean(_luminance(lin)))
    gain = _MEAN_LUM_TARGET / mean_lum if mean_lum > 1e-6 else 1.0
    return cv2.resize(lin * gain, (out_w, out_w // 2), interpolation=cv2.INTER_AREA)


def extract_key_lights(
    env: np.ndarray, k: int = 2, nms_deg: float = 25.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily pick the K brightest dome regions as directional lights.

    Works on a small (128x64) copy. Each pick claims the disc of texels within
    ``nms_deg/2`` of its peak: the light's radiance is the summed ``L*domega``
    over the disc (so a big soft window and a small bright lamp compare fairly),
    and the disc is zeroed in the returned residual map so ``build_irradiance``
    doesn't count the same energy again.

    Returns ``(dirs (K,3) float32, radiance (K,3) float32, residual (64,128,3))``.
    """
    import cv2

    h, w = 64, 128
    small = cv2.resize(env, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    dirs = _texel_dirs(w, h)
    domega = _texel_solid_angles(w, h)
    residual = small.copy()
    cos_disc = math.cos(math.radians(nms_deg / 2.0))
    key_dirs, key_rad = [], []
    for _ in range(max(0, int(k))):
        score = _luminance(residual) * domega
        r, c = np.unravel_index(int(np.argmax(score)), score.shape)
        if score[r, c] <= 0.0:
            break
        d0 = dirs[r, c]
        disc = (dirs @ d0) > cos_disc
        radiance = (residual * domega[..., None])[disc].sum(axis=0)
        key_dirs.append(d0)
        key_rad.append(radiance)
        residual[disc] = 0.0
    if not key_dirs:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), residual.astype(np.float32))
    return (
        np.asarray(key_dirs, dtype=np.float32),
        np.asarray(key_rad, dtype=np.float32),
        residual.astype(np.float32),
    )


def build_irradiance(env_small: np.ndarray, out_wh: tuple[int, int] = (64, 32)) -> np.ndarray:
    """Cosine-convolve a small linear equirect into ``E(n)/pi``: the diffuse term
    is then just ``albedo * irr(n)`` in the kernel. One matmul (~50 MFLOP)."""
    h_s, w_s = env_small.shape[:2]
    src_dirs = _texel_dirs(w_s, h_s).reshape(-1, 3)
    weighted = env_small.reshape(-1, 3).astype(np.float64) * _texel_solid_angles(w_s, h_s).reshape(-1, 1)
    w_o, h_o = out_wh
    out_dirs = _texel_dirs(w_o, h_o).reshape(-1, 3)
    cosines = np.clip(out_dirs @ src_dirs.T, 0.0, None)
    irr = (cosines @ weighted) / math.pi
    return irr.reshape(h_o, w_o, 3).astype(np.float32)


def prefilter_env(
    env: np.ndarray,
    out_wh: tuple[int, int] = (256, 128),
    sigmas_deg: tuple[float, ...] = (4.0, 12.0, 30.0),
    peak_clamp: float = 2.0,
) -> np.ndarray:
    """Roughness-prefiltered specular levels: the equirect blurred at increasing
    angular sigmas (longitude wrap-padded so the seam doesn't show; equirect pole
    distortion is accepted at this fidelity). Returns ``(L, H, W, 3)`` float32.

    ``peak_clamp`` caps the SPECULAR env only (~4x the normalized mean luminance):
    extreme sources like sun slits otherwise reflect off even matte surfaces as a
    smeared view-tracking streak. Diffuse irradiance and the extracted key lights
    keep the full energy, so the scene's lighting is unchanged."""
    import cv2

    w_o, h_o = out_wh
    base = cv2.resize(np.minimum(env, peak_clamp), (w_o, h_o), interpolation=cv2.INTER_AREA)
    levels = []
    for sd in sigmas_deg:
        sigma_px = max(sd / 360.0 * w_o, 0.5)
        pad = int(np.ceil(3.0 * sigma_px))
        padded = np.pad(base, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
        blurred = cv2.GaussianBlur(padded, (0, 0), sigma_px)
        levels.append(blurred[:, pad:-pad])
    return np.stack(levels).astype(np.float32)
