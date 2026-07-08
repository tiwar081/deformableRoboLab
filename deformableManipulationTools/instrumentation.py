"""Centralized per-frame grip instrumentation + physics-graph output (shared by EVERY demo via
the run harness; never wired per-demo). Records, once per rendered frame, three signals of the
gripper-PAD ↔ object contact — penetration depth, grip force, and the number of contact points —
and saves them as a single 3-panel PNG with green/red dotted verticals marking each grasp/release
trigger, so the grasped intervals are visible at a glance.

VBD path only: the metrics are defined on the dynamic finger PROXIES (the two pads), so a demo with
no proxies (the rigid-only MuJoCo grasp) records nothing. The contact geometry is read host-side
(numpy) from the PUBLIC Newton contact buffers — penetration uses Newton's own definition
(``C_n = (margin0+margin1) − n·(p1−p0)``, see ``newton`` ``contact_surface_separation``); nothing in
``_external`` is imported at runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------------------
# Host-side contact metrics over the public Newton contact buffers.
# ------------------------------------------------------------------------------------------------
def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row-vectors ``v`` (m,3) by xyzw quaternions ``q`` (m,4)."""
    u = q[:, :3]
    w = q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _to_world(body_q: np.ndarray, bodies: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map body-frame points ``pts`` (m,3) on ``bodies`` (m,) to world via ``body_q`` (nb,7=pos+xyzw)."""
    xf = body_q[bodies]
    return xf[:, :3] + _qrot(xf[:, 3:7], pts)


def gripper_contact_metrics(body_q, shape_body, pad_bodies, proxy_bodies,
                            rigid=None, soft=None, particle_q=None, particle_radius=None):
    """``(max_penetration_m, n_contact_points)`` for the gripper PADS against the object.

    A contact qualifies when one side is a pad and the other is a non-proxy object body (rigid) or
    when a particle touches a pad shape (soft). Penetration is the deepest qualifying overlap
    (``max(0, −separation)``); the count is every qualifying generated contact constraint.
    Args are host (numpy) arrays already copied off device.
    """
    pad_bodies = np.asarray(pad_bodies)
    proxy_bodies = np.asarray(proxy_bodies)
    max_pen = 0.0
    n_contacts = 0

    if rigid is not None and int(rigid["count"]) > 0:
        cc = int(rigid["count"])
        s0 = np.asarray(rigid["shape0"][:cc]); s1 = np.asarray(rigid["shape1"][:cc])
        valid = (s0 >= 0) & (s1 >= 0)
        if valid.any():
            b0 = np.where(s0 >= 0, shape_body[np.clip(s0, 0, None)], -1)
            b1 = np.where(s1 >= 0, shape_body[np.clip(s1, 0, None)], -1)
            pad0, pad1 = np.isin(b0, pad_bodies), np.isin(b1, pad_bodies)
            px0, px1 = np.isin(b0, proxy_bodies), np.isin(b1, proxy_bodies)
            qual = valid & ((pad0 & ~px1) | (pad1 & ~px0))
            idx = np.nonzero(qual)[0]
            if idx.size:
                cp0 = _to_world(body_q, b0[idx], np.asarray(rigid["point0"][:cc])[idx])
                cp1 = _to_world(body_q, b1[idx], np.asarray(rigid["point1"][:cc])[idx])
                nrm = np.asarray(rigid["normal"][:cc])[idx]
                m0 = np.asarray(rigid["margin0"][:cc])[idx]
                m1 = np.asarray(rigid["margin1"][:cc])[idx]
                sep = np.einsum("ij,ij->i", nrm, cp1 - cp0) - (m0 + m1)
                pen = np.maximum(0.0, -sep)
                n_contacts += int(idx.size)
                if pen.size:
                    max_pen = max(max_pen, float(pen.max()))

    if soft is not None and int(soft["count"]) > 0:
        sc = int(soft["count"])
        pid = np.asarray(soft["particle"][:sc]); shp = np.asarray(soft["shape"][:sc])
        valid = (pid >= 0) & (shp >= 0)
        if valid.any():
            body = np.where(shp >= 0, shape_body[np.clip(shp, 0, None)], -1)
            qual = valid & np.isin(body, pad_bodies)
            idx = np.nonzero(qual)[0]
            if idx.size:
                bx = _to_world(body_q, body[idx], np.asarray(soft["body_pos"][:sc])[idx])
                nrm = np.asarray(soft["normal"][:sc])[idx]
                pq = particle_q[pid[idx]]
                pr = particle_radius[pid[idx]]
                pen = -(np.einsum("ij,ij->i", nrm, pq - bx) - pr)
                n_contacts += int(idx.size)
                pos = pen[pen > 0.0]
                if pos.size:
                    max_pen = max(max_pen, float(pos.max()))

    return max_pen, n_contacts


# ------------------------------------------------------------------------------------------------
# Recorder + 3-panel PNG.
# ------------------------------------------------------------------------------------------------
class PhysicsRecorder:
    """Accumulates the per-frame grip triplet and writes the 3-panel physics PNG. One per run,
    created by the run harness only when ``--output`` requests graphs AND the demo is on the VBD
    (proxy) path."""

    def __init__(self, fps: float, grasp_windows):
        self.fps = float(fps)
        self.windows = list(grasp_windows or [])
        self.penetration_mm: list[float] = []
        self.force_n: list[float] = []          # raw min-of-pads reaction magnitude
        self.squeeze_n: list[float] = []        # closing-axis projected squeeze (the regulated quantity)
        self.contacts: list[int] = []

    def record(self, penetration_m: float, force_n: float, squeeze_n: float, n_contacts: int) -> None:
        self.penetration_mm.append(float(penetration_m) * 1.0e3)
        self.force_n.append(float(force_n))
        self.squeeze_n.append(float(squeeze_n))
        self.contacts.append(int(n_contacts))

    @property
    def frames(self) -> int:
        return len(self.penetration_mm)

    def _grasp_release_frames(self):
        """``(grasp_frames, release_frames)`` as frame indices from the GraspWindow trigger times."""
        grasp, release = [], []
        for w in self.windows:
            if np.isfinite(w.close_start):
                grasp.append(int(round(w.close_start * self.fps)))
            if np.isfinite(w.release_start):
                release.append(int(round(w.release_start * self.fps)))
        return grasp, release

    def _force_refs(self):
        """Distinct ``(target, engage, deadband)`` triples [N] across the grasp windows — the
        admittance regulator's reference levels for the projected-squeeze signal. Resolved exactly
        as the kernel does: per-window target, clamped engage, and the TARGET-RELATIVE deadband via
        the ONE centralized derivation ``GripConfig.window_params`` (never re-derived here)."""
        from .params import GRIP
        refs, seen = [], set()
        for w in self.windows:
            target = float(w.force_target) if w.force_target is not None else float(GRIP.force_target)
            engage = min(max(GRIP.engage_frac * target, GRIP.engage_floor), GRIP.engage_cap)
            _, deadband = GRIP.window_params(target)
            key = (round(target, 6), round(engage, 6))
            if key not in seen:
                seen.add(key)
                refs.append((target, float(engage), float(deadband)))
        return refs

    def save(self, path, title: str | None = None) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        x = np.arange(self.frames)
        grasp, release = self._grasp_release_frames()

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        def _triggers(ax):
            first_g, first_r = True, True
            for gf in grasp:
                ax.axvline(gf, color="green", linestyle=":", linewidth=1.4,
                           label="grasp triggered" if first_g else None)
                first_g = False
            for rf in release:
                ax.axvline(rf, color="red", linestyle=":", linewidth=1.4,
                           label="release triggered" if first_r else None)
                first_r = False

        # --- penetration ---
        ax = axes[0]
        ax.plot(x, self.penetration_mm, color="tab:blue", linewidth=1.2)
        ax.set_title("gripper penetration", fontsize=11, loc="left")
        ax.set_ylabel("penetration [mm]")
        ax.grid(True, alpha=0.3)
        _triggers(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=9)

        # --- force: regulated projected squeeze (primary) + raw pad |F| (secondary) + control refs ---
        ax = axes[1]
        refs = self._force_refs()
        first_t, first_d, first_e = True, True, True
        for target, engage, deadband in refs:
            ax.axhspan(target - deadband, target + deadband, color="orange", alpha=0.13,
                       label="target ± deadband" if first_d else None)
            ax.axhline(target, color="darkorange", linestyle=":", linewidth=1.5,
                       label="target force" if first_t else None)
            ax.axhline(engage, color="purple", linestyle=":", linewidth=1.3,
                       label="engage force" if first_e else None)
            first_t = first_d = first_e = False
        ax.plot(x, self.squeeze_n, color="tab:red", linewidth=1.3, label="grip force (closing-axis squeeze)")
        ax.plot(x, self.force_n, color="tab:red", linewidth=0.9, alpha=0.35, label="pad |F| (raw)")
        ax.set_title("gripper force", fontsize=11, loc="left")
        ax.set_ylabel("grip force [N]")
        ax.grid(True, alpha=0.3)
        _triggers(ax)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

        # --- contact points ---
        ax = axes[2]
        ax.plot(x, self.contacts, color="tab:green", linewidth=1.2)
        ax.set_title("gripper–object contact points", fontsize=11, loc="left")
        ax.set_ylabel("# contact points")
        ax.grid(True, alpha=0.3)
        _triggers(ax)

        axes[2].set_xlabel("frame")
        fig.suptitle(title or "Gripper physics", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
