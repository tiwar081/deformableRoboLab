"""Central project settings, loaded once from the repo-root ``settings.yaml``.

This is the single place OUTSIDE the package source where project-wide choices are made — most
importantly which **robot** every demo, visualization, and renderer uses. ``params.py`` consumes
:data:`SETTINGS` to pick the active :class:`~deformableManipulationTools.params.RobotConfig`, so a
one-line edit to ``settings.yaml`` switches the whole codebase between robots.

Kept dependency-light (``yaml`` + ``pathlib``) and free of any ``newton``/``warp`` import so the
lowest-level modules (``params.py``) can import it without a cycle. Missing keys fall back to the
defaults below, so a deleted or partial ``settings.yaml`` still runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root = parent of the package directory; settings.yaml lives there (outside all folders).
REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "settings.yaml"

# Defaults applied when settings.yaml (or a given key) is absent.
_DEFAULTS = {
    "robot": "franka_panda_isaacsim",
    "render": {"style": "mp4", "table": "maple", "background": "home_office"},
    "device": "cuda:0",
}


@dataclass(frozen=True)
class Settings:
    robot: str                 # active robot key into params.ROBOTS
    render_style: str          # default --output-style: usd | mp4 | mp4_advanced (CLI overrides)
    render_table: str          # default scenic work-table texture (CLI --table overrides)
    render_background: str     # default scenic dome background (CLI --background overrides)
    device: str | None         # default compute device (CLI --device overrides; None = Warp default)


def _load() -> Settings:
    data = {}
    if SETTINGS_PATH.exists():
        data = yaml.safe_load(SETTINGS_PATH.read_text()) or {}
    render = {**_DEFAULTS["render"], **(data.get("render") or {})}
    return Settings(
        robot=data.get("robot", _DEFAULTS["robot"]),
        render_style=render["style"],
        render_table=render["table"],
        render_background=render["background"],
        device=data.get("device", _DEFAULTS["device"]),
    )


SETTINGS = _load()
