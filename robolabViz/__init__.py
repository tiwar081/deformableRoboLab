"""RoboLab-style visualization layer for the Newton split-solver demos.

This package reproduces the graphics used by NVIDIA RoboLab's
``examples/run_recorded.py`` (Isaac Sim RTX rendering, DROID camera rig,
home-office dome light, maple work table, Franka pedestal) on top of this
repo's Newton physics, without importing anything from ``_external/``.

Pipeline:

- ``config``       — dataclasses describing the scene (fixtures, lights,
                     cameras, object styling). ``droid_scene_config()`` returns
                     RoboLab's exact values; every field is overridable.
- ``robot_usd``    — one-shot URDF -> USD conversion via Isaac Sim's importer
                     (subprocess, cached under ``assets/robots/``).
- ``robot_fk``     — uncollapsed FK mirror of the simulated robot so every
                     URDF link (including fixed ones) gets a world transform.
- ``stage``        — authors the full scene as a time-sampled USD with pure
                     ``pxr`` (no kit needed) and puppeteers it per frame.
- ``raycast``      — warp-based raycast preview renderer producing real frames
                     from the configured cameras on any CUDA GPU (used for
                     wrist-camera tuning and verification on machines without
                     RT cores, where Isaac Sim cannot render).
- ``isaac_render`` — offline RTX renderer for the produced USD (requires an
                     RTX-capable GPU; H100/H200 have no RT cores and cannot
                     run it).
"""

from .config import (  # noqa: F401
    CameraConfig,
    DomeLightConfig,
    FixtureConfig,
    ObjectStyle,
    RoboLabSceneConfig,
    SphereLightConfig,
    WristCameraConfig,
    droid_scene_config,
)
