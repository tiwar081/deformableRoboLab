"""Trajectory generation — the stage between task generation and simulation.

Turns a generated pipeline run (``scene.json`` + ``task.json``) into an EXECUTED robot policy:
online grasp selection over the precomputed grasp library (physics-tiered re-rank + score-weighted
sampling), a Bezier-legged pick-and-place plan with collision-driven control-point insertion, a
headless rollout that measures the grasp and the task goal, and a bounded (2-attempt) LLM recovery
loop for failed grasps. The plan is written as ``traj.json`` beside the run's demo file, which
``agentic_pipeline.build.demo_from_dir`` turns into the demo's policy — so the standard runner and
renderer replay exactly what was evaluated.

    from deformableManipulationTools.traj_gen import generate_trajectory
    report = generate_trajectory("outputs/agenticPipeline/<run>")

CLI (select -> plan -> rollout -> retries -> final mp4 render):

    .venv/bin/python -m deformableManipulationTools.traj_gen <run_dir> [--device cuda:0]

Docs: docs/trajPipeline/trajectory-generation.md. Invariants (no GPU):
``python -m deformableManipulationTools.traj_gen.selftest``.
"""
from .policy import PickSpec, PlanError, TrajPlan, plan_pick_place, resolve_place, SUPPORTED_GOALS
from .selection import TaskRanking, draw, physics_tier, rank_for_task
from .stage import generate_trajectory

__all__ = ["generate_trajectory", "rank_for_task", "draw", "physics_tier", "TaskRanking",
           "PickSpec", "PlanError", "TrajPlan", "plan_pick_place", "resolve_place",
           "SUPPORTED_GOALS"]
