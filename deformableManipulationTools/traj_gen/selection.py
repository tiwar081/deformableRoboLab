"""Online grasp choice for the trajectory stage: PHYSICS-TIERED re-rank + weighted sampling.

``grasp_select.select_grasps`` already assembles the pool per the candidate-status taxonomy
(grasp-library.md "Candidate statuses"): weak/``retreated`` candidates are EXCLUDED from the default
pool (the user's rule — a retreated candidate is not a legitimate grasp, its effective score is 0
and it is never sampled) and ``seat_blocked`` candidates are dropped unconditionally (the merge
already discards them from records; any met here are pre-migration leftovers). This module is the
CALLER-side re-rank the selection result was designed for ("a caller re-ranks ``result.ranked``
itself"):

1. **Physics tier is the primary key** — the full-catalog shake (1719 trials) measured every
   candidate, so the ranking now leads with that evidence: tier 0 = measured HELD
   (``object_in_gripper == 1``), tier 1 = never physics-tested (skip/no sidecar — "unknown", not
   "bad"), tier 2 = measured DROP. Within a tier the ``grasp_select`` score (quality motion metrics,
   region proximity, awkwardness, containment) orders candidates as before.
2. **Sampling is score-weighted random, not argmin** — the pick is drawn from
   ``p ∝ exp(-(score + tier_penalty)/T)`` so better-scoring candidates are preferred but the stage
   still explores (retries never re-draw an already-tried id). ``temperature=0`` degenerates to the
   deterministic best, which is what the selftest pins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grasp_select import RobotState, ScoredGrasp, SelectionResult, select_grasps
from ..grasp_select.scoring import sample_index

# Additive sampling-cost penalty per physics tier (scores are ~[0, 1] weighted means).
# Held evidence dominates: an untested candidate must beat a held one by a wide score margin to be
# sampled ahead of it, and a measured drop is close to (but not exactly) unsampleable.
TIER_PENALTY = {0: 0.0, 1: 0.35, 2: 0.90}
# Default sampling temperature: sharp preference for the best few, non-zero exploration.
DEFAULT_TEMPERATURE = 0.08


def physics_tier(candidate) -> int:
    """0 = measured held, 1 = no physics measurement, 2 = measured drop."""
    q = getattr(candidate, "quality", None) or {}
    held = q.get("object_in_gripper")
    if not getattr(candidate, "quality_source", None) or held is None:
        return 1
    return 0 if float(held) >= 0.5 else 2


@dataclass(frozen=True)
class RankedGrasp:
    grasp: ScoredGrasp
    tier: int
    cost: float                   # score.total + TIER_PENALTY[tier] — the sampling cost

    @property
    def id(self) -> str:
        return self.grasp.id


@dataclass
class TaskRanking:
    """The physics-re-ranked candidate list for one placed object, plus the raw selection result."""
    ranked: list = field(default_factory=list)          # list[RankedGrasp], best first
    selection: SelectionResult | None = None

    def __len__(self) -> int:
        return len(self.ranked)

    def by_id(self, cid: str) -> RankedGrasp | None:
        return next((r for r in self.ranked if r.id == cid), None)

    def tiers(self) -> dict:
        out = {0: 0, 1: 0, 2: 0}
        for r in self.ranked:
            out[r.tier] += 1
        return out

    def report(self) -> str:
        t = self.tiers()
        lines = [f"physics re-rank: {len(self.ranked)} selectable — "
                 f"{t[0]} measured-held, {t[1]} untested, {t[2]} measured-drop"]
        for i, r in enumerate(self.ranked[:10]):
            g = r.grasp
            lines.append(f"  #{i:<2d} T{r.tier} cost={r.cost:.3f} {g.id:34s} "
                         f"[{g.score.describe()}] {g.command.describe()}")
        if self.selection is not None:
            lines.append(self.selection.report())
        return "\n".join(lines)


def rank_for_task(record, placement, robot: RobotState, *, obstacles=(),
                  regions=None) -> TaskRanking:
    """Run the full online selection for a placed object, then re-rank by physics tier.

    ``select_grasps`` runs at temperature 0 (its own sampler is unused — this module samples);
    everything it rejects (status/bucket/reach/clearance/projection/depth) stays rejected, with the
    reasons preserved in ``ranking.selection``."""
    result = select_grasps(record, placement, robot, obstacles=obstacles, regions=regions,
                           temperature=0.0)
    ranked = [RankedGrasp(grasp=g, tier=physics_tier(g.candidate),
                          cost=float(g.score.total) + TIER_PENALTY[physics_tier(g.candidate)])
              for g in result.ranked]
    ranked.sort(key=lambda r: (r.cost, r.id))
    return TaskRanking(ranked=ranked, selection=result)


def draw(ranking: TaskRanking, *, temperature: float = DEFAULT_TEMPERATURE,
         rng: np.random.Generator | None = None, exclude=()) -> RankedGrasp | None:
    """Score-weighted random pick among candidates not in ``exclude`` (already-tried ids).

    Softmin over the tiered sampling cost (``grasp_select.scoring.sample_index`` — the one
    temperature-sampling definition); ``temperature=0`` is the deterministic best remaining."""
    pool = [r for r in ranking.ranked if r.id not in set(exclude)]
    if not pool:
        return None
    idx = sample_index([r.cost for r in pool], temperature,
                       rng if rng is not None else np.random.default_rng(0))
    return pool[idx]
