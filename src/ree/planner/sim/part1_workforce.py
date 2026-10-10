"""Workforce for the planner simulation, with scores from Part 1 and hidden true skills.

Only the simulation and the evaluation read the hidden skills; the planner sees the
scores through the score provider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ree.api.provider import InMemoryScoreProvider
from ree.config import Config, make_rng
from ree.domain.models import CashierScoreRecord
from ree.engine.pipeline import run as score_store
from ree.planner.config import PlannerConfig
from ree.planner.roster.model import UNRATED_TIER, ScoreInputs, clipped_z
from ree.simulator.run import simulate_store

PART_ONE_SEED_OFFSET = 1000
TIER_SHARE = 0.2


@dataclass(frozen=True)
class WorkerTruth:
    """Hidden true speed and traits of a simulated cashier."""

    worker_id: str
    log_speed: float
    peak_effect: float
    newcomer: bool
    tenure_weeks: float


@dataclass
class Part1Workforce:
    """A simulated workforce: its score provider, hidden truth and score records."""

    provider: InMemoryScoreProvider
    truth: dict[str, WorkerTruth]
    records: list[CashierScoreRecord]


def part1_config(config: PlannerConfig, seed: int, count: int) -> Config:
    """Part 1 configuration for a store of the given size and seed."""
    return Config().with_overrides(
        {
            "seed": PART_ONE_SEED_OFFSET + seed,
            "simulator.population.n_cashiers": count,
            "simulator.n_weeks": config.workforce.part1_weeks,
            "engine.intervals.n_bootstrap": config.workforce.part1_bootstrap,
        }
    )


def part1_workforce(config: PlannerConfig, seed: int, count: int) -> Part1Workforce:
    """Simulate a Part 1 store and score it, giving the provider and the hidden truth."""
    part_one = part1_config(config, seed, count)
    store = simulate_store(part_one)
    records = score_store(store.data, part_one)
    tau = part_one.simulator.population.learning_tau_weeks
    truth: dict[str, WorkerTruth] = {}
    for row in store.ground_truth.itertuples(index=False):
        deficit = float(row.learning_deficit) * float(np.exp(-float(row.tenure_weeks) / tau))
        truth[str(row.worker_id)] = WorkerTruth(
            worker_id=str(row.worker_id),
            log_speed=float(row.true_speed) - deficit,
            peak_effect=float(row.peak_effect),
            newcomer=bool(row.is_newcomer),
            tenure_weeks=float(row.tenure_weeks),
        )
    return Part1Workforce(InMemoryScoreProvider(records), truth, records)


def service_multiplier(truth: WorkerTruth, rush: bool) -> float:
    """Factor on service time of a cashier, from true speed and the peak effect."""
    return float(np.exp(-(truth.log_speed + (truth.peak_effect if rush else 0.0))))


def oracle_inputs(truth: dict[str, WorkerTruth], config: PlannerConfig) -> ScoreInputs:
    """Score inputs built from the true speeds, as if the scores were perfect."""
    ids = sorted(truth)
    speeds = np.array([truth[w].log_speed for w in ids])
    spread = float(speeds.std()) or 1.0
    standard = (speeds - float(speeds.mean())) / spread
    order = np.argsort(standard)
    cut = max(1, round(TIER_SHARE * len(ids)))
    tiers = {ids[i]: "B" for i in range(len(ids))}
    for index in order[:cut]:
        tiers[ids[int(index)]] = "C"
    for index in order[-cut:]:
        tiers[ids[int(index)]] = "A"
    clip = config.objective.score_clip
    z = {w: clipped_z(50.0 + 15.0 * float(standard[i]), clip) for i, w in enumerate(ids)}
    return ScoreInputs(z, tiers, frozenset())


def shuffled_inputs(scores: ScoreInputs, seed: int) -> ScoreInputs:
    """Score inputs handed to the wrong workers, as a placebo."""
    ids = sorted(scores.z)
    rng = make_rng(seed, "planner-placebo")
    permutation = [int(i) for i in rng.permutation(len(ids))]
    donor = {ids[i]: ids[permutation[i]] for i in range(len(ids))}
    return ScoreInputs(
        z={w: scores.z[donor[w]] for w in ids},
        tiers={w: scores.tiers[donor[w]] for w in ids},
        exploration=frozenset(w for w in ids if donor[w] in scores.exploration),
    )


def unrated_share(scores: ScoreInputs) -> float:
    """Share of workers who are unrated."""
    ids = sorted(scores.tiers)
    return sum(1 for w in ids if scores.tiers[w] == UNRATED_TIER) / max(len(ids), 1)
