"""Bootstrap intervals around the combined score."""

from __future__ import annotations

import numpy as np

from ree.config import CombineConfig, IntervalConfig, make_rng
from ree.domain.models import Context, Interval
from ree.engine.combine import combined_score
from ree.engine.shrinkage import shrink, to_score_scale
from ree.engine.speed_v1 import V1Fit
from ree.engine.speed_v2 import V2Fit


def joint_speed_draws(
    fit: V1Fit | V2Fit,
    worker_ids: list[str],
    period: str,
    n_valid: np.ndarray,
    reference: np.ndarray,
    seed: int,
    n_bootstrap: int,
    shrinkage_k: float,
    combine: CombineConfig,
) -> np.ndarray:
    """Bootstrap draws of the speed score of all cashiers at once.

    Each draw is shrunk and standardised against the rated group, so the score keeps the
    effect of the reference group moving with it.
    """
    raw = np.zeros((n_bootstrap, len(worker_ids)))
    for column, worker_id in enumerate(worker_ids):
        rng = make_rng(seed, f"bootstrap:speed:{worker_id}:{period}")
        raw[:, column] = fit.bootstrap(worker_id, period, rng, n_bootstrap)
    shrunk = np.asarray(shrink(raw, n_valid[None, :].astype(np.float64), shrinkage_k))
    selected = shrunk[:, reference] if reference.any() else shrunk
    mean = selected.mean(axis=1, keepdims=True)
    spread = selected.std(axis=1, keepdims=True)
    spread = np.where(spread > 0, spread, 1.0)
    z = np.clip((shrunk - mean) / spread, -combine.z_cap, combine.z_cap)
    return to_score_scale(z, combine.score_center, combine.score_scale)


def accuracy_draws_with_fallback(
    draws: np.ndarray,
    point_score: float,
    weeks_of_data: int,
    cohort_scores: np.ndarray,
    config: IntervalConfig,
    combine: CombineConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Accuracy draws, or normal draws around the point score when there are too few weeks
    of data to bootstrap.
    """
    if weeks_of_data >= config.min_weeks_for_bootstrap or len(cohort_scores) < 2:
        return draws
    spread = float(np.std(cohort_scores))
    fallback = point_score + spread * rng.standard_normal(len(draws))
    low = combine.score_center - combine.score_scale * combine.z_cap
    high = combine.score_center + combine.score_scale * combine.z_cap
    return np.clip(fallback, low, high)


def combined_interval(
    speed_draws: np.ndarray,
    accuracy_draws: np.ndarray,
    point_score: float,
    config: IntervalConfig,
    combine: CombineConfig,
    context: Context,
) -> Interval:
    """Interval of the combined score from the draws; it always contains the point score."""
    draws = combined_score(speed_draws, accuracy_draws, combine, context)
    tail = (1.0 - config.level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(lower=float(min(lower, point_score)), upper=float(max(upper, point_score)))
