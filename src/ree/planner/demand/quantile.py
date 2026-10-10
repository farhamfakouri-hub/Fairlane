"""Negative binomial quantiles of customer counts, and the fit of their dispersion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

MIN_DISPERSION = 3.0
MAX_DISPERSION = 2000.0


def nbinom_parameters(mean: float, dispersion: float) -> tuple[float, float]:
    """Shape and success probability of the negative binomial for a mean and a dispersion."""
    return dispersion, dispersion / (dispersion + mean)


def quantile_count(mean: float, dispersion: float, probability: float) -> int:
    """Customer count at a quantile level for a given mean and dispersion."""
    if mean <= 0.0:
        return 0
    shape, success = nbinom_parameters(mean, dispersion)
    return int(stats.nbinom.ppf(probability, shape, success))


def fit_dispersion(means: Sequence[float], counts: Sequence[float]) -> float:
    """Estimate the dispersion from forecast means and actual counts, within fixed limits."""
    forecast = np.asarray(means, dtype=np.float64)
    actual = np.asarray(counts, dtype=np.float64)
    keep = forecast > 0.0
    if keep.sum() < 10:
        return MAX_DISPERSION
    excess = float(np.sum((actual[keep] - forecast[keep]) ** 2 - forecast[keep]))
    if excess <= 0.0:
        return MAX_DISPERSION
    estimate = float(np.sum(forecast[keep] ** 2)) / excess
    return float(min(max(estimate, MIN_DISPERSION), MAX_DISPERSION))


def exceedance_rate(counts: Sequence[float], planned: Sequence[float]) -> float:
    """Share of slots in which the actual count was above the planned one."""
    actual = np.asarray(counts, dtype=np.float64)
    quantile = np.asarray(planned, dtype=np.float64)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(actual > quantile))
