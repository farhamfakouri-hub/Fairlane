"""Rank metrics that compare estimated scores with the hidden true values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fill_missing(estimate: pd.Series) -> pd.Series:
    """Replace missing estimates by the median (by zero when all are missing)."""
    if estimate.isna().all():
        return estimate.fillna(0.0)
    return estimate.fillna(estimate.median())


def _align(estimate: pd.Series, truth: pd.Series) -> tuple[pd.Series, pd.Series]:
    common = truth.index.intersection(estimate.index)
    truth = truth.loc[common].astype(float)
    estimate = fill_missing(estimate.reindex(common).astype(float))
    keep = np.isfinite(truth.to_numpy())
    return estimate[keep], truth[keep]


def spearman(estimate: pd.Series, truth: pd.Series) -> float:
    """Spearman rank correlation between the estimates and the truth."""
    estimate, truth = _align(estimate, truth)
    if estimate.nunique() < 2 or len(estimate) < 3:
        return float("nan")
    return float(stats.spearmanr(estimate, truth).statistic)


def kendall(estimate: pd.Series, truth: pd.Series) -> float:
    """Kendall rank correlation between the estimates and the truth."""
    estimate, truth = _align(estimate, truth)
    if estimate.nunique() < 2 or len(estimate) < 3:
        return float("nan")
    return float(stats.kendalltau(estimate, truth).statistic)


def _top(values: pd.Series, k: int) -> set[str]:
    ordered = values.sort_values(ascending=False, kind="stable")
    return set(ordered.index[:k])


def top_k_precision(estimate: pd.Series, truth: pd.Series, k: int = 5) -> float:
    """Share of the true top k that the estimates also place in their top k."""
    estimate, truth = _align(estimate, truth)
    k = min(k, len(truth))
    if k == 0:
        return float("nan")
    return len(_top(estimate, k) & _top(truth, k)) / k


def mean_absolute_rank_error(estimate: pd.Series, truth: pd.Series) -> float:
    """Average distance, in rank positions, between estimate and truth."""
    estimate, truth = _align(estimate, truth)
    if len(truth) == 0:
        return float("nan")
    estimated_rank = estimate.rank(ascending=False, method="average")
    true_rank = truth.rank(ascending=False, method="average")
    return float((estimated_rank - true_rank).abs().mean())


def exposure_leakage(estimate: pd.Series, truth: pd.Series, exposure: pd.Series) -> float:
    """How much an estimate follows an exposure such as basket size, beyond the true skill.

    It is the coefficient on the exposure when the estimate is regressed on the truth and
    the exposure, all standardised.
    """
    common = truth.index.intersection(estimate.index).intersection(exposure.index)
    frame = pd.DataFrame(
        {
            "estimate": fill_missing(estimate.reindex(common).astype(float)),
            "truth": truth.loc[common].astype(float),
            "exposure": exposure.loc[common].astype(float),
        }
    ).dropna()
    if len(frame) < 5 or frame["exposure"].std() == 0 or frame["estimate"].std() == 0:
        return float("nan")
    z = (frame - frame.mean()) / frame.std(ddof=0)
    design = np.column_stack([np.ones(len(z)), z["truth"], z["exposure"]])
    coefficients, *_ = np.linalg.lstsq(design, z["estimate"].to_numpy(), rcond=None)
    return float(coefficients[2])


def evaluate(estimate: pd.Series, truth: pd.Series, k: int = 5) -> dict[str, float]:
    """All the rank metrics for one set of estimates."""
    return {
        "spearman": spearman(estimate, truth),
        "kendall": kendall(estimate, truth),
        "top_k_precision": top_k_precision(estimate, truth, k),
        "mean_abs_rank_error": mean_absolute_rank_error(estimate, truth),
    }
