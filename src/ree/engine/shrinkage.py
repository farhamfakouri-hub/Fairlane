"""Shrinkage, standardisation and the 50 + 15 z score scale."""

from __future__ import annotations

import numpy as np


def shrink(
    raw: np.ndarray | float,
    n_transactions: np.ndarray | float,
    k: float,
    center: float = 0.0,
) -> np.ndarray | float:
    """Pull a raw value towards a center in proportion to how little data supports it.

    The raw value keeps the weight n / (n + k), where n is the number of transactions.
    """
    weight = np.asarray(n_transactions, dtype=np.float64) / (
        np.asarray(n_transactions, dtype=np.float64) + k
    )
    return center + weight * (np.asarray(raw, dtype=np.float64) - center)


def shrinkage_weight(n_transactions: np.ndarray | float, k: float) -> np.ndarray | float:
    """Weight n / (n + k) that the raw value keeps in shrinkage."""
    counts = np.asarray(n_transactions, dtype=np.float64)
    return counts / (counts + k)


def cohort_statistics(values: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Mean and spread of the finite values inside a reference group.

    Returns 0 and 1 when the group has fewer than two values.
    """
    selected = np.asarray(values, dtype=np.float64)[np.asarray(reference, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if len(selected) < 2:
        return 0.0, 1.0
    spread = float(selected.std())
    return float(selected.mean()), spread if spread > 0 else 1.0


def standard_scores(values: np.ndarray, mean: float, spread: float) -> np.ndarray:
    """Standard scores: values minus the mean, divided by the spread."""
    return (np.asarray(values, dtype=np.float64) - mean) / spread


def to_score_scale(z: np.ndarray | float, center: float = 50.0, scale: float = 15.0) -> np.ndarray:
    """Put standard scores on the reporting scale, 50 + 15 z by default."""
    return center + scale * np.asarray(z, dtype=np.float64)


def log_speed_ratio(expected: np.ndarray | float, observed: np.ndarray | float) -> np.ndarray:
    """Log of expected over observed duration; positive means faster than expected."""
    return np.log(np.asarray(expected, dtype=np.float64) / np.asarray(observed, dtype=np.float64))
