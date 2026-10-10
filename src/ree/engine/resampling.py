"""Resampling helpers for the bootstrap over sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_counts(rng: np.random.Generator, n_sessions: int, n_bootstrap: int) -> np.ndarray:
    """How often each session is drawn when sessions are resampled with replacement."""
    return rng.multinomial(n_sessions, np.full(n_sessions, 1.0 / n_sessions), size=n_bootstrap)


def weighted_median_rows(sorted_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted median of sorted values, one for each row of weights."""
    cumulative = np.cumsum(weights, axis=1)
    total = cumulative[:, -1]
    half = 0.5 * total[:, None]
    lower = (cumulative >= half).argmax(axis=1)
    upper = (cumulative > half).argmax(axis=1)
    result = 0.5 * (sorted_values[lower] + sorted_values[upper])
    return np.where(total > 0, result, np.nan)


def session_codes(
    workers: np.ndarray, periods: np.ndarray, sessions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Number the sessions inside each cashier and period, and count them."""
    frame = pd.DataFrame({"worker": workers, "period": periods, "session": sessions})
    grouped = frame.groupby(["worker", "period"], sort=False)["session"]
    local = grouped.transform(lambda values: pd.factorize(values)[0]).to_numpy(dtype=np.int64)
    total = grouped.transform("nunique").to_numpy(dtype=np.int64)
    return local, total
