"""Speed score, method v2: duration regressed on basket content, then a median log ratio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ree.config import SpeedConfig
from ree.engine.resampling import session_codes, session_counts, weighted_median_rows

MIN_EXPECTED_SECONDS = 1.0
MIN_DURATION_SECONDS = 0.5
MIN_ROWS_FOR_MAD = 10
MAD_TO_SIGMA = 1.4826
DESIGN_COLUMNS: tuple[str, ...] = (
    "n_items",
    "n_hand_keyed",
    "n_tagged",
    "has_loyalty",
    "n_vouchers",
)
PRICE_TERMS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "lambda", "mu")
Z_95 = 1.96


@dataclass
class PriceFit:
    """Result of a regression: coefficients, standard errors and residuals."""

    theta: np.ndarray
    std_errors: np.ndarray
    residuals: np.ndarray


@dataclass
class V2Fit:
    """Result of the v2 fit: scores, per-transaction scores, price list and bootstrap data."""

    scores: pd.DataFrame
    row_scores: pd.DataFrame
    price_list: pd.DataFrame
    method: str = "v2"
    _values: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = field(
        default_factory=dict, repr=False
    )

    def worker_scores(self) -> pd.DataFrame:
        """Raw score and number of valid transactions of each cashier and period."""
        return self.scores

    def bootstrap(
        self, worker_id: str, period: str, rng: np.random.Generator, n_bootstrap: int
    ) -> np.ndarray:
        """Bootstrap draws of the raw score of one cashier and period, resampling sessions."""
        entry = self._values.get((worker_id, period))
        if entry is None:
            return np.zeros(n_bootstrap)
        values, sessions, n_sessions = entry
        counts = session_counts(rng, n_sessions, n_bootstrap)
        return weighted_median_rows(values, counts[:, sessions])


def fit_linear(
    design: np.ndarray, durations: np.ndarray, estimator: Literal["ols", "huber"] = "ols"
) -> PriceFit:
    """Regress duration on the design matrix, by least squares or with a robust Huber fit."""
    response = np.asarray(durations, dtype=np.float64)
    matrix = np.asarray(design, dtype=np.float64)
    if estimator == "huber":
        result = sm.RLM(response, matrix, M=sm.robust.norms.HuberT()).fit()
    else:
        result = sm.OLS(response, matrix).fit()
    return PriceFit(
        theta=np.asarray(result.params),
        std_errors=np.asarray(result.bse),
        residuals=response - matrix @ np.asarray(result.params),
    )


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Design matrix: a constant plus items, hand-keyed items, tagged items, loyalty card
    and vouchers.
    """
    columns = [np.ones(len(frame))]
    columns.extend(frame[name].to_numpy(dtype=np.float64) for name in DESIGN_COLUMNS)
    return np.column_stack(columns)


def mad_keep_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Mask of values within a number of robust standard deviations of the median."""
    median = np.median(values)
    spread = MAD_TO_SIGMA * np.median(np.abs(values - median))
    if spread <= 0:
        return np.ones(len(values), dtype=bool)
    return np.abs(values - median) <= threshold * spread


def grouped_mad_keep_mask(
    values: np.ndarray, workers: np.ndarray, periods: np.ndarray, threshold: float
) -> np.ndarray:
    """The same mask for each cashier and period; groups that are too small keep every value."""
    frame = pd.DataFrame({"value": values, "worker": workers, "period": periods})
    keys = [frame["worker"], frame["period"]]
    median = frame["value"].groupby(keys).transform("median")
    deviation = (frame["value"] - median).abs()
    spread = MAD_TO_SIGMA * deviation.groupby(keys).transform("median")
    size = frame["value"].groupby(keys).transform("size")
    keep = (deviation <= threshold * spread) | (spread <= 0) | (size < MIN_ROWS_FOR_MAD)
    return keep.to_numpy(dtype=bool)


def fit_v2(frame: pd.DataFrame, durations: np.ndarray, config: SpeedConfig) -> V2Fit:
    """Fit the v2 speed score.

    A price list is estimated for each lane group and period. Each transaction's log ratio
    of expected to actual duration is centered on its median, extreme values are dropped,
    and a cashier's score is the median of what remains.
    """
    work = frame[["worker_id", "period", "lane_group", "basket_class", *DESIGN_COLUMNS]].copy()
    work["duration"] = np.maximum(np.asarray(durations, dtype=np.float64), MIN_DURATION_SECONDS)
    log_ratio = np.full(len(work), np.nan)
    price_rows: list[dict[str, object]] = []
    positions = np.arange(len(work))

    for (lane_group, period), context in work.groupby(["lane_group", "period"], sort=True):
        context_positions = positions[work.index.get_indexer(context.index)]
        fit = fit_linear(design_matrix(context), context["duration"].to_numpy(), config.regression)
        expected = np.maximum(design_matrix(context) @ fit.theta, MIN_EXPECTED_SECONDS)
        ratio = np.log(expected / context["duration"].to_numpy())
        log_ratio[context_positions] = ratio - float(np.median(ratio))
        for term, estimate, error in zip(PRICE_TERMS, fit.theta, fit.std_errors, strict=True):
            price_rows.append(
                {
                    "lane_group": lane_group,
                    "period": period,
                    "term": term,
                    "estimate": float(estimate),
                    "std_error": float(error),
                    "n_rows": len(context),
                }
            )

    kept = grouped_mad_keep_mask(
        log_ratio,
        work["worker_id"].to_numpy(),
        work["period"].to_numpy(),
        config.mad_threshold,
    )
    row_scores = pd.DataFrame(
        {
            "worker_id": work["worker_id"].to_numpy(),
            "period": work["period"].to_numpy(),
            "basket_class": work["basket_class"].to_numpy(),
            "log_ratio": np.where(kept, log_ratio, np.nan),
        },
        index=work.index,
    )
    used = row_scores.dropna(subset=["log_ratio"])
    grouped = used.groupby(["worker_id", "period"])["log_ratio"]
    scores = grouped.agg(raw_score="median", n_valid="size").reset_index()
    sessions = (
        frame["session_id"].to_numpy() if "session_id" in frame.columns else np.arange(len(frame))
    )
    used_sessions = pd.Series(sessions, index=work.index).loc[used.index].to_numpy()
    local, totals = session_codes(
        used["worker_id"].to_numpy(), used["period"].to_numpy(), used_sessions
    )
    values: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    positions = pd.Series(
        np.arange(len(used)), index=pd.MultiIndex.from_frame(used[["worker_id", "period"]])
    )
    for key, rows in positions.groupby(level=[0, 1]):
        index = rows.to_numpy()
        order = np.argsort(used["log_ratio"].to_numpy()[index], kind="stable")
        ordered = index[order]
        values[key] = (
            used["log_ratio"].to_numpy(dtype=np.float64)[ordered],
            local[ordered],
            int(totals[ordered[0]]),
        )
    return V2Fit(
        scores=scores,
        row_scores=row_scores,
        price_list=pd.DataFrame(price_rows),
        _values=values,
    )


def sanity_warnings(price_list: pd.DataFrame, config: SpeedConfig) -> list[str]:
    """Warn when the estimated cost of removing a tag or hand keying is out of range."""
    messages: list[str] = []
    low, high = config.delta_expected_range
    for row in price_list.itertuples(index=False):
        margin = Z_95 * row.std_error
        if row.term == "delta" and (row.estimate + margin < low or row.estimate - margin > high):
            messages.append(
                f"price list {row.lane_group}/{row.period}: tag removal cost delta = "
                f"{row.estimate:.2f} s is outside the expected {low:.0f} to {high:.0f} s"
            )
        if row.term == "gamma" and row.estimate - margin >= config.gamma_expected_max:
            messages.append(
                f"price list {row.lane_group}/{row.period}: hand-keyed cost gamma = "
                f"{row.estimate:.2f} s is above the expected {config.gamma_expected_max:.0f} s"
            )
    return messages
