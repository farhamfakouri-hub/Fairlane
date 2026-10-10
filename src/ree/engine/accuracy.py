"""Accuracy score: cash discrepancies and void rates, shrunk, standardised and combined."""

from __future__ import annotations

import warnings
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ree.config import AccuracyConfig, CombineConfig, ReviewConfig, make_rng
from ree.engine.shrinkage import cohort_statistics, shrink, standard_scores

ATTRIBUTION_COLUMNS = ["worker_id", "week_index", "delta", "cash"]


@dataclass
class WeeklyRecord:
    """Weekly inputs of one cashier: weights, excess discrepancy, cash, transactions, voids."""

    weight: np.ndarray
    excess: np.ndarray
    cash: np.ndarray
    transactions: np.ndarray
    voids: np.ndarray


@dataclass
class AccuracyParameters:
    """Center, mean and spread of the error, void and manual-entry rates in the rated group."""

    error_center: float
    error_mean: float
    error_spread: float
    void_center: float
    void_mean: float
    void_spread: float
    manual_center: float
    manual_mean: float
    manual_spread: float


def standardize_matrix(
    raw: np.ndarray,
    effective: np.ndarray,
    k: float,
    reference: np.ndarray,
    cap: float,
) -> np.ndarray:
    """Shrink and standardise a matrix of bootstrap rates (draws by cashiers).

    The mean and spread of each draw come from the reference group, and values are capped.
    """
    finite = np.isfinite(raw)
    usable = finite & reference[None, :]
    masked = np.where(usable, raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmean(masked, axis=1, keepdims=True)
    center = np.where(np.isfinite(center), center, 0.0)
    shrunk = np.asarray(
        shrink(np.where(finite, raw, center), np.where(finite, effective, 0.0), k, center)
    )
    shrunk_masked = np.where(usable, shrunk, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(shrunk_masked, axis=1, keepdims=True)
        spread = np.nanstd(shrunk_masked, axis=1, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    return np.clip((shrunk - mean) / spread, -cap, cap)


@dataclass
class AccuracyFit:
    """Result of the accuracy fit: score table, weekly records and bootstrap over weeks."""

    table: pd.DataFrame
    weekly: dict[str, WeeklyRecord]
    parameters: AccuracyParameters
    config: AccuracyConfig
    combine: CombineConfig

    def joint_bootstrap(
        self, worker_ids: list[str], reference: np.ndarray, seed: int, n_bootstrap: int
    ) -> np.ndarray:
        """Bootstrap draws of the accuracy score of all cashiers at once, resampling weeks."""
        columns = len(worker_ids)
        error_raw = np.full((n_bootstrap, columns), np.nan)
        error_weeks = np.zeros((n_bootstrap, columns))
        void_raw = np.full((n_bootstrap, columns), np.nan)
        transactions = np.zeros((n_bootstrap, columns))
        has_record = np.zeros(columns, dtype=bool)
        for column, worker_id in enumerate(worker_ids):
            record = self.weekly.get(worker_id)
            if record is None or len(record.weight) == 0:
                continue
            has_record[column] = True
            rng = make_rng(seed, f"bootstrap:accuracy:{worker_id}")
            weeks = len(record.weight)
            draws = rng.integers(0, weeks, size=(n_bootstrap, weeks))
            weight = record.weight[draws]
            numerator = (weight * record.excess[draws]).sum(axis=1)
            denominator = (weight * record.cash[draws]).sum(axis=1)
            defined = denominator >= self.config.min_cash_euros
            error_raw[:, column] = np.where(
                defined, numerator / np.where(defined, denominator, 1.0), np.nan
            )
            error_weeks[:, column] = np.where(
                defined, (weight * (record.cash[draws] > 0)).sum(axis=1), 0.0
            )
            weighted_transactions = (weight * record.transactions[draws]).sum(axis=1)
            positive = weighted_transactions > 0
            void_raw[:, column] = np.where(
                positive,
                (weight * record.voids[draws]).sum(axis=1)
                / np.where(positive, weighted_transactions, 1.0)
                * 100.0,
                np.nan,
            )
            transactions[:, column] = weighted_transactions
        cap = self.combine.z_cap
        z_error = standardize_matrix(
            error_raw, error_weeks, self.config.shrinkage_k_weeks, reference, cap
        )
        z_void = standardize_matrix(
            void_raw, transactions, self.config.shrinkage_k_transactions, reference, cap
        )
        blended = np.where(
            np.isfinite(error_raw),
            self.config.error_weight * z_error + self.config.void_weight * z_void,
            z_void,
        )
        scores = self.combine.score_center - self.combine.score_scale * blended
        scores[:, ~has_record] = self.combine.score_center
        return scores


def attribute_discrepancies(cash_sheet: pd.DataFrame, data_start: pd.Timestamp) -> pd.DataFrame:
    """Share each drawer's discrepancy among its cashiers by the cash they handled.

    Returns one row per cashier and week. Drawers with an unknown owner are left out.
    """
    if cash_sheet.empty:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in ATTRIBUTION_COLUMNS})
    sheet = cash_sheet.copy()
    sheet["week_index"] = ((pd.to_datetime(sheet["date"]) - data_start).dt.days // 7).astype(
        np.int64
    )
    unclear = sheet["worker_id"].isna().groupby(sheet["drawer_id"]).transform("max")
    known = sheet[~unclear.astype(bool)].copy()
    if known.empty:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in ATTRIBUTION_COLUMNS})
    drawer_cash = known.groupby("drawer_id")["cash_handled"].transform("sum")
    drawer_size = known.groupby("drawer_id")["cash_handled"].transform("size")
    share = np.where(
        drawer_cash > 0,
        known["cash_handled"] / drawer_cash.where(drawer_cash > 0, 1.0),
        1.0 / drawer_size,
    )
    drawer_discrepancy = known.groupby("drawer_id")["till_discrepancy"].transform("first")
    known["attributed"] = drawer_discrepancy * share
    return known.groupby(["worker_id", "week_index"], as_index=False).agg(
        delta=("attributed", "sum"), cash=("cash_handled", "sum")
    )


def weekly_records(
    features: pd.DataFrame,
    attributed: pd.DataFrame,
    as_of_week: int,
    config: AccuracyConfig,
) -> pd.DataFrame:
    """Weekly activity, cash and discrepancy of each cashier inside the scoring window.

    Each week gets a weight that halves with age and an excess over the tolerance.
    """
    accuracy_rows = features[features["valid_for_accuracy"]]
    activity = (
        accuracy_rows.groupby(["worker_id", "week_index"])
        .agg(
            transactions=("transaction_id", "size"),
            voids=("n_void_errors", "sum"),
            manual=("n_manual_entries", "sum"),
        )
        .reset_index()
    )
    weekly = activity.merge(attributed, on=["worker_id", "week_index"], how="outer")
    for column in ("transactions", "voids", "manual", "delta", "cash"):
        weekly[column] = weekly[column].fillna(0.0)
    weekly["weeks_ago"] = as_of_week - weekly["week_index"]
    weekly = weekly[(weekly["weeks_ago"] >= 0) & (weekly["weeks_ago"] < config.window_weeks)].copy()
    weekly["weight"] = 0.5 ** (weekly["weeks_ago"] / config.half_life_weeks)
    weekly["excess"] = np.maximum(0.0, weekly["delta"].abs() - config.tolerance_euros)
    return weekly.sort_values(["worker_id", "week_index"]).reset_index(drop=True)


def accuracy_scores(
    numerator: np.ndarray,
    denominator: np.ndarray,
    effective_weeks: np.ndarray,
    void_total: np.ndarray,
    effective_transactions: np.ndarray,
    parameters: AccuracyParameters,
    config: AccuracyConfig,
    combine: CombineConfig,
) -> dict[str, np.ndarray]:
    """Accuracy scores from weighted totals.

    Error and void rates are shrunk, standardised and blended; a higher score is better.
    """
    defined = denominator >= config.min_cash_euros
    safe_denominator = np.where(defined, denominator, 1.0)
    error_raw = np.where(defined, numerator / safe_denominator, np.nan)
    has_transactions = effective_transactions > 0
    void_raw = np.where(
        has_transactions,
        void_total / np.where(has_transactions, effective_transactions, 1.0) * 100.0,
        np.nan,
    )
    error_shrunk = shrink(
        np.where(defined, error_raw, parameters.error_center),
        np.where(defined, effective_weeks, 0.0),
        config.shrinkage_k_weeks,
        parameters.error_center,
    )
    void_shrunk = shrink(
        np.where(has_transactions, void_raw, parameters.void_center),
        effective_transactions,
        config.shrinkage_k_transactions,
        parameters.void_center,
    )
    cap = combine.z_cap
    z_error = np.clip(
        standard_scores(error_shrunk, parameters.error_mean, parameters.error_spread), -cap, cap
    )
    z_void = np.clip(
        standard_scores(void_shrunk, parameters.void_mean, parameters.void_spread), -cap, cap
    )
    blended = np.where(
        defined,
        config.error_weight * z_error + config.void_weight * z_void,
        z_void,
    )
    return {
        "error_rate_raw": error_raw,
        "error_rate_shrunk": np.asarray(error_shrunk, dtype=np.float64),
        "void_rate_raw": void_raw,
        "void_rate_shrunk": np.asarray(void_shrunk, dtype=np.float64),
        "z_error": z_error,
        "z_void": z_void,
        "error_defined": defined,
        "accuracy_score": combine.score_center - combine.score_scale * blended,
    }


def _cohort(
    raw: np.ndarray, effective: np.ndarray, k: float, reference: np.ndarray
) -> tuple[float, float, float]:
    usable = reference & np.isfinite(raw)
    center = float(np.mean(raw[usable])) if usable.any() else 0.0
    shrunk = shrink(
        np.where(np.isfinite(raw), raw, center),
        np.where(np.isfinite(raw), effective, 0.0),
        k,
        center,
    )
    mean, spread = cohort_statistics(np.asarray(shrunk), usable)
    return center, mean, spread


def fit_accuracy(
    features: pd.DataFrame,
    cash_sheet: pd.DataFrame,
    data_start: pd.Timestamp,
    as_of_week: int,
    config: AccuracyConfig,
    combine: CombineConfig,
    rated: Collection[str] | None = None,
) -> AccuracyFit:
    """Fit the accuracy score from the features and the cash sheet as of a given week."""
    attributed = attribute_discrepancies(cash_sheet, data_start)
    weekly = weekly_records(features, attributed, as_of_week, config)
    weekly["weighted_excess"] = weekly["weight"] * weekly["excess"]
    weekly["weighted_cash"] = weekly["weight"] * weekly["cash"]
    weekly["weighted_cash_weeks"] = weekly["weight"] * (weekly["cash"] > 0)
    weekly["weighted_transactions"] = weekly["weight"] * weekly["transactions"]
    weekly["weighted_voids"] = weekly["weight"] * weekly["voids"]
    weekly["weighted_manual"] = weekly["weight"] * weekly["manual"]
    totals = weekly.groupby("worker_id").agg(
        numerator=("weighted_excess", "sum"),
        denominator=("weighted_cash", "sum"),
        effective_weeks=("weighted_cash_weeks", "sum"),
        effective_transactions=("weighted_transactions", "sum"),
        void_total=("weighted_voids", "sum"),
        manual_total=("weighted_manual", "sum"),
        weeks_with_data=("week_index", "nunique"),
    )
    totals["delta_sum"] = weekly.groupby("worker_id")["delta"].sum()
    workers = totals.index.to_numpy()
    reference = (
        np.isin(workers, list(rated)) if rated is not None else np.ones(len(workers), dtype=bool)
    )

    denominator = totals["denominator"].to_numpy()
    defined = denominator >= config.min_cash_euros
    error_raw = np.where(
        defined, totals["numerator"].to_numpy() / np.where(defined, denominator, 1.0), np.nan
    )
    transactions = totals["effective_transactions"].to_numpy()
    has_transactions = transactions > 0
    safe_transactions = np.where(has_transactions, transactions, 1.0)
    void_raw = np.where(
        has_transactions, totals["void_total"].to_numpy() / safe_transactions * 100.0, np.nan
    )
    manual_raw = np.where(
        has_transactions, totals["manual_total"].to_numpy() / safe_transactions * 100.0, np.nan
    )

    error_center, error_mean, error_spread = _cohort(
        error_raw, totals["effective_weeks"].to_numpy(), config.shrinkage_k_weeks, reference
    )
    void_center, void_mean, void_spread = _cohort(
        void_raw, transactions, config.shrinkage_k_transactions, reference
    )
    manual_center, manual_mean, manual_spread = _cohort(
        manual_raw, transactions, config.shrinkage_k_transactions, reference
    )
    parameters = AccuracyParameters(
        error_center=error_center,
        error_mean=error_mean,
        error_spread=error_spread,
        void_center=void_center,
        void_mean=void_mean,
        void_spread=void_spread,
        manual_center=manual_center,
        manual_mean=manual_mean,
        manual_spread=manual_spread,
    )
    scored = accuracy_scores(
        totals["numerator"].to_numpy(),
        denominator,
        totals["effective_weeks"].to_numpy(),
        totals["void_total"].to_numpy(),
        transactions,
        parameters,
        config,
        combine,
    )
    manual_shrunk = shrink(
        np.where(has_transactions, manual_raw, manual_center),
        transactions,
        config.shrinkage_k_transactions,
        manual_center,
    )
    table = pd.DataFrame(
        {
            "worker_id": workers,
            "weeks_with_data": totals["weeks_with_data"].to_numpy(),
            "effective_weeks": totals["effective_weeks"].to_numpy(),
            "effective_transactions": transactions,
            "cash_weighted": denominator,
            "net_discrepancy": totals["delta_sum"].to_numpy(),
            "manual_rate_raw": manual_raw,
            "manual_rate_shrunk": np.asarray(manual_shrunk, dtype=np.float64),
            "z_manual": standard_scores(np.asarray(manual_shrunk), manual_mean, manual_spread),
            **scored,
        }
    ).set_index("worker_id")

    records = {
        worker_id: WeeklyRecord(
            weight=group["weight"].to_numpy(),
            excess=group["excess"].to_numpy(),
            cash=group["cash"].to_numpy(),
            transactions=group["transactions"].to_numpy(),
            voids=group["voids"].to_numpy(),
        )
        for worker_id, group in weekly.groupby("worker_id")
    }
    return AccuracyFit(
        table=table, weekly=records, parameters=parameters, config=config, combine=combine
    )


def review_flags(
    speed_z: pd.Series,
    void_z: pd.Series,
    manual_z: pd.Series,
    config: ReviewConfig,
) -> pd.Series:
    """Flag cashiers whose speed is very high together with a high void or manual-entry rate.

    Returns the reason for each flagged cashier and None for the others.
    """
    frame = pd.DataFrame({"speed_z": speed_z, "void_z": void_z, "manual_z": manual_z})
    reasons: list[str | None] = []
    for row in frame.itertuples():
        speed_high = row.speed_z >= config.speed_z_threshold
        void_high = row.void_z >= config.rate_z_threshold
        manual_high = row.manual_z >= config.rate_z_threshold
        if speed_high and (void_high or manual_high):
            signals = []
            if void_high:
                signals.append(f"void rate (z = {row.void_z:+.1f})")
            if manual_high:
                signals.append(f"manual-entry rate (z = {row.manual_z:+.1f})")
            reasons.append(
                f"speed is very high (z = {row.speed_z:+.1f}) together with a high "
                f"{' and '.join(signals)}: worth a human look for skipped scans"
            )
        else:
            reasons.append(None)
    return pd.Series(reasons, index=frame.index, dtype="object")
