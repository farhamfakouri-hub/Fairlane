"""Transaction features: one row per complete transaction with counts, timings and validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ree.config import Config
from ree.domain.baskets import basket_class_labels
from ree.domain.calendar import StoreCalendar, week_start_monday
from ree.domain.events import (
    EVENT_CODES,
    NON_ERROR_VOID_REASONS,
    event_type_codes,
)
from ree.domain.models import StoreData
from ree.engine.pause_rule import (
    GapTable,
    PauseThresholds,
    extract_gaps,
    thresholds_from_config,
    transaction_timing,
    tune_thresholds,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "session_id",
    "worker_id",
    "register",
    "lane_group",
    "period",
    "started_at",
    "date",
    "week_index",
    "n_items",
    "n_scanned",
    "n_hand_keyed",
    "n_tagged",
    "has_loyalty",
    "n_vouchers",
    "basket_class",
    "size_bin",
    "duration_raw",
    "duration_capped",
    "duration_hard_stop",
    "max_gap",
    "n_gaps",
    "n_voids",
    "n_void_errors",
    "n_void_excluded",
    "n_manual_entries",
    "explicit_interruption",
    "interrupted_by_gap",
    "interrupted",
    "has_exception",
    "valid_for_speed",
    "valid_for_accuracy",
)


@dataclass
class TransactionFeatures:
    """The feature table with its gaps, the pause thresholds used and the first Monday."""

    frame: pd.DataFrame
    gaps: GapTable
    thresholds: PauseThresholds
    data_start: pd.Timestamp

    @property
    def speed_rows(self) -> pd.DataFrame:
        """Rows that are valid for the speed score."""
        return self.frame[self.frame["valid_for_speed"]]


def _count(index: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(index[mask], minlength=size)


def size_bins(n_items: np.ndarray, bin_width: int, max_binned_items: int) -> np.ndarray:
    """Basket-size bin of each item count; counts above the maximum share the last bin."""
    last_bin = (max_binned_items - 1) // bin_width
    return np.minimum((np.asarray(n_items) - 1) // bin_width, last_bin).astype(np.int64)


def data_start_monday(data: StoreData) -> pd.Timestamp:
    """Monday of the first week of data."""
    return pd.Timestamp(week_start_monday(pd.DatetimeIndex([data.first_instant]))[0])


def resolve_thresholds(
    gaps: GapTable, config: Config, override: PauseThresholds | None
) -> PauseThresholds:
    """Choose the pause thresholds: given ones, tuned on the data, or configured."""
    if override is not None:
        return override
    if config.engine.pause.tune_from_data:
        return tune_thresholds(gaps, config.engine.pause)
    return thresholds_from_config(config.engine.pause)


def build_features(
    data: StoreData, config: Config, thresholds: PauseThresholds | None = None
) -> TransactionFeatures:
    """Build the feature table of every complete transaction.

    A transaction is valid for speed when it has an item and a positive capped duration
    and is not interrupted, voided or hand-corrected. All complete transactions count
    for accuracy.
    """
    pause = config.engine.pause
    speed = config.engine.speed
    calendar = StoreCalendar(config.store)

    gaps = extract_gaps(data.events)
    resolved = resolve_thresholds(gaps, config, thresholds)
    timing = transaction_timing(gaps, resolved, pause.hard_stop_seconds)

    events = data.events
    codes = event_type_codes(events["event_type"])
    transaction_ids = events["transaction_id"].to_numpy(dtype=np.int64)
    unique_ids, inverse = np.unique(transaction_ids, return_inverse=True)
    size = len(unique_ids)
    n_scanned = _count(inverse, codes == EVENT_CODES["item_scanned"], size)
    n_hand_keyed = _count(inverse, codes == EVENT_CODES["item_hand_keyed"], size)
    n_tagged = _count(inverse, codes == EVENT_CODES["tag_removed"], size)
    n_vouchers = _count(inverse, codes == EVENT_CODES["voucher_applied"], size)
    n_loyalty = _count(inverse, codes == EVENT_CODES["loyalty_scanned"], size)
    n_manual = _count(inverse, codes == EVENT_CODES["manual_entry"], size)
    n_price_check = _count(inverse, codes == EVENT_CODES["price_check"], size)
    n_supervisor = _count(inverse, codes == EVENT_CODES["supervisor_call"], size)
    n_cheque = _count(inverse, codes == EVENT_CODES["cheque_payment"], size)

    void_mask = codes == EVENT_CODES["void"]
    void_reasons = events["detail"].to_numpy(dtype=object)[void_mask]
    void_excluded_flag = np.isin(void_reasons, NON_ERROR_VOID_REASONS)
    void_index = inverse[void_mask]
    n_voids = np.bincount(void_index, minlength=size)
    n_void_excluded = np.bincount(void_index[void_excluded_flag], minlength=size)

    counts = pd.DataFrame(
        {
            "transaction_id": unique_ids,
            "n_scanned": n_scanned,
            "n_hand_keyed": n_hand_keyed,
            "n_tagged": n_tagged,
            "n_vouchers": n_vouchers,
            "has_loyalty": (n_loyalty > 0).astype(np.int64),
            "n_voids": n_voids,
            "n_void_excluded": n_void_excluded,
            "n_manual_entries": n_manual,
            "explicit_interruption": (n_price_check + n_supervisor + n_cheque) > 0,
        }
    )
    frame = timing[timing["n_gaps"] > 0].merge(counts, on="transaction_id", how="left")

    headers = data.transactions[["transaction_id", "session_id", "register", "started_at"]].merge(
        data.sessions[["session_id", "worker_id"]], on="session_id", how="left"
    )
    frame = frame.merge(headers, on="transaction_id", how="inner")
    frame["started_at"] = pd.to_datetime(frame["started_at"])

    start = data_start_monday(data)
    frame["n_items"] = frame["n_scanned"] + frame["n_hand_keyed"]
    frame["n_void_errors"] = frame["n_voids"] - frame["n_void_excluded"]
    frame["lane_group"] = calendar.lane_groups(frame["register"].to_numpy())
    frame["period"] = calendar.period_labels(frame["started_at"])
    frame["date"] = frame["started_at"].dt.normalize()
    frame["week_index"] = ((frame["date"] - start).dt.days // 7).astype(np.int64)
    frame["basket_class"] = basket_class_labels(frame["n_items"].to_numpy())
    frame["size_bin"] = size_bins(
        frame["n_items"].to_numpy(), speed.bin_width, speed.max_binned_items
    )
    frame["interrupted"] = frame["interrupted_by_gap"] | frame["explicit_interruption"]
    frame["has_exception"] = (frame["n_voids"] + frame["n_manual_entries"]) > 0
    frame["valid_for_accuracy"] = True
    frame["valid_for_speed"] = (
        ~frame["interrupted"]
        & ~frame["has_exception"]
        & (frame["n_items"] >= 1)
        & (frame["duration_capped"] > 0)
    )
    frame = frame[list(FEATURE_COLUMNS)].sort_values("transaction_id").reset_index(drop=True)
    return TransactionFeatures(frame=frame, gaps=gaps, thresholds=resolved, data_start=start)


def weeks_of_data(data: StoreData) -> pd.Series:
    """Number of distinct weeks in which each cashier has a session."""
    start = data_start_monday(data)
    sessions = data.sessions
    week = ((sessions["login_ts"].dt.normalize() - start).dt.days // 7).astype(np.int64)
    return pd.Series(week.to_numpy()).groupby(sessions["worker_id"].to_numpy()).nunique()


def as_of_week_index(data: StoreData) -> int:
    """Index of the as-of week, counted in weeks from the first Monday of the data."""
    start = data_start_monday(data)
    return int((data.as_of.normalize() - start).days // 7)
