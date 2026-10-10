"""Pause rule: reads the gaps between register events and decides which time counts as work."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ree.config import PauseRuleConfig
from ree.domain.events import (
    EVENT_CODES,
    GAP_EVENT_TYPES,
    ITEM_EVENT_TYPES,
    event_type_codes,
)

PAYMENT_CODE = EVENT_CODES["payment_pressed"]
ITEM_CODES = np.array([EVENT_CODES[name] for name in ITEM_EVENT_TYPES])
WINDOW_CODES = np.array([*(EVENT_CODES[name] for name in GAP_EVENT_TYPES), PAYMENT_CODE])


@dataclass(frozen=True)
class PauseThresholds:
    """Gap limits in seconds.

    A gap longer than the idle limit is cut to it; a gap longer than the stop limit
    marks the transaction as interrupted.
    """

    t_idle_seconds: float
    t_stop_seconds: float


@dataclass
class GapTable:
    """Gaps between consecutive events, from the first item to payment, of complete transactions."""

    transaction_ids: np.ndarray
    transaction_index: np.ndarray
    gap_seconds: np.ndarray
    closing_event_code: np.ndarray

    @property
    def n_transactions(self) -> int:
        """Number of transactions in the table."""
        return len(self.transaction_ids)


def thresholds_from_config(config: PauseRuleConfig) -> PauseThresholds:
    """Pause thresholds as set in the configuration."""
    return PauseThresholds(config.t_idle_seconds, config.t_stop_seconds)


def capped_duration(gaps: Sequence[float], t_idle_seconds: float) -> float:
    """Sum of the gaps, each cut to the idle limit."""
    return float(sum(min(gap, t_idle_seconds) for gap in gaps))


def is_interrupted(gaps: Sequence[float], t_stop_seconds: float) -> bool:
    """Whether any gap is longer than the stop limit."""
    return any(gap > t_stop_seconds for gap in gaps)


def hard_stop_duration(gaps: Sequence[float], hard_stop_seconds: float) -> float:
    """Sum of the gaps up to a hard-stop limit; longer gaps are left out."""
    return float(sum(gap for gap in gaps if gap <= hard_stop_seconds))


def extract_gaps(events: pd.DataFrame) -> GapTable:
    """Find the gaps of every complete transaction.

    A transaction is complete when it has an item event and a payment. Gaps are measured
    between events from the first item to the payment.
    """
    codes = event_type_codes(events["event_type"])
    transaction = events["transaction_id"].to_numpy(dtype=np.int64)
    offsets = events["offset_s"].to_numpy(dtype=np.float64)
    order = np.lexsort((offsets, transaction))
    codes = codes[order]
    transaction = transaction[order]
    offsets = offsets[order]

    transaction_ids, transaction_index = np.unique(transaction, return_inverse=True)
    count = len(transaction_ids)
    item_mask = np.isin(codes, ITEM_CODES)
    first_item = np.full(count, np.inf)
    np.minimum.at(first_item, transaction_index[item_mask], offsets[item_mask])
    has_payment = np.zeros(count, dtype=bool)
    has_payment[transaction_index[codes == PAYMENT_CODE]] = True
    complete = has_payment & np.isfinite(first_item)

    in_window = np.isin(codes, WINDOW_CODES) & (offsets >= first_item[transaction_index])
    window_transaction = transaction_index[in_window]
    window_offsets = offsets[in_window]
    window_codes = codes[in_window]
    same_transaction = window_transaction[1:] == window_transaction[:-1]
    gaps = (window_offsets[1:] - window_offsets[:-1])[same_transaction]
    gap_transaction = window_transaction[1:][same_transaction]
    closing_codes = window_codes[1:][same_transaction]
    keep = complete[gap_transaction]
    return GapTable(
        transaction_ids=transaction_ids,
        transaction_index=gap_transaction[keep],
        gap_seconds=gaps[keep],
        closing_event_code=closing_codes[keep],
    )


def max_gap_per_transaction(gap_table: GapTable) -> np.ndarray:
    """Longest gap of each transaction, zero for none."""
    maxima = np.zeros(gap_table.n_transactions)
    np.maximum.at(maxima, gap_table.transaction_index, gap_table.gap_seconds)
    return maxima


def transaction_timing(
    gap_table: GapTable, thresholds: PauseThresholds, hard_stop_seconds: float
) -> pd.DataFrame:
    """Duration measures of each transaction: raw, capped and hard-stop totals,
    number of gaps, longest gap and whether a gap interrupted it.
    """
    count = gap_table.n_transactions
    index = gap_table.transaction_index
    gaps = gap_table.gap_seconds
    max_gap = max_gap_per_transaction(gap_table)
    return pd.DataFrame(
        {
            "transaction_id": gap_table.transaction_ids,
            "n_gaps": np.bincount(index, minlength=count),
            "duration_raw": np.bincount(index, weights=gaps, minlength=count),
            "duration_capped": np.bincount(
                index, weights=np.minimum(gaps, thresholds.t_idle_seconds), minlength=count
            ),
            "duration_hard_stop": np.bincount(
                index, weights=np.where(gaps <= hard_stop_seconds, gaps, 0.0), minlength=count
            ),
            "max_gap": max_gap,
            "interrupted_by_gap": max_gap > thresholds.t_stop_seconds,
        }
    )


def tune_thresholds(gap_table: GapTable, config: PauseRuleConfig) -> PauseThresholds:
    """Choose the thresholds from the data.

    The idle limit is a quantile of the gaps in transactions without a long pause. The
    stop limit is the configured one or a multiple of the idle limit, whichever is larger.
    """
    max_gap = max_gap_per_transaction(gap_table)
    typical = max_gap[gap_table.transaction_index] <= config.t_stop_seconds
    typical_gaps = gap_table.gap_seconds[typical]
    if len(typical_gaps) == 0:
        return thresholds_from_config(config)
    t_idle = float(np.quantile(typical_gaps, config.tune_quantile))
    t_stop = max(config.t_stop_seconds, config.tune_stop_multiplier * t_idle)
    return PauseThresholds(t_idle_seconds=t_idle, t_stop_seconds=t_stop)
