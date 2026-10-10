"""Plain-English reasons behind a score."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ree.config import AccuracyConfig, CombineConfig
from ree.domain.baskets import BASKET_CLASSES
from ree.domain.models import Context
from ree.engine.accuracy import AccuracyParameters
from ree.engine.combine import accuracy_weight, speed_weight

MIN_POINTS = 1.0
MAX_DRIVERS = 4
BASKET_LABELS = {"small": "small baskets", "medium": "medium baskets", "big": "big baskets"}
CONTEXTS: tuple[Context, ...] = ("peak", "off")
NEUTRAL_DRIVER = "close to the store average on every measured aspect"


@dataclass(frozen=True)
class DriverInputs:
    """Everything needed to explain the scores of a run."""

    row_scores: pd.DataFrame
    speed_spread: dict[str, float]
    accuracy_table: pd.DataFrame
    accuracy_parameters: AccuracyParameters
    accuracy_config: AccuracyConfig
    combine: CombineConfig
    provisional: pd.Series
    weeks_of_data: pd.Series
    n_valid: pd.DataFrame
    min_weeks: int
    min_transactions: int


def _speed_points(worker_id: str, inputs: DriverInputs) -> dict[str, float]:
    rows = inputs.row_scores[inputs.row_scores["worker_id"] == worker_id].dropna(
        subset=["log_ratio"]
    )
    totals = rows.groupby("period").size()
    grand_total = float(totals.sum())
    points = {name: 0.0 for name in BASKET_CLASSES}
    if grand_total == 0:
        return points
    stats = rows.groupby(["period", "basket_class"])["log_ratio"].agg(["median", "size"])
    for (period, basket_class), row in stats.iterrows():
        omega = speed_weight(inputs.combine, period)
        spread = inputs.speed_spread[period]
        period_share = float(totals[period]) / grand_total
        class_share = float(row["size"]) / float(totals[period])
        points[basket_class] += (
            period_share
            * omega
            * inputs.combine.score_scale
            * class_share
            * float(row["median"])
            / spread
        )
    return points


def _accuracy_points(worker_id: str, inputs: DriverInputs) -> list[tuple[float, str]]:
    if worker_id not in inputs.accuracy_table.index:
        return []
    row = inputs.accuracy_table.loc[worker_id]
    omega = sum(accuracy_weight(inputs.combine, context) for context in CONTEXTS) / len(CONTEXTS)
    error_weight = inputs.accuracy_config.error_weight if row["error_defined"] else 0.0
    void_weight = 1.0 - error_weight
    scale = inputs.combine.score_scale
    candidates: list[tuple[float, str]] = []
    if row["error_defined"]:
        points = -omega * scale * error_weight * float(row["z_error"])
        text = (
            "cash discrepancies above the store norm"
            if points < 0
            else "cash discrepancies below the store norm"
        )
        candidates.append((points, text))
    points = -omega * scale * void_weight * float(row["z_void"])
    rate = float(row["void_rate_raw"]) if pd.notna(row["void_rate_raw"]) else 0.0
    center = inputs.accuracy_parameters.void_center
    quality = "high" if points < 0 else "low"
    candidates.append(
        (
            points,
            f"{quality} void rate ({rate:.1f} per 100 transactions, store average {center:.1f})",
        )
    )
    return candidates


def _caveats(worker_id: str, inputs: DriverInputs) -> list[str]:
    messages: list[str] = []
    weeks = int(inputs.weeks_of_data.get(worker_id, 0))
    if bool(inputs.provisional.get(worker_id, False)):
        messages.append(f"provisional: {weeks} weeks of data, learning curve still in progress")
    smallest = int(inputs.n_valid.loc[worker_id].min()) if worker_id in inputs.n_valid.index else 0
    if weeks < inputs.min_weeks or smallest < inputs.min_transactions:
        messages.append(f"limited data: {smallest} timed transactions in the thinner period")
    return messages


def build_drivers(worker_ids: list[str], inputs: DriverInputs) -> dict[str, list[str]]:
    """Write the reasons for each cashier's score.

    Caveats about limited data come first. Then come the largest contributions, from speed
    by basket class, cash discrepancies and void rate, up to four reasons in all.
    """
    drivers: dict[str, list[str]] = {}
    for worker_id in worker_ids:
        candidates: list[tuple[float, str]] = []
        for basket_class, points in _speed_points(worker_id, inputs).items():
            label = BASKET_LABELS[basket_class]
            text = (
                f"faster than the store median on {label}"
                if points > 0
                else f"slower than the store median on {label}"
            )
            candidates.append((points, text))
        candidates.extend(_accuracy_points(worker_id, inputs))
        significant = sorted(
            (item for item in candidates if abs(item[0]) >= MIN_POINTS),
            key=lambda item: -abs(item[0]),
        )
        messages = _caveats(worker_id, inputs)
        room = max(MAX_DRIVERS - len(messages), 0)
        messages.extend(text for _, text in significant[:room])
        drivers[worker_id] = messages if messages else [NEUTRAL_DRIVER]
    return drivers
