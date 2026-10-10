"""Tiers A, B and C from scores and intervals, and the confidence taken from interval width."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ree.config import TierConfig

UNRATED = "unrated"


def interval_confidence(width: np.ndarray | float, width_limit: float) -> np.ndarray | float:
    """Confidence between 0 and 1: one minus the interval width over its limit."""
    return np.clip(1.0 - np.asarray(width, dtype=np.float64) / width_limit, 0.0, 1.0)


def _group_tiers(group: pd.DataFrame, config: TierConfig) -> pd.Series:
    percentile = group["score"].rank(method="average", pct=True)
    tiers = pd.Series("B", index=group.index, dtype="object")
    top = percentile > 1.0 - config.top_share
    bottom = percentile <= config.bottom_share
    tiers[top] = "A"
    tiers[bottom] = "C"
    if not config.conservative:
        return tiers
    middle = group.loc[~top & ~bottom, "score"]
    if top.any() and len(middle):
        cut_high = 0.5 * (group.loc[top, "score"].min() + middle.max())
        tiers[top & (group["lower"] < cut_high)] = "B"
    if bottom.any() and len(middle):
        cut_low = 0.5 * (group.loc[bottom, "score"].max() + middle.min())
        tiers[bottom & (group["upper"] > cut_low)] = "B"
    return tiers


def assign_tiers(
    scores: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    provisional: pd.Series,
    weeks_of_data: pd.Series,
    config: TierConfig,
    width_limit: float,
) -> pd.Series:
    """Assign tier A, B, C or unrated to each cashier.

    Only cashiers with a narrow enough interval and enough weeks of data are ranked, in
    separate groups for established and provisional cashiers. Groups that are too small
    stay unrated.
    """
    frame = pd.DataFrame(
        {
            "score": scores,
            "lower": lower,
            "upper": upper,
            "eligible": ((upper - lower) <= width_limit)
            & (weeks_of_data >= config.min_weeks_of_data),
            "provisional": provisional,
        }
    )
    tiers = pd.Series(UNRATED, index=frame.index, dtype="object")
    for flag in (False, True):
        group = frame[(frame["provisional"] == flag) & frame["eligible"]]
        if len(group) < config.min_group_size:
            continue
        tiers.loc[group.index] = _group_tiers(group, config)
    return tiers
