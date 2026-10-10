"""Intraday demand profile: how the customers of a day spread over its slots."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from ree.planner.config import PlannerConfig
from ree.planner.domain.demand_types import PlannerCalendar

REFERENCE_MONDAY = date(2026, 11, 16)


def normal_date(weekday: int) -> date:
    """A fixed reference date for a weekday, used to get its normal profile."""
    return REFERENCE_MONDAY + timedelta(days=weekday)


def slot_weights(calendar: PlannerCalendar, config: PlannerConfig, day: date) -> np.ndarray:
    """Relative weight of each slot of a day: rush slots count more, quiet edges less."""
    starts = calendar.slot_starts(day)
    weights = np.ones(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        if calendar.slot_is_rush(day, start):
            weights[index] = config.demand.rush_intensity
    if len(weights) > 2:
        for edge in (0, len(weights) - 1):
            if not calendar.slot_is_rush(day, starts[edge]):
                weights[edge] *= config.demand.edge_taper
    return weights


def normalise(values: np.ndarray) -> np.ndarray:
    """Scale values so that they sum to one."""
    total = float(values.sum())
    return values / total if total > 0.0 else values


def default_share(calendar: PlannerCalendar, config: PlannerConfig, weekday: int) -> np.ndarray:
    """Default share of the day's customers falling in each slot of a weekday."""
    return normalise(slot_weights(calendar, config, normal_date(weekday)))


def shape_adjustment(calendar: PlannerCalendar, config: PlannerConfig, day: date) -> np.ndarray:
    """Ratio of the slot weights of a day to the normal ones of its weekday."""
    actual = slot_weights(calendar, config, day)
    reference = slot_weights(calendar, config, normal_date(day.weekday()))
    return actual / reference
