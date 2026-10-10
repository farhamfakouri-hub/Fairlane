"""Register profiles and utilisation tables of the simulated store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ree.config import Config, make_rng
from ree.domain.calendar import LANE_GROUPS, MINUTES_PER_DAY, StoreCalendar

MIN_TAGGED_SHARE = 0.005
MAX_TAGGED_SHARE = 0.45


@dataclass(frozen=True)
class RegisterProfiles:
    """Settings of each register: lane group, basket mixture, tagged and cash share, slowness."""

    registers: np.ndarray
    lane_codes: np.ndarray
    mixtures: np.ndarray
    tagged_share: np.ndarray
    cash_share: np.ndarray
    lane_slowness: np.ndarray

    def index_of(self, registers: np.ndarray) -> np.ndarray:
        """Zero-based array positions of register numbers."""
        return np.asarray(registers, dtype=np.int64) - 1


def build_register_profiles(config: Config, calendar: StoreCalendar) -> RegisterProfiles:
    """Build the profile of every register from the configuration.

    The tagged share gets a small random spread from one register to the next.
    """
    store = config.store
    timing = config.simulator.timing
    baskets = config.simulator.baskets
    errors = config.simulator.errors
    rng = make_rng(config.seed, "register_profiles")
    registers = np.arange(1, store.n_registers + 1)
    lane_codes = calendar.lane_group_codes(registers)
    mixtures_by_lane = np.array(
        [
            baskets.mixture_zone_a,
            baskets.mixture_zone_b,
            baskets.mixture_priority,
            baskets.mixture_card_only,
        ]
    )
    tagged_zone_multiplier = np.array(
        [
            baskets.tagged_zone_multiplier_zone_a,
            baskets.tagged_zone_multiplier_zone_b,
            baskets.tagged_zone_multiplier_priority,
            baskets.tagged_zone_multiplier_card_only,
        ]
    )
    sigma = baskets.tagged_register_log_sigma
    register_noise = np.exp(sigma * rng.standard_normal(len(registers)) - sigma**2 / 2)
    tagged_share = np.clip(
        timing.tagged_share * tagged_zone_multiplier[lane_codes] * register_noise,
        MIN_TAGGED_SHARE,
        MAX_TAGGED_SHARE,
    )
    cash_share = np.full(len(registers), errors.cash_share)
    cash_share[registers == store.card_only_register] = 0.0
    lane_slowness = np.ones(len(registers))
    lane_slowness[registers == store.priority_register] = timing.priority_lane_slowness
    return RegisterProfiles(
        registers=registers,
        lane_codes=lane_codes,
        mixtures=mixtures_by_lane[lane_codes],
        tagged_share=tagged_share,
        cash_share=cash_share,
        lane_slowness=lane_slowness,
    )


def utilization_tables(config: Config, calendar: StoreCalendar, days: list[date]) -> np.ndarray:
    """Share of time a customer is waiting, per day, lane group and minute of the day.

    It combines the lane level, rush windows, busy days and seasons, kept within limits.
    """
    traffic = config.simulator.traffic
    lane_utilization = np.array(
        [
            traffic.utilization_zone_a,
            traffic.utilization_zone_b,
            traffic.utilization_priority,
            traffic.utilization_card_only,
        ]
    )
    tables = np.zeros((len(days), len(LANE_GROUPS), MINUTES_PER_DAY), dtype=np.float64)
    for index, day in enumerate(days):
        weekday = day.weekday()
        season_traffic, _ = calendar.season(day)
        day_multiplier = season_traffic * (
            traffic.busy_day_multiplier if calendar.is_busy_day(weekday) else 1.0
        )
        period_multiplier = np.where(
            calendar.peak_table[weekday], traffic.peak_multiplier, traffic.off_peak_multiplier
        )
        raw = lane_utilization[:, None] * period_multiplier[None, :] * day_multiplier
        tables[index] = np.clip(raw, traffic.min_utilization, traffic.max_utilization)
    return tables
