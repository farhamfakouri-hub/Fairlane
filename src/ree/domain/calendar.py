"""Store calendar: opening hours, rush windows, lane groups, seasons and week starts."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ree.config import WEEKDAY_INDEX, StoreConfig, clock_to_minutes

LANE_GROUPS: tuple[str, ...] = ("zone_a", "zone_b", "priority", "card_only")
PERIODS: tuple[str, ...] = ("peak", "off")
MINUTES_PER_DAY = 1440


class StoreCalendar:
    """Answers calendar questions about the simulated store from its configuration."""

    def __init__(self, store: StoreConfig) -> None:
        self.store = store
        self.opening_minutes = np.zeros((7, 2), dtype=np.int64)
        for weekday in range(7):
            hours = store.opening.sunday if weekday == 6 else store.opening.monday_to_saturday
            self.opening_minutes[weekday] = (
                clock_to_minutes(hours[0]),
                clock_to_minutes(hours[1]),
            )
        self.peak_table = np.zeros((7, MINUTES_PER_DAY), dtype=bool)
        for window in store.rush_windows:
            start = clock_to_minutes(window.start)
            end = clock_to_minutes(window.end)
            for day in window.days:
                self.peak_table[WEEKDAY_INDEX[day], start:end] = True
        self.busy_weekdays = frozenset(WEEKDAY_INDEX[day] for day in store.busy_days)

    def open_close_minutes(self, weekday: int) -> tuple[int, int]:
        """Opening and closing time of a weekday, in minutes after midnight."""
        opening, closing = self.opening_minutes[weekday]
        return int(opening), int(closing)

    def is_busy_day(self, weekday: int) -> bool:
        """Whether the weekday is one of the store's busy days."""
        return weekday in self.busy_weekdays

    def lane_group(self, register: int) -> str:
        """Lane group of a register: zone_a, zone_b, priority or card_only."""
        if register == self.store.priority_register:
            return "priority"
        if register == self.store.card_only_register:
            return "card_only"
        if register <= self.store.busy_zone_last_register:
            return "zone_a"
        return "zone_b"

    def lane_groups(self, registers: np.ndarray) -> np.ndarray:
        """Lane group name of each register in an array."""
        registers = np.asarray(registers)
        groups = np.full(registers.shape, "zone_b", dtype=object)
        groups[registers <= self.store.busy_zone_last_register] = "zone_a"
        groups[registers == self.store.card_only_register] = "card_only"
        groups[registers == self.store.priority_register] = "priority"
        return groups

    def lane_group_codes(self, registers: np.ndarray) -> np.ndarray:
        """Integer code of the lane group of each register in an array."""
        names = self.lane_groups(registers)
        return np.array([LANE_GROUPS.index(name) for name in names], dtype=np.int64)

    def peak_flags(self, instants: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """Whether each instant falls inside a rush window."""
        index = pd.DatetimeIndex(instants)
        weekday = np.asarray(index.weekday)
        minute = np.asarray(index.hour) * 60 + np.asarray(index.minute)
        return self.peak_table[weekday, minute]

    def period_labels(self, instants: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """Label each instant "peak" or "off"."""
        return np.where(self.peak_flags(instants), "peak", "off")

    def season(self, day: date) -> tuple[float, float]:
        """Traffic and basket multipliers that apply on a date."""
        traffic = 1.0
        basket = 1.0
        marker = (day.month, day.day)
        for window in self.store.seasonal:
            start = _month_day(window.start)
            end = _month_day(window.end)
            inside = start <= marker <= end if start <= end else marker >= start or marker <= end
            if inside:
                traffic *= window.traffic_multiplier
                basket *= window.basket_multiplier
        return traffic, basket

    def minute_peak_hours(self, weekday: int, login_minute: int, logout_minute: int) -> float:
        """Hours of rush inside a login-to-logout span on a weekday."""
        window = self.peak_table[weekday, login_minute:logout_minute]
        return float(window.sum()) / 60.0


def _month_day(text: str) -> tuple[int, int]:
    month, day = text.split("-")
    return int(month), int(day)


def week_start_monday(instants: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monday (at midnight) of the week of each instant."""
    index = pd.DatetimeIndex(instants).normalize()
    return index - pd.to_timedelta(np.asarray(index.weekday), unit="D")
