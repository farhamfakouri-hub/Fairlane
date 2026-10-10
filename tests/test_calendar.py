from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ree.config import StoreConfig
from ree.domain.calendar import LANE_GROUPS, StoreCalendar, week_start_monday


@pytest.fixture(scope="module")
def calendar() -> StoreCalendar:
    return StoreCalendar(StoreConfig())


def test_lane_groups_follow_the_store_layout(calendar: StoreCalendar) -> None:
    expected = {1: "priority", 4: "card_only", 2: "zone_a", 3: "zone_a", 5: "zone_a", 7: "zone_a"}
    expected.update({register: "zone_b" for register in range(8, 16)})
    for register, group in expected.items():
        assert calendar.lane_group(register) == group
    registers = np.arange(1, 16)
    vectorised = calendar.lane_groups(registers)
    assert list(vectorised) == [calendar.lane_group(int(r)) for r in registers]
    assert set(vectorised) == set(LANE_GROUPS)


def test_peak_windows_per_day(calendar: StoreCalendar) -> None:
    def peak(text: str) -> bool:
        return bool(calendar.peak_flags(pd.DatetimeIndex([text]))[0])

    assert peak("2025-11-25 11:00")
    assert peak("2025-11-25 18:30")
    assert not peak("2025-11-25 09:00")
    assert not peak("2025-11-25 16:00")
    assert not peak("2025-11-25 14:00")
    assert peak("2025-11-28 08:45")
    assert peak("2025-11-29 08:45")
    assert not peak("2025-11-26 08:45")
    assert peak("2025-11-30 11:30")
    assert not peak("2025-11-30 09:30")
    assert not peak("2025-11-30 12:30")


def test_period_labels(calendar: StoreCalendar) -> None:
    labels = calendar.period_labels(pd.DatetimeIndex(["2025-11-25 11:00", "2025-11-25 16:00"]))
    assert list(labels) == ["peak", "off"]


def test_opening_hours(calendar: StoreCalendar) -> None:
    assert calendar.open_close_minutes(0) == (510, 1200)
    assert calendar.open_close_minutes(5) == (510, 1200)
    assert calendar.open_close_minutes(6) == (510, 750)
    assert calendar.is_busy_day(4) and calendar.is_busy_day(5) and calendar.is_busy_day(6)
    assert not calendar.is_busy_day(1)


def test_seasonal_multipliers(calendar: StoreCalendar) -> None:
    assert calendar.season(date(2025, 12, 29)) == (1.35, 1.20)
    assert calendar.season(date(2026, 1, 3)) == (1.35, 1.20)
    assert calendar.season(date(2025, 12, 28)) == (1.35, 1.20)
    assert calendar.season(date(2026, 1, 5)) == (1.0, 1.0)
    assert calendar.season(date(2026, 8, 30)) == (1.20, 1.10)
    assert calendar.season(date(2026, 9, 8)) == (1.0, 1.0)


def test_week_start_monday() -> None:
    starts = week_start_monday(
        pd.DatetimeIndex(["2025-11-26 10:00", "2025-11-30 23:00", "2025-12-01"])
    )
    assert [timestamp.strftime("%Y-%m-%d") for timestamp in starts] == [
        "2025-11-24",
        "2025-11-24",
        "2025-12-01",
    ]
