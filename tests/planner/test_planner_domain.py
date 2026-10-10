from __future__ import annotations

from datetime import date

import pytest

from ree.planner.config import PlannerConfig, clock_to_minutes
from ree.planner.domain.demand_types import PlannerCalendar
from ree.planner.domain.patterns import (
    PatternSpec,
    full_pattern_set,
    generate_patterns,
    solver_pattern_set,
)
from ree.planner.domain.rules import break_minutes, break_window
from ree.planner.domain.shifts import Block, make_block, pattern_burden

CONFIG = PlannerConfig()
SPEC = PatternSpec()
FULL = full_pattern_set(CONFIG, SPEC)
SOLVER = solver_pattern_set(CONFIG, SPEC)


def minutes(clock: str) -> int:
    return clock_to_minutes(clock)


def test_every_generated_pattern_respects_length_and_margin_rules() -> None:
    for pattern in (*FULL.weekday, *FULL.sunday):
        assert 180 <= pattern.register_minutes <= 570
        assert pattern.register_minutes % 5 == 0
        assert pattern.paid_minutes <= CONFIG.rules.max_daily_hours * 60
        for block in pattern.blocks:
            assert block.register_minutes >= SPEC.min_block_minutes
            assert block.register_start - block.paid_start == 10
            assert block.paid_end - block.register_end == 5


def test_weekday_patterns_follow_the_roster_sheet_ranges() -> None:
    continuous = [p for p in FULL.weekday if p.kind == "continuous"]
    morning = [p for p in continuous if p.register_end < minutes("17:00")]
    evening = [p for p in continuous if p.register_end >= minutes("21:30")]
    assert morning and evening
    for pattern in morning:
        assert minutes("08:20") <= pattern.paid_start <= minutes("10:00")
        assert minutes("12:55") <= pattern.paid_end <= minutes("16:20")
    for pattern in evening:
        assert minutes("14:05") <= pattern.paid_start <= minutes("17:30")
        assert pattern.paid_end == minutes("21:35")
        assert pattern.register_end == minutes("21:30")
    shortest = min(p.register_minutes for p in FULL.weekday)
    longest = max(p.register_minutes for p in FULL.weekday)
    assert shortest == 180 and longest <= 570


def test_split_patterns_have_one_cut_of_one_hour_or_ninety_minutes() -> None:
    splits = [p for p in FULL.weekday if p.kind == "split"]
    assert splits
    for pattern in splits:
        assert len(pattern.blocks) == 2
        assert pattern.cuts[0] in (60, 90)
        assert minutes("09:20") <= pattern.paid_start <= minutes("09:30")
    assert {p.cuts[0] for p in splits} == {60, 90}


def test_no_pattern_opens_and_closes_the_same_day() -> None:
    open_minute = minutes("08:30")
    close_minute = minutes("21:30")
    slack = CONFIG.rules.opening_slack_minutes
    for pattern in FULL.weekday:
        opens = pattern.register_start <= open_minute + slack
        closes = pattern.register_end >= close_minute
        assert not (opens and closes)


def test_sunday_patterns_stay_inside_the_sunday_hours() -> None:
    assert FULL.sunday
    for pattern in FULL.sunday:
        assert pattern.register_start >= minutes("08:30")
        assert pattern.register_end == minutes("12:30")
        assert 180 <= pattern.register_minutes <= 240


def test_solver_patterns_are_a_thinned_subset_on_the_thirty_minute_grid() -> None:
    assert len(SOLVER.weekday) == 39 and len(SOLVER.sunday) == 3
    for pattern in (*SOLVER.weekday, *SOLVER.sunday):
        assert pattern.pattern_id in FULL.by_id
        for block in pattern.blocks:
            assert block.register_start % 30 == 0 and block.register_end % 30 == 0


def test_generation_is_deterministic_and_ids_are_unique() -> None:
    again = generate_patterns(SPEC, CONFIG.rules, CONFIG.opening, "weekday")
    assert again == FULL.weekday
    ids = [p.pattern_id for p in FULL.weekday]
    assert len(ids) == len(set(ids))


def test_disabled_families_add_nothing_and_enabling_them_adds_patterns() -> None:
    payload = SPEC.model_dump(mode="python")
    for family in payload["weekday"]["families"]:
        if family["name"] == "midday":
            family["enabled"] = True
    enabled = PatternSpec.model_validate(payload)
    more = generate_patterns(enabled, CONFIG.rules, CONFIG.opening, "weekday")
    assert len(more) > len(FULL.weekday)


@pytest.mark.parametrize(
    ("register_minutes", "expected"),
    [(150, 0), (180, 10), (210, 15), (240, 15), (300, 15), (360, 20), (420, 25), (570, 30)],
)
def test_break_length_is_three_minutes_per_hour_rounded_up(
    register_minutes: int, expected: int
) -> None:
    assert break_minutes(register_minutes, CONFIG.rules) == expected


def test_break_window_is_centred_on_the_middle_of_the_block() -> None:
    six_hours = make_block(minutes("09:30"), minutes("15:30"), 10, 5)
    assert break_window(six_hours, CONFIG.rules) == (minutes("12:00"), minutes("13:00"))
    three_hours = make_block(minutes("09:00"), minutes("12:00"), 10, 5)
    low, high = break_window(three_hours, CONFIG.rules) or (0, 0)
    assert low == minutes("10:00") and high == minutes("11:00")
    assert break_window(Block(0, 100, 10, 95), CONFIG.rules) is None


def test_breaks_never_start_outside_the_register_window() -> None:
    for pattern in FULL.weekday[::25]:
        for block in pattern.blocks:
            window = break_window(block, CONFIG.rules)
            assert window is not None
            length = break_minutes(block.register_minutes, CONFIG.rules)
            assert block.register_start <= window[0] <= window[1]
            assert window[1] + length <= block.register_end


def test_rush_windows_follow_the_owner_answers() -> None:
    calendar = PlannerCalendar(CONFIG)
    monday = date(2026, 11, 16)
    friday = date(2026, 11, 20)
    saturday = date(2026, 11, 21)
    sunday = date(2026, 11, 22)
    assert calendar.rush_windows(monday) == ((630, 840), (1080, 1200))
    expected_friday = ((510, 540), (630, 840), (1080, 1260))
    assert calendar.rush_windows(friday) == expected_friday
    assert calendar.rush_windows(saturday) == expected_friday
    assert calendar.rush_windows(sunday) == ((660, 750),)
    for day in (monday, friday, saturday):
        _, close = calendar.opening(day)
        assert all(end <= close - 30 for _, end in calendar.rush_windows(day))
    assert calendar.rush_windows(sunday)[-1][1] == calendar.opening(sunday)[1]
    assert calendar.is_rush(friday, minutes("20:30"))
    assert not calendar.is_rush(monday, minutes("20:30"))
    assert not calendar.is_rush(saturday, minutes("21:00"))


def test_month_start_extends_the_weekday_evening_rush() -> None:
    calendar = PlannerCalendar(CONFIG)
    assert calendar.rush_windows(date(2026, 11, 30))[-1] == (1080, 1260)
    assert calendar.rush_windows(date(2026, 11, 3))[-1] == (1080, 1260)
    assert calendar.rush_windows(date(2026, 11, 17))[-1] == (1080, 1200)
    quiet = PlannerCalendar(CONFIG.with_overrides({"rush.month_start_extends_evening": False}))
    assert quiet.rush_windows(date(2026, 11, 30))[-1] == (1080, 1200)


def test_seasonal_multipliers_and_manager_override() -> None:
    calendar = PlannerCalendar(CONFIG)
    assert calendar.season(date(2026, 11, 17)) == (1.0, 1.0)
    assert calendar.season(date(2026, 12, 25)) == (1.35, 1.15)
    arrival, service = calendar.season(date(2026, 12, 29))
    assert arrival == pytest.approx(1.35 * 1.10) and service == pytest.approx(1.15)
    assert calendar.season(date(2027, 1, 2))[0] == pytest.approx(1.35 * 1.10)
    assert calendar.season(date(2027, 1, 4))[0] == pytest.approx(1.10)
    assert calendar.season(date(2027, 1, 10)) == (1.0, 1.0)
    assert calendar.season(date(2026, 8, 30))[0] == pytest.approx(1.15 * 1.10)
    overridden = PlannerCalendar(
        CONFIG.with_overrides({"season.manager_arrival_override": {"2026-11-17": 1.5}})
    )
    assert overridden.season(date(2026, 11, 17))[0] == pytest.approx(1.5)


def test_slots_and_floaters_follow_the_store_hours() -> None:
    calendar = PlannerCalendar(CONFIG)
    assert len(calendar.slot_starts(date(2026, 11, 2))) == 26
    assert len(calendar.slot_starts(date(2026, 11, 8))) == 8
    assert calendar.slot_starts(date(2026, 11, 2))[0] == minutes("08:30")
    assert calendar.floaters_required(minutes("13:59")) == 2
    assert calendar.floaters_required(minutes("14:00")) == 2
    assert calendar.week_start(date(2026, 11, 7)) == date(2026, 11, 1)
    assert calendar.week_start(date(2026, 11, 1)) == date(2026, 11, 1)


def test_pattern_burden_ranks_unpopular_shifts_higher() -> None:
    closing = next(
        p for p in FULL.weekday if p.register_end == minutes("21:30") and len(p.blocks) == 1
    )
    split = next(p for p in FULL.weekday if p.kind == "split")
    morning = next(
        p
        for p in FULL.weekday
        if p.register_start == minutes("10:00") and p.register_end < minutes("15:00")
    )
    sunday = FULL.sunday[0]
    burden = CONFIG.burden
    opening = CONFIG.opening
    assert pattern_burden(closing, 0, burden, opening) == burden.closing
    assert pattern_burden(closing, 5, burden, opening) == burden.closing + burden.saturday
    assert pattern_burden(closing, 4, burden, opening) == burden.closing + burden.friday_evening
    assert pattern_burden(split, 1, burden, opening) >= burden.split
    assert pattern_burden(morning, 1, burden, opening) == 0
    assert pattern_burden(sunday, 6, burden, opening) >= burden.sunday
