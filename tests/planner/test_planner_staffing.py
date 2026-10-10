from __future__ import annotations

from datetime import date
from math import exp, factorial, inf

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ree.planner.config import OPENING_ORDER, PlannerConfig
from ree.planner.demand.forecast import default_model, forecast_cycle, forecast_day
from ree.planner.domain.demand_types import PlannerCalendar
from ree.planner.staffing.erlang import (
    erlang_b,
    erlang_c,
    mean_wait,
    queue_limit_target_minutes,
    required_servers,
    wait_exceeds,
)
from ree.planner.staffing.opening_order import (
    last_register_to_close,
    next_register_to_open,
    registers_open,
)
from ree.planner.staffing.requirement import (
    build_requirement,
    estimate_service_minutes,
    requirement_for_cycle,
    slot_requirement,
)

CONFIG = PlannerConfig()
CALENDAR = PlannerCalendar(CONFIG)
MODEL = default_model(CALENDAR, CONFIG)


def closed_form_erlang_c(servers: int, load: float) -> float:
    tail = load**servers / factorial(servers) * servers / (servers - load)
    head = sum(load**k / factorial(k) for k in range(servers))
    return tail / (head + tail)


def test_the_design_document_worked_example_is_reproduced() -> None:
    assert erlang_c(3, 2.0) == pytest.approx(4 / 9)
    assert mean_wait(3, 2.0, 2.0) * 60 == pytest.approx(53.33, abs=0.01)
    assert mean_wait(3, 2.0, 2.0) == pytest.approx(0.889, abs=0.001)
    assert wait_exceeds(3, 2.0, 2.0, 3.0) == pytest.approx(0.0992, abs=0.0001)
    assert erlang_c(4, 2.0) == pytest.approx(0.174, abs=0.001)
    assert mean_wait(4, 2.0, 2.0) * 60 == pytest.approx(10.4, abs=0.1)
    assert wait_exceeds(4, 2.0, 2.0, 3.0) == pytest.approx(0.0087, abs=0.0001)
    assert erlang_c(5, 2.0) == pytest.approx(0.0597, abs=0.0005)
    assert wait_exceeds(5, 2.0, 2.0, 3.0) == pytest.approx(0.00066, abs=0.00002)


def test_two_registers_are_not_stable_for_two_erlangs() -> None:
    assert erlang_c(2, 2.0) == 1.0
    assert mean_wait(2, 2.0, 2.0) == inf
    assert wait_exceeds(2, 2.0, 2.0, 3.0) == 1.0


@given(st.integers(min_value=1, max_value=30), st.floats(min_value=0.01, max_value=0.99))
@settings(max_examples=80, deadline=None)
def test_the_recursion_matches_the_brute_force_formula(servers: int, utilisation: float) -> None:
    load = utilisation * servers
    assert erlang_c(servers, load) == pytest.approx(closed_form_erlang_c(servers, load), rel=1e-8)
    waiting = erlang_c(servers, load)
    assert 0.0 <= erlang_b(servers, load) <= waiting <= 1.0
    expected = waiting * exp(-(servers - load) * 3.0 / 2.0)
    assert wait_exceeds(servers, load, 2.0, 3.0) == pytest.approx(expected, rel=1e-9)


@given(
    st.floats(min_value=0.1, max_value=11.0),
    st.floats(min_value=0.0, max_value=2.0),
    st.sampled_from([0.1, 0.2, 0.35]),
)
@settings(max_examples=80, deadline=None)
def test_required_registers_never_decrease_with_demand(
    load: float, extra: float, alpha: float
) -> None:
    low, _ = required_servers(load, 2.0, 3.0, alpha, 2, 15)
    high, _ = required_servers(load + extra, 2.0, 3.0, alpha, 2, 15)
    assert high >= low >= 2


def test_the_requirement_is_the_smallest_stable_count_meeting_the_target() -> None:
    servers, saturated = required_servers(2.0, 2.0, 3.0, 0.20, 2, 15)
    assert (servers, saturated) == (3, False)
    assert required_servers(2.0, 2.0, 3.0, 0.05, 2, 15) == (4, False)
    assert required_servers(0.0, 2.0, 3.0, 0.20, 2, 15) == (2, False)
    assert required_servers(40.0, 2.0, 3.0, 0.20, 2, 15) == (15, True)


def test_the_queue_limit_converts_to_a_wait_target() -> None:
    assert queue_limit_target_minutes(5, 2.0, 3) == pytest.approx(10 / 3)
    assert queue_limit_target_minutes(5, 2.0, 0) == pytest.approx(10.0)


def test_registers_one_and_four_are_always_open_first() -> None:
    assert OPENING_ORDER[:2] == (1, 4)
    assert registers_open(2) == (1, 4)
    assert registers_open(7) == (1, 4, 2, 3, 5, 6, 7)
    assert registers_open(15) == tuple(OPENING_ORDER)
    assert registers_open(0) == ()
    assert next_register_to_open((1, 4, 2)) == 3
    assert last_register_to_close((1, 4, 2, 3, 5)) == 5
    assert next_register_to_open(OPENING_ORDER) is None
    for count in range(2, 16):
        assert {1, 4} <= set(registers_open(count))


def test_even_an_empty_slot_keeps_two_registers_open() -> None:
    requirement = slot_requirement(0, False, 600, 1.0, CALENDAR, CONFIG)
    assert requirement.registers == 2
    assert requirement.headcount == 2 + 3 + 2
    assert not requirement.saturated


def test_rush_slots_use_five_supervisors_and_calm_slots_three() -> None:
    calm = slot_requirement(30, False, 600, 1.0, CALENDAR, CONFIG)
    rush = slot_requirement(120, True, 660, 1.0, CALENDAR, CONFIG)
    assert (calm.k_sc, calm.sco_zone1, calm.sco_zone2) == (3, 1, 2)
    assert (rush.k_sc, rush.sco_zone1, rush.sco_zone2) == (5, 2, 3)
    assert rush.registers > calm.registers
    assert rush.headcount == max(rush.registers, 2) + 5 + rush.floaters
    assert calm.floaters >= 2 and rush.floaters >= 2


def test_the_headcount_adds_the_break_buffer_to_registers_and_supervisors() -> None:
    requirement = slot_requirement(150, True, 660, 1.0, CALENDAR, CONFIG)
    assert requirement.headcount == requirement.registers + requirement.k_sc + requirement.floaters
    wide = CONFIG.with_overrides({"staffing.break_share": 0.5})
    larger = slot_requirement(150, True, 660, 1.0, CALENDAR, wide)
    assert larger.floaters > requirement.floaters
    assert larger.headcount > requirement.headcount


def test_the_christmas_service_time_raises_the_requirement() -> None:
    normal = slot_requirement(120, True, 660, 1.0, CALENDAR, CONFIG)
    heavy = slot_requirement(120, True, 660, 1.15, CALENDAR, CONFIG)
    assert heavy.registers >= normal.registers
    assert heavy.offered_load > normal.offered_load


def test_a_saturated_slot_is_flagged_and_capped() -> None:
    saturated = slot_requirement(900, True, 660, 1.0, CALENDAR, CONFIG)
    assert saturated.saturated and saturated.registers == 15


def test_the_default_store_opens_eight_to_nine_registers_in_the_rush() -> None:
    forecasts = forecast_cycle(MODEL, CALENDAR, CONFIG, date(2026, 11, 15), 1)
    table = build_requirement(forecasts, CALENDAR, CONFIG)
    peaks = {day: max(row.registers for _, row in table.slots(day)) for day in table.dates()}
    assert max(peaks.values()) in (9, 10)
    assert all(6 <= peak <= 10 for peak in peaks.values())
    assert peaks[date(2026, 11, 21)] >= 8
    calm = min(row.registers for _, row in table.slots(date(2026, 11, 17)))
    assert 2 <= calm <= 3
    assert table.opening_order == CONFIG.staffing.opening_order


def test_the_weekly_requirement_fits_the_default_workforce() -> None:
    forecasts = forecast_cycle(MODEL, CALENDAR, CONFIG, date(2026, 11, 15), 1)
    table = build_requirement(forecasts, CALENDAR, CONFIG)
    required = sum(table.staff_minutes(day) for day in table.dates()) / 60.0
    shifts = required / 6.0
    paid = required + 0.25 * shifts
    assert 800 < paid < 1050


def test_the_requirement_cycle_covers_three_weeks_and_respects_the_season() -> None:
    table = requirement_for_cycle(MODEL, CALENDAR, CONFIG, date(2026, 12, 20), 3)
    assert len(table.dates()) == 21
    christmas = date(2026, 12, 26)
    normal = date(2026, 11, 21)
    busy = sum(row.planned_customers for _, row in table.slots(christmas))
    ordinary = sum(
        row.planned_customers
        for _, row in build_requirement(
            [forecast_day(MODEL, CALENDAR, CONFIG, normal)], CALENDAR, CONFIG
        ).slots(normal)
    )
    assert busy > ordinary * 1.2
    assert table.slots(christmas)[0][1].offered_load > 0


def test_service_time_is_estimated_from_busy_gaps_between_transaction_starts() -> None:
    starts = {
        1: [0, 120, 240, 360, 1500, 1620, 1740],
        2: [0, 118, 244, 358, 3000],
    }
    assert estimate_service_minutes(starts) == pytest.approx(2.0, abs=0.05)
    with pytest.raises(ValueError):
        estimate_service_minutes({1: [0, 1000]})
