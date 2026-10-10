from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from planner_factories import (
    CONFIG,
    OPENING_ORDER,
    RUSH,
    TUESDAY,
    evening_pattern,
    flat_day,
    morning_pattern,
    pattern,
    requirement_for,
    slot,
    staggered_day,
    worker,
)

from ree.planner.checker.checks import check_breaks, check_daily
from ree.planner.daily.breaks import break_requests, schedule_breaks
from ree.planner.daily.daily_plan import plan_day_from_shifts
from ree.planner.daily.posts import post_importance, required_posts
from ree.planner.domain.plan import BREAK, FLOATER, is_register_post, is_self_checkout_post
from ree.planner.roster.model import neutral_scores


def test_required_posts_follow_the_priority_order() -> None:
    assert required_posts(slot(4, 1, 2), OPENING_ORDER) == [
        "R01",
        "R04",
        "SCO1-1",
        "R02",
        "R03",
        "SCO2-1",
        "SCO2-2",
    ]
    assert required_posts(slot(2, 0, 0), OPENING_ORDER) == ["R01", "R04"]
    assert required_posts(slot(3, 0, 1), OPENING_ORDER) == ["R01", "R04", "SCO2-1", "R02"]


def test_the_importance_of_a_post_follows_its_place_in_the_opening_order() -> None:
    assert post_importance("R01", OPENING_ORDER, CONFIG) == CONFIG.daily.importance_priority
    assert post_importance("R04", OPENING_ORDER, CONFIG) == CONFIG.daily.importance_priority
    assert post_importance("R02", OPENING_ORDER, CONFIG) == CONFIG.daily.importance_main
    assert post_importance("R12", OPENING_ORDER, CONFIG) == CONFIG.daily.importance_other
    assert post_importance("SCO1-1", OPENING_ORDER, CONFIG) == CONFIG.daily.importance_sco


def test_break_requests_have_the_length_and_window_of_the_rules() -> None:
    requests = break_requests({"A": morning_pattern()}, CONFIG.rules)
    assert len(requests) == 1
    request = requests[0]
    assert request.minutes == 25
    assert (request.low, request.high, request.nominal) == (690, 750, 720)
    assert break_requests({"A": pattern((600, 750))}, CONFIG.rules) == []
    split = break_requests({"A": pattern((510, 750), (990, 1290))}, CONFIG.rules)
    assert [(item.block_index, item.minutes) for item in split] == [(0, 15), (1, 15)]


def test_breaks_fall_inside_their_windows_with_the_right_length_and_at_most_two_at_once() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert not check_breaks(result.plan, CONFIG.rules)
    assert len(result.plan.breaks) == 8
    assert result.metrics.peak_simultaneous_breaks <= CONFIG.rules.max_simultaneous_breaks
    assert result.metrics.capacity_gap_minutes == 0


def test_breaks_stay_out_of_the_rush_when_the_window_allows_it() -> None:
    _, _, shifts, requirement, _ = staggered_day(rush=(630, 720))
    schedule = schedule_breaks(TUESDAY, shifts, requirement, CONFIG)
    assert all(entry.start >= 720 for entry in schedule.entries)


def test_identical_shifts_cannot_all_break_inside_the_limit_and_the_limit_case_is_flagged() -> None:
    _, workers, shifts, requirement, scores = flat_day(7, 4, 1)
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert result.metrics.peak_simultaneous_breaks == 3
    assert result.report.count("BREAKS_LIMIT_CASE") == 1
    assert result.report.count("TOO_MANY_BREAKS") == 0


def test_a_break_that_would_leave_too_few_people_on_duty_is_reported() -> None:
    _, workers, shifts, requirement, scores = flat_day(5, 4, 1)
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert result.metrics.capacity_gap_minutes > 0
    assert any("COVERAGE_GAP" in alert for alert in result.plan.alerts)
    assert result.report.count("COVERAGE_GAP") == 1


def test_every_required_post_is_staffed_and_the_plan_passes_the_checker() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert result.report.errors == []
    assert result.report.count("POST_EMPTY") == 0
    assert result.plan.alerts == []


def test_self_checkout_only_workers_never_hold_a_register_and_others_need_clearance() -> None:
    flags = {
        "W03": {"self_checkout_only": True, "cleared_self_checkout": True},
        "W04": {"cleared_self_checkout": False},
    }
    _, workers, shifts, requirement, scores = staggered_day(flags=flags, with_sco=True)
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    sco_segments = 0
    for segment in result.plan.segments:
        if segment.worker_id == "W03":
            assert not is_register_post(segment.post)
            sco_segments += is_self_checkout_post(segment.post)
        if segment.worker_id == "W04":
            assert not is_self_checkout_post(segment.post)
    assert sco_segments > 0
    assert result.report.errors == []


def test_strong_workers_hold_the_busiest_registers_in_the_rush() -> None:
    names, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    neutral = plan_day_from_shifts(
        TUESDAY, shifts, workers, requirement, neutral_scores(names), CONFIG
    )
    assert result.metrics.priority_score_gap > 1.0
    assert abs(neutral.metrics.priority_score_gap) < 0.5
    on_priority = {"R01", "R04"}
    strong = {"W01", "W02"}
    total = held = 0
    for minute in range(RUSH[0], RUSH[1], 5):
        for worker_id, post in result.plan.occupants(minute).items():
            if post in on_priority:
                total += 1
                held += worker_id in strong
    assert total > 0 and held / total >= 0.8


def test_a_weak_worker_floats_before_a_strong_one_when_there_are_spare_people() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    floating = {s.worker_id for s in result.plan.segments if s.post == FLOATER and s.start >= 630}
    assert floating & {"W07", "W08"}
    assert "W01" not in floating


def test_every_break_from_a_required_post_has_a_relief_handover_that_the_checker_accepts() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    plan = result.plan
    relief = {(h.minute, h.replaces): h for h in plan.handovers if h.reason == "relief"}
    breaks_from_posts = 0
    for entry in plan.breaks:
        before = plan.post_at(entry.worker_id, entry.start - 5)
        if before is not None and before not in (FLOATER, BREAK):
            breaks_from_posts += 1
            handover = relief.get((entry.start, entry.worker_id))
            assert handover is not None, (entry, before)
            assert handover.to_post == before
    assert breaks_from_posts > 0
    assert len(relief) == breaks_from_posts
    assert result.report.count("HANDOVER_INCONSISTENT") == 0
    assert result.report.count("RELIEF_INVALID") == 0
    assert len(result.relief_lines) == breaks_from_posts
    assert all(" takes " in line and "-minute break" in line for line in result.relief_lines)


def test_a_worker_who_returns_from_a_break_is_back_on_a_post_or_floating() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    plan = result.plan
    for entry in plan.breaks:
        after = plan.post_at(entry.worker_id, entry.end)
        assert after is not None and after != BREAK


def test_the_daily_plan_is_deterministic() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    first = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    second = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert first.plan.segments == second.plan.segments
    assert first.plan.breaks == second.plan.breaks
    assert first.plan.handovers == second.plan.handovers


def test_posts_change_hands_rarely_when_nobody_goes_on_break() -> None:
    names, workers, shifts, requirement, scores = staggered_day()
    quiet = CONFIG.with_overrides({"rules.break_min_block_hours": 20.0})
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, quiet)
    assert result.plan.breaks == []
    assert len(result.plan.handovers) <= 2 * len(names)


def test_two_shifts_that_overlap_hand_over_without_leaving_the_priority_registers_empty() -> None:
    names = [f"W{index}" for index in range(1, 9)]
    workers = {name: worker(name, 29.0) for name in names}
    shifts = {
        name: morning_pattern() if index < 4 else evening_pattern()
        for index, name in enumerate(names)
    }
    requirement = requirement_for([TUESDAY], registers=3, k_sc=0, zone1=0)
    result = plan_day_from_shifts(
        TUESDAY, shifts, workers, requirement, neutral_scores(names), CONFIG
    )
    assert result.report.count("REGISTERS_1_4_UNSTAFFED") == 0
    assert result.report.count("SEGMENT_MISMATCH") == 0


def test_a_manual_change_that_breaks_a_rule_is_caught_by_the_checker() -> None:
    _, workers, shifts, requirement, scores = staggered_day()
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    plan = result.plan
    plan.breaks = [
        entry
        if entry.worker_id != "W01"
        else type(entry)(
            entry.worker_id, entry.day, entry.block_index, entry.start + 120, entry.minutes
        )
        for entry in plan.breaks
    ]
    report = check_daily(plan, CONFIG.rules, requirement)
    assert report.errors


@settings(max_examples=25, deadline=None)
@given(
    pairs=st.integers(min_value=2, max_value=5),
    values=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False), min_size=10, max_size=10
    ),
)
def test_random_teams_always_get_a_legal_plan_with_every_post_staffed(
    pairs: int, values: list[float]
) -> None:
    _, workers, shifts, requirement, scores = staggered_day(pairs=pairs, z=values)
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert result.report.errors == []
    assert result.report.count("POST_EMPTY") == 0
    assert result.report.count("COVERAGE_GAP") == 0
    assert result.metrics.peak_simultaneous_breaks <= CONFIG.rules.max_simultaneous_breaks
    for worker_id in workers:
        covered = sum(s.end - s.start for s in result.plan.segments if s.worker_id == worker_id)
        assert covered == shifts[worker_id].register_minutes


def test_the_score_gap_is_zero_when_no_rush_slot_exists() -> None:
    _, workers, shifts, requirement, scores = staggered_day(rush=(0, 0))
    result = plan_day_from_shifts(TUESDAY, shifts, workers, requirement, scores, CONFIG)
    assert result.metrics.priority_score_gap == pytest.approx(0.0)


@pytest.fixture(scope="module")
def solved_week():
    from ree.planner.roster.solver import plan_roster
    from ree.planner.sim.roster_gate import scenario_inputs

    config, workers, requirement, scores = scenario_inputs(3, store="small")
    solution = plan_roster(workers, requirement, config, scores, weeks=1)
    return config, requirement, scores, solution


def test_a_solved_roster_gets_a_legal_daily_plan_on_every_day(solved_week) -> None:
    from ree.planner.daily.daily_plan import plan_days

    config, requirement, scores, solution = solved_week
    results = plan_days(solution.plan, requirement, config, scores)
    assert len(results) == 7
    for day, result in results.items():
        assert result.report.errors == [], (day, result.report.summary())
        assert result.metrics.peak_simultaneous_breaks <= config.rules.max_simultaneous_breaks
        assert result.report.count("HANDOVER_INCONSISTENT") == 0
        assert result.report.count("RELIEF_INVALID") == 0
        for worker_id, pattern_ in result.plan.shifts.items():
            covered = sum(s.end - s.start for s in result.plan.segments if s.worker_id == worker_id)
            assert covered == pattern_.register_minutes
        warned = {item.code for item in result.report.warnings}
        assert warned <= {"COVERAGE_GAP", "POST_EMPTY", "BREAKS_LIMIT_CASE"}


def test_every_alert_of_a_solved_week_comes_from_a_named_checker_rule(solved_week) -> None:
    from ree.planner.daily.daily_plan import plan_days

    config, requirement, scores, solution = solved_week
    for result in plan_days(solution.plan, requirement, config, scores).values():
        for alert in result.plan.alerts:
            assert any(
                code in alert for code in ("COVERAGE_GAP", "POST_EMPTY", "BREAKS_LIMIT_CASE")
            )
