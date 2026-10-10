from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import timedelta

import pytest
from planner_factories import (
    MONDAY,
    RULES,
    START,
    evening_pattern,
    minutes,
    morning_pattern,
    pattern,
    requirement_for,
    valid_daily,
    valid_roster,
    worker,
)

from ree.planner.checker.checks import check, check_daily, check_roster
from ree.planner.checker.report import CATALOGUE
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import (
    Assignment,
    BreakEntry,
    DailyPlan,
    Handover,
    PostSegment,
    RosterPlan,
)
from ree.planner.domain.shifts import Block, ShiftPattern

RosterMutation = Callable[[RosterPlan, RequirementTable], tuple[RosterPlan, RequirementTable]]
DailyMutation = Callable[[DailyPlan, RequirementTable], tuple[DailyPlan, RequirementTable]]


def copy_roster(plan: RosterPlan) -> RosterPlan:
    return RosterPlan(plan.start_date, plan.weeks, dict(plan.workers), list(plan.assignments))


def copy_daily(plan: DailyPlan) -> DailyPlan:
    return DailyPlan(
        plan.day,
        dict(plan.shifts),
        dict(plan.workers),
        list(plan.segments),
        list(plan.breaks),
        list(plan.handovers),
        list(plan.alerts),
        plan.grid_minutes,
    )


def replace_pattern(plan: RosterPlan, worker_id: str, day, new: ShiftPattern) -> RosterPlan:
    kept = [a for a in plan.assignments if not (a.worker_id == worker_id and a.day == day)]
    return RosterPlan(
        plan.start_date, plan.weeks, plan.workers, [*kept, Assignment(worker_id, day, new)]
    )


def broken_block(**changes: int) -> ShiftPattern:
    base = morning_pattern().blocks[0]
    return ShiftPattern("X", "continuous", (dataclasses.replace(base, **changes),))


def test_a_hand_made_valid_roster_has_no_violations() -> None:
    plan, requirement = valid_roster()
    report = check_roster(plan, RULES, requirement)
    assert report.violations == [], report.summary()
    assert report.ok


def test_a_hand_made_valid_daily_plan_has_no_violations() -> None:
    plan, requirement = valid_daily()
    report = check_daily(plan, RULES, requirement)
    assert report.violations == [], report.summary()


def test_check_dispatches_on_the_plan_type() -> None:
    roster, requirement = valid_roster()
    daily, daily_requirement = valid_daily()
    assert check(roster, RULES, requirement).ok
    assert check(daily, RULES, daily_requirement).ok
    with pytest.raises(TypeError):
        check("plan", RULES)


def mutate_unknown_worker(plan, requirement):
    out = copy_roster(plan)
    out.assignments.append(Assignment("X99", MONDAY, morning_pattern()))
    return out, requirement


def mutate_second_pattern(plan, requirement):
    out = copy_roster(plan)
    out.assignments.append(Assignment("A", MONDAY, evening_pattern()))
    return out, requirement


def mutate_paid_window_inside_register(plan, requirement):
    return replace_pattern(plan, "A", MONDAY, broken_block(paid_end=minutes("15:00"))), requirement


def mutate_margin(plan, requirement):
    block = morning_pattern().blocks[0]
    return replace_pattern(
        plan, "A", MONDAY, broken_block(paid_start=block.paid_start + 1)
    ), requirement


def mutate_short_block(plan, requirement):
    short = pattern((minutes("09:00"), minutes("11:30")))
    return replace_pattern(plan, "A", MONDAY, short), requirement


def mutate_long_day(plan, requirement):
    long_pattern = pattern((minutes("11:00"), minutes("21:30")))
    return replace_pattern(plan, "A", MONDAY, long_pattern), requirement


def mutate_three_blocks(plan, requirement):
    triple = pattern(
        (minutes("09:00"), minutes("12:00")),
        (minutes("13:00"), minutes("16:00")),
        (minutes("17:00"), minutes("20:00")),
    )
    return replace_pattern(plan, "A", MONDAY, triple), requirement


def mutate_short_cut(plan, requirement):
    split = pattern((minutes("09:30"), minutes("13:30")), (minutes("14:00"), minutes("19:00")))
    return replace_pattern(plan, "A", MONDAY, split), requirement


def mutate_before_opening(plan, requirement):
    early = pattern((minutes("08:00"), minutes("15:00")))
    return replace_pattern(plan, "A", MONDAY, early), requirement


def mutate_open_and_close(plan, requirement):
    all_day = pattern((minutes("09:00"), minutes("21:30")))
    return replace_pattern(plan, "A", MONDAY, all_day), requirement


def mutate_blocked_day(plan, requirement):
    out = copy_roster(plan)
    out.workers["A"] = dataclasses.replace(out.workers["A"], blocked={MONDAY: ((0, 1440),)})
    return out, requirement


def mutate_soft_block(plan, requirement):
    out = copy_roster(plan)
    out.workers["A"] = dataclasses.replace(
        out.workers["A"], soft_blocked={MONDAY: ((minutes("10:00"), minutes("12:00")),)}
    )
    return out, requirement


def mutate_missing_shift(plan, requirement):
    return plan.without("A", MONDAY), requirement


def mutate_contract_too_high(plan, requirement):
    out = copy_roster(plan)
    out.workers["A"] = dataclasses.replace(out.workers["A"], contract_hours=35.0)
    return out, requirement


def mutate_seven_days(plan, requirement):
    out = copy_roster(plan)
    for offset in range(7):
        day = START + timedelta(days=offset)
        if out.pattern_of("A", day) is None:
            out.assignments.append(Assignment("A", day, morning_pattern()))
    out.workers["A"] = dataclasses.replace(out.workers["A"], contract_hours=50.75)
    return out, requirement


def mutate_short_rest(plan, requirement):
    out = replace_pattern(plan, "A", MONDAY + timedelta(days=0), evening_pattern())
    return out, requirement


def mutate_no_evening(plan, requirement):
    out = copy_roster(plan)
    out.assignments = [a for a in out.assignments if a.worker_id not in ("C", "D", "G")]
    for name in "CDG":
        out.workers[name] = dataclasses.replace(out.workers[name], contract_hours=0.0)
    return out, requirement


def mutate_higher_headcount(plan, requirement):
    return plan, requirement_for(requirement.dates(), headcount=5)


def mutate_self_checkout_need(plan, requirement):
    out = copy_roster(plan)
    for name in out.workers:
        out.workers[name] = dataclasses.replace(out.workers[name], cleared_self_checkout=False)
    return out, requirement_for(requirement.dates(), headcount=2, k_sc=1, zone1=1)


ROSTER_CASES: list[tuple[str, RosterMutation, set[str]]] = [
    ("unknown worker", mutate_unknown_worker, {"UNKNOWN_WORKER"}),
    ("second pattern", mutate_second_pattern, {"TWO_PATTERNS_SAME_DAY"}),
    ("window order", mutate_paid_window_inside_register, {"PATTERN_WINDOW_ORDER"}),
    ("margins", mutate_margin, {"PATTERN_MARGINS"}),
    ("short block", mutate_short_block, {"BLOCK_TOO_SHORT"}),
    ("long day", mutate_long_day, {"MAX_DAILY_HOURS"}),
    ("three blocks", mutate_three_blocks, {"TOO_MANY_CUTS"}),
    ("short cut", mutate_short_cut, {"CUT_LENGTH"}),
    ("before opening", mutate_before_opening, {"OUTSIDE_OPENING_HOURS"}),
    ("open and close", mutate_open_and_close, {"OPEN_AND_CLOSE"}),
    ("blocked day", mutate_blocked_day, {"AVAILABILITY"}),
    ("lost shift", mutate_missing_shift, {"WEEKLY_HOURS_CONTRACT"}),
    ("contract too high", mutate_contract_too_high, {"WEEKLY_HOURS_CONTRACT"}),
    (
        "seven days",
        mutate_seven_days,
        {"MAX_CONSECUTIVE_DAYS", "WEEKLY_REST", "MAX_WEEKLY_HOURS", "AVERAGE_WEEKLY_HOURS"},
    ),
    ("short rest", mutate_short_rest, {"MIN_REST"}),
    ("nobody in the evening", mutate_no_evening, {"REGISTERS_1_4_UNSTAFFED", "UNDERSTAFFED"}),
    ("higher headcount", mutate_higher_headcount, {"UNDERSTAFFED"}),
    ("self-checkout cover", mutate_self_checkout_need, {"SELF_CHECKOUT_UNCOVERED"}),
]


@pytest.mark.parametrize(
    ("name", "mutation", "expected"), ROSTER_CASES, ids=[c[0] for c in ROSTER_CASES]
)
def test_every_planted_roster_violation_is_rejected_with_the_right_reason(
    name: str, mutation: RosterMutation, expected: set[str]
) -> None:
    plan, requirement = valid_roster()
    broken, new_requirement = mutation(plan, requirement)
    report = check_roster(broken, RULES, new_requirement)
    assert expected <= report.codes(), f"{name}: {report.summary()}"
    for code in expected:
        severity = CATALOGUE[code][0]
        assert code in report.codes(severity)


def test_a_possible_absence_is_a_warning_not_an_error() -> None:
    plan, requirement = valid_roster()
    broken, new_requirement = mutate_soft_block(plan, requirement)
    report = check_roster(broken, RULES, new_requirement)
    assert report.ok and "POTENTIAL_ABSENCE" in report.codes("warning")


def test_the_roster_report_names_the_worker_and_the_day() -> None:
    plan, requirement = valid_roster()
    broken, new_requirement = mutate_blocked_day(plan, requirement)
    report = check_roster(broken, RULES, new_requirement)
    hit = next(item for item in report.violations if item.code == "AVAILABILITY")
    assert hit.worker_id == "A" and hit.day == MONDAY
    assert "AVAILABILITY" in report.summary()


def replace_break(plan: DailyPlan, worker_id: str, **changes) -> DailyPlan:
    out = copy_daily(plan)
    out.breaks = [
        dataclasses.replace(item, **changes) if item.worker_id == worker_id else item
        for item in plan.breaks
    ]
    return out


def move_break(plan: DailyPlan, worker_id: str, start: int) -> DailyPlan:
    out = replace_break(plan, worker_id, start=start)
    out.segments = [
        dataclasses.replace(s, start=start, end=start + 25)
        if (s.worker_id == worker_id and s.post == "BREAK")
        else s
        for s in plan.segments
    ]
    return out


def daily_missing_break(plan, requirement):
    out = copy_daily(plan)
    out.breaks = [b for b in plan.breaks if b.worker_id != "A"]
    out.segments = [s for s in plan.segments if not (s.worker_id == "A" and s.post == "BREAK")]
    return out, requirement


def daily_wrong_length(plan, requirement):
    return replace_break(plan, "A", minutes=10), requirement


def daily_moved_break(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        s
        for s in plan.segments
        if not (s.worker_id == "B" and s.start >= 745)
        and not (s.worker_id == "F" and s.start >= 690 and s.end <= 930)
    ]
    out.segments += [
        PostSegment("B", plan.day, 745, 755, "FLOAT"),
        PostSegment("B", plan.day, 755, 930, "R01"),
        PostSegment("F", plan.day, 690, 755, "R01"),
        PostSegment("F", plan.day, 755, 780, "BREAK"),
        PostSegment("F", plan.day, 780, 930, "FLOAT"),
    ]
    out.breaks = [
        dataclasses.replace(b, start=755) if b.worker_id == "F" else b for b in plan.breaks
    ]
    out.handovers = [
        Handover(plan.day, 690, "F", "FLOAT", "R01", "relief", "A"),
        Handover(plan.day, 720, "A", "FLOAT", "R04", "relief", "B"),
        Handover(plan.day, 755, "B", "FLOAT", "R01", "relief", "F"),
    ]
    return out, requirement


def daily_far_break(plan, requirement):
    return move_break(plan, "A", minutes("09:30")), requirement


def daily_break_outside_shift(plan, requirement):
    out = copy_daily(plan)
    out.breaks = [*plan.breaks, BreakEntry("A", plan.day, 0, 480, 25)]
    return out, requirement


def daily_duplicate_break(plan, requirement):
    out = copy_daily(plan)
    out.breaks = [*plan.breaks, BreakEntry("A", plan.day, 0, 700, 25)]
    return out, requirement


def daily_unknown_break(plan, requirement):
    out = copy_daily(plan)
    out.breaks = [*plan.breaks, BreakEntry("A", plan.day, 3, 700, 25)]
    return out, requirement


def daily_limit_case(plan, requirement):
    out = plan
    for name in "ABF":
        out = move_break(out, name, minutes("11:30"))
    return out, requirement


def daily_too_many_breaks(plan, requirement):
    out, _ = daily_limit_case(plan, requirement)
    return replace_break(out, "C", start=minutes("11:30")), requirement


def daily_break_segment_missing(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        dataclasses.replace(s, post="FLOAT") if (s.worker_id == "A" and s.post == "BREAK") else s
        for s in plan.segments
    ]
    return out, requirement


def daily_double_booked(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        dataclasses.replace(s, end=s.end + 10) if (s.worker_id == "A" and s.post == "R01") else s
        for s in plan.segments
    ]
    return out, requirement


def daily_gap_in_timeline(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        s
        for s in plan.segments
        if not (s.worker_id == "F" and s.post == "FLOAT" and s.start == 775)
    ]
    return out, requirement


def daily_double_post(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        dataclasses.replace(s, post="R04")
        if (s.worker_id == "F" and s.post == "FLOAT" and s.start == 510)
        else s
        for s in plan.segments
    ]
    return out, requirement


def daily_register_left_empty(plan, requirement):
    out = copy_daily(plan)
    out.segments = [
        dataclasses.replace(s, post="FLOAT") if (s.worker_id == "F" and s.post == "R01") else s
        for s in plan.segments
    ]
    out.handovers = [h for h in plan.handovers if h.worker_id != "F"]
    return out, requirement


def daily_coverage_gap(plan, requirement):
    return plan, requirement_for([plan.day], headcount=3, registers=3, k_sc=0, floaters=1)


def daily_self_checkout_only(plan, requirement):
    out = copy_daily(plan)
    out.workers["A"] = dataclasses.replace(out.workers["A"], self_checkout_only=True)
    return out, requirement


def daily_bad_handover(plan, requirement):
    out = copy_daily(plan)
    out.handovers = [*plan.handovers, Handover(plan.day, 800, "A", "R04", "R01", "move")]
    return out, requirement


def daily_bad_relief(plan, requirement):
    out = copy_daily(plan)
    out.handovers = [*plan.handovers, Handover(plan.day, 930, "C", "FLOAT", "R01", "relief", "A")]
    return out, requirement


def daily_bad_margins(plan, requirement):
    out = copy_daily(plan)
    out.shifts["A"] = ShiftPattern("X", "continuous", (Block(500, 940, 510, 930),))
    return out, requirement


DAILY_CASES: list[tuple[str, DailyMutation, set[str]]] = [
    ("missing break", daily_missing_break, {"BREAK_MISSING"}),
    ("wrong length", daily_wrong_length, {"BREAK_LENGTH"}),
    ("moved break", daily_moved_break, {"BREAK_MOVED"}),
    ("far break", daily_far_break, {"BREAK_OUTSIDE_WINDOW"}),
    ("break outside shift", daily_break_outside_shift, {"BREAK_OUTSIDE_SHIFT"}),
    ("duplicate break", daily_duplicate_break, {"BREAK_DUPLICATE"}),
    ("unknown break", daily_unknown_break, {"BREAK_UNKNOWN"}),
    ("limit case", daily_limit_case, {"BREAKS_LIMIT_CASE"}),
    ("too many breaks", daily_too_many_breaks, {"TOO_MANY_BREAKS"}),
    ("break segment missing", daily_break_segment_missing, {"BREAK_SEGMENT_MISMATCH"}),
    ("double booked", daily_double_booked, {"DOUBLE_BOOKED"}),
    ("gap in timeline", daily_gap_in_timeline, {"SEGMENT_MISMATCH"}),
    ("double post", daily_double_post, {"POST_DOUBLE_OCCUPIED"}),
    ("register left empty", daily_register_left_empty, {"REGISTERS_1_4_UNSTAFFED"}),
    ("coverage gap", daily_coverage_gap, {"COVERAGE_GAP", "POST_EMPTY"}),
    (
        "self-checkout only worker on a register",
        daily_self_checkout_only,
        {"SELF_CHECKOUT_ONLY_VIOLATION"},
    ),
    ("inconsistent handover", daily_bad_handover, {"HANDOVER_INCONSISTENT"}),
    ("invalid relief", daily_bad_relief, {"RELIEF_INVALID"}),
    ("margins on a daily shift", daily_bad_margins, {"PATTERN_MARGINS"}),
]


@pytest.mark.parametrize(
    ("name", "mutation", "expected"), DAILY_CASES, ids=[c[0] for c in DAILY_CASES]
)
def test_every_planted_daily_violation_is_rejected_with_the_right_reason(
    name: str, mutation: DailyMutation, expected: set[str]
) -> None:
    plan, requirement = valid_daily()
    broken, new_requirement = mutation(plan, requirement)
    report = check_daily(broken, RULES, new_requirement)
    assert expected <= report.codes(), f"{name}: {report.summary()}"


def test_moved_breaks_are_warnings_and_far_breaks_are_errors() -> None:
    plan, requirement = valid_daily()
    moved, _ = daily_moved_break(plan, requirement)
    far, _ = daily_far_break(plan, requirement)
    moved_report = check_daily(moved, RULES, requirement)
    assert moved_report.ok, moved_report.summary()
    assert moved_report.codes("warning") == {"BREAK_MOVED"}
    assert not check_daily(far, RULES, requirement).ok


def test_the_catalogue_describes_every_code_the_checker_can_raise() -> None:
    assert all(severity in ("error", "warning") for severity, _ in CATALOGUE.values())
    assert all(text for _, text in CATALOGUE.values())
    assert len(CATALOGUE) >= 30
    raised = set()
    for _, _mutation, expected in ROSTER_CASES:
        raised |= expected
    for _, _mutation, expected in DAILY_CASES:
        raised |= expected
    raised |= {"POTENTIAL_ABSENCE"}
    assert raised <= set(CATALOGUE)
    assert worker("Z").worker_id == "Z"
    assert set(CATALOGUE) - raised == set()


def test_posts_dataclasses_can_be_constructed_with_defaults() -> None:
    segment = PostSegment("A", MONDAY, 510, 520, "R01")
    assert segment.end - segment.start == 10
    assert DailyPlan(MONDAY, {}, {}).post_at("A", 510) is None
