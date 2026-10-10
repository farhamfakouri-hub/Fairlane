"""Independent plan checker: tests rosters and daily plans against the labour and store rules.

It shares no code with the solvers, so a plan from any source can be judged by it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from itertools import pairwise

import numpy as np

from ree.planner.checker.report import Report, Violation, violation
from ree.planner.config import LabourRules
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import (
    BREAK,
    FLOATER,
    DailyPlan,
    RosterPlan,
    is_register_post,
    is_self_checkout_post,
    register_number,
    register_post,
    self_checkout_zone,
)
from ree.planner.domain.rules import block_break_length, break_window, grid_down
from ree.planner.domain.shifts import DAY_MINUTES, ShiftPattern

EPSILON = 1e-6
PRIORITY_REGISTERS = (1, 4)

Problem = tuple[str, str, float | None]


def pattern_problems(pattern: ShiftPattern, rules: LabourRules) -> list[Problem]:
    """Problems inside one shift pattern: window order, margins, block length, hours, cuts."""
    problems: list[Problem] = []
    if not pattern.blocks:
        return [("PATTERN_WINDOW_ORDER", "the pattern has no blocks", None)]
    tolerance = rules.margin_tolerance_minutes
    previous_end: int | None = None
    for index, block in enumerate(pattern.blocks, start=1):
        ordered = block.paid_start <= block.register_start < block.register_end <= block.paid_end
        if not ordered:
            problems.append(
                (
                    "PATTERN_WINDOW_ORDER",
                    f"block {index}: register window leaves its paid window",
                    None,
                )
            )
        if previous_end is not None and block.paid_start < previous_end:
            problems.append(
                ("PATTERN_WINDOW_ORDER", f"block {index} overlaps the previous block", None)
            )
        previous_end = block.paid_end
        before = block.register_start - block.paid_start
        after = block.paid_end - block.register_end
        wrong_before = abs(before - rules.clock_in_margin_minutes) > tolerance
        wrong_after = abs(after - rules.clock_out_margin_minutes) > tolerance
        if wrong_before or wrong_after:
            problems.append(
                (
                    "PATTERN_MARGINS",
                    f"block {index}: paid {before} minutes before and {after} after the register, "
                    f"expected {rules.clock_in_margin_minutes} "
                    f"and {rules.clock_out_margin_minutes}",
                    None,
                )
            )
        if block.register_minutes + EPSILON < rules.min_block_hours * 60:
            problems.append(
                (
                    "BLOCK_TOO_SHORT",
                    f"block {index} has {block.register_minutes} register minutes, "
                    f"the minimum is {rules.min_block_hours * 60:.0f}",
                    float(block.register_minutes),
                )
            )
    if pattern.paid_minutes > rules.max_daily_hours * 60 + EPSILON:
        problems.append(
            (
                "MAX_DAILY_HOURS",
                f"{pattern.paid_minutes / 60:.2f} paid hours, the daily maximum is "
                f"{rules.max_daily_hours:g}",
                pattern.paid_minutes / 60,
            )
        )
    cuts = pattern.cuts
    if len(cuts) > rules.max_cuts_per_day:
        problems.append(
            ("TOO_MANY_CUTS", f"{len(cuts)} cuts, the limit is {rules.max_cuts_per_day}", None)
        )
    for cut in cuts:
        if not rules.cut_minutes <= cut <= rules.cut_max_minutes:
            problems.append(
                (
                    "CUT_LENGTH",
                    f"a cut of {cut} minutes, allowed {rules.cut_minutes} "
                    f"to {rules.cut_max_minutes}",
                    float(cut),
                )
            )
    return problems


def check_assignments(
    plan: RosterPlan, rules: LabourRules, requirement: RequirementTable | None
) -> list[Violation]:
    """Checks on single assignments.

    These are a known worker, one pattern a day, the shape of the pattern, availability,
    opening hours and not covering both the opening and the closing.
    """
    found: list[Violation] = []
    counts: dict[tuple[str, date], int] = defaultdict(int)
    for assignment in plan.assignments:
        worker = plan.workers.get(assignment.worker_id)
        if worker is None:
            found.append(
                violation(
                    "UNKNOWN_WORKER",
                    f"worker {assignment.worker_id} is not in the plan",
                    assignment.worker_id,
                    assignment.day,
                )
            )
            continue
        counts[(assignment.worker_id, assignment.day)] += 1
        pattern = assignment.pattern
        for code, message, amount in pattern_problems(pattern, rules):
            found.append(
                violation(code, message, assignment.worker_id, assignment.day, None, amount)
            )
        for block in pattern.blocks:
            if worker.is_blocked(assignment.day, block.paid_start, block.paid_end):
                found.append(
                    violation(
                        "AVAILABILITY",
                        "assigned while blocked or on leave",
                        assignment.worker_id,
                        assignment.day,
                        block.paid_start,
                    )
                )
            elif worker.is_softly_blocked(assignment.day, block.paid_start, block.paid_end):
                found.append(
                    violation(
                        "POTENTIAL_ABSENCE",
                        "assigned in a window marked as a possible absence",
                        assignment.worker_id,
                        assignment.day,
                        block.paid_start,
                    )
                )
        hours = requirement.opening_hours(assignment.day) if requirement is not None else None
        if hours is not None and pattern.blocks:
            open_minute, close_minute = hours
            if pattern.register_start < open_minute or pattern.register_end > close_minute:
                found.append(
                    violation(
                        "OUTSIDE_OPENING_HOURS",
                        f"register time {pattern.register_start} to {pattern.register_end} "
                        f"outside {open_minute} to {close_minute}",
                        assignment.worker_id,
                        assignment.day,
                    )
                )
            long_day = close_minute - open_minute > rules.max_daily_hours * 60
            opens = pattern.register_start <= open_minute + rules.opening_slack_minutes
            closes = pattern.register_end >= close_minute - rules.closing_slack_minutes
            if long_day and opens and closes:
                found.append(
                    violation(
                        "OPEN_AND_CLOSE",
                        "the same pattern covers the opening and the closing",
                        assignment.worker_id,
                        assignment.day,
                    )
                )
    for (worker_id, day), count in counts.items():
        if count > 1:
            found.append(
                violation(
                    "TWO_PATTERNS_SAME_DAY",
                    f"{count} patterns on the same day",
                    worker_id,
                    day,
                    None,
                    float(count),
                )
            )
    return found


def _worker_day_patterns(plan: RosterPlan) -> dict[str, dict[int, ShiftPattern]]:
    table: dict[str, dict[int, ShiftPattern]] = defaultdict(dict)
    for assignment in plan.assignments:
        index = (assignment.day - plan.start_date).days
        table[assignment.worker_id].setdefault(index, assignment.pattern)
    return table


def check_worker_rules(plan: RosterPlan, rules: LabourRules) -> list[Violation]:
    """Checks over each worker's weeks.

    These are contract and maximum hours, rest between shifts, weekly rest and the number
    of consecutive working days.
    """
    found: list[Violation] = []
    table = _worker_day_patterns(plan)
    total_days = 7 * plan.weeks
    for worker_id, worker in plan.workers.items():
        days = table.get(worker_id, {})
        tolerance = (
            worker.hours_tolerance
            if worker.hours_tolerance is not None
            else rules.hours_tolerance_hours
        )
        weekly: list[float] = []
        for week in range(plan.weeks):
            paid = sum(
                days[index].paid_minutes for index in range(7 * week, 7 * week + 7) if index in days
            )
            hours = paid / 60.0
            weekly.append(hours)
            week_start = plan.start_date.fromordinal(plan.start_date.toordinal() + 7 * week)
            if abs(hours - worker.contract_hours) > tolerance + EPSILON:
                found.append(
                    violation(
                        "WEEKLY_HOURS_CONTRACT",
                        f"{hours:.2f} paid hours in the week, contract {worker.contract_hours:g} "
                        f"with tolerance {tolerance:g}",
                        worker_id,
                        week_start,
                        None,
                        hours - worker.contract_hours,
                    )
                )
            if hours > rules.max_weekly_hours + EPSILON:
                found.append(
                    violation(
                        "MAX_WEEKLY_HOURS",
                        f"{hours:.2f} paid hours, the weekly maximum is {rules.max_weekly_hours:g}",
                        worker_id,
                        week_start,
                        None,
                        hours,
                    )
                )
        if weekly and sum(weekly) / len(weekly) > rules.average_weekly_hours + EPSILON:
            found.append(
                violation(
                    "AVERAGE_WEEKLY_HOURS",
                    f"average {sum(weekly) / len(weekly):.2f} paid hours a week over {len(weekly)} "
                    f"weeks, the limit is {rules.average_weekly_hours:g}",
                    worker_id,
                    plan.start_date,
                    None,
                    sum(weekly) / len(weekly),
                )
            )
        found.extend(_rest_violations(plan, worker_id, days, rules))
        found.extend(_streak_violations(plan, worker_id, days, rules, total_days))
        found.extend(_weekly_rest_violations(plan, worker_id, days, rules))
    return found


def _date_of(plan: RosterPlan, index: int) -> date:
    return plan.start_date.fromordinal(plan.start_date.toordinal() + index)


def _rest_violations(
    plan: RosterPlan, worker_id: str, days: dict[int, ShiftPattern], rules: LabourRules
) -> list[Violation]:
    found: list[Violation] = []
    for index in sorted(days):
        following = days.get(index + 1)
        if following is None:
            continue
        rest = following.paid_start + DAY_MINUTES - days[index].paid_end
        if rest + EPSILON < rules.min_rest_hours * 60:
            found.append(
                violation(
                    "MIN_REST",
                    f"{rest / 60:.2f} hours of rest before the next shift, the minimum is "
                    f"{rules.min_rest_hours:g}",
                    worker_id,
                    _date_of(plan, index + 1),
                    None,
                    rest / 60,
                )
            )
    return found


def _streak_violations(
    plan: RosterPlan,
    worker_id: str,
    days: dict[int, ShiftPattern],
    rules: LabourRules,
    total_days: int,
) -> list[Violation]:
    found: list[Violation] = []
    streak = 0
    for index in range(total_days):
        streak = streak + 1 if index in days else 0
        if streak == rules.max_consecutive_days + 1:
            found.append(
                violation(
                    "MAX_CONSECUTIVE_DAYS",
                    f"more than {rules.max_consecutive_days} consecutive working days",
                    worker_id,
                    _date_of(plan, index),
                    None,
                    float(streak),
                )
            )
    return found


def _weekly_rest_violations(
    plan: RosterPlan, worker_id: str, days: dict[int, ShiftPattern], rules: LabourRules
) -> list[Violation]:
    blocks: list[tuple[int, int]] = []
    for index in sorted(days):
        for block in days[index].blocks:
            blocks.append(
                (index * DAY_MINUTES + block.paid_start, index * DAY_MINUTES + block.paid_end)
            )
    blocks.sort()
    needed = rules.weekly_rest_hours * 60
    gaps: list[tuple[float, float]] = []
    previous_end = float("-inf")
    for start, end in blocks:
        gaps.append((previous_end, float(start)))
        previous_end = float(max(previous_end, end))
    gaps.append((previous_end, float("inf")))
    day_part = (rules.weekly_rest_hours - rules.min_rest_hours) * 60
    found: list[Violation] = []
    for week in range(plan.weeks):
        week_start = week * 7 * DAY_MINUTES
        week_end = week_start + 7 * DAY_MINUTES
        satisfied = any(
            end - start >= needed - EPSILON
            and min(end, week_end) - max(start, week_start) >= day_part - EPSILON
            for start, end in gaps
        )
        if not satisfied:
            found.append(
                violation(
                    "WEEKLY_REST",
                    f"no rest of {rules.weekly_rest_hours:g} consecutive hours in week {week + 1}",
                    worker_id,
                    _date_of(plan, 7 * week),
                )
            )
    return found


def check_coverage(
    plan: RosterPlan, rules: LabourRules, requirement: RequirementTable
) -> list[Violation]:
    """Checks of a roster against the requirement: registers 1 and 4, headcount, self-checkout."""
    found: list[Violation] = []
    by_day: dict[date, list] = defaultdict(list)
    for assignment in plan.assignments:
        if assignment.worker_id in plan.workers:
            by_day[assignment.day].append(assignment)
    for day in plan.dates():
        hours = requirement.opening_hours(day)
        if hours is None:
            continue
        open_minute, close_minute = hours
        length = close_minute - open_minute
        everyone = np.zeros(length, dtype=np.int64)
        registers = np.zeros(length, dtype=np.int64)
        cleared = np.zeros(length, dtype=np.int64)
        for assignment in by_day.get(day, []):
            worker = plan.workers[assignment.worker_id]
            for block in assignment.pattern.blocks:
                low = max(block.register_start, open_minute) - open_minute
                high = min(block.register_end, close_minute) - open_minute
                if high <= low:
                    continue
                everyone[low:high] += 1
                if not worker.self_checkout_only:
                    registers[low:high] += 1
                if worker.cleared_self_checkout or worker.self_checkout_only:
                    cleared[low:high] += 1
        short = np.flatnonzero(registers < len(PRIORITY_REGISTERS))
        if short.size:
            found.append(
                violation(
                    "REGISTERS_1_4_UNSTAFFED",
                    f"registers 1 and 4 need two register workers; short for {short.size} minutes",
                    None,
                    day,
                    open_minute + int(short[0]),
                    float(short.size),
                )
            )
        for slot_start, slot in requirement.slots(day):
            low = slot_start - open_minute
            high = min(low + requirement.slot_minutes, length)
            need = slot.headcount * (high - low)
            have = int(everyone[low:high].sum())
            if have < need:
                found.append(
                    violation(
                        "UNDERSTAFFED",
                        f"{have / (high - low):.1f} people on duty, {slot.headcount} needed",
                        None,
                        day,
                        slot_start,
                        float(need - have),
                    )
                )
            sco_have = int(cleared[low:high].sum())
            sco_need = slot.k_sc * (high - low)
            if sco_have < sco_need:
                found.append(
                    violation(
                        "SELF_CHECKOUT_UNCOVERED",
                        f"{sco_have / (high - low):.1f} cleared workers on duty, "
                        f"{slot.k_sc} needed",
                        None,
                        day,
                        slot_start,
                        float(sco_need - sco_have),
                    )
                )
    return found


def check_roster(
    plan: RosterPlan, rules: LabourRules, requirement: RequirementTable | None = None
) -> Report:
    """Check a roster against every rule; coverage is checked when a requirement is given."""
    report = Report()
    report.extend(check_assignments(plan, rules, requirement))
    report.extend(check_worker_rules(plan, rules))
    if requirement is not None:
        report.extend(check_coverage(plan, rules, requirement))
    return report


def _window_mask(windows: list[tuple[int, int]], origin: int, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in windows:
        low = max(start - origin, 0)
        high = min(end - origin, length)
        if high > low:
            mask[low:high] = True
    return mask


def check_breaks(plan: DailyPlan, rules: LabourRules) -> list[Violation]:
    """Checks of the breaks of a day: presence, length, window and limits on simultaneous breaks."""
    found: list[Violation] = []
    known: set[tuple[str, int]] = set()
    for worker_id, pattern in plan.shifts.items():
        for index, block in enumerate(pattern.blocks):
            known.add((worker_id, index))
            expected = block_break_length(block, rules)
            entries = [
                item
                for item in plan.breaks
                if item.worker_id == worker_id and item.block_index == index
            ]
            if expected == 0:
                continue
            if not entries:
                found.append(
                    violation(
                        "BREAK_MISSING",
                        f"block {index + 1} needs a break of {expected} minutes",
                        worker_id,
                        plan.day,
                    )
                )
                continue
            if len(entries) > 1:
                found.append(
                    violation(
                        "BREAK_DUPLICATE",
                        f"block {index + 1} has {len(entries)} breaks",
                        worker_id,
                        plan.day,
                    )
                )
            window = break_window(block, rules)
            nominal = grid_down(block.register_start + block.register_minutes // 2)
            for entry in entries:
                if entry.minutes != expected:
                    found.append(
                        violation(
                            "BREAK_LENGTH",
                            f"break of {entry.minutes} minutes, expected {expected}",
                            worker_id,
                            plan.day,
                            entry.start,
                            float(entry.minutes),
                        )
                    )
                if entry.start < block.register_start or entry.end > block.register_end:
                    found.append(
                        violation(
                            "BREAK_OUTSIDE_SHIFT",
                            "the break is not inside the register time of its block",
                            worker_id,
                            plan.day,
                            entry.start,
                        )
                    )
                    continue
                if window is not None and not window[0] <= entry.start <= window[1]:
                    deviation = abs(entry.start - nominal)
                    code = (
                        "BREAK_MOVED"
                        if deviation <= rules.max_break_shift_minutes
                        else "BREAK_OUTSIDE_WINDOW"
                    )
                    found.append(
                        violation(
                            code,
                            f"break starts {deviation} minutes from the middle of the block, "
                            f"window {window[0]} to {window[1]}",
                            worker_id,
                            plan.day,
                            entry.start,
                            float(deviation),
                        )
                    )
    for entry in plan.breaks:
        if (entry.worker_id, entry.block_index) not in known:
            found.append(
                violation(
                    "BREAK_UNKNOWN",
                    "the break belongs to no block of the worker's shift",
                    entry.worker_id,
                    plan.day,
                    entry.start,
                )
            )
    found.extend(_simultaneous_breaks(plan, rules))
    return found


def _simultaneous_breaks(plan: DailyPlan, rules: LabourRules) -> list[Violation]:
    if not plan.breaks:
        return []
    low = min(item.start for item in plan.breaks)
    high = max(item.end for item in plan.breaks)
    counts = np.zeros(high - low, dtype=np.int64)
    for item in plan.breaks:
        counts[item.start - low : item.end - low] += 1
    peak = int(counts.max())
    first = low + int(np.argmax(counts))
    if peak > rules.limit_case_simultaneous_breaks:
        return [
            violation(
                "TOO_MANY_BREAKS",
                f"{peak} workers on break at the same time",
                None,
                plan.day,
                first,
                float(peak),
            )
        ]
    if peak > rules.max_simultaneous_breaks:
        return [
            violation(
                "BREAKS_LIMIT_CASE",
                f"{peak} workers on break at the same time (limit case)",
                None,
                plan.day,
                first,
                float(peak),
            )
        ]
    return []


def check_segments(plan: DailyPlan) -> list[Violation]:
    """Checks that post segments cover the register time exactly, with no double booking."""
    found: list[Violation] = []
    by_worker: dict[str, list] = defaultdict(list)
    for segment in plan.segments:
        by_worker[segment.worker_id].append(segment)
    for worker_id in by_worker:
        if worker_id not in plan.shifts:
            found.append(
                violation(
                    "UNKNOWN_WORKER", "segments for a worker without a shift", worker_id, plan.day
                )
            )
    for worker_id, pattern in plan.shifts.items():
        if not pattern.blocks:
            continue
        segments = sorted(by_worker.get(worker_id, []), key=lambda item: (item.start, item.end))
        for earlier, later in pairwise(segments):
            if later.start < earlier.end:
                found.append(
                    violation(
                        "DOUBLE_BOOKED",
                        f"{earlier.post} and {later.post} overlap",
                        worker_id,
                        plan.day,
                        later.start,
                        float(earlier.end - later.start),
                    )
                )
        windows = [(block.register_start, block.register_end) for block in pattern.blocks]
        origin = min(start for start, _ in windows)
        length = max(end for _, end in windows) - origin
        wanted = _window_mask(windows, origin, length)
        covered = np.zeros(length, dtype=np.int64)
        outside = 0
        for segment in segments:
            low = segment.start - origin
            high = segment.end - origin
            inside_low = max(low, 0)
            inside_high = min(high, length)
            if inside_high > inside_low:
                covered[inside_low:inside_high] += 1
            outside += max(0, -low) + max(0, high - length)
        mismatch = (
            int(np.sum(wanted & (covered == 0))) + int(np.sum(~wanted & (covered > 0))) + outside
        )
        if mismatch:
            found.append(
                violation(
                    "SEGMENT_MISMATCH",
                    f"the timeline differs from the register time by {mismatch} minutes",
                    worker_id,
                    plan.day,
                    None,
                    float(mismatch),
                )
            )
        break_segments = sorted((s.start, s.end) for s in segments if s.post == BREAK)
        break_entries = sorted(
            (item.start, item.end) for item in plan.breaks if item.worker_id == worker_id
        )
        if break_segments != break_entries:
            found.append(
                violation(
                    "BREAK_SEGMENT_MISMATCH",
                    "break entries and break segments disagree",
                    worker_id,
                    plan.day,
                )
            )
        worker = plan.workers.get(worker_id)
        if worker is not None and worker.self_checkout_only:
            for segment in segments:
                if is_register_post(segment.post):
                    found.append(
                        violation(
                            "SELF_CHECKOUT_ONLY_VIOLATION",
                            f"a self-checkout-only worker holds {segment.post}",
                            worker_id,
                            plan.day,
                            segment.start,
                        )
                    )
    return found


def check_posts(plan: DailyPlan, requirement: RequirementTable | None) -> list[Violation]:
    """Checks that each required post is held by one worker, and registers 1 and 4 are staffed."""
    found: list[Violation] = []
    by_post: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for segment in plan.segments:
        if segment.post not in (FLOATER, BREAK):
            by_post[segment.post].append((segment.start, segment.end))
    for post, intervals in by_post.items():
        intervals.sort()
        for earlier, later in pairwise(intervals):
            if later[0] < earlier[1]:
                found.append(
                    violation(
                        "POST_DOUBLE_OCCUPIED",
                        f"{post} is held by two workers at once",
                        None,
                        plan.day,
                        later[0],
                        float(earlier[1] - later[0]),
                    )
                )
    hours = requirement.opening_hours(plan.day) if requirement is not None else None
    if requirement is None or hours is None:
        return found
    open_minute, close_minute = hours
    length = close_minute - open_minute
    registers_present = {
        number: np.zeros(length, dtype=bool)
        for number in range(1, len(requirement.opening_order) + 1)
    }
    zone_counts = {1: np.zeros(length, dtype=np.int64), 2: np.zeros(length, dtype=np.int64)}
    for segment in plan.segments:
        low = max(segment.start, open_minute) - open_minute
        high = min(segment.end, close_minute) - open_minute
        if high <= low:
            continue
        if is_register_post(segment.post):
            number = register_number(segment.post)
            if number in registers_present:
                registers_present[number][low:high] = True
        elif is_self_checkout_post(segment.post):
            zone = self_checkout_zone(segment.post)
            if zone in zone_counts:
                zone_counts[zone][low:high] += 1
    missing_priority = 0
    missing_other = 0
    first_priority: int | None = None
    for offset in range(length):
        minute = open_minute + offset
        slot = requirement.at(plan.day, minute)
        if slot is None:
            continue
        for number in requirement.opening_order[: slot.registers]:
            if not registers_present[number][offset]:
                if number in PRIORITY_REGISTERS:
                    missing_priority += 1
                    if first_priority is None:
                        first_priority = minute
                else:
                    missing_other += 1
        if zone_counts[1][offset] < slot.sco_zone1:
            missing_other += slot.sco_zone1 - int(zone_counts[1][offset])
        if zone_counts[2][offset] < slot.sco_zone2:
            missing_other += slot.sco_zone2 - int(zone_counts[2][offset])
    if missing_priority:
        found.append(
            violation(
                "REGISTERS_1_4_UNSTAFFED",
                f"registers 1 and 4 unstaffed for {missing_priority} post-minutes",
                None,
                plan.day,
                first_priority,
                float(missing_priority),
            )
        )
    if missing_other:
        found.append(
            violation(
                "POST_EMPTY",
                f"required posts empty for {missing_other} post-minutes",
                None,
                plan.day,
                None,
                float(missing_other),
            )
        )
    return found


def check_capacity(plan: DailyPlan, requirement: RequirementTable | None) -> list[Violation]:
    """Checks that people on duty minus people on break meet the requirement of each slot."""
    hours = requirement.opening_hours(plan.day) if requirement is not None else None
    if requirement is None or hours is None:
        return []
    open_minute, close_minute = hours
    length = close_minute - open_minute
    on_duty = np.zeros(length, dtype=np.int64)
    for pattern in plan.shifts.values():
        for block in pattern.blocks:
            low = max(block.register_start, open_minute) - open_minute
            high = min(block.register_end, close_minute) - open_minute
            if high > low:
                on_duty[low:high] += 1
    on_break = np.zeros(length, dtype=np.int64)
    for item in plan.breaks:
        low = max(item.start, open_minute) - open_minute
        high = min(item.end, close_minute) - open_minute
        if high > low:
            on_break[low:high] += 1
    deficit = 0
    first: int | None = None
    for offset in range(length):
        slot = requirement.at(plan.day, open_minute + offset)
        if slot is None:
            continue
        needed = slot.registers + slot.k_sc
        gap = needed - int(on_duty[offset] - on_break[offset])
        if gap > 0:
            deficit += gap
            if first is None:
                first = open_minute + offset
    if not deficit:
        return []
    return [
        violation(
            "COVERAGE_GAP",
            f"people on duty minus people on break fall short by {deficit} person-minutes",
            None,
            plan.day,
            first,
            float(deficit),
        )
    ]


def check_handovers(plan: DailyPlan) -> list[Violation]:
    """Checks that handovers and relief match the posts in the timeline."""
    found: list[Violation] = []
    step = plan.grid_minutes
    for handover in plan.handovers:
        before = plan.post_at(handover.worker_id, handover.minute - step)
        after = plan.post_at(handover.worker_id, handover.minute)
        if before != handover.from_post or after != handover.to_post:
            found.append(
                violation(
                    "HANDOVER_INCONSISTENT",
                    f"the timeline shows {before} then {after}, the handover says "
                    f"{handover.from_post} then {handover.to_post}",
                    handover.worker_id,
                    plan.day,
                    handover.minute,
                )
            )
        if handover.replaces is not None:
            replaced_before = plan.post_at(handover.replaces, handover.minute - step)
            replaced_after = plan.post_at(handover.replaces, handover.minute)
            if replaced_after != BREAK or replaced_before != handover.to_post:
                found.append(
                    violation(
                        "RELIEF_INVALID",
                        f"{handover.replaces} is not leaving {handover.to_post} "
                        "for a break at this time",
                        handover.worker_id,
                        plan.day,
                        handover.minute,
                    )
                )
    return found


def check_daily(
    plan: DailyPlan, rules: LabourRules, requirement: RequirementTable | None = None
) -> Report:
    """Check a daily plan against every rule."""
    report = Report()
    for worker_id, pattern in plan.shifts.items():
        for code, message, amount in pattern_problems(pattern, rules):
            report.extend([violation(code, message, worker_id, plan.day, None, amount)])
    report.extend(check_breaks(plan, rules))
    report.extend(check_segments(plan))
    report.extend(check_posts(plan, requirement))
    report.extend(check_capacity(plan, requirement))
    report.extend(check_handovers(plan))
    return report


def check(
    plan: RosterPlan | DailyPlan, rules: LabourRules, requirement: RequirementTable | None = None
) -> Report:
    """Check a roster or a daily plan.

    Every plan a solver produces, and every manual edit, goes through this function.
    """
    if isinstance(plan, RosterPlan):
        return check_roster(plan, rules, requirement)
    if isinstance(plan, DailyPlan):
        return check_daily(plan, rules, requirement)
    raise TypeError(f"cannot check a {type(plan).__name__}")


def required_registers(requirement: RequirementTable, day: date, minute: int) -> tuple[str, ...]:
    """Register posts that must be staffed at a minute."""
    return tuple(register_post(number) for number in requirement.required_registers(day, minute))
