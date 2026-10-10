"""Drafts of live changes: open or close a register, cover an absence, re-plan the day."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ree.planner.config import PlannerConfig
from ree.planner.daily.breaks import STEP
from ree.planner.daily.posts import Timeline, assign_posts, can_hold
from ree.planner.daily.relief import (
    build_segments,
    derive_handovers,
    is_required_post,
)
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import (
    BREAK,
    FLOATER,
    DailyPlan,
    is_register_post,
    register_number,
    register_post,
)
from ree.planner.domain.shifts import minutes_to_clock
from ree.planner.live.busyness import expected_wait_minutes
from ree.planner.live.proposals import (
    CLOSE_REGISTER,
    COVER_ABSENCE,
    OPEN_REGISTER,
    REPLAN,
    Change,
    Draft,
)


def grid_up(minute: int) -> int:
    """Round a minute up to the 5-minute grid."""
    return -(-minute // STEP) * STEP


def timeline_of(plan: DailyPlan) -> Timeline:
    """Post of each worker at each grid minute, from the post segments."""
    cells: Timeline = {worker_id: {} for worker_id in plan.shifts}
    for segment in plan.segments:
        for minute in range(segment.start, segment.end, STEP):
            cells[segment.worker_id][minute] = segment.post
    return cells


def rebuilt(plan: DailyPlan, cells: Timeline) -> DailyPlan:
    """Daily plan rebuilt from a changed timeline."""
    shifts = {
        worker_id: pattern for worker_id, pattern in plan.shifts.items() if worker_id in cells
    }
    segments = build_segments(plan.day, {w: cells[w] for w in shifts}, shifts)
    handovers = derive_handovers(plan.day, {w: cells[w] for w in shifts})
    return replace(
        plan,
        shifts=shifts,
        workers={w: plan.workers[w] for w in shifts},
        segments=segments,
        handovers=handovers,
        breaks=[entry for entry in plan.breaks if entry.worker_id in shifts],
        alerts=[],
    )


def diff_cells(old: Timeline, new: Timeline) -> tuple[Change, ...]:
    """Differences between two timelines, as runs of the same change."""
    changes: list[Change] = []
    for worker_id in sorted(set(old) | set(new)):
        before_cells = old.get(worker_id, {})
        after_cells = new.get(worker_id, {})
        run: Change | None = None
        for minute in sorted(set(before_cells) | set(after_cells)):
            before = before_cells.get(minute, "-")
            after = after_cells.get(minute, "-")
            if before == after:
                if run is not None:
                    changes.append(run)
                    run = None
                continue
            if (
                run is not None
                and run.before == before
                and run.after == after
                and run.end == minute
            ):
                run = replace(run, end=minute + STEP)
                continue
            if run is not None:
                changes.append(run)
            run = Change(worker_id, minute, minute + STEP, before, after)
        if run is not None:
            changes.append(run)
    return tuple(changes)


def open_registers_at(cells: Timeline, minute: int) -> list[int]:
    """Registers that are staffed at a minute."""
    return sorted(
        {
            register_number(post)
            for worker_cells in cells.values()
            for post in [worker_cells.get(minute)]
            if post is not None and is_register_post(post)
        }
    )


def floaters_at(plan: DailyPlan, cells: Timeline, minute: int) -> list[str]:
    """Workers floating at a minute who may take a register."""
    return [
        worker_id
        for worker_id in sorted(cells)
        if cells[worker_id].get(minute) == FLOATER
        and not plan.workers[worker_id].self_checkout_only
    ]


def _wait_pair(
    rate: float | None, service_minutes: float, before: int, after: int
) -> tuple[float | None, float | None]:
    if rate is None:
        return None, None
    return (
        expected_wait_minutes(rate, service_minutes, before),
        expected_wait_minutes(rate, service_minutes, after),
    )


def open_register(
    plan: DailyPlan,
    requirement: RequirementTable,
    z: Mapping[str, float],
    now: int,
    config: PlannerConfig,
    rate: float | None = None,
    service_minutes: float | None = None,
) -> Draft | None:
    """Draft that opens the next register with the best-scored floater after the freeze."""
    start = grid_up(now + config.live.freeze_minutes)
    cells = timeline_of(plan)
    opened = open_registers_at(cells, start)
    target = next((n for n in requirement.opening_order if n not in opened), None)
    if target is None:
        return None
    candidates = floaters_at(plan, cells, start)
    if not candidates:
        return None
    chosen = max(candidates, key=lambda w: (z.get(w, 0.0), w))
    post = register_post(target)
    updated = {worker_id: dict(worker_cells) for worker_id, worker_cells in cells.items()}
    minute = start
    while updated[chosen].get(minute) == FLOATER:
        updated[chosen][minute] = post
        minute += STEP
    before, after = _wait_pair(
        rate, service_minutes or config.staffing.service_minutes, len(opened), len(opened) + 1
    )
    reason = (
        f"Registers have been busier than {config.live.rho_open:.0%} for "
        f"{config.live.open_window_minutes} minutes: {chosen} opens {post} from "
        f"{minutes_to_clock(start)}"
    )
    return Draft(
        OPEN_REGISTER, reason, rebuilt(plan, updated), diff_cells(cells, updated), before, after
    )


def close_register(
    plan: DailyPlan,
    requirement: RequirementTable,
    now: int,
    config: PlannerConfig,
    rate: float | None = None,
    service_minutes: float | None = None,
) -> Draft | None:
    """Draft that closes the last register opened and lets its worker float."""
    start = grid_up(now + config.live.freeze_minutes)
    cells = timeline_of(plan)
    opened = open_registers_at(cells, start)
    minimum = config.staffing.minimum_open_registers
    if len(opened) <= minimum:
        return None
    order = list(requirement.opening_order)
    target = max(opened, key=lambda n: order.index(n))
    post = register_post(target)
    holders = [w for w in sorted(cells) if cells[w].get(start) == post]
    if not holders:
        return None
    updated = {worker_id: dict(worker_cells) for worker_id, worker_cells in cells.items()}
    for holder in holders:
        minute = start
        while updated[holder].get(minute) == post:
            updated[holder][minute] = FLOATER
            minute += STEP
    before, after = _wait_pair(
        rate, service_minutes or config.staffing.service_minutes, len(opened), len(opened) - 1
    )
    reason = (
        f"Registers have been quieter than {config.live.rho_close:.0%} for "
        f"{config.live.close_window_minutes} minutes: close {post} from "
        f"{minutes_to_clock(start)} and let {holders[0]} float"
    )
    return Draft(
        CLOSE_REGISTER, reason, rebuilt(plan, updated), diff_cells(cells, updated), before, after
    )


def cover_absence(
    plan: DailyPlan,
    worker_id: str,
    z: Mapping[str, float],
    now: int,
    config: PlannerConfig,
) -> Draft | None:
    """Draft in which floaters take over the posts of a worker who has not logged in."""
    if worker_id not in plan.shifts:
        return None
    start = grid_up(now)
    cells = timeline_of(plan)
    updated = {w: dict(worker_cells) for w, worker_cells in cells.items() if w != worker_id}
    uncovered = 0
    cover: str | None = None
    for minute in sorted(m for m in cells[worker_id] if m >= start):
        post = cells[worker_id][minute]
        if post in (FLOATER, BREAK) or not is_required_post(post):
            continue
        if cover is not None and updated[cover].get(minute) != FLOATER:
            cover = None
        if cover is None:
            options = [
                w
                for w in sorted(updated)
                if updated[w].get(minute) == FLOATER and can_hold(plan.workers[w], post)
            ]
            cover = max(options, key=lambda w: (z.get(w, 0.0), w)) if options else None
        if cover is None:
            uncovered += STEP
            continue
        updated[cover][minute] = post
    result = rebuilt(plan, updated)
    reason = f"{worker_id} has not logged in: floaters take the posts they held"
    if uncovered:
        reason += f"; {uncovered} post-minutes stay uncovered"
    return Draft(COVER_ABSENCE, reason, result, diff_cells(cells, updated))


def replan_rest_of_day(
    plan: DailyPlan,
    requirement: RequirementTable,
    z: Mapping[str, float],
    now: int,
    config: PlannerConfig,
    reason: str,
) -> Draft:
    """Draft that re-assigns posts for the rest of the day, keeping changes small."""
    start = grid_up(now + config.live.freeze_minutes)
    cells = timeline_of(plan)
    fresh = assign_posts(
        plan.day,
        plan.shifts,
        plan.breaks,
        requirement,
        z,
        plan.workers,
        config,
        frozen=cells,
        from_minute=start,
        reference=cells,
        change_penalty=config.live.stability_penalty,
    )
    return Draft(REPLAN, reason, rebuilt(plan, fresh), diff_cells(cells, fresh))
