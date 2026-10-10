"""Assignment of workers to posts, minute by minute, around breaks and floating time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
from scipy.optimize import linear_sum_assignment

from ree.planner.config import PlannerConfig
from ree.planner.daily.breaks import STEP, shift_span, slot_lookup
from ree.planner.domain.demand_types import RequirementTable, SlotRequirement
from ree.planner.domain.plan import (
    BREAK,
    FLOATER,
    BreakEntry,
    WorkerRecord,
    is_register_post,
    is_self_checkout_post,
    register_number,
    register_post,
    self_checkout_post,
)
from ree.planner.domain.shifts import ShiftPattern

FORBIDDEN_COST = 1.0e7
FILL_BONUS = 1.0e4
PRIORITY_STEP = 50.0

Timeline = dict[str, dict[int, str]]


def required_posts(slot: SlotRequirement, opening_order: Sequence[int]) -> list[str]:
    """Posts a slot needs, in the order they are filled.

    The two priority registers come first, then a self-checkout place, then the other
    registers and self-checkout places.
    """
    posts = [register_post(number) for number in opening_order[: min(2, slot.registers)]]
    supervision: list[str] = [
        self_checkout_post(1, index) for index in range(1, slot.sco_zone1 + 1)
    ] + [self_checkout_post(2, index) for index in range(1, slot.sco_zone2 + 1)]
    if supervision:
        posts.append(supervision.pop(0))
    posts.extend(register_post(number) for number in opening_order[2 : slot.registers])
    posts.extend(supervision)
    return posts


def post_importance(post: str, opening_order: Sequence[int], config: PlannerConfig) -> float:
    """How much a post matters, from its place in the opening order or its type."""
    daily = config.daily
    if is_self_checkout_post(post):
        return daily.importance_sco
    position = list(opening_order).index(register_number(post))
    if position < 2:
        return daily.importance_priority
    if position < daily.main_registers:
        return daily.importance_main
    return daily.importance_other


def can_hold(worker: WorkerRecord, post: str) -> bool:
    """Whether a worker may hold a post, given self-checkout-only and cleared status."""
    if worker.self_checkout_only and is_register_post(post):
        return False
    return not (is_self_checkout_post(post) and not worker.cleared_self_checkout)


def _pair_cost(
    worker: WorkerRecord,
    post: str,
    previous: str | None,
    skill: float,
    weight: float,
    importance: float,
    config: PlannerConfig,
    reference_post: str | None = None,
    change_penalty: float = 0.0,
) -> float:
    daily = config.daily
    is_post = post != FLOATER
    if is_post and not can_hold(worker, post):
        return FORBIDDEN_COST
    cost = 0.0
    if is_post:
        cost -= daily.theta * weight * skill * importance
        if worker.preferred_self_checkout:
            cost += (
                -daily.preferred_sco_bonus
                if is_self_checkout_post(post)
                else (daily.preferred_sco_bonus)
            )
    held_before = previous is not None and previous not in (FLOATER, BREAK)
    if held_before and post != previous:
        cost += daily.handover_cost
    if reference_post is not None and post != reference_post:
        cost += change_penalty
    return cost


def assign_posts(
    day: date,
    shifts: Mapping[str, ShiftPattern],
    breaks: Sequence[BreakEntry],
    requirement: RequirementTable,
    z: Mapping[str, float],
    workers: Mapping[str, WorkerRecord],
    config: PlannerConfig,
    *,
    frozen: Timeline | None = None,
    from_minute: int | None = None,
    reference: Timeline | None = None,
    change_penalty: float = 0.0,
) -> Timeline:
    """Assign every worker to a post, a break or floating on a 5-minute grid.

    Each minute is a minimum-cost matching: better scores go to the more important posts,
    and a change of post costs something. Minutes before from_minute can be frozen, and a
    reference timeline can be kept stable with a change penalty.
    """
    open_minute, close_minute = requirement.opening_hours(day) or shift_span(shifts)
    times = range(open_minute, close_minute, STEP)
    slots = slot_lookup(requirement, day, times)
    on_break = {(entry.worker_id, minute) for entry in breaks for minute in _cells(entry)}
    timeline: Timeline = {worker_id: {} for worker_id in shifts}
    previous: dict[str, str | None] = {worker_id: None for worker_id in shifts}
    order = requirement.opening_order
    daily = config.daily
    for minute in times:
        if frozen is not None and from_minute is not None and minute < from_minute:
            for worker_id in shifts:
                kept = frozen.get(worker_id, {}).get(minute)
                if kept is not None:
                    timeline[worker_id][minute] = kept
                previous[worker_id] = kept
            continue
        present = [
            worker_id
            for worker_id in sorted(shifts)
            if any(block.on_register_at(minute) for block in shifts[worker_id].blocks)
        ]
        for worker_id in shifts:
            if worker_id not in present:
                previous[worker_id] = None
        resting = [worker_id for worker_id in present if (worker_id, minute) in on_break]
        for worker_id in resting:
            timeline[worker_id][minute] = BREAK
            previous[worker_id] = BREAK
        free = [worker_id for worker_id in present if (worker_id, minute) not in on_break]
        if not free:
            continue
        slot = slots[minute]
        posts = required_posts(slot, order)[: len(free)] if slot is not None else []
        columns = posts + [FLOATER] * len(free)
        weight = 1.0 if slot is not None and slot.rush else daily.calm_weight
        cost = np.zeros((len(free), len(columns)))
        for row, worker_id in enumerate(free):
            skill = z.get(worker_id, 0.0)
            reference_post = reference.get(worker_id, {}).get(minute) if reference else None
            for column, post in enumerate(columns):
                importance = 0.0 if post == FLOATER else post_importance(post, order, config)
                cost[row, column] = _pair_cost(
                    workers[worker_id],
                    post,
                    previous[worker_id],
                    skill,
                    weight,
                    importance,
                    config,
                    reference_post,
                    change_penalty,
                )
                if post != FLOATER and column < len(posts):
                    cost[row, column] -= FILL_BONUS - PRIORITY_STEP * column
        rows, chosen = linear_sum_assignment(cost)
        for row, column in zip(rows, chosen, strict=True):
            worker_id = free[row]
            timeline[worker_id][minute] = columns[column]
            previous[worker_id] = columns[column]
    return timeline


def _cells(entry: BreakEntry) -> range:
    return range(entry.start, entry.end, STEP)
