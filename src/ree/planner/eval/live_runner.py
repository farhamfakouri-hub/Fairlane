"""Simulation of a plan with live re-planning by a simulated manager."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from ree.planner.config import PlannerConfig
from ree.planner.daily.daily_plan import DailyResult
from ree.planner.domain.plan import DailyPlan, is_register_post, register_number
from ree.planner.eval.plans import PlanBundle
from ree.planner.eval.world import SECONDS, DayRun, World, rush_lookup
from ree.planner.live.busyness import Reading, RegisterUsage, blended_rate, busyness
from ree.planner.live.proposals import Proposal, ProposalBook
from ree.planner.live.triggers import LiveController
from ree.planner.sim.part1_workforce import WorkerTruth, service_multiplier
from ree.planner.sim.queue_sim import (
    Observation,
    QueueSettings,
    StaffInterval,
    simulate_day,
)

CHECKPOINT_MINUTES = 5
LIVE_PLAN = "L"


@dataclass
class LiveStats:
    """Counts of the proposals, openings, closings and flips in a run."""

    proposed: int = 0
    accepted: int = 0
    opened: int = 0
    closed: int = 0
    covered: int = 0
    flips: int = 0


def register_intervals(plan: DailyPlan, absent: set[str]) -> dict[int, list[StaffInterval]]:
    """Register staffing of a daily plan, grouped by register."""
    grouped: dict[int, list[StaffInterval]] = defaultdict(list)
    for segment in plan.segments:
        if is_register_post(segment.post) and segment.worker_id not in absent:
            grouped[register_number(segment.post)].append(
                StaffInterval(
                    register_number(segment.post),
                    segment.start * SECONDS,
                    segment.end * SECONDS,
                    segment.worker_id,
                )
            )
    return {register: sorted(items, key=lambda i: i.start) for register, items in grouped.items()}


class LiveSession:
    """Controller that reads checkpoints, raises proposals and applies the accepted ones."""

    def __init__(
        self,
        result: DailyResult,
        bundle: PlanBundle,
        config: PlannerConfig,
        day: date,
        absent: set[str],
        policy: str,
    ) -> None:
        self.day = day
        self.config = config
        self.bundle = bundle
        self.absent = absent
        self.policy = policy
        self.book = ProposalBook(result.plan, bundle.requirement, config.rules)
        self.controller = LiveController(self.book, bundle.scores.z, config)
        self.current = register_intervals(result.plan, absent)
        self.stats = LiveStats()
        self.starts = {
            worker_id: pattern.register_start for worker_id, pattern in result.plan.shifts.items()
        }
        self.last_direction: str | None = None

    def __call__(self, observation: Observation) -> Mapping[int, Sequence[StaffInterval]] | None:
        minute = int(observation.second // 60)
        usage = [RegisterUsage(r, opened, busy) for r, opened, busy in observation.registers]
        reading = Reading(
            minute,
            busyness(usage),
            sum(1 for item in usage if item.open_seconds > 0.0),
            observation.waiting,
        )
        window_minutes = observation.window_seconds / 60.0
        observed = observation.arrivals / window_minutes if window_minutes else 0.0
        slot = self.bundle.requirement.at(self.day, minute)
        planned = (
            slot.planned_customers / self.bundle.requirement.slot_minutes
            if slot is not None
            else 0.0
        )
        rate = blended_rate(observed, planned, self.config.live)
        logged_in = {w for w in self.book.plan.shifts if w not in self.absent}
        made = self.controller.check_logins(minute, self.starts, logged_in)
        made += self.controller.observe(reading, rate)
        return self._decide(made, minute)

    def _decide(
        self, proposals: Sequence[Proposal], minute: int
    ) -> Mapping[int, Sequence[StaffInterval]] | None:
        changed = False
        for proposal in proposals:
            self.stats.proposed += 1
            if self.policy != "accept_all" or proposal.state != "pending":
                continue
            try:
                self.book.accept(proposal.proposal_id, minute, "accepted by the simulated manager")
            except ValueError:
                continue
            self.stats.accepted += 1
            changed = True
            if proposal.kind == "open_register":
                self.stats.opened += 1
                self.stats.flips += int(self.last_direction == "close")
                self.last_direction = "open"
            elif proposal.kind == "close_register":
                self.stats.closed += 1
                self.stats.flips += int(self.last_direction == "open")
                self.last_direction = "close"
            elif proposal.kind == "cover_absence":
                self.stats.covered += 1
        if not changed:
            return None
        updated = register_intervals(self.book.plan, self.absent)
        changes: dict[int, Sequence[StaffInterval]] = {}
        for register in set(updated) | set(self.current):
            new = updated.get(register, [])
            if new != self.current.get(register, []):
                changes[register] = new
        self.current = updated
        return changes


def simulate_plan_live(
    bundle: PlanBundle,
    world: World,
    truth: dict[str, WorkerTruth],
    policy: str = "accept_all",
) -> tuple[list[DayRun], LiveStats]:
    """Run a plan through a world with live re-planning, and count what happened."""
    config = world.config
    positions = tuple((r, i) for i, r in enumerate(bundle.requirement.opening_order))
    runs: list[DayRun] = []
    total = LiveStats()
    for day in world.days:
        result = bundle.daily.get(day)
        hours = bundle.requirement.opening_hours(day)
        if result is None or hours is None:
            continue
        absent = {
            worker_id
            for worker_id in result.plan.shifts
            if world.absence_draw.get((worker_id, day), 1.0) < world.scenario.absence_rate
        }
        session = LiveSession(result, bundle, config, day, absent, policy)
        intervals = [item for items in session.current.values() for item in items]
        rush = rush_lookup(bundle.requirement, day)
        cache: dict[tuple[str, bool], float] = {}

        def multiplier(worker_id: str, second: float, rush=rush, cache=cache) -> float:
            flag = rush(second)
            key = (worker_id, flag)
            if key not in cache:
                cache[key] = service_multiplier(truth[worker_id], flag)
            return cache[key]

        settings = QueueSettings(
            discipline=config.service.queue_discipline,
            position_penalty=config.service.position_penalty,
            queue_limit=config.staffing.queue_limit_customers,
            close_second=hours[1] * SECONDS,
            register_positions=positions,
            checkpoint_seconds=CHECKPOINT_MINUTES * SECONDS,
            checkpoint_start=(hours[0] + CHECKPOINT_MINUTES) * SECONDS,
            checkpoint_end=(hours[1] - CHECKPOINT_MINUTES) * SECONDS,
        )
        outcome = simulate_day(world.customers[day], intervals, multiplier, settings, session)
        flags = np.array([rush(float(t)) for t in outcome.arrival], dtype=bool)
        runs.append(DayRun(day, outcome, flags, (hours[1] - hours[0]) * SECONDS, len(absent)))
        for name in ("proposed", "accepted", "opened", "closed", "covered", "flips"):
            setattr(total, name, getattr(total, name) + getattr(session.stats, name))
    return runs, total
