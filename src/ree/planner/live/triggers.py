"""Live controller: turns busyness readings and login checks into proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ree.planner.config import PlannerConfig
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.live.busyness import BusynessTracker, Reading
from ree.planner.live.proposals import (
    CLOSE_REGISTER,
    OPEN_REGISTER,
    PENDING,
    REPLAN,
    STALE,
    Proposal,
    ProposalBook,
)
from ree.planner.live.replan import (
    close_register,
    cover_absence,
    floaters_at,
    grid_up,
    open_register,
    replan_rest_of_day,
    timeline_of,
)


@dataclass
class LiveController:
    """Watches readings and logins and raises proposals; nothing changes until a manager accepts."""

    book: ProposalBook
    z: Mapping[str, float]
    config: PlannerConfig
    tracker: BusynessTracker = field(default_factory=BusynessTracker)
    last_minute: dict[str, int] = field(default_factory=dict)
    absence_proposals: dict[str, int] = field(default_factory=dict)

    def _since(self, kind: str, now: int) -> int | None:
        previous = self.last_minute.get(kind)
        return None if previous is None else now - previous

    def _allowed(self, kind: str, now: int) -> bool:
        elapsed = self._since(kind, now)
        return elapsed is None or elapsed >= self.config.live.same_proposal_gap_minutes

    def _pending(self, kind: str) -> bool:
        return any(p.kind == kind and p.state == PENDING for p in self.book.proposals.values())

    def _quiet_since(self, kind: str, now: int, minutes: int) -> bool:
        elapsed = self._since(kind, now)
        return elapsed is None or elapsed >= minutes

    def observe(self, reading: Reading, rate: float | None = None) -> list[Proposal]:
        """Add a reading and return any proposals it triggers.

        Sustained high busyness proposes opening a register; sustained low busyness, with
        floaters to spare, proposes closing one.
        """
        now = reading.minute
        self.tracker.add(reading)
        live = self.config.live
        made: list[Proposal] = []
        if (
            self._allowed(OPEN_REGISTER, now)
            and not self._pending(OPEN_REGISTER)
            and self._quiet_since(CLOSE_REGISTER, now, live.open_window_minutes)
            and self.tracker.sustained_above(now, live.open_window_minutes, live.rho_open)
        ):
            draft = open_register(
                self.book.plan, self.book.requirement, self.z, now, self.config, rate
            )
            if draft is not None:
                made.append(self.book.propose(draft, now))
                self.last_minute[OPEN_REGISTER] = now
        elif (
            self._allowed(CLOSE_REGISTER, now)
            and not self._pending(CLOSE_REGISTER)
            and self._quiet_since(OPEN_REGISTER, now, live.close_window_minutes)
            and self.tracker.sustained_below(now, live.close_window_minutes, live.rho_close)
            and self._floaters(now) > 1
        ):
            draft = close_register(self.book.plan, self.book.requirement, now, self.config, rate)
            if draft is not None:
                made.append(self.book.propose(draft, now))
                self.last_minute[CLOSE_REGISTER] = now
        return made

    def _absence_handled(self, worker_id: str) -> bool:
        known = self.absence_proposals.get(worker_id)
        if known is None:
            return False
        if known == 0:
            return True
        return self.book.proposals[known].state != STALE

    def _floaters(self, now: int) -> int:
        start = grid_up(now + self.config.live.freeze_minutes)
        cells = timeline_of(self.book.plan)
        return len(floaters_at(self.book.plan, cells, start))

    def check_logins(
        self, now: int, shift_starts: Mapping[str, int], logged_in: set[str]
    ) -> list[Proposal]:
        """Propose cover for workers who have not logged in after the lateness delay."""
        made: list[Proposal] = []
        late = self.config.live.lateness_minutes
        for worker_id in sorted(shift_starts):
            if worker_id in logged_in or self._absence_handled(worker_id):
                continue
            if worker_id not in self.book.plan.shifts or now < shift_starts[worker_id] + late:
                continue
            draft = cover_absence(self.book.plan, worker_id, self.z, now, self.config)
            if draft is not None:
                proposal = self.book.propose(draft, now)
                self.absence_proposals[worker_id] = proposal.proposal_id
                made.append(proposal)
            else:
                self.absence_proposals[worker_id] = 0
        return made

    def manager_event(self, requirement: RequirementTable, now: int, reason: str) -> Proposal:
        """Propose a re-plan of the rest of the day for a reason given by a manager."""
        draft = replan_rest_of_day(self.book.plan, requirement, self.z, now, self.config, reason)
        self.last_minute[REPLAN] = now
        return self.book.propose(draft, now)
