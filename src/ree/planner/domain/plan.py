"""Plan data: workers, roster assignments, and daily plans with posts, breaks and handovers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

from ree.planner.domain.shifts import DAY_MINUTES, ShiftPattern

FLOATER = "FLOAT"
BREAK = "BREAK"
REGISTER_PREFIX = "R"
SELF_CHECKOUT_PREFIX = "SCO"
Window = tuple[int, int]


def register_post(number: int) -> str:
    """Post name of a register, such as "R04"."""
    return f"{REGISTER_PREFIX}{number:02d}"


def self_checkout_post(zone: int, index: int) -> str:
    """Post name of a self-checkout place, such as "SCO1-2"."""
    return f"{SELF_CHECKOUT_PREFIX}{zone}-{index}"


def is_register_post(post: str) -> bool:
    """Whether a post name is a register."""
    return post.startswith(REGISTER_PREFIX) and post[1:].isdigit()


def register_number(post: str) -> int:
    """Register number of a register post."""
    return int(post[1:])


def is_self_checkout_post(post: str) -> bool:
    """Whether a post name is a self-checkout place."""
    return post.startswith(SELF_CHECKOUT_PREFIX)


def self_checkout_zone(post: str) -> int:
    """Zone number of a self-checkout post."""
    return int(post[len(SELF_CHECKOUT_PREFIX)])


@dataclass(frozen=True)
class WorkerRecord:
    """A worker as the planner sees them: contract hours, abilities and unavailable times."""

    worker_id: str
    contract_hours: float
    contract_type: str = "full_time"
    hours_tolerance: float | None = None
    self_checkout_only: bool = False
    preferred_self_checkout: bool = False
    cleared_self_checkout: bool = True
    blocked: Mapping[date, tuple[Window, ...]] = field(default_factory=dict)
    soft_blocked: Mapping[date, tuple[Window, ...]] = field(default_factory=dict)

    def is_blocked(self, day: date, start: int, end: int) -> bool:
        """Whether a time the worker cannot work overlaps the interval on the day."""
        return any(
            start < blocked_end and blocked_start < end
            for blocked_start, blocked_end in self.blocked.get(day, ())
        )

    def is_softly_blocked(self, day: date, start: int, end: int) -> bool:
        """Whether a time the worker would rather avoid overlaps the interval."""
        return any(
            start < blocked_end and blocked_start < end
            for blocked_start, blocked_end in self.soft_blocked.get(day, ())
        )


@dataclass(frozen=True)
class Assignment:
    """A shift pattern given to a worker on a day."""

    worker_id: str
    day: date
    pattern: ShiftPattern


@dataclass
class RosterPlan:
    """A roster: the workers and the assignments over the weeks of a cycle."""

    start_date: date
    weeks: int
    workers: dict[str, WorkerRecord]
    assignments: list[Assignment] = field(default_factory=list)

    def dates(self) -> list[date]:
        """Every date of the cycle."""
        return [self.start_date + timedelta(days=offset) for offset in range(7 * self.weeks)]

    def week_of(self, day: date) -> int:
        """Index of the week of a date inside the cycle."""
        return (day - self.start_date).days // 7

    def by_worker_day(self) -> dict[tuple[str, date], list[ShiftPattern]]:
        """Patterns of each worker and day."""
        grouped: dict[tuple[str, date], list[ShiftPattern]] = defaultdict(list)
        for assignment in self.assignments:
            grouped[(assignment.worker_id, assignment.day)].append(assignment.pattern)
        return dict(grouped)

    def pattern_of(self, worker_id: str, day: date) -> ShiftPattern | None:
        """Pattern of a worker on a day, or None."""
        for assignment in self.assignments:
            if assignment.worker_id == worker_id and assignment.day == day:
                return assignment.pattern
        return None

    def worked_on(self, day: date) -> dict[str, ShiftPattern]:
        """Pattern of each worker who works on a day."""
        return {a.worker_id: a.pattern for a in self.assignments if a.day == day}

    def with_assignment(self, assignment: Assignment) -> RosterPlan:
        """Copy of the roster with one more assignment."""
        return RosterPlan(
            self.start_date, self.weeks, self.workers, [*self.assignments, assignment]
        )

    def without(self, worker_id: str, day: date) -> RosterPlan:
        """Copy of the roster without the assignment of a worker on a day."""
        kept = [a for a in self.assignments if not (a.worker_id == worker_id and a.day == day)]
        return RosterPlan(self.start_date, self.weeks, self.workers, kept)


@dataclass(frozen=True)
class BreakEntry:
    """A scheduled break: its block, start and length."""

    worker_id: str
    day: date
    block_index: int
    start: int
    minutes: int

    @property
    def end(self) -> int:
        """End minute of the break."""
        return self.start + self.minutes


@dataclass(frozen=True)
class PostSegment:
    """A stretch of time that a worker spends at one post."""

    worker_id: str
    day: date
    start: int
    end: int
    post: str


@dataclass(frozen=True)
class Handover:
    """A change of post at a minute, with the reason and the worker being replaced."""

    day: date
    minute: int
    worker_id: str
    from_post: str
    to_post: str
    reason: str
    replaces: str | None = None


@dataclass
class DailyPlan:
    """The plan of one day: shifts, post segments, breaks, handovers and alerts."""

    day: date
    shifts: dict[str, ShiftPattern]
    workers: dict[str, WorkerRecord]
    segments: list[PostSegment] = field(default_factory=list)
    breaks: list[BreakEntry] = field(default_factory=list)
    handovers: list[Handover] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    grid_minutes: int = 5

    def post_at(self, worker_id: str, minute: int) -> str | None:
        """Post of a worker at a minute, or None."""
        for segment in self.segments:
            if segment.worker_id == worker_id and segment.start <= minute < segment.end:
                return segment.post
        return None

    def occupants(self, minute: int) -> dict[str, str]:
        """Post of every worker who is at a post at a minute."""
        return {
            segment.worker_id: segment.post
            for segment in self.segments
            if segment.start <= minute < segment.end
        }


def day_start_minute(day: date, origin: date) -> int:
    """Minute offset of a day from an origin date."""
    return (day - origin).days * DAY_MINUTES
