"""Planner service: generates rosters, checks edits, plans days and handles live proposals.

Plans are proposals. A manager's override is applied only when it adds no rule error
and comes with a short reason, and everything is written to the audit log.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from ree.api.provider import ScoreProvider
from ree.planner.api.audit_store import AuditStore
from ree.planner.checker.checks import check_daily, check_roster
from ree.planner.checker.report import Report
from ree.planner.config import PlannerConfig
from ree.planner.daily.breaks import BreakSchedule
from ree.planner.daily.daily_plan import DailyResult, plan_day, plan_day_from_shifts
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import Assignment, BreakEntry, RosterPlan, WorkerRecord
from ree.planner.domain.shifts import ShiftPattern, make_block, minutes_to_clock, pattern_id_for
from ree.planner.live.busyness import Reading
from ree.planner.live.proposals import Proposal, ProposalBook
from ree.planner.live.triggers import LiveController
from ree.planner.roster.explain import Reason, explain_roster, trade_offs
from ree.planner.roster.model import ScoreInputs, roster_scores, score_inputs
from ree.planner.roster.solver import RosterSolution, plan_roster
from ree.planner.sim.demo import DemoData, demo_data

MIN_REASON_CHARACTERS = 3
WORKER_TEXT = {
    "RUSH_TOP_SCORE": "You work a busy period. Among the people available that day your "
    "recent peak score is among the strongest; scores only help choose between shifts of the "
    "same hours and the same burden, never how much you work.",
    "RUSH_EXPLORATION": "You were placed in a busy period so that new register data can be "
    "collected; your score is still provisional.",
    "BURDEN_ROTATION": "This shift is one that is shared out in turn; you have had fewer of "
    "this kind than average over the last weeks.",
    "SAME_AS_BEFORE": "This is the same shift as in your previous roster.",
    "COVERAGE_NEED": "The store needs people at this time.",
    "CONTRACT_HOURS": "Your shifts add up to your contract hours.",
}

Loader = Callable[..., DemoData]


@dataclass(frozen=True)
class Edit:
    """A manager's change of one worker's shift on a day; no pattern removes the shift."""

    worker_id: str
    day: date
    pattern_id: str | None


@dataclass
class RosterVersion:
    """A numbered roster with its inputs, reasons and notes."""

    number: int
    config: PlannerConfig
    workers: list[WorkerRecord]
    requirement: RequirementTable
    scores: ScoreInputs
    solution: RosterSolution
    explanations: dict[tuple[str, date], list[Reason]]
    notes: list[str]


@dataclass(frozen=True)
class EditOutcome:
    """Result of checking or applying an edit: applied or not, and the new rule errors."""

    applied: bool
    report: Report
    new_errors: tuple[str, ...]
    message: str


def parse_pattern(pattern_id: str, config: PlannerConfig) -> ShiftPattern:
    """Shift pattern from its code, such as "C0830-1630" or "S0930-1330+1430-1930"."""
    body = pattern_id[1:]
    blocks = []
    for part in body.split("+"):
        try:
            first, last = part.split("-")
            start = int(first[:2]) * 60 + int(first[2:])
            end = int(last[:2]) * 60 + int(last[2:])
        except ValueError as error:
            raise ValueError(f"{pattern_id!r} is not a shift code") from error
        if end <= start:
            raise ValueError(f"{pattern_id!r} ends before it starts")
        blocks.append(
            make_block(
                start,
                end,
                config.rules.clock_in_margin_minutes,
                config.rules.clock_out_margin_minutes,
            )
        )
    kind = "split" if len(blocks) > 1 else "continuous"
    return ShiftPattern(pattern_id_for(kind, tuple(blocks)), kind, tuple(blocks))


def _error_counts(report: Report) -> Counter[str]:
    return Counter(item.code for item in report.errors)


class PlannerService:
    """Holds the current roster and everything a manager does with it."""

    def __init__(
        self,
        loader: Loader = demo_data,
        audit: AuditStore | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.loader = loader
        self.audit = audit if audit is not None else AuditStore()
        self.clock = clock
        self.version: RosterVersion | None = None
        self.counter = 0
        self.daily_cache: dict[date, DailyResult] = {}
        self.live: dict[date, tuple[ProposalBook, LiveController]] = {}

    def _log(self, kind: str, subject: str, detail: dict[str, object]) -> None:
        self.audit.append(kind, subject, detail, self.clock())

    def current(self) -> RosterVersion:
        """Current roster version; raises LookupError when none has been generated."""
        if self.version is None:
            raise LookupError("no roster has been generated yet")
        return self.version

    def generate(
        self,
        store: str = "small",
        seed: int = 3,
        weeks: int = 1,
        backend: str | None = None,
        posts_only: bool = False,
    ) -> RosterVersion:
        """Generate a roster for a simulated store and make it current."""
        extra: dict[str, object] = {"solver.backend": backend} if backend else {}
        if posts_only:
            extra["objective.scores_in_roster"] = False
        data = self.loader(store, seed, weeks, **extra)
        return self.load(data, weeks, f"generated for store {store}, seed {seed}")

    def load(
        self,
        data: DemoData,
        weeks: int = 1,
        origin: str = "loaded",
        provider: ScoreProvider | None = None,
    ) -> RosterVersion:
        """Solve a roster for given data and make it current."""
        ids = [w.worker_id for w in data.workers]
        scores = score_inputs(provider or data.provider, ids, data.config)
        solution = plan_roster(data.workers, data.requirement, data.config, scores, weeks=weeks)
        self.counter += 1
        self._adopt(data.config, data.workers, data.requirement, scores, solution, origin)
        self._log(
            "roster_generated",
            f"v{self.counter}",
            {
                "origin": origin,
                "status": solution.status,
                "errors": len(solution.report.errors) if solution.report else 0,
                "shifts": len(solution.plan.assignments),
            },
        )
        return self.current()

    def _adopt(
        self,
        config: PlannerConfig,
        workers: list[WorkerRecord],
        requirement: RequirementTable,
        scores: ScoreInputs,
        solution: RosterSolution,
        origin: str,
    ) -> None:
        used = roster_scores(scores, config)
        explanations = explain_roster(solution.plan, requirement, used, config)
        shortfall = solution.report.shortfall_minutes() if solution.report else 0.0
        gap = solution.metrics.max_tier_gap if solution.metrics else 0.0
        notes = [
            *solution.notes,
            *trade_offs(solution.plan, requirement, used, config, solution.relaxed, shortfall, gap),
        ]
        if not config.objective.scores_in_roster:
            notes.insert(0, "Posts-only mode: the roster ignores the scores, the posts use them.")
        self.version = RosterVersion(
            self.counter, config, workers, requirement, scores, solution, explanations, notes
        )
        self.daily_cache.clear()
        self.live.clear()

    def _with_edit(self, edit: Edit) -> RosterPlan:
        version = self.current()
        if edit.worker_id not in version.solution.plan.workers:
            raise KeyError(f"unknown worker {edit.worker_id!r}")
        plan = version.solution.plan.without(edit.worker_id, edit.day)
        if edit.pattern_id is None:
            return plan
        pattern = parse_pattern(edit.pattern_id, version.config)
        return plan.with_assignment(Assignment(edit.worker_id, edit.day, pattern))

    def check_edit(self, edit: Edit) -> EditOutcome:
        """Check an edit against the rules without applying it."""
        version = self.current()
        edited = self._with_edit(edit)
        report = check_roster(edited, version.config.rules, version.requirement)
        before = version.solution.report
        introduced = tuple(
            sorted((_error_counts(report) - _error_counts(before or Report())).elements())
        )
        message = (
            "The edit keeps the roster legal."
            if not introduced
            else "The edit breaks: " + ", ".join(introduced)
        )
        return EditOutcome(False, report, introduced, message)

    def override(self, edit: Edit, reason: str, by: str | None = None) -> EditOutcome:
        """Apply an edit when it adds no rule error and a short reason is given.

        Every attempt, applied or blocked, is written to the audit log.
        """
        if len(reason.strip()) < MIN_REASON_CHARACTERS:
            raise ValueError("an override needs a short reason")
        outcome = self.check_edit(edit)
        subject = f"{edit.worker_id} {edit.day.isoformat()}"
        detail = {
            "pattern": edit.pattern_id,
            "reason": reason.strip(),
            "by": by,
            "new_errors": list(outcome.new_errors),
        }
        if outcome.new_errors:
            self._log("override_blocked", subject, detail)
            return outcome
        version = self.current()
        edited = self._with_edit(edit)
        solution = dataclasses.replace(
            version.solution, plan=edited, report=outcome.report, notes=[*version.solution.notes]
        )
        self.counter += 1
        self._adopt(
            version.config,
            version.workers,
            version.requirement,
            version.scores,
            solution,
            "override",
        )
        self._log("override", subject, detail)
        return dataclasses.replace(outcome, applied=True, message="The override was applied.")

    def daily(self, day: date) -> DailyResult:
        """Plan of one day, cached until the roster changes."""
        if day in self.daily_cache:
            return self.daily_cache[day]
        version = self.current()
        if not any(a.day == day for a in version.solution.plan.assignments):
            raise LookupError(f"nobody works on {day.isoformat()}")
        result = plan_day(
            version.solution.plan, day, version.requirement, version.config, version.scores
        )
        self.daily_cache[day] = result
        return result

    def move_break(
        self,
        day: date,
        worker_id: str,
        block_index: int,
        start: int,
        reason: str = "",
        apply: bool = False,
    ) -> EditOutcome:
        """Check a move of a break and, when asked, apply it.

        Applying needs a short reason and a move that adds no rule error.
        """
        version = self.current()
        result = self.daily(day)
        if not any(
            e.worker_id == worker_id and e.block_index == block_index for e in result.plan.breaks
        ):
            raise KeyError(f"{worker_id} has no break in block {block_index}")
        entries = [
            dataclasses.replace(entry, start=start)
            if entry.worker_id == worker_id and entry.block_index == block_index
            else entry
            for entry in result.plan.breaks
        ]
        schedule = BreakSchedule(entries, "MANUAL", 0, _peak(entries))
        edited = plan_day_from_shifts(
            day,
            result.plan.shifts,
            result.plan.workers,
            version.requirement,
            version.scores,
            version.config,
            schedule,
        )
        before = _error_counts(result.report)
        introduced = tuple(sorted((_error_counts(edited.report) - before).elements()))
        applied = False
        message = "The move keeps the day legal."
        if introduced:
            message = "The move breaks: " + ", ".join(introduced)
        if apply:
            if len(reason.strip()) < MIN_REASON_CHARACTERS:
                raise ValueError("a manual change needs a short reason")
            subject = f"{worker_id} {day.isoformat()}"
            detail = {"start": start, "reason": reason.strip(), "new_errors": list(introduced)}
            if introduced:
                self._log("break_move_blocked", subject, detail)
            else:
                self.daily_cache[day] = edited
                self.live.pop(day, None)
                self._log("break_moved", subject, detail)
                applied = True
                message = "The break was moved."
        return EditOutcome(applied, edited.report, introduced, message)

    def check_swap(self, first: str, second: str, day: date) -> EditOutcome:
        """Check whether two workers can swap their shifts on a day; nothing changes."""
        version = self.current()
        plan = version.solution.plan
        pattern_a, pattern_b = plan.pattern_of(first, day), plan.pattern_of(second, day)
        if pattern_a is None or pattern_b is None:
            raise LookupError("both workers must work on the day to swap shifts")
        swapped = (
            plan.without(first, day)
            .without(second, day)
            .with_assignment(Assignment(first, day, pattern_b))
            .with_assignment(Assignment(second, day, pattern_a))
        )
        report = check_roster(swapped, version.config.rules, version.requirement)
        before = version.solution.report
        introduced = tuple(
            sorted((_error_counts(report) - _error_counts(before or Report())).elements())
        )
        valid = not introduced
        self._log(
            "swap_checked",
            f"{first} {second} {day.isoformat()}",
            {"valid": valid, "new_errors": list(introduced)},
        )
        message = "The swap is legal." if valid else "The swap breaks: " + ", ".join(introduced)
        return EditOutcome(False, report, introduced, message)

    def _live_for(self, day: date) -> tuple[ProposalBook, LiveController]:
        if day not in self.live:
            version = self.current()
            result = self.daily(day)
            book = ProposalBook(result.plan, version.requirement, version.config.rules)
            self.live[day] = (book, LiveController(book, version.scores.z, version.config))
        return self.live[day]

    def observe(
        self,
        day: date,
        minute: int,
        busyness: float,
        open_registers: int,
        waiting: int = 0,
        arrival_rate: float | None = None,
        logged_in: list[str] | None = None,
    ) -> list[Proposal]:
        """Give a live reading to the controller of a day and return the proposals it raises."""
        book, controller = self._live_for(day)
        made = controller.observe(Reading(minute, busyness, open_registers, waiting), arrival_rate)
        if logged_in is not None:
            starts = {w: p.register_start for w, p in book.plan.shifts.items()}
            made += controller.check_logins(minute, starts, set(logged_in))
        for proposal in made:
            self._log(
                "proposal",
                f"{day.isoformat()} #{proposal.proposal_id}",
                {"kind": proposal.kind, "reason": proposal.reason, "minute": minute},
            )
        return made

    def proposals(self, day: date) -> list[Proposal]:
        """Proposals made so far for a day."""
        if day not in self.live:
            return []
        return list(self.live[day][0].proposals.values())

    def decide(self, day: date, proposal_id: int, accept: bool, note: str = "") -> Proposal:
        """Accept or reject a live proposal, with a note."""
        book, _ = self._live_for(day)
        proposal = book.proposals[proposal_id]
        minute = proposal.minute
        if accept:
            book.accept(proposal_id, minute, note)
            result = self.daily(day)
            self.daily_cache[day] = dataclasses.replace(
                result,
                plan=book.plan,
                report=check_daily(
                    book.plan, self.current().config.rules, self.current().requirement
                ),
            )
        else:
            book.reject(proposal_id, minute, note)
        self._log(
            "proposal_accepted" if accept else "proposal_rejected",
            f"{day.isoformat()} #{proposal_id}",
            {"kind": proposal.kind, "note": note},
        )
        return proposal

    def worker_view(self, worker_id: str) -> dict[str, object]:
        """A worker's own shifts with breaks and the reasons for them."""
        version = self.current()
        plan = version.solution.plan
        if worker_id not in plan.workers:
            raise KeyError(f"unknown worker {worker_id!r}")
        shifts = []
        total = 0.0
        for assignment in sorted(
            (a for a in plan.assignments if a.worker_id == worker_id), key=lambda a: a.day
        ):
            pattern = assignment.pattern
            total += pattern.paid_minutes / 60.0
            reasons = version.explanations.get((worker_id, assignment.day), [])
            breaks = [
                {"start": minutes_to_clock(item.start), "minutes": item.minutes}
                for item in self.daily(assignment.day).plan.breaks
                if item.worker_id == worker_id
            ]
            shifts.append(
                {
                    "day": assignment.day.isoformat(),
                    "weekday": assignment.day.strftime("%A"),
                    "blocks": [
                        _block_text(b.register_start, b.register_end) for b in pattern.blocks
                    ],
                    "paid_hours": round(pattern.paid_minutes / 60.0, 2),
                    "breaks": breaks,
                    "why": [WORKER_TEXT.get(r.code, r.text) for r in reasons],
                }
            )
        return {
            "worker_id": worker_id,
            "contract_hours": plan.workers[worker_id].contract_hours,
            "paid_hours": round(total, 2),
            "shifts": shifts,
        }

    def audit_log(self, kind: str | None = None):
        """Audit records, optionally of one kind only."""
        return self.audit.records(kind)


def _block_text(start: int, end: int) -> str:
    return f"{minutes_to_clock(start)} to {minutes_to_clock(end)}"


def _peak(entries: list[BreakEntry]) -> int:
    peak = 0
    for entry in entries:
        for minute in range(entry.start, entry.end, 5):
            peak = max(peak, sum(1 for other in entries if other.start <= minute < other.end))
    return peak
