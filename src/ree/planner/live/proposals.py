"""Proposals for live changes, with their state, audit log and acceptance rules."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ree.planner.checker.checks import check_daily
from ree.planner.config import LabourRules
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.plan import DailyPlan

PENDING = "pending"
ACCEPTED = "accepted"
REJECTED = "rejected"
STALE = "stale"

OPEN_REGISTER = "open_register"
CLOSE_REGISTER = "close_register"
COVER_ABSENCE = "cover_absence"
REPLAN = "replan"


@dataclass(frozen=True)
class Change:
    """A change of post of one worker over a time span."""

    worker_id: str
    start: int
    end: int
    before: str
    after: str


@dataclass(frozen=True)
class Draft:
    """A proposed change before it is registered: kind, reason, resulting plan and changes."""

    kind: str
    reason: str
    result: DailyPlan
    changes: tuple[Change, ...]
    wait_before: float | None = None
    wait_after: float | None = None


@dataclass
class Proposal:
    """A registered proposal that waits for a manager's decision."""

    proposal_id: int
    minute: int
    kind: str
    reason: str
    changes: tuple[Change, ...]
    cost: int
    wait_before: float | None
    wait_after: float | None
    base_version: int
    result: DailyPlan
    new_errors: tuple[str, ...]
    state: str = PENDING
    decided_at: int | None = None
    note: str = ""


@dataclass(frozen=True)
class AuditEntry:
    """One line of the audit log: what happened to which proposal."""

    minute: int
    action: str
    proposal_id: int
    kind: str
    note: str


def error_counts(plan: DailyPlan, rules: LabourRules, requirement: RequirementTable) -> Counter:
    """Number of rule errors of each code in a daily plan."""
    report = check_daily(plan, rules, requirement)
    return Counter(item.code for item in report.errors)


@dataclass
class ProposalBook:
    """Current plan, proposals and audit log; only an accepted proposal changes the plan."""

    plan: DailyPlan
    requirement: RequirementTable
    rules: LabourRules
    version: int = 0
    proposals: dict[int, Proposal] = field(default_factory=dict)
    audit: list[AuditEntry] = field(default_factory=list)

    def _log(self, minute: int, action: str, proposal: Proposal, note: str = "") -> None:
        self.audit.append(AuditEntry(minute, action, proposal.proposal_id, proposal.kind, note))

    def propose(self, draft: Draft, minute: int) -> Proposal:
        """Register a draft as a proposal and note the rule errors it would add."""
        before = error_counts(self.plan, self.rules, self.requirement)
        after = error_counts(draft.result, self.rules, self.requirement)
        introduced = tuple(sorted((after - before).elements()))
        proposal = Proposal(
            proposal_id=len(self.proposals) + 1,
            minute=minute,
            kind=draft.kind,
            reason=draft.reason,
            changes=draft.changes,
            cost=len({(c.worker_id, c.start) for c in draft.changes}),
            wait_before=draft.wait_before,
            wait_after=draft.wait_after,
            base_version=self.version,
            result=draft.result,
            new_errors=introduced,
        )
        self.proposals[proposal.proposal_id] = proposal
        self._log(minute, "proposed", proposal, draft.reason)
        return proposal

    def pending(self) -> list[Proposal]:
        """Proposals that still wait for a decision."""
        return [p for p in self.proposals.values() if p.state == PENDING]

    def accept(self, proposal_id: int, minute: int, note: str = "") -> DailyPlan:
        """Accept a proposal and make its plan current.

        A stale proposal, or one that would add rule errors, is refused. Accepting one proposal
        makes the other pending ones stale.
        """
        proposal = self.proposals[proposal_id]
        if proposal.state != PENDING:
            raise ValueError(f"proposal {proposal_id} is {proposal.state}")
        if proposal.base_version != self.version:
            proposal.state = STALE
            self._log(minute, "stale", proposal, "the plan changed after the proposal was made")
            raise ValueError(f"proposal {proposal_id} is based on an older plan")
        if proposal.new_errors:
            self._log(minute, "blocked", proposal, ", ".join(proposal.new_errors))
            raise ValueError(f"proposal {proposal_id} would break {', '.join(proposal.new_errors)}")
        proposal.state = ACCEPTED
        proposal.decided_at = minute
        proposal.note = note
        self.plan = proposal.result
        self.version += 1
        self._log(minute, "accepted", proposal, note)
        for other in self.pending():
            other.state = STALE
            self._log(minute, "stale", other, "another proposal was accepted first")
        return self.plan

    def reject(self, proposal_id: int, minute: int, note: str = "") -> None:
        """Reject a proposal and keep the reason."""
        proposal = self.proposals[proposal_id]
        if proposal.state != PENDING:
            raise ValueError(f"proposal {proposal_id} is {proposal.state}")
        proposal.state = REJECTED
        proposal.decided_at = minute
        proposal.note = note
        self._log(minute, "rejected", proposal, note)

    def acceptance_rate(self) -> float:
        """Share of decided proposals that were accepted."""
        decided = [p for p in self.proposals.values() if p.state in (ACCEPTED, REJECTED)]
        if not decided:
            return 0.0
        return sum(1 for p in decided if p.state == ACCEPTED) / len(decided)
