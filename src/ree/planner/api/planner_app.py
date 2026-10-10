"""FastAPI application of the planner: roster, daily plans, live proposals and audit log."""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query, status

from ree.planner.api.schemas import (
    AssignmentOut,
    AuditOut,
    BreakMoveIn,
    BreakOut,
    ChangeOut,
    DailyOut,
    DecisionIn,
    EditIn,
    EditOut,
    GenerateRequest,
    HandoverOut,
    HealthOut,
    ObservationIn,
    ProposalOut,
    ReasonOut,
    ReportOut,
    RosterOut,
    SegmentOut,
    SwapIn,
    ViolationOut,
)
from ree.planner.api.service import Edit, EditOutcome, PlannerService, RosterVersion
from ree.planner.checker.report import Report
from ree.planner.daily.daily_plan import DailyResult
from ree.planner.live.proposals import Proposal

API_TITLE = "Shift planner"
API_DESCRIPTION = (
    "Roster, daily plan and live proposals for the cashier team. Every plan is a proposal: "
    "nothing changes until a manager accepts it, every override keeps a short reason, and "
    "every manual edit is checked by an independent plan checker."
)


def report_out(report: Report) -> ReportOut:
    """Response model of a check report."""
    return ReportOut(
        ok=report.ok,
        errors=len(report.errors),
        warnings=len(report.warnings),
        shortfall_minutes=report.shortfall_minutes(),
        violations=[
            ViolationOut(
                code=item.code,
                severity=item.severity,
                message=item.message,
                worker_id=item.worker_id,
                day=item.day,
                minute=item.minute,
                amount=item.amount,
            )
            for item in [*report.errors, *report.warnings]
        ],
    )


def roster_out(version: RosterVersion) -> RosterOut:
    """Response model of a roster version, with fairness measures and reasons."""
    plan = version.solution.plan
    report = version.solution.report or Report()
    metrics = version.solution.metrics
    assignments = [
        AssignmentOut(
            worker_id=a.worker_id,
            day=a.day,
            pattern_id=a.pattern.pattern_id,
            blocks=[
                f"{b.register_start // 60:02d}:{b.register_start % 60:02d}-"
                f"{b.register_end // 60:02d}:{b.register_end % 60:02d}"
                for b in a.pattern.blocks
            ],
            paid_hours=round(a.pattern.paid_minutes / 60.0, 2),
            reasons=[
                ReasonOut(code=r.code, text=r.text)
                for r in version.explanations.get((a.worker_id, a.day), [])
            ],
        )
        for a in sorted(plan.assignments, key=lambda a: (a.day, a.worker_id))
    ]
    return RosterOut(
        version=version.number,
        status=version.solution.status,
        backend=version.solution.backend,
        relaxed=list(version.solution.relaxed),
        notes=list(version.notes),
        report=report_out(report),
        fairness={
            "burden_spread": metrics.burden_spread if metrics else 0.0,
            "max_tier_gap": metrics.max_tier_gap if metrics else 0.0,
            "max_hours_deviation": metrics.max_hours_deviation if metrics else 0.0,
        },
        assignments=assignments,
    )


def edit_out(outcome: EditOutcome) -> EditOut:
    """Response model of an edit outcome."""
    return EditOut(
        applied=outcome.applied,
        message=outcome.message,
        new_errors=list(outcome.new_errors),
        report=report_out(outcome.report),
    )


def daily_out(result: DailyResult) -> DailyOut:
    """Response model of a daily plan."""
    plan = result.plan
    metrics = result.metrics
    return DailyOut(
        day=plan.day,
        report=report_out(result.report),
        alerts=list(plan.alerts),
        breaks=[
            BreakOut(
                worker_id=b.worker_id, block_index=b.block_index, start=b.start, minutes=b.minutes
            )
            for b in plan.breaks
        ],
        segments=[
            SegmentOut(worker_id=s.worker_id, start=s.start, end=s.end, post=s.post)
            for s in plan.segments
        ],
        handovers=[
            HandoverOut(
                minute=h.minute,
                worker_id=h.worker_id,
                from_post=h.from_post,
                to_post=h.to_post,
                reason=h.reason,
                replaces=h.replaces,
            )
            for h in plan.handovers
        ],
        relief=list(result.relief_lines),
        metrics={
            "breaks": float(metrics.breaks),
            "peak_simultaneous_breaks": float(metrics.peak_simultaneous_breaks),
            "capacity_gap_minutes": float(metrics.capacity_gap_minutes),
            "handovers": float(metrics.handovers),
            "relief_handovers": float(metrics.relief_handovers),
            "priority_score_gap": metrics.priority_score_gap,
        },
    )


def proposal_out(proposal: Proposal) -> ProposalOut:
    """Response model of a live proposal."""
    return ProposalOut(
        proposal_id=proposal.proposal_id,
        minute=proposal.minute,
        kind=proposal.kind,
        reason=proposal.reason,
        state=proposal.state,
        cost=proposal.cost,
        wait_before=_finite(proposal.wait_before),
        wait_after=_finite(proposal.wait_after),
        new_errors=list(proposal.new_errors),
        note=proposal.note,
        changes=[
            ChangeOut(
                worker_id=c.worker_id, start=c.start, end=c.end, before=c.before, after=c.after
            )
            for c in proposal.changes
        ],
    )


def _finite(value: float | None) -> float | None:
    if value is None or value in (float("inf"), float("-inf")) or value != value:
        return None
    return value


def create_planner_app(service: PlannerService | None = None) -> FastAPI:
    """Build the planner API around a planner service."""
    planner = service if service is not None else PlannerService()
    app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version="0.1.0")

    def guarded(call):
        try:
            return call()
        except LookupError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except KeyError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error.args[0])) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/health", response_model=HealthOut, tags=["service"])
    def health() -> HealthOut:
        """Service status and whether a roster exists."""
        version = planner.version
        return HealthOut(
            status="ok",
            has_roster=version is not None,
            version=version.number if version else None,
            shifts=len(version.solution.plan.assignments) if version else 0,
        )

    @app.post("/roster/generate", response_model=RosterOut, tags=["roster"])
    def generate(request: GenerateRequest) -> RosterOut:
        """Generate a roster for a simulated store and make it current."""
        version = guarded(
            lambda: planner.generate(
                request.store, request.seed, request.weeks, request.backend, request.posts_only
            )
        )
        return roster_out(version)

    @app.get("/roster", response_model=RosterOut, tags=["roster"])
    def roster() -> RosterOut:
        """The current roster with its checks, notes and reasons."""
        return roster_out(guarded(planner.current))

    @app.post("/roster/check", response_model=EditOut, tags=["roster"])
    def check(edit: EditIn) -> EditOut:
        """Check a change of one shift against the rules without applying it."""
        outcome = guarded(
            lambda: planner.check_edit(Edit(edit.worker_id, edit.day, edit.pattern_id))
        )
        return edit_out(outcome)

    @app.post("/roster/override", response_model=EditOut, tags=["roster"])
    def override(edit: EditIn) -> EditOut:
        """Apply a change of one shift.

        It needs a short reason and must not add rule errors; every attempt is logged.
        """
        outcome = guarded(
            lambda: planner.override(
                Edit(edit.worker_id, edit.day, edit.pattern_id), edit.reason or "", edit.by
            )
        )
        return edit_out(outcome)

    @app.post("/swaps/check", response_model=EditOut, tags=["roster"])
    def swap(request: SwapIn) -> EditOut:
        """Check a swap of two workers' shifts on a day without applying it."""
        return edit_out(
            guarded(lambda: planner.check_swap(request.first, request.second, request.day))
        )

    @app.get("/daily/{day}", response_model=DailyOut, tags=["daily"])
    def daily(day: date) -> DailyOut:
        """The plan of one day: breaks, posts, handovers and checks."""
        return daily_out(guarded(lambda: planner.daily(day)))

    @app.post("/daily/{day}/break", response_model=EditOut, tags=["daily"])
    def move_break(day: date, move: BreakMoveIn) -> EditOut:
        """Check a move of a break and, when asked, apply it with a reason."""
        outcome = guarded(
            lambda: planner.move_break(
                day, move.worker_id, move.block_index, move.start, move.reason or "", move.apply
            )
        )
        return edit_out(outcome)

    @app.post("/live/{day}/observe", response_model=list[ProposalOut], tags=["live"])
    def observe(day: date, observation: ObservationIn) -> list[ProposalOut]:
        """Send a live reading of the registers; the planner may answer with proposals."""
        made = guarded(
            lambda: planner.observe(
                day,
                observation.minute,
                observation.busyness,
                observation.open_registers,
                observation.waiting,
                observation.arrival_rate,
                observation.logged_in,
            )
        )
        return [proposal_out(item) for item in made]

    @app.get("/proposals/{day}", response_model=list[ProposalOut], tags=["live"])
    def proposals(day: date) -> list[ProposalOut]:
        """Live proposals made for a day."""
        return [proposal_out(item) for item in guarded(lambda: planner.proposals(day))]

    @app.post("/proposals/{day}/{proposal_id}/decision", response_model=ProposalOut, tags=["live"])
    def decide(day: date, proposal_id: int, decision: DecisionIn) -> ProposalOut:
        """Accept or reject a live proposal."""
        try:
            made = planner.decide(day, proposal_id, decision.accept, decision.note)
        except KeyError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown proposal") from error
        except LookupError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except ValueError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        return proposal_out(made)

    @app.get("/workers/{worker_id}/plan", tags=["workers"])
    def worker_plan(worker_id: str) -> dict[str, object]:
        """A worker's own plan, with breaks and plain-language reasons."""
        return guarded(lambda: planner.worker_view(worker_id))

    @app.get("/audit", response_model=list[AuditOut], tags=["audit"])
    def audit(kind: str | None = Query(default=None)) -> list[AuditOut]:
        """The audit log, optionally of one kind only."""
        return [
            AuditOut(
                record_id=item.record_id,
                at=item.at,
                kind=item.kind,
                subject=item.subject,
                detail=item.detail,
            )
            for item in planner.audit_log(kind)
        ]

    return app
