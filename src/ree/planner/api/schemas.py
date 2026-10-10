"""Request and response models of the planner API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Base of the API models: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class GenerateRequest(Strict):
    """Request to generate a roster for a simulated store."""

    store: Literal["tiny", "small", "medium", "eval", "full"] = "small"
    seed: int = 3
    weeks: int = Field(default=1, ge=1, le=3)
    backend: Literal["cp_sat", "heuristic"] | None = None
    posts_only: bool = False


class ViolationOut(Strict):
    """One rule violation."""

    code: str
    severity: str
    message: str
    worker_id: str | None = None
    day: date | None = None
    minute: int | None = None
    amount: float | None = None


class ReportOut(Strict):
    """Result of a check: counts, uncovered minutes and the violations."""

    ok: bool
    errors: int
    warnings: int
    shortfall_minutes: float
    violations: list[ViolationOut]


class ReasonOut(Strict):
    """A reason for a shift."""

    code: str
    text: str


class AssignmentOut(Strict):
    """A worker's shift on a day, with its reasons."""

    worker_id: str
    day: date
    pattern_id: str
    blocks: list[str]
    paid_hours: float
    reasons: list[ReasonOut]


class RosterOut(Strict):
    """A roster version with its status, notes, checks, fairness measures and shifts."""

    version: int
    status: str
    backend: str
    relaxed: list[str]
    notes: list[str]
    report: ReportOut
    fairness: dict[str, float]
    assignments: list[AssignmentOut]


class EditIn(Strict):
    """A requested change of a shift, with the reason and who asked."""

    worker_id: str
    day: date
    pattern_id: str | None = None
    reason: str | None = None
    by: str | None = None


class EditOut(Strict):
    """Result of checking or applying an edit."""

    applied: bool
    message: str
    new_errors: list[str]
    report: ReportOut


class SwapIn(Strict):
    """Two workers who would swap their shifts on a day."""

    first: str
    second: str
    day: date


class BreakOut(Strict):
    """A scheduled break."""

    worker_id: str
    block_index: int
    start: int
    minutes: int


class SegmentOut(Strict):
    """A stretch of time at one post."""

    worker_id: str
    start: int
    end: int
    post: str


class HandoverOut(Strict):
    """A change of post at a minute."""

    minute: int
    worker_id: str
    from_post: str
    to_post: str
    reason: str
    replaces: str | None = None


class DailyOut(Strict):
    """The plan of one day: checks, alerts, breaks, posts, handovers and relief."""

    day: date
    report: ReportOut
    alerts: list[str]
    breaks: list[BreakOut]
    segments: list[SegmentOut]
    handovers: list[HandoverOut]
    relief: list[str]
    metrics: dict[str, float]


class BreakMoveIn(Strict):
    """A requested move of a break; it is applied only when apply is true."""

    worker_id: str
    block_index: int = 0
    start: int = Field(ge=0, le=1440)
    reason: str | None = None
    apply: bool = False


class ObservationIn(Strict):
    """A live reading of the registers at a minute of the day."""

    minute: int = Field(ge=0, le=1440)
    busyness: float = Field(ge=0.0, le=1.0)
    open_registers: int = Field(ge=0)
    waiting: int = Field(default=0, ge=0)
    arrival_rate: float | None = Field(default=None, ge=0.0)
    logged_in: list[str] | None = None


class ChangeOut(Strict):
    """A change of post inside a proposal."""

    worker_id: str
    start: int
    end: int
    before: str
    after: str


class ProposalOut(Strict):
    """A live proposal with its reason, cost, expected waits and changes."""

    proposal_id: int
    minute: int
    kind: str
    reason: str
    state: str
    cost: int
    wait_before: float | None
    wait_after: float | None
    new_errors: list[str]
    note: str
    changes: list[ChangeOut]


class DecisionIn(Strict):
    """A manager's decision on a proposal."""

    accept: bool
    note: str = ""


class AuditOut(Strict):
    """A line of the audit log."""

    record_id: int
    at: str
    kind: str
    subject: str
    detail: dict[str, object]


class HealthOut(Strict):
    """Service status and whether a roster exists."""

    status: str
    has_roster: bool
    version: int | None
    shifts: int
