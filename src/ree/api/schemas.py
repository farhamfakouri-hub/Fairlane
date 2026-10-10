"""Response models of the score API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ree.domain.models import AppealState, CashierScoreRecord, Context, Interval, Tier

APPEAL_PATH = "/appeals"


class AppealInfo(BaseModel):
    """State of a cashier's latest appeal, and how to ask for a review."""

    model_config = ConfigDict(extra="forbid")

    status: AppealState = "none"
    last_appeal_id: int | None = None
    how_to_request: str = Field(
        default=f"POST {APPEAL_PATH} with the worker id and a reason to ask for a review",
        description="Every score can be contested; a manager reviews the request.",
    )


class ScoreResponse(CashierScoreRecord):
    """A score record together with the state of the cashier's appeal."""

    appeal: AppealInfo


class RankedScoreResponse(BaseModel):
    """One line of a ranking in a context."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    rank: int
    context: Context
    score: float
    effective_score: float = Field(
        description="Score used for ordering; unrated workers are placed at the cohort average."
    )
    tier: Tier
    unrated: bool
    interval: Interval
    confidence: float = Field(ge=0.0, le=1.0)
    provisional: bool
    review_flag: str | None


class RankingResponse(BaseModel):
    """A ranking of cashiers with the settings that produced it."""

    model_config = ConfigDict(extra="forbid")

    context: Context
    as_of: datetime | None
    min_confidence: float
    ranking: list[RankedScoreResponse]


class HealthResponse(BaseModel):
    """Service status, model version, number of cashiers and time of the latest run."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model_version: str | None
    n_workers: int
    as_of: datetime | None
