"""FastAPI application that serves cashier scores, rankings and appeals."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, status

from ree.api.provider import InMemoryScoreProvider, ScoreProvider
from ree.api.schemas import (
    AppealInfo,
    HealthResponse,
    RankedScoreResponse,
    RankingResponse,
    ScoreResponse,
)
from ree.config import Config, load_config
from ree.domain.models import Appeal, AppealRequest, CashierScoreRecord, Context
from ree.storage import db

API_TITLE = "Cashier efficiency scores"
API_DESCRIPTION = (
    "Read-only scores for the shift planner (Part 2) and the review screen. "
    "Scores are decision support: a manager validates every plan, and any worker can contest "
    "their own score through POST /appeals."
)


def load_provider(config: Config) -> ScoreProvider:
    """Load the stored scores of a configuration into a provider; empty when there are none."""
    directory = db.store_directory(config)
    if not (directory / db.DATABASE_FILE).exists():
        return InMemoryScoreProvider([])
    connection = db.connect(directory)
    try:
        return InMemoryScoreProvider(db.load_scores(connection))
    finally:
        connection.close()


def create_app(config: Config | None = None, provider: ScoreProvider | None = None) -> FastAPI:
    """Build the API application.

    The settings and the score provider are loaded from the configuration unless given.
    """
    settings = config if config is not None else load_config()
    scores = provider if provider is not None else load_provider(settings)
    directory = db.store_directory(settings)
    app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version="0.1.0")

    def appeal_info(worker_id: str) -> AppealInfo:
        connection = db.connect(directory)
        try:
            appeals = db.list_appeals(connection, worker_id)
        finally:
            connection.close()
        if not appeals:
            return AppealInfo()
        latest = appeals[-1]
        return AppealInfo(status=latest.status, last_appeal_id=latest.appeal_id)

    def to_response(record: CashierScoreRecord) -> ScoreResponse:
        return ScoreResponse(**record.model_dump(), appeal=appeal_info(record.worker_id))

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Report that the service is up, with the model version and the number of cashiers."""
        records = scores.list_scores()
        return HealthResponse(
            status="ok",
            model_version=records[0].model_version if records else None,
            n_workers=len(records),
            as_of=scores.latest_as_of(),
        )

    @app.get("/scores", response_model=list[ScoreResponse], tags=["scores"])
    def list_scores() -> list[ScoreResponse]:
        """All current score records, each with the state of the cashier's appeal."""
        return [to_response(record) for record in scores.list_scores()]

    @app.get("/scores/ranking", response_model=RankingResponse, tags=["scores"])
    def ranking(
        context: Annotated[Context, Query(description="peak or off-peak score")],
        min_confidence: Annotated[
            float,
            Query(
                ge=0.0,
                le=1.0,
                description=(
                    "Workers whose confidence is below this value are treated as unrated: "
                    "they stay in the list, placed at the cohort average and flagged."
                ),
            ),
        ] = 0.0,
        as_of: Annotated[
            datetime | None, Query(description="use the latest run up to this time")
        ] = None,
    ) -> RankingResponse:
        """Cashiers ranked by their peak or off-peak score.

        Unrated cashiers are placed at the average of the rated ones and flagged.
        """
        entries = scores.get_ranking(context, as_of, min_confidence)
        return RankingResponse(
            context=context,
            as_of=as_of or scores.latest_as_of(),
            min_confidence=min_confidence,
            ranking=[RankedScoreResponse(context=context, **vars(entry)) for entry in entries],
        )

    @app.get("/scores/{worker_id}", response_model=ScoreResponse, tags=["scores"])
    def get_score(worker_id: str) -> ScoreResponse:
        """Score record of one cashier, or 404 when there is none."""
        record = scores.get_score(worker_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no score for {worker_id}")
        return to_response(record)

    @app.post(
        "/appeals", response_model=Appeal, status_code=status.HTTP_201_CREATED, tags=["appeals"]
    )
    def create_appeal(request: AppealRequest) -> Appeal:
        """Record a cashier's request to have a score reviewed."""
        record = scores.get_score(request.worker_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no score for {request.worker_id}")
        connection = db.connect(directory)
        try:
            return db.create_appeal(connection, request, record.model_version)
        finally:
            connection.close()

    @app.get("/appeals", response_model=list[Appeal], tags=["appeals"])
    def list_appeals(worker_id: str | None = None) -> list[Appeal]:
        """Stored review requests, for one cashier or for all."""
        connection = db.connect(directory)
        try:
            return db.list_appeals(connection, worker_id)
        finally:
            connection.close()

    return app
