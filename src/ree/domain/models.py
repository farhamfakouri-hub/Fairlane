"""Data models shared by the engine, the API and the screens: scores, appeals, audit rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ree.config import Config
from ree.domain.events import (
    CASH_SHEET_COLUMNS,
    EVENT_COLUMNS,
    SESSION_COLUMNS,
    TRANSACTION_COLUMNS,
    WORKER_COLUMNS,
)

Context = Literal["peak", "off"]
Tier = Literal["A", "B", "C", "unrated"]
AppealState = Literal["none", "open", "under_review", "resolved"]


class Interval(BaseModel):
    """A confidence interval around a score."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered(self) -> Interval:
        """The lower bound may not be above the upper bound."""
        if self.lower > self.upper:
            raise ValueError("interval lower bound above upper bound")
        return self

    @property
    def width(self) -> float:
        """Distance between the upper and the lower bound."""
        return self.upper - self.lower


class ScoreComponents(BaseModel):
    """The parts of a score: speed and accuracy scores, sample sizes and confidence."""

    model_config = ConfigDict(extra="forbid")

    speed_score_peak: float
    speed_score_off: float
    accuracy_score: float
    n_valid_peak: int
    n_valid_off: int
    error_rate: float | None
    void_rate_per_100: float | None
    confidence_peak: float = Field(ge=0.0, le=1.0)
    confidence_off: float = Field(ge=0.0, le=1.0)
    speed_method: str


class CashierScoreRecord(BaseModel):
    """The score of one cashier: peak and off-peak scores, intervals, tiers and reasons."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    score_peak: float
    score_off: float
    interval_peak: Interval
    interval_off: Interval
    tier_peak: Tier
    tier_off: Tier
    n_transactions: int = Field(ge=0)
    weeks_of_data: int = Field(ge=0)
    provisional: bool
    review_flag: str | None
    drivers: list[str]
    model_version: str
    computed_at: datetime
    components: ScoreComponents

    @field_validator("worker_id")
    @classmethod
    def worker_id_pseudonymous(cls, value: str) -> str:
        """The worker id must be a non-empty token without spaces: a pseudonym, never a name."""
        if not value or " " in value:
            raise ValueError("worker_id must be a non-empty pseudonymous token")
        return value

    def score_for(self, context: Context) -> float:
        """Score in the peak or the off-peak context."""
        return self.score_peak if context == "peak" else self.score_off

    def interval_for(self, context: Context) -> Interval:
        """Interval of the score in the peak or the off-peak context."""
        return self.interval_peak if context == "peak" else self.interval_off

    def tier_for(self, context: Context) -> Tier:
        """Tier in the peak or the off-peak context."""
        return self.tier_peak if context == "peak" else self.tier_off

    def confidence_for(self, context: Context) -> float:
        """Confidence from the interval width in the given context."""
        if context == "peak":
            return self.components.confidence_peak
        return self.components.confidence_off


class AppealRequest(BaseModel):
    """A cashier's request to review their score."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    reason: str = Field(min_length=3, max_length=2000)


class Appeal(BaseModel):
    """A stored appeal with its state and the model version it refers to."""

    model_config = ConfigDict(extra="forbid")

    appeal_id: int
    worker_id: str
    reason: str
    requested_at: datetime
    status: AppealState
    model_version: str | None


class AuditRecord(BaseModel):
    """One row of the audit log: what was scored, with which model and settings."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_version: str
    config_hash: str
    code_hash: str
    as_of: datetime
    n_workers: int
    n_transactions: int
    speed_method: str
    warnings: list[str]
    started_wallclock_utc: datetime


@dataclass
class StoreData:
    """Everything the engine may read about a store: no ground truth."""

    workers: pd.DataFrame
    sessions: pd.DataFrame
    transactions: pd.DataFrame
    events: pd.DataFrame
    cash_sheet: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def as_of(self) -> pd.Timestamp:
        """Time of the last logout: the instant the data are scored as of."""
        return pd.Timestamp(self.sessions["logout_ts"].max())

    @property
    def first_instant(self) -> pd.Timestamp:
        """Time of the first login."""
        return pd.Timestamp(self.sessions["login_ts"].min())

    def validate(self) -> None:
        """Check that all tables have their columns and that the keys match."""
        required = {
            "workers": (self.workers, WORKER_COLUMNS),
            "sessions": (self.sessions, SESSION_COLUMNS),
            "transactions": (self.transactions, TRANSACTION_COLUMNS),
            "events": (self.events, EVENT_COLUMNS),
            "cash_sheet": (self.cash_sheet, CASH_SHEET_COLUMNS),
        }
        for name, (frame, columns) in required.items():
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"{name} is missing columns {missing}")
        unknown_sessions = set(self.transactions["session_id"]) - set(self.sessions["session_id"])
        if unknown_sessions:
            raise ValueError("transactions reference unknown sessions")
        unknown_workers = set(self.sessions["worker_id"]) - set(self.workers["worker_id"])
        if unknown_workers:
            raise ValueError("sessions reference unknown workers")
        unknown_transactions = set(self.events["transaction_id"].unique()) - set(
            self.transactions["transaction_id"]
        )
        if unknown_transactions:
            raise ValueError("events reference unknown transactions")

    def slice_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> StoreData:
        """Copy of the data restricted to sessions, transactions and cash rows in a span."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        sessions = self.sessions[
            (self.sessions["login_ts"] >= start) & (self.sessions["login_ts"] < end)
        ]
        transactions = self.transactions[
            (self.transactions["started_at"] >= start) & (self.transactions["started_at"] < end)
        ]
        events = self.events[self.events["transaction_id"].isin(transactions["transaction_id"])]
        sheet_dates = pd.to_datetime(self.cash_sheet["date"])
        cash_sheet = self.cash_sheet[(sheet_dates >= start.normalize()) & (sheet_dates < end)]
        return StoreData(
            workers=self.workers,
            sessions=sessions.reset_index(drop=True),
            transactions=transactions.reset_index(drop=True),
            events=events.reset_index(drop=True),
            cash_sheet=cash_sheet.reset_index(drop=True),
            metadata=dict(self.metadata),
        )


@dataclass
class SimulatedStore:
    """A simulated store: the data, the hidden ground truth and the configuration."""

    data: StoreData
    ground_truth: pd.DataFrame
    config: Config
    latent: pd.DataFrame | None = None
