"""Read-only access to cashier scores: the interface the shift planner depends on."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from ree.domain.models import CashierScoreRecord, Context, Interval, Tier

UNRATED_TIER = "unrated"


@dataclass(frozen=True)
class RankedScore:
    """One line of a ranking: score, score used for ordering, tier, interval and flags."""

    worker_id: str
    rank: int
    score: float
    effective_score: float
    tier: Tier
    unrated: bool
    interval: Interval
    confidence: float
    provisional: bool
    review_flag: str | None


class ScoreProvider(Protocol):
    """Read-only access to cashier scores.

    Consumers such as the shift planner see scores and intervals only through this interface.
    """

    def get_ranking(
        self, context: Context, as_of: datetime | None = None, min_confidence: float = 0.0
    ) -> list[RankedScore]:
        """Rank the cashiers in a context, using the latest run up to as_of.

        A cashier is unrated when the tier is unrated or the confidence is below min_confidence;
        unrated cashiers stay in the list at the average score of the rated ones.
        """
        ...

    def get_score(self, worker_id: str) -> CashierScoreRecord | None:
        """Score record of one cashier, or None when there is none."""
        ...

    def list_scores(self) -> list[CashierScoreRecord]:
        """Score records of the latest run, ordered by worker id."""
        ...

    def latest_as_of(self) -> datetime | None:
        """Time of the latest run, or None when there are no scores."""
        ...


class InMemoryScoreProvider:
    """Score provider over a list of records, grouped into runs by computation time."""

    def __init__(self, records: Sequence[CashierScoreRecord]) -> None:
        grouped: dict[datetime, dict[str, CashierScoreRecord]] = defaultdict(dict)
        for record in records:
            grouped[record.computed_at][record.worker_id] = record
        self._runs = dict(sorted(grouped.items()))

    def latest_as_of(self) -> datetime | None:
        """Time of the latest run, or None when there are no scores."""
        return max(self._runs) if self._runs else None

    def _select_run(self, as_of: datetime | None) -> dict[str, CashierScoreRecord]:
        eligible = [stamp for stamp in self._runs if as_of is None or stamp <= as_of]
        if not eligible:
            return {}
        return self._runs[max(eligible)]

    def list_scores(self) -> list[CashierScoreRecord]:
        """Score records of the latest run, ordered by worker id."""
        run = self._select_run(None)
        return [run[worker_id] for worker_id in sorted(run)]

    def get_score(self, worker_id: str) -> CashierScoreRecord | None:
        """Score record of one cashier in the latest run, or None."""
        return self._select_run(None).get(worker_id)

    def get_ranking(
        self, context: Context, as_of: datetime | None = None, min_confidence: float = 0.0
    ) -> list[RankedScore]:
        """Rank the cashiers in a context, using the latest run up to as_of.

        Unrated cashiers are placed at the average score of the rated ones and flagged. Equal
        scores are ordered by worker id.
        """
        run = self._select_run(as_of)
        if not run:
            return []
        rated_scores = [
            record.score_for(context)
            for record in run.values()
            if record.tier_for(context) != UNRATED_TIER
            and record.confidence_for(context) >= min_confidence
        ]
        average = float(np.mean(rated_scores)) if rated_scores else 50.0
        entries = []
        for record in run.values():
            unrated = (
                record.tier_for(context) == UNRATED_TIER
                or record.confidence_for(context) < min_confidence
            )
            entries.append(
                (
                    average if unrated else record.score_for(context),
                    record,
                    unrated,
                )
            )
        entries.sort(key=lambda item: (-item[0], item[1].worker_id))
        return [
            RankedScore(
                worker_id=record.worker_id,
                rank=position,
                score=record.score_for(context),
                effective_score=effective,
                tier=record.tier_for(context),
                unrated=unrated,
                interval=record.interval_for(context),
                confidence=record.confidence_for(context),
                provisional=record.provisional,
                review_flag=record.review_flag,
            )
            for position, (effective, record, unrated) in enumerate(entries, start=1)
        ]
