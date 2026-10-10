from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ree.config import Config
from ree.domain.events import EVENT_CODES, make_events_frame
from ree.domain.models import (
    CashierScoreRecord,
    Interval,
    ScoreComponents,
    SimulatedStore,
    StoreData,
)


def events_from_sequence(
    transaction_id: int,
    kinds_and_offsets: list[tuple[str, float]],
    details: dict[int, str] | None = None,
) -> pd.DataFrame:
    codes = np.array([EVENT_CODES[kind] for kind, _ in kinds_and_offsets])
    offsets = np.array([offset for _, offset in kinds_and_offsets], dtype=np.float64)
    detail = [None] * len(kinds_and_offsets)
    for index, value in (details or {}).items():
        detail[index] = value
    return make_events_frame(
        np.full(len(kinds_and_offsets), transaction_id, dtype=np.int64),
        codes,
        offsets,
        np.array(detail, dtype=object),
    )


@dataclass
class TransactionSpec:
    events: list[tuple[str, float]]
    worker_id: str = "W001"
    register: int = 2
    started_at: str = "2025-11-25 09:00:00"
    details: dict[int, str] | None = None


def store_from_specs(specs: Sequence[TransactionSpec]) -> StoreData:
    worker_ids = sorted({spec.worker_id for spec in specs})
    workers = pd.DataFrame(
        {
            "worker_id": worker_ids,
            "hire_date": pd.to_datetime(["2024-01-08"] * len(worker_ids)),
            "contract_type": ["full_time"] * len(worker_ids),
            "contract_hours": [35.0] * len(worker_ids),
        }
    )
    session_keys = sorted({(spec.worker_id, spec.register) for spec in specs})
    sessions = pd.DataFrame(
        {
            "session_id": list(range(1, len(session_keys) + 1)),
            "worker_id": [key[0] for key in session_keys],
            "register": [key[1] for key in session_keys],
            "login_ts": pd.to_datetime(["2025-11-25 08:30:00"] * len(session_keys)),
            "logout_ts": pd.to_datetime(["2025-11-25 20:00:00"] * len(session_keys)),
        }
    )
    session_of = {key: index + 1 for index, key in enumerate(session_keys)}
    transactions = pd.DataFrame(
        {
            "transaction_id": list(range(1, len(specs) + 1)),
            "session_id": [session_of[(spec.worker_id, spec.register)] for spec in specs],
            "register": [spec.register for spec in specs],
            "started_at": pd.to_datetime([spec.started_at for spec in specs]),
        }
    )
    frames = []
    for index, spec in enumerate(specs, start=1):
        sequence = [("transaction_started", 0.0), *spec.events]
        detail = {key + 1: value for key, value in (spec.details or {}).items()}
        frames.append(events_from_sequence(index, sequence, detail))
    events = (
        pd.concat(frames, ignore_index=True)
        if frames
        else events_from_sequence(0, [("transaction_started", 0.0)]).iloc[0:0]
    )
    cash_sheet = pd.DataFrame(
        {
            "date": pd.to_datetime([]),
            "register": pd.Series([], dtype="int64"),
            "drawer_id": pd.Series([], dtype="object"),
            "worker_id": pd.Series([], dtype="object"),
            "cash_handled": pd.Series([], dtype="float64"),
            "till_discrepancy": pd.Series([], dtype="float64"),
        }
    )
    return StoreData(
        workers=workers,
        sessions=sessions,
        transactions=transactions,
        events=events,
        cash_sheet=cash_sheet,
    )


def make_tiny_store() -> SimulatedStore:
    workers = pd.DataFrame(
        {
            "worker_id": ["W001", "W002"],
            "hire_date": pd.to_datetime(["2024-01-08", "2025-11-10"]),
            "contract_type": ["full_time", "student"],
            "contract_hours": [35.0, 20.0],
        }
    )
    sessions = pd.DataFrame(
        {
            "session_id": [1, 2],
            "worker_id": ["W001", "W002"],
            "register": [2, 9],
            "login_ts": pd.to_datetime(["2025-11-25 08:30:00", "2025-11-25 14:00:00"]),
            "logout_ts": pd.to_datetime(["2025-11-25 14:30:00", "2025-11-25 20:00:00"]),
        }
    )
    transactions = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "session_id": [1, 1, 2],
            "register": [2, 2, 9],
            "started_at": pd.to_datetime(
                ["2025-11-25 09:00:00", "2025-11-25 09:05:00", "2025-11-25 15:00:00"]
            ),
        }
    )
    events = pd.concat(
        [
            events_from_sequence(
                1,
                [
                    ("transaction_started", 0.0),
                    ("item_scanned", 5.0),
                    ("item_scanned", 8.0),
                    ("payment_pressed", 30.0),
                ],
            ),
            events_from_sequence(
                2,
                [
                    ("transaction_started", 0.0),
                    ("item_scanned", 4.0),
                    ("void", 6.0),
                    ("payment_pressed", 20.0),
                ],
                {2: "cashier_error"},
            ),
            events_from_sequence(
                3,
                [
                    ("transaction_started", 0.0),
                    ("item_hand_keyed", 6.0),
                    ("payment_pressed", 22.0),
                ],
            ),
        ],
        ignore_index=True,
    )
    cash_sheet = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-11-25", "2025-11-25"]),
            "register": [2, 9],
            "drawer_id": ["2025-11-25-2-W001", "2025-11-25-9-W002"],
            "worker_id": ["W001", "W002"],
            "cash_handled": [450.0, 380.0],
            "till_discrepancy": [-1.5, 0.2],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "worker_id": ["W001", "W002"],
            "true_speed": [0.1, -0.2],
            "peak_effect": [0.0, 0.05],
            "error_propensity": [-0.3, 0.8],
            "contract_type": ["full_time", "student"],
            "tenure_weeks": [95.0, 3.0],
            "learning_deficit": [0.2, 0.3],
        }
    )
    data = StoreData(
        workers=workers,
        sessions=sessions,
        transactions=transactions,
        events=events,
        cash_sheet=cash_sheet,
        metadata={"note": "tiny"},
    )
    return SimulatedStore(data=data, ground_truth=ground_truth, config=Config())


def make_record(
    worker_id: str = "W001", score_peak: float = 55.0, score_off: float = 48.0
) -> CashierScoreRecord:
    return CashierScoreRecord(
        worker_id=worker_id,
        score_peak=score_peak,
        score_off=score_off,
        interval_peak=Interval(lower=score_peak - 3, upper=score_peak + 3),
        interval_off=Interval(lower=score_off - 4, upper=score_off + 4),
        tier_peak="A",
        tier_off="B",
        n_transactions=1234,
        weeks_of_data=12,
        provisional=False,
        review_flag=None,
        drivers=["faster than the store median on big baskets"],
        model_version="ree-test",
        computed_at=datetime(2026, 1, 25, 20, 0, 0),
        components=ScoreComponents(
            speed_score_peak=score_peak,
            speed_score_off=score_off,
            accuracy_score=50.0,
            n_valid_peak=500,
            n_valid_off=700,
            error_rate=0.0004,
            void_rate_per_100=2.1,
            confidence_peak=0.8,
            confidence_off=0.7,
            speed_method="v1",
        ),
    )
