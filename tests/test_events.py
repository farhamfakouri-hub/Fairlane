from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ree.domain.events import (
    EVENT_CODES,
    EVENT_TYPE_ORDER,
    GAP_EVENT_TYPES,
    EventType,
    RegisterEvent,
    event_type_codes,
    make_events_frame,
    validate_events_frame,
)


def test_event_schema_contains_every_required_event() -> None:
    required = {
        "transaction_started",
        "item_scanned",
        "item_hand_keyed",
        "tag_removed",
        "voucher_applied",
        "loyalty_scanned",
        "payment_pressed",
    }
    assert required <= set(EVENT_TYPE_ORDER)
    assert {"supervisor_call", "cheque_payment", "price_check"} <= set(EVENT_TYPE_ORDER)
    assert set(GAP_EVENT_TYPES) == {
        "item_scanned",
        "item_hand_keyed",
        "tag_removed",
        "voucher_applied",
        "loyalty_scanned",
    }


def test_register_event_validation() -> None:
    event = RegisterEvent(transaction_id=1, event_type=EventType.ITEM_SCANNED, offset_s=3.5)
    assert event.detail is None
    with pytest.raises(ValidationError):
        RegisterEvent(transaction_id=1, event_type=EventType.ITEM_SCANNED, offset_s=-1.0)
    with pytest.raises(ValidationError):
        RegisterEvent(transaction_id=1, event_type=EventType.ITEM_SCANNED, offset_s=1.0, detail="x")
    with pytest.raises(ValidationError):
        RegisterEvent(transaction_id=1, event_type=EventType.VOID, offset_s=1.0, detail="bogus")
    ok = RegisterEvent(
        transaction_id=1, event_type=EventType.VOID, offset_s=1.0, detail="customer_change"
    )
    assert ok.detail == "customer_change"


def _frame(
    types: list[str], offsets: list[float], details: list[str | None] | None = None
) -> pd.DataFrame:
    codes = np.array([EVENT_CODES[name] for name in types])
    return make_events_frame(np.ones(len(types), dtype=np.int64), codes, np.array(offsets), details)


def test_validate_events_frame_accepts_a_clean_transaction() -> None:
    frame = _frame(
        ["transaction_started", "item_scanned", "item_scanned", "payment_pressed"],
        [0.0, 5.0, 8.0, 30.0],
    )
    validate_events_frame(frame)


def test_validate_events_frame_rejects_bad_data() -> None:
    with pytest.raises(ValueError):
        validate_events_frame(_frame(["item_scanned", "payment_pressed"], [-1.0, 5.0]))
    with pytest.raises(ValueError):
        validate_events_frame(_frame(["payment_pressed", "payment_pressed"], [1.0, 2.0]))
    with pytest.raises(ValueError):
        validate_events_frame(_frame(["void", "payment_pressed"], [1.0, 2.0], ["whatever", None]))
    with pytest.raises(ValueError):
        validate_events_frame(_frame(["item_scanned"], [1.0], ["oops"]))
    with pytest.raises(ValueError):
        validate_events_frame(pd.DataFrame({"transaction_id": [1]}))


def test_event_type_codes_reject_unknown_types() -> None:
    with pytest.raises(ValueError):
        event_type_codes(pd.Series(["item_scanned", "teleport"]))
    assert list(event_type_codes(pd.Series(["item_scanned"]))) == [EVENT_CODES["item_scanned"]]
