"""Register event types, table schemas and checks for the events table."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class EventType(StrEnum):
    """Kinds of register event: item scans, tags, vouchers, payment, exceptions."""

    TRANSACTION_STARTED = "transaction_started"
    ITEM_SCANNED = "item_scanned"
    ITEM_HAND_KEYED = "item_hand_keyed"
    TAG_REMOVED = "tag_removed"
    VOUCHER_APPLIED = "voucher_applied"
    LOYALTY_SCANNED = "loyalty_scanned"
    PAYMENT_PRESSED = "payment_pressed"
    SUPERVISOR_CALL = "supervisor_call"
    CHEQUE_PAYMENT = "cheque_payment"
    PRICE_CHECK = "price_check"
    VOID = "void"
    MANUAL_ENTRY = "manual_entry"


class VoidReason(StrEnum):
    """Why a transaction was voided; only cashier errors count against accuracy."""

    CASHIER_ERROR = "cashier_error"
    SUPERVISOR_REQUESTED = "supervisor_requested"
    CUSTOMER_CHANGE = "customer_change"
    UNKNOWN = "unknown"


EVENT_TYPE_ORDER: tuple[str, ...] = tuple(event.value for event in EventType)
EVENT_CODES: dict[str, int] = {name: code for code, name in enumerate(EVENT_TYPE_ORDER)}

ITEM_EVENT_TYPES: tuple[str, ...] = (
    EventType.ITEM_SCANNED.value,
    EventType.ITEM_HAND_KEYED.value,
)
GAP_EVENT_TYPES: tuple[str, ...] = (
    *ITEM_EVENT_TYPES,
    EventType.TAG_REMOVED.value,
    EventType.VOUCHER_APPLIED.value,
    EventType.LOYALTY_SCANNED.value,
)
INTERRUPTION_EVENT_TYPES: tuple[str, ...] = (
    EventType.SUPERVISOR_CALL.value,
    EventType.CHEQUE_PAYMENT.value,
    EventType.PRICE_CHECK.value,
)
EXCEPTION_EVENT_TYPES: tuple[str, ...] = (
    *INTERRUPTION_EVENT_TYPES,
    EventType.VOID.value,
    EventType.MANUAL_ENTRY.value,
)
NON_ERROR_VOID_REASONS: tuple[str, ...] = (
    VoidReason.SUPERVISOR_REQUESTED.value,
    VoidReason.CUSTOMER_CHANGE.value,
)

EVENT_COLUMNS: tuple[str, ...] = ("transaction_id", "event_type", "offset_s", "detail")
TRANSACTION_COLUMNS: tuple[str, ...] = ("transaction_id", "session_id", "register", "started_at")
SESSION_COLUMNS: tuple[str, ...] = ("session_id", "worker_id", "register", "login_ts", "logout_ts")
WORKER_COLUMNS: tuple[str, ...] = ("worker_id", "hire_date", "contract_type", "contract_hours")
CASH_SHEET_COLUMNS: tuple[str, ...] = (
    "date",
    "register",
    "drawer_id",
    "worker_id",
    "cash_handled",
    "till_discrepancy",
)


class RegisterEvent(BaseModel):
    """One timed event inside a transaction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    event_type: EventType
    offset_s: float
    detail: str | None = None

    @field_validator("offset_s")
    @classmethod
    def offset_not_negative(cls, value: float) -> float:
        """The offset must be a finite number of seconds that is not negative."""
        if value < 0 or not np.isfinite(value):
            raise ValueError("offset_s must be a finite non-negative number")
        return value

    @model_validator(mode="after")
    def detail_only_for_void(self) -> RegisterEvent:
        """Only void events carry a detail, and it must be a known void reason."""
        if self.detail is not None and self.event_type is not EventType.VOID:
            raise ValueError("detail is only allowed on void events")
        if self.detail is not None:
            VoidReason(self.detail)
        return self


def event_type_codes(event_types: pd.Series) -> np.ndarray:
    """Map event type names to integer codes, rejecting unknown names."""
    categorical = pd.Categorical(event_types.astype(str), categories=list(EVENT_TYPE_ORDER))
    codes = np.asarray(categorical.codes, dtype=np.int16)
    if (codes < 0).any():
        unknown = sorted(set(event_types.astype(str)) - set(EVENT_TYPE_ORDER))
        raise ValueError(f"unknown event types: {unknown}")
    return codes


def make_events_frame(
    transaction_id: np.ndarray,
    event_type_code: np.ndarray,
    offset_s: np.ndarray,
    detail: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build an events table from arrays of ids, type codes and offsets."""
    event_type = pd.Categorical.from_codes(
        np.asarray(event_type_code, dtype=np.int64), categories=list(EVENT_TYPE_ORDER)
    )
    if detail is None:
        detail_values = pd.Series([None] * len(transaction_id), dtype="object")
    else:
        detail_values = pd.Series(detail, dtype="object")
    return pd.DataFrame(
        {
            "transaction_id": np.asarray(transaction_id, dtype=np.int64),
            "event_type": event_type,
            "offset_s": np.asarray(offset_s, dtype=np.float64),
            "detail": detail_values,
        }
    )


def validate_events_frame(events: pd.DataFrame) -> None:
    """Check an events table: columns, offsets, single start and payment, void details."""
    missing = [name for name in EVENT_COLUMNS if name not in events.columns]
    if missing:
        raise ValueError(f"events frame is missing columns {missing}")
    codes = event_type_codes(events["event_type"])
    offsets = events["offset_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(offsets).all() or (offsets < 0).any():
        raise ValueError("offsets must be finite and non-negative")
    transaction_ids = events["transaction_id"].to_numpy()
    for event_type in (EventType.PAYMENT_PRESSED, EventType.TRANSACTION_STARTED):
        mask = codes == EVENT_CODES[event_type.value]
        counts = np.bincount(pd.factorize(transaction_ids[mask])[0], minlength=0)
        if counts.size and counts.max() > 1:
            raise ValueError(f"more than one {event_type.value} in a transaction")
    void_mask = codes == EVENT_CODES[EventType.VOID.value]
    details = events["detail"].astype("object").to_numpy()
    allowed = {reason.value for reason in VoidReason}
    bad_void = [d for d in details[void_mask] if d is not None and d == d and d not in allowed]
    if bad_void:
        raise ValueError(f"unknown void reasons {sorted(set(bad_void))}")
    other_detail = details[~void_mask]
    if any(d is not None and d == d for d in other_detail):
        raise ValueError("detail is only allowed on void events")
