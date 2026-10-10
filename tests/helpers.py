from __future__ import annotations

import re

import pandas as pd

FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"customer|client|barcode|\bean\b|gtin|\bsku\b|item_name|item_id|product|card_number|"
    r"loyalty_id|email|phone|address|surname|first_name|last_name|full_name",
    re.IGNORECASE,
)

GROUND_TRUTH_COLUMNS = (
    "true_speed",
    "peak_effect",
    "error_propensity",
    "learning_deficit",
    "realized_log_speed_off",
    "realized_log_speed_peak",
)


def event_durations(events: pd.DataFrame) -> pd.Series:
    item_events = events[events["event_type"].isin(["item_scanned", "item_hand_keyed"])]
    first_item = item_events.groupby("transaction_id")["offset_s"].min()
    payment = (
        events[events["event_type"] == "payment_pressed"]
        .groupby("transaction_id")["offset_s"]
        .max()
    )
    return (payment - first_item).dropna()
