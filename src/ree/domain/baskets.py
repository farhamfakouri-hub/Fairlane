"""Basket size classes: small under 20 items, medium 20 to 39, big 40 and more."""

from __future__ import annotations

import numpy as np

SMALL_MAX_ITEMS = 19
MEDIUM_MAX_ITEMS = 39
BIG_MIN_ITEMS = 40
BASKET_CLASSES: tuple[str, ...] = ("small", "medium", "big")
SMALL_CLASS = 0
MEDIUM_CLASS = 1
BIG_CLASS = 2


def basket_class_codes(item_counts: np.ndarray) -> np.ndarray:
    """Map item counts to class codes 0 (small), 1 (medium) and 2 (big)."""
    counts = np.asarray(item_counts)
    return np.where(
        counts <= SMALL_MAX_ITEMS,
        SMALL_CLASS,
        np.where(counts <= MEDIUM_MAX_ITEMS, MEDIUM_CLASS, BIG_CLASS),
    ).astype(np.int64)


def basket_class_labels(item_counts: np.ndarray) -> np.ndarray:
    """Map item counts to the labels small, medium and big."""
    codes = basket_class_codes(item_counts)
    return np.asarray(BASKET_CLASSES, dtype=object)[codes]
