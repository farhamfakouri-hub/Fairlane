"""Basket sizes: how many items a customer brings, by lane mixture and season."""

from __future__ import annotations

import numpy as np

from ree.config import BasketConfig
from ree.domain.baskets import (
    BIG_CLASS,
    BIG_MIN_ITEMS,
    MEDIUM_CLASS,
    MEDIUM_MAX_ITEMS,
    SMALL_CLASS,
    SMALL_MAX_ITEMS,
)

MEDIUM_MIN_ITEMS = SMALL_MAX_ITEMS + 1


def sample_item_counts(
    rng: np.random.Generator,
    mixtures: np.ndarray,
    basket_multiplier: np.ndarray,
    config: BasketConfig,
) -> np.ndarray:
    """Draw item counts: pick a size class from the lane mixture, then a count inside it.

    The count is scaled by the seasonal basket multiplier and is at least one.
    """
    count = mixtures.shape[0]
    class_draw = rng.random(count)
    cumulative = np.cumsum(mixtures, axis=1)
    classes = np.minimum((class_draw[:, None] > cumulative).sum(axis=1), BIG_CLASS)

    small = np.rint(
        np.exp(
            np.log(config.small_median_items) + config.small_log_sigma * rng.standard_normal(count)
        )
    )
    small = np.clip(small, 1, SMALL_MAX_ITEMS)

    span = MEDIUM_MAX_ITEMS - MEDIUM_MIN_ITEMS + 1
    decay = config.medium_decay_items
    truncated_uniform = rng.random(count)
    medium_excess = -decay * np.log1p(-truncated_uniform * (1.0 - np.exp(-span / decay)))
    medium = MEDIUM_MIN_ITEMS + np.minimum(np.floor(medium_excess), span - 1)

    big_excess = np.exp(
        np.log(config.big_excess_median_items)
        + config.big_excess_log_sigma * rng.standard_normal(count)
    )
    big = np.minimum(BIG_MIN_ITEMS + np.floor(big_excess), config.big_max_items)

    items = np.where(classes == SMALL_CLASS, small, np.where(classes == MEDIUM_CLASS, medium, big))
    items = np.maximum(1, np.rint(items * basket_multiplier))
    return items.astype(np.int64)
