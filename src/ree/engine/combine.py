"""Weights that combine the speed and accuracy scores in each context."""

from __future__ import annotations

import numpy as np

from ree.config import CombineConfig
from ree.domain.models import Context


def speed_weight(config: CombineConfig, context: Context) -> float:
    """Weight of the speed score in the peak or the off-peak context."""
    if context == "peak":
        return config.peak_speed_weight
    return config.off_peak_speed_weight


def accuracy_weight(config: CombineConfig, context: Context) -> float:
    """Weight of the accuracy score: one minus the speed weight."""
    return 1.0 - speed_weight(config, context)


def combined_score(
    speed_score: np.ndarray | float,
    accuracy_score: np.ndarray | float,
    config: CombineConfig,
    context: Context,
) -> np.ndarray | float:
    """Weighted mean of the speed and accuracy scores."""
    omega_speed = speed_weight(config, context)
    return omega_speed * np.asarray(speed_score) + (1.0 - omega_speed) * np.asarray(accuracy_score)
