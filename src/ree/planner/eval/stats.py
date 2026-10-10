"""Bootstrap statistics for the evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ree.config import make_rng

BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True)
class PairedDifference:
    """Mean difference between two plans over seeds, with its bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int

    @property
    def verdict(self) -> str:
        """Whether the plan is "higher", "lower" or shows "no clear difference"."""
        if self.low > 0.0:
            return "higher"
        if self.high < 0.0:
            return "lower"
        return "no clear difference"


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 7,
    level: float = 0.95,
) -> PairedDifference:
    """Paired bootstrap of the mean difference between two samples."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired samples must be non-empty and of the same length")
    differences = a - b
    rng = make_rng(seed, "planner-bootstrap")
    picks = rng.integers(0, differences.size, size=(replicates, differences.size))
    means = differences[picks].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return PairedDifference(float(differences.mean()), float(low), float(high), int(a.size))


def bootstrap_mean(
    values: Sequence[float],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 7,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of a sample with its bootstrap interval."""
    data = np.asarray(values, dtype=float)
    rng = make_rng(seed, "planner-bootstrap-mean")
    picks = rng.integers(0, data.size, size=(replicates, data.size))
    means = data[picks].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(data.mean()), float(low), float(high)
