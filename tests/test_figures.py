from __future__ import annotations

from pathlib import Path

from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.experiments.figures import make_gap_distribution_figure

MINIMUM_FIGURE_BYTES = 20_000


def test_gap_distribution_figure_is_written(
    default_config: Config, default_store: SimulatedStore, tmp_path: Path
) -> None:
    target = tmp_path / "gap_distribution.png"
    written = make_gap_distribution_figure(default_config, default_store, target)
    assert written == target
    assert target.exists()
    assert target.stat().st_size > MINIMUM_FIGURE_BYTES
