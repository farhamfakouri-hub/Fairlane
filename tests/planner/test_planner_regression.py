from __future__ import annotations

import hashlib
import json

from ree.config import Config
from ree.engine.pipeline import run
from ree.simulator.run import simulate_store

PART_ONE_DIGEST = "55e404de18df414c3ea90494148f92d0d7da779bfe0fa9aed6fe9dee96b98716"


def test_part_one_scores_are_unchanged_by_the_planner() -> None:
    config = Config().with_overrides(
        {
            "seed": 4242,
            "simulator.population.n_cashiers": 8,
            "simulator.n_weeks": 4,
            "engine.intervals.n_bootstrap": 40,
        }
    )
    records = run(simulate_store(config).data, config)
    rows = [
        (
            record.worker_id,
            round(record.score_peak, 4),
            round(record.score_off, 4),
            record.tier_peak,
            record.tier_off,
            record.provisional,
        )
        for record in sorted(records, key=lambda item: item.worker_id)
    ]
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == PART_ONE_DIGEST
