"""Small ready-made data set for the screens and the smoke check."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ree.api.provider import ScoreProvider
from ree.planner.config import PlannerConfig
from ree.planner.demand.forecast import default_model
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable
from ree.planner.domain.plan import WorkerRecord
from ree.planner.sim.part1_workforce import part1_workforce
from ree.planner.sim.scenarios import make_workforce, store_config
from ree.planner.staffing.requirement import requirement_for_cycle


@dataclass
class DemoData:
    """Configuration, workers, requirement and score provider of one demo store."""

    config: PlannerConfig
    workers: list[WorkerRecord]
    requirement: RequirementTable
    provider: ScoreProvider


def demo_data(store: str, seed: int, weeks: int = 1, **extra: object) -> DemoData:
    """Build the demo data of a store size: workers, scores from Part 1 and the requirement."""
    config = store_config(store, **extra)
    workforce = part1_workforce(config, seed, config.workforce.n_workers)
    calendar = PlannerCalendar(config)
    start = date.fromisoformat(config.start_date)
    requirement = requirement_for_cycle(
        default_model(calendar, config), calendar, config, start, weeks
    )
    workers = make_workforce(config, seed=seed, weeks=weeks)
    return DemoData(config, workers, requirement, workforce.provider)
