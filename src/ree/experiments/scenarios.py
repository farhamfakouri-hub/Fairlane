"""The simulated scenarios on which the scoring methods are compared."""

from __future__ import annotations

from dataclasses import dataclass

from ree.config import Config

NO_NEWCOMERS = "simulator.population.newcomer_share"
SHARED_TILL_PROBABILITY = "simulator.scheduler.shared_till_probability"
CUSTOMER_DELAY = "simulator.timing.customer_delay_probability_per_transaction"
INTERRUPTION = "simulator.timing.interruption_probability_per_transaction"


@dataclass(frozen=True)
class Scenario:
    """A named set of changes to the default configuration."""

    name: str
    title: str
    description: str
    overrides: tuple[tuple[str, object], ...]

    def apply(self, config: Config, seed: int, reduced: bool) -> Config:
        """Configuration of this scenario for a seed; reduced runs use fewer cashiers and weeks."""
        values: dict[str, object] = dict(self.overrides)
        values["seed"] = seed
        if reduced:
            experiments = config.experiments
            values["simulator.population.n_cashiers"] = experiments.reduced_cashiers
            weeks = int(values.get("simulator.n_weeks", config.simulator.n_weeks))
            values["simulator.n_weeks"] = min(weeks, experiments.reduced_weeks)
        return config.with_overrides(values)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="balanced",
        title="Balanced assignment",
        description="Random shifts and registers, 12 weeks, veterans only.",
        overrides=((NO_NEWCOMERS, 0.0),),
    ),
    Scenario(
        name="biased",
        title="Biased assignment",
        description=(
            "Strong cashiers get more peak hours, the card-only lane and smaller baskets; "
            "weak cashiers get big baskets and the priority lane."
        ),
        overrides=((NO_NEWCOMERS, 0.0), ("simulator.scheduler.mode", "biased")),
    ),
    Scenario(
        name="low_data",
        title="Low data (2 weeks)",
        description="Only two weeks of register data.",
        overrides=((NO_NEWCOMERS, 0.0), ("simulator.n_weeks", 2)),
    ),
    Scenario(
        name="heavy_interruptions",
        title="Heavy interruptions",
        description="Customer delays in 15 percent and interruptions in 8 percent of transactions.",
        overrides=((NO_NEWCOMERS, 0.0), (CUSTOMER_DELAY, 0.15), (INTERRUPTION, 0.08)),
    ),
    Scenario(
        name="newcomers",
        title="Newcomers with learning curves",
        description="A third of the cashiers are hired during the window and speed up with tenure.",
        overrides=((NO_NEWCOMERS, 0.32),),
    ),
    Scenario(
        name="shared_tills",
        title="Shared tills",
        description="Cashiers share a drawer on most days they meet at the same register.",
        overrides=((NO_NEWCOMERS, 0.0), (SHARED_TILL_PROBABILITY, 0.9)),
    ),
)
SCENARIO_BY_NAME: dict[str, Scenario] = {scenario.name: scenario for scenario in SCENARIOS}
