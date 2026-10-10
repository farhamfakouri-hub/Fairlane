from __future__ import annotations

import pytest

from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.engine.pipeline import run_with_artifacts
from ree.simulator.run import simulate_store


@pytest.fixture(scope="session")
def default_config() -> Config:
    return Config()


@pytest.fixture(scope="session")
def default_store(default_config: Config) -> SimulatedStore:
    return simulate_store(default_config)


@pytest.fixture(scope="session")
def small_config() -> Config:
    return Config().with_overrides({"simulator.population.n_cashiers": 8, "simulator.n_weeks": 2})


@pytest.fixture(scope="session")
def small_store(small_config: Config) -> SimulatedStore:
    return simulate_store(small_config)


@pytest.fixture(scope="session")
def artifacts(default_store: SimulatedStore, default_config: Config):
    return run_with_artifacts(default_store.data, default_config)
