from __future__ import annotations

import pytest

from ree.config import Config
from ree.experiments.calibration import (
    benchmark_coverage,
    coverage_from,
    dispersion_from,
    run_replicates,
)

REPLICATES = 6


@pytest.fixture(scope="module")
def replicate_data():
    config = Config().with_overrides(
        {
            "simulator.population.n_cashiers": 12,
            "simulator.n_weeks": 6,
            "engine.intervals.n_bootstrap": 100,
        }
    )
    return run_replicates(config, REPLICATES, jobs=1)


def test_benchmark_coverage_accounts_for_the_noisy_target() -> None:
    assert benchmark_coverage(0.90, 3) < benchmark_coverage(0.90, 12) < 0.90
    assert benchmark_coverage(0.90, 10) == pytest.approx(0.881, abs=0.003)


def test_speed_intervals_are_calibrated_by_the_session_cluster_bootstrap(replicate_data) -> None:
    summary = dispersion_from(replicate_data).summary
    for component in ("speed_peak", "speed_off"):
        assert 0.7 < summary.loc[component, "pooled_ratio"] < 1.4, component


def test_combined_interval_coverage_is_measured_and_not_catastrophic(replicate_data) -> None:
    result = coverage_from(replicate_data)
    assert result.n_intervals > 100
    assert 0.55 < result.replicate_coverage <= 1.0
    assert result.benchmark_coverage < result.nominal_level
    assert result.mean_width > 0.0
    assert set(result.coverage_by_context) == {"peak", "off"}


def test_accuracy_dispersion_is_reported_with_a_ratio_per_worker(replicate_data) -> None:
    result = dispersion_from(replicate_data)
    accuracy = result.table[result.table["component"] == "accuracy"]
    assert len(accuracy) >= 8
    assert (accuracy["empirical_sd"] > 0).all() and (accuracy["bootstrap_se"] > 0).all()
    assert set(result.summary.index) == {
        "accuracy",
        "combined_off",
        "combined_peak",
        "speed_off",
        "speed_peak",
    }
