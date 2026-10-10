from __future__ import annotations

import numpy as np
import pandas as pd

from helpers import FORBIDDEN_COLUMN_PATTERN, GROUND_TRUTH_COLUMNS
from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.simulator.workers import generate_workers


def test_population_distributions_match_the_configuration() -> None:
    config = Config().with_overrides({"simulator.population.n_cashiers": 6000})
    _, truth = generate_workers(config)
    population = config.simulator.population
    assert abs(truth["true_speed"].std() - population.true_speed_sd) < 0.01
    assert abs(truth["true_speed"].mean()) < 0.01
    assert abs(truth["error_propensity"].std() - 1.0) < 0.03
    correlation = np.corrcoef(truth["true_speed"], truth["error_propensity"])[0, 1]
    assert abs(correlation - population.speed_error_correlation) < 0.04
    assert abs(truth["peak_effect"].std() - population.peak_effect_sd) < 0.005


def test_contract_split_and_hiring_dates() -> None:
    config = Config()
    workers, truth = generate_workers(config)
    population = config.simulator.population
    assert workers["worker_id"].is_unique
    students = (workers["contract_type"] == "student").sum()
    assert students == round(population.student_share * population.n_cashiers)
    assert set(workers["contract_type"]) == {"student", "full_time"}
    hire = pd.to_datetime(workers["hire_date"])
    start = pd.Timestamp(config.simulator.start_date)
    end = start + pd.Timedelta(weeks=config.simulator.n_weeks)
    newcomers = truth["is_newcomer"].to_numpy()
    assert newcomers.sum() == round(population.newcomer_share * population.n_cashiers)
    assert ((hire[newcomers] >= start) & (hire[newcomers] < end)).all()
    assert (hire[~newcomers] < start).all()
    expected_tenure = (end - hire).dt.days / 7.0
    assert np.allclose(truth["tenure_weeks"], expected_tenure)


def test_ground_truth_never_appears_in_the_engine_visible_data(
    default_store: SimulatedStore,
) -> None:
    data = default_store.data
    frames = {
        "workers": data.workers,
        "sessions": data.sessions,
        "transactions": data.transactions,
        "events": data.events,
        "cash_sheet": data.cash_sheet,
    }
    for name, frame in frames.items():
        leaked = set(GROUND_TRUTH_COLUMNS) & set(frame.columns)
        assert leaked == set(), f"{name} leaks {leaked}"
        offenders = [column for column in frame.columns if FORBIDDEN_COLUMN_PATTERN.search(column)]
        assert offenders == [], f"{name} has identity-like columns {offenders}"
    assert set(GROUND_TRUTH_COLUMNS) <= set(default_store.ground_truth.columns)


def test_realized_speed_columns_track_the_latent_speed(default_store: SimulatedStore) -> None:
    truth = default_store.ground_truth
    veterans = truth[~truth["is_newcomer"]]
    gap = (veterans["realized_log_speed_off"] - veterans["true_speed"]).abs()
    assert gap.max() < 0.05
    peak_gap = (
        veterans["realized_log_speed_peak"] - (veterans["true_speed"] + veterans["peak_effect"])
    ).abs()
    assert peak_gap.max() < 0.05
