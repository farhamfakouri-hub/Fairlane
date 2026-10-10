"""Cashier population of the simulation: hidden skills, contracts and hire dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ree.config import Config, make_rng


def generate_workers(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the cashiers and their hidden ground truth.

    Returns the public worker table (id, hire date, contract) and the ground-truth table
    (true speed, peak effect, error propensity, newcomer status and learning deficit).
    """
    simulator = config.simulator
    population = simulator.population
    rng = make_rng(config.seed, "workers")
    count = population.n_cashiers
    worker_ids = [f"W{index:03d}" for index in range(1, count + 1)]

    speed_z = rng.standard_normal(count)
    independent_z = rng.standard_normal(count)
    correlation = population.speed_error_correlation
    error_propensity = correlation * speed_z + np.sqrt(1.0 - correlation**2) * independent_z
    true_speed = population.true_speed_sd * speed_z
    peak_effect = rng.normal(population.peak_effect_mean, population.peak_effect_sd, count)

    student_count = round(population.student_share * count)
    student_mask = np.zeros(count, dtype=bool)
    student_mask[rng.permutation(count)[:student_count]] = True
    contract_type = np.where(student_mask, "student", "full_time")
    contract_hours = np.where(
        student_mask, population.student_weekly_hours, population.full_time_weekly_hours
    )

    newcomer_count = round(population.newcomer_share * count)
    newcomer_mask = np.zeros(count, dtype=bool)
    newcomer_mask[rng.permutation(count)[:newcomer_count]] = True
    latest_hire_week = max(0, simulator.n_weeks - 1)
    newcomer_hire_week = rng.integers(0, latest_hire_week + 1, size=count)
    veteran_tenure_weeks = rng.integers(
        population.veteran_min_tenure_weeks, population.veteran_max_tenure_weeks + 1, size=count
    )
    learning_deficit = rng.uniform(
        population.newcomer_deficit_low, population.newcomer_deficit_high, size=count
    )

    start = pd.Timestamp(simulator.start_date)
    hire_offset_days = np.where(newcomer_mask, newcomer_hire_week * 7, -veteran_tenure_weeks * 7)
    hire_date = start + pd.to_timedelta(hire_offset_days, unit="D")
    end = start + pd.Timedelta(days=7 * simulator.n_weeks)
    tenure_weeks_at_end = np.asarray((end - hire_date).days, dtype=np.float64) / 7.0
    tenure_weeks_at_start = np.asarray((start - hire_date).days, dtype=np.float64) / 7.0

    workers = pd.DataFrame(
        {
            "worker_id": worker_ids,
            "hire_date": pd.DatetimeIndex(hire_date).astype("datetime64[ns]"),
            "contract_type": contract_type,
            "contract_hours": contract_hours.astype(np.float64),
        }
    )
    ground_truth = pd.DataFrame(
        {
            "worker_id": worker_ids,
            "true_speed": true_speed,
            "peak_effect": peak_effect,
            "error_propensity": error_propensity,
            "contract_type": contract_type,
            "tenure_weeks": tenure_weeks_at_end,
            "tenure_weeks_at_start": tenure_weeks_at_start,
            "is_newcomer": newcomer_mask,
            "learning_deficit": learning_deficit,
        }
    )
    return workers, ground_truth
