"""Calibration of the intervals: do they contain the score of fresh data as often as claimed?"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ree.config import Config
from ree.engine.pipeline import run_with_artifacts
from ree.simulator.run import simulate_store

CONTEXTS: tuple[str, ...] = ("peak", "off")
MIN_REPLICATES = 3
Z_90 = 1.645


@dataclass
class ReplicateData:
    """Scores and components of the same cashiers over several simulated replicates."""

    n_replicates: int
    level: float
    scores: pd.DataFrame
    components: pd.DataFrame


@dataclass
class CalibrationResult:
    """Coverage of the intervals over the replicates, with the benchmark for that count."""

    n_replicates: int
    nominal_level: float
    replicate_coverage: float
    coverage_by_context: dict[str, float]
    benchmark_coverage: float
    mean_width: float
    n_intervals: int
    table: pd.DataFrame


@dataclass
class DispersionResult:
    """Ratio of the real spread between replicates to the bootstrap standard error."""

    n_replicates: int
    table: pd.DataFrame
    summary: pd.DataFrame


def benchmark_coverage(level: float, n_replicates: int) -> float:
    """Coverage expected of a perfectly calibrated interval.

    The target is the mean of the other replicates, which is itself noisy, so even a
    perfect interval covers a little less than its nominal level.
    """
    inflation = np.sqrt(1.0 + 1.0 / (n_replicates - 1))
    return float(2.0 * norm.cdf(norm.ppf(0.5 + level / 2.0) / inflation) - 1.0)


def _half_width(draws: np.ndarray, level: float) -> float:
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return float(upper - lower) / 2.0


def _replicate_outputs(config_json: str, replicate: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = Config.model_validate_json(config_json).with_overrides(
        {"simulator.replicate": replicate}
    )
    store = simulate_store(config)
    artifacts = run_with_artifacts(store.data, config)
    table = artifacts.worker_table
    level = config.engine.intervals.level
    min_weeks = config.engine.tiers.min_weeks_of_data
    settled = (~table["provisional"]) & (table["weeks_of_data"] >= min_weeks)

    score_frames = []
    component_rows = []
    for context in CONTEXTS:
        eligible = settled & table[f"rated_{context}"]
        score_frames.append(
            pd.DataFrame(
                {
                    "replicate": replicate,
                    "worker_id": table["worker_id"],
                    "context": context,
                    "score": table[f"score_{context}"],
                    "lower": table[f"lower_{context}"],
                    "upper": table[f"upper_{context}"],
                    "eligible": eligible,
                }
            )
        )
        for column, worker_id in enumerate(table["worker_id"]):
            speed_half = _half_width(artifacts.speed_draws[context][:, column], level)
            interval_half = (
                float(
                    table[f"upper_{context}"].iloc[column] - table[f"lower_{context}"].iloc[column]
                )
                / 2.0
            )
            component_rows.append(
                (
                    replicate,
                    worker_id,
                    f"speed_{context}",
                    float(table[f"speed_score_{context}"].iloc[column]),
                    speed_half,
                    bool(eligible.iloc[column]),
                )
            )
            component_rows.append(
                (
                    replicate,
                    worker_id,
                    f"combined_{context}",
                    float(table[f"score_{context}"].iloc[column]),
                    interval_half,
                    bool(eligible.iloc[column]),
                )
            )
    for column, worker_id in enumerate(table["worker_id"]):
        component_rows.append(
            (
                replicate,
                worker_id,
                "accuracy",
                float(table["accuracy_score"].iloc[column]),
                _half_width(artifacts.accuracy_draws[:, column], level),
                bool(settled.iloc[column]),
            )
        )
    components = pd.DataFrame(
        component_rows,
        columns=["replicate", "worker_id", "component", "point", "half_width", "eligible"],
    )
    return pd.concat(score_frames, ignore_index=True), components


def run_replicates(config: Config, n_replicates: int, jobs: int = 1) -> ReplicateData:
    """Simulate and score several replicates of the same cashiers, in parallel if asked."""
    if n_replicates < MIN_REPLICATES:
        raise ValueError(f"need at least {MIN_REPLICATES} replicates")
    payload = config.model_dump_json()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outputs = list(
                pool.map(_replicate_outputs, [payload] * n_replicates, range(n_replicates))
            )
    else:
        outputs = [_replicate_outputs(payload, replicate) for replicate in range(n_replicates)]
    return ReplicateData(
        n_replicates=n_replicates,
        level=config.engine.intervals.level,
        scores=pd.concat([item[0] for item in outputs], ignore_index=True),
        components=pd.concat([item[1] for item in outputs], ignore_index=True),
    )


def coverage_from(data: ReplicateData) -> CalibrationResult:
    """Share of intervals that contain the mean score of the other replicates."""
    table = data.scores.copy()
    keys = ["worker_id", "context"]
    totals = table.groupby(keys)["score"].transform("sum")
    table["target"] = (totals - table["score"]) / (data.n_replicates - 1)
    table["covered"] = (table["lower"] <= table["target"]) & (table["target"] <= table["upper"])
    table["width"] = table["upper"] - table["lower"]
    evaluated = table[table.groupby(keys)["eligible"].transform("all")]
    return CalibrationResult(
        n_replicates=data.n_replicates,
        nominal_level=data.level,
        replicate_coverage=float(evaluated["covered"].mean()),
        coverage_by_context={
            context: float(evaluated.loc[evaluated["context"] == context, "covered"].mean())
            for context in CONTEXTS
        },
        benchmark_coverage=benchmark_coverage(data.level, data.n_replicates),
        mean_width=float(evaluated["width"].mean()),
        n_intervals=len(evaluated),
        table=table,
    )


def dispersion_from(data: ReplicateData) -> DispersionResult:
    """Compare the spread between replicates of each component with its bootstrap error."""
    rows = data.components
    keys = ["worker_id", "component"]
    rows = rows[rows.groupby(keys)["eligible"].transform("all")]
    table = (
        rows.groupby(keys)
        .agg(
            empirical_sd=("point", lambda values: float(np.std(values, ddof=1))),
            bootstrap_se=("half_width", lambda values: float(np.mean(values)) / Z_90),
        )
        .reset_index()
    )
    table["ratio"] = table["empirical_sd"] / table["bootstrap_se"]
    summaries = {name: _summarise(group) for name, group in table.groupby("component")}
    summary = pd.DataFrame(summaries, index=["n", "median_ratio", "q25", "q75", "pooled_ratio"]).T
    summary.index.name = "component"
    summary["n"] = summary["n"].astype(int)
    return DispersionResult(n_replicates=data.n_replicates, table=table, summary=summary)


def _summarise(group: pd.DataFrame) -> pd.Series:
    ratio = group["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    pooled = np.sqrt(np.mean(group["empirical_sd"] ** 2) / np.mean(group["bootstrap_se"] ** 2))
    return pd.Series(
        {
            "n": len(group),
            "median_ratio": float(ratio.median()),
            "q25": float(ratio.quantile(0.25)),
            "q75": float(ratio.quantile(0.75)),
            "pooled_ratio": float(pooled),
        }
    )


def run_calibration(config: Config, n_replicates: int, jobs: int = 1) -> CalibrationResult:
    """Run the replicates and compute the coverage of the intervals."""
    return coverage_from(run_replicates(config, n_replicates, jobs))


def run_dispersion(config: Config, n_replicates: int, jobs: int = 1) -> DispersionResult:
    """Run the replicates and compute the dispersion ratios."""
    return dispersion_from(run_replicates(config, n_replicates, jobs))
