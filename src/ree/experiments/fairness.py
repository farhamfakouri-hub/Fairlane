"""Bias audit: score gaps between groups of cashiers at the same hidden skill."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ree.config import Config, FairnessConfig
from ree.engine.pipeline import run_with_artifacts
from ree.experiments.report import confidence_interval
from ree.simulator.run import simulate_store

TENURE_LABELS: tuple[str, ...] = (
    "under 4 weeks",
    "4 to 12 weeks",
    "12 to 26 weeks",
    "26 weeks and more",
)
OUTCOMES: tuple[str, ...] = (
    "speed_peak",
    "speed_off",
    "accuracy",
    "combined_peak",
    "combined_off",
)
ATTRIBUTES: tuple[str, ...] = ("contract_type", "tenure_band")
MIN_WEEKS_FOR_ACCURACY = 4
Z_95 = 1.96


@dataclass
class BiasAudit:
    """Result of the audit of one store: the gap of each group and whether it is within limits."""

    rows: pd.DataFrame
    gap_limit: float
    slope_limit: float

    @property
    def failures(self) -> pd.DataFrame:
        """Rows outside the configured limits."""
        return self.rows[~self.rows["within_limits"]]

    @property
    def passes(self) -> bool:
        """Whether every group is within the limits."""
        return self.failures.empty


@dataclass
class AuditStudy:
    """Audit rows and inputs gathered from several independent stores."""

    rows: pd.DataFrame
    inputs: pd.DataFrame


def tenure_bands(tenure_weeks: pd.Series, edges: tuple[int, ...]) -> pd.Series:
    """Label each tenure in weeks with its band, such as under 4 weeks."""
    bins = [-np.inf, *edges, np.inf]
    labels = list(TENURE_LABELS[: len(bins) - 1])
    return pd.cut(tenure_weeks, bins=bins, labels=labels, right=False).astype(str)


def _standardise(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64)
    spread = array.std()
    return (array - array.mean()) / (spread if spread > 0 else 1.0)


def adjusted_gap(
    outcome: np.ndarray, controls: np.ndarray, group: np.ndarray
) -> tuple[float, float, float]:
    """Gap in an outcome between a group and everyone else, at the same hidden skill.

    Returns the gap in points, the gap in slope relative to the slope itself, and the
    standard error of the gap.
    """
    ones = np.ones(len(outcome))
    design = np.column_stack([ones, controls, group.astype(float)])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    dof = max(len(outcome) - design.shape[1], 1)
    covariance = float(residual @ residual) / dof * np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    primary = controls[:, 0]
    interaction = np.column_stack([design, primary * group.astype(float)])
    full, *_ = np.linalg.lstsq(interaction, outcome, rcond=None)
    slope = float(full[1])
    relative = float(full[-1]) / slope if abs(slope) > 1e-9 else float("nan")
    return float(coefficients[-1]), relative, standard_error


def _outcome_inputs(worker_table: pd.DataFrame, truth: pd.DataFrame, outcome: str) -> pd.DataFrame:
    joined = worker_table.set_index("worker_id").join(truth, how="inner", rsuffix="_truth")
    accuracy_truth = -joined["error_propensity"]
    if outcome == "accuracy":
        keep = joined["weeks_of_data"] >= MIN_WEEKS_FOR_ACCURACY
        return pd.DataFrame({"outcome": joined["accuracy_score"], "primary": accuracy_truth}).loc[
            keep
        ]
    kind, context = outcome.split("_")
    keep = joined[f"n_valid_{context}"] >= 1
    speed_truth = joined[f"realized_log_speed_{context}"]
    if kind == "speed":
        columns = {"outcome": joined[f"speed_score_{context}"], "primary": speed_truth}
    else:
        columns = {
            "outcome": joined[f"score_{context}"],
            "primary": speed_truth,
            "secondary": accuracy_truth,
        }
    return pd.DataFrame(columns).loc[keep].dropna()


def audit_inputs(
    worker_table: pd.DataFrame,
    workers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: FairnessConfig,
) -> pd.DataFrame:
    """Outcome, hidden skill and group labels of each cashier, stacked for every outcome."""
    truth = ground_truth.set_index("worker_id")
    labels = pd.DataFrame(
        {
            "contract_type": workers.set_index("worker_id")["contract_type"].astype(str),
            "tenure_band": tenure_bands(truth["tenure_weeks"], config.tenure_band_edges_weeks),
        }
    )
    frames = []
    for outcome in OUTCOMES:
        data = _outcome_inputs(worker_table, truth, outcome).join(labels, how="left")
        data["outcome_name"] = outcome
        frames.append(data.rename_axis("worker_id").reset_index())
    return pd.concat(frames, ignore_index=True)


def _control_columns(data: pd.DataFrame) -> list[str]:
    return [name for name in ("primary", "secondary") if name in data and data[name].notna().any()]


def _gap_rows(data: pd.DataFrame, config: FairnessConfig) -> list[dict[str, object]]:
    rows = []
    columns = _control_columns(data)
    controls = np.column_stack([_standardise(data[name]) for name in columns])
    outcome = data["outcome"].to_numpy(dtype=np.float64)
    for attribute in ATTRIBUTES:
        labels = data[attribute]
        for group_name in sorted(set(labels.dropna())):
            member = (labels == group_name).to_numpy()
            if member.sum() < config.min_group_size or (~member).sum() < config.min_group_size:
                continue
            gap, slope_gap, error = adjusted_gap(outcome, controls, member)
            rows.append(
                {
                    "attribute": attribute,
                    "group": group_name,
                    "n_group": int(member.sum()),
                    "n_rest": int((~member).sum()),
                    "adjusted_gap_points": gap,
                    "standard_error": error,
                    "relative_slope_gap": slope_gap,
                }
            )
    return rows


def run_bias_audit(
    worker_table: pd.DataFrame,
    workers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: FairnessConfig,
) -> BiasAudit:
    """Estimate the adjusted gap of every group and outcome and check it against the limits."""
    inputs = audit_inputs(worker_table, workers, ground_truth, config)
    rows = []
    for outcome, data in inputs.groupby("outcome_name", sort=False):
        for row in _gap_rows(data.reset_index(drop=True), config):
            rows.append({"outcome": outcome, **row})
    table = pd.DataFrame(rows)
    if table.empty:
        table["within_limits"] = pd.Series(dtype=bool)
    else:
        slope_ok = table["relative_slope_gap"].abs().fillna(0.0) <= config.max_slope_gap
        gap_ok = table["adjusted_gap_points"].abs() <= config.max_adjusted_gap_points
        table["within_limits"] = slope_ok & gap_ok
    return BiasAudit(
        rows=table, gap_limit=config.max_adjusted_gap_points, slope_limit=config.max_slope_gap
    )


def audit_from_config(
    worker_table: pd.DataFrame,
    workers: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: Config,
) -> BiasAudit:
    """Run the audit with the limits of a full configuration."""
    return run_bias_audit(worker_table, workers, ground_truth, config.fairness)


def pooled_audit(inputs: pd.DataFrame, config: FairnessConfig) -> pd.DataFrame:
    """Pool several stores in one regression with a separate intercept for each store.

    Returns the mean gap of each group with a 95% interval.
    """
    rows = []
    for outcome, data in inputs.groupby("outcome_name", sort=False):
        data = data.reset_index(drop=True)
        columns = _control_columns(data)
        standardised = data.groupby("store")[columns].transform(_standardise_group)
        centred = data["outcome"] - data.groupby("store")["outcome"].transform("mean")
        controls = standardised[columns].to_numpy(dtype=np.float64)
        for attribute in ATTRIBUTES:
            labels = data[attribute]
            for group_name in sorted(set(labels.dropna())):
                member = (labels == group_name).to_numpy()
                if member.sum() < config.min_group_size or (~member).sum() < config.min_group_size:
                    continue
                gap, slope_gap, error = adjusted_gap(
                    centred.to_numpy(dtype=np.float64), controls, member
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "attribute": attribute,
                        "group": group_name,
                        "workers": int(member.sum()),
                        "mean gap (points)": gap,
                        "low": gap - Z_95 * error,
                        "high": gap + Z_95 * error,
                        "relative slope gap": slope_gap,
                    }
                )
    return pd.DataFrame(rows)


def shrinkage_comparison(
    with_shrinkage: pd.DataFrame, without_shrinkage: pd.DataFrame
) -> pd.DataFrame:
    """Compare the speed gaps by tenure band with and without shrinkage."""

    def speed_by_tenure(table: pd.DataFrame) -> pd.DataFrame:
        selected = table[
            (table["attribute"] == "tenure_band") & table["outcome"].str.startswith("speed")
        ]
        return selected.set_index(["outcome", "group"])

    base = speed_by_tenure(with_shrinkage)
    other = speed_by_tenure(without_shrinkage)
    rows = []
    for key, row in base.iterrows():
        if key not in other.index:
            continue
        counterpart = other.loc[key]
        rows.append(
            {
                "outcome": key[0],
                "group": key[1],
                "workers": int(row["workers"]),
                "gap with shrinkage": float(row["mean gap (points)"]),
                "interval with shrinkage": f"{row['low']:.2f} to {row['high']:.2f}",
                "gap without shrinkage": float(counterpart["mean gap (points)"]),
                "interval without shrinkage": (
                    f"{counterpart['low']:.2f} to {counterpart['high']:.2f}"
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "outcome",
            "group",
            "workers",
            "gap with shrinkage",
            "interval with shrinkage",
            "gap without shrinkage",
            "interval without shrinkage",
        ],
    )


def _standardise_group(values: pd.Series) -> pd.Series:
    spread = values.std(ddof=0)
    return (values - values.mean()) / (spread if spread > 0 else 1.0)


def _audit_task(config_json: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = Config.model_validate_json(config_json).with_overrides({"seed": seed})
    store = simulate_store(config)
    artifacts = run_with_artifacts(store.data, config)
    inputs = audit_inputs(
        artifacts.worker_table, store.data.workers, store.ground_truth, config.fairness
    )
    audit = run_bias_audit(
        artifacts.worker_table, store.data.workers, store.ground_truth, config.fairness
    )
    return audit.rows.assign(store=seed), inputs.assign(store=seed)


def run_audit_seeds(config: Config, seeds: list[int], jobs: int = 1) -> AuditStudy:
    """Simulate, score and audit several stores, in parallel if asked."""
    payload = config.model_dump_json()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_audit_task, [payload] * len(seeds), seeds))
    else:
        results = [_audit_task(payload, seed) for seed in seeds]
    return AuditStudy(
        rows=pd.concat([item[0] for item in results], ignore_index=True),
        inputs=pd.concat([item[1] for item in results], ignore_index=True),
    )


def summarize_audit(rows: pd.DataFrame, level: float) -> pd.DataFrame:
    """Mean gap and confidence interval of each group over the stores."""
    summary = []
    for (outcome, attribute, group), part in rows.groupby(["outcome", "attribute", "group"]):
        mean, low, high = confidence_interval(part["adjusted_gap_points"], level)
        summary.append(
            {
                "outcome": outcome,
                "attribute": attribute,
                "group": group,
                "stores": len(part),
                "mean gap (points)": mean,
                "low": low,
                "high": high,
                "mean relative slope gap": float(part["relative_slope_gap"].mean()),
            }
        )
    return pd.DataFrame(summary)
