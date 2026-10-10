from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers import FORBIDDEN_COLUMN_PATTERN
from ree.config import Config, FairnessConfig
from ree.domain.models import CashierScoreRecord, SimulatedStore
from ree.engine.pipeline import PipelineArtifacts, score_directory
from ree.experiments.fairness import (
    TENURE_LABELS,
    adjusted_gap,
    audit_from_config,
    pooled_audit,
    run_audit_seeds,
    run_bias_audit,
    shrinkage_comparison,
    summarize_audit,
    tenure_bands,
)
from ree.simulator.run import simulate_to_directory
from ree.storage import db

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_NAMES = {
    "contract_type",
    "contract_hours",
    "tenure_weeks",
    "absence",
    "sick_leave",
    "maternity",
    "gender",
    "age",
    "nationality",
    "religion",
    "disability",
}
ABSENCE_PATTERN = re.compile(r"absen|sick|maternit|leave|illness|pregnan", re.IGNORECASE)


def test_tenure_bands_follow_the_configured_edges() -> None:
    weeks = pd.Series([0.0, 3.9, 4.0, 11.0, 12.0, 25.9, 26.0, 200.0])
    bands = tenure_bands(weeks, (4, 12, 26)).tolist()
    assert bands == [
        TENURE_LABELS[0],
        TENURE_LABELS[0],
        TENURE_LABELS[1],
        TENURE_LABELS[1],
        TENURE_LABELS[2],
        TENURE_LABELS[2],
        TENURE_LABELS[3],
        TENURE_LABELS[3],
    ]


def test_adjusted_gap_recovers_a_planted_offset_and_slope_difference() -> None:
    rng = np.random.default_rng(1)
    skill = rng.normal(size=400)
    group = rng.random(400) < 0.4
    offset = 50 + 12 * skill + 4.0 * group + rng.normal(0, 1.0, 400)
    gap, slope, error = adjusted_gap(offset, skill[:, None], group)
    assert gap == pytest.approx(4.0, abs=0.4)
    assert 0.0 < error < 0.4 and abs(gap - 4.0) < 4 * error
    assert abs(slope) < 0.1
    flatter = 50 + 12 * skill * np.where(group, 0.5, 1.0) + rng.normal(0, 1.0, 400)
    gap_flat, slope_flat, _ = adjusted_gap(flatter, skill[:, None], group)
    assert abs(gap_flat) < 0.6
    assert slope_flat == pytest.approx(-0.5, abs=0.1)


def _synthetic_inputs(
    stores: int = 8, workers: int = 25, student_gap: float = 4.0, newcomer_lift: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frames = []
    for store in range(stores):
        skill = rng.normal(size=workers)
        student = rng.random(workers) < 0.4
        veteran = rng.random(workers) < 0.85
        frames.append(
            pd.DataFrame(
                {
                    "store": store,
                    "worker_id": [f"S{store}W{i}" for i in range(workers)],
                    "outcome_name": "speed_off",
                    "outcome": 50
                    + 12 * skill
                    + student_gap * student
                    + newcomer_lift * ~veteran
                    + rng.normal(0, 1.5, workers),
                    "primary": skill * 0.1,
                    "contract_type": np.where(student, "student", "full_time"),
                    "tenure_band": np.where(veteran, "26 weeks and more", "under 4 weeks"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_pooled_audit_recovers_a_planted_gap_with_an_honest_interval() -> None:
    pooled = pooled_audit(_synthetic_inputs(), FairnessConfig())
    student = pooled[(pooled["attribute"] == "contract_type") & (pooled["group"] == "student")]
    assert len(student) == 1
    row = student.iloc[0]
    assert row["low"] < 4.0 < row["high"] and row["workers"] > 50
    assert 3.0 < row["mean gap (points)"] < 5.0
    tenure = pooled[pooled["attribute"] == "tenure_band"]
    assert not tenure.empty
    assert (tenure["low"] < 0.0).all() and (tenure["high"] > 0.0).all()
    clean = pooled_audit(_synthetic_inputs(student_gap=0.0), FairnessConfig())
    clean_row = clean[(clean["attribute"] == "contract_type") & (clean["group"] == "student")]
    assert clean_row["low"].iloc[0] < 0.0 < clean_row["high"].iloc[0]


def test_shrinkage_comparison_pairs_the_tenure_gaps_of_two_audits() -> None:
    lifted = pooled_audit(_synthetic_inputs(newcomer_lift=4.0), FairnessConfig())
    flat = pooled_audit(_synthetic_inputs(newcomer_lift=0.0), FairnessConfig())
    comparison = shrinkage_comparison(lifted, flat)
    assert set(comparison["outcome"]) == {"speed_off"}
    assert set(comparison["group"]) == {"under 4 weeks", "26 weeks and more"}
    newcomers = comparison[comparison["group"] == "under 4 weeks"].iloc[0]
    assert newcomers["gap with shrinkage"] > newcomers["gap without shrinkage"] + 3.0
    assert abs(newcomers["gap without shrinkage"]) < 1.5
    assert " to " in newcomers["interval with shrinkage"]
    assert newcomers["workers"] > 5
    empty = shrinkage_comparison(lifted.iloc[0:0], flat)
    assert empty.empty and list(empty.columns) == list(comparison.columns)


def test_audit_study_runs_on_small_stores_and_pools_them() -> None:
    config = Config().with_overrides(
        {
            "simulator.population.n_cashiers": 12,
            "simulator.n_weeks": 4,
            "engine.intervals.n_bootstrap": 40,
        }
    )
    study = run_audit_seeds(config, [11, 12], jobs=1)
    assert set(study.inputs["store"]) == {11, 12}
    assert {"outcome_name", "outcome", "primary", "contract_type", "tenure_band"} <= set(
        study.inputs.columns
    )
    assert "standard_error" in study.rows.columns
    pooled = pooled_audit(study.inputs, config.fairness)
    assert {"contract_type"} <= set(pooled["attribute"])
    assert (pooled["high"] > pooled["low"]).all()


def test_audit_summary_pools_stores_with_intervals() -> None:
    rows = pd.DataFrame(
        {
            "outcome": ["speed_off"] * 6,
            "attribute": ["contract_type"] * 6,
            "group": ["student"] * 6,
            "adjusted_gap_points": [0.5, -0.4, 0.2, 0.1, -0.2, 0.0],
            "relative_slope_gap": [0.1] * 6,
        }
    )
    summary = summarize_audit(rows, 0.95)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["stores"] == 6 and row["low"] < row["mean gap (points)"] < row["high"]


def test_default_store_passes_the_configured_bias_limits(
    artifacts: PipelineArtifacts, default_store: SimulatedStore, default_config: Config
) -> None:
    audit = audit_from_config(
        artifacts.worker_table,
        default_store.data.workers,
        default_store.ground_truth,
        default_config,
    )
    assert not audit.rows.empty
    assert {"contract_type"} <= set(audit.rows["attribute"])
    assert audit.passes, audit.failures.to_string()
    assert (
        audit.rows["adjusted_gap_points"].abs().max()
        <= default_config.fairness.max_adjusted_gap_points
    )


def test_audit_fails_when_a_planted_bias_exceeds_the_limit(
    artifacts: PipelineArtifacts, default_store: SimulatedStore, default_config: Config
) -> None:
    contract = default_store.data.workers.set_index("worker_id")["contract_type"]
    biased = artifacts.worker_table.copy()
    students = biased["worker_id"].map(contract) == "student"
    biased.loc[students, "score_off"] += 12.0
    audit = audit_from_config(
        biased, default_store.data.workers, default_store.ground_truth, default_config
    )
    assert not audit.passes
    failing = audit.failures
    assert (failing["outcome"] == "combined_off").any()
    assert (failing["attribute"] == "contract_type").all()


def test_the_limits_come_from_the_configuration(
    artifacts: PipelineArtifacts, default_store: SimulatedStore
) -> None:
    strict = FairnessConfig(max_adjusted_gap_points=0.05)
    audit = run_bias_audit(
        artifacts.worker_table, default_store.data.workers, default_store.ground_truth, strict
    )
    assert not audit.passes and audit.gap_limit == 0.05


def test_groups_below_the_minimum_size_are_not_audited(
    artifacts: PipelineArtifacts, default_store: SimulatedStore
) -> None:
    config = FairnessConfig(min_group_size=14)
    audit = run_bias_audit(
        artifacts.worker_table, default_store.data.workers, default_store.ground_truth, config
    )
    assert audit.rows.empty or (audit.rows[["n_group", "n_rest"]] >= 14).all().all()


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def test_engine_source_never_references_protected_or_demographic_fields() -> None:
    offenders = {}
    for path in sorted((ROOT / "src" / "ree" / "engine").glob("*.py")):
        found = _identifiers(path) & PROTECTED_NAMES
        if found:
            offenders[path.name] = sorted(found)
    assert offenders == {}


def test_hire_date_is_the_only_tenure_input_and_only_the_pipeline_reads_it() -> None:
    readers = [
        path.name
        for path in sorted((ROOT / "src" / "ree" / "engine").glob("*.py"))
        if "hire_date" in _identifiers(path)
    ]
    assert readers == ["pipeline.py"]


def test_no_table_or_output_has_absence_or_identity_fields(tmp_path: Path) -> None:
    config = Config().with_overrides(
        {
            "simulator.population.n_cashiers": 8,
            "simulator.n_weeks": 4,
            "paths.data_dir": str(tmp_path),
            "paths.store_name": "privacy",
            "engine.intervals.n_bootstrap": 50,
        }
    )
    directory = simulate_to_directory(config)
    score_directory(config)
    columns = db.list_all_columns(directory)
    assert {"workers", "sessions", "cash_sheet", "scores", "audit_log", "appeals"} <= set(columns)
    for table, names in columns.items():
        for name in names:
            assert not FORBIDDEN_COLUMN_PATTERN.search(name), (table, name)
            assert not ABSENCE_PATTERN.search(name), (table, name)
    payload = json.loads((directory / db.SCORES_FILE).read_text(encoding="utf-8"))
    keys: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(key)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    assert not [
        key for key in keys if FORBIDDEN_COLUMN_PATTERN.search(key) or ABSENCE_PATTERN.search(key)
    ]
    assert all(re.fullmatch(r"W\d{3}", record["worker_id"]) for record in payload)
    assert set(CashierScoreRecord.model_fields) >= {"drivers", "model_version", "review_flag"}
