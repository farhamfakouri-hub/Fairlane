from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ree.cli import main
from ree.config import Config, dump_config
from ree.domain.models import CashierScoreRecord, SimulatedStore
from ree.engine.pipeline import (
    PipelineArtifacts,
    model_version,
    run,
    run_with_artifacts,
    scoring_config_hash,
    source_hash,
)
from ree.simulator.run import simulate_to_directory
from ree.storage import db

ALLOWED_TIERS = {"A", "B", "C", "unrated"}


def test_records_validate_and_cover_every_worker(
    artifacts: PipelineArtifacts, default_store: SimulatedStore
) -> None:
    workers = set(default_store.data.sessions["worker_id"])
    assert {record.worker_id for record in artifacts.records} == workers
    for record in artifacts.records:
        assert CashierScoreRecord.model_validate(record.model_dump()) == record
        assert record.tier_peak in ALLOWED_TIERS and record.tier_off in ALLOWED_TIERS
        assert record.drivers
        assert record.model_version == artifacts.audit.model_version
        assert record.computed_at == default_store.data.as_of.to_pydatetime()
        assert record.n_transactions > 0 and record.weeks_of_data > 0


def test_scores_are_finite_bounded_and_inside_their_intervals(artifacts: PipelineArtifacts) -> None:
    for record in artifacts.records:
        for context in ("peak", "off"):
            score = record.score_for(context)
            interval = record.interval_for(context)
            assert np.isfinite(score)
            assert 0.0 <= score <= 100.0
            assert interval.lower <= score <= interval.upper
            assert 0.0 <= record.confidence_for(context) <= 1.0
    scores = np.array([record.score_peak for record in artifacts.records])
    assert 40.0 < scores.mean() < 60.0


def test_provisional_workers_are_labelled_and_never_tiered_alone(
    artifacts: PipelineArtifacts, default_store: SimulatedStore
) -> None:
    table = artifacts.worker_table.set_index("worker_id")
    provisional = table.index[table["provisional"]]
    assert len(provisional) >= 1
    for worker_id in provisional:
        record = next(r for r in artifacts.records if r.worker_id == worker_id)
        assert record.provisional
        assert any(text.startswith("provisional") for text in record.drivers)
    assert (table.loc[provisional, "tier_peak"] == "unrated").all()
    veterans = table.index[~table["provisional"] & (table["weeks_of_data"] >= 12)]
    assert (table.loc[veterans, "tier_peak"] != "unrated").mean() > 0.7


def test_rerun_in_process_is_byte_identical(
    artifacts: PipelineArtifacts, default_store: SimulatedStore, default_config: Config
) -> None:
    again = run_with_artifacts(default_store.data, default_config)
    assert db.scores_to_json(artifacts.records) == db.scores_to_json(again.records)
    pd.testing.assert_frame_equal(artifacts.worker_table, again.worker_table)
    assert run(default_store.data, default_config) == artifacts.records


def _small_directory_config(tmp_path: Path) -> Config:
    return Config().with_overrides(
        {
            "simulator.population.n_cashiers": 8,
            "simulator.n_weeks": 4,
            "paths.data_dir": str(tmp_path),
            "paths.store_name": "pipeline",
            "engine.intervals.n_bootstrap": 60,
        }
    )


def test_directory_scoring_is_byte_identical_and_writes_one_audit_row(tmp_path: Path) -> None:
    from ree.engine.pipeline import score_directory

    config = _small_directory_config(tmp_path)
    directory = simulate_to_directory(config)
    first_summary = score_directory(config)
    first_bytes = (directory / db.SCORES_FILE).read_bytes()
    score_directory(config)
    assert (directory / db.SCORES_FILE).read_bytes() == first_bytes
    assert "scored 8 cashiers" in first_summary

    connection = db.connect(directory)
    try:
        log = db.load_audit_log(connection)
        records = db.load_scores(connection)
    finally:
        connection.close()
    assert len(log) == 1
    row = log.iloc[0]
    assert row["model_version"] == model_version(config)
    assert row["config_hash"] == scoring_config_hash(config)
    assert row["code_hash"] == source_hash()
    assert row["n_workers"] == 8 and row["speed_method"] == config.engine.speed.method
    assert len(records) == 8
    assert db.scores_to_json(records) == first_bytes.decode("utf-8")


def test_engine_path_never_reads_the_ground_truth_table(tmp_path: Path) -> None:
    from ree.engine.pipeline import score_directory

    config = _small_directory_config(tmp_path)
    directory = simulate_to_directory(config)
    guarded = db.connect(directory, deny_ground_truth=True)
    try:
        with pytest.raises(Exception, match="prohibited"):
            guarded.execute("SELECT * FROM ground_truth").fetchall()
    finally:
        guarded.close()
    summary = score_directory(config)
    assert "scored 8 cashiers" in summary
    assert not hasattr(db.load_store_data(directory), "ground_truth")


def test_model_version_follows_the_scoring_configuration_only(default_config: Config) -> None:
    base = model_version(default_config)
    assert base == model_version(Config())
    changed_engine = default_config.with_overrides({"engine.speed.shrinkage_k": 200.0})
    assert model_version(changed_engine) != base
    changed_store = default_config.with_overrides({"store.priority_register": 2})
    assert model_version(changed_store) != base
    simulation_only = default_config.with_overrides({"simulator.n_weeks": 5, "seed": 3})
    assert model_version(simulation_only) == base
    assert base.startswith("ree-")


def test_contract_type_never_influences_any_score(
    default_store: SimulatedStore, default_config: Config, artifacts: PipelineArtifacts
) -> None:
    data = default_store.data
    shuffled = data.workers.copy()
    shuffled["contract_type"] = np.random.default_rng(1).permutation(shuffled["contract_type"])
    shuffled["contract_hours"] = np.random.default_rng(2).permutation(shuffled["contract_hours"])
    altered = dataclasses.replace(data, workers=shuffled)
    assert db.scores_to_json(run(altered, default_config)) == db.scores_to_json(artifacts.records)


def test_tenure_only_acts_through_the_provisional_label(
    default_store: SimulatedStore, default_config: Config, artifacts: PipelineArtifacts
) -> None:
    data = default_store.data
    veterans = default_store.ground_truth.set_index("worker_id")["is_newcomer"]
    veteran_ids = set(veterans.index[~veterans])
    earlier = data.workers.copy()
    mask = earlier["worker_id"].isin(veteran_ids)
    earlier.loc[mask, "hire_date"] = earlier.loc[mask, "hire_date"] - pd.Timedelta(weeks=30)
    altered = dataclasses.replace(data, workers=earlier)
    assert db.scores_to_json(run(altered, default_config)) == db.scores_to_json(artifacts.records)


def test_score_command_runs_from_a_yaml_configuration(tmp_path: Path) -> None:
    config = _small_directory_config(tmp_path)
    path = tmp_path / "config.yaml"
    dump_config(config, path)
    assert main(["simulate", "--config", str(path)]) == 0
    assert main(["score", "--config", str(path), "--method", "v2"]) == 0
    records = db.load_scores(db.connect(tmp_path / "pipeline"))
    assert len(records) == 8
    assert {record.components.speed_method for record in records} == {"v2"}
