from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from factories import make_record, make_tiny_store
from ree.config import Config
from ree.domain.models import AppealRequest, AuditRecord
from ree.storage import db

FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"customer|client|barcode|\bean\b|gtin|\bsku\b|item_name|item_id|product|card_number|"
    r"loyalty_id|email|phone|address|surname|first_name|last_name|full_name",
    re.IGNORECASE,
)


def test_store_roundtrip(tmp_path: Path) -> None:
    simulated = make_tiny_store()
    db.save_store(simulated, tmp_path)
    loaded = db.load_store_data(tmp_path)
    pd.testing.assert_frame_equal(
        loaded.sessions.reset_index(drop=True),
        simulated.data.sessions,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(loaded.workers, simulated.data.workers, check_dtype=False)
    assert len(loaded.events) == len(simulated.data.events)
    assert list(loaded.cash_sheet["worker_id"]) == ["W001", "W002"]
    truth = db.load_ground_truth(tmp_path)
    assert list(truth["worker_id"]) == ["W001", "W002"]


def test_engine_connection_cannot_read_ground_truth(tmp_path: Path) -> None:
    db.save_store(make_tiny_store(), tmp_path)
    guarded = db.connect(tmp_path, deny_ground_truth=True)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            guarded.execute("SELECT * FROM ground_truth").fetchall()
        assert guarded.execute("SELECT COUNT(*) FROM workers").fetchone()[0] == 2
    finally:
        guarded.close()
    db.load_store_data(tmp_path, deny_ground_truth=True)


def test_no_customer_or_item_identity_columns(tmp_path: Path) -> None:
    db.save_store(make_tiny_store(), tmp_path)
    connection = db.connect(tmp_path)
    db.init_schema(connection)
    connection.close()
    columns = db.list_all_columns(tmp_path)
    assert {"workers", "sessions", "cash_sheet", "scores", "appeals", "events.parquet"} <= set(
        columns
    )
    offenders = [
        (table, name)
        for table, names in columns.items()
        for name in names
        if FORBIDDEN_COLUMN_PATTERN.search(name)
    ]
    assert offenders == []


def test_scores_roundtrip_and_audit_log(tmp_path: Path) -> None:
    connection = db.connect(tmp_path)
    records = [make_record("W001"), make_record("W002", 40.0, 61.0)]
    audit = AuditRecord(
        run_id="run-1",
        model_version="ree-test",
        config_hash="abc",
        code_hash="def",
        as_of=datetime(2026, 1, 25, 20, 0),
        n_workers=2,
        n_transactions=2468,
        speed_method="v1",
        warnings=["a warning"],
        started_wallclock_utc=datetime(2026, 2, 1, 8, 0),
    )
    price_list = pd.DataFrame(
        {
            "lane_group": ["zone_a"],
            "period": ["peak"],
            "term": ["beta"],
            "estimate": [2.9],
            "std_error": [0.01],
        }
    )
    db.save_scores(connection, "run-1", records, audit, price_list)
    loaded = db.load_scores(connection)
    assert loaded == records
    log = db.load_audit_log(connection)
    assert list(log["run_id"]) == ["run-1"]
    assert db.latest_run_id(connection) == "run-1"
    assert len(db.load_price_list(connection)) == 1
    db.write_scores_json(loaded, tmp_path / "scores.json")
    assert db.scores_to_json(loaded) == (tmp_path / "scores.json").read_text(encoding="utf-8")
    connection.close()


def test_appeals_are_stored_and_listed(tmp_path: Path) -> None:
    connection = db.connect(tmp_path)
    created = db.create_appeal(
        connection, AppealRequest(worker_id="W001", reason="I was on training"), "ree-test"
    )
    assert created.status == "open" and created.appeal_id == 1
    db.create_appeal(connection, AppealRequest(worker_id="W002", reason="Check my register"), None)
    assert len(db.list_appeals(connection)) == 2
    assert [a.worker_id for a in db.list_appeals(connection, "W001")] == ["W001"]
    connection.close()


def test_store_directory_uses_config_paths() -> None:
    config = Config().with_overrides({"paths.store_name": "demo"})
    assert db.store_directory(config) == Path("data") / "demo"
