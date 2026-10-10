"""SQLite and parquet storage of stores, scores, price lists, audit logs and appeals."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ree.config import Config, canonical_json
from ree.domain.models import (
    Appeal,
    AppealRequest,
    AppealState,
    AuditRecord,
    CashierScoreRecord,
    Interval,
    ScoreComponents,
    SimulatedStore,
    StoreData,
)

DATABASE_FILE = "store.db"
EVENTS_FILE = "events.parquet"
TRANSACTIONS_FILE = "transactions.parquet"
METADATA_FILE = "store_meta.json"
SCORES_FILE = "scores.json"
GROUND_TRUTH_TABLE = "ground_truth"
ENGINE_VISIBLE_TABLES: tuple[str, ...] = ("workers", "sessions", "cash_sheet")

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS workers (
        worker_id TEXT PRIMARY KEY,
        hire_date TEXT NOT NULL,
        contract_type TEXT,
        contract_hours REAL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY,
        worker_id TEXT NOT NULL REFERENCES workers(worker_id),
        register INTEGER NOT NULL,
        login_ts TEXT NOT NULL,
        logout_ts TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS cash_sheet (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        register INTEGER NOT NULL,
        drawer_id TEXT NOT NULL,
        worker_id TEXT,
        cash_handled REAL NOT NULL,
        till_discrepancy REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS scores (
        run_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        score_peak REAL NOT NULL,
        score_off REAL NOT NULL,
        interval_peak_lower REAL NOT NULL,
        interval_peak_upper REAL NOT NULL,
        interval_off_lower REAL NOT NULL,
        interval_off_upper REAL NOT NULL,
        tier_peak TEXT NOT NULL,
        tier_off TEXT NOT NULL,
        n_transactions INTEGER NOT NULL,
        weeks_of_data INTEGER NOT NULL,
        provisional INTEGER NOT NULL,
        review_flag TEXT,
        drivers_json TEXT NOT NULL,
        components_json TEXT NOT NULL,
        model_version TEXT NOT NULL,
        computed_at TEXT NOT NULL,
        PRIMARY KEY (run_id, worker_id)
    )""",
    """CREATE TABLE IF NOT EXISTS audit_log (
        run_id TEXT PRIMARY KEY,
        model_version TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        as_of TEXT NOT NULL,
        n_workers INTEGER NOT NULL,
        n_transactions INTEGER NOT NULL,
        speed_method TEXT NOT NULL,
        warnings_json TEXT NOT NULL,
        started_wallclock_utc TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS price_list (
        run_id TEXT NOT NULL,
        lane_group TEXT NOT NULL,
        period TEXT NOT NULL,
        term TEXT NOT NULL,
        estimate REAL NOT NULL,
        std_error REAL,
        PRIMARY KEY (run_id, lane_group, period, term)
    )""",
    """CREATE TABLE IF NOT EXISTS appeals (
        appeal_id INTEGER PRIMARY KEY AUTOINCREMENT,
        worker_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        status TEXT NOT NULL,
        model_version TEXT
    )""",
)


def store_directory(config: Config) -> Path:
    """Directory of the stored simulation for a configuration."""
    return Path(config.paths.data_dir) / config.paths.store_name


def connect(directory: str | Path, deny_ground_truth: bool = False) -> sqlite3.Connection:
    """Open the store database, creating the directory if needed.

    With deny_ground_truth the connection refuses to read the hidden ground-truth table.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path / DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    if deny_ground_truth:
        connection.set_authorizer(_deny_ground_truth_reads)
    return connection


def _deny_ground_truth_reads(
    action: int, argument_one: str | None, argument_two: str | None, *_: object
) -> int:
    if action == sqlite3.SQLITE_READ and argument_one == GROUND_TRUTH_TABLE:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def init_schema(connection: sqlite3.Connection) -> None:
    """Create the tables of the store database when they do not exist."""
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.commit()


def _iso(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%dT%H:%M:%S")


def save_store(simulated: SimulatedStore, directory: str | Path) -> Path:
    """Write a simulated store to a directory: database, parquet files and metadata."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    database_path = path / DATABASE_FILE
    if database_path.exists():
        database_path.unlink()
    connection = connect(path)
    try:
        init_schema(connection)
        data = simulated.data
        workers = data.workers.copy()
        workers["hire_date"] = pd.to_datetime(workers["hire_date"]).dt.strftime("%Y-%m-%d")
        workers.to_sql("workers", connection, if_exists="append", index=False)
        sessions = data.sessions.copy()
        sessions["login_ts"] = _iso(sessions["login_ts"])
        sessions["logout_ts"] = _iso(sessions["logout_ts"])
        sessions.to_sql("sessions", connection, if_exists="append", index=False)
        sheet = data.cash_sheet.copy()
        sheet["date"] = pd.to_datetime(sheet["date"]).dt.strftime("%Y-%m-%d")
        sheet.to_sql("cash_sheet", connection, if_exists="append", index=False)
        simulated.ground_truth.to_sql(
            GROUND_TRUTH_TABLE, connection, if_exists="replace", index=False
        )
        connection.commit()
    finally:
        connection.close()
    data.transactions.to_parquet(path / TRANSACTIONS_FILE, index=False)
    data.events.to_parquet(path / EVENTS_FILE, index=False)
    metadata = {"config": json.loads(canonical_json(simulated.config)), **data.metadata}
    (path / METADATA_FILE).write_text(
        json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
    )
    return path


def load_store_data(directory: str | Path, deny_ground_truth: bool = True) -> StoreData:
    """Load the data the engine may see.

    By default the connection refuses to read the ground-truth table.
    """
    path = Path(directory)
    connection = connect(path, deny_ground_truth=deny_ground_truth)
    try:
        workers = pd.read_sql("SELECT * FROM workers", connection)
        sessions = pd.read_sql("SELECT * FROM sessions", connection)
        cash_sheet = pd.read_sql("SELECT * FROM cash_sheet", connection)
    finally:
        connection.close()
    workers["hire_date"] = pd.to_datetime(workers["hire_date"])
    sessions["login_ts"] = pd.to_datetime(sessions["login_ts"])
    sessions["logout_ts"] = pd.to_datetime(sessions["logout_ts"])
    cash_sheet["date"] = pd.to_datetime(cash_sheet["date"])
    cash_sheet = cash_sheet.drop(columns=["row_id"])
    transactions = pd.read_parquet(path / TRANSACTIONS_FILE)
    events = pd.read_parquet(path / EVENTS_FILE)
    metadata_path = path / METADATA_FILE
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    data = StoreData(
        workers=workers,
        sessions=sessions,
        transactions=transactions,
        events=events,
        cash_sheet=cash_sheet,
        metadata=metadata,
    )
    data.validate()
    return data


def load_ground_truth(directory: str | Path) -> pd.DataFrame:
    """Load the hidden ground-truth table; only experiments and tests read it."""
    connection = connect(directory)
    try:
        return pd.read_sql(f"SELECT * FROM {GROUND_TRUTH_TABLE}", connection)
    finally:
        connection.close()


def load_store_config(directory: str | Path) -> Config:
    """Read back the configuration a store was simulated with."""
    metadata = json.loads((Path(directory) / METADATA_FILE).read_text(encoding="utf-8"))
    return Config.model_validate(metadata["config"])


def list_all_columns(directory: str | Path) -> dict[str, list[str]]:
    """Column names of every table and parquet file in a store, used by the privacy test."""
    path = Path(directory)
    connection = connect(path)
    try:
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        columns = {
            table: [row["name"] for row in connection.execute(f"PRAGMA table_info({table})")]
            for table in tables
        }
    finally:
        connection.close()
    for parquet_file in (TRANSACTIONS_FILE, EVENTS_FILE):
        if (path / parquet_file).exists():
            columns[parquet_file] = list(pq.read_schema(path / parquet_file).names)
    return columns


def _record_to_row(run_id: str, record: CashierScoreRecord) -> tuple[object, ...]:
    return (
        run_id,
        record.worker_id,
        record.score_peak,
        record.score_off,
        record.interval_peak.lower,
        record.interval_peak.upper,
        record.interval_off.lower,
        record.interval_off.upper,
        record.tier_peak,
        record.tier_off,
        record.n_transactions,
        record.weeks_of_data,
        int(record.provisional),
        record.review_flag,
        json.dumps(record.drivers),
        record.components.model_dump_json(),
        record.model_version,
        record.computed_at.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def save_scores(
    connection: sqlite3.Connection,
    run_id: str,
    records: Iterable[CashierScoreRecord],
    audit: AuditRecord,
    price_list: pd.DataFrame | None = None,
) -> None:
    """Save the score records, the audit record and the price list of one run."""
    init_schema(connection)
    connection.execute("DELETE FROM scores WHERE run_id = ?", (run_id,))
    connection.executemany(
        "INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [_record_to_row(run_id, record) for record in records],
    )
    connection.execute(
        "INSERT OR REPLACE INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            audit.run_id,
            audit.model_version,
            audit.config_hash,
            audit.code_hash,
            audit.as_of.strftime("%Y-%m-%dT%H:%M:%S"),
            audit.n_workers,
            audit.n_transactions,
            audit.speed_method,
            json.dumps(audit.warnings),
            audit.started_wallclock_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    if price_list is not None:
        connection.execute("DELETE FROM price_list WHERE run_id = ?", (run_id,))
        connection.executemany(
            "INSERT INTO price_list VALUES (?,?,?,?,?,?)",
            [
                (
                    run_id,
                    row.lane_group,
                    row.period,
                    row.term,
                    float(row.estimate),
                    None if pd.isna(row.std_error) else float(row.std_error),
                )
                for row in price_list.itertuples(index=False)
            ],
        )
    connection.commit()


def latest_run_id(connection: sqlite3.Connection) -> str | None:
    """Identifier of the most recent scoring run, or None."""
    row = connection.execute(
        "SELECT run_id FROM audit_log ORDER BY as_of DESC, started_wallclock_utc DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row["run_id"])


def load_scores(
    connection: sqlite3.Connection, run_id: str | None = None
) -> list[CashierScoreRecord]:
    """Load the score records of a run (the latest one by default)."""
    if run_id is None:
        run_id = latest_run_id(connection)
    if run_id is None:
        return []
    rows = connection.execute(
        "SELECT * FROM scores WHERE run_id = ? ORDER BY worker_id", (run_id,)
    ).fetchall()
    return [
        CashierScoreRecord(
            worker_id=row["worker_id"],
            score_peak=row["score_peak"],
            score_off=row["score_off"],
            interval_peak=Interval(
                lower=row["interval_peak_lower"], upper=row["interval_peak_upper"]
            ),
            interval_off=Interval(lower=row["interval_off_lower"], upper=row["interval_off_upper"]),
            tier_peak=row["tier_peak"],
            tier_off=row["tier_off"],
            n_transactions=row["n_transactions"],
            weeks_of_data=row["weeks_of_data"],
            provisional=bool(row["provisional"]),
            review_flag=row["review_flag"],
            drivers=json.loads(row["drivers_json"]),
            model_version=row["model_version"],
            computed_at=datetime.fromisoformat(row["computed_at"]),
            components=ScoreComponents.model_validate_json(row["components_json"]),
        )
        for row in rows
    ]


def load_price_list(connection: sqlite3.Connection, run_id: str | None = None) -> pd.DataFrame:
    """Load the price list of a run (the latest one by default)."""
    if run_id is None:
        run_id = latest_run_id(connection)
    if run_id is None:
        return pd.DataFrame(columns=["lane_group", "period", "term", "estimate", "std_error"])
    return pd.read_sql(
        "SELECT lane_group, period, term, estimate, std_error FROM price_list WHERE run_id = ?",
        connection,
        params=(run_id,),
    )


def load_audit_log(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load the audit log of all runs."""
    return pd.read_sql("SELECT * FROM audit_log ORDER BY as_of, started_wallclock_utc", connection)


def scores_to_json(records: Iterable[CashierScoreRecord]) -> str:
    """Serialise score records to JSON text."""
    payload = [record.model_dump(mode="json") for record in records]
    return json.dumps(payload, sort_keys=True, indent=2)


def write_scores_json(records: Iterable[CashierScoreRecord], path: str | Path) -> None:
    """Write score records to a JSON file."""
    Path(path).write_bytes(scores_to_json(records).encode("utf-8"))


def create_appeal(
    connection: sqlite3.Connection, request: AppealRequest, model_version: str | None
) -> Appeal:
    """Store a cashier's appeal against a score, linked to the model version."""
    init_schema(connection)
    requested_at = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    cursor = connection.execute(
        "INSERT INTO appeals (worker_id, reason, requested_at, status, model_version) "
        "VALUES (?,?,?,?,?)",
        (request.worker_id, request.reason, requested_at.isoformat(), "open", model_version),
    )
    connection.commit()
    return Appeal(
        appeal_id=int(cursor.lastrowid),
        worker_id=request.worker_id,
        reason=request.reason,
        requested_at=requested_at,
        status="open",
        model_version=model_version,
    )


def list_appeals(connection: sqlite3.Connection, worker_id: str | None = None) -> list[Appeal]:
    """List stored appeals, for one cashier or for all."""
    init_schema(connection)
    if worker_id is None:
        rows = connection.execute("SELECT * FROM appeals ORDER BY appeal_id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM appeals WHERE worker_id = ? ORDER BY appeal_id", (worker_id,)
        ).fetchall()
    return [
        Appeal(
            appeal_id=row["appeal_id"],
            worker_id=row["worker_id"],
            reason=row["reason"],
            requested_at=datetime.fromisoformat(row["requested_at"]),
            status=_appeal_state(row["status"]),
            model_version=row["model_version"],
        )
        for row in rows
    ]


def _appeal_state(value: str) -> AppealState:
    if value not in ("none", "open", "under_review", "resolved"):
        raise ValueError(f"unknown appeal status {value!r}")
    return value
