"""Audit log of the planner: roster generations, overrides, proposals and swap checks."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MEMORY = ":memory:"

SCHEMA = """
create table if not exists planner_audit (
    id integer primary key autoincrement,
    at text not null,
    kind text not null,
    subject text not null,
    detail text not null
)
"""


@dataclass(frozen=True)
class AuditRecord:
    """One line of the audit log."""

    record_id: int
    at: str
    kind: str
    subject: str
    detail: dict[str, object]


class AuditStore:
    """Append-only audit log kept in SQLite, in memory unless a path is given."""

    def __init__(self, path: str | Path = MEMORY) -> None:
        target = str(path)
        if target != MEMORY:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(target, check_same_thread=False)
        self.connection.execute(SCHEMA)
        self.connection.commit()

    def append(
        self, kind: str, subject: str, detail: dict[str, object], at: datetime | None = None
    ) -> AuditRecord:
        """Add a record and return it."""
        stamp = (at or datetime.now()).isoformat(timespec="seconds")
        cursor = self.connection.execute(
            "insert into planner_audit (at, kind, subject, detail) values (?, ?, ?, ?)",
            (stamp, kind, subject, json.dumps(detail, sort_keys=True, default=str)),
        )
        self.connection.commit()
        return AuditRecord(int(cursor.lastrowid or 0), stamp, kind, subject, detail)

    def records(self, kind: str | None = None) -> list[AuditRecord]:
        """Records in order, optionally of one kind only."""
        query = "select id, at, kind, subject, detail from planner_audit"
        parameters: tuple[str, ...] = ()
        if kind is not None:
            query += " where kind = ?"
            parameters = (kind,)
        rows = self.connection.execute(query + " order by id", parameters).fetchall()
        return [AuditRecord(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
