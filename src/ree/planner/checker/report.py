"""Violation records and the report of a plan check, with the catalogue of rule codes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Severity = Literal["error", "warning"]

CATALOGUE: dict[str, tuple[Severity, str]] = {
    "UNKNOWN_WORKER": ("error", "assignment to a worker who is not in the plan"),
    "TWO_PATTERNS_SAME_DAY": ("error", "a worker holds two patterns on the same day"),
    "PATTERN_WINDOW_ORDER": ("error", "blocks overlap or a register window leaves its paid window"),
    "PATTERN_MARGINS": ("error", "paid and register windows break the 10 and 5 minute margins"),
    "BLOCK_TOO_SHORT": ("error", "a block is shorter than the minimum shift"),
    "MAX_DAILY_HOURS": ("error", "more paid hours than the daily maximum"),
    "TOO_MANY_CUTS": ("error", "more cuts than the store rule allows"),
    "CUT_LENGTH": ("error", "a cut is outside the allowed length"),
    "OUTSIDE_OPENING_HOURS": ("error", "register time falls outside the opening hours"),
    "OPEN_AND_CLOSE": ("error", "one pattern covers both the opening and the closing"),
    "AVAILABILITY": ("error", "a worker is assigned while blocked or on leave"),
    "POTENTIAL_ABSENCE": (
        "warning",
        "a worker is assigned in a window marked as a possible absence",
    ),
    "WEEKLY_HOURS_CONTRACT": (
        "error",
        "weekly paid hours differ from the contract by more than the tolerance",
    ),
    "MAX_WEEKLY_HOURS": ("error", "more paid hours than the weekly maximum"),
    "AVERAGE_WEEKLY_HOURS": ("error", "average weekly hours above the limit"),
    "MIN_REST": ("error", "less than the minimum rest between two shifts"),
    "WEEKLY_REST": ("error", "no weekly rest gap in a week"),
    "MAX_CONSECUTIVE_DAYS": ("error", "more consecutive working days than allowed"),
    "REGISTERS_1_4_UNSTAFFED": ("error", "registers 1 and 4 are not both staffed"),
    "UNDERSTAFFED": ("warning", "fewer people on duty than the requirement"),
    "SELF_CHECKOUT_UNCOVERED": (
        "warning",
        "fewer cleared workers on duty than self-checkout supervision needs",
    ),
    "BREAK_MISSING": ("error", "a block that needs a break has none"),
    "BREAK_LENGTH": ("error", "a break has the wrong length"),
    "BREAK_OUTSIDE_WINDOW": ("error", "a break starts outside its legal window"),
    "BREAK_MOVED": ("warning", "a break was moved inside the soft tolerance"),
    "BREAK_OUTSIDE_SHIFT": ("error", "a break is not inside the register time of its block"),
    "BREAK_UNKNOWN": ("error", "a break belongs to no block"),
    "BREAK_DUPLICATE": ("error", "a block has more than one break"),
    "BREAKS_LIMIT_CASE": ("warning", "three workers are on break at once (limit case)"),
    "TOO_MANY_BREAKS": ("error", "more workers on break at once than the limit case allows"),
    "BREAK_SEGMENT_MISMATCH": ("error", "break entries and break segments disagree"),
    "DOUBLE_BOOKED": ("error", "a worker is in two places at the same time"),
    "SEGMENT_MISMATCH": ("error", "segments do not cover exactly the register time of the shift"),
    "POST_DOUBLE_OCCUPIED": ("error", "two workers hold the same post at the same time"),
    "POST_EMPTY": ("warning", "a required post is empty"),
    "COVERAGE_GAP": ("warning", "people on duty minus people on break fall below the requirement"),
    "SELF_CHECKOUT_ONLY_VIOLATION": ("error", "a self-checkout-only worker holds a register post"),
    "HANDOVER_INCONSISTENT": ("error", "a handover does not match the posts in the timeline"),
    "RELIEF_INVALID": ("error", "a relief handover does not cover a worker who is on break"),
}


@dataclass(frozen=True)
class Violation:
    """One rule that a plan breaks: its code, severity, message and where it happens."""

    code: str
    severity: Severity
    message: str
    worker_id: str | None = None
    day: date | None = None
    minute: int | None = None
    amount: float | None = None


def violation(
    code: str,
    message: str,
    worker_id: str | None = None,
    day: date | None = None,
    minute: int | None = None,
    amount: float | None = None,
) -> Violation:
    """Create a violation of a catalogued code; the severity comes from the catalogue."""
    return Violation(code, CATALOGUE[code][0], message, worker_id, day, minute, amount)


@dataclass
class Report:
    """The violations found when checking a plan."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        """Violations of severity error."""
        return [item for item in self.violations if item.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        """Violations of severity warning."""
        return [item for item in self.violations if item.severity == "warning"]

    @property
    def ok(self) -> bool:
        """Whether the plan has no errors; warnings do not count."""
        return not self.errors

    def extend(self, items: Iterable[Violation]) -> None:
        """Add violations to the report."""
        self.violations.extend(items)

    def merge(self, other: Report) -> Report:
        """A new report with the violations of both reports."""
        return Report([*self.violations, *other.violations])

    def codes(self, severity: Severity | None = None) -> set[str]:
        """The set of violation codes, optionally of one severity only."""
        return {
            item.code for item in self.violations if severity is None or item.severity == severity
        }

    def count(self, code: str) -> int:
        """Number of violations of one code."""
        return sum(1 for item in self.violations if item.code == code)

    def counts(self) -> dict[str, int]:
        """Number of violations of each code."""
        return dict(Counter(item.code for item in self.violations))

    def shortfall_minutes(self) -> float:
        """Total person-minutes by which the plan is understaffed."""
        return sum(item.amount or 0.0 for item in self.violations if item.code == "UNDERSTAFFED")

    def summary(self, limit: int = 6) -> str:
        """Short text with the numbers of errors and warnings and the first of each code."""
        if not self.violations:
            return "no violations"
        lines = [f"{len(self.errors)} errors, {len(self.warnings)} warnings"]
        for code, count in sorted(self.counts().items()):
            first = next(item for item in self.violations if item.code == code)
            where = " ".join(
                part
                for part in (first.worker_id, first.day.isoformat() if first.day else None)
                if part
            )
            lines.append(f"  {code} x{count} [{first.severity}] {where}: {first.message}")
            if len(lines) > limit:
                lines.append("  ...")
                break
        return chr(10).join(lines)
