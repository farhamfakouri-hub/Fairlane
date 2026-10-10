"""Break rules derived from the labour rules, and the table of rules with their status."""

from __future__ import annotations

from math import ceil

from ree.planner.config import LabourRules
from ree.planner.domain.shifts import Block

__all__ = [
    "RULE_NOTES",
    "LabourRules",
    "block_break_length",
    "break_minutes",
    "break_window",
    "grid_down",
    "grid_up",
    "labour_rules_markdown",
]

BREAK_GRID_MINUTES = 5


def grid_up(minute: int, step: int = BREAK_GRID_MINUTES) -> int:
    """Round a minute up to the grid."""
    return -(-minute // step) * step


def grid_down(minute: int, step: int = BREAK_GRID_MINUTES) -> int:
    """Round a minute down to the grid."""
    return (minute // step) * step


def break_minutes(register_minutes: int, rules: LabourRules) -> int:
    """Break length for a number of register minutes; none for blocks that are too short."""
    hours = register_minutes / 60.0
    if hours + 1e-9 < rules.break_min_block_hours:
        return 0
    raw = rules.break_minutes_per_hour * hours
    step = rules.break_rounding_minutes
    return int(ceil(raw / step - 1e-9) * step)


def block_break_length(block: Block, rules: LabourRules) -> int:
    """Break length of a block."""
    return break_minutes(block.register_minutes, rules)


def break_window(block: Block, rules: LabourRules) -> tuple[int, int] | None:
    """Earliest and latest start of the break of a block, around its middle.

    Returns None when the block has no break or the window is empty.
    """
    length = block_break_length(block, rules)
    if length == 0:
        return None
    nominal = grid_down(block.register_start + block.register_minutes // 2)
    low = grid_up(max(block.register_start, nominal - rules.break_window_tolerance_minutes))
    high = grid_down(
        min(block.register_end - length, nominal + rules.break_window_tolerance_minutes)
    )
    if low > high:
        return None
    return low, high


GENERAL = "General rule, confirm with HR and the collective agreement"
STORE = "Store practice, confirm"
ASSUMED = "Assumed default, confirm"

RULE_NOTES: dict[str, tuple[str, str]] = {
    "max_daily_hours": ("Maximum paid work per day (hours)", GENERAL),
    "min_rest_hours": ("Minimum rest between two shifts (hours)", GENERAL),
    "weekly_rest_hours": ("Weekly rest, consecutive hours", GENERAL),
    "max_weekly_hours": ("Maximum paid work per week (hours)", GENERAL),
    "average_weekly_hours": ("Maximum average paid work per week (hours)", GENERAL),
    "average_weeks": ("Weeks over which the average is taken", GENERAL),
    "max_consecutive_days": (
        "Maximum consecutive working days",
        "Follows from weekly rest, confirm",
    ),
    "max_cuts_per_day": (
        "Maximum cuts (split shifts) per day",
        "Store rule, confirm limits for part-time contracts",
    ),
    "cut_minutes": ("Default cut length (minutes)", STORE),
    "cut_max_minutes": ("Longest cut allowed (minutes)", STORE),
    "hours_tolerance_hours": (
        "Tolerance on weekly paid hours around the contract (hours)",
        ASSUMED,
    ),
    "min_block_hours": ("Minimum length of a shift block (hours)", STORE),
    "break_minutes_per_hour": ("Break minutes per hour worked, paid and counted as work", STORE),
    "break_rounding_minutes": ("Break length is rounded up to a multiple of (minutes)", ASSUMED),
    "break_min_block_hours": ("Blocks shorter than this have no break (hours)", ASSUMED),
    "break_window_tolerance_minutes": (
        "Break start tolerance around the middle of the block (minutes)",
        ASSUMED,
    ),
    "max_simultaneous_breaks": ("Workers on break at the same time", STORE),
    "limit_case_simultaneous_breaks": ("Workers on break at once in a flagged limit case", STORE),
    "max_break_shift_minutes": ("Soft limit for moving a break on the day (minutes)", STORE),
    "clock_in_margin_minutes": ("Paid minutes before the register start", STORE),
    "clock_out_margin_minutes": ("Paid minutes after the register end", STORE),
    "margin_tolerance_minutes": ("Tolerance on the two margins (minutes)", ASSUMED),
    "opening_slack_minutes": (
        "A register start within this of opening counts as an opening shift (minutes)",
        ASSUMED,
    ),
    "closing_slack_minutes": (
        "A register end within this of closing counts as a closing shift (minutes)",
        ASSUMED,
    ),
    "planning_weeks": ("Roster cycle length (weeks)", "Store answer: issued every 3 weeks"),
    "sunday_mandatory": (
        "Sunday work is mandatory and enters the burden rotation",
        "Store answer, confirm rules and compensation",
    ),
    "history_weeks": ("Weeks of burden history used for fairness", ASSUMED),
}


def labour_rules_markdown(rules: LabourRules) -> str:
    """Markdown table of the labour rules with defaults and status."""
    lines = [
        "| Parameter | Meaning | Default | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name in LabourRules.model_fields:
        meaning, status = RULE_NOTES[name]
        lines.append(f"| `{name}` | {meaning} | {getattr(rules, name)} | {status} |")
    return chr(10).join(lines)
