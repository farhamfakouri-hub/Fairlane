"""Generation of the allowed shift patterns from a specification of families and windows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ree.config import StrictModel
from ree.planner.config import (
    Clock,
    LabourRules,
    OpeningConfig,
    PlannerConfig,
    _clock,
    clock_to_minutes,
)
from ree.planner.domain.shifts import Block, ShiftPattern, make_block, pattern_id_for

DayKind = Literal["weekday", "sunday"]
SUNDAY_WEEKDAY = 6


class FamilySpec(StrictModel):
    """A family of shifts: start and end windows and, for split shifts, the cut lengths."""

    name: str
    kind: Literal["continuous", "split"] = "continuous"
    enabled: bool = True
    start_window: tuple[Clock, Clock]
    end_window: tuple[Clock, Clock] | None = None
    end_anchor: Literal["close"] | None = None
    first_end_window: tuple[Clock, Clock] | None = None
    cut_minutes: tuple[int, ...] = ()

    @field_validator("start_window", "end_window", "first_end_window")
    @classmethod
    def windows_valid(cls, value: tuple[str, str] | None) -> tuple[str, str] | None:
        """Windows must be valid clock times and start no later than end."""
        if value is None:
            return None
        for part in value:
            _clock(part)
        if clock_to_minutes(value[0]) > clock_to_minutes(value[1]):
            raise ValueError("window start must not be after its end")
        return value

    @model_validator(mode="after")
    def shape_is_complete(self) -> FamilySpec:
        """A family needs an end window or an end anchor; a split family also needs a first end
        window and cut lengths.
        """
        if self.end_window is None and self.end_anchor is None:
            raise ValueError("a family needs an end window or an end anchor")
        if self.kind == "split" and (self.first_end_window is None or not self.cut_minutes):
            raise ValueError("a split family needs first_end_window and cut_minutes")
        return self


class DayKindSpec(StrictModel):
    """The shift families of one kind of day."""

    families: list[FamilySpec]


def _default_weekday() -> DayKindSpec:
    return DayKindSpec(
        families=[
            FamilySpec(
                name="morning",
                start_window=("08:30", "10:10"),
                end_window=("12:55", "16:15"),
            ),
            FamilySpec(
                name="evening",
                start_window=("14:15", "17:40"),
                end_anchor="close",
            ),
            FamilySpec(
                name="split",
                kind="split",
                start_window=("09:30", "09:40"),
                first_end_window=("13:20", "13:40"),
                cut_minutes=(60, 90),
                end_window=("19:30", "20:10"),
            ),
            FamilySpec(
                name="midday",
                enabled=False,
                start_window=("10:30", "12:00"),
                end_window=("14:30", "16:30"),
            ),
        ]
    )


def _default_sunday() -> DayKindSpec:
    return DayKindSpec(
        families=[
            FamilySpec(
                name="sunday_morning",
                start_window=("08:30", "09:30"),
                end_anchor="close",
            )
        ]
    )


class PatternSpec(StrictModel):
    """Specification of the patterns: grids, length limits and families for each day kind."""

    grid_minutes: int = 5
    solver_grid_minutes: int = 30
    min_register_minutes: int = 180
    max_register_minutes: int = 570
    min_block_minutes: int = 180
    weekday: DayKindSpec = Field(default_factory=_default_weekday)
    sunday: DayKindSpec = Field(default_factory=_default_sunday)

    @model_validator(mode="after")
    def grids_are_multiples(self) -> PatternSpec:
        """The solver grid must be a multiple of the pattern grid."""
        if self.solver_grid_minutes % self.grid_minutes != 0:
            raise ValueError("solver_grid_minutes must be a multiple of grid_minutes")
        return self


def load_pattern_spec(path: str | Path | None = None) -> PatternSpec:
    """Load a pattern specification from YAML, or the defaults without a path."""
    if path is None:
        return PatternSpec()
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PatternSpec.model_validate(payload or {})


def dump_pattern_spec(spec: PatternSpec, path: str | Path) -> None:
    """Write a pattern specification to a YAML file."""
    text = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    Path(path).write_bytes(text.encode("utf-8"))


def _grid(window: tuple[str, str], step: int) -> Iterator[int]:
    low, high = clock_to_minutes(window[0]), clock_to_minutes(window[1])
    first = -(-low // step) * step
    yield from range(first, high + 1, step)


def _close_minute(day_kind: DayKind, opening: OpeningConfig) -> int:
    pair = opening.sunday if day_kind == "sunday" else opening.monday_to_saturday
    return clock_to_minutes(pair[1])


def _continuous(
    family: FamilySpec, step: int, close: int, margins: tuple[int, int]
) -> Iterator[tuple[Block, ...]]:
    ends: list[int]
    for start in _grid(family.start_window, step):
        if family.end_anchor == "close":
            ends = [close]
        else:
            assert family.end_window is not None
            ends = list(_grid(family.end_window, step))
        for end in ends:
            if end > start:
                yield (make_block(start, end, margins[0], margins[1]),)


def _split(family: FamilySpec, step: int, margins: tuple[int, int]) -> Iterator[tuple[Block, ...]]:
    assert family.first_end_window is not None and family.end_window is not None
    for start in _grid(family.start_window, step):
        for first_end in _grid(family.first_end_window, step):
            for cut in family.cut_minutes:
                second_start = first_end + cut
                if second_start % step != 0:
                    continue
                for second_end in _grid(family.end_window, step):
                    if second_end > second_start:
                        yield (
                            make_block(start, first_end, margins[0], margins[1]),
                            make_block(second_start, second_end, margins[0], margins[1]),
                        )


def generate_patterns(
    spec: PatternSpec,
    rules: LabourRules,
    opening: OpeningConfig,
    day_kind: DayKind,
    grid_minutes: int | None = None,
) -> tuple[ShiftPattern, ...]:
    """Generate every pattern a day kind allows.

    Patterns must respect the minimum and maximum register time, the block length and the
    daily paid-hours limit.
    """
    step = grid_minutes if grid_minutes is not None else spec.grid_minutes
    margins = (rules.clock_in_margin_minutes, rules.clock_out_margin_minutes)
    close = _close_minute(day_kind, opening)
    families = (spec.sunday if day_kind == "sunday" else spec.weekday).families
    found: dict[str, ShiftPattern] = {}
    for family in families:
        if not family.enabled:
            continue
        candidates = (
            _continuous(family, step, close, margins)
            if family.kind == "continuous"
            else _split(family, step, margins)
        )
        for blocks in candidates:
            total = sum(block.register_minutes for block in blocks)
            if not spec.min_register_minutes <= total <= spec.max_register_minutes:
                continue
            if any(block.register_minutes < spec.min_block_minutes for block in blocks):
                continue
            if sum(block.paid_minutes for block in blocks) > rules.max_daily_hours * 60:
                continue
            pattern = ShiftPattern(
                pattern_id=pattern_id_for(family.kind, blocks), kind=family.kind, blocks=blocks
            )
            found.setdefault(pattern.pattern_id, pattern)
    return tuple(
        sorted(found.values(), key=lambda item: (item.register_start, item.register_end, item.kind))
    )


class PatternSet:
    """The weekday and Sunday patterns, with a lookup by identifier."""

    def __init__(self, weekday: tuple[ShiftPattern, ...], sunday: tuple[ShiftPattern, ...]):
        self.weekday = weekday
        self.sunday = sunday
        self.by_id: dict[str, ShiftPattern] = {
            pattern.pattern_id: pattern for pattern in (*weekday, *sunday)
        }

    def for_weekday(self, weekday: int) -> tuple[ShiftPattern, ...]:
        """Patterns allowed on a weekday, counted with Monday as 0."""
        return self.sunday if weekday == SUNDAY_WEEKDAY else self.weekday

    def __len__(self) -> int:
        return len(self.by_id)


def solver_pattern_set(config: PlannerConfig, spec: PatternSpec | None = None) -> PatternSet:
    """Coarse pattern set on the solver grid, used by the roster solver."""
    used = spec if spec is not None else PatternSpec()
    grid = used.solver_grid_minutes
    return PatternSet(
        generate_patterns(used, config.rules, config.opening, "weekday", grid),
        generate_patterns(used, config.rules, config.opening, "sunday", grid),
    )


def full_pattern_set(config: PlannerConfig, spec: PatternSpec | None = None) -> PatternSet:
    """Fine pattern set on the 5-minute grid, used by the daily plan and the checker."""
    used = spec if spec is not None else PatternSpec()
    return PatternSet(
        generate_patterns(used, config.rules, config.opening, "weekday"),
        generate_patterns(used, config.rules, config.opening, "sunday"),
    )
