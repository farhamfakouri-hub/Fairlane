"""Requirement types and the planner calendar: opening hours, rush windows and seasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ree.planner.config import OPENING_ORDER, PlannerConfig, clock_to_minutes

SUNDAY = 6
SATURDAY = 5
FRIDAY = 4


def _month_day(value: str) -> tuple[int, int]:
    month, day = value.split("-")
    return int(month), int(day)


def _in_window(day: date, window: tuple[str, str]) -> bool:
    start, end = _month_day(window[0]), _month_day(window[1])
    here = (day.month, day.day)
    if start <= end:
        return start <= here <= end
    return here >= start or here <= end


@dataclass(frozen=True)
class SlotRequirement:
    """What one time slot needs: open registers, self-checkout places, relief and headcount."""

    registers: int
    k_sc: int
    sco_zone1: int
    sco_zone2: int
    floaters: int
    headcount: int
    rush: bool
    offered_load: float
    planned_customers: float
    saturated: bool = False


@dataclass
class RequirementTable:
    """Requirement of every slot of every day, with the order in which registers open."""

    slot_minutes: int
    days: dict[date, tuple[tuple[int, SlotRequirement], ...]] = field(default_factory=dict)
    opening_order: tuple[int, ...] = OPENING_ORDER

    def slots(self, day: date) -> tuple[tuple[int, SlotRequirement], ...]:
        """Slots of a day with their start minutes."""
        return self.days.get(day, ())

    def at(self, day: date, minute: int) -> SlotRequirement | None:
        """Requirement of the slot that contains a minute, or None."""
        for start, requirement in self.days.get(day, ()):
            if start <= minute < start + self.slot_minutes:
                return requirement
        return None

    def dates(self) -> list[date]:
        """Dates that have a requirement, sorted."""
        return sorted(self.days)

    def opening_hours(self, day: date) -> tuple[int, int] | None:
        """Start of the first slot and end of the last slot of a day, or None."""
        slots = self.days.get(day)
        if not slots:
            return None
        return slots[0][0], slots[-1][0] + self.slot_minutes

    def required_registers(self, day: date, minute: int) -> tuple[int, ...]:
        """Registers that must be open at a minute, in opening order."""
        requirement = self.at(day, minute)
        if requirement is None:
            return ()
        return tuple(self.opening_order[: requirement.registers])

    def staff_minutes(self, day: date) -> int:
        """Staff minutes a day needs: headcount times slot length, summed."""
        return sum(item.headcount * self.slot_minutes for _, item in self.slots(day))


class PlannerCalendar:
    """Answers calendar questions for the planner from its configuration."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self.slot_minutes = config.demand.slot_minutes

    def opening(self, day: date) -> tuple[int, int]:
        """Opening and closing minute of a day."""
        pair = (
            self.config.opening.sunday
            if day.weekday() == SUNDAY
            else self.config.opening.monday_to_saturday
        )
        return clock_to_minutes(pair[0]), clock_to_minutes(pair[1])

    def is_month_start(self, day: date) -> bool:
        """Whether the day falls in the busy days around the start of the month."""
        rush = self.config.rush
        return day.day >= rush.month_start_first_day or day.day <= rush.month_start_last_day

    def rush_windows(self, day: date) -> tuple[tuple[int, int], ...]:
        """Rush windows of a day as merged minute intervals inside opening hours."""
        weekday = day.weekday()
        open_minute, close_minute = self.opening(day)
        rush = self.config.rush
        if weekday == SUNDAY:
            windows = [
                (clock_to_minutes(rush.sunday_window[0]), clock_to_minutes(rush.sunday_window[1]))
            ]
        else:
            windows = [
                (clock_to_minutes(start), clock_to_minutes(end))
                for start, end in rush.weekday_windows
            ]
            extend = weekday in (FRIDAY, SATURDAY) or (
                rush.month_start_extends_evening and self.is_month_start(day)
            )
            if extend and windows:
                windows[-1] = (windows[-1][0], clock_to_minutes(rush.friday_saturday_evening_end))
            if weekday in (FRIDAY, SATURDAY):
                morning = rush.friday_saturday_morning_window
                windows.append((clock_to_minutes(morning[0]), clock_to_minutes(morning[1])))
        limit = close_minute if weekday == SUNDAY else close_minute - rush.quiet_last_minutes
        clipped = sorted(
            (max(start, open_minute), min(end, limit))
            for start, end in windows
            if min(end, limit) > max(start, open_minute)
        )
        merged: list[tuple[int, int]] = []
        for start, end in clipped:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return tuple(merged)

    def is_rush(self, day: date, minute: int) -> bool:
        """Whether a minute is inside a rush window."""
        return any(start <= minute < end for start, end in self.rush_windows(day))

    def slot_starts(self, day: date) -> tuple[int, ...]:
        """Start minutes of the slots of a day."""
        open_minute, close_minute = self.opening(day)
        return tuple(range(open_minute, close_minute, self.slot_minutes))

    def slot_is_rush(self, day: date, slot_start: int) -> bool:
        """Whether the middle of a slot is inside a rush window."""
        return self.is_rush(day, slot_start + self.slot_minutes // 2)

    def season(self, day: date) -> tuple[float, float]:
        """Multipliers of customer arrivals and service time that apply on a day."""
        season = self.config.season
        arrival, service = 1.0, 1.0
        if self.is_month_start(day):
            arrival *= season.month_start_arrival
            service *= season.month_start_service
        if _in_window(day, season.christmas_window):
            arrival *= season.christmas_arrival
            service *= season.christmas_service
        if _in_window(day, season.late_summer_window):
            arrival *= season.late_summer_arrival
            service *= season.late_summer_service
        arrival *= season.manager_arrival_override.get(day.isoformat(), 1.0)
        return arrival, service

    def floaters_required(self, minute: int) -> int:
        """Relief workers required at a minute, which changes in the afternoon."""
        staffing = self.config.staffing
        switch = clock_to_minutes(staffing.floater_switch_time)
        return staffing.floaters_morning if minute < switch else staffing.floaters_afternoon

    def week_start(self, day: date) -> date:
        """Sunday that starts the week of a date."""
        return day - timedelta(days=(day.weekday() + 1) % 7)

    def cycle_dates(self, start: date, weeks: int) -> list[date]:
        """Dates of a roster cycle of several weeks."""
        return [start + timedelta(days=offset) for offset in range(7 * weeks)]
