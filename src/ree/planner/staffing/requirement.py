"""Staffing requirement of each slot from the forecast: queueing, self-checkout and relief."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from itertools import pairwise
from math import ceil

from ree.planner.config import PlannerConfig
from ree.planner.demand.forecast import DayForecast, DemandModel, forecast_cycle
from ree.planner.domain.demand_types import PlannerCalendar, RequirementTable, SlotRequirement
from ree.planner.staffing.erlang import required_servers


def slot_requirement(
    planned_customers: float,
    rush: bool,
    slot_start: int,
    season_service: float,
    calendar: PlannerCalendar,
    config: PlannerConfig,
) -> SlotRequirement:
    """Requirement of one slot.

    Registers come from the queueing model, self-checkout places and relief workers from
    the configuration. Headcount is their sum.
    """
    staffing = config.staffing
    service = staffing.service_minutes * season_service * staffing.safety_multiplier
    rate = planned_customers / config.demand.slot_minutes
    load = rate * service
    alpha = staffing.alpha_rush if rush else staffing.alpha_calm
    registers, saturated = required_servers(
        load,
        service,
        staffing.t_target_minutes,
        alpha,
        staffing.minimum_open_registers,
        staffing.max_registers,
    )
    k_sc = staffing.k_sc_rush if rush else staffing.k_sc_calm
    zone1 = staffing.sco_zone1_rush if rush else staffing.sco_zone1_calm
    zone2 = staffing.sco_zone2_rush if rush else staffing.sco_zone2_calm
    floaters = calendar.floaters_required(slot_start)
    buffer = max(floaters, ceil(staffing.break_share * (registers + k_sc) - 1e-9))
    headcount = max(registers, staffing.minimum_open_registers) + k_sc + buffer
    return SlotRequirement(
        registers=registers,
        k_sc=k_sc,
        sco_zone1=zone1,
        sco_zone2=zone2,
        floaters=buffer,
        headcount=headcount,
        rush=rush,
        offered_load=load,
        planned_customers=float(planned_customers),
        saturated=saturated,
    )


def build_requirement(
    forecasts: Iterable[DayForecast], calendar: PlannerCalendar, config: PlannerConfig
) -> RequirementTable:
    """Requirement table from the forecasts of several days."""
    table = RequirementTable(
        slot_minutes=config.demand.slot_minutes, opening_order=config.staffing.opening_order
    )
    for forecast in forecasts:
        rows = []
        for start, planned, rush in zip(
            forecast.slot_starts, forecast.planned, forecast.rush, strict=True
        ):
            rows.append(
                (
                    start,
                    slot_requirement(
                        planned, rush, start, forecast.season_service, calendar, config
                    ),
                )
            )
        table.days[forecast.day] = tuple(rows)
    return table


def requirement_for_cycle(
    model: DemandModel,
    calendar: PlannerCalendar,
    config: PlannerConfig,
    start: date,
    weeks: int,
    uplifts: dict[date, float] | None = None,
) -> RequirementTable:
    """Forecast a roster cycle and turn it into a requirement table."""
    return build_requirement(
        forecast_cycle(model, calendar, config, start, weeks, uplifts), calendar, config
    )


def requirement_summary(table: RequirementTable) -> list[str]:
    """One line per day: register range, peak headcount and staff hours."""
    lines = []
    for day in table.dates():
        rows = table.slots(day)
        peak = max(row.registers for _, row in rows)
        calm = min(row.registers for _, row in rows)
        top = max(row.headcount for _, row in rows)
        hours = table.staff_minutes(day) / 60.0
        lines.append(
            f"{day.isoformat()} {day.strftime('%a')}: registers {calm} to {peak}, "
            f"headcount up to {top}, {hours:.0f} staff hours"
        )
    return lines


def estimate_service_minutes(
    starts_by_register: Mapping[int, Sequence[float]], idle_cutoff_seconds: float = 300.0
) -> float:
    """Median gap between transaction starts at a register, in minutes.

    Gaps above the idle cutoff are ignored.
    """
    gaps: list[float] = []
    for starts in starts_by_register.values():
        ordered = sorted(starts)
        gaps.extend(
            later - earlier
            for earlier, later in pairwise(ordered)
            if 0.0 < later - earlier <= idle_cutoff_seconds
        )
    if not gaps:
        raise ValueError("no busy gaps between transaction starts")
    return float(sorted(gaps)[len(gaps) // 2]) / 60.0
