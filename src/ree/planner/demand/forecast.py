"""Demand forecast: level and profile by weekday, customers by slot, and a rolling backtest."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ree.planner.config import WEEKDAYS_IN_ORDER, PlannerConfig
from ree.planner.demand.profile import default_share, normalise, shape_adjustment
from ree.planner.demand.quantile import exceedance_rate, fit_dispersion, quantile_count
from ree.planner.domain.demand_types import PlannerCalendar


@dataclass(frozen=True)
class DayCounts:
    """Customer counts of one day, slot by slot."""

    day: date
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        """Customers of the whole day."""
        return sum(self.counts)


@dataclass(frozen=True)
class DemandModel:
    """Forecast model: level and slot shares for each weekday, and the dispersion."""

    level: dict[int, float]
    shares: dict[int, tuple[float, ...]]
    dispersion: float
    history_days: int = 0


@dataclass(frozen=True)
class DayForecast:
    """Forecast of one day: mean and planned customers by slot, rush flags and season factors."""

    day: date
    slot_starts: tuple[int, ...]
    mean: tuple[float, ...]
    planned: tuple[int, ...]
    rush: tuple[bool, ...]
    season_arrival: float
    season_service: float


@dataclass(frozen=True)
class BacktestReport:
    """How well the forecast did in a rolling backtest: error, bias and exceedance rates."""

    wape: float
    bias: float
    exceedance_rush: float
    exceedance_calm: float
    nominal_rush: float
    nominal_calm: float
    slots: int
    dispersion: float


def default_model(calendar: PlannerCalendar, config: PlannerConfig) -> DemandModel:
    """Model built from the configured daily customers, used when there is no history."""
    level = {
        weekday: float(config.demand.daily_customers[WEEKDAYS_IN_ORDER[weekday]])
        for weekday in range(7)
    }
    shares = {
        weekday: tuple(float(v) for v in default_share(calendar, config, weekday))
        for weekday in range(7)
    }
    return DemandModel(level, shares, config.demand.dispersion_k)


def _day_shares(
    model_shares: dict[int, tuple[float, ...]],
    calendar: PlannerCalendar,
    config: PlannerConfig,
    day: date,
) -> np.ndarray:
    base = np.asarray(model_shares[day.weekday()], dtype=np.float64)
    return normalise(base * shape_adjustment(calendar, config, day))


def fit_model(
    history: Iterable[DayCounts], calendar: PlannerCalendar, config: PlannerConfig
) -> DemandModel:
    """Fit the model on history.

    The level of each weekday is a smoothed average and the profile is blended with the
    default; the dispersion is estimated from one-step forecast errors.
    """
    fallback = default_model(calendar, config)
    by_weekday: dict[int, list[DayCounts]] = defaultdict(list)
    ordered = sorted(history, key=lambda item: item.day)
    for item in ordered:
        by_weekday[item.day.weekday()].append(item)
    alpha = config.demand.level_ewma_alpha
    prior = config.demand.profile_prior_customers
    level = dict(fallback.level)
    shares = dict(fallback.shares)
    for weekday, days in by_weekday.items():
        totals = [item.total / calendar.season(item.day)[0] for item in days]
        estimate = totals[0]
        for value in totals[1:]:
            estimate = alpha * value + (1.0 - alpha) * estimate
        level[weekday] = float(estimate)
        plain = [item for item in days if not calendar.is_month_start(item.day)]
        if len(plain) >= config.demand.min_history_weeks:
            width = len(plain[0].counts)
            pooled = np.zeros(width, dtype=np.float64)
            for item in plain:
                pooled += np.asarray(item.counts, dtype=np.float64)
            base = np.asarray(fallback.shares[weekday], dtype=np.float64)
            blended = (pooled + prior * base) / (pooled.sum() + prior)
            shares[weekday] = tuple(float(v) for v in blended)
    means: list[float] = []
    counts: list[float] = []
    for days in by_weekday.values():
        previous: float | None = None
        for item in days:
            adjusted = item.total / calendar.season(item.day)[0]
            if previous is not None:
                forecast_total = previous * calendar.season(item.day)[0]
                slot_means = forecast_total * _day_shares(shares, calendar, config, item.day)
                means.extend(float(v) for v in slot_means)
                counts.extend(float(v) for v in item.counts)
            previous = adjusted if previous is None else alpha * adjusted + (1 - alpha) * previous
    dispersion = fit_dispersion(means, counts) if means else config.demand.dispersion_k
    return DemandModel(level, shares, dispersion, len(ordered))


def forecast_day(
    model: DemandModel,
    calendar: PlannerCalendar,
    config: PlannerConfig,
    day: date,
    uplift: float = 1.0,
) -> DayForecast:
    """Forecast of one day, with an optional uplift on the day's customers."""
    arrival, service = calendar.season(day)
    starts = calendar.slot_starts(day)
    shares = _day_shares(model.shares, calendar, config, day)
    total = model.level[day.weekday()] * arrival * uplift
    mean = tuple(float(v) for v in total * shares)
    rush = tuple(calendar.slot_is_rush(day, start) for start in starts)
    planned = tuple(
        quantile_count(
            value,
            model.dispersion,
            config.demand.quantile_rush if is_rush else config.demand.quantile_calm,
        )
        for value, is_rush in zip(mean, rush, strict=True)
    )
    return DayForecast(day, starts, mean, planned, rush, arrival, service)


def forecast_cycle(
    model: DemandModel,
    calendar: PlannerCalendar,
    config: PlannerConfig,
    start: date,
    weeks: int,
    uplifts: dict[date, float] | None = None,
) -> list[DayForecast]:
    """Forecasts of every day of a roster cycle."""
    uplift = uplifts or {}
    return [
        forecast_day(model, calendar, config, day, uplift.get(day, 1.0))
        for day in calendar.cycle_dates(start, weeks)
    ]


def rolling_backtest(
    history: Sequence[DayCounts],
    calendar: PlannerCalendar,
    config: PlannerConfig,
    min_weeks: int = 4,
) -> BacktestReport:
    """Forecast each week from the weeks before it and compare with what happened."""
    ordered = sorted(history, key=lambda item: item.day)
    first_week = calendar.week_start(ordered[0].day)
    weeks: dict[int, list[DayCounts]] = defaultdict(list)
    for item in ordered:
        weeks[(item.day - first_week).days // 7].append(item)
    forecasts: list[float] = []
    actuals: list[float] = []
    planned_rush: list[float] = []
    actual_rush: list[float] = []
    planned_calm: list[float] = []
    actual_calm: list[float] = []
    last_dispersion = config.demand.dispersion_k
    for week in sorted(weeks):
        if week < min_weeks:
            continue
        cutoff = first_week + timedelta(days=7 * week)
        training = [item for item in ordered if item.day < cutoff]
        model = fit_model(training, calendar, config)
        last_dispersion = model.dispersion
        for item in weeks[week]:
            forecast = forecast_day(model, calendar, config, item.day)
            for mean, planned, is_rush, actual in zip(
                forecast.mean, forecast.planned, forecast.rush, item.counts, strict=True
            ):
                forecasts.append(mean)
                actuals.append(float(actual))
                if is_rush:
                    planned_rush.append(float(planned))
                    actual_rush.append(float(actual))
                else:
                    planned_calm.append(float(planned))
                    actual_calm.append(float(actual))
    total_actual = float(np.sum(actuals)) if actuals else 0.0
    wape = (
        float(np.sum(np.abs(np.subtract(forecasts, actuals))) / total_actual)
        if total_actual
        else float("nan")
    )
    bias = (
        float((np.sum(forecasts) - total_actual) / total_actual) if total_actual else float("nan")
    )
    return BacktestReport(
        wape=wape,
        bias=bias,
        exceedance_rush=exceedance_rate(actual_rush, planned_rush),
        exceedance_calm=exceedance_rate(actual_calm, planned_calm),
        nominal_rush=1.0 - config.demand.quantile_rush,
        nominal_calm=1.0 - config.demand.quantile_calm,
        slots=len(actuals),
        dispersion=last_dispersion,
    )
