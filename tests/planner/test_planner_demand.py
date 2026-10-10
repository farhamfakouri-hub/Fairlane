from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from scipy import stats

from ree.config import make_rng
from ree.planner.config import PlannerConfig
from ree.planner.demand.forecast import (
    DayCounts,
    default_model,
    fit_model,
    forecast_cycle,
    forecast_day,
    rolling_backtest,
)
from ree.planner.demand.profile import default_share, normal_date, shape_adjustment, slot_weights
from ree.planner.demand.quantile import (
    exceedance_rate,
    fit_dispersion,
    nbinom_parameters,
    quantile_count,
)
from ree.planner.domain.demand_types import PlannerCalendar

CONFIG = PlannerConfig()
CALENDAR = PlannerCalendar(CONFIG)
TRUE_DISPERSION = 60.0


def synthetic_history(
    weeks: int, level_scale: float = 1.0, seed: int = 1, start: date = date(2026, 8, 30)
) -> list[DayCounts]:
    rng = make_rng(seed, "test-history")
    truth = default_model(CALENDAR, CONFIG)
    history: list[DayCounts] = []
    for offset in range(7 * weeks):
        day = start + timedelta(days=offset)
        arrival = CALENDAR.season(day)[0]
        shares = np.asarray(truth.shares[day.weekday()]) * shape_adjustment(CALENDAR, CONFIG, day)
        shares = shares / shares.sum()
        noise = float(np.exp(rng.normal(0.0, 0.05)))
        means = truth.level[day.weekday()] * level_scale * arrival * noise * shares
        counts = rng.negative_binomial(TRUE_DISPERSION, TRUE_DISPERSION / (TRUE_DISPERSION + means))
        history.append(DayCounts(day, tuple(int(v) for v in counts)))
    return history


def test_default_profiles_sum_to_one_and_peak_in_the_rush() -> None:
    for weekday in range(7):
        share = default_share(CALENDAR, CONFIG, weekday)
        assert share.sum() == pytest.approx(1.0)
        weights = slot_weights(CALENDAR, CONFIG, normal_date(weekday))
        assert weights.max() == CONFIG.demand.rush_intensity
        assert weights.min() == CONFIG.demand.edge_taper
    assert [normal_date(w).weekday() for w in range(7)] == list(range(7))
    assert all(not CALENDAR.is_month_start(normal_date(w)) for w in range(7))


def test_month_start_shifts_customers_into_the_extended_evening_rush() -> None:
    month_start = date(2026, 11, 3)
    plain = date(2026, 11, 17)
    assert month_start.weekday() == plain.weekday()
    model = default_model(CALENDAR, CONFIG)
    late = (20 * 60, 21 * 60)

    def late_share(day: date) -> float:
        forecast = forecast_day(model, CALENDAR, CONFIG, day)
        total = sum(forecast.mean)
        part = sum(
            m
            for s, m in zip(forecast.slot_starts, forecast.mean, strict=True)
            if late[0] <= s < late[1]
        )
        return part / total

    assert late_share(month_start) > late_share(plain) * 1.5


def test_forecast_means_follow_level_season_and_uplift() -> None:
    model = default_model(CALENDAR, CONFIG)
    normal = forecast_day(model, CALENDAR, CONFIG, date(2026, 11, 21))
    assert sum(normal.mean) == pytest.approx(model.level[5])
    christmas = forecast_day(model, CALENDAR, CONFIG, date(2026, 12, 26))
    assert sum(christmas.mean) == pytest.approx(model.level[5] * 1.35)
    assert christmas.season_service == pytest.approx(1.15)
    lifted = forecast_day(model, CALENDAR, CONFIG, date(2026, 11, 21), uplift=1.5)
    assert sum(lifted.mean) == pytest.approx(1.5 * sum(normal.mean))
    cycle = forecast_cycle(
        model, CALENDAR, CONFIG, date(2026, 11, 15), 3, {date(2026, 11, 21): 1.5}
    )
    assert len(cycle) == 21
    assert sum(cycle[6].mean) == pytest.approx(sum(lifted.mean))


def test_the_planning_quantile_is_higher_in_the_rush_than_in_calm_slots() -> None:
    forecast = forecast_day(default_model(CALENDAR, CONFIG), CALENDAR, CONFIG, date(2026, 11, 21))
    for mean, planned, rush in zip(forecast.mean, forecast.planned, forecast.rush, strict=True):
        probability = 0.8 if rush else 0.5
        shape, success = nbinom_parameters(mean, CONFIG.demand.dispersion_k)
        assert planned == int(stats.nbinom.ppf(probability, shape, success))
        assert planned >= int(mean) - 1
    rush_excess = [
        p / m for m, p, r in zip(forecast.mean, forecast.planned, forecast.rush, strict=True) if r
    ]
    calm_excess = [
        p / m
        for m, p, r in zip(forecast.mean, forecast.planned, forecast.rush, strict=True)
        if not r
    ]
    assert np.mean(rush_excess) > np.mean(calm_excess)
    assert quantile_count(0.0, 10.0, 0.8) == 0


def test_dispersion_is_recovered_from_negative_binomial_counts() -> None:
    rng = make_rng(3, "dispersion")
    means = rng.uniform(20, 200, 4000)
    counts = rng.negative_binomial(TRUE_DISPERSION, TRUE_DISPERSION / (TRUE_DISPERSION + means))
    estimate = fit_dispersion(means, counts)
    assert 0.75 * TRUE_DISPERSION < estimate < 1.3 * TRUE_DISPERSION
    assert fit_dispersion([1.0, 2.0], [1, 2]) >= 3.0
    assert exceedance_rate([1, 5, 9], [4, 4, 4]) == pytest.approx(2 / 3)


def test_the_model_recovers_level_and_profile_from_history() -> None:
    history = synthetic_history(10, level_scale=1.15)
    model = fit_model(history, CALENDAR, CONFIG)
    truth = default_model(CALENDAR, CONFIG)
    for weekday in range(7):
        assert model.level[weekday] == pytest.approx(truth.level[weekday] * 1.15, rel=0.12)
        learned = np.asarray(model.shares[weekday])
        assert learned.sum() == pytest.approx(1.0)
        assert np.abs(learned - np.asarray(truth.shares[weekday])).max() < 0.03
    assert 25 < model.dispersion < 140
    assert model.history_days == 70


def test_without_history_the_model_falls_back_to_the_rush_window_profile() -> None:
    model = fit_model([], CALENDAR, CONFIG)
    assert model == default_model(CALENDAR, CONFIG)


def test_the_rolling_backtest_reports_accuracy_bias_and_quantile_coverage() -> None:
    history = synthetic_history(14, level_scale=1.1, seed=5)
    report = rolling_backtest(history, CALENDAR, CONFIG, min_weeks=4)
    assert report.slots > 1500
    assert report.wape < 0.25
    assert abs(report.bias) < 0.06
    assert report.nominal_rush == pytest.approx(0.2) and report.nominal_calm == pytest.approx(0.5)
    assert abs(report.exceedance_rush - report.nominal_rush) < 0.08
    assert abs(report.exceedance_calm - report.nominal_calm) < 0.1
    assert 25 < report.dispersion < 140


def test_a_level_shift_shows_up_as_bias_in_the_backtest() -> None:
    steady = synthetic_history(12, seed=7)
    shifted = [
        DayCounts(
            item.day,
            tuple(int(c * (1.0 if item.day < date(2026, 11, 1) else 1.4)) for c in item.counts),
        )
        for item in steady
    ]
    assert abs(rolling_backtest(shifted, CALENDAR, CONFIG).bias) > abs(
        rolling_backtest(steady, CALENDAR, CONFIG).bias
    )
