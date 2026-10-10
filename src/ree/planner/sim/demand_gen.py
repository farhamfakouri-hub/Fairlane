"""Random customer arrivals, service times and patience for the queue simulator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ree.planner.config import ServiceConfig
from ree.planner.sim.queue_sim import DayCustomers

SECONDS_PER_MINUTE = 60.0


def lognormal_unit(rng: np.random.Generator, cv: float, size: int) -> np.ndarray:
    """Lognormal factors with mean one and a given coefficient of variation."""
    sigma = float(np.sqrt(np.log1p(cv * cv)))
    return rng.lognormal(-0.5 * sigma * sigma, sigma, size)


def draw_service(
    rng: np.random.Generator, service: ServiceConfig, season_service: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed and variable service seconds of customers, scaled by the season."""
    if service.distribution == "exponential":
        mean = (service.overhead_seconds + service.work_seconds) * season_service
        return np.zeros(count), rng.exponential(mean, count)
    fixed = np.full(count, service.overhead_seconds)
    variable = (
        service.work_seconds * season_service * lognormal_unit(rng, service.service_cv, count)
    )
    return fixed, variable


def draw_patience(rng: np.random.Generator, service: ServiceConfig, count: int) -> np.ndarray:
    """Seconds each customer is willing to wait; infinite when patience is switched off."""
    if service.patience_minutes <= 0.0:
        return np.full(count, np.inf)
    return rng.exponential(service.patience_minutes * SECONDS_PER_MINUTE, count)


def draw_arrivals(
    rng: np.random.Generator,
    slot_starts: Sequence[int],
    slot_means: Sequence[float],
    slot_minutes: int,
    dispersion: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Arrival times in seconds: negative binomial counts in each slot, spread uniformly
    inside it.
    """
    times: list[np.ndarray] = []
    for start, mean in zip(slot_starts, slot_means, strict=True):
        expected = max(mean * scale, 0.0)
        if expected <= 0.0:
            continue
        count = int(rng.negative_binomial(dispersion, dispersion / (dispersion + expected)))
        if count:
            offsets = rng.uniform(0.0, slot_minutes, count)
            times.append((start + offsets) * SECONDS_PER_MINUTE)
    if not times:
        return np.zeros(0)
    return np.sort(np.concatenate(times))


def draw_day_customers(
    rng: np.random.Generator,
    slot_starts: Sequence[int],
    slot_means: Sequence[float],
    slot_minutes: int,
    dispersion: float,
    service: ServiceConfig,
    season_service: float = 1.0,
    scale: float = 1.0,
) -> DayCustomers:
    """Arrivals, service times and patience of one day's customers."""
    arrival = draw_arrivals(rng, slot_starts, slot_means, slot_minutes, dispersion, scale)
    fixed, variable = draw_service(rng, service, season_service, len(arrival))
    return DayCustomers(arrival, fixed, variable, draw_patience(rng, service, len(arrival)))


def stationary_customers(
    rng: np.random.Generator,
    rate_per_second: float,
    count: int,
    mean_service_seconds: float,
    exponential: bool = True,
    patience_seconds: float = float("inf"),
) -> DayCustomers:
    """A steady stream of customers, used to compare the simulator with Erlang C."""
    arrival = np.cumsum(rng.exponential(1.0 / rate_per_second, count))
    if exponential:
        variable = rng.exponential(mean_service_seconds, count)
    else:
        variable = np.full(count, mean_service_seconds)
    return DayCustomers(arrival, np.zeros(count), variable, np.full(count, patience_seconds))
