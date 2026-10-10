"""Validation of the queue simulator against the Erlang C formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ree.config import make_rng
from ree.planner.sim.demand_gen import stationary_customers
from ree.planner.sim.queue_sim import QueueSettings, StaffInterval, simulate_day
from ree.planner.staffing.erlang import mean_wait, wait_exceeds

SERVICE_SECONDS = 120.0
TARGET_MINUTES = 3.0
CASES: tuple[tuple[int, float], ...] = ((2, 1.5), (3, 2.25), (4, 3.0), (6, 4.5), (8, 6.0))
WARM_UP_SHARE = 0.1


def simulate_stationary(
    servers: int, load: float, customers: int, seed: int, discipline: str
) -> tuple[float, float]:
    """Mean wait and share over target of a long steady stream, in minutes."""
    rng = make_rng(seed, "planner-erlang-check")
    stream = stationary_customers(rng, load / SERVICE_SECONDS, customers, SERVICE_SECONDS)
    horizon = float(stream.arrival[-1]) + 1.0e6
    intervals = [
        StaffInterval(index + 1, 0.0, horizon, f"S{index:02d}") for index in range(servers)
    ]
    outcome = simulate_day(
        stream, intervals, lambda worker, second: 1.0, QueueSettings(discipline=discipline)
    )
    settled = outcome.wait[int(WARM_UP_SHARE * customers) :]
    return float(settled.mean()) / 60.0, float((settled > TARGET_MINUTES * 60.0).mean())


def queue_validation(
    cases: tuple[tuple[int, float], ...] = CASES, customers: int = 60000, seeds: int = 4
) -> pd.DataFrame:
    """Compare simulated waits with Erlang C for several cases and queue disciplines."""
    rows: list[dict[str, float | int | str]] = []
    service_minutes = SERVICE_SECONDS / 60.0
    for servers, load in cases:
        analytic_wait = mean_wait(servers, load, service_minutes)
        analytic_tail = wait_exceeds(servers, load, service_minutes, TARGET_MINUTES)
        for discipline in ("shared", "shortest_line"):
            results = [
                simulate_stationary(servers, load, customers, seed, discipline)
                for seed in range(seeds)
            ]
            waits = np.array([item[0] for item in results])
            tails = np.array([item[1] for item in results])
            rows.append(
                {
                    "servers": servers,
                    "load": load,
                    "discipline": discipline,
                    "simulated_wait_min": float(waits.mean()),
                    "simulated_tail": float(tails.mean()),
                    "erlang_wait_min": float(analytic_wait),
                    "erlang_tail": float(analytic_tail),
                    "relative_gap": float(waits.mean() / analytic_wait - 1.0),
                }
            )
    return pd.DataFrame(rows)
