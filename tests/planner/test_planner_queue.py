from __future__ import annotations

import numpy as np
import pytest

from ree.config import make_rng
from ree.planner.config import ServiceConfig
from ree.planner.eval.validation import queue_validation, simulate_stationary
from ree.planner.sim.demand_gen import (
    draw_arrivals,
    draw_day_customers,
    draw_patience,
    draw_service,
    lognormal_unit,
    stationary_customers,
)
from ree.planner.sim.queue_sim import DayCustomers, QueueSettings, StaffInterval, simulate_day
from ree.planner.staffing.erlang import mean_wait, wait_exceeds


def customers(arrivals, variable, fixed=0.0, patience=float("inf")) -> DayCustomers:
    count = len(arrivals)
    return DayCustomers(
        np.array(arrivals, dtype=float),
        np.full(count, fixed),
        np.array(variable, dtype=float),
        np.full(count, patience),
    )


def unit(worker_id: str, second: float) -> float:
    return 1.0


def test_the_simulator_reproduces_the_erlang_c_waiting_time_of_a_shared_queue() -> None:
    for servers, load in ((2, 1.5), (4, 3.0)):
        results = [simulate_stationary(servers, load, 60000, seed, "shared") for seed in range(4)]
        wait = float(np.mean([r[0] for r in results]))
        tail = float(np.mean([r[1] for r in results]))
        assert wait == pytest.approx(mean_wait(servers, load, 2.0), rel=0.06)
        assert tail == pytest.approx(wait_exceeds(servers, load, 2.0, 3.0), abs=0.015)


def test_one_line_per_register_waits_longer_than_a_shared_queue() -> None:
    table = queue_validation(cases=((4, 3.0),), customers=30000, seeds=2)
    shared = float(table[table["discipline"] == "shared"]["simulated_wait_min"].iloc[0])
    lines = float(table[table["discipline"] == "shortest_line"]["simulated_wait_min"].iloc[0])
    assert lines > 1.15 * shared


def test_every_customer_is_either_served_or_leaves_and_waits_are_not_negative() -> None:
    rng = make_rng(3, "queue-conservation")
    stream = stationary_customers(rng, 2.7 / 120.0, 4000, 120.0, patience_seconds=300.0)
    horizon = float(stream.arrival[-1]) + 1.0e5
    intervals = [StaffInterval(r, 0.0, horizon, f"W{r}") for r in (1, 2, 3)]
    outcome = simulate_day(stream, intervals, unit, QueueSettings())
    assert np.all(outcome.served ^ outcome.abandoned)
    assert outcome.abandoned.any() and outcome.served.sum() > 3000
    assert np.all(outcome.wait >= 0.0)
    assert np.all(outcome.service[outcome.served] > 0.0)
    assert np.all(outcome.wait[outcome.abandoned] <= 300.0 + 1e-9)
    assert outcome.busy_seconds == pytest.approx(outcome.service[outcome.served].sum())
    assert 0.5 < outcome.busy_seconds / outcome.open_seconds <= 1.0


def test_a_single_server_serves_in_arrival_order() -> None:
    stream = customers([0.0, 1.0, 2.0, 3.0], [50.0, 50.0, 50.0, 50.0])
    outcome = simulate_day(stream, [StaffInterval(1, 0.0, 1000.0, "A")], unit, QueueSettings())
    starts = outcome.arrival + outcome.wait
    assert list(starts) == pytest.approx([0.0, 50.0, 100.0, 150.0])


def test_a_customer_who_loses_patience_leaves_after_exactly_that_long() -> None:
    stream = customers([0.0, 1.0], [1000.0, 10.0], patience=60.0)
    outcome = simulate_day(stream, [StaffInterval(1, 0.0, 5000.0, "A")], unit, QueueSettings())
    assert not outcome.abandoned[0]
    assert outcome.abandoned[1]
    assert outcome.wait[1] == pytest.approx(60.0)


def test_the_queue_of_a_register_that_closes_moves_to_the_open_register() -> None:
    stream = customers([0.0, 1.0, 2.0], [1000.0, 1000.0, 10.0])
    intervals = [StaffInterval(1, 0.0, 50.0, "A"), StaffInterval(2, 0.0, 10000.0, "B")]
    outcome = simulate_day(stream, intervals, unit, QueueSettings(close_second=1.0e9))
    assert outcome.register[0] == 1 and outcome.register[1] == 2
    assert outcome.register[2] == 2
    assert outcome.served.all()


def test_customers_wait_for_a_register_to_open_when_none_is_open() -> None:
    stream = customers([5.0], [30.0])
    outcome = simulate_day(
        stream, [StaffInterval(1, 100.0, 1000.0, "A")], unit, QueueSettings(close_second=1.0e9)
    )
    assert outcome.served[0]
    assert outcome.wait[0] == pytest.approx(95.0)


def test_the_speed_of_the_worker_at_the_register_scales_the_work_part_only() -> None:
    stream = customers([0.0, 300.0], [100.0, 100.0], fixed=20.0)
    intervals = [StaffInterval(1, 0.0, 200.0, "FAST"), StaffInterval(1, 200.0, 1000.0, "SLOW")]
    speeds = {"FAST": 0.5, "SLOW": 2.0}
    outcome = simulate_day(stream, intervals, lambda w, t: speeds[w], QueueSettings())
    assert list(outcome.service) == pytest.approx([70.0, 220.0])


def test_a_worker_change_at_the_same_register_does_not_close_it() -> None:
    stream = customers([95.0, 99.0], [30.0, 30.0])
    intervals = [StaffInterval(1, 0.0, 100.0, "A"), StaffInterval(1, 100.0, 1000.0, "B")]
    outcome = simulate_day(stream, intervals, unit, QueueSettings(close_second=1.0e9))
    assert outcome.served.all()
    assert outcome.wait[1] == pytest.approx(26.0)


def test_the_position_penalty_sends_customers_to_the_registers_near_the_entrance() -> None:
    stream = customers([0.0, 1.0], [500.0, 500.0])
    intervals = [StaffInterval(1, 0.0, 10000.0, "A"), StaffInterval(4, 0.0, 10000.0, "B")]
    positions = ((1, 0), (4, 1))
    plain = simulate_day(stream, intervals, unit, QueueSettings(register_positions=positions))
    biased = simulate_day(
        stream, intervals, unit, QueueSettings(position_penalty=5.0, register_positions=positions)
    )
    assert list(plain.register) == [1, 4]
    assert list(biased.register) == [1, 1]
    assert biased.wait[1] > plain.wait[1]


def test_the_shared_queue_gives_the_next_customer_to_whichever_register_frees_first() -> None:
    stream = customers([0.0, 1.0, 2.0], [100.0, 500.0, 10.0])
    intervals = [StaffInterval(1, 0.0, 10000.0, "A"), StaffInterval(2, 0.0, 10000.0, "B")]
    outcome = simulate_day(stream, intervals, unit, QueueSettings(discipline="shared"))
    assert outcome.register[2] == 1
    assert outcome.wait[2] == pytest.approx(98.0)


def test_the_queue_statistics_integrate_over_time() -> None:
    stream = customers([0.0, 10.0], [100.0, 10.0])
    outcome = simulate_day(
        stream, [StaffInterval(1, 0.0, 1000.0, "A")], unit, QueueSettings(close_second=1.0e9)
    )
    assert outcome.queue_area == pytest.approx(90.0)
    assert outcome.over_limit_seconds == 0.0


def test_the_time_over_the_queue_limit_is_measured() -> None:
    count = 8
    stream = customers(np.arange(count) * 1.0, [100.0] * count)
    outcome = simulate_day(
        stream,
        [StaffInterval(1, 0.0, 10000.0, "A")],
        unit,
        QueueSettings(queue_limit=5, close_second=1.0e9),
    )
    assert outcome.over_limit_seconds > 100.0


def test_the_simulation_is_deterministic() -> None:
    rng = make_rng(9, "queue-deterministic")
    stream = stationary_customers(rng, 2.0 / 120.0, 800, 120.0, patience_seconds=400.0)
    intervals = [StaffInterval(r, 0.0, 1.0e6, f"W{r}") for r in (1, 2, 3)]
    first = simulate_day(stream, intervals, unit, QueueSettings(position_penalty=0.2))
    second = simulate_day(stream, intervals, unit, QueueSettings(position_penalty=0.2))
    assert np.array_equal(first.wait, second.wait)
    assert np.array_equal(first.register, second.register)


def test_arrivals_follow_the_forecast_slot_means_and_stay_inside_their_slot() -> None:
    rng = make_rng(1, "arrival-test")
    starts = list(range(510, 600, 30))
    means = [20.0, 40.0, 30.0]
    totals = np.zeros(3)
    for _ in range(600):
        times = draw_arrivals(rng, starts, means, 30, 60.0)
        assert np.all(np.diff(times) >= 0.0)
        for index, start in enumerate(starts):
            inside = (times >= start * 60.0) & (times < (start + 30) * 60.0)
            totals[index] += inside.sum()
    assert totals / 600.0 == pytest.approx(means, rel=0.03)


def test_arrival_counts_are_over_dispersed_relative_to_poisson() -> None:
    rng = make_rng(2, "dispersion-test")
    counts = [len(draw_arrivals(rng, [510], [50.0], 30, 20.0)) for _ in range(3000)]
    assert np.var(counts) > 1.5 * np.mean(counts)
    assert np.mean(counts) == pytest.approx(50.0, rel=0.03)


def test_service_and_patience_draws_have_the_configured_means() -> None:
    rng = make_rng(4, "service-test")
    service = ServiceConfig()
    fixed, variable = draw_service(rng, service, 1.0, 40000)
    assert fixed[0] == service.overhead_seconds
    assert variable.mean() == pytest.approx(service.work_seconds, rel=0.02)
    exponential = ServiceConfig(distribution="exponential")
    _, total = draw_service(rng, exponential, 1.0, 40000)
    assert total.mean() == pytest.approx(120.0, rel=0.02)
    assert draw_patience(rng, service, 40000).mean() == pytest.approx(
        service.patience_minutes * 60.0, rel=0.02
    )
    assert np.isinf(draw_patience(rng, ServiceConfig(patience_minutes=0.0), 3)).all()
    assert lognormal_unit(rng, 0.6, 40000).mean() == pytest.approx(1.0, rel=0.02)
    _, christmas = draw_service(rng, service, 1.15, 40000)
    assert christmas.mean() == pytest.approx(1.15 * service.work_seconds, rel=0.02)


def test_a_day_of_customers_is_seeded() -> None:
    config = ServiceConfig()
    first = draw_day_customers(make_rng(5, "day"), [510, 540], [30.0, 30.0], 30, 60.0, config)
    again = draw_day_customers(make_rng(5, "day"), [510, 540], [30.0, 30.0], 30, 60.0, config)
    assert np.array_equal(first.arrival, again.arrival)
    assert np.array_equal(first.variable, again.variable)
    assert len(first) == len(first.arrival) > 0
