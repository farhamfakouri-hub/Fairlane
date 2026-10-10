"""End-to-end smoke check of the planner: Erlang C, queue simulator, roster, day and API."""

from __future__ import annotations

from dataclasses import dataclass

from ree.api.probe import get_json
from ree.planner.api.planner_app import create_planner_app
from ree.planner.api.service import PlannerService
from ree.planner.eval.validation import simulate_stationary
from ree.planner.staffing.erlang import erlang_c, mean_wait, wait_exceeds

SMOKE_STORE = "small"
SMOKE_SEED = 3
ERLANG_TOLERANCE = 0.06
QUEUE_CHECK_CUSTOMERS = 60000


@dataclass(frozen=True)
class Check:
    """Result of one smoke check."""

    name: str
    ok: bool
    detail: str


def erlang_check() -> Check:
    """Check the Erlang C formulas against a worked example."""
    probability = erlang_c(3, 2.0)
    wait = mean_wait(3, 2.0, 2.0)
    tail = wait_exceeds(3, 2.0, 2.0, 3.0)
    ok = abs(probability - 4 / 9) < 1e-9 and abs(wait - 0.8889) < 1e-3 and abs(tail - 0.0992) < 1e-3
    return Check(
        "Erlang C worked example",
        ok,
        f"C(3, 2) = {probability:.4f}, mean wait {wait:.3f} min, P(W > 3 min) = {tail:.4f}",
    )


def queue_check() -> Check:
    """Check that the queue simulator reproduces the Erlang C waiting time."""
    runs = [
        simulate_stationary(4, 3.0, QUEUE_CHECK_CUSTOMERS, seed, "shared")[0] for seed in range(4)
    ]
    simulated = sum(runs) / len(runs)
    expected = mean_wait(4, 3.0, 2.0)
    gap = simulated / expected - 1.0
    return Check(
        "queue simulator against Erlang C",
        abs(gap) < ERLANG_TOLERANCE,
        f"simulated {simulated:.3f} min, Erlang C {expected:.3f} min ({gap:+.1%})",
    )


def service_checks() -> list[Check]:
    """Solve a roster, plan a day and query the API."""
    service = PlannerService()
    version = service.generate(SMOKE_STORE, SMOKE_SEED, 1)
    report = version.solution.report
    errors = len(report.errors) if report else -1
    checks = [
        Check(
            "roster",
            errors == 0,
            f"{len(version.solution.plan.assignments)} shifts, {version.solution.backend} "
            f"{version.solution.status}, {errors} checker errors",
        )
    ]
    day = min(a.day for a in version.solution.plan.assignments if a.day.weekday() == 1)
    daily = service.daily(day)
    checks.append(
        Check(
            "daily plan",
            not daily.report.errors,
            f"{day.isoformat()}: {len(daily.plan.breaks)} breaks, "
            f"{daily.metrics.relief_handovers} relief handovers, "
            f"{len(daily.report.errors)} checker errors",
        )
    )
    app = create_planner_app(service)
    status, health = get_json(app, "/health")
    checks.append(
        Check(
            "planner API",
            status == 200 and bool(health["has_roster"]),
            f"/health answers {status} with {health['shifts']} shifts",
        )
    )
    status, body = get_json(app, f"/daily/{day.isoformat()}")
    checks.append(
        Check(
            "daily plan over the API", status == 200 and bool(body["segments"]), f"status {status}"
        )
    )
    return checks


def run_checks() -> list[Check]:
    """Run every smoke check."""
    return [erlang_check(), queue_check(), *service_checks()]
