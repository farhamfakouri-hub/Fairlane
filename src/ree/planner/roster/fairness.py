"""Fairness measures of a roster: burden rates, tier gaps, score correlations and stability."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy import stats

from ree.planner.config import PlannerConfig, clock_to_minutes
from ree.planner.domain.plan import RosterPlan
from ree.planner.domain.shifts import SATURDAY, SUNDAY, ShiftPattern, pattern_burden

BURDEN_SCALE = 10_000


def day_burden(pattern: ShiftPattern, day: date, config: PlannerConfig) -> int:
    """Burden points of a pattern on a day."""
    return pattern_burden(pattern, day.weekday(), config.burden, config.opening)


def worker_burden(plan: RosterPlan, worker_id: str, config: PlannerConfig) -> int:
    """Total burden points of a worker over the roster."""
    return sum(
        day_burden(a.pattern, a.day, config) for a in plan.assignments if a.worker_id == worker_id
    )


def burden_rates(
    plan: RosterPlan,
    config: PlannerConfig,
    history: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Burden per contract hour and week of each worker, with optional history."""
    history = history or {}
    weeks = plan.weeks + (config.rules.history_weeks if history else 0)
    rates: dict[str, float] = {}
    for worker_id, worker in plan.workers.items():
        if worker.contract_hours <= 0:
            continue
        total = worker_burden(plan, worker_id, config) + history.get(worker_id, 0.0)
        rates[worker_id] = total / (worker.contract_hours * weeks)
    return rates


def burden_spread(rates: Mapping[str, float]) -> float:
    """Highest burden rate minus the lowest."""
    if not rates:
        return 0.0
    return float(max(rates.values()) - min(rates.values()))


def tier_groups(tiers: Mapping[str, str]) -> dict[str, list[str]]:
    """Workers of each tier."""
    groups: dict[str, list[str]] = defaultdict(list)
    for worker_id, tier in sorted(tiers.items()):
        groups[tier].append(worker_id)
    return dict(groups)


def tier_gaps(rates: Mapping[str, float], tiers: Mapping[str, str]) -> dict[str, float]:
    """Mean burden rate of each tier relative to the overall mean, minus one."""
    if not rates:
        return {}
    overall = float(np.mean(list(rates.values())))
    gaps: dict[str, float] = {}
    for tier, members in tier_groups(tiers).items():
        values = [rates[w] for w in members if w in rates]
        if values and overall > 0.0:
            gaps[tier] = float(np.mean(values)) / overall - 1.0
    return gaps


def max_tier_gap(rates: Mapping[str, float], tiers: Mapping[str, str]) -> float:
    """Largest absolute tier gap."""
    gaps = tier_gaps(rates, tiers)
    return max((abs(value) for value in gaps.values()), default=0.0)


def shift_frequencies(plan: RosterPlan, config: PlannerConfig) -> dict[str, dict[str, float]]:
    """Share of weekend, late and split shifts of each worker."""
    close = clock_to_minutes(config.opening.monday_to_saturday[1])
    counts: dict[str, dict[str, float]] = {
        worker_id: {"shifts": 0.0, "weekend": 0.0, "late": 0.0, "split": 0.0}
        for worker_id in plan.workers
    }
    for assignment in plan.assignments:
        if assignment.worker_id not in counts:
            continue
        row = counts[assignment.worker_id]
        row["shifts"] += 1.0
        if assignment.day.weekday() in (SATURDAY, SUNDAY):
            row["weekend"] += 1.0
        if assignment.day.weekday() != SUNDAY and assignment.pattern.register_end >= close:
            row["late"] += 1.0
        if len(assignment.pattern.blocks) > 1:
            row["split"] += 1.0
    for row in counts.values():
        shifts = row["shifts"]
        for key in ("weekend", "late", "split"):
            row[key] = row[key] / shifts if shifts else 0.0
    return counts


def score_correlations(
    plan: RosterPlan, z: Mapping[str, float], config: PlannerConfig
) -> dict[str, float]:
    """Rank correlation of scores with the shift shares and with the burden."""
    frequencies = shift_frequencies(plan, config)
    workers = [w for w in sorted(plan.workers) if w in z and frequencies[w]["shifts"] > 0]
    result: dict[str, float] = {}
    if len(workers) < 3:
        return {key: 0.0 for key in ("weekend", "late", "split", "burden")}
    scores = [z[w] for w in workers]
    for key in ("weekend", "late", "split"):
        values = [frequencies[w][key] for w in workers]
        result[key] = _correlation(scores, values)
    rates = burden_rates(plan, config)
    result["burden"] = _correlation(scores, [rates.get(w, 0.0) for w in workers])
    return result


def _correlation(first: list[float], second: list[float]) -> float:
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(stats.spearmanr(first, second).statistic)


def hours_deviation(plan: RosterPlan) -> dict[str, float]:
    """Largest weekly deviation of paid hours from the contract for each worker."""
    result: dict[str, float] = {}
    for worker_id, worker in plan.workers.items():
        paid = defaultdict(float)
        for assignment in plan.assignments:
            if assignment.worker_id == worker_id:
                paid[plan.week_of(assignment.day)] += assignment.pattern.paid_minutes / 60.0
        weekly = [paid.get(week, 0.0) for week in range(plan.weeks)]
        result[worker_id] = max((abs(h - worker.contract_hours) for h in weekly), default=0.0)
    return result


def signed_hours_deviation(plan: RosterPlan) -> dict[str, float]:
    """Average signed deviation of weekly paid hours from the contract for each worker."""
    result: dict[str, float] = {}
    for worker_id, worker in plan.workers.items():
        paid = defaultdict(float)
        for assignment in plan.assignments:
            if assignment.worker_id == worker_id:
                paid[plan.week_of(assignment.day)] += assignment.pattern.paid_minutes / 60.0
        weekly = [paid.get(week, 0.0) - worker.contract_hours for week in range(plan.weeks)]
        result[worker_id] = sum(weekly) / len(weekly) if weekly else 0.0
    return result


def hours_score_correlation(plan: RosterPlan, z: Mapping[str, float]) -> float:
    """Rank correlation of scores with the signed hours deviation; it should be near zero."""
    deviation = signed_hours_deviation(plan)
    workers = [w for w in sorted(deviation) if w in z]
    if len(workers) < 3:
        return 0.0
    return _correlation([z[w] for w in workers], [deviation[w] for w in workers])


def tier_hours_gap(plan: RosterPlan, tiers: Mapping[str, str], top: str, bottom: str) -> float:
    """Difference in mean signed hours deviation between two tiers."""
    deviation = signed_hours_deviation(plan)
    groups = tier_groups(tiers)
    high = [deviation[w] for w in groups.get(top, []) if w in deviation]
    low = [deviation[w] for w in groups.get(bottom, []) if w in deviation]
    if not high or not low:
        return 0.0
    return float(np.mean(high)) - float(np.mean(low))


def stability(plan: RosterPlan, previous: Mapping[tuple[str, date], str]) -> float:
    """Share of assignments that are the same as in a previous roster."""
    if not previous:
        return 1.0
    current = {(a.worker_id, a.day): a.pattern.pattern_id for a in plan.assignments}
    keys = set(previous) | set(current)
    same = sum(1 for key in keys if previous.get(key) == current.get(key))
    return same / len(keys) if keys else 1.0


@dataclass(frozen=True)
class FairnessMetrics:
    """The fairness measures of one roster."""

    burden_spread: float
    max_tier_gap: float
    tier_gaps: dict[str, float]
    correlations: dict[str, float]
    max_hours_deviation: float
    stability: float


def fairness_metrics(
    plan: RosterPlan,
    config: PlannerConfig,
    tiers: Mapping[str, str],
    z: Mapping[str, float],
    history: Mapping[str, float] | None = None,
    previous: Mapping[tuple[str, date], str] | None = None,
) -> FairnessMetrics:
    """Compute the fairness measures of a roster."""
    rates = burden_rates(plan, config, history)
    return FairnessMetrics(
        burden_spread=burden_spread(rates),
        max_tier_gap=max_tier_gap(rates, tiers),
        tier_gaps=tier_gaps(rates, tiers),
        correlations=score_correlations(plan, z, config),
        max_hours_deviation=max(hours_deviation(plan).values(), default=0.0),
        stability=stability(plan, previous or {}),
    )
