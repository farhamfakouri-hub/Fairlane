from __future__ import annotations

import dataclasses
import time
from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ortools.sat.python import cp_model

from ree.planner.checker.checks import check_roster
from ree.planner.config import PlannerConfig
from ree.planner.domain.patterns import PatternSpec, solver_pattern_set
from ree.planner.roster.explain import explain_roster, rush_minutes
from ree.planner.roster.fairness import burden_rates, hours_deviation
from ree.planner.roster.model import (
    COVERAGE_TERMS,
    RosterProblem,
    ScoreInputs,
    build_model,
    neutral_scores,
    plan_from_start,
    roster_scores,
    term_cost,
    term_values,
)
from ree.planner.roster.refine import refine_scores
from ree.planner.roster.solver import plan_roster, solve_coverage
from ree.planner.sim.roster_gate import scenario_inputs, solve_scenario
from ree.planner.sim.scenarios import make_workforce, tiny_store_config


@pytest.fixture(scope="module")
def baseline():
    return solve_scenario(3)


def test_the_solver_pool_leaves_out_patterns_a_blocked_worker_cannot_take() -> None:
    config, workers, requirement, scores = scenario_inputs(3)
    problem = RosterProblem(
        date.fromisoformat(config.start_date),
        1,
        workers,
        solver_pattern_set(config, PatternSpec()),
        requirement,
        scores,
        config,
    )
    built = build_model(problem)
    blocked = [w for w in workers if w.blocked]
    assert blocked
    worker = blocked[0]
    index = workers.index(worker)
    for day in worker.blocked:
        di = built.days.index(day)
        for wi, d, pi in built.x:
            if wi == index and d == di:
                pattern = built.patterns_by_day[di][pi]
                assert not any(
                    worker.is_blocked(day, b.paid_start, b.paid_end) for b in pattern.blocks
                )
    assert len(built.x) > 1000


def test_a_solved_roster_has_exact_contract_hours_and_no_checker_errors(baseline) -> None:
    report = baseline.report
    assert report is not None and report.ok, report.summary()
    assert baseline.status in ("OPTIMAL", "FEASIBLE")
    assert max(hours_deviation(baseline.plan).values()) <= 1.0
    assert not baseline.relaxed
    assert "REGISTERS_1_4_UNSTAFFED" not in report.codes()


def test_the_solver_is_deterministic_with_a_fixed_seed(baseline) -> None:
    again = solve_scenario(3)
    first = sorted((a.worker_id, a.day, a.pattern.pattern_id) for a in baseline.plan.assignments)
    second = sorted((a.worker_id, a.day, a.pattern.pattern_id) for a in again.plan.assignments)
    assert first == second
    assert again.objective == baseline.objective


def test_the_fast_mode_respects_its_time_limit() -> None:
    config = tiny_store_config(**{"solver.mode": "fast", "solver.time_limit_seconds": 2.0})
    started = time.perf_counter()
    solution = solve_scenario(4, config)
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0
    assert solution.report is not None


def test_the_posts_only_mode_gives_the_roster_the_same_inputs_as_neutral_scores() -> None:
    config = tiny_store_config(**{"solver.backend": "heuristic"})
    off = config.with_overrides({"objective.scores_in_roster": False})
    _, workers, requirement, scores = scenario_inputs(4, config)
    assert any(abs(z) > 0.0 for z in scores.z.values())
    ids = [w.worker_id for w in workers]
    ignoring = plan_roster(workers, requirement, off, scores, weeks=1)
    neutral = plan_roster(workers, requirement, config, neutral_scores(ids), weeks=1)
    assert ignoring.plan.assignments == neutral.plan.assignments
    assert roster_scores(scores, config) is scores
    flat = roster_scores(scores, off)
    assert set(flat.z.values()) == {0.0} and flat.exploration == frozenset()
    assert set(flat.z) == set(scores.z)
    assert len(set(flat.tiers.values())) == 1


def test_the_heuristic_fallback_builds_legal_rosters() -> None:
    config = tiny_store_config(**{"solver.backend": "heuristic"})
    for seed in range(5):
        solution = solve_scenario(seed, config)
        assert solution.backend == "heuristic"
        assert solution.report is not None
        legal = solution.report.codes("error") - {"REGISTERS_1_4_UNSTAFFED"}
        assert legal == set()


def test_an_impossible_contract_is_reported_not_hidden() -> None:
    config, workers, requirement, scores = scenario_inputs(5)
    first = workers[0]
    start = date.fromisoformat(config.start_date)
    blocked_all = {
        start + timedelta(days=offset): ((0, 1440),)
        for offset in range(7)
        if (start + timedelta(days=offset)).weekday() != 6
    }
    squeezed = [dataclasses.replace(first, contract_hours=21.0, blocked=blocked_all), *workers[1:]]
    solution = plan_roster(squeezed, requirement, config, scores, weeks=1)
    assert solution.report is not None
    assert "WEEKLY_HOURS_CONTRACT" in solution.report.codes("error")
    assert solution.backend == "heuristic" or solution.relaxed
    assert solution.notes


def test_the_tier_fairness_guard_keeps_every_score_tier_near_the_average(baseline) -> None:
    config, _, _, _ = scenario_inputs(3)
    metrics = baseline.metrics
    assert metrics is not None
    assert metrics.max_tier_gap <= config.objective.tier_epsilon + 0.1
    assert metrics.max_hours_deviation <= 1.0


def test_new_workers_get_peak_slots_to_learn_from() -> None:
    config, workers, requirement, scores = scenario_inputs(7)
    solution = plan_roster(workers, requirement, config, scores, weeks=1)
    quota = config.objective.exploration_hours_per_week * 60
    gaps = []
    for worker_id in sorted(scores.exploration):
        minutes = sum(
            rush_minutes(a.pattern, a.day, requirement)
            for a in solution.plan.assignments
            if a.worker_id == worker_id
        )
        gaps.append(max(0.0, quota - minutes))
    assert gaps
    assert sum(gaps) / len(gaps) <= quota / 2


def test_the_workforce_generator_is_seeded_and_respects_the_contract_mix() -> None:
    config = PlannerConfig()
    first = make_workforce(config, seed=1)
    again = make_workforce(config, seed=1)
    other = make_workforce(config, seed=2)
    assert first == again and first != other
    assert len(first) == 40
    hours = {w.contract_hours for w in first}
    assert {15.0, 21.0} & hours and {35.0, 30.0} & hours
    only = [w for w in first if w.self_checkout_only]
    assert 2 <= len(only) <= 4
    assert all(w.cleared_self_checkout for w in only)
    assert tiny_store_config().workforce.n_workers == 10


def test_the_checker_agrees_with_the_solver_report(baseline) -> None:
    config, _, requirement, _ = scenario_inputs(3)
    again = check_roster(baseline.plan, config.rules, requirement)
    assert again.counts() == baseline.report.counts()


def _problem_for(seed: int, last_roster=None) -> RosterProblem:
    config, workers, requirement, scores = scenario_inputs(seed)
    return RosterProblem(
        date.fromisoformat(config.start_date),
        1,
        workers,
        solver_pattern_set(config, PatternSpec()),
        requirement,
        scores,
        config,
        last_roster=dict(last_roster or {}),
    )


def test_the_objective_is_the_sum_of_its_named_terms(baseline) -> None:
    assert baseline.objective is not None
    assert sum(baseline.terms.values()) == pytest.approx(baseline.objective)
    assert baseline.terms["floor"] == 0.0


def test_the_final_roster_has_no_score_improving_legal_swap_left(baseline) -> None:
    refined, swaps = refine_scores(_problem_for(3), baseline.plan)
    assert swaps == []
    assert refined.assignments == baseline.plan.assignments


def test_every_assignment_has_a_plain_reason_and_the_contract_line(baseline) -> None:
    config, _, requirement, scores = scenario_inputs(3)
    explanations = explain_roster(baseline.plan, requirement, scores, config)
    assert len(explanations) == len(baseline.plan.assignments)
    for (worker_id, _), reasons in explanations.items():
        assert reasons and reasons[-1].code == "CONTRACT_HOURS"
        codes = {reason.code for reason in reasons}
        if "RUSH_EXPLORATION" in codes:
            assert worker_id in scores.exploration
        if "RUSH_TOP_SCORE" in codes:
            assert scores.z[worker_id] > 0.0
    assert any(
        reason.code in ("RUSH_TOP_SCORE", "RUSH_EXPLORATION", "COVERAGE_NEED")
        for reasons in explanations.values()
        for reason in reasons
    )


def test_the_stability_term_charges_every_assignment_that_differs_from_last_time(baseline) -> None:
    plan = baseline.plan
    previous = {(a.worker_id, a.day): a.pattern.pattern_id for a in plan.assignments}
    altered = dict(previous)
    changed_keys = sorted(altered)[:3]
    for key in changed_keys:
        altered[key] = "no-such-pattern"
    problem = _problem_for(3, altered)
    built = build_model(problem)
    ids = [w.worker_id for w in problem.workers]
    chosen = {
        (ids.index(a.worker_id), built.days.index(a.day)): a.pattern for a in plan.assignments
    }
    for (wi, di, pi), variable in built.x.items():
        taken = chosen.get((wi, di))
        built.model.add(
            variable == int(taken is not None and built.patterns_by_day[di][pi] == taken)
        )
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    status = solver.solve(built.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), solver.status_name(status)
    expected = len(changed_keys) * problem.config.objective.stability_weight
    assert term_values(built, solver)["stability"] == expected


def _swap_scores(scores: ScoreInputs, first: str, second: str) -> ScoreInputs:
    def exchanged(mapping: dict) -> dict:
        result = dict(mapping)
        result[first], result[second] = mapping[second], mapping[first]
        return result

    exploration = set(scores.exploration)
    if (first in exploration) != (second in exploration):
        exploration ^= {first, second}
    return ScoreInputs(exchanged(scores.z), exchanged(scores.tiers), frozenset(exploration))


def _widest_pair_with_equal_contracts(workers, scores: ScoreInputs) -> tuple[str, str]:
    rated = [w for w in workers if w.worker_id not in scores.exploration]
    pairs = [
        (abs(scores.z[a.worker_id] - scores.z[b.worker_id]), a.worker_id, b.worker_id)
        for i, a in enumerate(rated)
        for b in rated[i + 1 :]
        if a.contract_hours == b.contract_hours
    ]
    _, first, second = max(pairs)
    return first, second


def _paid_hours(plan, worker_id: str) -> float:
    return sum(a.pattern.paid_minutes for a in plan.assignments if a.worker_id == worker_id) / 60.0


@pytest.mark.parametrize("seed", [3, 7])
def test_swapping_the_scores_of_two_workers_keeps_their_hours_and_burden_within_tolerance(
    seed: int, baseline
) -> None:
    config, workers, requirement, scores = scenario_inputs(seed)
    before = baseline if seed == 3 else plan_roster(workers, requirement, config, scores, weeks=1)
    first, second = _widest_pair_with_equal_contracts(workers, scores)
    after = plan_roster(workers, requirement, config, _swap_scores(scores, first, second), weeks=1)
    tolerance = config.rules.hours_tolerance_hours
    for solution in (before, after):
        assert solution.report is not None and solution.report.ok
        assert solution.metrics is not None
        assert solution.metrics.max_hours_deviation <= tolerance
        assert solution.metrics.max_tier_gap <= config.objective.tier_epsilon
    rates_before = burden_rates(before.plan, config)
    rates_after = burden_rates(after.plan, config)
    allowed = config.objective.tier_epsilon * (sum(rates_before.values()) / len(rates_before))
    contract = {w.worker_id: w.contract_hours for w in workers}
    for worker_id in (first, second):
        assert abs(_paid_hours(after.plan, worker_id) - contract[worker_id]) <= tolerance
        assert abs(rates_after[worker_id] - rates_before[worker_id]) <= allowed


def _coverage_first_inputs():
    config = tiny_store_config(**{"solver.coverage_first": True, "solver.deterministic_time": 6.0})
    return scenario_inputs(3, config)


@pytest.fixture(scope="module")
def coverage_first():
    config, workers, requirement, scores = _coverage_first_inputs()
    start = solve_coverage(workers, requirement, config, weeks=1)
    neutral = neutral_scores([w.worker_id for w in workers])
    return {
        "config": config,
        "requirement": requirement,
        "scores": scores,
        "start": start,
        "with_scores": plan_roster(
            workers, requirement, config, scores, weeks=1, coverage_start=start
        ),
        "neutral": plan_roster(
            workers, requirement, config, neutral, weeks=1, coverage_start=start
        ),
    }


def test_coverage_first_never_gives_up_the_coverage_of_its_first_stage(coverage_first) -> None:
    start = coverage_first["start"]
    assert start is not None
    for name in ("with_scores", "neutral"):
        solution = coverage_first[name]
        assert solution.report is not None and solution.report.ok
        assert term_cost(solution.terms, COVERAGE_TERMS) <= start.cost
        assert any("coverage first" in note for note in solution.notes)


def test_coverage_first_gives_every_score_vector_the_same_coverage_cost(coverage_first) -> None:
    first = coverage_first["with_scores"]
    second = coverage_first["neutral"]
    assert first.report.shortfall_minutes() == second.report.shortfall_minutes()


def _weekly_paid_minutes(plan) -> dict[tuple[str, int], int]:
    totals: dict[tuple[str, int], int] = {}
    for assignment in plan.assignments:
        key = (assignment.worker_id, plan.week_of(assignment.day))
        totals[key] = totals.get(key, 0) + assignment.pattern.paid_minutes
    return totals


def test_scores_never_change_anyones_weekly_paid_minutes_in_coverage_first_mode(
    coverage_first,
) -> None:
    scored = _weekly_paid_minutes(coverage_first["with_scores"].plan)
    neutral = _weekly_paid_minutes(coverage_first["neutral"].plan)
    assert scored == neutral
    assert scored


def test_swapping_the_scores_of_two_workers_changes_nobodys_weekly_paid_minutes(
    coverage_first,
) -> None:
    config, workers, requirement, scores = _coverage_first_inputs()
    first, second = _widest_pair_with_equal_contracts(workers, scores)
    swapped = plan_roster(
        workers,
        requirement,
        config,
        _swap_scores(scores, first, second),
        weeks=1,
        coverage_start=coverage_first["start"],
    )
    assert swapped.report is not None and swapped.report.ok
    assert _weekly_paid_minutes(swapped.plan) == _weekly_paid_minutes(
        coverage_first["with_scores"].plan
    )


def test_coverage_first_is_deterministic(coverage_first) -> None:
    config, workers, requirement, scores = _coverage_first_inputs()
    again = plan_roster(
        workers, requirement, config, scores, weeks=1, coverage_start=coverage_first["start"]
    )
    first = sorted(
        (a.worker_id, a.day, a.pattern.pattern_id)
        for a in coverage_first["with_scores"].plan.assignments
    )
    second = sorted((a.worker_id, a.day, a.pattern.pattern_id) for a in again.plan.assignments)
    assert first == second


def test_the_coverage_start_can_be_turned_back_into_the_same_plan(coverage_first) -> None:
    config, workers, requirement, _ = _coverage_first_inputs()
    start = coverage_first["start"]
    problem = RosterProblem(
        date.fromisoformat(config.start_date),
        1,
        workers,
        solver_pattern_set(config, PatternSpec()),
        requirement,
        neutral_scores([w.worker_id for w in workers]),
        config,
    )
    plan = plan_from_start(problem, start)
    assert {(a.worker_id, a.day, a.pattern.pattern_id) for a in plan.assignments} == set(
        start.assignments
    )


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_the_heuristic_roster_of_any_seed_passes_every_legal_rule_of_the_checker(seed: int) -> None:
    config = tiny_store_config(**{"solver.backend": "heuristic"})
    solution = solve_scenario(seed, config)
    assert solution.report is not None
    assert solution.report.codes("error") <= {"REGISTERS_1_4_UNSTAFFED"}
    assert max(hours_deviation(solution.plan).values()) <= 1.0
