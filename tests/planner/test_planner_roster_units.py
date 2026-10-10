from __future__ import annotations

from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from planner_factories import (
    CONFIG,
    MONDAY,
    START,
    evening_pattern,
    morning_pattern,
    pattern,
    requirement_for,
    score_record,
    valid_roster,
    with_rush,
    worker,
)

from ree.api.provider import InMemoryScoreProvider
from ree.planner.checker.checks import check_worker_rules
from ree.planner.config import PlannerConfig, clock_to_minutes
from ree.planner.domain.demand_types import RequirementTable
from ree.planner.domain.patterns import PatternSpec, solver_pattern_set
from ree.planner.domain.plan import Assignment, RosterPlan
from ree.planner.roster.explain import explain_roster, rush_minutes, trade_offs
from ree.planner.roster.fairness import (
    burden_rates,
    burden_spread,
    day_burden,
    hours_deviation,
    hours_score_correlation,
    max_tier_gap,
    score_correlations,
    shift_frequencies,
    signed_hours_deviation,
    stability,
    tier_gaps,
    tier_hours_gap,
    worker_burden,
)
from ree.planner.roster.model import (
    RosterProblem,
    ScoreInputs,
    clipped_z,
    neutral_scores,
    rush_weights,
    score_inputs,
)
from ree.planner.roster.refine import refine_scores, repair_tiers

TUESDAY = START + timedelta(days=2)
FRIDAY = START + timedelta(days=5)
SATURDAY = START + timedelta(days=6)
EVENING_RUSH = (18 * 60, 20 * 60)


def test_scores_become_clipped_confidence_weighted_z_values() -> None:
    provider = InMemoryScoreProvider(
        [
            score_record("W001", 65.0, 0.8, "A"),
            score_record("W002", 95.0, 1.0, "A"),
            score_record("W003", 20.0, 0.5, "C"),
            score_record("W004", 70.0, 0.9, "A", provisional=True),
            score_record("W005", 50.0, 0.9, "unrated"),
        ]
    )
    ids = ["W001", "W002", "W003", "W004", "W005", "W006"]
    inputs = score_inputs(provider, ids, CONFIG)
    assert inputs.z["W001"] == pytest.approx(0.8)
    assert inputs.z["W002"] == pytest.approx(2.0)
    assert inputs.z["W003"] == pytest.approx(-1.0)
    assert inputs.tiers == {
        "W001": "A",
        "W002": "A",
        "W003": "C",
        "W004": "unrated",
        "W005": "unrated",
        "W006": "unrated",
    }
    assert inputs.exploration == frozenset({"W004", "W005", "W006"})
    assert all(inputs.z[w] == 0.0 for w in ("W004", "W005", "W006"))


def test_the_planner_reads_scores_only_through_the_provider_interface() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_score(self, worker_id: str):
            self.calls.append(worker_id)
            return None

        def get_ranking(self, *args, **kwargs):
            raise AssertionError("ranking is not needed to build the roster inputs")

        def list_scores(self):
            raise AssertionError("listing every score is not needed")

        def latest_as_of(self):
            return None

    recorder = Recorder()
    inputs = score_inputs(recorder, ["W001", "W002"], CONFIG)
    assert recorder.calls == ["W001", "W002"]
    assert inputs.exploration == frozenset({"W001", "W002"})


def test_neutral_scores_carry_no_information() -> None:
    inputs = neutral_scores(["a", "b"])
    assert inputs.z == {"a": 0.0, "b": 0.0}
    assert set(inputs.tiers.values()) == {"all"}
    assert inputs.exploration == frozenset()


def test_the_z_value_is_clipped_at_the_configured_limit() -> None:
    assert clipped_z(50.0, 2.0) == 0.0
    assert clipped_z(65.0, 2.0) == pytest.approx(1.0)
    assert clipped_z(100.0, 2.0) == 2.0
    assert clipped_z(0.0, 2.0) == -2.0


def test_rush_weights_run_from_zero_to_one_across_the_offered_load() -> None:
    calm = requirement_for([MONDAY])
    table = with_rush(calm, MONDAY, 630, 840, load=5.0)
    weights = rush_weights(table)
    assert weights[(MONDAY, 630)] == 1.0
    assert weights[(MONDAY, 510)] == 0.0
    assert max(weights.values()) == 1.0 and min(weights.values()) == 0.0
    assert set(rush_weights(calm).values()) == {0.0}
    assert rush_weights(RequirementTable(slot_minutes=30)) == {}


def test_the_burden_of_a_shift_adds_up_its_unpopular_features() -> None:
    assert day_burden(pattern((510, 750)), START, CONFIG) == 4
    assert day_burden(evening_pattern(), FRIDAY, CONFIG) == 3
    assert day_burden(evening_pattern(), SATURDAY, CONFIG) == 4
    assert day_burden(morning_pattern(), MONDAY, CONFIG) == 1
    assert day_burden(pattern((510, 750), (990, 1290)), TUESDAY, CONFIG) == 5
    assert day_burden(pattern((600, 1080)), TUESDAY, CONFIG) == 0


def _burden_plan() -> RosterPlan:
    workers = {"A": worker("A"), "B": worker("B")}
    assignments = [
        Assignment("A", MONDAY, morning_pattern()),
        Assignment("A", TUESDAY, evening_pattern()),
        Assignment("B", SATURDAY, evening_pattern()),
    ]
    return RosterPlan(START, 1, workers, assignments)


def test_burden_totals_and_rates_have_known_values() -> None:
    plan = _burden_plan()
    assert worker_burden(plan, "A", CONFIG) == 3
    assert worker_burden(plan, "B", CONFIG) == 4
    rates = burden_rates(plan, CONFIG)
    assert rates["A"] == pytest.approx(3 / 29.0)
    assert rates["B"] == pytest.approx(4 / 29.0)
    with_history = burden_rates(plan, CONFIG, {"A": 8.0})
    weeks = 1 + CONFIG.rules.history_weeks
    assert with_history["A"] == pytest.approx(11 / (29.0 * weeks))
    assert with_history["B"] == pytest.approx(4 / (29.0 * weeks))


def test_spread_and_tier_gaps_have_known_values() -> None:
    assert burden_spread({"a": 0.5, "b": 1.25}) == pytest.approx(0.75)
    assert burden_spread({}) == 0.0
    rates = {"a": 1.0, "b": 2.0, "c": 3.0}
    tiers = {"a": "A", "b": "B", "c": "C"}
    gaps = tier_gaps(rates, tiers)
    assert gaps == {"A": pytest.approx(-0.5), "B": pytest.approx(0.0), "C": pytest.approx(0.5)}
    assert max_tier_gap(rates, tiers) == pytest.approx(0.5)
    assert tier_gaps({}, tiers) == {}
    assert max_tier_gap({"a": 1.0, "b": 1.0}, {"a": "A", "b": "B"}) == 0.0


def test_shift_frequencies_count_weekend_late_and_split_shifts() -> None:
    frequencies = shift_frequencies(_burden_plan(), CONFIG)
    assert frequencies["A"] == {"shifts": 2.0, "weekend": 0.0, "late": 0.5, "split": 0.0}
    assert frequencies["B"] == {"shifts": 1.0, "weekend": 1.0, "late": 1.0, "split": 0.0}


def test_the_score_correlation_is_one_when_burden_follows_the_score() -> None:
    workers = {name: worker(name) for name in ("W1", "W2", "W3")}
    assignments = [
        Assignment("W1", MONDAY, morning_pattern()),
        Assignment("W1", TUESDAY, morning_pattern()),
        Assignment("W2", MONDAY, morning_pattern()),
        Assignment("W2", SATURDAY, morning_pattern()),
        Assignment("W3", SATURDAY, morning_pattern()),
        Assignment("W3", START, morning_pattern()),
    ]
    plan = RosterPlan(START, 1, workers, assignments)
    z = {"W1": -1.0, "W2": 0.0, "W3": 1.0}
    correlations = score_correlations(plan, z, CONFIG)
    assert correlations["weekend"] == pytest.approx(1.0)
    assert correlations["burden"] == pytest.approx(1.0)
    assert correlations["late"] == 0.0 and correlations["split"] == 0.0
    inverse = score_correlations(plan, {"W1": 1.0, "W2": 0.0, "W3": -1.0}, CONFIG)
    assert inverse["weekend"] == pytest.approx(-1.0)
    few = score_correlations(plan, {"W1": 0.0, "W2": 1.0}, CONFIG)
    assert set(few.values()) == {0.0}


def test_hours_deviation_is_zero_for_a_full_contract_and_known_otherwise() -> None:
    full, _ = valid_roster(4)
    assert set(hours_deviation(full).values()) == {0.0}
    short, _ = valid_roster(3)
    assert all(value == pytest.approx(7.25) for value in hours_deviation(short).values())


def _contract_plan(contracts: dict[str, float]) -> RosterPlan:
    workers = {name: worker(name, hours) for name, hours in contracts.items()}
    days = [START + timedelta(days=1 + offset) for offset in range(4)]
    assignments = [Assignment(name, day, morning_pattern()) for name in contracts for day in days]
    return RosterPlan(START, 1, workers, assignments)


def test_the_signed_hours_deviation_is_paid_minus_contract_per_week() -> None:
    assert set(signed_hours_deviation(valid_roster(4)[0]).values()) == {0.0}
    assert all(
        v == pytest.approx(-7.25) for v in signed_hours_deviation(valid_roster(3)[0]).values()
    )
    assert all(
        v == pytest.approx(7.25) for v in signed_hours_deviation(valid_roster(5)[0]).values()
    )
    plan = _contract_plan({"A": 28.0, "B": 30.0})
    assert signed_hours_deviation(plan) == pytest.approx({"A": 1.0, "B": -1.0})


def test_the_score_correlates_with_the_hours_deviation_only_when_they_move_together() -> None:
    contracts = {"A": 28.0, "B": 28.0, "F": 29.0, "G": 29.0, "C": 30.0, "D": 30.0}
    plan = _contract_plan(contracts)
    aligned = {"A": 2.0, "B": 1.0, "F": 0.0, "G": -0.5, "C": -1.0, "D": -2.0}
    assert hours_score_correlation(plan, aligned) > 0.9
    assert hours_score_correlation(plan, {k: -v for k, v in aligned.items()}) < -0.9
    assert hours_score_correlation(plan, {k: 0.0 for k in aligned}) == 0.0
    assert hours_score_correlation(plan, {"A": 1.0, "B": 2.0}) == 0.0


def test_the_tier_hours_gap_is_the_difference_of_the_mean_deviations() -> None:
    plan = _contract_plan({"A": 28.0, "B": 28.0, "F": 29.0, "G": 29.0, "C": 30.0, "D": 30.0})
    tiers = {"A": "A", "B": "A", "F": "B", "G": "B", "C": "C", "D": "C"}
    assert tier_hours_gap(plan, tiers, "A", "C") == pytest.approx(2.0)
    assert tier_hours_gap(plan, tiers, "C", "A") == pytest.approx(-2.0)
    assert tier_hours_gap(plan, tiers, "A", "Z") == 0.0


def test_stability_counts_the_assignments_that_stay_the_same() -> None:
    plan, _ = valid_roster(2)
    previous = {(a.worker_id, a.day): a.pattern.pattern_id for a in plan.assignments}
    assert stability(plan, previous) == 1.0
    assert stability(plan, {}) == 1.0
    changed = dict(previous)
    first = plan.assignments[0]
    changed[(first.worker_id, first.day)] = "other"
    assert stability(plan, changed) == pytest.approx(11 / 12)
    changed[("Z", MONDAY)] = "other"
    assert stability(plan, changed) == pytest.approx(11 / 13)


def _explain_inputs() -> tuple[RosterPlan, RequirementTable, ScoreInputs]:
    names = ("A", "B", "C", "D")
    workers = {name: worker(name, 7.25) for name in names}
    assignments = [Assignment(name, TUESDAY, morning_pattern()) for name in names]
    plan = RosterPlan(START, 1, workers, assignments)
    requirement = with_rush(requirement_for([TUESDAY], headcount=4), TUESDAY, 630, 840)
    scores = ScoreInputs(
        z={"A": 1.5, "B": 0.5, "C": -1.0, "D": 0.0},
        tiers={"A": "A", "B": "B", "C": "C", "D": "unrated"},
        exploration=frozenset({"D"}),
    )
    return plan, requirement, scores


def _codes(explanations, name: str, day) -> list[str]:
    return [reason.code for reason in explanations[(name, day)]]


def test_explanations_name_the_peak_reason_for_each_worker() -> None:
    plan, requirement, scores = _explain_inputs()
    explanations = explain_roster(plan, requirement, scores, CONFIG)
    assert "RUSH_TOP_SCORE" in _codes(explanations, "A", TUESDAY)
    assert "RUSH_EXPLORATION" in _codes(explanations, "D", TUESDAY)
    for name in ("B", "C"):
        codes = _codes(explanations, name, TUESDAY)
        assert "COVERAGE_NEED" in codes and "RUSH_TOP_SCORE" not in codes
    assert rush_minutes(morning_pattern(), TUESDAY, requirement) == 210


def test_every_explanation_states_the_contract_hours_and_has_at_most_four_reasons() -> None:
    plan, requirement, scores = _explain_inputs()
    previous = {("A", TUESDAY): morning_pattern().pattern_id}
    explanations = explain_roster(plan, requirement, scores, CONFIG, last_roster=previous)
    for reasons in explanations.values():
        assert 1 <= len(reasons) <= 4
        assert reasons[-1].code == "CONTRACT_HOURS"
        assert "7.25 paid hours" in reasons[-1].text
    assert "SAME_AS_BEFORE" in _codes(explanations, "A", TUESDAY)
    assert "SAME_AS_BEFORE" not in _codes(explanations, "B", TUESDAY)


def test_an_unpopular_shift_for_a_worker_with_little_burden_is_explained_by_rotation() -> None:
    workers = {"X": worker("X"), "Y": worker("Y")}
    assignments = [
        Assignment("X", SATURDAY, evening_pattern()),
        Assignment("Y", MONDAY, evening_pattern()),
        Assignment("Y", TUESDAY, evening_pattern()),
        Assignment("Y", START + timedelta(days=3), evening_pattern()),
    ]
    plan = RosterPlan(START, 1, workers, assignments)
    requirement = requirement_for([MONDAY, TUESDAY, START + timedelta(days=3), SATURDAY])
    scores = neutral_scores(["X", "Y"])
    explanations = explain_roster(plan, requirement, scores, CONFIG)
    rotation = [r for r in explanations[("X", SATURDAY)] if r.code == "BURDEN_ROTATION"]
    assert rotation and "Saturday" in rotation[0].text
    assert "BURDEN_ROTATION" not in _codes(explanations, "Y", MONDAY)


def test_the_trade_off_notes_report_what_gave_way() -> None:
    plan, requirement, scores = _explain_inputs()
    assert trade_offs(plan, requirement, scores, CONFIG, [], 0.0, 0.0) == []
    notes = trade_offs(plan, requirement, scores, CONFIG, ["weekly_rest"], 90.0, 0.5)
    text = " ".join(notes)
    assert "weekly rest" in text
    assert "1.5 person-hours" in text
    assert "50%" in text
    calm = requirement_for([TUESDAY], headcount=4)
    missing = trade_offs(plan, calm, scores, CONFIG, [], 0.0, 0.0)
    assert missing == ["1 new workers got fewer peak slots than the learning quota"]


def _two_shift_problem(
    plan: RosterPlan,
    z: dict[str, float],
    requirement: RequirementTable,
    config: PlannerConfig = CONFIG,
) -> RosterProblem:
    tiers = {name: "all" for name in plan.workers}
    return RosterProblem(
        START,
        1,
        list(plan.workers.values()),
        solver_pattern_set(config, PatternSpec()),
        requirement,
        ScoreInputs(z, tiers, frozenset()),
        config,
    )


def _rush_requirement(days) -> RequirementTable:
    return with_rush(requirement_for(days), TUESDAY, *EVENING_RUSH, load=3.0)


def _midday() -> object:
    return pattern((clock_to_minutes("10:00"), clock_to_minutes("17:00")))


def _afternoon() -> object:
    return pattern((clock_to_minutes("12:00"), clock_to_minutes("19:00")))


def test_the_refinement_puts_the_stronger_score_on_the_rush_shift() -> None:
    workers = {"A": worker("A", 7.25), "B": worker("B", 7.25)}
    plan = RosterPlan(
        START,
        1,
        workers,
        [Assignment("A", TUESDAY, _midday()), Assignment("B", TUESDAY, _afternoon())],
    )
    problem = _two_shift_problem(plan, {"A": 1.5, "B": -1.0}, _rush_requirement([TUESDAY]))
    refined, swaps = refine_scores(problem, plan)
    assert refined.pattern_of("A", TUESDAY) == _afternoon()
    assert refined.pattern_of("B", TUESDAY) == _midday()
    assert len(swaps) == 1
    assert (swaps[0].first, swaps[0].second) == ("A", "B")
    assert swaps[0].gain == pytest.approx(2.5 * 2.0)
    again, more = refine_scores(problem, refined)
    assert more == [] and again.assignments == refined.assignments


def test_the_refinement_never_trades_hours_or_burden_for_a_score() -> None:
    workers = {"A": worker("A", 7.25), "B": worker("B", 7.25)}
    scores = {"A": 1.5, "B": -1.0}
    requirement = _rush_requirement([TUESDAY])
    closing = [
        Assignment("A", TUESDAY, morning_pattern()),
        Assignment("B", TUESDAY, evening_pattern()),
    ]
    plan = RosterPlan(START, 1, workers, closing)
    refined, swaps = refine_scores(_two_shift_problem(plan, scores, requirement), plan)
    assert swaps == [] and refined.assignments == plan.assignments
    longer = pattern((clock_to_minutes("12:00"), clock_to_minutes("19:30")))
    unequal = [Assignment("A", TUESDAY, _midday()), Assignment("B", TUESDAY, longer)]
    plan = RosterPlan(START, 1, workers, unequal)
    refined, swaps = refine_scores(_two_shift_problem(plan, scores, requirement), plan)
    assert swaps == [] and refined.assignments == plan.assignments


def test_the_negative_control_swaps_shifts_of_different_hours_and_burden() -> None:
    workers = {"A": worker("A", 7.25), "B": worker("B", 7.25)}
    plan = RosterPlan(
        START,
        1,
        workers,
        [Assignment("A", TUESDAY, morning_pattern()), Assignment("B", TUESDAY, evening_pattern())],
    )
    control = CONFIG.with_overrides({"objective.neutral_swaps": False})
    problem = _two_shift_problem(plan, {"A": 1.5, "B": -1.0}, _rush_requirement([TUESDAY]), control)
    refined, swaps = refine_scores(problem, plan)
    assert len(swaps) == 1
    assert refined.pattern_of("A", TUESDAY) == evening_pattern()
    assert swaps[0].gain == pytest.approx(10.0)
    assert worker_burden(refined, "A", CONFIG) != worker_burden(plan, "A", CONFIG)


def test_the_refinement_never_makes_a_shift_swap_that_breaks_a_labour_rule() -> None:
    workers = {"A": worker("A", 7.25), "B": worker("B", 14.5)}
    plan = RosterPlan(
        START,
        1,
        workers,
        [
            Assignment("A", TUESDAY, morning_pattern()),
            Assignment("B", MONDAY, evening_pattern()),
            Assignment("B", TUESDAY, evening_pattern()),
        ],
    )
    problem = _two_shift_problem(plan, {"A": 1.5, "B": -1.0}, _rush_requirement([MONDAY, TUESDAY]))
    refined, swaps = refine_scores(problem, plan)
    assert swaps == []
    assert refined.assignments == plan.assignments


def test_the_refinement_does_nothing_when_every_score_is_neutral() -> None:
    workers = {"A": worker("A", 7.25), "B": worker("B", 7.25)}
    plan = RosterPlan(
        START,
        1,
        workers,
        [Assignment("A", TUESDAY, morning_pattern()), Assignment("B", TUESDAY, evening_pattern())],
    )
    problem = _two_shift_problem(plan, {"A": 0.0, "B": 0.0}, _rush_requirement([TUESDAY]))
    refined, swaps = refine_scores(problem, plan)
    assert swaps == [] and refined.assignments == plan.assignments


@settings(max_examples=60, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False), min_size=6, max_size=6
    )
)
def test_the_strongest_scores_end_up_on_the_rush_shift_and_nothing_else_changes(
    scores: list[float],
) -> None:
    names = [f"W{index}" for index in range(6)]
    workers = {name: worker(name, 7.25) for name in names}
    assignments = [
        Assignment(name, TUESDAY, _midday() if index < 3 else _afternoon())
        for index, name in enumerate(names)
    ]
    plan = RosterPlan(START, 1, workers, assignments)
    z = dict(zip(names, scores, strict=True))
    problem = _two_shift_problem(plan, z, _rush_requirement([TUESDAY]))
    refined, _ = refine_scores(problem, plan)
    assert sorted(a.pattern.pattern_id for a in refined.assignments) == sorted(
        a.pattern.pattern_id for a in plan.assignments
    )
    assert sorted(a.worker_id for a in refined.assignments) == names
    assert set(hours_deviation(refined).values()) == {0.0}
    assert not [v for v in check_worker_rules(refined, CONFIG.rules) if v.severity == "error"]
    for name in names:
        assert worker_burden(refined, name, CONFIG) == worker_burden(plan, name, CONFIG)
    on_rush = sorted(z[a.worker_id] for a in refined.assignments if a.pattern == _afternoon())
    assert sum(on_rush) == pytest.approx(sum(sorted(scores)[3:]))


def _tier_problem(plan: RosterPlan, tiers: dict[str, str], limit: float = 0.1) -> RosterProblem:
    config = CONFIG.with_overrides({"objective.tier_epsilon": limit})
    requirement = _rush_requirement([TUESDAY])
    return RosterProblem(
        START,
        1,
        list(plan.workers.values()),
        solver_pattern_set(config, PatternSpec()),
        requirement,
        ScoreInputs({name: 0.0 for name in plan.workers}, tiers, frozenset()),
        config,
    )


def _four_workers(heavy: tuple[str, str]) -> tuple[RosterPlan, dict[str, str]]:
    names = ["A1", "A2", "C1", "C2"]
    workers = {name: worker(name, 7.25) for name in names}
    assignments = [
        Assignment(
            name,
            TUESDAY,
            evening_pattern() if name in heavy else morning_pattern(),
        )
        for name in names
    ]
    tiers = {"A1": "A", "A2": "A", "C1": "C", "C2": "C"}
    return RosterPlan(START, 1, workers, assignments), tiers


def test_the_tier_repair_moves_the_heavy_shift_away_from_the_over_burdened_tier() -> None:
    plan, tiers = _four_workers(heavy=("A1", "A2"))
    problem = _tier_problem(plan, tiers)
    before = max_tier_gap(burden_rates(plan, problem.config), tiers)
    repaired, swaps = repair_tiers(problem, plan)
    after = max_tier_gap(burden_rates(repaired, problem.config), tiers)
    assert before > 0.1 and after < before
    assert swaps and all(tiers[s.first] != tiers[s.second] for s in swaps)
    assert sorted(a.pattern.pattern_id for a in repaired.assignments) == sorted(
        a.pattern.pattern_id for a in plan.assignments
    )
    assert set(hours_deviation(repaired).values()) == {0.0}


def test_the_tier_repair_leaves_a_roster_alone_when_the_tiers_are_within_the_limit() -> None:
    plan, tiers = _four_workers(heavy=("A1", "C1"))
    problem = _tier_problem(plan, tiers, limit=0.35)
    repaired, swaps = repair_tiers(problem, plan)
    assert swaps == [] and repaired.assignments == plan.assignments


def test_the_tier_repair_never_breaks_the_rest_rule() -> None:
    names = ["A1", "C1"]
    workers = {"A1": worker("A1", 14.5), "C1": worker("C1", 14.5)}
    plan = RosterPlan(
        START,
        1,
        workers,
        [
            Assignment("A1", MONDAY, evening_pattern()),
            Assignment("A1", TUESDAY, evening_pattern()),
            Assignment("C1", MONDAY, morning_pattern()),
            Assignment("C1", TUESDAY, morning_pattern()),
        ],
    )
    problem = _tier_problem(plan, {"A1": "A", "C1": "C"}, limit=0.05)
    repaired, _ = repair_tiers(problem, plan)
    assert not [v for v in check_worker_rules(repaired, CONFIG.rules) if v.severity == "error"]
    assert names


@settings(max_examples=40, deadline=None)
@given(
    heavy=st.lists(st.booleans(), min_size=8, max_size=8),
    tier_codes=st.lists(st.sampled_from(["A", "B", "C"]), min_size=8, max_size=8),
)
def test_the_tier_repair_never_widens_the_gap_and_is_idempotent(
    heavy: list[bool], tier_codes: list[str]
) -> None:
    names = [f"W{index}" for index in range(8)]
    workers = {name: worker(name, 7.25) for name in names}
    plan = RosterPlan(
        START,
        1,
        workers,
        [
            Assignment(name, TUESDAY, evening_pattern() if flag else morning_pattern())
            for name, flag in zip(names, heavy, strict=True)
        ],
    )
    tiers = dict(zip(names, tier_codes, strict=True))
    problem = _tier_problem(plan, tiers, limit=0.05)
    before = max_tier_gap(burden_rates(plan, problem.config), tiers)
    repaired, swaps = repair_tiers(problem, plan)
    after = max_tier_gap(burden_rates(repaired, problem.config), tiers)
    assert after <= before + 1e-9
    assert sorted(a.pattern.pattern_id for a in repaired.assignments) == sorted(
        a.pattern.pattern_id for a in plan.assignments
    )
    again, more = repair_tiers(problem, repaired)
    assert more == [] and again.assignments == repaired.assignments
    assert len(swaps) <= 60
