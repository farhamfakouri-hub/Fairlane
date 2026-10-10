from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ree.api.provider import InMemoryScoreProvider
from ree.planner.cli import main
from ree.planner.eval.figures import write_figures
from ree.planner.eval.metrics import week_metrics
from ree.planner.eval.plans import (
    ALL_PLANS,
    PLAN_NAMES,
    POSTS_ONLY_PLAN,
    build_inputs,
    build_plan,
    flat_requirement,
    plan_specification,
    restrict_scores,
)
from ree.planner.eval.report import (
    ablation_statement,
    build_report,
    live_statement,
    placebo_statement,
    tally,
)
from ree.planner.eval.run_eval import evaluate_seed
from ree.planner.eval.stats import bootstrap_mean, paired_bootstrap
from ree.planner.eval.validation import queue_validation
from ree.planner.eval.world import SCENARIOS, DayRun
from ree.planner.roster.model import neutral_scores
from ree.planner.sim.part1_workforce import (
    WorkerTruth,
    oracle_inputs,
    service_multiplier,
    shuffled_inputs,
)
from ree.planner.sim.queue_sim import DayOutcome
from ree.planner.sim.scenarios import store_config


def fast_config(**extra: object):
    return store_config(
        "tiny", **{"solver.backend": "heuristic", "evaluation.replications": 2, **extra}
    )


@pytest.fixture(scope="module")
def inputs():
    return build_inputs(fast_config(), 1)


@pytest.fixture(scope="module")
def evaluated() -> pd.DataFrame:
    config = fast_config()
    rows = evaluate_seed(config, 1, list(config.evaluation.scenarios), PLAN_NAMES)
    return pd.DataFrame(rows)


def test_the_part_one_workforce_gives_scores_and_hidden_truth_for_every_worker(inputs) -> None:
    ids = [f"W{index:03d}" for index in range(1, 11)]
    assert sorted(inputs.workforce.truth) == ids
    assert isinstance(inputs.workforce.provider, InMemoryScoreProvider)
    assert all(inputs.workforce.provider.get_score(w) is not None for w in ids)
    rated = [w for w in ids if w not in inputs.scores.exploration]
    z = [inputs.scores.z[w] for w in rated]
    speed = [inputs.workforce.truth[w].log_speed for w in rated]
    assert np.corrcoef(z, speed)[0, 1] > 0.3
    assert [w.worker_id for w in inputs.workers] == ids


def test_the_service_multiplier_follows_the_hidden_speed_and_the_peak_effect() -> None:
    truth = WorkerTruth("W001", log_speed=0.2, peak_effect=-0.1, newcomer=False, tenure_weeks=50.0)
    assert service_multiplier(truth, False) == pytest.approx(np.exp(-0.2))
    assert service_multiplier(truth, True) == pytest.approx(np.exp(-0.1))


def test_the_oracle_scores_are_the_standardised_true_speeds(inputs) -> None:
    oracle = oracle_inputs(inputs.workforce.truth, inputs.config)
    ids = sorted(inputs.workforce.truth)
    values = np.array([oracle.z[w] for w in ids])
    assert values.mean() == pytest.approx(0.0, abs=0.1)
    assert oracle.exploration == frozenset()
    tiers = [oracle.tiers[w] for w in ids]
    assert tiers.count("A") == 2 and tiers.count("C") == 2
    fastest = max(ids, key=lambda w: inputs.workforce.truth[w].log_speed)
    assert oracle.tiers[fastest] == "A"


def test_the_placebo_scores_are_the_real_scores_moved_to_other_workers(inputs) -> None:
    placebo = shuffled_inputs(inputs.scores, 1)
    assert sorted(placebo.z.values()) == pytest.approx(sorted(inputs.scores.z.values()))
    assert sorted(placebo.tiers.values()) == sorted(inputs.scores.tiers.values())
    assert len(placebo.exploration) == len(inputs.scores.exploration)
    assert placebo.z != inputs.scores.z
    assert shuffled_inputs(inputs.scores, 1).z == placebo.z
    assert shuffled_inputs(inputs.scores, 2).z != placebo.z


def test_plan_a_staffs_the_same_registers_in_every_rush_slot_and_every_calm_slot(inputs) -> None:
    flat = inputs.flat
    rush = {row.registers for d in flat.dates() for _, row in flat.slots(d) if row.rush}
    calm = {row.registers for d in flat.dates() for _, row in flat.slots(d) if not row.rush}
    assert len(rush) == 1 and len(calm) == 1
    original = inputs.requirement
    for day in original.dates():
        for (_, before), (_, after) in zip(original.slots(day), flat.slots(day), strict=True):
            assert (before.k_sc, before.floaters, before.rush) == (
                after.k_sc,
                after.floaters,
                after.rush,
            )
    assert flat_requirement(flat).days == flat.days


def test_each_plan_uses_its_requirement_and_its_scores(inputs) -> None:
    neutral = neutral_scores([w.worker_id for w in inputs.workers])
    assert plan_specification(inputs, "A") == (inputs.flat, neutral)
    assert plan_specification(inputs, "B") == (inputs.requirement, neutral)
    assert plan_specification(inputs, "C") == (inputs.requirement, inputs.scores)
    assert plan_specification(inputs, "D") == (inputs.requirement, inputs.oracle)
    assert plan_specification(inputs, "E") == (inputs.requirement, inputs.placebo)
    with pytest.raises(ValueError):
        plan_specification(inputs, "Z")


def test_a_plan_bundle_holds_a_roster_and_a_daily_plan_for_each_working_day(inputs) -> None:
    bundle = build_plan(inputs, "C")
    assert bundle.roster.report is not None
    days = {a.day for a in bundle.roster.plan.assignments}
    assert set(bundle.daily) == days


def test_plan_f_keeps_the_roster_of_plan_b_and_uses_the_scores_in_the_posts(inputs) -> None:
    neutral = build_plan(inputs, "B")
    posts_only = build_plan(inputs, POSTS_ONLY_PLAN, base=neutral)
    assert posts_only.name == POSTS_ONLY_PLAN
    assert posts_only.roster is neutral.roster
    assert posts_only.scores == inputs.scores
    assert set(posts_only.daily) == set(neutral.daily)
    assert any(
        posts_only.daily[day].plan.segments != neutral.daily[day].plan.segments
        for day in neutral.daily
    )
    rebuilt = build_plan(inputs, POSTS_ONLY_PLAN)
    assert rebuilt.roster.plan.assignments == neutral.roster.plan.assignments


def test_restricting_scores_keeps_only_the_listed_workers(inputs) -> None:
    kept = {"W001", "W002", "W003"}
    restricted = restrict_scores(inputs.scores, kept)
    assert set(restricted.z) == kept and set(restricted.tiers) == kept
    assert restricted.exploration <= kept


def run_of(waits, abandoned, rush, busy=300.0, opened=600.0, area=120.0, over=30.0) -> DayRun:
    count = len(waits)
    outcome = DayOutcome(
        arrival=np.arange(count, dtype=float),
        wait=np.array(waits, dtype=float),
        served=~np.array(abandoned, dtype=bool),
        abandoned=np.array(abandoned, dtype=bool),
        register=np.ones(count, dtype=np.int64),
        service=np.full(count, 60.0),
        busy_seconds=busy,
        open_seconds=opened,
        queue_area=area,
        over_limit_seconds=over,
    )
    return DayRun(date(2026, 11, 2), outcome, np.array(rush, dtype=bool), 1000.0, 0)


def test_the_week_metrics_have_known_values() -> None:
    run = run_of([0.0, 60.0, 200.0, 100.0], [False, False, False, True], [False, False, True, True])
    metrics = week_metrics([run], 180.0)
    assert metrics["customers"] == 4
    assert metrics["mean_wait_s"] == pytest.approx(90.0)
    assert metrics["p95_wait_s"] == pytest.approx(185.0)
    assert metrics["share_over_target"] == pytest.approx(0.5)
    assert metrics["abandon_rate"] == pytest.approx(0.25)
    assert metrics["rush_mean_wait_s"] == pytest.approx(150.0)
    assert metrics["rush_share_over_target"] == pytest.approx(1.0)
    assert metrics["idle_share"] == pytest.approx(0.5)
    assert metrics["mean_queue_per_register"] == pytest.approx(0.2)
    assert metrics["share_time_over_limit"] == pytest.approx(0.03)


def test_the_week_metrics_combine_days_and_survive_an_empty_week() -> None:
    first = run_of([10.0, 30.0], [False, False], [False, True])
    second = run_of([50.0, 70.0], [False, False], [True, True], busy=100.0, opened=400.0)
    metrics = week_metrics([first, second], 180.0)
    assert metrics["mean_wait_s"] == pytest.approx(40.0)
    assert metrics["idle_share"] == pytest.approx(1.0 - 400.0 / 1000.0)
    assert week_metrics([], 180.0)["customers"] == 0.0


def test_a_paired_bootstrap_of_a_constant_difference_is_that_difference() -> None:
    result = paired_bootstrap([5.0, 6.0, 7.0, 8.0], [10.0, 11.0, 12.0, 13.0])
    assert (result.mean, result.low, result.high) == (-5.0, -5.0, -5.0)
    assert result.verdict == "lower"
    assert paired_bootstrap([1.0, 2.0], [1.0, 2.0]).verdict == "no clear difference"
    assert paired_bootstrap([9.0, 9.0], [1.0, 1.0]).verdict == "higher"
    assert paired_bootstrap([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == paired_bootstrap(
        [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
    )
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_the_bootstrap_interval_of_a_mean_contains_the_mean() -> None:
    mean, low, high = bootstrap_mean([1.0, 2.0, 3.0, 4.0, 10.0])
    assert low <= mean <= high
    assert mean == pytest.approx(4.0)


def test_the_evaluation_gives_one_row_per_scenario_and_plan(evaluated) -> None:
    main = evaluated[evaluated["scenario"].isin(SCENARIOS)]
    assert len(main) == len(SCENARIOS) * len(ALL_PLANS)
    assert set(evaluated["plan"]) == set(ALL_PLANS)
    assert set(evaluated["scenario"]) == set(SCENARIOS) | {"layout_0", "layout_0.3"}
    assert "live_proposed" in evaluated.columns
    for column in ("mean_wait_s", "p95_wait_s", "share_over_target", "roster_errors", "paid_hours"):
        assert column in evaluated.columns


def test_every_plan_faces_the_same_customers_within_a_scenario_and_seed(evaluated) -> None:
    grouped = evaluated.groupby(["scenario", "seed"])["customers"].nunique()
    assert (grouped == 1).all()


def test_the_scenarios_change_the_world_as_described(evaluated) -> None:
    plan = evaluated[(evaluated["plan"] == "B") & (evaluated["seed"] == 1)].set_index("scenario")
    normal = plan.loc["normal_week"]
    assert plan.loc["saturday_surge", "customers"] > normal["customers"]
    assert plan.loc["season_peak", "customers"] > 1.2 * normal["customers"]
    assert plan.loc["small_team", "paid_hours"] < normal["paid_hours"]


def test_the_evaluation_is_deterministic() -> None:
    config = fast_config()
    first = evaluate_seed(config, 1, ["normal_week"], ["B", "C"])
    second = evaluate_seed(config, 1, ["normal_week"], ["B", "C"])
    assert first == second


def crafted(c_values, e_values, b_values) -> pd.DataFrame:
    rows = []
    for scenario in ("normal_week", "saturday_surge"):
        for seed in range(1, 5):
            for plan, values in (("B", b_values), ("C", c_values), ("E", e_values)):
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "plan": plan,
                        "share_over_target": values[seed - 1],
                        "mean_wait_s": values[seed - 1] * 100,
                    }
                )
    return pd.DataFrame(rows)


def test_the_placebo_statement_says_so_when_the_scores_help() -> None:
    frame = crafted([0.05, 0.051, 0.049, 0.05], [0.08, 0.081, 0.079, 0.08], [0.08] * 4)
    assert "better than the placebo in most scenarios" in placebo_statement(frame)


def test_the_placebo_statement_admits_when_the_scores_cannot_be_told_from_chance() -> None:
    same = [0.07, 0.071, 0.069, 0.07]
    frame = crafted(same, same, [0.08] * 4)
    assert "cannot be told apart from the placebo" in placebo_statement(frame)


def live_frame(normal_shift: float, absence_shift: float) -> pd.DataFrame:
    rows = []
    for scenario, shift in (("normal_week", normal_shift), ("absences", absence_shift)):
        for seed in range(1, 5):
            base = 0.05 + 0.001 * seed
            for plan, offset in (("C", 0.0), ("L", shift)):
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "plan": plan,
                        "share_over_target": base + offset,
                        "mean_wait_s": (base + offset) * 100,
                        "live_flips": 3.0,
                    }
                )
    return pd.DataFrame(rows)


def test_the_live_statement_names_the_scenarios_where_live_replanning_helped_and_hurt() -> None:
    statement = live_statement(live_frame(0.01, -0.01))
    assert "The result is mixed" in statement
    assert "lower in the absences scenario and higher in the normal week scenario" in statement


def test_the_live_statement_admits_when_live_replanning_makes_no_difference() -> None:
    assert "cannot be told apart from the static plan" in live_statement(live_frame(0.0, 0.0))


def test_the_live_statement_does_not_claim_a_help_it_did_not_show() -> None:
    statement = live_statement(live_frame(0.01, 0.01))
    assert "did not reduce waiting in any scenario" in statement
    assert "mixed" not in statement


def test_the_live_statement_reports_a_help_that_never_hurt() -> None:
    statement = live_statement(live_frame(-0.01, 0.0))
    assert "never made waiting worse" in statement
    assert "the normal week scenario" in statement


def ablation_frame(posts_shift: float, full_shift: float) -> pd.DataFrame:
    rows = []
    for scenario in ("normal_week", "saturday_surge"):
        for seed in range(1, 5):
            base = 0.08 + 0.001 * seed
            for plan, offset in (("B", 0.0), ("F", posts_shift), ("C", full_shift)):
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "plan": plan,
                        "share_over_target": base + offset,
                        "mean_wait_s": (base + offset) * 100,
                    }
                )
    return pd.DataFrame(rows)


def test_the_ablation_statement_credits_the_posts_when_the_roster_adds_nothing() -> None:
    statement = ablation_statement(ablation_frame(-0.02, -0.02))
    assert "lower the mean wait against plan B in 2 of 2 scenarios" in statement
    assert "F captures 100% of the mean-wait reduction of C" in statement
    assert "Most of the gain comes from choosing who stands at which register" in statement
    assert "keeps the roster of plan B" in statement


def test_the_ablation_statement_credits_both_steps_when_each_helps() -> None:
    statement = ablation_statement(ablation_frame(-0.01, -0.03))
    assert "F captures 33% of the mean-wait reduction of C" in statement
    assert "Both uses of the scores help" in statement


def test_the_ablation_statement_says_so_when_the_posts_alone_do_not_help() -> None:
    statement = ablation_statement(ablation_frame(0.0, -0.02))
    assert "The gain depends on the roster" in statement
    assert "F captures 0% of the mean-wait reduction of C" in statement


def test_the_ablation_statement_admits_when_nothing_helps() -> None:
    statement = ablation_statement(ablation_frame(0.0, 0.0))
    assert "no reduction in the mean wait" in statement
    assert "Neither step gives a clear gain" in statement


def test_the_ablation_statement_is_empty_without_plan_f() -> None:
    frame = ablation_frame(-0.01, -0.02)
    assert ablation_statement(frame[frame["plan"] != "F"]) == ""


def test_the_tally_counts_scenarios_by_the_reading_of_the_interval() -> None:
    frame = crafted([0.05, 0.051, 0.049, 0.05], [0.08, 0.081, 0.079, 0.08], [0.08] * 4)
    counts, differences = tally(frame, "C", "E", "share_over_target", 100.0)
    assert (counts.lower, counts.tie, counts.higher) == (2, 0, 0)
    assert set(differences) == {"normal_week", "saturday_surge"}


def test_the_figures_and_the_report_are_written(evaluated, tmp_path) -> None:
    validation = queue_validation(cases=((2, 1.5),), customers=3000, seeds=1)
    paths = write_figures(evaluated, validation, tmp_path)
    assert set(paths) == {"waits", "differences", "alignment", "fairness", "validation"}
    assert all(path.exists() and path.stat().st_size > 1000 for path in paths.values())
    text = build_report(evaluated, validation, fast_config(), paths, "tiny")
    for heading in (
        "# Planner evaluation",
        "## Waiting times",
        "## Queue length and idle capacity",
        "## Paired differences",
        "## Honest reading",
        "## Live re-planning",
        "## Fairness in every scenario",
        "## Validation of the queue simulator",
        "## What this does not prove",
    ):
        assert heading in text
    assert "Placebo test" in text
    assert "normal week" in text and "small team" in text
    assert "Customers waiting per open register" in text
    assert "Idle share of open-register time (%)" in text
    assert "Where the benefit of the scores comes from" not in text


def test_the_evaluate_command_writes_results_figures_and_a_report(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    config = fast_config(**{"paths.outputs_dir": str(tmp_path), "evaluation.replications": 1})
    monkeypatch.setattr(cli, "evaluation_config", lambda store: config)
    code = main(["evaluate", "--reduced", "--store", "tiny", "--seeds", "1", "--jobs", "1"])
    assert code == 0
    reduced = tmp_path / "reduced"
    assert (reduced / "results.csv").exists()
    assert (reduced / "evaluation.md").exists()
    assert (reduced / "figures" / "planner_waits.png").exists()
    assert "F" in set(pd.read_csv(reduced / "results.csv")["plan"])
    report_text = (reduced / "evaluation.md").read_text(encoding="utf-8")
    assert "Where the benefit of the scores comes from" in report_text
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    assert images and all((reduced / image).exists() for image in images)
    assert "wrote" in capsys.readouterr().out


def test_the_report_command_needs_results_first(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    monkeypatch.setattr(cli, "evaluation_config", lambda store: config)
    assert main(["report", "--reduced", "--store", "tiny"]) == 1
    assert "run `planner evaluate` first" in capsys.readouterr().out


def test_the_queue_check_command_reports_the_gap_to_erlang_c(
    monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    monkeypatch.setattr(cli, "queue_validation", lambda: queue_validation(((4, 3.0),), 60000, 4))
    assert main(["queue-check"]) == 0
    assert "largest gap for the shared queue" in capsys.readouterr().out
