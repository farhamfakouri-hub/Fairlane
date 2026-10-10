from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ree.config import Config
from ree.experiments.figures import (
    make_calibration_figure,
    make_learning_curve_figure,
    make_method_comparison_figure,
    make_scatter_figure,
)
from ree.experiments.methods import PERIODS, SPEED_METHODS
from ree.experiments.report import (
    confidence_interval,
    default_method_section,
    design_method_outcomes,
    markdown_table,
    paired_difference,
    summarize,
    v1_versus_v2,
    winner_sentences,
)
from ree.experiments.run_all import run_task
from ree.experiments.scenarios import SCENARIO_BY_NAME, SCENARIOS

LEVEL = 0.95
METHOD_NAMES = list(SPEED_METHODS)


def _results() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    profile = {
        "naive_items_per_shift": (0.60, 0.05),
        "hard_5s_stop": (0.45, 0.05),
        "v1_no_pause": (0.974, 0.0001),
        "v1_pause": (0.987, 0.0001),
        "v2_ols_pause": (0.989, 0.003),
        "v2_huber_pause": (0.990, 0.0001),
    }
    rows = []
    for scenario in SCENARIOS:
        for seed in range(8):
            shared = rng.normal(0, 0.003)
            for method, (level, spread) in profile.items():
                noise = rng.normal(0, spread)
                for period in PERIODS:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "seed": seed,
                            "method": method,
                            "period": period,
                            "target": "latent",
                            "spearman": level + shared + noise,
                            "kendall": level - 0.05,
                            "top_k_precision": 0.9,
                            "mean_abs_rank_error": 1.0,
                            "leakage_basket": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_confidence_interval_uses_t_quantiles_and_respects_bounds() -> None:
    values = pd.Series([0.9, 0.95, 0.99, 1.0, 0.97])
    mean, low, high = confidence_interval(values, 0.95)
    assert mean == pytest.approx(values.mean())
    assert low < mean < high
    _, _, clipped_high = confidence_interval(values, 0.95, (None, 1.0))
    assert clipped_high <= 1.0
    assert confidence_interval(pd.Series([0.5]), 0.95) == (0.5, 0.5, 0.5)
    assert np.isnan(confidence_interval(pd.Series([np.nan]), 0.95)[0])


def test_summary_and_paired_difference_are_computed_over_seeds() -> None:
    results = _results()
    summary = summarize(results, "spearman", LEVEL)
    assert set(summary["scenario"]) == {scenario.name for scenario in SCENARIOS}
    assert (summary["n"] == 8).all()
    best = summary[summary["scenario"] == "balanced"].set_index("method")["mean"]
    assert best["v2_huber_pause"] > best["v1_pause"] > best["naive_items_per_shift"]
    mean, low, _ = paired_difference(
        results, "balanced", "spearman", "v2_huber_pause", "v1_pause", LEVEL
    )
    assert 0.002 < mean < 0.004 and low > 0.0


def test_winner_text_separates_ties_trivial_gaps_and_clear_losses() -> None:
    results = _results()
    summary = summarize(results, "spearman", LEVEL)
    sentences = winner_sentences(results, summary, LEVEL, METHOD_NAMES)
    assert len(sentences) == len(SCENARIOS)
    first = sentences[0]
    assert "v2 regression (Huber) + pause rule" in first
    assert "Not distinguishable" in first and "v2 regression (OLS) + pause rule" in first
    assert "Detectably but trivially lower" in first and "v1 bins + pause rule (-0.003)" in first
    assert "Clearly lower" in first and "naive items per shift" in first
    outcomes = "\n".join(design_method_outcomes(results, summary, LEVEL, METHOD_NAMES))
    assert "Where the naive baseline wins or ties" in outcomes
    assert "Nowhere: it is clearly worse" in outcomes
    assert "Naive items-per-shift baseline, mean Spearman by scenario" in outcomes


def test_default_method_follows_the_highest_mean_and_v1_v2_table_has_a_verdict() -> None:
    results = _results()
    summary = summarize(results, "spearman", LEVEL)
    section = "\n".join(default_method_section(summary))
    assert "Highest: **v2 regression (Huber) + pause rule**" in section
    table = v1_versus_v2(results, summary, LEVEL)
    assert len(table) == len(SCENARIOS)
    assert set(table["verdict"]) <= {"tie", "v1 better", "v2 better"}


def test_markdown_table_renders_floats_and_missing_values() -> None:
    frame = pd.DataFrame({"a": ["x", "y"], "b": [0.123456, np.nan]})
    text = markdown_table(frame, 2)
    assert text.splitlines()[0] == "| a | b |"
    assert "| x | 0.12 |" in text and "| y | n/a |" in text


def test_scenarios_apply_overrides_seed_and_reduced_sizes() -> None:
    config = Config()
    biased = SCENARIO_BY_NAME["biased"].apply(config, 7, reduced=False)
    assert biased.simulator.scheduler.mode == "biased" and biased.seed == 7
    low = SCENARIO_BY_NAME["low_data"].apply(config, 7, reduced=True)
    assert low.simulator.n_weeks == 2
    assert low.simulator.population.n_cashiers == config.experiments.reduced_cashiers
    newcomers = SCENARIO_BY_NAME["newcomers"].apply(config, 7, reduced=True)
    assert newcomers.simulator.n_weeks == config.experiments.reduced_weeks
    assert newcomers.simulator.population.newcomer_share == 0.32
    shared = SCENARIO_BY_NAME["shared_tills"].apply(config, 7, reduced=False)
    assert shared.simulator.scheduler.shared_till_probability == 0.9


@pytest.fixture(scope="module")
def balanced_output():
    return run_task(Config().model_dump_json(), "balanced", 1000, True, True)


def test_task_runner_returns_every_method_period_and_target(balanced_output) -> None:
    metrics = balanced_output.metrics
    assert set(metrics["method"]) == set(METHOD_NAMES)
    assert set(metrics["period"]) == {"off", "peak"}
    assert set(metrics["target"]) == {"latent", "realized"}
    assert len(metrics) == len(METHOD_NAMES) * 2 * 2
    assert (
        metrics[["spearman", "kendall", "top_k_precision", "mean_abs_rank_error"]]
        .notna()
        .all()
        .all()
    )
    assert set(balanced_output.accuracy["variant"]) == {
        "split_by_cash_share",
        "full_to_each_cashier",
        "drop_shared_tills",
    }
    assert balanced_output.scatter is not None and len(balanced_output.scatter) > 0
    latent = metrics[metrics["target"] == "latent"].groupby("method")["spearman"].mean()
    assert latent["v2_ols_pause"] > 0.8 > latent["hard_5s_stop"]


def test_task_runner_is_deterministic_and_biased_assignment_fools_the_naive_baseline(
    balanced_output,
) -> None:
    again = run_task(Config().model_dump_json(), "balanced", 1000, True, False)
    pd.testing.assert_frame_equal(balanced_output.metrics, again.metrics)
    biased = run_task(Config().model_dump_json(), "biased", 1000, True, False)
    latent = biased.metrics[biased.metrics["target"] == "latent"]
    naive = latent[latent["method"] == "naive_items_per_shift"]["spearman"].mean()
    v1 = latent[latent["method"] == "v1_pause"]["spearman"].mean()
    leak = latent.groupby("method")["leakage_basket"].mean()
    assert naive < v1 - 0.4
    assert leak["naive_items_per_shift"] > leak["v1_pause"] + 0.3


def test_experiment_figures_are_written(balanced_output, tmp_path) -> None:
    metrics = balanced_output.metrics
    summary = summarize(
        metrics.assign(scenario="balanced").pipe(
            lambda frame: pd.concat(
                [frame.assign(scenario=scenario.name) for scenario in SCENARIOS], ignore_index=True
            )
        ),
        "spearman",
        LEVEL,
    )
    curve = pd.DataFrame(
        {
            "seed": 0,
            "weeks": [1, 2, 3] * len(METHOD_NAMES),
            "method": [name for name in METHOD_NAMES for _ in range(3)],
            "spearman": np.linspace(0.5, 0.99, 3 * len(METHOD_NAMES)),
        }
    )
    calibration = pd.DataFrame(
        {
            "component": ["accuracy", "speed_off", "speed_peak", "combined_off", "combined_peak"],
            "n": 10,
            "median_ratio": 1.1,
            "pooled_ratio": [1.4, 1.1, 1.0, 1.3, 1.3],
        }
    )
    paths = [
        make_scatter_figure(balanced_output.scatter, tmp_path / "scatter.png"),
        make_method_comparison_figure(summary, LEVEL, tmp_path / "comparison.png"),
        make_learning_curve_figure(curve, tmp_path / "curve.png"),
        make_calibration_figure(calibration, tmp_path / "calibration.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 20_000
