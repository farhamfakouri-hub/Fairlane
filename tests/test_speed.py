from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ree.config import Config, SpeedConfig
from ree.domain.models import SimulatedStore
from ree.engine.features import build_features
from ree.engine.shrinkage import (
    cohort_statistics,
    log_speed_ratio,
    shrink,
    shrinkage_weight,
    standard_scores,
    to_score_scale,
)
from ree.engine.speed_v1 import fit_v1
from ree.engine.speed_v2 import (
    DESIGN_COLUMNS,
    PRICE_TERMS,
    design_matrix,
    fit_linear,
    fit_v2,
    grouped_mad_keep_mask,
    mad_keep_mask,
    sanity_warnings,
)

RELAXED = SpeedConfig(min_store_bin_transactions=1, min_cashier_bin_transactions=5)


def test_worked_example_a_ols_on_four_baskets() -> None:
    items = np.array([5.0, 10.0, 20.0, 40.0])
    durations = np.array([42.0, 53.0, 88.0, 142.0])
    fit = fit_linear(np.column_stack([np.ones(4), items]), durations)
    alpha, beta = fit.theta
    assert beta == pytest.approx(2.90, abs=0.005)
    assert alpha == pytest.approx(26.8, abs=0.05)
    assert fit.residuals.sum() == pytest.approx(0.0, abs=1e-9)
    normal_equations = np.linalg.solve(
        np.column_stack([np.ones(4), items]).T @ np.column_stack([np.ones(4), items]),
        np.column_stack([np.ones(4), items]).T @ durations,
    )
    assert np.allclose(fit.theta, normal_equations)


def test_worked_example_b_log_ratio_cancels_where_the_plain_ratio_does_not() -> None:
    ratios = log_speed_ratio(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
    assert ratios.mean() == pytest.approx(0.0, abs=1e-15)
    assert np.mean([2.0, 0.5]) == pytest.approx(1.25)
    assert np.exp(ratios[0]) == pytest.approx(2.0)
    assert np.exp(ratios[1]) == pytest.approx(0.5)


def test_worked_example_c_shrinkage_halves_the_score() -> None:
    assert shrink(0.8, 150, 150) == pytest.approx(0.4)
    assert shrinkage_weight(150, 150) == pytest.approx(0.5)


def test_shrinkage_properties() -> None:
    assert shrink(0.2, 150, 150, center=0.1) == pytest.approx(0.15)
    assert shrink(0.5, 0, 150) == pytest.approx(0.0)
    assert shrink(0.5, 1e9, 150) == pytest.approx(0.5, abs=1e-6)
    weights = shrinkage_weight(np.array([10, 100, 1000, 10000]), 150)
    assert (np.diff(weights) > 0).all()
    assert (shrink(-0.4, np.array([10, 1000]), 150) > -0.4).all()


def test_cohort_standardisation_and_score_scale() -> None:
    values = np.array([0.0, 0.1, 0.2, 0.3, 9.0])
    reference = np.array([True, True, True, True, False])
    mean, spread = cohort_statistics(values, reference)
    scores = standard_scores(values, mean, spread)
    assert scores[:4].mean() == pytest.approx(0.0, abs=1e-12)
    assert scores[:4].std() == pytest.approx(1.0)
    assert scores[4] > 10
    assert to_score_scale(np.array([0.0, 1.0, -2.0])).tolist() == [50.0, 65.0, 20.0]
    assert cohort_statistics(values, np.array([True, False, False, False, False])) == (0.0, 1.0)
    assert cohort_statistics(np.ones(4), np.ones(4, dtype=bool)) == (1.0, 1.0)


def _rows(spec: list[tuple[str, str, str, int, float, int]]) -> pd.DataFrame:
    records = []
    for worker, period, lane, size_bin, duration, count in spec:
        for _ in range(count):
            records.append(
                {
                    "worker_id": worker,
                    "period": period,
                    "lane_group": lane,
                    "size_bin": size_bin,
                    "basket_class": "small",
                    "duration": duration,
                }
            )
    return pd.DataFrame(records)


def test_v1_matches_a_hand_computed_two_cell_example() -> None:
    frame = _rows(
        [
            ("A", "off", "zone_a", 0, 10.0, 5),
            ("B", "off", "zone_a", 0, 20.0, 5),
            ("A", "off", "zone_a", 1, 30.0, 15),
            ("B", "off", "zone_a", 1, 40.0, 15),
        ]
    )
    fit = fit_v1(frame, frame["duration"].to_numpy(), RELAXED)
    scores = fit.scores.set_index("worker_id")["raw_score"]
    expected_a = 0.25 * np.log(15 / 10) + 0.75 * np.log(35 / 30)
    expected_b = 0.25 * np.log(15 / 20) + 0.75 * np.log(35 / 40)
    assert scores["A"] == pytest.approx(expected_a)
    assert scores["B"] == pytest.approx(expected_b)
    assert fit.scores.set_index("worker_id")["n_valid"].tolist() == [20, 20]


def test_zero_durations_are_floored_so_scores_stay_finite() -> None:
    frame = _rows(
        [
            ("A", "off", "zone_a", 0, 0.0, 12),
            ("B", "off", "zone_a", 0, 20.0, 12),
            ("C", "off", "zone_a", 0, 25.0, 12),
        ]
    )
    scores = fit_v1(frame, frame["duration"].to_numpy(), RELAXED).scores.set_index("worker_id")
    assert np.isfinite(scores["raw_score"]).all()
    assert (
        scores.loc["A", "raw_score"] > scores.loc["B", "raw_score"] > scores.loc["C", "raw_score"]
    )


def test_v1_ignores_thin_cells_and_renormalises_the_weights() -> None:
    frame = _rows(
        [
            ("A", "off", "zone_a", 0, 10.0, 4),
            ("B", "off", "zone_a", 0, 20.0, 12),
            ("A", "off", "zone_a", 1, 30.0, 10),
            ("B", "off", "zone_a", 1, 40.0, 10),
        ]
    )
    fit = fit_v1(frame, frame["duration"].to_numpy(), RELAXED)
    scores = fit.scores.set_index("worker_id")
    assert scores.loc["A", "n_valid"] == 10
    store_median_bin1 = 35.0
    assert scores.loc["A", "raw_score"] == pytest.approx(np.log(store_median_bin1 / 30.0))


def test_v1_does_not_penalise_a_cashier_for_getting_big_baskets() -> None:
    rng = np.random.default_rng(4)
    base = {0: 40.0, 1: 60.0, 2: 90.0, 3: 140.0}
    records = []
    mixes = {"small_mix": [0.7, 0.2, 0.1, 0.0], "big_mix": [0.0, 0.1, 0.3, 0.6], "even": [0.25] * 4}
    for worker, mix in mixes.items():
        bins = rng.choice(4, size=400, p=mix)
        for size_bin in bins:
            records.append(
                {
                    "worker_id": worker,
                    "period": "off",
                    "lane_group": "zone_a",
                    "size_bin": int(size_bin),
                    "basket_class": "small",
                    "duration": base[int(size_bin)] * np.exp(rng.normal(0, 0.05)),
                }
            )
    frame = pd.DataFrame(records)
    scores = fit_v1(frame, frame["duration"].to_numpy(), SpeedConfig()).scores
    assert scores["raw_score"].abs().max() < 0.05


def test_v1_recovers_planted_speeds_and_is_scale_free() -> None:
    rng = np.random.default_rng(11)
    speeds = rng.normal(0, 0.2, size=12)
    records = []
    for worker, speed in enumerate(speeds):
        for _ in range(300):
            size_bin = int(rng.integers(0, 6))
            base = 25.0 + 14.0 * (size_bin + 1)
            duration = (25.0 + (base - 25.0) * np.exp(-speed)) * np.exp(rng.normal(0, 0.08))
            records.append(
                {
                    "worker_id": f"W{worker:02d}",
                    "period": "off",
                    "lane_group": "zone_a",
                    "size_bin": size_bin,
                    "basket_class": "small",
                    "duration": duration,
                }
            )
    frame = pd.DataFrame(records)
    fit = fit_v1(frame, frame["duration"].to_numpy(), SpeedConfig())
    scores = fit.scores.sort_values("worker_id")["raw_score"].to_numpy()
    assert stats.spearmanr(scores, speeds).statistic > 0.95
    scaled = fit_v1(frame, 7.0 * frame["duration"].to_numpy(), SpeedConfig())
    assert np.allclose(scaled.scores.sort_values("worker_id")["raw_score"], scores)


def test_v1_bootstrap_is_seeded_and_centred_on_the_score() -> None:
    frame = _rows(
        [
            ("A", "off", "zone_a", 0, 10.0, 40),
            ("B", "off", "zone_a", 0, 20.0, 40),
        ]
    )
    frame["duration"] = frame["duration"] * np.exp(
        np.random.default_rng(1).normal(0, 0.1, len(frame))
    )
    fit = fit_v1(frame, frame["duration"].to_numpy(), RELAXED)
    first = fit.bootstrap("A", "off", np.random.default_rng(5), 300)
    again = fit.bootstrap("A", "off", np.random.default_rng(5), 300)
    assert np.array_equal(first, again)
    raw = fit.scores.set_index("worker_id").loc["A", "raw_score"]
    assert abs(first.mean() - raw) < 0.05
    assert first.std() > 0
    assert fit.bootstrap("missing", "off", np.random.default_rng(5), 10).tolist() == [0.0] * 10


def _synthetic_context(rng: np.random.Generator, n_rows: int = 6000) -> pd.DataFrame:
    n_items = rng.integers(1, 60, size=n_rows)
    n_keyed = rng.binomial(n_items, 0.15)
    n_tagged = rng.binomial(n_items - n_keyed, 0.08)
    loyalty = rng.integers(0, 2, size=n_rows)
    vouchers = rng.integers(0, 3, size=n_rows)
    return pd.DataFrame(
        {
            "n_items": n_items,
            "n_hand_keyed": n_keyed,
            "n_tagged": n_tagged,
            "has_loyalty": loyalty,
            "n_vouchers": vouchers,
        }
    )


TRUE_THETA = np.array([25.0, 2.9, 0.7, 6.0, 8.0, 4.0])


def test_v2_ols_recovers_the_planted_price_list_and_huber_resists_outliers() -> None:
    rng = np.random.default_rng(2)
    frame = _synthetic_context(rng)
    design = design_matrix(frame)
    clean = design @ TRUE_THETA + rng.normal(0, 4.0, len(frame))
    ols = fit_linear(design, clean, "ols")
    assert np.allclose(ols.theta, TRUE_THETA, atol=[1.0, 0.1, 0.4, 0.5, 0.7, 0.7])
    contaminated = clean.copy()
    hit = rng.random(len(frame)) < 0.06
    contaminated[hit] += rng.uniform(40, 120, hit.sum())
    biased_ols = fit_linear(design, contaminated, "ols")
    robust = fit_linear(design, contaminated, "huber")
    assert abs(biased_ols.theta[0] - 25.0) > 3.0
    assert abs(robust.theta[0] - 25.0) < abs(biased_ols.theta[0] - 25.0)
    assert abs(robust.theta[1] - 2.9) < 0.15


def test_design_columns_and_terms_line_up() -> None:
    assert len(PRICE_TERMS) == len(DESIGN_COLUMNS) + 1
    assert design_matrix(_synthetic_context(np.random.default_rng(0), 5)).shape == (5, 6)


def test_mad_filter_removes_only_extreme_rows() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0, 0.1, 1000)
    values[:5] = [3.0, -4.0, 2.5, -2.5, 5.0]
    keep = mad_keep_mask(values, 3.0)
    assert not keep[:5].any()
    assert keep[5:].mean() > 0.99
    assert mad_keep_mask(np.ones(10), 3.0).all()


def test_grouped_mad_filter_is_relative_to_each_cashier() -> None:
    rng = np.random.default_rng(4)
    slow = rng.normal(-0.6, 0.08, 400)
    typical = rng.normal(0.0, 0.08, 3600)
    slow[:6] = [-3.0, -2.5, 1.5, -3.5, -2.8, -3.2]
    values = np.concatenate([slow, typical])
    workers = np.array(["slow"] * 400 + [f"typical{i}" for i in range(9) for _ in range(400)])
    periods = np.array(["off"] * 4000)
    keep = grouped_mad_keep_mask(values, workers, periods, 3.0)
    assert not keep[:6].any()
    assert keep[6:400].mean() > 0.97
    assert keep[400:].mean() > 0.97
    store_wide = mad_keep_mask(values, 3.0)
    assert store_wide[6:400].mean() < 0.1


def test_v2_keeps_the_transactions_of_a_consistently_slow_cashier() -> None:
    rng = np.random.default_rng(12)
    speeds = np.concatenate([rng.normal(0, 0.1, 9), [-0.7]])
    frames = []
    for worker, speed in enumerate(speeds):
        context = _synthetic_context(rng, 500)
        scaled = np.array([25.0, 2.9 / np.exp(speed), 0.7 / np.exp(speed), 6.0, 8.0, 4.0])
        context["duration"] = (design_matrix(context) @ scaled) * np.exp(rng.normal(0, 0.08, 500))
        context["worker_id"] = f"W{worker:02d}"
        context["lane_group"] = "zone_a"
        context["period"] = "off"
        context["basket_class"] = "small"
        frames.append(context)
    frame = pd.concat(frames, ignore_index=True)
    fit = fit_v2(frame, frame["duration"].to_numpy(), SpeedConfig())
    scores = fit.scores.set_index("worker_id")
    assert scores.loc["W09", "n_valid"] > 0.95 * 500
    assert scores["raw_score"].idxmin() == "W09"


def test_v2_ranks_cashiers_and_recentres_each_context() -> None:
    rng = np.random.default_rng(8)
    speeds = rng.normal(0, 0.2, size=10)
    frames = []
    for lane in ("zone_a", "zone_b"):
        for worker, speed in enumerate(speeds):
            context = _synthetic_context(rng, 500)
            scaled = np.array([25.0, 2.9 / np.exp(speed), 0.7 / np.exp(speed), 6.0, 8.0, 4.0])
            noise = np.exp(rng.normal(0, 0.1, len(context)))
            context["duration"] = (design_matrix(context) @ scaled) * noise
            context["worker_id"] = f"W{worker:02d}"
            context["lane_group"] = lane
            context["period"] = "off"
            context["basket_class"] = "small"
            frames.append(context)
    frame = pd.concat(frames, ignore_index=True)
    fit = fit_v2(frame, frame["duration"].to_numpy(), SpeedConfig())
    scores = fit.scores.sort_values("worker_id")["raw_score"].to_numpy()
    assert stats.spearmanr(scores, speeds).statistic > 0.9
    centred = fit.row_scores.assign(lane=frame["lane_group"]).dropna(subset=["log_ratio"])
    medians = centred.groupby("lane")["log_ratio"].median()
    assert np.allclose(medians, 0.0, atol=0.02)
    assert set(fit.price_list["term"]) == set(PRICE_TERMS)
    assert len(fit.price_list) == 2 * len(PRICE_TERMS)


def test_v2_bootstrap_is_seeded() -> None:
    rng = np.random.default_rng(9)
    context = _synthetic_context(rng, 400)
    context["duration"] = design_matrix(context) @ TRUE_THETA * np.exp(rng.normal(0, 0.1, 400))
    context["worker_id"] = "W01"
    context["lane_group"] = "zone_a"
    context["period"] = "off"
    context["basket_class"] = "small"
    fit = fit_v2(context, context["duration"].to_numpy(), SpeedConfig())
    first = fit.bootstrap("W01", "off", np.random.default_rng(1), 100)
    assert np.array_equal(first, fit.bootstrap("W01", "off", np.random.default_rng(1), 100))
    assert abs(first.mean() - fit.scores.iloc[0]["raw_score"]) < 0.02


def test_sanity_warnings_fire_only_on_clear_contradictions() -> None:
    config = SpeedConfig()

    def table(delta: float, gamma: float, error: float = 0.1) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lane_group": ["zone_a"] * 2,
                "period": ["off"] * 2,
                "term": ["delta", "gamma"],
                "estimate": [delta, gamma],
                "std_error": [error, error],
            }
        )

    assert sanity_warnings(table(6.0, 0.7), config) == []
    assert sanity_warnings(table(4.9, 0.7, 0.2), config) == []
    assert len(sanity_warnings(table(3.0, 0.7), config)) == 1
    assert len(sanity_warnings(table(9.0, 0.7), config)) == 1
    assert len(sanity_warnings(table(6.0, 1.8), config)) == 1
    assert len(sanity_warnings(table(3.0, 1.8), config)) == 2


@pytest.fixture(scope="module")
def default_features(default_store: SimulatedStore, default_config: Config):
    return build_features(default_store.data, default_config)


def _spearman(scores: pd.DataFrame, period: str, target: pd.Series) -> float:
    selected = scores[scores["period"] == period].set_index("worker_id")["raw_score"]
    common = selected.index.intersection(target.index)
    return float(stats.spearmanr(selected[common], target[common]).statistic)


def test_v1_and_v2_recover_the_hidden_order_on_the_simulated_store(
    default_store: SimulatedStore, default_config: Config, default_features
) -> None:
    rows = default_features.speed_rows
    truth = default_store.ground_truth.set_index("worker_id")
    veterans = truth[~truth["is_newcomer"]]
    durations = rows["duration_capped"].to_numpy()
    speed_config = default_config.engine.speed
    fits = {
        "v1": fit_v1(rows, durations, speed_config).scores,
        "v2": fit_v2(rows, durations, speed_config).scores,
        "huber": fit_v2(
            rows, durations, speed_config.model_copy(update={"regression": "huber"})
        ).scores,
    }
    for name, scores in fits.items():
        assert _spearman(scores, "off", veterans["true_speed"]) > 0.9, name
        assert _spearman(scores, "peak", veterans["true_speed"] + veterans["peak_effect"]) > 0.85, (
            name
        )


def test_price_list_on_the_simulated_store_passes_the_sanity_checks(
    default_config: Config, default_features
) -> None:
    rows = default_features.speed_rows
    fit = fit_v2(rows, rows["duration_capped"].to_numpy(), default_config.engine.speed)
    estimates = fit.price_list.pivot_table(
        index=["lane_group", "period"], columns="term", values="estimate"
    )
    assert (estimates["gamma"] < 1.0).all()
    assert estimates["delta"].between(4.0, 7.5).all()
    assert sanity_warnings(fit.price_list, default_config.engine.speed) == []
    assert estimates.loc["priority", "beta"].min() > 1.15 * estimates.loc["zone_a", "beta"].max()


def test_hard_stop_timing_destroys_the_signal_that_capping_keeps(
    default_store: SimulatedStore, default_config: Config, default_features
) -> None:
    rows = default_features.speed_rows
    truth = default_store.ground_truth.set_index("worker_id")
    veterans = truth[~truth["is_newcomer"]]
    capped = fit_v1(rows, rows["duration_capped"].to_numpy(), default_config.engine.speed)
    hard = fit_v1(rows, rows["duration_hard_stop"].to_numpy(), default_config.engine.speed)
    good = _spearman(capped.scores, "off", veterans["true_speed"])
    bad = _spearman(hard.scores, "off", veterans["true_speed"])
    assert good > 0.9
    assert good - bad > 0.3
