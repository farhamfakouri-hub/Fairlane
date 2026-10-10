from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ree.config import AccuracyConfig, CombineConfig, Config, ReviewConfig
from ree.domain.models import SimulatedStore
from ree.engine.accuracy import (
    AccuracyParameters,
    accuracy_scores,
    attribute_discrepancies,
    fit_accuracy,
    review_flags,
    weekly_records,
)
from ree.engine.features import as_of_week_index, build_features

DATA_START = pd.Timestamp("2025-11-24")
CONFIG = AccuracyConfig()
COMBINE = CombineConfig()


def _sheet(rows: list[tuple[int, str, str | None, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [DATA_START + pd.Timedelta(days=7 * week + 1) for week, *_ in rows],
            "register": [3] * len(rows),
            "drawer_id": [row[1] for row in rows],
            "worker_id": [row[2] for row in rows],
            "cash_handled": [row[3] for row in rows],
            "till_discrepancy": [row[4] for row in rows],
        }
    )


def _features(spec: dict[str, dict[int, tuple[int, int, int]]]) -> pd.DataFrame:
    records = []
    transaction_id = 0
    for worker, weeks in spec.items():
        for week, (transactions, void_errors, manual) in weeks.items():
            for index in range(transactions):
                transaction_id += 1
                records.append(
                    {
                        "transaction_id": transaction_id,
                        "worker_id": worker,
                        "week_index": week,
                        "n_void_errors": 1 if index < void_errors else 0,
                        "n_manual_entries": 1 if index < manual else 0,
                        "valid_for_accuracy": True,
                    }
                )
    return pd.DataFrame(records)


def test_single_owner_drawers_are_attributed_directly_and_summed_per_week() -> None:
    sheet = _sheet(
        [
            (0, "d1", "A", 800.0, -2.0),
            (0, "d2", "A", 200.0, 0.5),
            (1, "d3", "B", 500.0, 4.0),
        ]
    )
    result = attribute_discrepancies(sheet, DATA_START).set_index(["worker_id", "week_index"])
    assert result.loc[("A", 0), "delta"] == pytest.approx(-1.5)
    assert result.loc[("A", 0), "cash"] == pytest.approx(1000.0)
    assert result.loc[("B", 1), "delta"] == pytest.approx(4.0)


def test_shared_tills_are_split_by_cash_handled() -> None:
    sheet = _sheet([(2, "shared", "A", 600.0, -10.0), (2, "shared", "B", 400.0, -10.0)])
    result = attribute_discrepancies(sheet, DATA_START).set_index("worker_id")
    assert result.loc["A", "delta"] == pytest.approx(-6.0)
    assert result.loc["B", "delta"] == pytest.approx(-4.0)
    assert result["delta"].sum() == pytest.approx(-10.0)


def test_unclear_owners_drop_the_whole_drawer() -> None:
    sheet = _sheet(
        [
            (0, "mixed", "A", 600.0, -10.0),
            (0, "mixed", None, 400.0, -10.0),
            (0, "clean", "A", 300.0, 1.0),
        ]
    )
    result = attribute_discrepancies(sheet, DATA_START).set_index("worker_id")
    assert result.loc["A", "delta"] == pytest.approx(1.0)
    assert result.loc["A", "cash"] == pytest.approx(300.0)


def test_zero_cash_drawers_split_equally_and_empty_sheets_are_handled() -> None:
    sheet = _sheet([(0, "z", "A", 0.0, 3.0), (0, "z", "B", 0.0, 3.0)])
    result = attribute_discrepancies(sheet, DATA_START).set_index("worker_id")
    assert result["delta"].tolist() == [1.5, 1.5]
    assert attribute_discrepancies(sheet.iloc[0:0], DATA_START).empty


def _single_worker_fit(deltas: dict[int, float], as_of: int = 11, cash: float = 1000.0):
    sheet = _sheet([(week, f"d{week}", "A", cash, delta) for week, delta in deltas.items()])
    weeks = {week: (50, 0, 0) for week in deltas}
    return fit_accuracy(_features({"A": weeks}), sheet, DATA_START, as_of, CONFIG, COMBINE)


def test_error_rate_matches_the_hand_computed_value() -> None:
    fit = _single_worker_fit({9: 0.5, 10: 3.0, 11: -4.0})
    weights = np.array([0.5 ** (2 / 4), 0.5 ** (1 / 4), 1.0])
    excess = np.array([0.0, 2.0, 3.0])
    expected = (weights * excess).sum() / (weights * 1000.0).sum()
    assert fit.table.loc["A", "error_rate_raw"] == pytest.approx(expected)
    assert expected == pytest.approx(4.6818 / 2548.0, rel=1e-3)


def test_tolerance_is_applied_to_the_weekly_total_and_both_signs_count() -> None:
    quiet = _single_worker_fit({9: 0.9, 10: -1.0, 11: 0.4})
    assert quiet.table.loc["A", "error_rate_raw"] == pytest.approx(0.0)
    surplus = _single_worker_fit({11: 5.0})
    shortage = _single_worker_fit({11: -5.0})
    assert surplus.table.loc["A", "error_rate_raw"] == pytest.approx(
        shortage.table.loc["A", "error_rate_raw"]
    )
    assert surplus.table.loc["A", "error_rate_raw"] == pytest.approx(4.0 / 1000.0)


def test_old_mistakes_fade_with_the_half_life_and_leave_the_window() -> None:
    recent = _single_worker_fit({week: (6.0 if week == 10 else 0.0) for week in range(12)})
    older = _single_worker_fit({week: (6.0 if week == 2 else 0.0) for week in range(12)})
    ratio = recent.table.loc["A", "error_rate_raw"] / older.table.loc["A", "error_rate_raw"]
    assert ratio == pytest.approx(0.5 ** ((1 - 9) / 4))
    outside = _single_worker_fit(
        {week: (6.0 if week == 0 else 0.0) for week in range(12)}, as_of=12
    )
    assert outside.table.loc["A", "error_rate_raw"] == pytest.approx(0.0)


def test_weekly_records_window_and_weights() -> None:
    sheet = _sheet([(week, f"d{week}", "A", 1000.0, 0.0) for week in range(14)])
    attributed = attribute_discrepancies(sheet, DATA_START)
    features = _features({"A": {week: (10, 0, 0) for week in range(14)}})
    weekly = weekly_records(features, attributed, as_of_week=13, config=CONFIG)
    assert weekly["weeks_ago"].tolist() == list(range(11, -1, -1))
    assert weekly["weight"].iloc[-1] == 1.0
    assert weekly["weight"].iloc[-5] == pytest.approx(0.5)


COHORT_SIZE = 24
BAD_WEEK_EUROS = -9.0


def _cohort_with_one_bad_week() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(5)
    rows = []
    spec: dict[str, dict[int, tuple[int, int, int]]] = {}

    def weekly_discrepancy() -> float:
        if rng.random() < 0.15:
            return float(rng.choice([-1.0, 1.0]) * 3.0 * np.exp(0.8 * rng.standard_normal()))
        return float(rng.normal(0, 0.3))

    for index in range(COHORT_SIZE):
        worker = f"W{index:02d}"
        spec[worker] = {week: (100, 2, 1) for week in range(12)}
        for week in range(12):
            rows.append((week, f"{worker}-{week}", worker, 5000.0, weekly_discrepancy()))
    spec["X"] = {week: (100, 2, 1) for week in range(12)}
    for week in range(12):
        rows.append((week, f"X-{week}", "X", 5000.0, BAD_WEEK_EUROS if week == 8 else 0.1))
    spec["Y"] = {11: (100, 2, 1)}
    rows.append((11, "Y-11", "Y", 5000.0, BAD_WEEK_EUROS))
    return _features(spec), _sheet(rows)


def test_a_single_bad_week_is_shrunk_not_condemned() -> None:
    features, sheet = _cohort_with_one_bad_week()
    rated = [f"W{index:02d}" for index in range(COHORT_SIZE)]
    uncapped = CombineConfig(z_cap=1e9)
    shrunk = fit_accuracy(features, sheet, DATA_START, 11, CONFIG, uncapped, rated=rated).table
    raw_config = AccuracyConfig(shrinkage_k_weeks=0.0)
    unshrunk = fit_accuracy(
        features, sheet, DATA_START, 11, raw_config, uncapped, rated=rated
    ).table

    center = float(np.mean(unshrunk.loc[rated, "error_rate_raw"]))
    deviation_raw = unshrunk.loc["Y", "error_rate_raw"] - center
    deviation_shrunk = shrunk.loc["Y", "error_rate_shrunk"] - center
    assert shrunk.loc["Y", "effective_weeks"] == pytest.approx(1.0)
    assert deviation_shrunk == pytest.approx(
        deviation_raw * (1.0 / (1.0 + CONFIG.shrinkage_k_weeks))
    )
    assert shrunk.loc["Y", "z_error"] < 0.6 * unshrunk.loc["Y", "z_error"]
    assert shrunk.loc["X", "z_error"] == pytest.approx(unshrunk.loc["X", "z_error"])
    assert shrunk.loc["Y", "accuracy_score"] > unshrunk.loc["Y", "accuracy_score"]


def test_the_z_cap_bounds_the_penalty_of_any_single_observation() -> None:
    features, sheet = _cohort_with_one_bad_week()
    rated = [f"W{index:02d}" for index in range(COHORT_SIZE)]
    table = fit_accuracy(features, sheet, DATA_START, 11, CONFIG, COMBINE, rated=rated).table
    cap = COMBINE.z_cap
    assert table["z_error"].max() <= cap + 1e-12
    assert table["z_void"].abs().max() <= cap + 1e-12
    floor = COMBINE.score_center - COMBINE.score_scale * cap
    assert table["accuracy_score"].min() >= floor - 1e-9
    assert table["accuracy_score"].max() <= COMBINE.score_center + COMBINE.score_scale * cap + 1e-9
    huge = _single_worker_fit({11: -500.0})
    assert huge.table.loc["A", "accuracy_score"] >= floor - 1e-9


def test_void_rate_excludes_non_error_voids_and_is_per_hundred_transactions() -> None:
    features = _features({"A": {11: (200, 4, 6)}, "B": {11: (200, 0, 0)}})
    sheet = _sheet([(11, "a", "A", 1000.0, 0.0), (11, "b", "B", 1000.0, 0.0)])
    table = fit_accuracy(features, sheet, DATA_START, 11, CONFIG, COMBINE).table
    assert table.loc["A", "void_rate_raw"] == pytest.approx(2.0)
    assert table.loc["A", "manual_rate_raw"] == pytest.approx(3.0)
    assert table.loc["B", "void_rate_raw"] == pytest.approx(0.0)


def test_workers_without_cash_are_scored_on_voids_alone() -> None:
    features = _features(
        {
            "card": {week: (100, 1, 0) for week in range(12)},
            "A": {week: (100, 2, 0) for week in range(12)},
            "B": {week: (100, 4, 0) for week in range(12)},
        }
    )
    sheet = _sheet(
        [
            (week, f"{worker}-{week}", worker, 3000.0, 0.0)
            for week in range(12)
            for worker in ("A", "B")
        ]
    )
    table = fit_accuracy(features, sheet, DATA_START, 11, CONFIG, COMBINE).table
    assert not table.loc["card", "error_defined"]
    assert table.loc["A", "error_defined"]
    expected = 50.0 - 15.0 * table.loc["card", "z_void"]
    assert table.loc["card", "accuracy_score"] == pytest.approx(expected)
    assert table.loc["card", "accuracy_score"] > table.loc["B", "accuracy_score"]


def test_accuracy_formula_uses_the_documented_weights() -> None:
    parameters = AccuracyParameters(
        error_center=0.0,
        error_mean=0.0,
        error_spread=1.0,
        void_center=2.0,
        void_mean=2.0,
        void_spread=1.0,
        manual_center=0.0,
        manual_mean=0.0,
        manual_spread=1.0,
    )
    result = accuracy_scores(
        numerator=np.array([1e6]),
        denominator=np.array([1e6]),
        effective_weeks=np.array([1e12]),
        void_total=np.array([1e10]),
        effective_transactions=np.array([1e12]),
        parameters=parameters,
        config=CONFIG,
        combine=COMBINE,
    )
    assert result["z_error"][0] == pytest.approx(1.0, abs=1e-6)
    assert result["z_void"][0] == pytest.approx(-1.0, abs=1e-6)
    assert result["accuracy_score"][0] == pytest.approx(50.0 - 15.0 * (0.6 * 1.0 + 0.4 * -1.0))


def test_review_flags_need_speed_and_a_high_error_signal_together() -> None:
    speed = pd.Series([1.5, 0.5, 2.0, 1.8, 1.0], index=list("abcde"))
    voids = pd.Series([2.0, 3.0, 0.5, 2.0, 0.2], index=list("abcde"))
    manual = pd.Series([0.0, 3.0, 2.0, 2.0, 0.0], index=list("abcde"))
    flags = review_flags(speed, voids, manual, ReviewConfig())
    assert flags["a"] is not None and "void rate" in flags["a"]
    assert flags["b"] is None
    assert flags["c"] is not None and "manual-entry" in flags["c"]
    assert "void rate" in flags["d"] and "manual-entry" in flags["d"]
    assert flags["e"] is None
    assert all("human" in text for text in flags.dropna())


def test_joint_bootstrap_is_seeded_and_centred() -> None:
    features, sheet = _cohort_with_one_bad_week()
    fit = fit_accuracy(features, sheet, DATA_START, 11, CONFIG, COMBINE)
    workers = list(fit.table.index)
    reference = np.array([worker.startswith("W") for worker in workers])
    first = fit.joint_bootstrap(workers, reference, 3, 200)
    assert np.array_equal(first, fit.joint_bootstrap(workers, reference, 3, 200))
    assert first.shape == (200, len(workers))
    column = workers.index("W03")
    assert abs(first[:, column].mean() - fit.table.loc["W03", "accuracy_score"]) < 8.0
    assert first[:, column].std() > 0.0
    unknown = fit.joint_bootstrap(["unknown"], np.array([False]), 3, 5)
    assert unknown.tolist() == [[50.0]] * 5
    assert set(fit.weekly["Y"].weight.tolist()) == {1.0}


def test_accuracy_tracks_hidden_error_propensity_on_the_simulated_store(
    default_store: SimulatedStore, default_config: Config
) -> None:
    features = build_features(default_store.data, default_config)
    fit = fit_accuracy(
        features.frame,
        default_store.data.cash_sheet,
        features.data_start,
        as_of_week_index(default_store.data),
        default_config.engine.accuracy,
        default_config.engine.combine,
    )
    truth = default_store.ground_truth.set_index("worker_id")["error_propensity"]
    table = fit.table
    assert stats.spearmanr(table["accuracy_score"], truth[table.index]).statistic < -0.5
    assert stats.spearmanr(table["void_rate_shrunk"], truth[table.index]).statistic > 0.7
    assert stats.spearmanr(table["error_rate_shrunk"], truth[table.index]).statistic > 0.25
    assert table["accuracy_score"].mean() == pytest.approx(50.0, abs=2.0)
