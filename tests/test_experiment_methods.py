from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.engine.accuracy import attribute_discrepancies
from ree.experiments.methods import (
    ACCURACY_VARIANTS,
    SPEED_METHODS,
    MethodInputs,
    accuracy_scores_by_variant,
    build_method_inputs,
    cash_sheet_variant,
    estimates_by_period,
    naive_items_per_shift,
)
from ree.experiments.metrics import (
    evaluate,
    exposure_leakage,
    fill_missing,
    kendall,
    mean_absolute_rank_error,
    spearman,
    top_k_precision,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=[f"W{i:02d}" for i in range(len(values))], dtype=float)


def test_rank_metrics_on_perfect_and_reversed_orders() -> None:
    truth = _series(list(range(10)))
    assert spearman(truth, truth) == pytest.approx(1.0)
    assert kendall(truth, truth) == pytest.approx(1.0)
    assert top_k_precision(truth, truth, 5) == 1.0
    assert mean_absolute_rank_error(truth, truth) == 0.0
    reversed_estimate = -truth
    assert spearman(reversed_estimate, truth) == pytest.approx(-1.0)
    assert top_k_precision(reversed_estimate, truth, 5) == 0.0
    assert mean_absolute_rank_error(reversed_estimate, truth) == pytest.approx(5.0)


def test_top_k_precision_counts_the_overlap() -> None:
    truth = _series(list(range(10)))
    estimate = truth.copy()
    estimate["W09"], estimate["W00"] = 0.0, 9.0
    assert top_k_precision(estimate, truth, 5) == pytest.approx(0.8)


def test_missing_estimates_are_placed_at_the_median() -> None:
    estimate = _series([1.0, 2.0, np.nan, 4.0, 5.0])
    filled = fill_missing(estimate)
    assert filled["W02"] == 3.0
    truth = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(estimate, truth) == pytest.approx(1.0)
    assert np.isnan(spearman(_series([1.0, 1.0, 1.0, 1.0]), _series([1.0, 2.0, 3.0, 4.0])))


def test_evaluate_returns_the_four_rank_metrics() -> None:
    truth = _series(list(np.linspace(0, 1, 12)))
    metrics = evaluate(truth + 0.01, truth)
    assert set(metrics) == {"spearman", "kendall", "top_k_precision", "mean_abs_rank_error"}


def test_exposure_leakage_detects_dependence_on_assignment() -> None:
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.normal(size=60), index=[f"W{i}" for i in range(60)])
    exposure = pd.Series(rng.normal(size=60), index=truth.index)
    leaky = truth + 0.8 * exposure
    clean = truth + 0.05 * pd.Series(rng.normal(size=60), index=truth.index)
    assert exposure_leakage(leaky, truth, exposure) > 0.5
    assert abs(exposure_leakage(clean, truth, exposure)) < 0.15


def test_naive_baseline_is_items_per_logged_in_hour() -> None:
    frame = pd.DataFrame(
        {
            "worker_id": ["A", "A", "A", "B"],
            "period": ["peak", "peak", "off", "peak"],
            "n_items": [30, 70, 20, 50],
        }
    )
    hours = pd.DataFrame(
        {
            "worker_id": ["A", "B"],
            "week_index": [0, 0],
            "peak_hours": [4.0, 5.0],
            "off_hours": [2.0, 0.0],
        }
    )
    inputs = MethodInputs(frame, hours, pd.DataFrame(), pd.Timestamp("2025-11-24"), 0, Config())
    estimates = naive_items_per_shift(inputs).set_index(["worker_id", "period"])["estimate"]
    assert estimates[("A", "peak")] == pytest.approx(25.0)
    assert estimates[("A", "off")] == pytest.approx(10.0)
    assert estimates[("B", "peak")] == pytest.approx(10.0)


@pytest.fixture(scope="module")
def default_inputs(default_store: SimulatedStore, default_config: Config) -> MethodInputs:
    return build_method_inputs(default_store.data, default_config)


def test_method_inputs_cover_the_store_and_truncate_by_week(
    default_store: SimulatedStore, default_inputs: MethodInputs
) -> None:
    assert len(default_inputs.frame) == len(default_store.data.transactions)
    sessions = default_store.data.sessions
    hours = (sessions["logout_ts"] - sessions["login_ts"]).dt.total_seconds().sum() / 3600
    table = default_inputs.session_hours
    assert (table["peak_hours"] + table["off_hours"]).sum() == pytest.approx(hours, rel=0.01)
    first_weeks = default_inputs.truncated(2)
    assert first_weeks.frame["week_index"].max() == 1
    assert first_weeks.session_hours["week_index"].max() == 1
    assert first_weeks.cash_sheet["date"].max() < default_inputs.data_start + pd.Timedelta(weeks=2)


def test_every_speed_method_scores_both_periods(default_inputs: MethodInputs) -> None:
    for name, method in SPEED_METHODS.items():
        estimates = method(default_inputs)
        assert set(estimates["period"]) == {"peak", "off"}, name
        by_period = estimates_by_period(estimates)
        assert len(by_period["off"]) >= 20, name
        assert by_period["off"].notna().mean() > 0.9, name


def test_methods_rank_as_the_design_document_predicts(
    default_store: SimulatedStore, default_inputs: MethodInputs
) -> None:
    truth = default_store.ground_truth.set_index("worker_id")
    veterans = truth[~truth["is_newcomer"]]
    scores = {}
    for name, method in SPEED_METHODS.items():
        off = estimates_by_period(method(default_inputs))["off"]
        scores[name] = spearman(off.reindex(veterans.index), veterans["true_speed"])
    assert scores["v1_pause"] > 0.9 and scores["v2_ols_pause"] > 0.9
    assert scores["hard_5s_stop"] < scores["v1_pause"] - 0.3
    assert scores["naive_items_per_shift"] < scores["v1_pause"]


def test_accuracy_attribution_variants_on_a_hand_built_sheet() -> None:
    sheet = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-11-25"] * 3),
            "register": [3, 3, 5],
            "drawer_id": ["shared", "shared", "solo"],
            "worker_id": ["A", "B", "C"],
            "cash_handled": [600.0, 400.0, 500.0],
            "till_discrepancy": [-10.0, -10.0, 2.0],
        }
    )
    start = pd.Timestamp("2025-11-24")
    split = attribute_discrepancies(cash_sheet_variant(sheet, "split_by_cash_share"), start)
    assert split.set_index("worker_id")["delta"].round(6).to_dict() == {
        "A": -6.0,
        "B": -4.0,
        "C": 2.0,
    }
    full = attribute_discrepancies(cash_sheet_variant(sheet, "full_to_each_cashier"), start)
    assert full.set_index("worker_id")["delta"].to_dict() == {"A": -10.0, "B": -10.0, "C": 2.0}
    dropped = attribute_discrepancies(cash_sheet_variant(sheet, "drop_shared_tills"), start)
    assert dropped["worker_id"].tolist() == ["C"]
    with pytest.raises(ValueError):
        cash_sheet_variant(sheet, "bogus")


def test_accuracy_variants_all_track_error_propensity(
    default_store: SimulatedStore, default_inputs: MethodInputs
) -> None:
    truth = default_store.ground_truth.set_index("worker_id")["error_propensity"]
    for variant in ACCURACY_VARIANTS:
        scores = accuracy_scores_by_variant(default_inputs, variant)
        assert spearman(scores, -truth) > 0.5, variant
