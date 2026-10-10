from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ree.config import CombineConfig, IntervalConfig, TierConfig
from ree.engine.combine import accuracy_weight, combined_score, speed_weight
from ree.engine.intervals import accuracy_draws_with_fallback, combined_interval
from ree.engine.resampling import (
    session_codes,
    session_counts,
    weighted_median_rows,
)
from ree.engine.speed_v2 import PRICE_TERMS, design_matrix, fit_v2
from ree.engine.tiers import UNRATED, assign_tiers, interval_confidence

COMBINE = CombineConfig()


def test_combined_score_uses_the_documented_context_weights() -> None:
    assert speed_weight(COMBINE, "peak") == 0.65
    assert accuracy_weight(COMBINE, "peak") == pytest.approx(0.35)
    assert speed_weight(COMBINE, "off") == 0.5
    assert combined_score(60.0, 40.0, COMBINE, "peak") == pytest.approx(0.65 * 60 + 0.35 * 40)
    assert combined_score(60.0, 40.0, COMBINE, "off") == pytest.approx(50.0)
    both = combined_score(np.array([60.0, 50.0]), np.array([40.0, 50.0]), COMBINE, "peak")
    assert both.tolist() == pytest.approx([53.0, 50.0])


def test_weighted_median_matches_numpy_for_unit_weights() -> None:
    rng = np.random.default_rng(1)
    for size in (7, 8, 101, 400):
        values = np.sort(rng.normal(size=size))
        result = weighted_median_rows(values, np.ones((1, size)))
        assert result[0] == pytest.approx(np.median(values))


def test_weighted_median_equals_the_median_of_repeated_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert weighted_median_rows(values, np.array([[1, 0, 0, 3]]))[0] == 4.0
    assert weighted_median_rows(values, np.array([[2, 2, 0, 0]]))[0] == 1.5
    assert np.isnan(weighted_median_rows(values, np.zeros((1, 4))))[0]


def test_session_counts_and_codes() -> None:
    counts = session_counts(np.random.default_rng(2), 6, 50)
    assert counts.shape == (50, 6)
    assert (counts.sum(axis=1) == 6).all()
    workers = np.array(["A", "A", "A", "B", "B"])
    periods = np.array(["off"] * 5)
    sessions = np.array([10, 10, 20, 30, 30])
    local, total = session_codes(workers, periods, sessions)
    assert local.tolist() == [0, 0, 1, 0, 0]
    assert total.tolist() == [2, 2, 2, 1, 1]


def _synthetic_context(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    n_items = rng.integers(1, 40, size=rows)
    return pd.DataFrame(
        {
            "n_items": n_items,
            "n_hand_keyed": rng.binomial(n_items, 0.15),
            "n_tagged": rng.binomial(n_items, 0.07),
            "has_loyalty": rng.integers(0, 2, size=rows),
            "n_vouchers": rng.integers(0, 3, size=rows),
        }
    )


def test_session_clustering_widens_the_speed_bootstrap() -> None:
    rng = np.random.default_rng(6)
    sessions_per_worker = 40
    rows_per_session = 60
    frames = []
    for worker in range(6):
        for session in range(sessions_per_worker):
            context = _synthetic_context(rng, rows_per_session)
            effect = np.exp(rng.normal(0, 0.12))
            base = design_matrix(context) @ np.array([25.0, 2.9, 0.7, 6.0, 8.0, 4.0])
            context["duration"] = base * effect * np.exp(rng.normal(0, 0.05, rows_per_session))
            context["worker_id"] = f"W{worker}"
            context["session_id"] = worker * 1000 + session
            context["lane_group"] = "zone_a"
            context["period"] = "off"
            context["basket_class"] = "small"
            frames.append(context)
    frame = pd.concat(frames, ignore_index=True)
    from ree.config import SpeedConfig

    clustered = fit_v2(frame, frame["duration"].to_numpy(), SpeedConfig())
    flat = fit_v2(frame.drop(columns=["session_id"]), frame["duration"].to_numpy(), SpeedConfig())
    clustered_se = clustered.bootstrap("W0", "off", np.random.default_rng(1), 400).std()
    flat_se = flat.bootstrap("W0", "off", np.random.default_rng(1), 400).std()
    assert clustered_se > 2.0 * flat_se
    assert set(PRICE_TERMS) == set(clustered.price_list["term"])


def _tier_frame(count: int = 20, half_width: float = 1.0, weeks: int = 12):
    scores = pd.Series(np.arange(1.0, count + 1.0), index=[f"W{i:02d}" for i in range(count)])
    return (
        scores,
        scores - half_width,
        scores + half_width,
        pd.Series(False, index=scores.index),
        pd.Series(weeks, index=scores.index),
    )


def test_percentile_tiers_without_the_conservative_rule() -> None:
    scores, lower, upper, provisional, weeks = _tier_frame()
    config = TierConfig(conservative=False)
    tiers = assign_tiers(scores, lower, upper, provisional, weeks, config, width_limit=20.0)
    assert (tiers == "A").sum() == 4 and (tiers == "C").sum() == 4 and (tiers == "B").sum() == 12
    assert tiers.iloc[-1] == "A" and tiers.iloc[0] == "C"


def test_tiers_are_withheld_for_wide_intervals_and_thin_history() -> None:
    scores, lower, upper, provisional, weeks = _tier_frame()
    upper = upper.copy()
    upper.iloc[-1] = scores.iloc[-1] + 50.0
    weeks = weeks.copy()
    weeks.iloc[0] = 2
    tiers = assign_tiers(
        scores, lower, upper, provisional, weeks, TierConfig(conservative=False), 20.0
    )
    assert tiers.iloc[-1] == UNRATED
    assert tiers.iloc[0] == UNRATED
    assert (tiers == UNRATED).sum() == 2


def test_provisional_workers_are_tiered_only_among_provisional_peers() -> None:
    scores, lower, upper, provisional, weeks = _tier_frame(count=24)
    provisional = provisional.copy()
    provisional.iloc[:3] = True
    config = TierConfig(conservative=False)
    tiers = assign_tiers(scores, lower, upper, provisional, weeks, config, 20.0)
    assert set(tiers.iloc[:3]) <= {"A", "B", "C"}
    assert tiers.iloc[:3].tolist() == ["B", "B", "A"]
    provisional.iloc[:2] = True
    provisional.iloc[2] = False
    only_two = assign_tiers(scores, lower, upper, provisional, weeks, config, 20.0)
    assert (only_two.iloc[:2] == UNRATED).all()


def test_conservative_tiers_need_the_interval_to_clear_the_cutoff() -> None:
    scores, lower, upper, provisional, weeks = _tier_frame(half_width=0.4)
    lenient = assign_tiers(
        scores, lower, upper, provisional, weeks, TierConfig(conservative=False), 20.0
    )
    narrow = assign_tiers(scores, lower, upper, provisional, weeks, TierConfig(), 20.0)
    assert (narrow == lenient).all()
    wide_lower = scores - 2.5
    wide_upper = scores + 2.5
    wide = assign_tiers(scores, wide_lower, wide_upper, provisional, weeks, TierConfig(), 20.0)
    assert (wide == "A").sum() < (lenient == "A").sum()
    assert wide.iloc[-1] == "A"
    assert wide.iloc[-4] == "B"
    assert (wide == "C").sum() < (lenient == "C").sum()


def test_interval_confidence_scales_with_the_width_limit() -> None:
    assert interval_confidence(0.0, 20.0) == 1.0
    assert interval_confidence(10.0, 20.0) == pytest.approx(0.5)
    assert interval_confidence(40.0, 20.0) == 0.0


def test_combined_interval_always_contains_the_point_score() -> None:
    rng = np.random.default_rng(3)
    speed = 55.0 + rng.normal(0, 1.0, 300)
    accuracy = 40.0 + rng.normal(0, 4.0, 300)
    point = 80.0
    interval = combined_interval(speed, accuracy, point, IntervalConfig(), COMBINE, "peak")
    assert interval.lower <= point <= interval.upper
    inside = combined_interval(speed, accuracy, 50.0, IntervalConfig(), COMBINE, "peak")
    assert inside.lower < 50.0 < inside.upper
    assert inside.width > 0


def test_accuracy_draws_fall_back_to_the_cohort_spread_for_thin_history() -> None:
    draws = np.full(200, 33.0)
    cohort = np.linspace(30.0, 70.0, 25)
    config = IntervalConfig()
    kept = accuracy_draws_with_fallback(
        draws, 33.0, 8, cohort, config, COMBINE, np.random.default_rng(1)
    )
    assert np.array_equal(kept, draws)
    widened = accuracy_draws_with_fallback(
        draws, 33.0, 1, cohort, config, COMBINE, np.random.default_rng(1)
    )
    assert abs(widened.mean() - 33.0) < 3.0
    assert widened.std() == pytest.approx(cohort.std(), rel=0.2)
    assert widened.min() >= 50.0 - 15.0 * COMBINE.z_cap
