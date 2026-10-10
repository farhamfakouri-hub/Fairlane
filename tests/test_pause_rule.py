from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from factories import TransactionSpec, events_from_sequence, store_from_specs
from ree.config import Config, PauseRuleConfig
from ree.domain.events import GAP_EVENT_TYPES
from ree.engine.features import build_features
from ree.engine.pause_rule import (
    GapTable,
    PauseThresholds,
    capped_duration,
    extract_gaps,
    hard_stop_duration,
    is_interrupted,
    transaction_timing,
    tune_thresholds,
)

DEFAULT_THRESHOLDS = PauseThresholds(t_idle_seconds=15.0, t_stop_seconds=60.0)
HARD_STOP = 5.0


def _timing(events: pd.DataFrame, thresholds: PauseThresholds = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    return transaction_timing(extract_gaps(events), thresholds, HARD_STOP).set_index(
        "transaction_id"
    )


def test_scalar_rule_on_a_hand_computed_basket() -> None:
    gaps = [3.0, 20.0, 3.0, 70.0]
    assert capped_duration(gaps, 15.0) == 3 + 15 + 3 + 15
    assert sum(gaps) == 96.0
    assert is_interrupted(gaps, 60.0)
    assert not is_interrupted([3.0, 20.0, 60.0], 60.0)
    assert hard_stop_duration(gaps, 5.0) == 6.0


def test_gap_extraction_on_a_hand_computed_transaction() -> None:
    events = events_from_sequence(
        1,
        [
            ("transaction_started", 0.0),
            ("item_scanned", 5.0),
            ("item_scanned", 8.0),
            ("tag_removed", 15.0),
            ("loyalty_scanned", 40.0),
            ("payment_pressed", 45.0),
        ],
    )
    timing = _timing(events)
    row = timing.loc[1]
    assert row["n_gaps"] == 4
    assert row["duration_raw"] == pytest.approx(40.0)
    assert row["duration_capped"] == pytest.approx(3 + 7 + 15 + 5)
    assert row["duration_hard_stop"] == pytest.approx(3 + 5)
    assert row["max_gap"] == pytest.approx(25.0)
    assert not row["interrupted_by_gap"]


def test_time_before_the_first_scan_and_after_payment_is_never_counted() -> None:
    events = events_from_sequence(
        1,
        [
            ("transaction_started", 0.0),
            ("loyalty_scanned", 2.0),
            ("item_scanned", 40.0),
            ("item_scanned", 43.0),
            ("payment_pressed", 50.0),
        ],
    )
    row = _timing(events).loc[1]
    assert row["duration_raw"] == pytest.approx(10.0)
    assert row["n_gaps"] == 2


def test_exception_events_do_not_split_the_gap_sequence() -> None:
    plain = events_from_sequence(
        1,
        [
            ("transaction_started", 0.0),
            ("item_scanned", 5.0),
            ("item_scanned", 9.0),
            ("payment_pressed", 12.0),
        ],
    )
    with_exceptions = events_from_sequence(
        1,
        [
            ("transaction_started", 0.0),
            ("item_scanned", 5.0),
            ("void", 6.0),
            ("manual_entry", 7.0),
            ("price_check", 8.0),
            ("item_scanned", 9.0),
            ("payment_pressed", 12.0),
        ],
        {2: "cashier_error"},
    )
    assert _timing(plain).loc[1, "duration_raw"] == _timing(with_exceptions).loc[1, "duration_raw"]


def test_incomplete_transactions_have_no_timing() -> None:
    no_payment = events_from_sequence(
        1, [("transaction_started", 0.0), ("item_scanned", 5.0), ("item_scanned", 9.0)]
    )
    no_items = events_from_sequence(2, [("transaction_started", 0.0), ("payment_pressed", 9.0)])
    complete = events_from_sequence(
        3, [("transaction_started", 0.0), ("item_scanned", 5.0), ("payment_pressed", 9.0)]
    )
    timing = _timing(pd.concat([no_payment, no_items, complete], ignore_index=True))
    assert timing.loc[[1, 2], "n_gaps"].tolist() == [0, 0]
    assert timing.loc[3, "n_gaps"] == 1


def test_extraction_does_not_depend_on_row_order() -> None:
    events = pd.concat(
        [
            events_from_sequence(
                t,
                [
                    ("transaction_started", 0.0),
                    ("item_scanned", 4.0),
                    ("tag_removed", 12.0 + t),
                    ("payment_pressed", 40.0 + 3 * t),
                ],
            )
            for t in range(1, 6)
        ],
        ignore_index=True,
    )
    shuffled = events.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(_timing(events), _timing(shuffled))


def test_threshold_tuning_ignores_interrupted_baskets() -> None:
    rng = np.random.default_rng(0)
    typical_gaps = np.exp(rng.normal(np.log(3.0), 0.4, size=10_000))
    typical_index = np.repeat(np.arange(1000), 10)
    interrupted_gaps = np.full(30, 120.0)
    interrupted_index = np.arange(1000, 1030)
    table = GapTable(
        transaction_ids=np.arange(1030),
        transaction_index=np.concatenate([typical_index, interrupted_index]),
        gap_seconds=np.concatenate([typical_gaps, interrupted_gaps]),
        closing_event_code=np.zeros(10_030, dtype=np.int64),
    )
    config = PauseRuleConfig()
    tuned = tune_thresholds(table, config)
    expected = float(np.quantile(typical_gaps, config.tune_quantile))
    assert tuned.t_idle_seconds == pytest.approx(expected)
    assert 7.0 < tuned.t_idle_seconds < 20.0
    assert tuned.t_stop_seconds == pytest.approx(max(60.0, 3.0 * tuned.t_idle_seconds))


def test_tuning_on_the_simulated_store_places_the_cap_in_the_tail(default_store) -> None:
    table = extract_gaps(default_store.data.events)
    tuned = tune_thresholds(table, PauseRuleConfig())
    assert 15.0 < tuned.t_idle_seconds < 30.0
    assert tuned.t_stop_seconds >= 60.0


gap_values = st.floats(min_value=0.0, max_value=400.0, allow_nan=False, allow_infinity=False)
gap_lists = st.lists(gap_values, min_size=1, max_size=40)
cap_values = st.floats(min_value=0.1, max_value=200.0, allow_nan=False)


@given(gaps=gap_lists, cap=cap_values)
def test_capping_never_increases_the_duration(gaps: list[float], cap: float) -> None:
    assert capped_duration(gaps, cap) <= sum(gaps) + 1e-9
    assert capped_duration(gaps, cap) >= 0.0


@given(gaps=gap_lists, first=cap_values, second=cap_values)
def test_duration_is_monotone_in_the_cap(gaps: list[float], first: float, second: float) -> None:
    low, high = sorted((first, second))
    assert capped_duration(gaps, low) <= capped_duration(gaps, high) + 1e-9


@given(gaps=gap_lists)
def test_a_cap_above_every_gap_changes_nothing(gaps: list[float]) -> None:
    assert capped_duration(gaps, max(gaps) + 1.0) == pytest.approx(sum(gaps))
    assert hard_stop_duration(gaps, max(gaps) + 1.0) == pytest.approx(sum(gaps))


@given(gaps=gap_lists, stop=cap_values)
def test_interruption_flag_matches_the_longest_gap(gaps: list[float], stop: float) -> None:
    assert is_interrupted(gaps, stop) == (max(gaps) > stop)


transaction_strategy = st.tuples(
    st.lists(st.sampled_from(GAP_EVENT_TYPES), min_size=0, max_size=12),
    st.lists(gap_values, min_size=13, max_size=13),
)


def _spec_from_draw(kinds: list[str], gaps: list[float]) -> tuple[TransactionSpec, list[float]]:
    used = gaps[: len(kinds) + 1]
    offset = 5.0
    events = [("item_scanned", offset)]
    for kind, gap in zip(kinds, used[:-1], strict=True):
        offset += gap
        events.append((kind, offset))
    offset += used[-1]
    events.append(("payment_pressed", offset))
    return TransactionSpec(events=events), used


@settings(max_examples=60, deadline=None)
@given(draws=st.lists(transaction_strategy, min_size=1, max_size=6))
def test_vectorised_rule_matches_the_scalar_rule(
    draws: list[tuple[list[str], list[float]]],
) -> None:
    specs_and_gaps = [_spec_from_draw(kinds, gaps) for kinds, gaps in draws]
    data = store_from_specs([spec for spec, _ in specs_and_gaps])
    timing = _timing(data.events)
    for index, (_, gaps) in enumerate(specs_and_gaps, start=1):
        row = timing.loc[index]
        assert row["duration_raw"] == pytest.approx(sum(gaps), abs=1e-6)
        assert row["duration_capped"] == pytest.approx(capped_duration(gaps, 15.0), abs=1e-6)
        assert row["duration_hard_stop"] == pytest.approx(hard_stop_duration(gaps, 5.0), abs=1e-6)
        assert row["max_gap"] == pytest.approx(max(gaps), abs=1e-6)
        assert bool(row["interrupted_by_gap"]) == is_interrupted(gaps, 60.0)
        assert row["duration_capped"] <= row["duration_raw"] + 1e-9


@settings(max_examples=40, deadline=None)
@given(draws=st.lists(transaction_strategy, min_size=1, max_size=5))
def test_interrupted_transactions_leave_speed_but_stay_for_accuracy(
    draws: list[tuple[list[str], list[float]]],
) -> None:
    specs_and_gaps = [_spec_from_draw(kinds, gaps) for kinds, gaps in draws]
    data = store_from_specs([spec for spec, _ in specs_and_gaps])
    frame = build_features(data, Config()).frame.set_index("transaction_id")
    assert len(frame) == len(specs_and_gaps)
    for index, (_, gaps) in enumerate(specs_and_gaps, start=1):
        row = frame.loc[index]
        assert row["valid_for_accuracy"]
        if is_interrupted(gaps, 60.0):
            assert row["interrupted"]
            assert not row["valid_for_speed"]
        else:
            assert row["valid_for_speed"] == (row["duration_capped"] > 0)
