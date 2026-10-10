from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factories import TransactionSpec, store_from_specs
from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.engine.features import (
    as_of_week_index,
    build_features,
    size_bins,
    weeks_of_data,
)
from ree.engine.pause_rule import PauseThresholds

SCAN_CYCLE = [("item_scanned", 5.0), ("item_scanned", 8.0), ("item_hand_keyed", 11.0)]


def _basket(extra: list[tuple[str, float]], end: float, **kwargs) -> TransactionSpec:
    return TransactionSpec(events=[*SCAN_CYCLE, *extra, ("payment_pressed", end)], **kwargs)


def test_counts_bins_and_contexts_for_a_hand_built_store() -> None:
    specs = [
        _basket(
            [("tag_removed", 18.0), ("loyalty_scanned", 26.0), ("voucher_applied", 30.0)],
            40.0,
            register=2,
            started_at="2025-11-25 09:00:00",
        ),
        _basket([], 25.0, register=4, started_at="2025-11-25 11:00:00", worker_id="W002"),
        _basket([], 25.0, register=9, started_at="2025-11-29 08:45:00"),
        _basket([], 25.0, register=1, started_at="2025-11-30 11:30:00"),
    ]
    frame = build_features(store_from_specs(specs), Config()).frame.set_index("transaction_id")
    first = frame.loc[1]
    assert (first["n_scanned"], first["n_hand_keyed"], first["n_items"]) == (2, 1, 3)
    assert (first["n_tagged"], first["has_loyalty"], first["n_vouchers"]) == (1, 1, 1)
    assert first["basket_class"] == "small"
    assert first["size_bin"] == 0
    assert first["lane_group"] == "zone_a"
    assert first["period"] == "off"
    assert first["worker_id"] == "W001"
    assert first["week_index"] == 0
    assert first["duration_raw"] == pytest.approx(35.0)
    assert first["duration_capped"] == pytest.approx(3 + 3 + 7 + 8 + 4 + 10)
    assert frame.loc[2, "lane_group"] == "card_only"
    assert frame.loc[2, "period"] == "peak"
    assert frame.loc[2, "worker_id"] == "W002"
    assert frame.loc[3, "lane_group"] == "zone_b"
    assert frame.loc[3, "period"] == "peak"
    assert frame.loc[4, "lane_group"] == "priority"
    assert frame.loc[4, "period"] == "peak"


def test_capped_duration_matches_the_hand_computation() -> None:
    spec = _basket([("tag_removed", 18.0), ("loyalty_scanned", 44.0)], 49.0)
    row = build_features(store_from_specs([spec]), Config()).frame.iloc[0]
    expected_gaps = [3.0, 3.0, 7.0, 26.0, 5.0]
    assert row["duration_raw"] == pytest.approx(sum(expected_gaps))
    assert row["duration_capped"] == pytest.approx(3 + 3 + 7 + 15 + 5)
    assert row["max_gap"] == pytest.approx(26.0)
    assert row["n_gaps"] == 5


def test_interrupted_transactions_are_flagged_but_kept() -> None:
    long_gap = _basket([("loyalty_scanned", 90.0)], 95.0)
    explicit = TransactionSpec(
        events=[
            ("item_scanned", 5.0),
            ("price_check", 6.0),
            ("item_scanned", 9.0),
            ("payment_pressed", 14.0),
        ]
    )
    clean = _basket([], 25.0)
    frame = build_features(store_from_specs([long_gap, explicit, clean]), Config()).frame.set_index(
        "transaction_id"
    )
    assert frame.loc[1, "interrupted_by_gap"] and not frame.loc[1, "explicit_interruption"]
    assert frame.loc[2, "explicit_interruption"] and not frame.loc[2, "interrupted_by_gap"]
    assert frame.loc[[1, 2], "interrupted"].all()
    assert not frame.loc[[1, 2], "valid_for_speed"].any()
    assert frame["valid_for_accuracy"].all()
    assert frame.loc[3, "valid_for_speed"]


def test_voids_and_manual_entries_are_exceptions_with_reasons() -> None:
    specs = [
        TransactionSpec(
            events=[
                ("item_scanned", 5.0),
                ("void", 6.0),
                ("void", 7.0),
                ("void", 7.5),
                ("item_scanned", 9.0),
                ("payment_pressed", 14.0),
            ],
            details={1: "cashier_error", 2: "customer_change", 3: "unknown"},
        ),
        TransactionSpec(
            events=[("item_scanned", 5.0), ("manual_entry", 6.0), ("payment_pressed", 14.0)]
        ),
    ]
    frame = build_features(store_from_specs(specs), Config()).frame.set_index("transaction_id")
    assert frame.loc[1, "n_voids"] == 3
    assert frame.loc[1, "n_void_excluded"] == 1
    assert frame.loc[1, "n_void_errors"] == 2
    assert frame.loc[2, "n_manual_entries"] == 1
    assert frame["has_exception"].all()
    assert not frame["valid_for_speed"].any()
    assert frame["valid_for_accuracy"].all()


def test_thresholds_can_be_overridden_or_tuned() -> None:
    spec = _basket([("loyalty_scanned", 44.0)], 49.0)
    data = store_from_specs([spec])
    loose = build_features(data, Config(), PauseThresholds(40.0, 200.0)).frame.iloc[0]
    assert loose["duration_capped"] == pytest.approx(loose["duration_raw"])
    strict = build_features(data, Config(), PauseThresholds(10.0, 200.0)).frame.iloc[0]
    assert strict["duration_capped"] < loose["duration_capped"]
    tuned_config = Config().with_overrides({"engine.pause.tune_from_data": True})
    tuned = build_features(data, tuned_config)
    assert tuned.thresholds.t_stop_seconds >= 60.0


def test_size_bins_are_five_items_wide_and_open_ended() -> None:
    items = np.array([1, 5, 6, 10, 11, 79, 80, 81, 400])
    assert size_bins(items, 5, 80).tolist() == [0, 0, 1, 1, 2, 15, 15, 15, 15]
    assert size_bins(items, 10, 80).tolist() == [0, 0, 0, 0, 1, 7, 7, 7, 7]


def test_features_on_the_simulated_store_agree_with_the_latent_truth(
    default_store: SimulatedStore, default_config: Config
) -> None:
    features = build_features(default_store.data, default_config)
    frame = features.frame.set_index("transaction_id")
    latent = default_store.latent.set_index("transaction_id").loc[frame.index]
    assert len(frame) == len(default_store.data.transactions)
    assert (frame["n_items"] == latent["n_items"]).all()
    assert (frame["n_tagged"] == latent["n_tagged"]).all()
    assert (frame["n_hand_keyed"] == latent["n_hand_keyed"]).all()
    assert (frame["has_loyalty"] == latent["has_loyalty"].astype(int)).all()
    assert np.allclose(frame["duration_raw"], latent["true_duration_seconds"], atol=1e-6)
    assert (frame["duration_capped"] <= frame["duration_raw"] + 1e-9).all()
    assert (frame["interrupted_by_gap"][latent["interruption"]]).all()
    assert frame["valid_for_speed"].mean() > 0.9
    assert frame["week_index"].between(0, default_config.simulator.n_weeks - 1).all()
    assert frame["valid_for_accuracy"].all()
    peak_truth = latent["is_peak"].to_numpy()
    assert ((frame["period"] == "peak").to_numpy() == peak_truth).all()


def test_weeks_of_data_and_as_of_week(default_store: SimulatedStore) -> None:
    data = default_store.data
    weeks = weeks_of_data(data)
    assert weeks.max() == default_store.config.simulator.n_weeks
    assert weeks.min() >= 1
    assert as_of_week_index(data) == default_store.config.simulator.n_weeks - 1
    newcomers = default_store.ground_truth.set_index("worker_id")["is_newcomer"]
    assert weeks[newcomers[newcomers].index].max() < weeks.max() + 1


def test_empty_and_unsorted_inputs_are_handled() -> None:
    specs = [_basket([], 25.0), _basket([], 30.0)]
    data = store_from_specs(specs)
    data.events = data.events.sample(frac=1.0, random_state=1).reset_index(drop=True)
    frame = build_features(data, Config()).frame
    assert frame["transaction_id"].tolist() == [1, 2]
    assert pd.api.types.is_bool_dtype(frame["valid_for_speed"])
