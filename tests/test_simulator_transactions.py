from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from helpers import FORBIDDEN_COLUMN_PATTERN, event_durations
from ree.config import Config
from ree.domain.baskets import basket_class_labels
from ree.domain.models import SimulatedStore
from ree.engine.pause_rule import extract_gaps
from ree.simulator.run import simulate_store, simulate_to_directory
from ree.storage.db import list_all_columns, load_store_data

ITEM_TYPES = ["item_scanned", "item_hand_keyed"]
LANES = {"zone_a": [2, 3, 5, 6, 7], "zone_b": list(range(8, 16))}


def _item_counts(store: SimulatedStore) -> pd.Series:
    items = store.data.events[store.data.events["event_type"].isin(ITEM_TYPES)]
    return items.groupby("transaction_id").size()


def test_simulation_is_deterministic_by_seed(small_config: Config) -> None:
    first = simulate_store(small_config)
    second = simulate_store(small_config)
    pd.testing.assert_frame_equal(first.data.events, second.data.events)
    pd.testing.assert_frame_equal(first.data.transactions, second.data.transactions)
    pd.testing.assert_frame_equal(first.data.sessions, second.data.sessions)
    pd.testing.assert_frame_equal(first.data.cash_sheet, second.data.cash_sheet)
    pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)
    other = simulate_store(small_config.with_overrides({"seed": small_config.seed + 1}))
    assert len(other.data.events) != len(first.data.events)
    assert not np.allclose(
        other.ground_truth["true_speed"].to_numpy(), first.ground_truth["true_speed"].to_numpy()
    )


def test_event_streams_are_well_formed(small_store: SimulatedStore) -> None:
    events = small_store.data.events
    transactions = small_store.data.transactions
    assert list(transactions["transaction_id"]) == list(range(1, len(transactions) + 1))
    assert set(events["transaction_id"].unique()) == set(transactions["transaction_id"])
    transaction_ids = events["transaction_id"].to_numpy()
    offsets = events["offset_s"].to_numpy()
    same = transaction_ids[1:] == transaction_ids[:-1]
    assert (np.diff(offsets)[same] >= 0).all()
    assert (np.diff(transaction_ids) >= 0).all()
    counts = (
        events.groupby(["transaction_id", "event_type"], observed=True).size().unstack(fill_value=0)
    )
    assert (counts["transaction_started"] == 1).all()
    assert (counts["payment_pressed"] == 1).all()
    started = events[events["event_type"] == "transaction_started"]
    assert (started["offset_s"] == 0.0).all()
    ordinary = events[~events["event_type"].isin(["transaction_started", "void", "manual_entry"])]
    first_event_type = ordinary.groupby("transaction_id")["event_type"].first()
    assert set(first_event_type.astype(str)) <= set(ITEM_TYPES)
    last_event_type = ordinary.groupby("transaction_id")["event_type"].last()
    assert set(last_event_type.astype(str)) == {"payment_pressed"}


def test_event_gaps_add_up_to_the_latent_duration(small_store: SimulatedStore) -> None:
    durations = event_durations(small_store.data.events)
    latent = small_store.latent.set_index("transaction_id")["true_duration_seconds"]
    assert np.allclose(durations.sort_index().to_numpy(), latent.sort_index().to_numpy(), atol=1e-6)


def test_transactions_fit_inside_their_sessions_without_overlap(
    small_store: SimulatedStore,
) -> None:
    transactions = small_store.data.transactions.merge(
        small_store.latent[["transaction_id", "true_duration_seconds"]], on="transaction_id"
    ).merge(small_store.data.sessions, on="session_id", suffixes=("", "_session"))
    end = transactions["started_at"] + pd.to_timedelta(
        transactions["true_duration_seconds"], unit="s"
    )
    assert (transactions["started_at"] >= transactions["login_ts"]).all()
    assert (end <= transactions["logout_ts"]).all()
    ordered = transactions.sort_values(["session_id", "started_at"])
    next_start = ordered.groupby("session_id")["started_at"].shift(-1)
    gap = (
        next_start
        - (ordered["started_at"] + pd.to_timedelta(ordered["true_duration_seconds"], unit="s"))
    ).dt.total_seconds()
    assert (gap.dropna() > 0).all()


def test_basket_class_shares_match_the_store(default_store: SimulatedStore) -> None:
    labels = pd.Series(basket_class_labels(_item_counts(default_store).to_numpy()))
    shares = labels.value_counts(normalize=True)
    assert abs(shares["small"] - 0.55) < 0.04
    assert abs(shares["medium"] - 0.35) < 0.04
    assert abs(shares["big"] - 0.10) < 0.03


def test_item_mix_matches_the_configuration(default_store: SimulatedStore) -> None:
    latent = default_store.latent
    keyed_share = latent["n_hand_keyed"].sum() / latent["n_items"].sum()
    assert abs(keyed_share - 0.15) < 0.01
    assert abs(latent["has_loyalty"].mean() - 0.60) < 0.01
    assert 0.1 < latent["n_vouchers"].mean() < 0.5
    assert latent["n_items"].min() >= 1
    assert abs(latent["customer_delay"].mean() - 0.04) < 0.005
    assert abs(latent["interruption"].mean() - 0.015) < 0.003


def test_register_zones_get_different_baskets(default_store: SimulatedStore) -> None:
    latent = default_store.latent.merge(
        default_store.data.transactions[["transaction_id", "register"]], on="transaction_id"
    )
    mean_items = latent.groupby("register")["n_items"].mean()
    assert mean_items[4] < mean_items[[2, 3, 5, 6, 7]].min()
    assert mean_items[[2, 3, 5, 6, 7]].mean() > mean_items[list(range(8, 16))].mean()


@pytest.mark.parametrize("lane", ["zone_a", "zone_b"])
def test_store_level_price_list_recovers_the_planted_parameters(
    default_store: SimulatedStore, lane: str
) -> None:
    latent = default_store.latent.merge(
        default_store.data.transactions[["transaction_id", "register"]], on="transaction_id"
    )
    subset = latent[latent["register"].isin(LANES[lane]) & ~latent["interruption"]]
    design = np.column_stack(
        [
            np.ones(len(subset)),
            subset["n_items"],
            subset["n_hand_keyed"],
            subset["n_tagged"],
            subset["has_loyalty"].astype(float),
            subset["n_vouchers"],
        ]
    )
    alpha, beta, gamma, delta, loyalty, voucher = np.linalg.lstsq(
        design, subset["true_duration_seconds"].to_numpy(), rcond=None
    )[0]
    assert 22.0 < alpha < 29.0
    assert 2.75 < beta < 3.15
    assert 0.2 < gamma < 1.1
    assert 5.0 < delta < 7.0
    assert 7.0 < loyalty < 9.0
    assert 3.2 < voucher < 4.8


def test_gap_distribution_has_the_long_tail_the_pause_rule_targets(
    default_store: SimulatedStore,
) -> None:
    gaps = extract_gaps(default_store.data.events).gap_seconds
    assert np.median(gaps) < 4.0
    assert 0.2 < (gaps > 5.0).mean() < 0.4
    assert 0.01 < (gaps > 15.0).mean() < 0.05
    assert 0.0003 < (gaps > 60.0).mean() < 0.003
    assert gaps.max() > 120.0
    assert np.quantile(gaps, 0.999) > 2.0 * np.quantile(gaps, 0.99)


def test_explicit_exception_events_are_only_emitted_when_enabled(small_config: Config) -> None:
    default_events = simulate_store(small_config).data.events
    explicit = {"supervisor_call", "cheque_payment", "price_check"}
    assert not default_events["event_type"].astype(str).isin(explicit).any()
    enabled = simulate_store(
        small_config.with_overrides({"simulator.timing.emit_exception_events": True})
    )
    emitted = enabled.data.events["event_type"].astype(str).isin(explicit)
    assert emitted.sum() == int(enabled.latent["interruption"].sum()) > 0


def test_newcomers_follow_a_learning_curve() -> None:
    config = Config().with_overrides(
        {
            "simulator.population.n_cashiers": 16,
            "simulator.population.newcomer_share": 0.5,
            "simulator.n_weeks": 10,
        }
    )
    store = simulate_store(config)
    truth = store.ground_truth.set_index("worker_id")
    workers = store.data.workers.set_index("worker_id")
    latent = store.latent.merge(
        store.data.transactions[["transaction_id", "started_at"]], on="transaction_id"
    )
    worker_id = np.array(workers.index)[latent["worker_index"].to_numpy()]
    latent["worker_id"] = worker_id
    latent = latent[truth.loc[worker_id, "is_newcomer"].to_numpy()]
    hire = workers.loc[latent["worker_id"], "hire_date"].to_numpy()
    latent = latent.assign(
        weeks_since_hire=(latent["started_at"].to_numpy() - hire) / np.timedelta64(7, "D")
    )
    baseline = (
        truth.loc[latent["worker_id"], "true_speed"].to_numpy()
        + latent["is_peak"].to_numpy() * truth.loc[latent["worker_id"], "peak_effect"].to_numpy()
    )
    latent = latent.assign(learning_component=latent["effective_log_speed"].to_numpy() - baseline)
    bins = pd.cut(latent["weeks_since_hire"], [0, 2, 4, 8, 20], right=False)
    means = latent.groupby(bins, observed=True)["learning_component"].agg(["mean", "count"])
    assert (means["count"] > 200).all()
    assert (np.diff(means["mean"].to_numpy()) > 0).all()
    assert means["mean"].iloc[0] < -0.10
    assert means["mean"].iloc[-1] > -0.06


def test_cash_sheet_properties(default_store: SimulatedStore) -> None:
    sheet = default_store.data.cash_sheet
    assert not (sheet["register"] == default_store.config.store.card_only_register).any()
    assert (sheet["cash_handled"] > 0).all()
    latent = default_store.latent
    assert abs(sheet["cash_handled"].sum() - latent["cash_amount"].sum()) < 0.01 * len(sheet) + 1.0

    truth = default_store.ground_truth.set_index("worker_id")
    single = sheet.dropna(subset=["worker_id"])
    single = single[single.groupby("drawer_id")["worker_id"].transform("nunique") == 1]
    absolute = (
        single.assign(absolute=single["till_discrepancy"].abs())
        .groupby("worker_id")["absolute"]
        .sum()
    )
    correlation = stats.spearmanr(truth.loc[absolute.index, "error_propensity"], absolute).statistic
    assert correlation > 0.4


def test_void_and_manual_entry_rates_follow_error_propensity(default_store: SimulatedStore) -> None:
    events = default_store.data.events
    transactions = default_store.data.transactions.merge(
        default_store.data.sessions[["session_id", "worker_id"]], on="session_id"
    )
    owner = transactions.set_index("transaction_id")["worker_id"]
    truth = default_store.ground_truth.set_index("worker_id")
    per_worker = transactions.groupby("worker_id").size()
    voids = events[
        (events["event_type"] == "void") & events["detail"].isin(["cashier_error", "unknown"])
    ]
    void_rate = (
        voids.groupby(owner.reindex(voids["transaction_id"]).to_numpy()).size() / per_worker * 100
    )
    manual = events[events["event_type"] == "manual_entry"]
    manual_rate = (
        manual.groupby(owner.reindex(manual["transaction_id"]).to_numpy()).size() / per_worker * 100
    )
    for rate in (void_rate, manual_rate):
        rate = rate.dropna()
        assert stats.spearmanr(truth.loc[rate.index, "error_propensity"], rate).statistic > 0.5
    reasons = events.loc[events["event_type"] == "void", "detail"].value_counts(normalize=True)
    assert {"cashier_error", "customer_change", "supervisor_requested", "unknown"} <= set(
        reasons.index
    )


def test_store_roundtrips_through_a_directory_without_identity_columns(
    small_config: Config, tmp_path: Path
) -> None:
    directory = simulate_to_directory(small_config, tmp_path / "store")
    loaded = load_store_data(directory)
    fresh = simulate_store(small_config)
    assert len(loaded.events) == len(fresh.data.events)
    assert loaded.sessions["session_id"].tolist() == fresh.data.sessions["session_id"].tolist()
    offenders = [
        (table, column)
        for table, columns in list_all_columns(directory).items()
        if table != "ground_truth"
        for column in columns
        if FORBIDDEN_COLUMN_PATTERN.search(column)
    ]
    assert offenders == []
