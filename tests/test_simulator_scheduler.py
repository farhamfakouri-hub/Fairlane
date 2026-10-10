from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ree.config import Config
from ree.domain.calendar import StoreCalendar
from ree.domain.models import SimulatedStore
from ree.simulator.run import simulate_store

HANDOVER_TOLERANCE_SECONDS = 600


def _with_day(sessions: pd.DataFrame) -> pd.DataFrame:
    frame = sessions.copy()
    frame["day"] = frame["login_ts"].dt.normalize()
    return frame


def test_no_worker_is_double_booked(default_store: SimulatedStore) -> None:
    frame = _with_day(default_store.data.sessions).sort_values(["worker_id", "login_ts"])
    next_login = frame.groupby("worker_id")["login_ts"].shift(-1)
    overlap = (frame["logout_ts"] - next_login).dt.total_seconds()
    assert (overlap.dropna() <= 0).all()


def test_registers_only_overlap_during_handovers(default_store: SimulatedStore) -> None:
    frame = _with_day(default_store.data.sessions).sort_values(["register", "login_ts"])
    previous_logout = frame.groupby("register")["logout_ts"].shift(1)
    overlap = (previous_logout - frame["login_ts"]).dt.total_seconds()
    assert overlap.dropna().max() <= HANDOVER_TOLERANCE_SECONDS


def test_sessions_respect_opening_hours_and_hiring(default_store: SimulatedStore) -> None:
    sessions = _with_day(default_store.data.sessions)
    calendar = StoreCalendar(default_store.config.store)
    weekday = sessions["login_ts"].dt.weekday.to_numpy()
    open_minute = calendar.opening_minutes[weekday, 0]
    close_minute = calendar.opening_minutes[weekday, 1]
    login_minute = (sessions["login_ts"] - sessions["day"]).dt.total_seconds().to_numpy() / 60
    logout_minute = (sessions["logout_ts"] - sessions["day"]).dt.total_seconds().to_numpy() / 60
    assert (login_minute >= open_minute - 60).all()
    assert (logout_minute <= close_minute + 15).all()
    hire = default_store.data.workers.set_index("worker_id")["hire_date"]
    assert (sessions["day"] >= sessions["worker_id"].map(hire)).all()
    assert set(sessions["register"]) == set(range(1, 16))


def test_priority_and_card_only_registers_are_staffed_most_of_the_day(
    default_store: SimulatedStore,
) -> None:
    sessions = _with_day(default_store.data.sessions)
    calendar = StoreCalendar(default_store.config.store)
    days = sessions["day"].drop_duplicates()
    weekdays = [day for day in days if day.weekday() != 6]
    for register in (
        default_store.config.store.priority_register,
        default_store.config.store.card_only_register,
    ):
        staffed_minutes = 0.0
        open_minutes = 0.0
        for day in weekdays:
            opening, closing = calendar.open_close_minutes(day.weekday())
            covered = np.zeros(closing - opening, dtype=bool)
            for row in sessions[
                (sessions["day"] == day) & (sessions["register"] == register)
            ].itertuples():
                start = int((row.login_ts - day).total_seconds() // 60) - opening
                end = int((row.logout_ts - day).total_seconds() // 60) - opening
                covered[max(start, 0) : max(end, 0)] = True
            staffed_minutes += covered.sum()
            open_minutes += len(covered)
        assert staffed_minutes / open_minutes > 0.85


def _hours_share(store: SimulatedStore, peak: bool) -> pd.Series:
    calendar = StoreCalendar(store.config.store)
    sessions = store.data.sessions
    day = sessions["login_ts"].dt.normalize()
    start = ((sessions["login_ts"] - day).dt.total_seconds() // 60).astype(int)
    end = ((sessions["logout_ts"] - day).dt.total_seconds() // 60).astype(int)
    weekday = sessions["login_ts"].dt.weekday.to_numpy()
    hours = (end - start).to_numpy() / 60.0
    peak_hours = np.array(
        [
            calendar.minute_peak_hours(int(w), int(s), int(e))
            for w, s, e in zip(weekday, start, end, strict=True)
        ]
    )
    frame = pd.DataFrame(
        {
            "worker_id": sessions["worker_id"],
            "hours": hours,
            "peak_hours": peak_hours,
            "card_only_hours": np.where(
                sessions["register"] == store.config.store.card_only_register, hours, 0.0
            ),
        }
    )
    grouped = frame.groupby("worker_id").sum()
    return grouped["peak_hours" if peak else "card_only_hours"] / grouped["hours"]


@pytest.fixture(scope="module")
def biased_store() -> SimulatedStore:
    return simulate_store(Config().with_overrides({"simulator.scheduler.mode": "biased"}))


def test_biased_mode_favours_strong_cashiers(
    biased_store: SimulatedStore, default_store: SimulatedStore
) -> None:
    truth = biased_store.ground_truth.set_index("worker_id")
    order = truth["true_speed"].sort_values(ascending=False).index
    group_size = round(0.4 * len(order))
    strong, weak = order[:group_size], order[-group_size:]
    biased_peak = _hours_share(biased_store, peak=True)
    biased_card = _hours_share(biased_store, peak=False)
    assert biased_peak[strong].mean() - biased_peak[weak].mean() > 0.04
    assert biased_card[strong].mean() > biased_card[weak].mean() + 0.02

    neutral_truth = default_store.ground_truth.set_index("worker_id")
    neutral_order = neutral_truth["true_speed"].sort_values(ascending=False).index
    neutral_peak = _hours_share(default_store, peak=True)
    random_gap = (
        neutral_peak[neutral_order[:group_size]].mean()
        - neutral_peak[neutral_order[-group_size:]].mean()
    )
    assert abs(random_gap) < 0.04


def test_shared_tills_follow_the_configured_probability() -> None:
    base = {"simulator.population.n_cashiers": 12, "simulator.n_weeks": 3}
    none_shared = simulate_store(
        Config().with_overrides({**base, "simulator.scheduler.shared_till_probability": 0.0})
    )
    all_shared = simulate_store(
        Config().with_overrides({**base, "simulator.scheduler.shared_till_probability": 1.0})
    )

    def shared_drawers(store: SimulatedStore) -> int:
        sheet = store.data.cash_sheet.dropna(subset=["worker_id"])
        return int((sheet.groupby("drawer_id")["worker_id"].nunique() > 1).sum())

    assert shared_drawers(none_shared) == 0
    assert shared_drawers(all_shared) >= 10


def test_unknown_owners_are_rare_but_present(default_store: SimulatedStore) -> None:
    sheet = default_store.data.cash_sheet
    share = sheet["worker_id"].isna().mean()
    assert 0.0 < share < 0.05
