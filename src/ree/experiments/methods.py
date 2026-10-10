"""The speed methods compared in the experiments, from a naive baseline to the final engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ree.config import Config
from ree.domain.calendar import StoreCalendar
from ree.domain.models import StoreData
from ree.engine.accuracy import fit_accuracy
from ree.engine.features import as_of_week_index, build_features, data_start_monday
from ree.engine.shrinkage import shrink
from ree.engine.speed_v1 import fit_v1
from ree.engine.speed_v2 import fit_v2

PERIODS: tuple[str, ...] = ("off", "peak")
MINUTES_PER_HOUR = 60.0

METHOD_LABELS: dict[str, str] = {
    "naive_items_per_shift": "naive items per shift",
    "hard_5s_stop": "hard 5 s stop",
    "v1_no_pause": "v1 bins, no pause rule",
    "v1_pause": "v1 bins + pause rule",
    "v2_ols_pause": "v2 regression (OLS) + pause rule",
    "v2_huber_pause": "v2 regression (Huber) + pause rule",
}
METHOD_FAMILY: dict[str, str] = {
    "naive_items_per_shift": "baseline",
    "hard_5s_stop": "baseline",
    "v1_no_pause": "v1",
    "v1_pause": "v1",
    "v2_ols_pause": "v2",
    "v2_huber_pause": "v2",
}
ACCURACY_VARIANTS: tuple[str, ...] = (
    "split_by_cash_share",
    "full_to_each_cashier",
    "drop_shared_tills",
)


@dataclass
class MethodInputs:
    """Inputs shared by the methods: features, hours worked, cash sheet and the as-of week."""

    frame: pd.DataFrame
    session_hours: pd.DataFrame
    cash_sheet: pd.DataFrame
    data_start: pd.Timestamp
    as_of_week: int
    config: Config

    def truncated(self, weeks: int) -> MethodInputs:
        """The same inputs restricted to the first weeks of data."""
        return MethodInputs(
            frame=self.frame[self.frame["week_index"] < weeks],
            session_hours=self.session_hours[self.session_hours["week_index"] < weeks],
            cash_sheet=self.cash_sheet[
                (self.cash_sheet["date"] - self.data_start).dt.days // 7 < weeks
            ],
            data_start=self.data_start,
            as_of_week=weeks - 1,
            config=self.config,
        )


def session_hours_table(data: StoreData, config: Config) -> pd.DataFrame:
    """Peak and off-peak hours of each session, used by the naive baseline."""
    calendar = StoreCalendar(config.store)
    sessions = data.sessions
    day = sessions["login_ts"].dt.normalize()
    login = ((sessions["login_ts"] - day).dt.total_seconds() // 60).astype(int).to_numpy()
    logout = np.minimum(
        np.ceil((sessions["logout_ts"] - day).dt.total_seconds() / 60.0).astype(int).to_numpy(),
        1440,
    )
    weekday = sessions["login_ts"].dt.weekday.to_numpy()
    peak = np.array(
        [
            calendar.minute_peak_hours(int(w), int(a), int(b))
            for w, a, b in zip(weekday, login, logout, strict=True)
        ]
    )
    total = (logout - login) / MINUTES_PER_HOUR
    start = data_start_monday(data)
    return pd.DataFrame(
        {
            "worker_id": sessions["worker_id"].to_numpy(),
            "week_index": ((day - start).dt.days // 7).to_numpy(),
            "peak_hours": peak,
            "off_hours": total - peak,
        }
    )


def build_method_inputs(data: StoreData, config: Config) -> MethodInputs:
    """Build the inputs of every method from store data."""
    features = build_features(data, config)
    return MethodInputs(
        frame=features.frame,
        session_hours=session_hours_table(data, config),
        cash_sheet=data.cash_sheet,
        data_start=features.data_start,
        as_of_week=as_of_week_index(data),
        config=config,
    )


def _shrunk(scores: pd.DataFrame, k: float) -> pd.DataFrame:
    estimate = shrink(scores["raw_score"].to_numpy(), scores["n_valid"].to_numpy(), k)
    return pd.DataFrame(
        {
            "worker_id": scores["worker_id"].to_numpy(),
            "period": scores["period"].to_numpy(),
            "estimate": np.asarray(estimate, dtype=np.float64),
        }
    )


def naive_items_per_shift(inputs: MethodInputs) -> pd.DataFrame:
    """Baseline: items per logged hour, ignoring basket mix and interruptions."""
    items = inputs.frame.groupby(["worker_id", "period"])["n_items"].sum()
    hours = inputs.session_hours.groupby("worker_id")[["peak_hours", "off_hours"]].sum()
    rows = []
    for (worker_id, period), total in items.items():
        logged = hours.loc[worker_id, f"{period}_hours"] if worker_id in hours.index else 0.0
        rows.append((worker_id, period, total / logged if logged > 0 else np.nan))
    return pd.DataFrame(rows, columns=["worker_id", "period", "estimate"])


def _rows_without_exceptions(inputs: MethodInputs) -> pd.DataFrame:
    frame = inputs.frame
    return frame[~frame["has_exception"] & ~frame["explicit_interruption"]]


def hard_5s_stop(inputs: MethodInputs) -> pd.DataFrame:
    """Baseline: v1 bins where any gap above the hard-stop limit is dropped."""
    rows = _rows_without_exceptions(inputs)
    fit = fit_v1(rows, rows["duration_hard_stop"].to_numpy(), inputs.config.engine.speed)
    return _shrunk(fit.scores, inputs.config.engine.speed.shrinkage_k)


def v1_no_pause(inputs: MethodInputs) -> pd.DataFrame:
    """v1 bins on raw durations, without the pause rule."""
    rows = _rows_without_exceptions(inputs)
    fit = fit_v1(rows, rows["duration_raw"].to_numpy(), inputs.config.engine.speed)
    return _shrunk(fit.scores, inputs.config.engine.speed.shrinkage_k)


def v1_pause(inputs: MethodInputs) -> pd.DataFrame:
    """v1 bins on durations capped by the pause rule."""
    rows = inputs.frame[inputs.frame["valid_for_speed"]]
    fit = fit_v1(rows, rows["duration_capped"].to_numpy(), inputs.config.engine.speed)
    return _shrunk(fit.scores, inputs.config.engine.speed.shrinkage_k)


def _v2(inputs: MethodInputs, regression: str) -> pd.DataFrame:
    rows = inputs.frame[inputs.frame["valid_for_speed"]]
    speed = inputs.config.engine.speed.model_copy(update={"regression": regression})
    fit = fit_v2(rows, rows["duration_capped"].to_numpy(), speed)
    return _shrunk(fit.scores, speed.shrinkage_k)


def v2_ols_pause(inputs: MethodInputs) -> pd.DataFrame:
    """v2 regression fitted by ordinary least squares, with the pause rule."""
    return _v2(inputs, "ols")


def v2_huber_pause(inputs: MethodInputs) -> pd.DataFrame:
    """v2 regression fitted with a robust Huber estimator, with the pause rule."""
    return _v2(inputs, "huber")


SPEED_METHODS: dict[str, Callable[[MethodInputs], pd.DataFrame]] = {
    "naive_items_per_shift": naive_items_per_shift,
    "hard_5s_stop": hard_5s_stop,
    "v1_no_pause": v1_no_pause,
    "v1_pause": v1_pause,
    "v2_ols_pause": v2_ols_pause,
    "v2_huber_pause": v2_huber_pause,
}


def cash_sheet_variant(sheet: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Change the cash sheet to test how shared tills are treated.

    The variants split the discrepancy by cash handled, give all of it to each cashier,
    or drop shared tills.
    """
    if variant == "split_by_cash_share":
        return sheet
    shared = sheet.groupby("drawer_id")["drawer_id"].transform("size") > 1
    if variant == "drop_shared_tills":
        return sheet[~shared]
    if variant == "full_to_each_cashier":
        altered = sheet.copy()
        altered.loc[shared, "drawer_id"] = (
            altered.loc[shared, "drawer_id"] + "|" + altered.loc[shared, "worker_id"].astype(str)
        )
        return altered
    raise ValueError(f"unknown accuracy variant {variant!r}")


def accuracy_scores_by_variant(inputs: MethodInputs, variant: str) -> pd.Series:
    """Accuracy scores of every cashier under one treatment of shared tills."""
    engine = inputs.config.engine
    fit = fit_accuracy(
        inputs.frame,
        cash_sheet_variant(inputs.cash_sheet, variant),
        inputs.data_start,
        inputs.as_of_week,
        engine.accuracy,
        engine.combine,
    )
    return fit.table["accuracy_score"]


def estimates_by_period(estimates: pd.DataFrame) -> dict[str, pd.Series]:
    """Split an estimate table into one series for each period."""
    return {
        period: estimates[estimates["period"] == period].set_index("worker_id")["estimate"]
        for period in PERIODS
    }
