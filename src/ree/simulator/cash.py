"""Cash sheet of the simulation: cash handled and the till discrepancy of each drawer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ree.config import Config, make_rng, replicate_stream
from ree.domain.events import CASH_SHEET_COLUMNS


def build_cash_sheet(config: Config, sessions: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
    """Build the cash sheet from the cash and mistake amounts of the hidden transactions.

    There is one row per drawer and cashier. The discrepancy belongs to the whole drawer
    (its mistakes plus noise), and a small share of drawers lose the name of one owner.
    """
    errors = config.simulator.errors
    rng = make_rng(config.seed, replicate_stream(config, "cash_sheet"))
    per_session = (
        latent.groupby("session_id", sort=True)[["cash_amount", "mistake_amount"]]
        .sum()
        .reset_index()
    )
    frame = sessions[["session_id", "worker_id", "register", "date", "drawer_id"]].merge(
        per_session, on="session_id", how="inner"
    )
    per_worker = (
        frame.groupby(["drawer_id", "worker_id"], sort=True)
        .agg(
            date=("date", "first"),
            register=("register", "first"),
            cash_handled=("cash_amount", "sum"),
            mistakes=("mistake_amount", "sum"),
        )
        .reset_index()
    )
    per_drawer = (
        per_worker.groupby("drawer_id", sort=True)
        .agg(cash_total=("cash_handled", "sum"), mistakes_total=("mistakes", "sum"))
        .reset_index()
    )
    per_drawer["till_discrepancy"] = (
        per_drawer["mistakes_total"]
        + errors.natural_noise_sd * rng.standard_normal(len(per_drawer))
    ).round(2)
    sheet = per_worker.merge(
        per_drawer[["drawer_id", "cash_total", "till_discrepancy"]], on="drawer_id", how="left"
    )
    sheet = sheet[sheet["cash_total"] > 0.0].copy()
    sheet["cash_handled"] = sheet["cash_handled"].round(2)

    sheet = sheet.sort_values(["date", "register", "drawer_id", "worker_id"], kind="stable")
    sheet = sheet.reset_index(drop=True)
    sheet["worker_id"] = sheet["worker_id"].astype(object)
    drawers = sheet["drawer_id"].drop_duplicates().to_numpy()
    unknown = drawers[rng.random(len(drawers)) < errors.unknown_owner_probability]
    for drawer in unknown:
        rows = np.flatnonzero((sheet["drawer_id"] == drawer).to_numpy())
        sheet.loc[sheet.index[rows[0]], "worker_id"] = None
    sheet["date"] = pd.to_datetime(sheet["date"]).astype("datetime64[ns]")
    return sheet[list(CASH_SHEET_COLUMNS)]
