"""Run a complete store simulation and write it to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ree.config import Config
from ree.domain.calendar import StoreCalendar
from ree.domain.events import SESSION_COLUMNS, WORKER_COLUMNS
from ree.domain.models import SimulatedStore, StoreData
from ree.simulator.cash import build_cash_sheet
from ree.simulator.scheduler import generate_sessions
from ree.simulator.store import build_register_profiles
from ree.simulator.transactions import simulate_transactions
from ree.simulator.workers import generate_workers
from ree.storage.db import save_store, store_directory

LATENT_FILE = "latent.parquet"


def _realized_speed_table(latent: pd.DataFrame, worker_ids: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"worker_id": worker_ids.to_numpy()}, index=np.arange(len(worker_ids)))
    for flag, label in ((False, "off"), (True, "peak")):
        part = (
            latent[latent["is_peak"] == flag]
            .groupby("worker_index")["effective_log_speed"]
            .agg(["mean", "count"])
        )
        table[f"realized_log_speed_{label}"] = part["mean"].reindex(table.index)
        table[f"n_transactions_{label}"] = part["count"].reindex(table.index).fillna(0).astype(int)
    overall = latent.groupby("worker_index")["effective_log_speed"].mean()
    table["realized_log_speed_all"] = overall.reindex(table.index)
    return table


def simulate_store(config: Config) -> SimulatedStore:
    """Simulate a store: workers, sessions, transactions, events and the cash sheet.

    The returned store keeps the data separate from the hidden ground truth.
    """
    calendar = StoreCalendar(config.store)
    profiles = build_register_profiles(config, calendar)
    workers, ground_truth = generate_workers(config)
    sessions, report = generate_sessions(config, workers, ground_truth, calendar, profiles)
    bundle = simulate_transactions(config, sessions, workers, ground_truth, profiles, calendar)

    sessions = sessions.copy()
    sessions["logout_ts"] = sessions["date"] + pd.to_timedelta(
        np.rint(bundle.session_logout).astype(np.int64), unit="s"
    )
    sessions["logout_ts"] = sessions["logout_ts"].astype("datetime64[ns]")
    cash_sheet = build_cash_sheet(config, sessions, bundle.latent)

    ground_truth = ground_truth.merge(
        _realized_speed_table(bundle.latent, ground_truth["worker_id"]), on="worker_id", how="left"
    )
    data = StoreData(
        workers=workers[list(WORKER_COLUMNS)],
        sessions=sessions[list(SESSION_COLUMNS)].reset_index(drop=True),
        transactions=bundle.transactions,
        events=bundle.events,
        cash_sheet=cash_sheet,
        metadata={
            "seed": config.seed,
            "n_transactions": len(bundle.transactions),
            "n_events": len(bundle.events),
            "dropped_shifts": report.dropped_shifts,
            "dropped_segments": report.dropped_segments,
            "simulation_rounds": bundle.rounds,
        },
    )
    latent = bundle.latent.merge(sessions[["session_id", "drawer_id"]], on="session_id", how="left")
    return SimulatedStore(data=data, ground_truth=ground_truth, config=config, latent=latent)


def simulate_to_directory(config: Config, directory: str | Path | None = None) -> Path:
    """Simulate a store and save it, with its hidden table, in a directory."""
    simulated = simulate_store(config)
    target = Path(directory) if directory is not None else store_directory(config)
    save_store(simulated, target)
    if simulated.latent is not None:
        simulated.latent.to_parquet(target / LATENT_FILE, index=False)
    return target
