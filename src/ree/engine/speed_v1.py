"""Speed score, method v1: the store median over the cashier median in each cell."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ree.config import SpeedConfig
from ree.engine.resampling import session_codes, session_counts, weighted_median_rows

MIN_DURATION_SECONDS = 0.5
CELL_KEYS = ["period", "lane_group", "size_bin"]
WORKER_CELL_KEYS = ["worker_id", *CELL_KEYS]

CellValues = tuple[int, list[tuple[float, np.ndarray, np.ndarray]]]


@dataclass
class V1Fit:
    """Result of the v1 fit: scores, per-transaction scores, cells and bootstrap data."""

    scores: pd.DataFrame
    row_scores: pd.DataFrame
    cells: pd.DataFrame
    method: str = "v1"
    _cell_values: dict[tuple[str, str], CellValues] = field(default_factory=dict, repr=False)

    def worker_scores(self) -> pd.DataFrame:
        """Raw score and number of valid transactions of each cashier and period."""
        return self.scores

    def bootstrap(
        self, worker_id: str, period: str, rng: np.random.Generator, n_bootstrap: int
    ) -> np.ndarray:
        """Bootstrap draws of the raw score of one cashier and period, resampling sessions."""
        entry = self._cell_values.get((worker_id, period))
        if entry is None:
            return np.zeros(n_bootstrap)
        n_sessions, cells = entry
        counts = session_counts(rng, n_sessions, n_bootstrap)
        weighted_total = np.zeros(n_bootstrap)
        weighted_ratio = np.zeros(n_bootstrap)
        for store_median, durations, sessions in cells:
            weights = counts[:, sessions]
            cell_weight = weights.sum(axis=1)
            medians = weighted_median_rows(durations, weights)
            usable = cell_weight > 0
            ratio = np.where(usable, np.log(store_median / np.where(usable, medians, 1.0)), 0.0)
            weighted_total += np.where(usable, cell_weight, 0.0)
            weighted_ratio += np.where(usable, cell_weight * ratio, 0.0)
        return np.where(weighted_total > 0, weighted_ratio / np.maximum(weighted_total, 1.0), 0.0)


def fit_v1(frame: pd.DataFrame, durations: np.ndarray, config: SpeedConfig) -> V1Fit:
    """Fit the v1 speed score.

    Cells combine period, lane group and basket-size bin. In each cell the score is the log
    of the store median over the cashier median; cells are weighted by transaction count
    and small cells are dropped.
    """
    sessions = (
        frame["session_id"].to_numpy() if "session_id" in frame.columns else np.arange(len(frame))
    )
    work = pd.DataFrame(
        {
            "worker_id": frame["worker_id"].to_numpy(),
            "period": frame["period"].to_numpy(),
            "lane_group": frame["lane_group"].to_numpy(),
            "size_bin": frame["size_bin"].to_numpy(),
            "basket_class": frame["basket_class"].to_numpy(),
            "duration": np.maximum(np.asarray(durations, dtype=np.float64), MIN_DURATION_SECONDS),
        },
        index=frame.index,
    )
    store = (
        work.groupby(CELL_KEYS)["duration"].agg(store_median="median", store_n="size").reset_index()
    )
    cells = (
        work.groupby(WORKER_CELL_KEYS)["duration"]
        .agg(cell_median="median", cell_n="size")
        .reset_index()
        .merge(store, on=CELL_KEYS, how="left")
    )
    cells = cells[
        (cells["cell_n"] >= config.min_cashier_bin_transactions)
        & (cells["store_n"] >= config.min_store_bin_transactions)
    ].copy()
    cells["log_ratio"] = np.log(cells["store_median"] / cells["cell_median"])
    totals = cells.groupby(["worker_id", "period"])["cell_n"].transform("sum")
    cells["weight"] = cells["cell_n"] / totals
    cells["weighted"] = cells["weight"] * cells["log_ratio"]
    scores = (
        cells.groupby(["worker_id", "period"])
        .agg(raw_score=("weighted", "sum"), n_valid=("cell_n", "sum"))
        .reset_index()
    )
    row_scores = work.merge(
        cells[[*WORKER_CELL_KEYS, "log_ratio"]], on=WORKER_CELL_KEYS, how="left"
    )
    row_scores.index = work.index
    row_scores = row_scores[["worker_id", "period", "basket_class", "log_ratio"]]

    local_sessions, session_totals = session_codes(
        work["worker_id"].to_numpy(), work["period"].to_numpy(), sessions
    )
    work["session_code"] = local_sessions
    work["session_total"] = session_totals
    keyed = cells.set_index(WORKER_CELL_KEYS)
    cell_values: dict[tuple[str, str], CellValues] = {}
    for key, group in work.groupby(WORKER_CELL_KEYS):
        if key not in keyed.index:
            continue
        worker_id, period, _, _ = key
        order = np.argsort(group["duration"].to_numpy(), kind="stable")
        entry = (
            float(keyed.loc[key, "store_median"]),
            group["duration"].to_numpy(dtype=np.float64)[order],
            group["session_code"].to_numpy(dtype=np.int64)[order],
        )
        total = int(group["session_total"].iloc[0])
        existing = cell_values.setdefault((worker_id, period), (total, []))
        existing[1].append(entry)
    return V1Fit(scores=scores, row_scores=row_scores, cells=cells, _cell_values=cell_values)
