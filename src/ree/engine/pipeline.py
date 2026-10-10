"""Scoring pipeline: from store data to cashier score records, with an audit trail."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ree import __version__
from ree.config import Config, make_rng
from ree.domain.models import (
    AuditRecord,
    CashierScoreRecord,
    Context,
    Interval,
    ScoreComponents,
    StoreData,
)
from ree.engine.accuracy import AccuracyFit, fit_accuracy, review_flags
from ree.engine.combine import combined_score
from ree.engine.drivers import DriverInputs, build_drivers
from ree.engine.features import TransactionFeatures, as_of_week_index, build_features, weeks_of_data
from ree.engine.intervals import (
    accuracy_draws_with_fallback,
    combined_interval,
    joint_speed_draws,
)
from ree.engine.pause_rule import PauseThresholds
from ree.engine.shrinkage import cohort_statistics, shrink, to_score_scale
from ree.engine.speed_v1 import V1Fit, fit_v1
from ree.engine.speed_v2 import V2Fit, fit_v2, sanity_warnings
from ree.engine.tiers import assign_tiers, interval_confidence
from ree.storage import db

CONTEXTS: tuple[Context, ...] = ("peak", "off")
DAYS_PER_WEEK = 7.0


@dataclass
class PipelineArtifacts:
    """Records and intermediate results of a scoring run, kept for experiments and screens."""

    records: list[CashierScoreRecord]
    features: TransactionFeatures
    speed_fit: V1Fit | V2Fit
    accuracy_fit: AccuracyFit
    worker_table: pd.DataFrame
    price_list: pd.DataFrame | None
    warnings: list[str]
    audit: AuditRecord
    thresholds: PauseThresholds
    speed_draws: dict[str, np.ndarray]
    accuracy_draws: np.ndarray


def source_hash() -> str:
    """Hash of the source files of the engine."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def scoring_config_hash(config: Config) -> str:
    """Hash of the engine and store settings."""
    payload = {
        "engine": config.engine.model_dump(mode="json"),
        "store": config.store.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def model_version(config: Config) -> str:
    """Label of the scoring model: package version, settings hash and source hash."""
    return f"ree-{__version__}-{scoring_config_hash(config)[:8]}-{source_hash()[:8]}"


def _fit_speed(frame: pd.DataFrame, config: Config) -> V1Fit | V2Fit:
    rows = frame[frame["valid_for_speed"]]
    durations = rows["duration_capped"].to_numpy()
    if config.engine.speed.method == "v1":
        return fit_v1(rows, durations, config.engine.speed)
    return fit_v2(rows, durations, config.engine.speed)


def _tenure_weeks(data: StoreData, worker_ids: list[str]) -> pd.Series:
    hire = data.workers.set_index("worker_id")["hire_date"].reindex(worker_ids)
    return (data.as_of - pd.to_datetime(hire)).dt.total_seconds() / 86400.0 / DAYS_PER_WEEK


def _speed_table(
    fit: V1Fit | V2Fit,
    worker_ids: list[str],
    provisional: pd.Series,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    engine = config.engine
    scores = fit.worker_scores()
    table = pd.DataFrame(index=pd.Index(worker_ids, name="worker_id"))
    cohort: dict[str, tuple[float, float]] = {}
    for context in CONTEXTS:
        selected = scores[scores["period"] == context].set_index("worker_id")
        raw = selected["raw_score"].reindex(worker_ids).fillna(0.0).to_numpy()
        n_valid = selected["n_valid"].reindex(worker_ids).fillna(0).astype(int).to_numpy()
        shrunk = np.asarray(shrink(raw, n_valid, engine.speed.shrinkage_k), dtype=np.float64)
        rated = (n_valid >= engine.min_transactions_rated) & ~provisional.to_numpy()
        mean, spread = cohort_statistics(shrunk, rated)
        z = np.clip((shrunk - mean) / spread, -engine.combine.z_cap, engine.combine.z_cap)
        table[f"raw_{context}"] = raw
        table[f"n_valid_{context}"] = n_valid
        table[f"shrunk_{context}"] = shrunk
        table[f"rated_{context}"] = rated
        table[f"z_{context}"] = z
        table[f"speed_score_{context}"] = to_score_scale(
            z, engine.combine.score_center, engine.combine.score_scale
        )
        cohort[context] = (mean, spread)
    return table, cohort


def run_with_artifacts(
    data: StoreData, config: Config, wallclock: datetime | None = None
) -> PipelineArtifacts:
    """Score every cashier and return the records with the intermediate results.

    The audit record in the result says which settings and code produced the scores.
    """
    started = wallclock or datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    data.validate()
    engine = config.engine
    features = build_features(data, config)
    frame = features.frame
    fit = _fit_speed(frame, config)
    worker_ids = sorted(data.sessions["worker_id"].unique().tolist())

    tenure = _tenure_weeks(data, worker_ids)
    provisional = pd.Series(
        (tenure < engine.provisional_weeks).to_numpy(), index=worker_ids, dtype=bool
    )
    weeks = weeks_of_data(data).reindex(worker_ids).fillna(0).astype(int)
    transaction_counts = frame.groupby("worker_id").size().reindex(worker_ids).fillna(0).astype(int)

    table, cohort = _speed_table(fit, worker_ids, provisional, config)
    table["provisional"] = provisional
    table["weeks_of_data"] = weeks
    table["n_transactions"] = transaction_counts
    rated_overall = table.index[table["rated_peak"] | table["rated_off"]]

    accuracy = fit_accuracy(
        frame,
        data.cash_sheet,
        features.data_start,
        as_of_week_index(data),
        engine.accuracy,
        engine.combine,
        rated=set(rated_overall),
    )
    accuracy_table = accuracy.table.reindex(worker_ids)
    table["accuracy_score"] = accuracy_table["accuracy_score"].fillna(engine.combine.score_center)
    cohort_accuracy = accuracy.table.loc[
        accuracy.table.index.intersection(rated_overall), "accuracy_score"
    ].to_numpy()

    for context in CONTEXTS:
        table[f"score_{context}"] = combined_score(
            table[f"speed_score_{context}"].to_numpy(),
            table["accuracy_score"].to_numpy(),
            engine.combine,
            context,
        )

    lower = {context: [] for context in CONTEXTS}
    upper = {context: [] for context in CONTEXTS}
    bootstrap_count = engine.intervals.n_bootstrap
    seed = engine.intervals.seed
    accuracy_matrix = accuracy.joint_bootstrap(
        worker_ids, table.index.isin(rated_overall), seed, bootstrap_count
    )
    speed_matrices = {
        context: joint_speed_draws(
            fit,
            worker_ids,
            context,
            table[f"n_valid_{context}"].to_numpy(),
            table[f"rated_{context}"].to_numpy(),
            seed,
            bootstrap_count,
            engine.speed.shrinkage_k,
            engine.combine,
        )
        for context in CONTEXTS
    }
    for column, worker_id in enumerate(worker_ids):
        accuracy_draws = accuracy_draws_with_fallback(
            accuracy_matrix[:, column],
            float(table.loc[worker_id, "accuracy_score"]),
            int(table.loc[worker_id, "weeks_of_data"]),
            cohort_accuracy,
            engine.intervals,
            engine.combine,
            make_rng(seed, f"bootstrap:fallback:{worker_id}"),
        )
        for context in CONTEXTS:
            interval = combined_interval(
                speed_matrices[context][:, column],
                accuracy_draws,
                float(table.loc[worker_id, f"score_{context}"]),
                engine.intervals,
                engine.combine,
                context,
            )
            lower[context].append(interval.lower)
            upper[context].append(interval.upper)

    for context in CONTEXTS:
        table[f"lower_{context}"] = lower[context]
        table[f"upper_{context}"] = upper[context]
        table[f"width_{context}"] = table[f"upper_{context}"] - table[f"lower_{context}"]
        table[f"confidence_{context}"] = interval_confidence(
            table[f"width_{context}"].to_numpy(), engine.intervals.width_limit
        )
        table[f"tier_{context}"] = assign_tiers(
            table[f"score_{context}"],
            table[f"lower_{context}"],
            table[f"upper_{context}"],
            table["provisional"],
            table["weeks_of_data"],
            engine.tiers,
            engine.intervals.width_limit,
        )

    speed_z_high = table[["z_peak", "z_off"]].max(axis=1)
    flags = review_flags(
        speed_z_high,
        accuracy_table["z_void"].reindex(worker_ids).fillna(0.0),
        accuracy_table["z_manual"].reindex(worker_ids).fillna(0.0),
        engine.review,
    )
    table["review_flag"] = flags

    driver_inputs = DriverInputs(
        row_scores=fit.row_scores,
        speed_spread={context: cohort[context][1] for context in CONTEXTS},
        accuracy_table=accuracy_table,
        accuracy_parameters=accuracy.parameters,
        accuracy_config=engine.accuracy,
        combine=engine.combine,
        provisional=table["provisional"],
        weeks_of_data=table["weeks_of_data"],
        n_valid=table[["n_valid_peak", "n_valid_off"]],
        min_weeks=engine.tiers.min_weeks_of_data,
        min_transactions=engine.min_transactions_rated,
    )
    drivers = build_drivers(worker_ids, driver_inputs)

    version = model_version(config)
    as_of = data.as_of.to_pydatetime()
    records = [
        _record(worker_id, table, accuracy_table, drivers, version, as_of, engine.speed.method)
        for worker_id in worker_ids
    ]

    price_list = fit.price_list if isinstance(fit, V2Fit) else None
    warnings = sanity_warnings(price_list, engine.speed) if price_list is not None else []
    unrated = int((table["tier_peak"] == "unrated").sum() + (table["tier_off"] == "unrated").sum())
    if unrated:
        warnings.append(
            f"{unrated} worker-context scores are unrated (wide interval or little data)"
        )
    audit = AuditRecord(
        run_id=f"{version}-{data.as_of:%Y%m%d%H%M%S}",
        model_version=version,
        config_hash=scoring_config_hash(config),
        code_hash=source_hash(),
        as_of=as_of,
        n_workers=len(worker_ids),
        n_transactions=len(frame),
        speed_method=engine.speed.method,
        warnings=warnings,
        started_wallclock_utc=started,
    )
    return PipelineArtifacts(
        records=records,
        features=features,
        speed_fit=fit,
        accuracy_fit=accuracy,
        worker_table=table.reset_index(),
        price_list=price_list,
        warnings=warnings,
        audit=audit,
        thresholds=features.thresholds,
        speed_draws=speed_matrices,
        accuracy_draws=accuracy_matrix,
    )


def _record(
    worker_id: str,
    table: pd.DataFrame,
    accuracy_table: pd.DataFrame,
    drivers: dict[str, list[str]],
    version: str,
    as_of: datetime,
    method: str,
) -> CashierScoreRecord:
    row = table.loc[worker_id]
    accuracy_row = accuracy_table.loc[worker_id] if worker_id in accuracy_table.index else None
    error_rate = None
    void_rate = None
    if accuracy_row is not None:
        error_rate = _optional(accuracy_row["error_rate_raw"])
        void_rate = _optional(accuracy_row["void_rate_raw"])
    review = row["review_flag"]
    return CashierScoreRecord(
        worker_id=worker_id,
        score_peak=float(row["score_peak"]),
        score_off=float(row["score_off"]),
        interval_peak=Interval(lower=float(row["lower_peak"]), upper=float(row["upper_peak"])),
        interval_off=Interval(lower=float(row["lower_off"]), upper=float(row["upper_off"])),
        tier_peak=row["tier_peak"],
        tier_off=row["tier_off"],
        n_transactions=int(row["n_transactions"]),
        weeks_of_data=int(row["weeks_of_data"]),
        provisional=bool(row["provisional"]),
        review_flag=None if review is None or pd.isna(review) else str(review),
        drivers=drivers[worker_id],
        model_version=version,
        computed_at=as_of,
        components=ScoreComponents(
            speed_score_peak=float(row["speed_score_peak"]),
            speed_score_off=float(row["speed_score_off"]),
            accuracy_score=float(row["accuracy_score"]),
            n_valid_peak=int(row["n_valid_peak"]),
            n_valid_off=int(row["n_valid_off"]),
            error_rate=error_rate,
            void_rate_per_100=void_rate,
            confidence_peak=float(row["confidence_peak"]),
            confidence_off=float(row["confidence_off"]),
            speed_method=method,
        ),
    )


def _optional(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def run(data: StoreData, config: Config) -> list[CashierScoreRecord]:
    """Score every cashier and return the records."""
    return run_with_artifacts(data, config).records


def score_directory(config: Config) -> str:
    """Score a stored simulation, save the scores, price list and audit, and return a summary."""
    directory = db.store_directory(config)
    data = db.load_store_data(directory)
    artifacts = run_with_artifacts(data, config)
    connection = db.connect(directory)
    try:
        db.save_scores(
            connection,
            artifacts.audit.run_id,
            artifacts.records,
            artifacts.audit,
            artifacts.price_list,
        )
    finally:
        connection.close()
    db.write_scores_json(artifacts.records, directory / db.SCORES_FILE)
    lines = [
        f"scored {len(artifacts.records)} cashiers from {artifacts.audit.n_transactions} "
        f"transactions with method {artifacts.audit.speed_method}",
        f"model version {artifacts.audit.model_version}",
        f"scores written to {directory / db.SCORES_FILE} and {directory / db.DATABASE_FILE}",
        *[f"warning: {message}" for message in artifacts.warnings],
    ]
    return "\n".join(lines)
