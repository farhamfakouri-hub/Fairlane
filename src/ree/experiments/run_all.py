"""Run all the experiments and write the report, tables and figures."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ree.config import Config
from ree.experiments import figures
from ree.experiments.calibration import coverage_from, dispersion_from, run_replicates
from ree.experiments.fairness import pooled_audit, run_audit_seeds, shrinkage_comparison
from ree.experiments.methods import (
    ACCURACY_VARIANTS,
    METHOD_LABELS,
    PERIODS,
    SPEED_METHODS,
    MethodInputs,
    accuracy_scores_by_variant,
    build_method_inputs,
    estimates_by_period,
)
from ree.experiments.metrics import evaluate, exposure_leakage, kendall, spearman
from ree.experiments.report import (
    HEADLINE,
    LATENT,
    METRIC_LABELS,
    REALIZED,
    confidence_interval,
    default_method_section,
    design_method_outcomes,
    markdown_table,
    scenario_descriptions,
    scenario_matrix,
    summarize,
    v1_versus_v2,
    winner_sentences,
)
from ree.experiments.scenarios import SCENARIO_BY_NAME, SCENARIOS
from ree.simulator.run import simulate_store

METRICS = tuple(METRIC_LABELS)
SCATTER_SCENARIO = "balanced"
LEARNING_SEEDS_FULL = 4
CALIBRATION_REDUCED_WEEKS = 6


@dataclass
class TaskOutput:
    """Metrics, accuracy results and optional scatter data for one scenario and seed."""

    metrics: pd.DataFrame
    accuracy: pd.DataFrame
    scatter: pd.DataFrame | None


def _targets(truth: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    return {
        LATENT: {"off": truth["true_speed"], "peak": truth["true_speed"] + truth["peak_effect"]},
        REALIZED: {
            "off": truth["realized_log_speed_off"],
            "peak": truth["realized_log_speed_peak"],
        },
    }


def _exposure(inputs: MethodInputs) -> dict[str, pd.Series]:
    mean_items = inputs.frame.groupby(["worker_id", "period"])["n_items"].mean()
    return {period: mean_items.xs(period, level="period") for period in PERIODS}


def run_task(
    config_json: str, scenario_name: str, seed: int, reduced: bool, keep_scatter: bool
) -> TaskOutput:
    """Simulate one scenario with one seed and score every method against the hidden truth."""
    base = Config.model_validate_json(config_json)
    config = SCENARIO_BY_NAME[scenario_name].apply(base, seed, reduced)
    store = simulate_store(config)
    inputs = build_method_inputs(store.data, config)
    truth = store.ground_truth.set_index("worker_id")
    targets = _targets(truth)
    exposure = _exposure(inputs)
    top_k = config.experiments.top_k

    metric_rows: list[dict[str, object]] = []
    scatter_rows: list[pd.DataFrame] = []
    for method, function in SPEED_METHODS.items():
        estimates = estimates_by_period(function(inputs))
        for period in PERIODS:
            for target_name, by_period in targets.items():
                target = by_period[period]
                scores = evaluate(estimates[period], target, top_k)
                leakage = (
                    exposure_leakage(estimates[period], target, exposure[period])
                    if target_name == LATENT
                    else float("nan")
                )
                metric_rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method,
                        "period": period,
                        "target": target_name,
                        **scores,
                        "leakage_basket": leakage,
                    }
                )
            if keep_scatter:
                latent = targets[LATENT][period]
                scatter_rows.append(
                    pd.DataFrame(
                        {
                            "method": method,
                            "period": period,
                            "worker_id": latent.index,
                            "truth": latent.to_numpy(),
                            "estimate": estimates[period].reindex(latent.index).to_numpy(),
                        }
                    )
                )

    accuracy_rows = []
    error_truth = -truth["error_propensity"]
    for variant in ACCURACY_VARIANTS:
        scores = accuracy_scores_by_variant(inputs, variant)
        accuracy_rows.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "variant": variant,
                "spearman": spearman(scores, error_truth),
                "kendall": kendall(scores, error_truth),
            }
        )
    return TaskOutput(
        metrics=pd.DataFrame(metric_rows),
        accuracy=pd.DataFrame(accuracy_rows),
        scatter=pd.concat(scatter_rows, ignore_index=True) if scatter_rows else None,
    )


def _map(function, arguments: list[tuple], jobs: int) -> list:
    if jobs <= 1:
        return [function(*item) for item in arguments]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(function, *item) for item in arguments]
        return [future.result() for future in futures]


def run_scenarios(
    config: Config, reduced: bool, jobs: int, n_seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Run every scenario for every seed, in parallel if asked."""
    payload = config.model_dump_json()
    seeds = [config.experiments.base_seed + index for index in range(n_seeds)]
    arguments = [
        (
            payload,
            scenario.name,
            seed,
            reduced,
            scenario.name == SCATTER_SCENARIO and seed == seeds[0],
        )
        for scenario in SCENARIOS
        for seed in seeds
    ]
    outputs = _map(run_task, arguments, jobs)
    metrics = pd.concat([item.metrics for item in outputs], ignore_index=True)
    accuracy = pd.concat([item.accuracy for item in outputs], ignore_index=True)
    scatter = next((item.scatter for item in outputs if item.scatter is not None), None)
    return metrics, accuracy, scatter


def run_learning_task(config_json: str, seed: int, reduced: bool) -> pd.DataFrame:
    """For one seed, score the methods on a growing number of weeks of data."""
    base = Config.model_validate_json(config_json)
    config = SCENARIO_BY_NAME[SCATTER_SCENARIO].apply(base, seed, reduced)
    store = simulate_store(config)
    inputs = build_method_inputs(store.data, config)
    truth = store.ground_truth.set_index("worker_id")
    targets = _targets(truth)[LATENT]
    rows = []
    for weeks in config.experiments.learning_curve_weeks:
        if weeks > config.simulator.n_weeks:
            continue
        window = inputs.truncated(weeks)
        for method, function in SPEED_METHODS.items():
            estimates = estimates_by_period(function(window))
            values = [spearman(estimates[period], targets[period]) for period in PERIODS]
            rows.append(
                {
                    "seed": seed,
                    "weeks": weeks,
                    "method": method,
                    "spearman": float(np.nanmean(values)),
                }
            )
    return pd.DataFrame(rows)


def run_learning_curve(config: Config, reduced: bool, jobs: int, n_seeds: int) -> pd.DataFrame:
    """Learning curve of the methods over several seeds."""
    payload = config.model_dump_json()
    seeds = [config.experiments.base_seed + 5000 + index for index in range(n_seeds)]
    frames = _map(run_learning_task, [(payload, seed, reduced) for seed in seeds], jobs)
    return pd.concat(frames, ignore_index=True)


def _calibration_config(config: Config, reduced: bool) -> Config:
    scenario = SCENARIO_BY_NAME[SCATTER_SCENARIO]
    calibrated = scenario.apply(config, config.experiments.base_seed + 9000, False)
    if not reduced:
        return calibrated
    return calibrated.with_overrides(
        {
            "simulator.population.n_cashiers": config.experiments.reduced_cashiers,
            "simulator.n_weeks": CALIBRATION_REDUCED_WEEKS,
        }
    )


def _ci_cell(
    values: pd.Series,
    level: float,
    digits: int = 3,
    bounds: tuple[float | None, float | None] = (-1.0, 1.0),
) -> str:
    mean, low, high = confidence_interval(values, level, bounds)
    return f"{mean:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def _per_seed_mean(frame: pd.DataFrame, keys: list[str], column: str) -> pd.DataFrame:
    return frame.groupby(keys, sort=False)[column].mean().reset_index()


def build_report(
    config: Config,
    reduced: bool,
    n_seeds: int,
    metrics: pd.DataFrame,
    accuracy: pd.DataFrame,
    curve: pd.DataFrame,
    coverage_text: str,
    fairness_text: str,
    figure_names: list[str],
) -> str:
    """Write the Markdown text of the experiment report."""
    level = config.experiments.confidence_level
    methods = list(SPEED_METHODS)
    experiments = config.experiments
    lines = ["# Experiment results", ""]
    mode = "REDUCED RUN (development only, not the published numbers)" if reduced else "Full run"
    lines += [
        f"{mode}. Generated by `src/ree/experiments/run_all.py` (`python -m ree.cli experiments`).",
        f"Seeds: {n_seeds} per scenario (base seed {experiments.base_seed}); intervals are "
        f"{level:.0%} t-intervals over seeds. Every number below is computed from the simulated "
        "stores of this run.",
        "",
        "## Scenarios",
        "",
        scenario_descriptions(),
        "",
        "## Which method wins where",
        "",
    ]
    summary = summarize(metrics, HEADLINE, level)
    lines += winner_sentences(metrics, summary, level, methods)
    lines += ["", *design_method_outcomes(metrics, summary, level, methods)]
    lines += ["", "## Default speed method", ""]
    lines += default_method_section(summary)
    lines += ["", "**v1 against v2 (paired difference of mean Spearman over seeds)**", ""]
    lines += [markdown_table(v1_versus_v2(metrics, summary, level), 4)]
    lines += ["", "## Spearman correlation with true speed (mean of off-peak and peak)", ""]
    lines += [markdown_table(scenario_matrix(summary, methods)), ""]
    for metric in METRICS[1:]:
        lines += [f"## {METRIC_LABELS[metric]}", ""]
        part = summarize(metrics, metric, level)
        digits = 2 if metric == "mean_abs_rank_error" else 3
        lines += [markdown_table(scenario_matrix(part, methods, digits)), ""]

    realized = summarize(metrics, HEADLINE, level, target=REALIZED)
    newcomers = realized[realized["scenario"] == "newcomers"].set_index("method")
    latent = summary[summary["scenario"] == "newcomers"].set_index("method")
    lines += [
        "## Newcomers: latent skill against the speed actually shown",
        "",
        "Newcomers speed up with tenure, so a method that measures their current speed correctly "
        "scores lower against their latent (long-run) skill than against the speed they showed.",
        "",
        markdown_table(
            pd.DataFrame(
                {
                    "method": [METHOD_LABELS[name] for name in methods],
                    "Spearman vs latent skill": [
                        f"{latent.loc[name, 'mean']:.3f}" for name in methods
                    ],
                    "Spearman vs realized speed": [
                        f"{newcomers.loc[name, 'mean']:.3f}" for name in methods
                    ],
                }
            )
        ),
        "",
    ]

    leakage_rows = []
    per_seed = _per_seed_mean(
        metrics[metrics["target"] == LATENT], ["scenario", "seed", "method"], "leakage_basket"
    )
    for name in methods:
        row: dict[str, object] = {"method": METHOD_LABELS[name]}
        for scenario in ("balanced", "biased"):
            values = per_seed[(per_seed["scenario"] == scenario) & (per_seed["method"] == name)][
                "leakage_basket"
            ]
            row[scenario] = _ci_cell(values, level, 3, (None, None))
        leakage_rows.append(row)
    lines += [
        "## Fairness: does the average basket size a cashier meets leak into the score?",
        "",
        "Coefficient of the standardised mean basket size in a regression of the standardised "
        "estimate on the standardised true skill and the basket size. Zero means the method "
        "ranks cashiers by skill only; the biased scenario gives strong cashiers smaller baskets.",
        "",
        markdown_table(pd.DataFrame(leakage_rows)),
        "",
    ]

    accuracy_rows = []
    for variant in ACCURACY_VARIANTS:
        row = {"attribution": variant.replace("_", " ")}
        for scenario in ("balanced", "shared_tills"):
            values = accuracy[
                (accuracy["scenario"] == scenario) & (accuracy["variant"] == variant)
            ]["spearman"]
            row[scenario] = _ci_cell(values, level)
        accuracy_rows.append(row)
    pivot_accuracy = accuracy.pivot_table(
        index=["scenario", "seed"], columns="variant", values="spearman"
    ).reset_index()
    paired_lines = []
    for scenario in ("balanced", "shared_tills"):
        part = pivot_accuracy[pivot_accuracy["scenario"] == scenario]
        mean, low, high = confidence_interval(
            part["split_by_cash_share"] - part["drop_shared_tills"], level
        )
        paired_lines.append(
            f"{scenario}: split by cash share minus drop shared tills = {mean:+.3f} "
            f"({low:+.3f} to {high:+.3f})"
        )
    lines += [
        "## Accuracy score with shared tills",
        "",
        "Spearman correlation of the accuracy score with minus the hidden error propensity, "
        "for three ways of attributing a shared drawer's cash discrepancy.",
        "",
        markdown_table(pd.DataFrame(accuracy_rows)),
        "",
        "Paired difference over seeds: " + "; ".join(paired_lines) + ".",
        "",
    ]

    lines += ["## Learning curve (balanced scenario, Spearman against true speed)", ""]
    pivot = curve.groupby(["weeks", "method"])["spearman"].mean().unstack("method")
    pivot = pivot[methods].reset_index()
    pivot.columns = ["weeks of data", *[METHOD_LABELS[name] for name in methods]]
    lines += [markdown_table(pivot), ""]
    lines += ["## Fairness audit by contract type and tenure", "", fairness_text, ""]
    lines += ["## Interval calibration", "", coverage_text, ""]
    if figure_names:
        lines += ["## Figures", ""]
        lines += [f"![{name}](../figures/{name})" for name in figure_names]
        lines.append("")
    return "\n".join(lines)


def fairness_section(config: Config, reduced: bool, jobs: int) -> tuple[str, pd.DataFrame]:
    """Run the bias audit on independent stores, with and without shrinkage, and describe it."""
    n_stores = 2 if reduced else config.fairness.audit_seeds
    audited = (
        config.with_overrides(
            {"simulator.population.n_cashiers": config.experiments.reduced_cashiers}
        )
        if reduced
        else config
    )
    seeds = [config.experiments.base_seed + 7000 + index for index in range(n_stores)]
    study = run_audit_seeds(audited, seeds, jobs)
    pooled = pooled_audit(study.inputs, config.fairness)
    unshrunk_study = run_audit_seeds(
        audited.with_overrides({"engine.speed.shrinkage_k": 0.0}), seeds, jobs
    )
    comparison = shrinkage_comparison(pooled, pooled_audit(unshrunk_study.inputs, config.fairness))
    worst = (
        float(study.rows["adjusted_gap_points"].abs().max()) if len(study.rows) else float("nan")
    )
    workers = int(study.inputs[["store", "worker_id"]].drop_duplicates().shape[0])
    text = (
        "Score gap in points between a group and everyone else at the same hidden skill (skill "
        "is the speed actually shown in the period for speed and combined scores, minus the "
        f"error propensity for accuracy), from one regression over {workers} workers in "
        f"{n_stores} independent stores with a separate intercept per store and 95% intervals. "
        f"Configured limit: {config.fairness.max_adjusted_gap_points:.0f} points. Largest "
        f"single-store gap for a group of at least {config.fairness.min_group_size} workers: "
        f"{worst:.1f} points (single stores are noisy, which is why the pooled estimate is the "
        "one to read).\n\n" + markdown_table(pooled, 2)
    )
    if not comparison.empty:
        text += (
            "\n\nSpeed scores by tenure band again with speed shrinkage switched off "
            "(`engine.speed.shrinkage_k` = 0), on the same stores:\n\n"
            + markdown_table(comparison, 2)
        )
    return text, pooled


def coverage_section(config: Config, reduced: bool, jobs: int) -> tuple[str, pd.DataFrame]:
    """Run the interval calibration on fresh replicates and describe the coverage."""
    replicates = 4 if reduced else config.experiments.calibration_replicates
    data = run_replicates(_calibration_config(config, reduced), replicates, jobs)
    coverage = coverage_from(data)
    dispersion = dispersion_from(data)
    summary = dispersion.summary.reset_index()[["component", "n", "median_ratio", "pooled_ratio"]]
    by_context = coverage.coverage_by_context
    text = (
        f"{replicates} replicates of fresh data for the same hidden cashiers "
        "(`experiments/calibration.py`). "
        f"A {coverage.nominal_level:.0%} interval contained the mean score on the other "
        f"replicates {coverage.replicate_coverage:.1%} of the time over {coverage.n_intervals} "
        f"intervals (peak {by_context['peak']:.1%}, off-peak {by_context['off']:.1%}); "
        f"a perfectly calibrated interval would score about {coverage.benchmark_coverage:.1%} "
        f"at this replicate count. Mean interval width {coverage.mean_width:.1f} points.\n\n"
        "Ratio of the replicate-to-replicate SD of a score to the bootstrap SE "
        "(1.0 is calibrated):\n\n" + markdown_table(summary, 2)
    )
    return text, summary


def output_directories(config: Config, reduced: bool) -> tuple[Path, Path]:
    """Directories of the reports and figures, created when missing; reduced runs have their own."""
    root = Path(config.paths.outputs_dir) / ("reduced" if reduced else "")
    reports, pictures = root / "reports", root / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    pictures.mkdir(parents=True, exist_ok=True)
    return reports, pictures


def run_all(
    config: Config,
    reduced: bool = False,
    jobs: int | None = None,
    seeds: int | None = None,
    make_figures: bool = True,
) -> Path:
    """Run every experiment, write the tables, figures and results.md, and return the report path.

    A reduced run uses fewer cashiers, weeks and seeds and is for development only.
    """
    workers = jobs if jobs is not None else config.experiments.jobs
    workers = max(1, min(workers, os.cpu_count() or 1))
    n_seeds = (
        seeds
        if seeds is not None
        else (config.experiments.reduced_seeds if reduced else config.experiments.n_seeds)
    )
    reports, pictures = output_directories(config, reduced)
    metrics, accuracy, scatter = run_scenarios(config, reduced, workers, n_seeds)
    curve = run_learning_curve(
        config, reduced, workers, 1 if reduced else min(n_seeds, LEARNING_SEEDS_FULL)
    )
    coverage_text, coverage_table = coverage_section(config, reduced, workers)
    fairness_text, fairness_table = fairness_section(config, reduced, workers)

    metrics.to_csv(reports / "results_by_seed.csv", index=False)
    accuracy.to_csv(reports / "accuracy_by_seed.csv", index=False)
    curve.to_csv(reports / "learning_curve.csv", index=False)
    coverage_table.to_csv(reports / "interval_calibration.csv", index=False)
    fairness_table.to_csv(reports / "fairness_audit.csv", index=False)

    figure_names: list[str] = []
    if make_figures:
        level = config.experiments.confidence_level
        summary = summarize(metrics, HEADLINE, level)
        figure_names = figures.write_experiment_figures(
            config, pictures, scatter, summary, curve, coverage_table, level
        )
    report = build_report(
        config,
        reduced,
        n_seeds,
        metrics,
        accuracy,
        curve,
        coverage_text,
        fairness_text,
        figure_names,
    )
    target = reports / "results.md"
    target.write_bytes(report.encode("utf-8"))
    print(f"wrote {target}")
    return target


SMOKE_CASHIERS = 8
SMOKE_WEEKS = 4
SMOKE_BOOTSTRAP = 60
SMOKE_MIN_RECOVERY = 0.8


@dataclass
class SmokeStep:
    """Outcome of one step of the smoke check."""

    name: str
    passed: bool
    seconds: float
    detail: str


def _run_step(name: str, action: Callable[[], str]) -> SmokeStep:
    started = time.perf_counter()
    try:
        detail = action()
        passed = True
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        passed = False
    return SmokeStep(name, passed, time.perf_counter() - started, detail)


def _smoke_experiment(config: Config, jobs: int) -> str:
    metrics, _, _ = run_scenarios(config, True, jobs, 1)
    latent = metrics[metrics["target"] == LATENT]
    by_method = latent.groupby("method")["spearman"].mean()
    biased = latent[latent["scenario"] == "biased"].groupby("method")["spearman"].mean()
    if by_method["v2_ols_pause"] < SMOKE_MIN_RECOVERY:
        raise AssertionError(f"v2 recovered only {by_method['v2_ols_pause']:.2f}")
    if by_method["v2_ols_pause"] <= by_method["hard_5s_stop"]:
        raise AssertionError("v2 does not beat the hard 5 s stop")
    if biased["naive_items_per_shift"] >= biased["v2_ols_pause"]:
        raise AssertionError("biased assignment did not fool the naive baseline")
    return (
        f"v2 {by_method['v2_ols_pause']:.2f}, hard stop {by_method['hard_5s_stop']:.2f}, "
        f"naive under biased assignment {biased['naive_items_per_shift']:.2f}"
    )


def run_smoke(config: Config) -> int:
    """End-to-end check on a small store.

    It simulates, scores twice, queries the API, runs a reduced experiment and opens the
    review screen. Returns 0 when every step passes.
    """
    from ree.api.app import create_app
    from ree.api.probe import get_json
    from ree.engine.pipeline import score_directory
    from ree.simulator.run import simulate_to_directory
    from ree.storage import db
    from ree.ui.smoke import run_ui_smoke

    smoke = config.with_overrides(
        {
            "simulator.population.n_cashiers": SMOKE_CASHIERS,
            "simulator.n_weeks": SMOKE_WEEKS,
            "paths.store_name": "smoke",
            "engine.intervals.n_bootstrap": SMOKE_BOOTSTRAP,
        }
    )
    directory = db.store_directory(smoke)
    jobs = max(1, min(config.experiments.jobs, os.cpu_count() or 1))

    def simulate() -> str:
        simulate_to_directory(smoke)
        data = db.load_store_data(directory)
        if len(data.transactions) < 500:
            raise AssertionError("too few transactions")
        return f"{len(data.transactions)} transactions, {len(data.events)} events"

    def score() -> str:
        score_directory(smoke)
        first = (directory / db.SCORES_FILE).read_bytes()
        score_directory(smoke)
        if (directory / db.SCORES_FILE).read_bytes() != first:
            raise AssertionError("rerun is not byte-identical")
        connection = db.connect(directory)
        try:
            records = db.load_scores(connection)
            audit = db.load_audit_log(connection)
        finally:
            connection.close()
        if len(records) != SMOKE_CASHIERS or len(audit) != 1:
            raise AssertionError("unexpected number of records or audit rows")
        return f"{len(records)} records, byte-identical rerun, 1 audit row"

    def api() -> str:
        app = create_app(smoke)
        health_status, health = get_json(app, "/health")
        ranking_status, ranking = get_json(app, "/scores/ranking", "context=peak")
        if health_status != 200 or ranking_status != 200:
            raise AssertionError("API did not answer")
        if health["status"] != "ok" or len(ranking["ranking"]) != SMOKE_CASHIERS:
            raise AssertionError("API answered with an unexpected body")
        return f"{len(ranking['ranking'])} workers ranked"

    def interface() -> str:
        problems = run_ui_smoke(smoke, directory / "ui")
        broken = {view: errors for view, errors in problems.items() if errors}
        if broken:
            raise AssertionError(str(broken))
        return f"{len(problems) - 1} views load without errors"

    steps = [
        _run_step("simulate a small store", simulate),
        _run_step("score it (twice)", score),
        _run_step("score API", api),
        _run_step("reduced experiment", lambda: _smoke_experiment(config, jobs)),
        _run_step("review screen", interface),
    ]
    width = max(len(step.name) for step in steps)
    print()
    for step in steps:
        mark = "PASS" if step.passed else "FAIL"
        print(f"[{mark}] {step.name.ljust(width)}  {step.seconds:5.1f} s  {step.detail}")
    passed = all(step.passed for step in steps)
    total = sum(step.seconds for step in steps)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'} in {total:.0f} s")
    return 0 if passed else 1
