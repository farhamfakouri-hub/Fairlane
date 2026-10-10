"""Run the evaluation of the plans over seeds and scenarios."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ree.planner.config import PlannerConfig
from ree.planner.eval.live_runner import simulate_plan_live
from ree.planner.eval.metrics import (
    priority_true_gap,
    register_staffing,
    roster_metrics,
    week_metrics,
)
from ree.planner.eval.plans import (
    EVALUATED_PLANS,
    LIVE_PLAN,
    PlanBundle,
    build_inputs,
    build_plan,
)
from ree.planner.eval.world import SCENARIOS, build_world, scenario_config, simulate_plan
from ree.planner.sim.scenarios import store_config

DEFAULT_STORE = "eval"
FIRST_SEED = 1
REDUCED_SCENARIOS = ("normal_week", "saturday_surge")
RESULTS_FILE = "results.csv"
LIVE_COLUMNS = (
    "live_proposed",
    "live_accepted",
    "live_opened",
    "live_closed",
    "live_covered",
    "live_flips",
)


def evaluation_config(store: str = DEFAULT_STORE, **extra: object) -> PlannerConfig:
    """Configuration of the evaluation store."""
    return store_config(store, **extra)


def evaluate_seed(
    config: PlannerConfig,
    seed: int,
    scenarios: Sequence[str],
    plans: Sequence[str],
    live: bool = True,
) -> list[dict[str, object]]:
    """Build and test every plan on every scenario for one seed; one row for each."""
    groups: dict[str, list[str]] = {}
    for name in scenarios:
        groups.setdefault(SCENARIOS[name].group, []).append(name)
    target = config.evaluation.wait_target_minutes * 60.0
    rows: list[dict[str, object]] = []
    for names in groups.values():
        sample = SCENARIOS[names[0]]
        group_config = scenario_config(config, sample)
        inputs = build_inputs(group_config, seed, team_share=sample.team_share)
        bundles: dict[str, PlanBundle] = {}
        for plan in plans:
            bundles[plan] = build_plan(inputs, plan, base=bundles.get("B"))
        tiers = inputs.scores.tiers
        for name in names:
            scenario = SCENARIOS[name]
            with_live = live and "C" in bundles
            collected: dict[str, list[dict[str, float]]] = {plan: [] for plan in bundles}
            if with_live:
                collected[LIVE_PLAN] = []
            worlds = [
                build_world(
                    group_config,
                    scenario,
                    seed,
                    [w.worker_id for w in inputs.workers],
                    replication=replication,
                )
                for replication in range(config.evaluation.replications)
            ]
            for world in worlds:
                for plan, bundle in bundles.items():
                    runs = simulate_plan(bundle, world, inputs.workforce.truth)
                    collected[plan].append(week_metrics(runs, target))
                if with_live:
                    runs, stats = simulate_plan_live(bundles["C"], world, inputs.workforce.truth)
                    collected[LIVE_PLAN].append(
                        week_metrics(runs, target)
                        | {
                            "live_proposed": float(stats.proposed),
                            "live_accepted": float(stats.accepted),
                            "live_opened": float(stats.opened),
                            "live_closed": float(stats.closed),
                            "live_covered": float(stats.covered),
                            "live_flips": float(stats.flips),
                        }
                    )
            for plan in collected:
                bundle = bundles["C" if plan == LIVE_PLAN else plan]
                rows.append(
                    result_row(
                        seed, name, plan, collected[plan], bundle, group_config, tiers, inputs
                    )
                )
            if name == "normal_week":
                rows.extend(
                    layout_rows(seed, bundles, worlds, group_config, inputs, target, tiers, config)
                )
    return rows


def result_row(seed, scenario, plan, results, bundle, group_config, tiers, inputs):
    """One result row: averaged week metrics with the plan's roster measures."""
    averaged = {key: float(np.mean([item[key] for item in results])) for key in results[0]}
    return {
        "seed": seed,
        "scenario": scenario,
        "plan": plan,
        **{key: 0.0 for key in LIVE_COLUMNS},
        **averaged,
        **roster_metrics(bundle, group_config, tiers),
        **register_staffing(bundle),
        "priority_true_gap": priority_true_gap(bundle, inputs.workforce.truth),
    }


def layout_rows(seed, bundles, worlds, group_config, inputs, target, tiers, config):
    """Rows for the queue-layout sensitivity runs."""
    rows: list[dict[str, object]] = []
    for penalty in config.evaluation.layout_penalties:
        variant = f"layout_{penalty:g}"
        changed = group_config.with_overrides({"service.position_penalty": penalty})
        for plan, bundle in bundles.items():
            results = [
                week_metrics(
                    simulate_plan(
                        bundle, dataclasses.replace(world, config=changed), inputs.workforce.truth
                    ),
                    target,
                )
                for world in worlds
            ]
            rows.append(
                result_row(seed, variant, plan, results, bundle, group_config, tiers, inputs)
            )
    return rows


def _evaluate_job(job: tuple[PlannerConfig, int, Sequence[str], Sequence[str], bool]):
    return evaluate_seed(*job)


def run_evaluation(
    config: PlannerConfig,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    plans: Sequence[str] = EVALUATED_PLANS,
    jobs: int = 2,
    progress: bool = False,
    live: bool = True,
) -> pd.DataFrame:
    """Evaluate many seeds, in parallel if asked, and return one table."""
    work = [(config, seed, tuple(scenarios), tuple(plans), live) for seed in seeds]
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    if jobs <= 1 or len(work) == 1:
        for done, job in enumerate(work, start=1):
            rows.extend(_evaluate_job(job))
            if progress:
                elapsed = time.perf_counter() - started
                print(f"seed {job[1]} done ({done}/{len(work)}, {elapsed:.0f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, chunk in enumerate(pool.map(_evaluate_job, work), start=1):
                rows.extend(chunk)
                if progress:
                    print(
                        f"{done}/{len(work)} seeds done ({time.perf_counter() - started:.0f} s)",
                        flush=True,
                    )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["scenario", "plan", "seed"]).reset_index(drop=True)


def results_path(config: PlannerConfig, reduced: bool) -> Path:
    """Path of the results table."""
    folder = "reduced" if reduced else "reports"
    return Path(config.paths.outputs_dir) / folder / RESULTS_FILE


def evaluation_plan(
    config: PlannerConfig, reduced: bool, seeds: int | None = None
) -> tuple[list[int], tuple[str, ...]]:
    """Seeds and scenarios to run, fewer for a reduced run."""
    count = (
        seeds
        if seeds is not None
        else (config.evaluation.reduced_seeds if reduced else config.evaluation.n_seeds)
    )
    scenarios = REDUCED_SCENARIOS if reduced else tuple(config.evaluation.scenarios)
    return list(range(FIRST_SEED, FIRST_SEED + count)), scenarios
