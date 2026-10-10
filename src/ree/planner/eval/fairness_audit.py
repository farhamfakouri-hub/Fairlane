"""Fairness audit of the roster: burden, hours, pay and score effects, with a negative control."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ree.config import make_rng
from ree.planner.config import PlannerConfig
from ree.planner.domain.plan import RosterPlan
from ree.planner.eval.plans import PlanInputs, build_inputs, build_plan
from ree.planner.eval.stats import bootstrap_mean
from ree.planner.roster.explain import rush_minutes
from ree.planner.roster.fairness import (
    burden_rates,
    burden_spread,
    hours_deviation,
    hours_score_correlation,
    max_tier_gap,
    score_correlations,
    tier_hours_gap,
)
from ree.planner.roster.model import ScoreInputs
from ree.planner.roster.solver import RosterSolution, plan_roster, solve_coverage

AUDIT_SEEDS = 6
NOISE_LEVELS = (0.25, 0.5, 1.0)
REPORT_FILE = "fairness_audit.md"
TABLE_FILE = "fairness_audit.csv"


def _rush_by_worker(plan: RosterPlan, inputs: PlanInputs) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for assignment in plan.assignments:
        totals[assignment.worker_id] += rush_minutes(
            assignment.pattern, assignment.day, inputs.requirement
        )
    return totals


def _paid_hours(plan: RosterPlan) -> dict[str, float]:
    paid: dict[str, float] = defaultdict(float)
    for assignment in plan.assignments:
        paid[assignment.worker_id] += assignment.pattern.paid_minutes / 60.0
    return paid


def _noisy(scores: ScoreInputs, sigma: float, seed: int, clip: float) -> ScoreInputs:
    rng = make_rng(seed, f"planner-audit-noise:{sigma}")
    ids = sorted(scores.z)
    noise = rng.normal(0.0, sigma, len(ids))
    z = {
        w: (
            scores.z[w]
            if w in scores.exploration
            else float(np.clip(scores.z[w] + noise[i], -clip, clip))
        )
        for i, w in enumerate(ids)
    }
    return ScoreInputs(z, dict(scores.tiers), scores.exploration)


def _solve(
    inputs: PlanInputs, scores: ScoreInputs, config: PlannerConfig | None = None
) -> RosterSolution:
    used = config if config is not None else inputs.config
    start = inputs.coverage.get("forecast")
    if used.solver.coverage_first and start is None:
        start = solve_coverage(inputs.workers, inputs.requirement, used, weeks=1)
        inputs.coverage["forecast"] = start
    return plan_roster(
        inputs.workers, inputs.requirement, used, scores, weeks=1, coverage_start=start
    )


def _row(
    seed: int,
    variant: str,
    solution: RosterSolution,
    inputs: PlanInputs,
    reference: RosterSolution,
    baseline: RosterSolution,
) -> dict[str, object]:
    plan = solution.plan
    rates = burden_rates(plan, inputs.config)
    tiers = {w: t for w, t in inputs.scores.tiers.items() if w in rates}
    rated = {w: z for w, z in inputs.scores.z.items() if w in rates}
    correlations = score_correlations(plan, rated, inputs.config)
    rush = _rush_by_worker(plan, inputs)
    truth = inputs.workforce.truth
    total_rush = sum(rush.values()) or 1
    alignment = sum(truth[w].log_speed * minutes for w, minutes in rush.items()) / total_rush
    exploring = [rush.get(w, 0) for w in sorted(inputs.scores.exploration) if w in rates]
    mine = {(a.worker_id, a.day, a.pattern.pattern_id) for a in plan.assignments}
    other = {(a.worker_id, a.day, a.pattern.pattern_id) for a in reference.plan.assignments}
    days = {(a.worker_id, a.day) for a in plan.assignments}
    base_days = {(a.worker_id, a.day) for a in baseline.plan.assignments}
    base_patterns = {(a.worker_id, a.day): a.pattern.pattern_id for a in baseline.plan.assignments}
    swapped = sum(
        1
        for a in plan.assignments
        if base_patterns.get((a.worker_id, a.day), a.pattern.pattern_id) != a.pattern.pattern_id
    )
    paid, base_paid = _paid_hours(plan), _paid_hours(baseline.plan)
    blocked = {w.worker_id: len(w.blocked) for w in inputs.workers}
    ids = sorted(rates)
    availability = (
        float(stats.spearmanr([blocked[w] for w in ids], [rates[w] for w in ids]).statistic)
        if len({blocked[w] for w in ids}) > 1
        else 0.0
    )
    return {
        "seed": seed,
        "variant": variant,
        "max_tier_gap": max_tier_gap(rates, tiers),
        "burden_spread": burden_spread(rates),
        "max_hours_deviation": max(hours_deviation(plan).values()),
        "paid_difference_vs_B": max(abs(paid[w] - base_paid[w]) for w in paid),
        "days_changed_vs_B": float(len(days ^ base_days)),
        "patterns_changed_vs_B": swapped / max(len(plan.assignments), 1),
        "corr_weekend": correlations["weekend"],
        "corr_late": correlations["late"],
        "corr_split": correlations["split"],
        "corr_burden": correlations["burden"],
        "corr_hours": hours_score_correlation(plan, rated),
        "tier_hours_gap": tier_hours_gap(plan, tiers, "A", "C"),
        "corr_availability_burden": availability,
        "exploration_rush_minutes": float(np.mean(exploring)) if exploring else 0.0,
        "true_alignment": alignment,
        "overlap_with_C": len(mine & other) / max(len(mine | other), 1),
        "shortfall_minutes": float(solution.report.shortfall_minutes()) if solution.report else 0.0,
    }


def audit_seed(config: PlannerConfig, seed: int) -> list[dict[str, object]]:
    """Build the plans and variants of one seed and measure each."""
    inputs = build_inputs(config, seed)
    bundles = {name: build_plan(inputs, name) for name in ("B", "C", "E")}
    baseline = bundles["B"].roster
    reference = bundles["C"].roster
    rows = [
        _row(seed, name, bundle.roster, inputs, reference, baseline)
        for name, bundle in bundles.items()
    ]
    clip = config.objective.score_clip
    for sigma in NOISE_LEVELS:
        noisy = _noisy(inputs.scores, sigma, seed, clip)
        rows.append(
            _row(seed, f"C with noise {sigma}", _solve(inputs, noisy), inputs, reference, baseline)
        )
    unquota = config.with_overrides({"objective.exploration_hours_per_week": 0.0})
    rows.append(
        _row(
            seed,
            "C without learning quota",
            _solve(inputs, inputs.scores, unquota),
            inputs,
            reference,
            baseline,
        )
    )
    control = config.with_overrides({"objective.neutral_swaps": False})
    rows.append(
        _row(
            seed,
            CONTROL,
            _solve(inputs, inputs.scores, control),
            inputs,
            reference,
            baseline,
        )
    )
    return rows


def _job(job: tuple[PlannerConfig, int]) -> list[dict[str, object]]:
    return audit_seed(*job)


def run_audit(
    config: PlannerConfig, seeds: Sequence[int], jobs: int = 2, progress: bool = False
) -> pd.DataFrame:
    """Run the audit over seeds, in parallel if asked."""
    work = [(config, seed) for seed in seeds]
    rows: list[dict[str, object]] = []
    if jobs <= 1 or len(work) == 1:
        for done, item in enumerate(work, start=1):
            rows.extend(_job(item))
            if progress:
                print(f"audit seed {item[1]} done ({done}/{len(work)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, chunk in enumerate(pool.map(_job, work), start=1):
                rows.extend(chunk)
                if progress:
                    print(f"audit {done}/{len(work)} seeds done", flush=True)
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


BURDEN_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("max_tier_gap", "Largest tier gap in burden", 3),
    ("burden_spread", "Burden spread per paid hour", 3),
    ("max_hours_deviation", "Largest hours deviation (h)", 2),
    ("paid_difference_vs_B", "Largest paid-hours difference to plan B (h)", 2),
    ("tier_hours_gap", "Paid minus contract hours, tier A minus tier C (h)", 2),
    ("days_changed_vs_B", "Worker-days that differ from plan B", 1),
    ("patterns_changed_vs_B", "Share of shifts that differ from plan B", 2),
)
EFFECT_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("corr_weekend", "Score against weekend share", 2),
    ("corr_late", "Score against late share", 2),
    ("corr_split", "Score against split share", 2),
    ("corr_burden", "Score against burden", 2),
    ("corr_hours", "Score against hours deviation", 2),
    ("corr_availability_burden", "Blocked days against burden", 2),
    ("exploration_rush_minutes", "Rush minutes of new workers", 0),
    ("true_alignment", "True log-speed per rush minute", 3),
    ("overlap_with_C", "Shifts shared with plan C", 2),
)
ORDER = (
    "B",
    "C",
    "E",
    "C with noise 0.25",
    "C with noise 0.5",
    "C with noise 1.0",
    "C without learning quota",
    "C with hours-changing swaps",
)
CONTROL = "C with hours-changing swaps"
GUARDED = tuple(name for name in ORDER if name.startswith("C") and name != CONTROL)


def _table(frame: pd.DataFrame, columns: tuple[tuple[str, str, int], ...]) -> list[str]:
    header = ["Variant", *[label for _, label, _ in columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for variant in ORDER:
        part = frame[frame["variant"] == variant].sort_values("seed")
        if part.empty:
            continue
        cells = [variant]
        for column, _, digits in columns:
            mean, low, high = bootstrap_mean(part[column].to_list())
            cells.append(f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _by_seed(frame: pd.DataFrame, variant: str, column: str) -> pd.Series:
    part = frame[frame["variant"] == variant].sort_values("seed")
    return part.set_index("seed")[column]


def audit_reading(frame: pd.DataFrame, config: PlannerConfig) -> list[str]:
    """Plain sentences saying what the audit found."""
    if not {"B", "C", "E"} <= set(frame["variant"]):
        return []
    seeds = int(frame["seed"].nunique())
    gap_c, gap_b = _by_seed(frame, "C", "max_tier_gap"), _by_seed(frame, "B", "max_tier_gap")
    above = int((gap_c > gap_b).sum())
    lines = [
        f"- Tier gap: plan C's largest tier gap is above plan B's in {above} of {seeds} seeds "
        f"(mean difference {float((gap_c - gap_b).mean()):+.3f})."
        + (
            " The guard keeps it inside the limit; it does not bring it down to the level of a "
            "plan that ignores the scores."
            if above * 2 > seeds
            else ""
        )
    ]
    for column, label in (
        ("corr_burden", "burden per paid hour"),
        ("corr_late", "share of late shifts"),
    ):
        c, b, e = (_by_seed(frame, name, column) for name in ("C", "B", "E"))
        lines.append(
            f"- Score against {label}: plan C {float(c.mean()):+.2f} (below zero in "
            f"{int((c < 0).sum())} of {seeds} seeds), plan B {float(b.mean()):+.2f}, placebo E "
            f"{float(e.mean()):+.2f}. A value below zero means that workers with higher scores "
            f"carry less of it."
        )
    same = all(
        np.allclose(_by_seed(frame, variant, column), _by_seed(frame, "B", column))
        for variant in ORDER
        if variant in set(frame["variant"])
        for column in ("corr_weekend", "corr_split")
    )
    hours = {name: _by_seed(frame, name, "corr_hours") for name in ("C", "B", "E")}
    gap_hours = _by_seed(frame, "C", "tier_hours_gap")
    lines.append(
        f"- Hours: the correlation of the score with the signed deviation of paid hours from the "
        f"contract (inside the tolerance of {config.rules.hours_tolerance_hours:.1f} h) is "
        f"{float(hours['C'].mean()):+.2f} in plan C, {float(hours['B'].mean()):+.2f} in plan B "
        f"and {float(hours['E'].mean()):+.2f} in the placebo E; paid minus contract hours of "
        f"tier A minus those of tier C: {float(gap_hours.mean()):+.2f} h a week in plan C."
    )
    changed_days = float(_by_seed(frame, "C", "days_changed_vs_B").mean())
    changed_patterns = float(_by_seed(frame, "C", "patterns_changed_vs_B").mean())
    lines.append(
        f"- What the scores change: against plan B, plan C moves {changed_days:.0f} worker-days "
        f"(a person working on a day when plan B does not) and gives a different shift of the "
        f"day to {changed_patterns:.0%} of its shifts."
        + (
            " The correlations of the score with the weekend share and with the split share are "
            "the same in every variant."
            if same
            else ""
        )
    )
    if CONTROL in set(frame["variant"]):
        control = _by_seed(frame, CONTROL, "corr_hours")
        control_gap = _by_seed(frame, CONTROL, "tier_hours_gap")
        lines.append(
            f"- Negative control: when shifts of different length and burden may be swapped "
            f"(`objective.neutral_swaps: false`, the behaviour of the first version), the "
            f"correlation of the score with the hours deviation is {float(control.mean()):+.2f} "
            f"and tier A minus tier C is {float(control_gap.mean()):+.2f} h a week, so the audit "
            f"detects a score that buys hours; the default swaps (equal hours, equal burden) "
            f"show {float(hours['C'].mean()):+.2f} and {float(gap_hours.mean()):+.2f} h."
        )
    quota = float(_by_seed(frame, "C", "exploration_rush_minutes").mean())
    floor = config.objective.exploration_hours_per_week * 60
    identical = bool(
        (_by_seed(frame, "C without learning quota", "overlap_with_C") == 1.0).all()
        if "C without learning quota" in set(frame["variant"])
        else False
    )
    lines.append(
        f"- Learning quota (a floor of {floor:.0f} rush minutes a week for each new worker): new "
        f"workers get {quota:.0f} rush minutes a week in plan C"
        + (
            ", and the roster without the quota is identical to plan C in every seed, so the "
            "quota does not bind here and this audit cannot show what it does."
            if identical
            else "."
        )
    )
    noisy = [
        name for name in ORDER if name.startswith("C with noise") and name in set(frame["variant"])
    ]
    if noisy:
        survive = ", ".join(
            f"{float(_by_seed(frame, name, 'overlap_with_C').mean()):.0%}" for name in noisy
        )
        levels = ", ".join(name.removeprefix("C with noise ") for name in noisy)
        unchanged = all(
            bool((_by_seed(frame, name, "overlap_with_C") == 1.0).all()) for name in noisy
        )
        lines.append(
            f"- Noise in the scores: with normal noise of {levels} (in z units) added to the "
            f"scores the planner sees, {survive} of plan C's shifts survive"
            + (
                ", so in these seeds the roster does not respond to the scores at all."
                if unchanged
                else "."
            )
        )
    return lines


def _paid_limit(config: PlannerConfig) -> float:
    if config.solver.coverage_first:
        return 0.0
    return 2 * config.rules.hours_tolerance_hours


def audit_passed(frame: pd.DataFrame, config: PlannerConfig) -> bool:
    """Whether the burden guard, the hours tolerance and the pay limit all hold."""
    guarded = frame[frame["variant"].isin(GUARDED)]
    scored = frame[frame["variant"] != CONTROL]
    return (
        float(guarded["max_tier_gap"].max()) <= config.objective.tier_epsilon
        and float(frame["max_hours_deviation"].max()) <= config.rules.hours_tolerance_hours + 1e-9
        and float(scored["paid_difference_vs_B"].max()) <= _paid_limit(config) + 1e-9
    )


def audit_report(frame: pd.DataFrame, config: PlannerConfig) -> str:
    """Markdown text of the audit."""
    limit = config.objective.tier_epsilon
    tolerance = config.rules.hours_tolerance_hours
    burden_lines = _table(frame, BURDEN_COLUMNS)
    effect_lines = _table(frame, EFFECT_COLUMNS)
    guarded = frame[frame["variant"].isin(GUARDED)]
    worst_gap = float(guarded["max_tier_gap"].max())
    unguarded_gap = float(frame[frame["variant"].isin(("B", "E"))]["max_tier_gap"].max())
    worst_hours = float(frame["max_hours_deviation"].max())
    worst_paid = float(frame[frame["variant"] != CONTROL]["paid_difference_vs_B"].max())
    paid_rule = (
        "none allowed: the score-blind coverage stage fixes the hours"
        if config.solver.coverage_first
        else "at most twice the tolerance"
    )
    checks = [
        f"- Largest score-tier burden gap over the plans that use the scores (C and its "
        f"variants) and every seed: {worst_gap:.3f} (limit {limit:.2f}): "
        f"{'within the limit' if worst_gap <= limit else 'ABOVE THE LIMIT'}.",
        f"- For comparison, plan B (no tiers) and the placebo E (shuffled tiers) reach "
        f"{unguarded_gap:.3f}, because no guard acts on the real tiers there.",
        f"- Largest hours deviation from a contract: {worst_hours:.2f} h "
        f"(tolerance {tolerance:.1f} h): "
        f"{'within the tolerance' if worst_hours <= tolerance + 1e-9 else 'ABOVE THE TOLERANCE'}.",
        f"- Largest paid-hours difference between a worker's weekly total in any variant (the "
        f"negative control apart) and in plan B: {worst_paid:.2f} h "
        f"({paid_rule}): "
        f"{'holds' if worst_paid <= _paid_limit(config) + 1e-9 else 'DOES NOT HOLD'}.",
    ]
    return "\n".join(
        [
            "# Fairness audit of the roster",
            "",
            "Generated by `python -m ree.cli planner audit`. Means over seeds with 95 percent "
            "bootstrap intervals in brackets; correlations are Spearman rank correlations "
            "between a worker's Part 1 score and the share of their shifts that are on a "
            "weekend, late or split, or their burden per paid hour.",
            "",
            "## Checks",
            "",
            *checks,
            "",
            "## Reading",
            "",
            *audit_reading(frame, config),
            "",
            "## Burden, hours and pay",
            "",
            *burden_lines,
            "",
            "## What the scores change and how it correlates with them",
            "",
            *effect_lines,
            "",
            "Rows C with noise 0.25 to 1.0 add normal noise (in z units) to the scores the "
            "planner sees; the last column shows how much of plan C survives. The true "
            "log-speed per rush minute (true skills are hidden from the planner) shows whether "
            "the strong workers still get the rush slots when the scores are noisy.",
            "",
        ]
    )


def audit_table_path(config: PlannerConfig, reduced: bool) -> Path:
    """Path of the audit table."""
    root = Path(config.paths.outputs_dir)
    return root / ("reduced" if reduced else "reports") / TABLE_FILE


def write_audit(
    frame: pd.DataFrame, config: PlannerConfig, reduced: bool, save_table: bool = True
) -> list[Path]:
    """Write the audit table and report."""
    table = audit_table_path(config, reduced)
    table.parent.mkdir(parents=True, exist_ok=True)
    report = table.parent / REPORT_FILE
    if save_table:
        frame.to_csv(table, index=False)
    report.write_text(audit_report(frame, config), encoding="utf-8")
    return [table, report] if save_table else [report]
