"""Markdown report of the planner evaluation, built from the results table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ree.planner.config import PlannerConfig
from ree.planner.eval.figures import SCENARIO_LABELS
from ree.planner.eval.plans import ALL_PLANS, EVALUATED_PLANS, PLAN_LABELS
from ree.planner.eval.stats import (
    BOOTSTRAP_REPLICATES,
    PairedDifference,
    bootstrap_mean,
    paired_bootstrap,
)
from ree.planner.eval.world import SCENARIOS

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("B", "A", "demand-aware against headcount only"),
    ("C", "B", "full planner against demand-aware without scores"),
    ("E", "B", "placebo (shuffled scores) against demand-aware without scores"),
    ("C", "E", "full planner against the placebo"),
    ("D", "C", "oracle (true skills) against the full planner"),
    ("L", "C", "live re-planning against the same plan without it"),
)
TABLE_METRICS: tuple[tuple[str, str, float], ...] = (
    ("mean_wait_s", "Mean wait (s)", 1.0),
    ("p95_wait_s", "95th percentile wait (s)", 1.0),
    ("share_over_target", "Waited over 3 min or left (%)", 100.0),
    ("abandon_rate", "Left without service (%)", 100.0),
    ("rush_mean_wait_s", "Rush mean wait (s)", 1.0),
)
QUEUE_METRICS: tuple[tuple[str, str, float, int], ...] = (
    ("mean_queue_per_register", "Customers waiting per open register", 1.0, 2),
    ("share_time_over_limit", "Open day with more than 5 waiting at a register (%)", 100.0, 1),
    ("idle_share", "Idle share of open-register time (%)", 100.0, 1),
)
VERDICT_METRICS: tuple[tuple[str, str, float], ...] = (
    ("mean_wait_s", "mean wait", 1.0),
    ("share_over_target", "share waiting over 3 minutes or leaving", 100.0),
)


@dataclass(frozen=True)
class Tally:
    """Counts of scenarios in which one plan is lower, tied or higher than another."""

    lower: int
    tie: int
    higher: int


def _vector(frame: pd.DataFrame, scenario: str, plan: str, metric: str) -> list[float]:
    part = frame[(frame["scenario"] == scenario) & (frame["plan"] == plan)].sort_values("seed")
    return [float(v) for v in part[metric]]


def _scenarios(frame: pd.DataFrame) -> list[str]:
    present = set(frame["scenario"])
    return [name for name in SCENARIO_LABELS if name in present]


def _difference(
    frame: pd.DataFrame, scenario: str, first: str, second: str, metric: str, scale: float
) -> PairedDifference:
    return paired_bootstrap(
        [v * scale for v in _vector(frame, scenario, first, metric)],
        [v * scale for v in _vector(frame, scenario, second, metric)],
    )


def tally(
    frame: pd.DataFrame, first: str, second: str, metric: str, scale: float
) -> tuple[Tally, dict[str, PairedDifference]]:
    """Compare two plans in every scenario and count the verdicts."""
    differences = {
        scenario: _difference(frame, scenario, first, second, metric, scale)
        for scenario in _scenarios(frame)
    }
    verdicts = [d.verdict for d in differences.values()]
    return (
        Tally(
            verdicts.count("lower"), verdicts.count("no clear difference"), verdicts.count("higher")
        ),
        differences,
    )


def _interval(values: Sequence[float], digits: int = 1) -> str:
    mean, low, high = bootstrap_mean(values)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def results_table(frame: pd.DataFrame) -> list[str]:
    """Wait results for each scenario and plan."""
    header = ["Scenario", "Plan", *[label for _, label, _ in TABLE_METRICS]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        for plan in ALL_PLANS:
            if plan not in set(frame["plan"]):
                continue
            cells = [SCENARIO_LABELS[scenario], PLAN_LABELS[plan]]
            for metric, _, scale in TABLE_METRICS:
                values = [v * scale for v in _vector(frame, scenario, plan, metric)]
                cells.append(_interval(values, 1 if scale == 1.0 else 2))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def difference_table(frame: pd.DataFrame) -> list[str]:
    """Paired differences between plans for each scenario."""
    header = ["Scenario", "Comparison", "Mean wait (s)", "Share over 3 min (points)", "Reading"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        for first, second, _ in PAIRS:
            if first not in set(frame["plan"]) or second not in set(frame["plan"]):
                continue
            wait = _difference(frame, scenario, first, second, "mean_wait_s", 1.0)
            share = _difference(frame, scenario, first, second, "share_over_target", 100.0)
            reading = _reading(wait, share)
            lines.append(
                f"| {SCENARIO_LABELS[scenario]} | {first} minus {second} | "
                f"{wait.mean:+.1f} [{wait.low:+.1f}, {wait.high:+.1f}] | "
                f"{share.mean:+.2f} [{share.low:+.2f}, {share.high:+.2f}] | {reading} |"
            )
    return lines


def queue_table(frame: pd.DataFrame) -> list[str]:
    """Queue length and idle share for each scenario and plan."""
    header = ["Scenario", "Plan", *[label for _, label, _, _ in QUEUE_METRICS]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        for plan in ALL_PLANS:
            if plan not in set(frame["plan"]):
                continue
            cells = [SCENARIO_LABELS[scenario], PLAN_LABELS[plan]]
            for metric, _, scale, digits in QUEUE_METRICS:
                values = [v * scale for v in _vector(frame, scenario, plan, metric)]
                cells.append(_interval(values, digits))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _reading(wait: PairedDifference, share: PairedDifference) -> str:
    if wait.verdict == "lower" and share.verdict != "higher":
        return "first is better"
    if share.verdict == "lower" and wait.verdict != "higher":
        return "first is better"
    if wait.verdict == "higher" or share.verdict == "higher":
        return "first is worse"
    return "no clear difference"


def staffing_table(frame: pd.DataFrame) -> list[str]:
    """Staffing and hand-over measures of each plan in the normal week."""
    columns = (
        ("planned_staff_hours", "Planned staff hours per week", 1.0, 0),
        ("rush_registers_staffed", "Registers staffed in rush", 1.0, 2),
        ("calm_registers_staffed", "Registers staffed outside rush", 1.0, 2),
        ("roster_shortfall_minutes", "Uncovered staff-minutes per week", 1.0, 0),
        ("break_gap_minutes", "Person-minutes below capacity from breaks", 1.0, 0),
        ("priority_true_gap", "True log-speed gap, registers 1 and 4 against others", 1.0, 3),
        ("handovers_per_shift", "Handovers per shift", 1.0, 2),
    )
    base = frame[frame["scenario"] == "normal_week"]
    header = ["Plan", *[label for _, label, _, _ in columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for plan in EVALUATED_PLANS:
        if plan not in set(base["plan"]):
            continue
        cells = [PLAN_LABELS[plan]]
        for metric, _, scale, digits in columns:
            values = [v * scale for v in _vector(base, "normal_week", plan, metric)]
            cells.append(_interval(values, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def fairness_table(frame: pd.DataFrame) -> list[str]:
    """Burden and hours of each plan in the normal week."""
    columns = (
        ("burden_spread", "Spread of burden per paid hour", 3),
        ("max_tier_gap", "Largest score-tier gap", 3),
        ("paid_hours", "Paid hours per week", 1),
    )
    base = frame[frame["scenario"] == "normal_week"]
    header = ["Plan", *[label for _, label, _ in columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for plan in EVALUATED_PLANS:
        if plan not in set(base["plan"]):
            continue
        cells = [PLAN_LABELS[plan]]
        for metric, _, digits in columns:
            cells.append(_interval(_vector(base, "normal_week", plan, metric), digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _tally_sentence(
    first: str, second: str, label: str, counts: Tally, name: str, total: int
) -> str:
    return (
        f"- {first} against {second} ({label}), {name}: lower in {counts.lower} of {total} "
        f"scenarios, no clear difference in {counts.tie}, higher in {counts.higher}."
    )


def reading_lines(frame: pd.DataFrame) -> list[str]:
    """One sentence for each pair of plans and metric, with the counts of verdicts."""
    lines: list[str] = []
    total = len(_scenarios(frame))
    for first, second, label in PAIRS:
        if first not in set(frame["plan"]) or second not in set(frame["plan"]):
            continue
        for metric, name, scale in VERDICT_METRICS:
            counts, _ = tally(frame, first, second, metric, scale)
            lines.append(_tally_sentence(first, second, label, counts, name, total))
    return lines


def placebo_statement(frame: pd.DataFrame) -> str:
    """What the comparison with the placebo shows about the scores."""
    if not {"C", "E", "B"} <= set(frame["plan"]):
        return ""
    total = len(_scenarios(frame))
    counts, _ = tally(frame, "C", "E", "share_over_target", 100.0)
    against_b, _ = tally(frame, "C", "B", "share_over_target", 100.0)
    placebo_b, _ = tally(frame, "E", "B", "share_over_target", 100.0)
    if counts.lower >= (total + 1) // 2 and counts.lower > counts.higher:
        verdict = (
            "The full planner does better than the placebo in most scenarios, so its benefit "
            "comes from the scores themselves and not from the shape of the plan around them."
        )
    elif counts.lower == 0 and counts.higher == 0:
        verdict = (
            "The full planner cannot be told apart from the placebo in any scenario. In this "
            "simulation the benefit of the scores is not separable from chance, and any "
            "improvement over the plan without scores must be credited to something else."
        )
    else:
        verdict = (
            "The full planner is not clearly better than the placebo across the scenarios, so the "
            "evidence that the scores themselves help is weak in this simulation."
        )
    return (
        f"Placebo test (share waiting over 3 minutes or leaving): C is lower than E in "
        f"{counts.lower} of {total} scenarios and higher in {counts.higher}; C is lower than B in "
        f"{against_b.lower} and higher in {against_b.higher}; E is lower than B in "
        f"{placebo_b.lower} and higher in {placebo_b.higher}. {verdict}"
    )


ABLATION_SETUP = (
    ", F the ablation (the roster of B with the Part 1 scores used only to choose who stands "
    "at which register, so it separates what the scores add in the posts from what they add in "
    "the roster)."
)
ABLATION_STEPS: tuple[tuple[str, str, str], ...] = (
    ("F", "B", "scores in the posts only against no scores"),
    ("C", "F", "scores in the roster as well against scores in the posts only"),
    ("C", "B", "full planner against no scores"),
)


def ablation_table(frame: pd.DataFrame) -> list[str]:
    """Where the benefit of the scores comes from, scenario by scenario."""
    header = ["Scenario"]
    for first, second, _ in ABLATION_STEPS:
        header.append(f"{first} minus {second}: mean wait (s)")
        header.append(f"{first} minus {second}: share (points)")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        cells = [SCENARIO_LABELS[scenario]]
        for first, second, _ in ABLATION_STEPS:
            wait = _difference(frame, scenario, first, second, "mean_wait_s", 1.0)
            share = _difference(frame, scenario, first, second, "share_over_target", 100.0)
            cells.append(f"{wait.mean:+.1f} [{wait.low:+.1f}, {wait.high:+.1f}]")
            cells.append(f"{share.mean:+.2f} [{share.low:+.2f}, {share.high:+.2f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def ablation_statement(frame: pd.DataFrame) -> str:
    """Plain statement of the ablation result."""
    if not {"B", "C", "F"} <= set(frame["plan"]):
        return ""
    total = len(_scenarios(frame))
    posts, _ = tally(frame, "F", "B", "mean_wait_s", 1.0)
    roster, _ = tally(frame, "C", "F", "mean_wait_s", 1.0)
    scenario = "normal_week"
    gain_posts = _mean(_vector(frame, scenario, "F", "mean_wait_s")) - _mean(
        _vector(frame, scenario, "B", "mean_wait_s")
    )
    gain_full = _mean(_vector(frame, scenario, "C", "mean_wait_s")) - _mean(
        _vector(frame, scenario, "B", "mean_wait_s")
    )
    if gain_full < 0.0:
        ratio = gain_posts / gain_full + 0.0
        captured = f"In the normal week F captures {ratio:.0%} of the mean-wait reduction of C."
    else:
        captured = "In the normal week the full planner shows no reduction in the mean wait."
    half = (total + 1) // 2
    if posts.lower >= half and roster.lower < half:
        reading = (
            "Most of the gain comes from choosing who stands at which register; using the scores "
            "in the roster as well does not add a gain that shows in most scenarios."
        )
    elif posts.lower >= half:
        reading = (
            "Both uses of the scores help: choosing who stands at which register and placing "
            "people in the roster each lower the mean wait in most scenarios."
        )
    elif roster.lower >= half:
        reading = (
            "The gain depends on the roster: choosing who stands at which register does not give "
            "a clear gain by itself in most scenarios."
        )
    else:
        reading = (
            "Neither step gives a clear gain in most scenarios on its own, so the evidence for "
            "either use of the scores is weak in this simulation."
        )
    return (
        f"Scores used only to choose who stands at which register (F) lower the mean wait "
        f"against plan B in {posts.lower} of {total} scenarios, with no clear difference in "
        f"{posts.tie} and higher in {posts.higher}. Adding the roster use of the scores (C "
        f"against F) lowers it further in {roster.lower}, with no clear difference in "
        f"{roster.tie} and higher in {roster.higher}. {captured} {reading} Plan F keeps the "
        f"roster of plan B, so it carries none of the effects of the scores on who works which "
        f"shift."
    )


def ablation_section(frame: pd.DataFrame) -> list[str]:
    """The ablation section of the report."""
    if not {"B", "C", "F"} <= set(frame["plan"]):
        return []
    return [
        "## Where the benefit of the scores comes from",
        "",
        "Plan F keeps the roster of plan B and uses the scores only in the posts, so the three "
        "comparisons split the effect of the full planner (C minus B) into the part from the "
        "posts (F minus B) and the part from the roster (C minus F). Below zero is better for "
        "the first plan.",
        "",
        *ablation_table(frame),
        "",
        ablation_statement(frame),
        "",
    ]


def sensitivity_variants(frame: pd.DataFrame) -> list[str]:
    """Layout variants present in the results."""
    layouts = sorted(
        {s for s in frame["scenario"] if str(s).startswith("layout_")},
        key=lambda s: float(str(s).split("_")[1]),
    )
    return layouts


def sensitivity_table(frame: pd.DataFrame, default_penalty: float) -> list[str]:
    """Results of the queue-layout sensitivity runs."""
    variants = sensitivity_variants(frame)
    if not variants:
        return []
    names = [("normal_week", f"{default_penalty:g} (the setting used above)")] + [
        (v, str(v).split("_")[1]) for v in variants
    ]
    header = [
        "Preference for the registers near the entrance",
        "Comparison",
        "Mean wait (s)",
        "Share over 3 min (points)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario, label in sorted(names, key=lambda item: float(item[1].split(" ")[0])):
        for first, second in (("C", "B"), ("E", "B"), ("D", "B")):
            wait = _difference(frame, scenario, first, second, "mean_wait_s", 1.0)
            share = _difference(frame, scenario, first, second, "share_over_target", 100.0)
            lines.append(
                f"| {label} | {first} minus {second} | "
                f"{wait.mean:+.1f} [{wait.low:+.1f}, {wait.high:+.1f}] | "
                f"{share.mean:+.2f} [{share.low:+.2f}, {share.high:+.2f}] |"
            )
    return lines


def stress_fairness_table(frame: pd.DataFrame, limit: float) -> list[str]:
    """Tier gaps in the stress scenarios, against the limit."""
    header = [
        "Scenario",
        "Plan",
        "Largest tier gap, mean over seeds",
        "Worst seed",
        f"Seeds above the limit of {limit:.0%}",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        for plan in ("A", "B", "C"):
            if plan not in set(frame["plan"]):
                continue
            values = _vector(frame, scenario, plan, "max_tier_gap")
            above = sum(1 for value in values if value > limit)
            lines.append(
                f"| {SCENARIO_LABELS[scenario]} | {PLAN_LABELS[plan]} | {_mean(values):.3f} | "
                f"{max(values):.3f} | {above} of {len(values)} |"
            )
    return lines


def live_table(frame: pd.DataFrame) -> list[str]:
    """Proposals and acceptance in live re-planning."""
    if "L" not in set(frame["plan"]):
        return []
    columns = (
        ("live_proposed", "Proposals per week"),
        ("live_accepted", "Accepted"),
        ("live_opened", "Registers opened"),
        ("live_closed", "Registers closed"),
        ("live_covered", "Absences covered"),
        ("live_flips", "Open then close or close then open"),
    )
    header = ["Scenario", *[label for _, label in columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in _scenarios(frame):
        cells = [SCENARIO_LABELS[scenario]]
        for metric, _ in columns:
            cells.append(f"{_mean(_vector(frame, scenario, 'L', metric)):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _named(differences: dict[str, PairedDifference], verdict: str) -> str:
    names = [
        SCENARIO_LABELS[name] for name, found in differences.items() if found.verdict == verdict
    ]
    if not names:
        return "no scenario"
    if len(names) == 1:
        return f"the {names[0]} scenario"
    return "the scenarios " + ", ".join(names)


def live_statement(frame: pd.DataFrame) -> str:
    """Plain statement of what live re-planning did."""
    if not {"L", "C"} <= set(frame["plan"]):
        return ""
    total = len(_scenarios(frame))
    wait, wait_by_scenario = tally(frame, "L", "C", "mean_wait_s", 1.0)
    share, _ = tally(frame, "L", "C", "share_over_target", 100.0)
    flips = _mean([v for s in _scenarios(frame) for v in _vector(frame, s, "L", "live_flips")])
    lower_in = _named(wait_by_scenario, "lower")
    higher_in = _named(wait_by_scenario, "higher")
    if wait.lower + wait.higher + share.lower + share.higher == 0:
        verdict = "Live re-planning cannot be told apart from the static plan in any scenario."
    elif wait.higher + share.higher == 0:
        verdict = (
            f"Live re-planning with an accepting manager never made waiting worse and reduced "
            f"the mean wait in {lower_in}."
        )
    elif wait.lower + share.lower == 0:
        verdict = (
            f"Live re-planning with an accepting manager did not reduce waiting in any "
            f"scenario and made the mean wait longer in {higher_in}."
        )
    else:
        verdict = (
            f"The result is mixed: the mean wait is lower in {lower_in} and higher in "
            f"{higher_in}. This simulation does not show that accepting every proposal beats "
            f"the static plan; the proposals are decision support for a manager who can decline "
            f"them."
        )
    return (
        f"Live re-planning (manager accepts every proposal): mean wait lower than the static "
        f"plan in {wait.lower} of {total} scenarios, no clear difference in {wait.tie}, higher "
        f"in {wait.higher}; share waiting over 3 minutes or leaving lower in {share.lower}, no "
        f"clear difference in {share.tie}, higher in {share.higher}. A register was opened and "
        f"closed again (or the reverse) {flips:.2f} times per week on average. {verdict}"
    )


def validation_table(validation: pd.DataFrame) -> list[str]:
    """Simulated waits next to Erlang C waits."""
    header = [
        "Open registers",
        "Load",
        "Queue",
        "Simulated wait (min)",
        "Erlang C wait (min)",
        "Gap",
        "Simulated share over 3 min",
        "Erlang C share over 3 min",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in validation.itertuples(index=False):
        queue = "shared" if row.discipline == "shared" else "one line per register"
        lines.append(
            f"| {row.servers} | {row.load:.2f} | {queue} | {row.simulated_wait_min:.3f} | "
            f"{row.erlang_wait_min:.3f} | {row.relative_gap:+.1%} | {row.simulated_tail:.3f} | "
            f"{row.erlang_tail:.3f} |"
        )
    return lines


LIMITS = (
    "Everything here comes from a simulated store: the arrivals follow the forecast model "
    "with negative binomial noise, the service times follow the structure of the Part 1 "
    "simulator, and the cashiers' hidden speeds are the Part 1 ground truth. A real store may "
    "behave differently and the first real test is a pilot with real tickets.",
    "The planning model assumes one shared queue (Erlang C). The simulated store has one "
    "line per register, where customers join the shortest line with a small preference for the "
    "registers nearest the entrance. Waits in that setting are longer than Erlang C predicts "
    "(see the validation table), so absolute waits are not comparable with the 3-minute target "
    "of the planning model, only the comparison between plans is.",
    "Plan A is a stylised stand-in for today's practice: the same number of open registers in "
    "every rush slot and in every calm slot, equal to the rounded weekly average of the "
    "demand-aware requirement. It is not calibrated on real data.",
    "Self-checkout customers are not simulated. Plans A to F are static; plan L adds the live "
    "controller with a manager who accepts every proposal, which is the most favourable "
    "assumption about how often advice is followed.",
    "Each seed is one simulated workforce with its own Part 1 scores; the intervals are 95 "
    "percent bootstrap intervals over a small number of seeds, so they are wide. A difference "
    "that is not clearly different from zero is reported as such and is not evidence of no "
    "effect.",
    "Contract hours are fixed for every plan. The plans decide where and when people work, "
    "not how much, so the comparison says nothing about cost.",
    "In these scenarios the roster that the scores produce is, apart from the tier guard "
    "acting in some seeds, the score-blind roster: swaps are limited to shifts of equal hours "
    "and equal burden, and the learning quota never binds. The measured benefit of the scores "
    "comes from the posts (plan F against plan B), not from the roster.",
)


def build_report(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    config: PlannerConfig,
    figure_paths: dict[str, Path],
    store: str,
    figure_prefix: str = "",
) -> str:
    """Build the whole evaluation report.

    The figures are linked as the prefix followed by the file name.
    """
    seeds = frame["seed"].nunique()
    customers = frame[(frame["scenario"] == "normal_week") & (frame["plan"] == "B")]["customers"]
    lines = [
        "# Planner evaluation",
        "",
        "Generated by `python -m ree.cli planner evaluate` (or `planner report` from the saved "
        "results). Every number below is computed from `results.csv` next to this file.",
        "",
        "## Set-up",
        "",
        f"- Store: `{store}`, {config.workforce.n_workers} cashiers with Part 1 scores, about "
        f"{customers.mean():.0f} customers in a week, one week of 7 days per run.",
        f"- {seeds} seeds. Each seed draws a Part 1 store (hidden speeds and the scores Part 1 "
        f"derives from simulated registers), a workforce with contracts and availability, and "
        f"{config.evaluation.replications} repetitions of the week's customers. Results are "
        f"averaged over the repetitions of a seed; intervals are 95 percent bootstrap intervals "
        f"over seeds ({BOOTSTRAP_REPLICATES} resamples).",
        "- Plans: A headcount only (the same registers open in every rush slot and in every calm "
        "slot), B demand-aware (forecast and Erlang C requirement, no scores), C the full planner "
        "(B with Part 1 scores in the posts and, in the roster, only through swaps of shifts with "
        "equal hours and equal burden, the tier guard and the learning quota), D the oracle (B "
        "with the hidden "
        "true speeds as scores), E the placebo (B with the Part 1 scores shuffled across "
        "workers)" + (ABLATION_SETUP if "F" in set(frame["plan"]) else "."),
        "- Coverage-first solving: for plans B to E the roster is first solved for coverage "
        "alone, with no scores, so all four have the same coverage; a second stage then keeps "
        "every worker's weekly paid minutes and the coverage and optimises burden, the tier "
        "guard and the learning quota, so no plan gives anyone different hours.",
        "- Scenarios: "
        + "; ".join(f"{SCENARIO_LABELS[name]} ({_describe(name)})" for name in _scenarios(frame))
        + ".",
        "",
        "## Waiting times (mean over seeds, 95 percent interval in brackets)",
        "",
        *results_table(frame),
        "",
        f"![Waits by plan]({figure_prefix}{figure_paths['waits'].name})",
        "",
        "## Queue length and idle capacity",
        "",
        "The manager tolerates no more than 5 customers waiting at every open register. The "
        "table gives the average number of customers waiting per open register, the share of "
        "the open day in which some open register had more than 5 waiting, and the share of "
        "open-register time spent idle.",
        "",
        *queue_table(frame),
        "",
        "## Paired differences",
        "",
        "Each row subtracts the second plan from the first within the same seeds. Below zero is "
        "better for the first plan.",
        "",
        *difference_table(frame),
        "",
        f"![Paired differences]({figure_prefix}{figure_paths['differences'].name})",
        "",
        "## Honest reading",
        "",
        *reading_lines(frame),
        "",
        placebo_statement(frame),
        "",
        *ablation_section(frame),
        "## Live re-planning",
        "",
        "The live controller watches the busyness of the open registers every 5 minutes, "
        "proposes opening the next register when it stays above 90 percent for 10 minutes and "
        "closing the last one when it stays below 55 percent for 20 minutes, and covers an "
        "absence from the floaters 10 minutes after the shift should have started. Nothing is "
        "applied without acceptance; in this table the simulated manager accepts every proposal "
        "that passes the plan checker.",
        "",
        *live_table(frame),
        "",
        live_statement(frame),
        "",
        "## Staffing and who stands on the busiest registers (normal week)",
        "",
        *staffing_table(frame),
        "",
        f"![Who holds the busiest registers]({figure_prefix}{figure_paths['alignment'].name})",
        "",
        "## How much the result depends on the layout of the store",
        "",
        "The simulated customers join the shortest line, with a preference for the registers "
        "nearest the entrance (registers 1 and 4 first). That preference is what makes it "
        "worthwhile to put the fastest cashiers there. The table repeats the normal week with "
        "no preference and with a stronger one; the rosters are the same, only the customers "
        "behave differently.",
        "",
        *sensitivity_table(frame, config.service.position_penalty),
        "",
        "## Fairness in every scenario",
        "",
        "The guard on the burden of each score tier is a priced term of the roster model, "
        "followed by a repair step that swaps shifts of equal paid minutes between tiers on the "
        "same day while it is exceeded; it is not a hard rule. Plans A and B do not use tiers, "
        "so their gap is chance. Plans D and E guard other tiers (true skills and shuffled "
        "scores), so their gap in Part 1 tiers is not shown. In the scenarios where there are "
        "too few people for the requirement the guard can be exceeded; the table counts how "
        "often.",
        "",
        *stress_fairness_table(frame, config.objective.tier_epsilon),
        "",
        "## Fairness and hours (normal week)",
        "",
        *fairness_table(frame),
        "",
        f"![Fairness]({figure_prefix}{figure_paths['fairness'].name})",
        "",
        "## Validation of the queue simulator",
        "",
        "A stationary queue with exponential service times and a shared line must reproduce the "
        "waiting time of Erlang C. The one-line-per-register discipline of the simulated store "
        "waits longer, which is the gap between the planning model and the store.",
        "",
        *validation_table(validation),
        "",
        f"![Erlang check]({figure_prefix}{figure_paths['validation'].name})",
        "",
        "## What this does not prove",
        "",
        *[f"- {item}" for item in LIMITS],
        "",
    ]
    return "\n".join(lines)


def _describe(name: str) -> str:
    return SCENARIOS[name].description
