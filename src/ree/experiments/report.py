"""Statistics and Markdown helpers for the experiment report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ree.experiments.methods import METHOD_LABELS
from ree.experiments.scenarios import SCENARIO_BY_NAME, SCENARIOS

METRIC_LABELS: dict[str, str] = {
    "spearman": "Spearman correlation with true speed",
    "kendall": "Kendall tau with true speed",
    "top_k_precision": "top-5 precision",
    "mean_abs_rank_error": "mean absolute rank error (positions, lower is better)",
}
METRIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "spearman": (-1.0, 1.0),
    "kendall": (-1.0, 1.0),
    "top_k_precision": (0.0, 1.0),
    "mean_abs_rank_error": (0.0, None),
}
HEADLINE = "spearman"
LATENT = "latent"
REALIZED = "realized"
DESIGN_METHODS: tuple[str, ...] = ("v1_pause", "v2_ols_pause", "v2_huber_pause")
NAIVE = "naive_items_per_shift"


def confidence_interval(
    values: pd.Series, level: float, bounds: tuple[float | None, float | None] = (None, None)
) -> tuple[float, float, float]:
    """Mean and t interval of a sample, clipped to optional bounds."""
    clean = values.dropna().to_numpy(dtype=np.float64)
    if len(clean) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean, mean
    scale = clean.std(ddof=1) / np.sqrt(len(clean))
    margin = float(stats.t.ppf(0.5 + level / 2.0, len(clean) - 1) * scale)
    low, high = mean - margin, mean + margin
    if bounds[0] is not None:
        low = max(low, bounds[0])
    if bounds[1] is not None:
        high = min(high, bounds[1])
    return mean, low, high


def per_seed_table(results: pd.DataFrame, target: str = LATENT) -> pd.DataFrame:
    """Metrics averaged over the periods for each scenario, seed and method."""
    selected = results[results["target"] == target]
    metrics = list(METRIC_LABELS)
    return (
        selected.groupby(["scenario", "seed", "method"], sort=False)[metrics].mean().reset_index()
    )


def summarize(
    results: pd.DataFrame, metric: str, level: float, target: str = LATENT
) -> pd.DataFrame:
    """Mean, interval and number of seeds of one metric for each scenario and method."""
    table = per_seed_table(results, target)
    rows = []
    for (scenario, method), group in table.groupby(["scenario", "method"], sort=False):
        mean, low, high = confidence_interval(group[metric], level, METRIC_BOUNDS[metric])
        rows.append((scenario, method, mean, low, high, int(group[metric].notna().sum())))
    return pd.DataFrame(rows, columns=["scenario", "method", "mean", "low", "high", "n"])


def paired_difference(
    results: pd.DataFrame, scenario: str, metric: str, first: str, second: str, level: float
) -> tuple[float, float, float]:
    """Mean difference between two methods over the seeds, with its paired interval."""
    table = per_seed_table(results)
    table = table[table["scenario"] == scenario].pivot(
        index="seed", columns="method", values=metric
    )
    return confidence_interval(table[first] - table[second], level)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Render a data frame as a Markdown table with a fixed number of decimals."""

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "n/a" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(str(column) for column in frame.columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in frame.columns) + " |")
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    )
    return "\n".join(lines)


def scenario_matrix(summary: pd.DataFrame, methods: list[str], digits: int = 3) -> pd.DataFrame:
    """Table of results with a row for each method and a column for each scenario."""
    rows = []
    for method in methods:
        row: dict[str, object] = {"method": METHOD_LABELS[method]}
        for scenario in SCENARIOS:
            cell = summary[(summary["scenario"] == scenario.name) & (summary["method"] == method)]
            if cell.empty:
                row[scenario.name] = "n/a"
                continue
            mean, low, high = cell.iloc[0][["mean", "low", "high"]]
            row[scenario.name] = f"{mean:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"
        rows.append(row)
    return pd.DataFrame(rows)


Gap = tuple[str, float, float, float]
PRACTICAL_DIFFERENCE = 0.005


def _comparison(
    results: pd.DataFrame, summary: pd.DataFrame, scenario: str, level: float, methods: list[str]
) -> tuple[str, list[Gap], list[Gap], list[Gap]]:
    part = summary[summary["scenario"] == scenario].set_index("method")
    best = str(part["mean"].idxmax())
    tied: list[Gap] = []
    marginal: list[Gap] = []
    worse: list[Gap] = []
    for method in methods:
        if method == best:
            continue
        mean, low, high = paired_difference(results, scenario, HEADLINE, best, method, level)
        if np.isnan(mean):
            continue
        if low <= 0.0 <= high:
            tied.append((method, mean, low, high))
        elif abs(mean) < PRACTICAL_DIFFERENCE:
            marginal.append((method, mean, low, high))
        else:
            worse.append((method, mean, low, high))
    return best, tied, marginal, worse


def winner_sentences(
    results: pd.DataFrame, summary: pd.DataFrame, level: float, methods: list[str]
) -> list[str]:
    """One sentence for each scenario naming the best method and which others tie with it,
    trail it trivially or are clearly lower.
    """
    sentences = []
    for scenario in SCENARIOS:
        part = summary[summary["scenario"] == scenario.name].set_index("method")
        if part.empty:
            continue
        best, tied, marginal, worse = _comparison(results, summary, scenario.name, level, methods)
        text = (
            f"- **{scenario.title}**: highest mean Spearman is {METHOD_LABELS[best]} "
            f"({part.loc[best, 'mean']:.3f})."
        )
        if tied:
            names = ", ".join(METHOD_LABELS[name] for name, *_ in tied)
            text += f" Not distinguishable from it over the seeds: {names}."
        if marginal:
            gaps = ", ".join(f"{METHOD_LABELS[name]} (-{gap:.3f})" for name, gap, *_ in marginal)
            text += f" Detectably but trivially lower (under {PRACTICAL_DIFFERENCE}): {gaps}."
        if worse:
            gaps = ", ".join(f"{METHOD_LABELS[name]} (-{gap:.3f})" for name, gap, *_ in worse)
            text += f" Clearly lower: {gaps}."
        sentences.append(text)
    return sentences


def naive_profile(summary: pd.DataFrame) -> str:
    """Sentence with the mean Spearman correlation of the naive baseline in each scenario."""
    part = summary[summary["method"] == NAIVE].set_index("scenario")["mean"]
    values = ", ".join(f"{name} {part[name]:.2f}" for name in part.index)
    return f"Naive items-per-shift baseline, mean Spearman by scenario: {values}."


def design_method_outcomes(
    results: pd.DataFrame, summary: pd.DataFrame, level: float, methods: list[str]
) -> list[str]:
    """List where the designed methods lose to another one and where the naive baseline
    wins or ties.
    """
    losses, trivial, naive_good = [], [], []
    for scenario in SCENARIOS:
        part = summary[summary["scenario"] == scenario.name].set_index("method")
        if part.empty:
            continue
        best, tied, marginal, worse = _comparison(results, summary, scenario.name, level, methods)
        for name, gap, low, high in worse:
            if name in DESIGN_METHODS:
                losses.append(
                    f"- {scenario.title}: {METHOD_LABELS[name]} is clearly beaten by "
                    f"{METHOD_LABELS[best]} (difference {gap:.3f}, paired {level:.0%} interval "
                    f"{low:.3f} to {high:.3f})."
                )
        for name, gap, *_ in marginal:
            if name in DESIGN_METHODS:
                trivial.append(
                    f"{scenario.title}: {METHOD_LABELS[name]} trails {METHOD_LABELS[best]} "
                    f"by {gap:.3f}"
                )
        if best == NAIVE:
            naive_good.append(f"- {scenario.title}: the naive baseline has the highest mean.")
        elif any(name == NAIVE for name, *_ in tied):
            naive_good.append(
                f"- {scenario.title}: the naive baseline is not distinguishable from the best."
            )
    first = ["**Where the design's own methods (v1 or v2 with the pause rule) lose**", ""]
    first += losses or ["- Nowhere by a practically relevant margin (0.005 Spearman or more)."]
    if trivial:
        first += ["", "Detectable but trivial gaps: " + "; ".join(trivial) + "."]
    second = ["", "**Where the naive baseline wins or ties**", ""]
    second += naive_good or [
        "- Nowhere: it is clearly worse than the best method in every scenario."
    ]
    second += ["", naive_profile(summary)]
    return [*first, *second]


def default_method_section(summary: pd.DataFrame) -> list[str]:
    """Choose the default method: the pause-rule version with the highest mean Spearman."""
    candidates = {
        name: float(summary[summary["method"] == name]["mean"].mean()) for name in DESIGN_METHODS
    }
    chosen = max(candidates, key=lambda name: candidates[name])
    rows = pd.DataFrame(
        {
            "method": [METHOD_LABELS[name] for name in candidates],
            "mean Spearman over the six scenarios": list(candidates.values()),
        }
    )
    return [
        "Rule from the method document: the version that recovers the planted order better "
        "becomes the default.",
        "",
        markdown_table(rows, 4),
        "",
        f"Highest: **{METHOD_LABELS[chosen]}**.",
    ]


def v1_versus_v2(results: pd.DataFrame, summary: pd.DataFrame, level: float) -> pd.DataFrame:
    """Paired comparison of v2 (least squares) with v1 in each scenario, with a verdict."""
    rows = []
    for scenario in SCENARIOS:
        part = summary[summary["scenario"] == scenario.name].set_index("method")
        if part.empty:
            continue
        mean, low, high = paired_difference(
            results, scenario.name, HEADLINE, "v2_ols_pause", "v1_pause", level
        )
        verdict = "tie" if low <= 0.0 <= high else ("v2 better" if mean > 0 else "v1 better")
        rows.append(
            {
                "scenario": scenario.name,
                "v1 + pause": part.loc["v1_pause", "mean"],
                "v2 OLS + pause": part.loc["v2_ols_pause", "mean"],
                "v2 minus v1": f"{mean:+.4f} ({low:+.4f} to {high:+.4f})",
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def scenario_descriptions() -> str:
    """Bullet list of the scenarios with their descriptions."""
    return "\n".join(
        f"- **{item.title}** (`{item.name}`): {item.description}" for item in SCENARIOS
    )


__all__ = [
    "DESIGN_METHODS",
    "HEADLINE",
    "LATENT",
    "METRIC_BOUNDS",
    "METRIC_LABELS",
    "NAIVE",
    "REALIZED",
    "SCENARIO_BY_NAME",
    "confidence_interval",
    "default_method_section",
    "design_method_outcomes",
    "markdown_table",
    "naive_profile",
    "paired_difference",
    "per_seed_table",
    "scenario_descriptions",
    "scenario_matrix",
    "summarize",
    "v1_versus_v2",
    "winner_sentences",
]
