"""Figures of the planner evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ree.experiments.figures import (
    AQUA,
    BLUE,
    INK_MUTED,
    INK_SECONDARY,
    ORANGE,
    apply_style,
    save_figure,
    style_axes,
)
from ree.planner.eval.plans import ALL_PLANS, PLAN_LABELS, PLAN_NAMES
from ree.planner.eval.stats import bootstrap_mean, paired_bootstrap

PURPLE = "#7c5cbf"
MAGENTA = "#b0307a"
OLIVE = "#8a7a1c"
PLAN_COLORS = {
    "A": INK_MUTED,
    "B": BLUE,
    "C": ORANGE,
    "D": AQUA,
    "E": PURPLE,
    "F": OLIVE,
    "L": MAGENTA,
}
PLAN_MARKERS = {"A": "s", "B": "o", "C": "D", "D": "^", "E": "v", "F": "X", "L": "P"}
ALIGNMENT_ORDER = ("A", "B", "F", "C", "D", "E")
SCENARIO_LABELS = {
    "normal_week": "normal week",
    "saturday_surge": "Saturday surge",
    "absences": "absences",
    "season_peak": "season peak",
    "forecast_error": "forecast error",
    "small_team": "small team",
}
MARKER_SIZE = 6.5
WHISKER_WIDTH = 1.4
COMPARISONS = (("C", "B"), ("E", "B"), ("D", "B"), ("L", "C"))
COMPARISON_COLORS = {
    ("C", "B"): ORANGE,
    ("E", "B"): PURPLE,
    ("D", "B"): AQUA,
    ("L", "C"): MAGENTA,
}


def _scenarios(frame: pd.DataFrame) -> list[str]:
    present = set(frame["scenario"])
    return [name for name in SCENARIO_LABELS if name in present]


def _values(frame: pd.DataFrame, scenario: str, plan: str, metric: str) -> np.ndarray:
    part = frame[(frame["scenario"] == scenario) & (frame["plan"] == plan)].sort_values("seed")
    return part[metric].to_numpy(dtype=float)


def _plan_legend(figure: Figure, axes: Axes, present: set[str]) -> None:
    handles = [
        axes.plot([], [], PLAN_MARKERS[p], color=PLAN_COLORS[p], label=PLAN_LABELS[p], ms=7)[0]
        for p in ALL_PLANS
        if p in present
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)


def _plan_panel(
    axes: Axes, frame: pd.DataFrame, metric: str, scale: float, title: str, unit: str
) -> None:
    scenarios = _scenarios(frame)
    plans = [p for p in ALL_PLANS if p in set(frame["plan"])]
    offsets = np.linspace(-0.32, 0.32, len(plans))
    for position, plan in enumerate(plans):
        for index, scenario in enumerate(scenarios):
            values = _values(frame, scenario, plan, metric) * scale
            mean, low, high = bootstrap_mean(values)
            x = index + offsets[position]
            axes.plot([x, x], [low, high], color=PLAN_COLORS[plan], lw=WHISKER_WIDTH)
            axes.plot(
                x,
                mean,
                PLAN_MARKERS[plan],
                color=PLAN_COLORS[plan],
                ms=MARKER_SIZE,
                mec="white",
                mew=0.8,
            )
    axes.set_xticks(range(len(scenarios)))
    axes.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    axes.set_ylabel(unit)
    axes.set_title(title)
    style_axes(axes, "y")


def make_wait_figure(frame: pd.DataFrame, path: Path) -> Path:
    """Waiting time of each plan in each scenario."""
    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    _plan_panel(left, frame, "mean_wait_s", 1.0, "Mean wait", "seconds")
    _plan_panel(
        right,
        frame,
        "share_over_target",
        100.0,
        "Waited over 3 minutes or left",
        "percent of customers",
    )
    _plan_legend(figure, left, set(frame["plan"]))
    return save_figure(figure, path)


def _forest_panel(
    axes: Axes, frame: pd.DataFrame, metric: str, scale: float, title: str, unit: str
) -> None:
    scenarios = _scenarios(frame)
    present = set(frame["plan"])
    pairs = [pair for pair in COMPARISONS if pair[0] in present and pair[1] in present]
    offsets = np.linspace(-0.27, 0.27, len(pairs))
    for position, (first, second) in enumerate(pairs):
        for index, scenario in enumerate(scenarios):
            difference = paired_bootstrap(
                _values(frame, scenario, first, metric) * scale,
                _values(frame, scenario, second, metric) * scale,
            )
            y = index + offsets[position]
            color = COMPARISON_COLORS[(first, second)]
            axes.plot([difference.low, difference.high], [y, y], color=color, lw=WHISKER_WIDTH)
            axes.plot(difference.mean, y, "o", color=color, ms=MARKER_SIZE, mec="white", mew=0.8)
    axes.axvline(0.0, color=INK_SECONDARY, lw=1.0)
    axes.set_yticks(range(len(scenarios)))
    axes.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    axes.invert_yaxis()
    axes.set_xlabel(unit)
    axes.set_title(title)
    style_axes(axes, "x")


def make_difference_figure(frame: pd.DataFrame, path: Path) -> Path:
    """Paired differences between plans, with intervals."""
    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    _forest_panel(
        left, frame, "mean_wait_s", 1.0, "Change in mean wait (below zero is better)", "seconds"
    )
    _forest_panel(
        right,
        frame,
        "share_over_target",
        100.0,
        "Change in share over 3 minutes (below zero is better)",
        "percentage points",
    )
    handles = [
        left.plot(
            [], [], "o", color=COMPARISON_COLORS[pair], label=f"{pair[0]} minus {pair[1]}", ms=7
        )[0]
        for pair in COMPARISONS
        if pair[0] in set(frame["plan"])
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    return save_figure(figure, path)


def make_alignment_figure(frame: pd.DataFrame, path: Path) -> Path:
    """Who holds the busiest registers in the rush, by plan."""
    apply_style()
    figure, axes = plt.subplots(1, 1, figsize=(6.4, 4.0), constrained_layout=True)
    base = frame[frame["scenario"] == "normal_week"]
    plans = [p for p in ALIGNMENT_ORDER if p in set(base["plan"])]
    for index, plan in enumerate(plans):
        values = _values(base, "normal_week", plan, "priority_true_gap")
        mean, low, high = bootstrap_mean(values)
        axes.plot([index, index], [low, high], color=PLAN_COLORS[plan], lw=WHISKER_WIDTH)
        axes.plot(
            index,
            mean,
            PLAN_MARKERS[plan],
            color=PLAN_COLORS[plan],
            ms=MARKER_SIZE + 1,
            mec="white",
        )
    axes.axhline(0.0, color=INK_SECONDARY, lw=1.0)
    axes.set_xticks(range(len(plans)))
    axes.set_xticklabels([PLAN_LABELS[p] for p in plans], rotation=20, ha="right")
    axes.set_ylabel("true log-speed, registers 1 and 4 minus others")
    axes.set_title("Who holds the busiest registers in the rush")
    style_axes(axes, "y")
    return save_figure(figure, path)


def make_validation_figure(validation: pd.DataFrame, path: Path) -> Path:
    """Simulated waits against Erlang C."""
    apply_style()
    figure, axes = plt.subplots(1, 1, figsize=(6.4, 4.0), constrained_layout=True)
    shared = validation[validation["discipline"] == "shared"]
    lines = validation[validation["discipline"] == "shortest_line"]
    axes.plot(
        shared["servers"],
        shared["erlang_wait_min"],
        "-",
        color=INK_SECONDARY,
        lw=2.0,
        label="Erlang C (shared queue)",
    )
    axes.plot(
        shared["servers"],
        shared["simulated_wait_min"],
        "o",
        color=BLUE,
        ms=MARKER_SIZE + 1,
        mec="white",
        label="simulated, shared queue",
    )
    axes.plot(
        lines["servers"],
        lines["simulated_wait_min"],
        "D",
        color=ORANGE,
        ms=MARKER_SIZE,
        mec="white",
        label="simulated, one line per register",
    )
    axes.set_xlabel("open registers (load 75 percent)")
    axes.set_ylabel("mean wait (minutes)")
    axes.set_title("The simulator reproduces Erlang C")
    axes.legend(loc="upper right")
    style_axes(axes, "y")
    return save_figure(figure, path)


def make_fairness_figure(frame: pd.DataFrame, path: Path) -> Path:
    """Burden spread and tier gap of each plan."""
    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    base = frame[frame["scenario"] == "normal_week"]
    plans = [p for p in PLAN_NAMES if p in set(base["plan"])]
    for axes, metric, title in (
        (left, "burden_spread", "Spread of burden per paid hour"),
        (right, "max_tier_gap", "Largest score-tier gap in burden"),
    ):
        for index, plan in enumerate(plans):
            values = _values(base, "normal_week", plan, metric)
            mean, low, high = bootstrap_mean(values)
            axes.plot([index, index], [low, high], color=PLAN_COLORS[plan], lw=WHISKER_WIDTH)
            axes.plot(
                index,
                mean,
                PLAN_MARKERS[plan],
                color=PLAN_COLORS[plan],
                ms=MARKER_SIZE + 1,
                mec="white",
            )
        axes.set_xticks(range(len(plans)))
        axes.set_xticklabels([PLAN_LABELS[p] for p in plans], rotation=20, ha="right")
        axes.set_title(title)
        style_axes(axes, "y")
    return save_figure(figure, path)


def write_figures(
    frame: pd.DataFrame, validation: pd.DataFrame, directory: Path
) -> dict[str, Path]:
    """Draw every figure and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "waits": make_wait_figure(frame, directory / "planner_waits.png"),
        "differences": make_difference_figure(frame, directory / "planner_differences.png"),
        "alignment": make_alignment_figure(frame, directory / "planner_alignment.png"),
        "fairness": make_fairness_figure(frame, directory / "planner_fairness.png"),
        "validation": make_validation_figure(validation, directory / "planner_erlang_check.png"),
    }
