"""Figures of the experiment report, drawn with matplotlib in one common style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ree.config import Config
from ree.domain.models import SimulatedStore
from ree.engine.pause_rule import extract_gaps, thresholds_from_config, tune_thresholds
from ree.experiments.methods import SPEED_METHODS
from ree.experiments.scenarios import SCENARIOS
from ree.simulator.run import simulate_store

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

FIGURE_DPI = 200
LINE_WIDTH = 2.0
MARKER_SIZE = 7.0
RING_WIDTH = 1.4


def apply_style() -> None:
    """Set the matplotlib style shared by all the figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": FIGURE_DPI,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def style_axes(axes: Axes, grid_axis: str = "y") -> None:
    """Give one axes the common look: light grid, no tick marks, soft axis lines."""
    axes.grid(False)
    axes.grid(True, axis=grid_axis)
    axes.tick_params(length=0)
    axes.spines["left"].set_color(BASELINE)
    axes.spines["bottom"].set_color(BASELINE)


def figures_directory(config: Config) -> Path:
    """Directory of the figures, created when missing."""
    path = Path(config.paths.outputs_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(figure: Figure, path: Path) -> Path:
    """Save a figure to a file and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    return path


def make_gap_distribution_figure(
    config: Config,
    simulated: SimulatedStore | None = None,
    output_path: Path | None = None,
) -> Path:
    """Histogram of the gaps between register events, showing the long tail that the pause
    rule targets.
    """
    apply_style()
    store = simulated if simulated is not None else simulate_store(config)
    table = extract_gaps(store.data.events)
    gaps = table.gap_seconds
    configured = thresholds_from_config(config.engine.pause)
    tuned = tune_thresholds(table, config.engine.pause)
    hard_stop = config.engine.pause.hard_stop_seconds

    shares = {
        "hard": float((gaps > hard_stop).mean()),
        "idle": float((gaps > configured.t_idle_seconds).mean()),
        "stop": float((gaps > configured.t_stop_seconds).mean()),
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.22})

    bins = np.arange(0.0, 40.5, 0.5)
    counts, edges = np.histogram(gaps, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    left.bar(centers, counts, width=0.42, color=BLUE, linewidth=0)
    left.set_yscale("log")
    left.set_xlim(0, 40)
    left.set_ylim(top=counts.max() * 3.5)
    left.set_xlabel("gap between consecutive events (seconds)")
    left.set_ylabel("number of gaps (log scale)")
    left.set_title("Most gaps are ordinary work; a thin tail is not")
    style_axes(left)

    marks = (
        (hard_stop, f"{hard_stop:.0f} s hard stop"),
        (configured.t_idle_seconds, f"T_idle {configured.t_idle_seconds:.0f} s"),
    )
    top = counts.max() * 3.0
    for position, label in marks:
        left.axvline(position, color=INK_SECONDARY, linewidth=1.2, ymax=0.93)
        left.text(position + 0.4, top, label, color=INK, fontsize=9, va="top", ha="left")
    left.text(
        0.98,
        0.88,
        f"{shares['hard']:.0%} of gaps are longer than {hard_stop:.0f} s:\n"
        "tag removal, loyalty card and closing\nare normal work, so a hard stop\ndeletes real time",
        transform=left.transAxes,
        ha="right",
        va="top",
        color=INK_SECONDARY,
        fontsize=9,
    )

    ordered = np.sort(gaps)
    probe = np.unique(
        np.concatenate([np.linspace(0.5, 40.0, 400), np.geomspace(40.0, ordered[-1], 200)])
    )
    survival = 1.0 - np.searchsorted(ordered, probe, side="right") / len(ordered)
    keep = survival > 0
    right.plot(probe[keep], survival[keep], color=BLUE, linewidth=LINE_WIDTH)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(1.0, 300.0)
    right.set_ylim(1e-5, 1.6)
    right.set_xlabel("gap length x (seconds, log scale)")
    right.set_ylabel("share of gaps longer than x (log scale)")
    right.set_title("The long tail the pause rule targets")
    style_axes(right, grid_axis="both")
    for position, label in (
        (configured.t_idle_seconds, f"T_idle {configured.t_idle_seconds:.0f} s"),
        (configured.t_stop_seconds, f"T_stop {configured.t_stop_seconds:.0f} s"),
    ):
        right.axvline(position, color=INK_SECONDARY, linewidth=1.2)
        right.text(
            position * 1.06,
            0.6,
            f"{label}\n{(gaps > position).mean():.2%} of gaps",
            color=INK,
            fontsize=9,
            va="top",
            ha="left",
        )
    right.text(
        0.03,
        0.40,
        "tuned from this store's\n"
        f"own gaps: T_idle {tuned.t_idle_seconds:.1f} s,\n"
        f"T_stop {tuned.t_stop_seconds:.0f} s",
        transform=right.transAxes,
        ha="left",
        va="top",
        color=INK_SECONDARY,
        fontsize=9,
    )

    target = output_path or figures_directory(config) / "gap_distribution.png"
    return save_figure(figure, target)


METHOD_STYLE: dict[str, dict[str, object]] = {
    "naive_items_per_shift": {"color": INK_MUTED, "marker": "s", "filled": True},
    "hard_5s_stop": {"color": INK_MUTED, "marker": "^", "filled": True},
    "v1_no_pause": {"color": BLUE, "marker": "o", "filled": False},
    "v1_pause": {"color": BLUE, "marker": "o", "filled": True},
    "v2_ols_pause": {"color": ORANGE, "marker": "o", "filled": True},
    "v2_huber_pause": {"color": ORANGE, "marker": "o", "filled": False},
}
SHORT_LABELS: dict[str, str] = {
    "naive_items_per_shift": "naive items per shift",
    "hard_5s_stop": "hard 5 s stop",
    "v1_no_pause": "v1, no pause rule",
    "v1_pause": "v1 + pause rule",
    "v2_ols_pause": "v2 OLS + pause rule",
    "v2_huber_pause": "v2 Huber + pause rule",
}


def _marker_kwargs(method: str, size: float = MARKER_SIZE) -> dict[str, object]:
    style = METHOD_STYLE[method]
    color = str(style["color"])
    filled = bool(style["filled"])
    return {
        "marker": style["marker"],
        "markersize": size,
        "markerfacecolor": color if filled else SURFACE,
        "markeredgecolor": color,
        "markeredgewidth": RING_WIDTH if filled else 1.8,
        "color": color,
    }


def _method_legend(figure: Figure, columns: int = 3) -> None:
    handles = [
        plt.Line2D([], [], linestyle="none", label=SHORT_LABELS[name], **_marker_kwargs(name))
        for name in SPEED_METHODS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=columns,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )


def make_scatter_figure(scatter: pd.DataFrame, path: Path) -> Path:
    """Estimated against true speed for each method."""
    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharex=True, sharey=True)
    off = scatter[scatter["period"] == "off"]
    for axis, method in zip(axes.ravel(), SPEED_METHODS, strict=True):
        part = off[off["method"] == method].dropna(subset=["estimate"])
        truth = (part["truth"] - part["truth"].mean()) / part["truth"].std(ddof=0)
        estimate = (part["estimate"] - part["estimate"].mean()) / part["estimate"].std(ddof=0)
        axis.plot([-3, 3], [-3, 3], color=BASELINE, linewidth=1.2, zorder=1)
        kwargs = _marker_kwargs(method, MARKER_SIZE)
        axis.plot(truth, estimate, linestyle="none", **kwargs)
        rho = float(pd.Series(truth).corr(pd.Series(estimate), method="spearman"))
        axis.set_title(SHORT_LABELS[method], fontsize=10.5)
        axis.text(
            0.04,
            0.95,
            f"Spearman {rho:.2f}",
            transform=axis.transAxes,
            va="top",
            color=INK,
            fontsize=10,
        )
        axis.set_xlim(-3, 3)
        axis.set_ylim(-3, 3)
        style_axes(axis, grid_axis="both")
    for axis in axes[1]:
        axis.set_xlabel("true speed (standardised)")
    for axis in axes[:, 0]:
        axis.set_ylabel("estimated speed (standardised)")
    figure.suptitle(
        "Estimated against true skill, off-peak, one balanced store (the line is perfect recovery)",
        x=0.08,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    return save_figure(figure, path)


def make_method_comparison_figure(summary: pd.DataFrame, level: float, path: Path) -> Path:
    """Spearman correlation of every method in every scenario, with intervals."""
    apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 7.4), sharex=True, sharey=True)
    order = list(SPEED_METHODS)
    positions = {name: len(order) - 1 - index for index, name in enumerate(order)}
    for axis, scenario in zip(axes.ravel(), SCENARIOS, strict=True):
        part = summary[summary["scenario"] == scenario.name].set_index("method")
        best = part["mean"].idxmax()
        for method in order:
            row = part.loc[method]
            y = positions[method]
            axis.plot(
                [row["low"], row["high"]],
                [y, y],
                color=_marker_kwargs(method)["color"],
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
                zorder=2,
            )
            axis.plot([row["mean"]], [y], linestyle="none", zorder=3, **_marker_kwargs(method))
        axis.text(
            part.loc[best, "high"] + 0.02,
            positions[best],
            f"{part.loc[best, 'mean']:.2f}",
            va="center",
            ha="left",
            color=INK,
            fontsize=9,
        )
        axis.set_title(scenario.title, fontsize=10.5)
        axis.set_xlim(-0.25, 1.12)
        axis.set_ylim(-0.7, len(order) - 0.3)
        axis.set_yticks([positions[name] for name in order])
        axis.set_yticklabels([SHORT_LABELS[name] for name in order], fontsize=8.5)
        style_axes(axis, grid_axis="x")
    for axis in axes[1]:
        axis.set_xlabel("Spearman correlation with true speed")
    figure.suptitle(
        f"Rank recovery by method and scenario (mean over seeds, {level:.0%} interval)",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(figure, path)


def make_learning_curve_figure(curve: pd.DataFrame, path: Path) -> Path:
    """How each method improves with more weeks of data."""
    apply_style()
    figure, (full, zoom) = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"wspace": 0.18})
    means = curve.groupby(["method", "weeks"])["spearman"].mean().reset_index()
    for method in SPEED_METHODS:
        part = means[means["method"] == method].sort_values("weeks")
        kwargs = _marker_kwargs(method, 6.0)
        full.plot(part["weeks"], part["spearman"], linewidth=LINE_WIDTH, **kwargs)
        if method not in ("naive_items_per_shift", "hard_5s_stop"):
            zoom.plot(part["weeks"], part["spearman"], linewidth=LINE_WIDTH, **kwargs)
    for axis in (full, zoom):
        axis.set_xlabel("weeks of register data")
        axis.set_xticks(sorted(means["weeks"].unique()))
        style_axes(axis, grid_axis="y")
    full.set_ylim(0.0, 1.02)
    full.set_ylabel("Spearman correlation with true speed")
    full.set_title("All six methods")
    zoom.set_ylim(0.8, 1.005)
    zoom.set_title("Zoom on the four timing-based methods")
    _method_legend(figure)
    figure.subplots_adjust(bottom=0.26)
    return save_figure(figure, path)


def make_calibration_figure(table: pd.DataFrame, path: Path) -> Path:
    """Ratio of the real spread of each score to its bootstrap error."""
    apply_style()
    labels = {
        "speed_off": "speed score, off-peak",
        "speed_peak": "speed score, peak",
        "accuracy": "accuracy score",
        "combined_off": "combined score, off-peak",
        "combined_peak": "combined score, peak",
    }
    ordered = [name for name in labels if name in set(table["component"])]
    rows = table.set_index("component").loc[ordered]
    figure, axis = plt.subplots(figsize=(8.2, 3.9))
    y = np.arange(len(ordered))[::-1]
    axis.barh(y, rows["pooled_ratio"], height=0.52, color=BLUE, linewidth=0, zorder=2)
    axis.axvline(1.0, color=INK_SECONDARY, linewidth=1.4, zorder=3)
    axis.text(
        1.0,
        len(ordered) - 0.35,
        "calibrated = 1.0",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
    )
    for position, value in zip(y, rows["pooled_ratio"], strict=True):
        axis.text(value + 0.03, position, f"{value:.2f}", va="center", fontsize=9.5, color=INK)
    axis.set_yticks(y)
    axis.set_yticklabels([labels[name] for name in ordered])
    axis.set_xlim(0, max(1.9, float(rows["pooled_ratio"].max()) + 0.3))
    axis.set_xlabel(
        "replicate-to-replicate SD divided by bootstrap SE (above 1 means intervals too narrow)"
    )
    axis.set_title("Are the intervals wide enough?")
    style_axes(axis, grid_axis="x")
    return save_figure(figure, path)


def write_experiment_figures(
    config: Config,
    directory: Path,
    scatter: pd.DataFrame | None,
    summary: pd.DataFrame,
    curve: pd.DataFrame,
    calibration: pd.DataFrame,
    level: float,
) -> list[str]:
    """Draw every figure of the report and return the file names."""
    names = []
    if scatter is not None:
        names.append(make_scatter_figure(scatter, directory / "scatter_true_vs_estimated.png").name)
    names.append(
        make_method_comparison_figure(summary, level, directory / "method_comparison.png").name
    )
    names.append(make_learning_curve_figure(curve, directory / "learning_curve.png").name)
    names.append(
        make_gap_distribution_figure(config, None, directory / "gap_distribution.png").name
    )
    names.append(make_calibration_figure(calibration, directory / "interval_calibration.png").name)
    return names
