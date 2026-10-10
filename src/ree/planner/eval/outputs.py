"""Write the evaluation report, queue validation and figures."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ree.planner.config import PlannerConfig
from ree.planner.eval.figures import write_figures
from ree.planner.eval.report import build_report
from ree.planner.eval.validation import queue_validation

REPORT_FILE = "evaluation.md"
VALIDATION_FILE = "queue_validation.csv"


def output_directories(config: PlannerConfig, reduced: bool) -> tuple[Path, Path]:
    """Report and figure directories; a reduced run has its own."""
    root = Path(config.paths.outputs_dir)
    if reduced:
        return root / "reduced", root / "reduced" / "figures"
    return root / "reports", root / "figures"


def write_outputs(
    frame: pd.DataFrame, config: PlannerConfig, store: str, reduced: bool
) -> list[Path]:
    """Write the report, the queue validation table and the figures; return their paths."""
    reports, figures = output_directories(config, reduced)
    reports.mkdir(parents=True, exist_ok=True)
    validation = queue_validation(customers=20000, seeds=2) if reduced else queue_validation()
    validation.to_csv(reports / VALIDATION_FILE, index=False)
    paths = write_figures(frame, validation, figures)
    figure_prefix = Path(os.path.relpath(figures, reports)).as_posix() + "/"
    text = build_report(frame, validation, config, paths, store, figure_prefix)
    report_path = reports / REPORT_FILE
    report_path.write_text(text, encoding="utf-8")
    return [report_path, reports / VALIDATION_FILE, *paths.values()]
