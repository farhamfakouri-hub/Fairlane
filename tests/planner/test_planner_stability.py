from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ree.planner.cli import main
from ree.planner.eval.stability import (
    BLOCKED_WORKER_DAYS,
    FULL_DAY,
    block_worker_days,
    changed_assignments,
    keyed,
    stability_report,
    stability_seed,
    stability_statement,
    write_stability,
)
from ree.planner.roster.solver import plan_roster
from ree.planner.sim.roster_gate import scenario_inputs
from ree.planner.sim.scenarios import store_config


def fast_config(**extra: object):
    return store_config("tiny", **{"solver.backend": "heuristic", **extra})


@pytest.fixture(scope="module")
def solved():
    config, workers, requirement, scores = scenario_inputs(100, fast_config())
    return workers, plan_roster(workers, requirement, config, scores, weeks=1).plan


def test_changed_assignments_counts_added_removed_and_replaced_shifts() -> None:
    monday, tuesday = date(2026, 11, 2), date(2026, 11, 3)
    before = {("W001", monday): "a", ("W002", monday): "b", ("W003", tuesday): "c"}
    after = {("W001", monday): "a", ("W002", monday): "x", ("W004", tuesday): "d"}
    assert changed_assignments(before, after) == 3
    assert changed_assignments(before, before) == 0
    assert changed_assignments({}, after) == 3


def test_blocking_blocks_exactly_the_chosen_worker_days_and_nothing_else(solved) -> None:
    workers, plan = solved
    updated, chosen = block_worker_days(workers, plan, 100)
    assert len(chosen) == len(set(chosen)) == BLOCKED_WORKER_DAYS
    assert set(chosen) <= set(keyed(plan))
    for before, after in zip(workers, updated, strict=True):
        added = {
            day: windows
            for day, windows in after.blocked.items()
            if before.blocked.get(day) != windows
        }
        expected = {day for worker_id, day in chosen if worker_id == before.worker_id}
        assert set(added) == expected
        assert all(windows == (FULL_DAY,) for windows in added.values())
        assert after.contract_hours == before.contract_hours
    assert block_worker_days(workers, plan, 100)[1] == chosen


def test_a_solved_roster_does_not_use_a_blocked_day_again(solved) -> None:
    workers, plan = solved
    updated, chosen = block_worker_days(workers, plan, 100)
    config, _, requirement, scores = scenario_inputs(100, fast_config())
    again = plan_roster(updated, requirement, config, scores, weeks=1)
    assert not set(chosen) & set(keyed(again.plan))


def test_the_stability_probe_gives_a_complete_reproducible_row() -> None:
    first = stability_seed(fast_config(), 100)
    assert first == stability_seed(fast_config(), 100)
    assert first["seed"] == 100
    assert first["blocked_worker_days"] == BLOCKED_WORKER_DAYS
    for key in (
        "changed_with_reference",
        "changed_without_reference",
        "shortfall_with_reference",
        "shortfall_without_reference",
        "errors_with_reference",
        "errors_without_reference",
    ):
        assert first[key] >= 0.0
    assert first["changed_with_reference"] >= BLOCKED_WORKER_DAYS


def crafted(with_reference: float, without_reference: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [1, 2],
            "assignments": [50.0, 50.0],
            "blocked_worker_days": [2.0, 2.0],
            "changed_with_reference": [with_reference, with_reference],
            "changed_without_reference": [without_reference, without_reference],
            "shortfall_with_reference": [0.0, 30.0],
            "shortfall_without_reference": [10.0, 20.0],
            "errors_with_reference": [0.0, 0.0],
            "errors_without_reference": [0.0, 0.0],
        }
    )


def test_the_statement_admits_when_the_solver_does_not_stay_close() -> None:
    text = stability_statement(crafted(30.0, 40.0), fast_config())
    assert "of 50 assignments and solving again changes 30 assignments on average" in text
    assert "and 40 when it is not" in text
    assert "The reference removes 20 of 80 changes (25%)" in text
    assert "does not stay close to the previous roster" in text


def test_the_statement_says_so_when_the_reference_does_not_reduce_changes() -> None:
    text = stability_statement(crafted(40.0, 30.0), fast_config())
    assert "The reference does not reduce the number of changes." in text


def test_the_statement_credits_a_solver_that_stays_close() -> None:
    text = stability_statement(crafted(5.0, 30.0), fast_config())
    assert "The solver stays close to the previous roster." in text
    assert "does not stay close" not in text


def test_the_report_and_files_are_written(tmp_path) -> None:
    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    frame = crafted(10.0, 20.0)
    text = stability_report(frame, config)
    assert "# Roster stability after a change" in text
    assert "Changed, previous roster as reference" in text
    paths = write_stability(frame, config, reduced=False)
    assert all(path.exists() for path in paths)
    assert (tmp_path / "reports" / "stability.md").exists()


def test_the_stability_command_runs_and_writes_its_files(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ree.planner.cli as cli

    config = fast_config(**{"paths.outputs_dir": str(tmp_path)})
    monkeypatch.setattr(cli, "store_config", lambda store, **extra: config)
    code = main(["stability", "--reduced", "--store", "tiny", "--seeds", "1", "--jobs", "1"])
    assert code == 0
    assert (tmp_path / "reduced" / "stability.csv").exists()
    assert (tmp_path / "reduced" / "stability.md").exists()
    assert "assignments changed" in capsys.readouterr().out
