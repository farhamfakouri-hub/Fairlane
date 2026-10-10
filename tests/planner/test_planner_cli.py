from __future__ import annotations

import pytest

from ree.cli import main as part_one_main
from ree.planner.cli import main


def test_planner_smoke_runs_through_the_planner_entry_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke", "--quick"]) == 0
    output = capsys.readouterr().out
    assert "planner foundation: ok" in output
    assert "patterns for the solver" in output


def test_planner_smoke_runs_through_the_main_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert part_one_main(["planner", "smoke", "--quick"]) == 0
    assert "planner foundation: ok" in capsys.readouterr().out


def test_dump_config_writes_both_files(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "planner.yaml"
    patterns_path = tmp_path / "patterns.yaml"
    arguments = ["dump-config", "--out", str(config_path), "--patterns-out", str(patterns_path)]
    assert main(arguments) == 0
    assert config_path.exists() and patterns_path.exists()
    assert "wrote" in capsys.readouterr().out


def test_requirement_command_prints_one_line_per_day(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["requirement", "--weeks", "1"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 7
    assert lines[0].startswith("2026-11-01 Sun")
    assert "staff hours" in lines[-1]


def test_roster_command_solves_and_reports_a_small_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["roster", "--store", "tiny", "--seed", "3", "--backend", "heuristic"])
    output = capsys.readouterr().out
    assert "store tiny: 10 workers, 1 week(s)" in output
    assert "status HEURISTIC with heuristic" in output
    assert "checker:" in output and "fairness:" in output


def test_roster_and_daily_commands_accept_the_posts_only_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["roster", "--store", "tiny", "--seed", "3", "--backend", "heuristic", "--posts-only"])
    assert "store tiny: 10 workers" in capsys.readouterr().out
    code = main(
        ["daily", "--store", "tiny", "--seed", "3", "--backend", "heuristic", "--posts-only"]
    )
    assert code in (0, 1)
    assert "roster: status HEURISTIC" in capsys.readouterr().out


def test_roster_command_rejects_an_unknown_store(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["roster", "--store", "huge"])
    assert "invalid choice" in capsys.readouterr().err


def test_roster_gate_command_writes_a_report_and_returns_the_verdict(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "gate.json"
    code = main(
        [
            "roster-gate",
            "--scenarios",
            "2",
            "--first-seed",
            "100",
            "--processes",
            "1",
            "--out",
            str(out),
        ]
    )
    assert out.exists()
    assert "roster gate:" in capsys.readouterr().out
    assert code in (0, 1)


def test_daily_command_prints_the_board_and_the_relief_chain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["daily", "--store", "tiny", "--backend", "heuristic", "--day", "2026-11-03"])
    output = capsys.readouterr().out
    assert "board for 2026-11-03" in output
    assert "relief chain" in output
    assert "2026-11-01 Sun:" in output


def test_the_full_smoke_checks_the_whole_chain_and_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke"]) == 0
    output = capsys.readouterr().out
    assert "planner smoke: ok" in output
    for name in (
        "Erlang C worked example",
        "queue simulator",
        "roster",
        "daily plan",
        "planner API",
    ):
        assert name in output
