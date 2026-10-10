from __future__ import annotations

from pathlib import Path

import pytest

from ree.cli import main
from ree.config import Config, dump_config
from ree.experiments import run_all


def _config(directory: Path) -> Config:
    return Config().with_overrides({"paths.data_dir": str(directory)})


def _stub_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_all, "_smoke_experiment", lambda config, jobs: "stubbed experiment")


def test_smoke_runs_every_step_and_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_experiment(monkeypatch)
    assert run_all.run_smoke(_config(tmp_path)) == 0
    output = capsys.readouterr().out
    assert output.count("[PASS]") == 5
    assert "[FAIL]" not in output
    assert "byte-identical rerun" in output
    assert "ALL CHECKS PASSED" in output


def test_smoke_is_reachable_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_experiment(monkeypatch)
    path = tmp_path / "config.yaml"
    dump_config(_config(tmp_path), path)
    assert main(["smoke", "--config", str(path)]) == 0
    assert "ALL CHECKS PASSED" in capsys.readouterr().out


def test_smoke_exits_non_zero_and_names_the_failing_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_experiment(config: Config, jobs: int) -> str:
        raise AssertionError("recovery below the floor")

    monkeypatch.setattr(run_all, "_smoke_experiment", failing_experiment)
    assert run_all.run_smoke(_config(tmp_path)) == 1
    output = capsys.readouterr().out
    assert output.count("[FAIL]") == 1
    assert "[FAIL] reduced experiment" in output
    assert "recovery below the floor" in output
    assert "SOME CHECKS FAILED" in output
    assert output.count("[PASS]") == 4
