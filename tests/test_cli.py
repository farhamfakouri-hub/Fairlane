from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import uvicorn

from ree.cli import main
from ree.config import DEFAULT_CONFIG_ENV, Config, dump_config


def test_ui_command_binds_to_localhost_and_forwards_the_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_call(command: list[str], env: dict[str, str] | None = None) -> int:
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    path = tmp_path / "config.yaml"
    dump_config(Config(), path)
    assert main(["ui", "--config", str(path), "--port", "8123"]) == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8123"
    assert command[command.index("--server.headless") + 1] == "true"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment[DEFAULT_CONFIG_ENV] == str(path.resolve())
    assert "http://127.0.0.1:8123" in capsys.readouterr().out


def test_ui_command_leaves_the_configuration_alone_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_call(command: list[str], env: dict[str, str] | None = None) -> int:
        captured["env"] = env
        return 0

    monkeypatch.delenv(DEFAULT_CONFIG_ENV, raising=False)
    monkeypatch.setattr(subprocess, "call", fake_call)
    assert main(["ui"]) == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert DEFAULT_CONFIG_ENV not in environment


def test_api_command_serves_on_localhost_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int) -> None:
        served.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    path = tmp_path / "config.yaml"
    dump_config(Config().with_overrides({"paths.data_dir": str(tmp_path)}), path)
    assert main(["api", "--config", str(path)]) == 0
    assert served["host"] == "127.0.0.1" and served["port"] == 8000
    assert served["app"] is not None
