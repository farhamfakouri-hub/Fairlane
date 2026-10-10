"""Smoke check of the Streamlit review screen with Streamlit's test runner."""

from __future__ import annotations

import os
from pathlib import Path

from streamlit.logger import set_log_level
from streamlit.testing.v1 import AppTest

from ree.config import DEFAULT_CONFIG_ENV, Config, dump_config

APP_PATH = Path(__file__).with_name("review_app.py")
DEFAULT_TIMEOUT_SECONDS = 240.0


def run_ui_smoke(
    config: Config, directory: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, list[str]]:
    """Open the screen and every view, and return the exceptions raised in each.

    An empty list for a view means that it loaded without errors.
    """
    set_log_level("ERROR")
    from ree.ui.review_app import VIEWS

    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "ui_config.yaml"
    dump_config(config, config_path)
    previous = os.environ.get(DEFAULT_CONFIG_ENV)
    os.environ[DEFAULT_CONFIG_ENV] = str(config_path)
    try:
        app = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
        app.run()
        problems = {"start": [str(item.value) for item in app.exception]}
        for view in VIEWS:
            app.sidebar.radio(key="view").set_value(view).run()
            problems[view] = [str(item.value) for item in app.exception]
    finally:
        if previous is None:
            os.environ.pop(DEFAULT_CONFIG_ENV, None)
        else:
            os.environ[DEFAULT_CONFIG_ENV] = previous
    return problems
