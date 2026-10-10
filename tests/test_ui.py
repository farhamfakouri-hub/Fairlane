from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ree.config import DEFAULT_CONFIG_ENV, Config, dump_config
from ree.storage import db
from ree.ui.review_app import VIEWS
from ree.ui.smoke import APP_PATH, run_ui_smoke

TIMEOUT = 240.0


def _small_config(directory: Path) -> Config:
    return Config().with_overrides(
        {
            "simulator.population.n_cashiers": 8,
            "simulator.n_weeks": 4,
            "engine.intervals.n_bootstrap": 40,
            "paths.data_dir": str(directory),
            "paths.store_name": "ui",
        }
    )


@pytest.fixture(scope="module")
def app(tmp_path_factory) -> AppTest:
    directory = tmp_path_factory.mktemp("ui")
    config = _small_config(directory)
    path = directory / "config.yaml"
    dump_config(config, path)
    os.environ[DEFAULT_CONFIG_ENV] = str(path)
    try:
        instance = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
        instance.run()
        yield instance
    finally:
        os.environ.pop(DEFAULT_CONFIG_ENV, None)


def _view(app: AppTest, name: str) -> AppTest:
    return app.sidebar.radio(key="view").set_value(name).run()


def test_every_view_loads_without_errors(tmp_path: Path) -> None:
    problems = run_ui_smoke(_small_config(tmp_path), tmp_path / "smoke")
    assert set(problems) == {"start", *VIEWS}
    assert problems == {name: [] for name in problems}


def test_manager_table_lists_every_cashier_with_scores_and_reasons(app: AppTest) -> None:
    _view(app, "Manager table")
    assert not app.exception
    frame = app.dataframe[0].value
    assert len(frame) == 8
    assert {"worker", "score_peak", "score_off", "tier_peak", "interval_peak", "drivers"} <= set(
        frame.columns
    )
    assert frame["drivers"].str.len().gt(0).all()
    assert len(app.metric) == 4


def test_manager_table_can_be_sorted_and_filtered(app: AppTest) -> None:
    _view(app, "Manager table")
    app.selectbox(key="sort_by").set_value("Worker").run()
    app.toggle(key="ascending").set_value(True).run()
    workers = app.dataframe[0].value["worker"].tolist()
    assert workers == sorted(workers)
    app.multiselect(key="tiers").set_value(["unrated"]).run()
    shown = app.dataframe[0].value
    assert ((shown["tier_peak"] == "unrated") | (shown["tier_off"] == "unrated")).all()
    app.multiselect(key="tiers").set_value(["A", "B", "C", "unrated"]).run()


def test_worker_view_explains_the_score_and_accepts_an_appeal(app: AppTest, tmp_path) -> None:
    _view(app, "Worker view")
    assert not app.exception
    worker = app.selectbox(key="worker").value
    text = " ".join(item.value for item in app.markdown)
    assert "How this score was computed" in " ".join(item.value for item in app.subheader)
    assert "Combined" in text and "Speed score" in text and "Accuracy score" in text
    app.text_area(key="appeal_reason").set_value("Please check my register assignments").run()
    submit = next(button for button in app.button if button.label == "Send review request")
    submit.click().run()
    assert not app.exception
    assert any("Review request" in item.value for item in app.success)
    config = Config.model_validate(
        __import__("ree.config", fromlist=["load_config"])
        .load_config(os.environ[DEFAULT_CONFIG_ENV])
        .model_dump()
    )
    connection = db.connect(db.store_directory(config))
    try:
        appeals = db.list_appeals(connection, worker)
    finally:
        connection.close()
    assert len(appeals) == 1 and appeals[0].status == "open"
    assert appeals[0].reason == "Please check my register assignments"


def test_empty_appeals_are_refused_politely(app: AppTest) -> None:
    _view(app, "Worker view")
    app.text_area(key="appeal_reason").set_value("  ").run()
    next(button for button in app.button if button.label == "Send review request").click().run()
    assert not app.exception
    assert any("few words" in item.value for item in app.warning)


def test_method_explainer_shows_the_formulas_and_a_live_example(app: AppTest) -> None:
    _view(app, "Method explainer")
    assert not app.exception
    assert len(app.latex) >= 6
    text = " ".join(item.value for item in app.markdown)
    assert "T_idle = 15 s" in text and "Expected time" in text
    first_tables = len(app.dataframe)
    assert first_tables >= 2
    next(button for button in app.button if button.key == "next_example").click().run()
    assert not app.exception


def test_the_simulator_control_regenerates_and_rescoring(app: AppTest) -> None:
    _view(app, "Manager table")
    before = app.dataframe[0].value["score_peak"].round(3).tolist()
    app.sidebar.number_input[0].set_value(777)
    submit = next(button for button in app.sidebar.button if "Regenerate" in button.label)
    submit.click().run()
    assert not app.exception
    after = app.dataframe[0].value["score_peak"].round(3).tolist()
    assert before != after
    assert len(after) == 8
