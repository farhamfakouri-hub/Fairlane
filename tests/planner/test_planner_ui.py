from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ree.planner.ui.manager_app import VIEWS

UI = Path(__file__).resolve().parents[2] / "src" / "ree" / "planner" / "ui"
MANAGER = UI / "manager_app.py"
WORKER = UI / "worker_app.py"
TIMEOUT = 240.0


@pytest.fixture(scope="module")
def manager() -> AppTest:
    app = AppTest.from_file(str(MANAGER), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception
    return app


def open_view(app: AppTest, name: str) -> AppTest:
    return app.sidebar.radio(key="view").set_value(name).run()


def planner_of(app: AppTest):
    return app.session_state["service"]


def test_every_manager_view_loads_without_errors(manager: AppTest) -> None:
    for name in VIEWS:
        open_view(manager, name)
        assert not manager.exception, (name, [str(e.value) for e in manager.exception])


def test_the_roster_view_shows_the_week_grid_and_the_reasons(manager: AppTest) -> None:
    open_view(manager, "Roster")
    grid = manager.dataframe[0].value
    assert grid.shape == (16, 7)
    assert len(manager.metric) == 4
    assignment = planner_of(manager).current().solution.plan.assignments[0]
    manager.selectbox(key="reason_worker").set_value(assignment.worker_id).run()
    manager.selectbox(key="reason_day").set_value(assignment.day).run()
    text = " ".join(item.value for item in manager.markdown)
    assert "CONTRACT_HOURS" in text


def test_the_day_view_shows_the_board_the_relief_chain_and_the_breaks(manager: AppTest) -> None:
    open_view(manager, "Day")
    assert len(manager.metric) == 4
    assert len(manager.dataframe[0].value) > 0
    assert any("takes" in item.value for item in manager.code)
    assert len(manager.dataframe[1].value) > 0


def shifted_code(blocks, minutes: int) -> str:
    first = blocks[0].register_start + minutes
    last = blocks[0].register_end + minutes
    return f"C{first // 60:02d}{first % 60:02d}-{last // 60:02d}{last % 60:02d}"


def a_legal_edit(planner):
    plan = planner.current().solution.plan
    from ree.planner.api.service import Edit

    for assignment in plan.assignments:
        if len(assignment.pattern.blocks) != 1:
            continue
        for minutes in (30, -30, 60, -60):
            code = shifted_code(assignment.pattern.blocks, minutes)
            if not planner.check_edit(Edit(assignment.worker_id, assignment.day, code)).new_errors:
                return assignment, code
    raise AssertionError("no legal edit found")


def test_an_edit_that_breaks_a_rule_shows_the_checker_reason_and_changes_nothing(
    manager: AppTest,
) -> None:
    open_view(manager, "Edit and check")
    planner = planner_of(manager)
    version = planner.current().number
    assignment = planner.current().solution.plan.assignments[0]
    manager.selectbox(key="edit_worker").set_value(assignment.worker_id).run()
    manager.selectbox(key="edit_day").set_value(assignment.day).run()
    manager.text_input(key="edit_code").set_value("C0830-2130").run()
    manager.button(key="check_edit").click().run()
    assert not manager.exception
    assert any("breaks" in item.value for item in manager.error)
    codes = set(manager.dataframe[-1].value["code"])
    assert codes & {"OPEN_AND_CLOSE", "SHIFT_TOO_LONG", "WEEKLY_HOURS_CONTRACT", "DAILY_HOURS"}
    manager.text_input(key="edit_reason").set_value("needs a long day").run()
    manager.button(key="apply_override").click().run()
    assert planner.current().number == version
    assert any(r.kind == "override_blocked" for r in planner.audit_log())


def test_an_override_without_a_reason_is_refused(manager: AppTest) -> None:
    open_view(manager, "Edit and check")
    planner = planner_of(manager)
    assignment, code = a_legal_edit(planner)
    version = planner.current().number
    manager.selectbox(key="edit_worker").set_value(assignment.worker_id).run()
    manager.selectbox(key="edit_day").set_value(assignment.day).run()
    manager.text_input(key="edit_code").set_value(code).run()
    manager.text_input(key="edit_reason").set_value("").run()
    manager.button(key="apply_override").click().run()
    assert any("short reason" in item.value for item in manager.warning)
    assert planner.current().number == version


def test_a_legal_override_with_a_reason_creates_a_new_roster_version(manager: AppTest) -> None:
    open_view(manager, "Edit and check")
    planner = planner_of(manager)
    assignment, code = a_legal_edit(planner)
    version = planner.current().number
    manager.selectbox(key="edit_worker").set_value(assignment.worker_id).run()
    manager.selectbox(key="edit_day").set_value(assignment.day).run()
    manager.text_input(key="edit_code").set_value(code).run()
    manager.text_input(key="edit_reason").set_value("moves half an hour for the delivery").run()
    manager.button(key="apply_override").click().run()
    assert not manager.exception
    assert planner.current().number == version + 1
    kinds = [r.kind for r in planner.audit_log()]
    assert "override" in kinds


def test_a_busy_spell_produces_a_proposal_that_waits_for_the_manager(manager: AppTest) -> None:
    open_view(manager, "Live proposals")
    planner = planner_of(manager)
    day = next(a.day for a in planner.current().solution.plan.assignments)
    manager.selectbox(key="live_day").set_value(day).run()
    for minute in (600, 605, 610):
        manager.number_input(key="reading_minute").set_value(minute).run()
        manager.slider(key="reading_busyness").set_value(0.97).run()
        manager.button(key="send_reading").click().run()
    proposals = planner.proposals(day)
    assert proposals and proposals[0].state == "pending"
    before = planner.daily(day).plan.segments
    assert any("open register" in item.value for item in manager.markdown)
    key = f"accept_{proposals[0].proposal_id}"
    manager.button(key=key).click().run()
    assert not manager.exception
    assert planner.proposals(day)[0].state == "accepted"
    assert planner.daily(day).plan.segments != before


def test_the_fairness_and_audit_views_show_their_tables(manager: AppTest) -> None:
    open_view(manager, "Fairness")
    assert len(manager.metric) == 3
    assert len(manager.dataframe) == 2
    open_view(manager, "Audit log")
    kinds = " ".join(manager.dataframe[0].value["kind"])
    assert "roster_generated" in kinds


def test_the_posts_only_checkbox_generates_a_roster_that_ignores_the_scores(
    manager: AppTest,
) -> None:
    open_view(manager, "Roster")
    manager.sidebar.selectbox(key="backend").set_value("heuristic")
    manager.sidebar.checkbox(key="posts_only").check()
    manager.sidebar.button(key="generate").click().run()
    assert not manager.exception
    version = planner_of(manager).current()
    assert any("Posts-only mode" in note for note in version.notes)
    codes = {reason.code for reasons in version.explanations.values() for reason in reasons}
    assert not codes & {"RUSH_TOP_SCORE", "RUSH_EXPLORATION"}
    assert any(abs(z) > 0.0 for z in version.scores.z.values())


@pytest.fixture(scope="module")
def worker() -> AppTest:
    app = AppTest.from_file(str(WORKER), default_timeout=TIMEOUT)
    app.run()
    assert not app.exception
    return app


def test_the_worker_screen_shows_hours_shifts_breaks_and_reasons(worker: AppTest) -> None:
    assert len(worker.metric) == 2
    text = " ".join(item.value for item in [*worker.markdown, *worker.caption])
    assert "contract hours" in text.lower()
    assert "paid hours" in text


def test_the_worker_screen_never_shows_scores_or_tiers(worker: AppTest) -> None:
    for frame in worker.dataframe:
        assert not {"score", "tier", "z"} & set(frame.value.columns)
    visible = " ".join(item.value for item in [*worker.markdown, *worker.caption]).lower()
    for forbidden in ("tier a", "tier c", " z =", "log-speed"):
        assert forbidden not in visible


def test_a_swap_request_is_validated_by_the_checker(worker: AppTest) -> None:
    worker.selectbox(key="worker").set_value(
        next(iter(worker.selectbox(key="worker").options))
    ).run()
    assert not worker.exception
    if worker.button(key="check_swap") is not None:
        worker.button(key="check_swap").click().run()
        assert not worker.exception
        assert worker.success or worker.error
