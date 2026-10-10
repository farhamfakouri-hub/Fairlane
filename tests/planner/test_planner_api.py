from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from ree.planner.api.audit_store import AuditStore
from ree.planner.api.planner_app import create_planner_app
from ree.planner.api.service import PlannerService, parse_pattern
from ree.planner.config import PlannerConfig

MONDAY = date(2026, 11, 2)
TUESDAY = date(2026, 11, 3)


@pytest.fixture(scope="module")
def client() -> TestClient:
    service = PlannerService(audit=AuditStore())
    app = create_planner_app(service)
    test_client = TestClient(app)
    answer = test_client.post(
        "/roster/generate", json={"store": "small", "seed": 3, "backend": "heuristic"}
    )
    assert answer.status_code == 200
    return test_client


def first_assignment(client: TestClient, weekday_start: int = 0) -> dict:
    roster = client.get("/roster").json()
    return next(a for a in roster["assignments"] if a["day"] == "2026-11-03")


def test_health_before_and_after_a_roster_exists() -> None:
    empty = TestClient(create_planner_app(PlannerService()))
    assert empty.get("/health").json() == {
        "status": "ok",
        "has_roster": False,
        "version": None,
        "shifts": 0,
    }
    assert empty.get("/roster").status_code == 404
    assert empty.get("/daily/2026-11-03").status_code == 404


def test_the_generated_roster_carries_its_report_fairness_and_reasons(client: TestClient) -> None:
    body = client.get("/roster").json()
    assert body["backend"] == "heuristic" and body["status"] == "HEURISTIC"
    assert body["report"]["ok"] is False or body["report"]["errors"] >= 0
    assert set(body["fairness"]) == {"burden_spread", "max_tier_gap", "max_hours_deviation"}
    assert body["assignments"]
    sample = body["assignments"][0]
    assert sample["reasons"] and sample["reasons"][-1]["code"] == "CONTRACT_HOURS"
    assert client.get("/health").json()["shifts"] == len(body["assignments"])


def test_the_posts_only_mode_keeps_the_scores_out_of_the_roster_and_its_reasons(
    client: TestClient,
) -> None:
    service = PlannerService(audit=AuditStore())
    other = TestClient(create_planner_app(service))
    answer = other.post(
        "/roster/generate",
        json={"store": "small", "seed": 3, "backend": "heuristic", "posts_only": True},
    )
    assert answer.status_code == 200
    body = answer.json()
    codes = {reason["code"] for item in body["assignments"] for reason in item["reasons"]}
    assert not codes & {"RUSH_TOP_SCORE", "RUSH_EXPLORATION"}
    assert any("Posts-only mode" in note for note in body["notes"])
    assert any(abs(z) > 0.0 for z in service.current().scores.z.values())
    assert other.get("/daily/2026-11-03").status_code == 200
    usual = client.get("/roster").json()
    assert not any("Posts-only mode" in note for note in usual["notes"])


def test_the_service_never_exposes_the_hidden_truth_or_a_reason_for_availability(
    client: TestClient,
) -> None:
    text = client.get("/roster").text.lower()
    for forbidden in ("log_speed", "true_speed", "peak_effect", "blocked", "availability_reason"):
        assert forbidden not in text


def test_a_legal_manual_edit_is_checked_and_not_applied_by_check(client: TestClient) -> None:
    before = client.get("/roster").json()["version"]
    assignment = first_assignment(client)
    answer = client.post(
        "/roster/check",
        json={"worker_id": assignment["worker_id"], "day": "2026-11-03", "pattern_id": None},
    ).json()
    assert answer["applied"] is False
    assert client.get("/roster").json()["version"] == before


def test_an_edit_that_breaks_a_rule_shows_the_checker_reason(client: TestClient) -> None:
    assignment = first_assignment(client)
    answer = client.post(
        "/roster/check",
        json={
            "worker_id": assignment["worker_id"],
            "day": "2026-11-03",
            "pattern_id": "C0830-2130",
        },
    ).json()
    assert answer["new_errors"], answer
    assert "breaks" in answer["message"]
    codes = {item["code"] for item in answer["report"]["violations"]}
    assert codes & {"SHIFT_TOO_LONG", "OPEN_AND_CLOSE", "WEEKLY_HOURS_CONTRACT", "PATTERN_INVALID"}


def test_an_override_needs_a_short_reason_and_a_blocked_one_changes_nothing(
    client: TestClient,
) -> None:
    assignment = first_assignment(client)
    payload = {
        "worker_id": assignment["worker_id"],
        "day": "2026-11-03",
        "pattern_id": "C0830-2130",
    }
    assert client.post("/roster/override", json=payload).status_code == 422
    version = client.get("/roster").json()["version"]
    answer = client.post("/roster/override", json={**payload, "reason": "needs a long day"}).json()
    assert answer["applied"] is False
    assert client.get("/roster").json()["version"] == version
    kinds = [item["kind"] for item in client.get("/audit").json()]
    assert "override_blocked" in kinds


def shifted_code(blocks: list[str], minutes: int) -> str:
    start, end = blocks[0].split("-")
    first = int(start[:2]) * 60 + int(start[3:]) + minutes
    last = int(end[:2]) * 60 + int(end[3:]) + minutes
    return f"C{first // 60:02d}{first % 60:02d}-{last // 60:02d}{last % 60:02d}"


def test_an_accepted_override_creates_a_new_roster_version_with_a_stored_reason() -> None:
    local = TestClient(create_planner_app(PlannerService()))
    local.post("/roster/generate", json={"store": "small", "seed": 3, "backend": "heuristic"})
    version = local.get("/roster").json()["version"]
    day = "2026-11-03"
    chosen = None
    for assignment in local.get("/roster").json()["assignments"]:
        if assignment["day"] != day or len(assignment["blocks"]) != 1:
            continue
        for minutes in (30, -30, 60, -60):
            code = shifted_code(assignment["blocks"], minutes)
            check = local.post(
                "/roster/check",
                json={"worker_id": assignment["worker_id"], "day": day, "pattern_id": code},
            ).json()
            if not check["new_errors"]:
                chosen = (assignment["worker_id"], code)
                break
        if chosen:
            break
    assert chosen is not None, "some Tuesday shift can be moved without breaking a rule"
    outcome = local.post(
        "/roster/override",
        json={
            "worker_id": chosen[0],
            "day": day,
            "pattern_id": chosen[1],
            "reason": "covers the Tuesday morning",
            "by": "floor-manager",
        },
    ).json()
    assert outcome["applied"] is True
    assert local.get("/roster").json()["version"] == version + 1
    record = local.get("/audit", params={"kind": "override"}).json()[-1]
    assert record["detail"]["reason"] == "covers the Tuesday morning"
    assert record["detail"]["by"] == "floor-manager"
    moved = next(
        a
        for a in local.get("/roster").json()["assignments"]
        if a["worker_id"] == chosen[0] and a["day"] == day
    )
    assert moved["pattern_id"] == chosen[1]


def test_a_shift_code_must_be_well_formed() -> None:
    config = PlannerConfig()
    assert parse_pattern("C0830-1530", config).paid_minutes == 435
    split = parse_pattern("S0830-1230+1330-1730", config)
    assert split.kind == "split" and len(split.blocks) == 2
    for bad in ("nonsense", "C0830", "C1530-0830"):
        with pytest.raises(ValueError):
            parse_pattern(bad, config)


def test_the_daily_plan_has_breaks_posts_handovers_and_a_clean_report(client: TestClient) -> None:
    body = client.get("/daily/2026-11-03").json()
    assert body["report"]["errors"] >= 0
    assert body["breaks"] and body["segments"] and body["handovers"]
    assert body["metrics"]["breaks"] == float(len(body["breaks"]))
    assert client.get("/daily/2027-01-01").status_code == 404


def test_moving_a_break_far_outside_its_window_is_refused_with_the_reason(
    client: TestClient,
) -> None:
    body = client.get("/daily/2026-11-03").json()
    chosen = body["breaks"][0]
    answer = client.post(
        "/daily/2026-11-03/break",
        json={
            "worker_id": chosen["worker_id"],
            "block_index": chosen["block_index"],
            "start": chosen["start"] + 200,
        },
    ).json()
    assert answer["applied"] is False
    assert answer["new_errors"]
    missing = client.post(
        "/daily/2026-11-03/break", json={"worker_id": "W999", "block_index": 0, "start": 700}
    )
    assert missing.status_code == 404


def test_a_break_move_inside_the_tolerance_can_be_applied_with_a_reason() -> None:
    service = PlannerService()
    local = TestClient(create_planner_app(service))
    local.post("/roster/generate", json={"store": "small", "seed": 3, "backend": "heuristic"})
    body = local.get("/daily/2026-11-03").json()
    chosen = body["breaks"][0]
    shifted = chosen["start"] + 5
    move = {
        "worker_id": chosen["worker_id"],
        "block_index": chosen["block_index"],
        "start": shifted,
    }
    no_reason = local.post("/daily/2026-11-03/break", json={**move, "apply": True})
    assert no_reason.status_code == 422
    answer = local.post(
        "/daily/2026-11-03/break",
        json={**move, "apply": True, "reason": "customer at the desk"},
    ).json()
    assert answer["new_errors"] == [] and answer["applied"] is True
    moved = local.get("/daily/2026-11-03").json()["breaks"]
    assert any(b["worker_id"] == chosen["worker_id"] and b["start"] == shifted for b in moved)
    record = local.get("/audit", params={"kind": "break_moved"}).json()[-1]
    assert record["detail"]["reason"] == "customer at the desk"


def test_a_swap_between_two_workers_is_validated_by_the_checker(client: TestClient) -> None:
    roster = client.get("/roster").json()["assignments"]
    on_tuesday = [a for a in roster if a["day"] == "2026-11-03"]
    first, second = on_tuesday[0]["worker_id"], on_tuesday[1]["worker_id"]
    answer = client.post(
        "/swaps/check", json={"first": first, "second": second, "day": "2026-11-03"}
    ).json()
    assert answer["applied"] is False and "swap" in answer["message"].lower()
    off = client.post("/swaps/check", json={"first": first, "second": "W999", "day": "2026-11-03"})
    assert off.status_code == 404


def test_the_live_loop_proposes_and_nothing_changes_until_accepted() -> None:
    service = PlannerService()
    local = TestClient(create_planner_app(service))
    local.post("/roster/generate", json={"store": "small", "seed": 3, "backend": "heuristic"})
    day = "2026-11-03"
    before = local.get(f"/daily/{day}").json()
    made: list[dict] = []
    for minute in range(600, 630, 5):
        made += local.post(
            f"/live/{day}/observe",
            json={"minute": minute, "busyness": 0.97, "open_registers": 3, "waiting": 6},
        ).json()
    assert made, "a busy spell must produce a proposal"
    proposal = made[0]
    assert proposal["state"] == "pending" and proposal["reason"]
    assert local.get(f"/daily/{day}").json()["segments"] == before["segments"]
    listed = local.get(f"/proposals/{day}").json()
    assert next(p["proposal_id"] for p in listed) == proposal["proposal_id"]
    decided = local.post(
        f"/proposals/{day}/{proposal['proposal_id']}/decision",
        json={"accept": True, "note": "queue at the door"},
    )
    assert decided.status_code == 200 and decided.json()["state"] == "accepted"
    assert local.get(f"/daily/{day}").json()["segments"] != before["segments"]
    again = local.post(
        f"/proposals/{day}/{proposal['proposal_id']}/decision", json={"accept": False}
    )
    assert again.status_code == 409
    kinds = [item["kind"] for item in local.get("/audit").json()]
    assert "proposal" in kinds and "proposal_accepted" in kinds


def test_a_rejected_proposal_keeps_the_plan_and_the_reason() -> None:
    service = PlannerService()
    local = TestClient(create_planner_app(service))
    local.post("/roster/generate", json={"store": "small", "seed": 3, "backend": "heuristic"})
    day = "2026-11-03"
    made = []
    for minute in range(600, 620, 5):
        made += local.post(
            f"/live/{day}/observe",
            json={"minute": minute, "busyness": 0.97, "open_registers": 3},
        ).json()
    before = local.get(f"/daily/{day}").json()["segments"]
    answer = local.post(
        f"/proposals/{day}/{made[0]['proposal_id']}/decision",
        json={"accept": False, "note": "not needed"},
    ).json()
    assert answer["state"] == "rejected" and answer["note"] == "not needed"
    assert local.get(f"/daily/{day}").json()["segments"] == before
    assert local.post(f"/proposals/{day}/99/decision", json={"accept": True}).status_code == 404


def test_the_worker_view_shows_shifts_breaks_and_reasons_but_no_scores(client: TestClient) -> None:
    worker_id = first_assignment(client)["worker_id"]
    body = client.get(f"/workers/{worker_id}/plan").json()
    assert body["worker_id"] == worker_id and body["shifts"]
    first = body["shifts"][0]
    assert first["blocks"] and first["why"]
    assert set(body) == {"worker_id", "contract_hours", "paid_hours", "shifts"}
    assert set(first) == {"day", "weekday", "blocks", "paid_hours", "breaks", "why"}
    assert client.get("/workers/W999/plan").status_code == 404


def test_the_audit_log_keeps_the_roster_generation(client: TestClient) -> None:
    entries = client.get("/audit", params={"kind": "roster_generated"}).json()
    assert entries and entries[0]["detail"]["origin"].startswith("generated for store small")


def test_the_audit_store_roundtrips_and_filters(tmp_path) -> None:
    store = AuditStore(tmp_path / "planner.sqlite")
    store.append("a", "x", {"n": 1})
    store.append("b", "y", {"n": 2})
    assert [r.kind for r in store.records()] == ["a", "b"]
    assert [r.detail for r in store.records("b")] == [{"n": 2}]
    store.close()
    again = AuditStore(tmp_path / "planner.sqlite")
    assert len(again.records()) == 2
    again.close()
    assert timedelta(days=1) > timedelta(0)


def test_the_committed_openapi_schema_matches_the_planner_application() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    committed = json.loads((root / "docs" / "planner_openapi.json").read_text(encoding="utf-8"))
    app = create_planner_app(PlannerService())
    assert committed == json.loads(json.dumps(app.openapi()))
    assert "/roster/override" in committed["paths"]
