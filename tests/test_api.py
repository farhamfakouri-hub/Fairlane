from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factories import make_record
from ree.api.app import create_app
from ree.api.probe import get_json
from ree.api.provider import InMemoryScoreProvider, ScoreProvider
from ree.config import Config
from ree.domain.models import CashierScoreRecord, Interval
from ree.engine.pipeline import PipelineArtifacts

ROOT = Path(__file__).resolve().parents[1]


def _record(
    worker_id: str,
    peak: float,
    tier: str = "B",
    confidence: float = 0.8,
    computed_at: datetime | None = None,
) -> CashierScoreRecord:
    record = make_record(worker_id, score_peak=peak, score_off=peak - 2.0)
    components = record.components.model_copy(
        update={"confidence_peak": confidence, "confidence_off": confidence}
    )
    update: dict[str, object] = {
        "tier_peak": tier,
        "tier_off": tier,
        "components": components,
        "interval_peak": Interval(lower=peak - 2, upper=peak + 2),
    }
    if computed_at is not None:
        update["computed_at"] = computed_at
    return record.model_copy(update=update)


@pytest.fixture
def records() -> list[CashierScoreRecord]:
    return [
        _record("W001", 70.0, "A"),
        _record("W002", 55.0, "B"),
        _record("W003", 40.0, "C"),
        _record("W004", 80.0, "unrated", confidence=0.1),
    ]


@pytest.fixture
def client(records: list[CashierScoreRecord], tmp_path: Path) -> TestClient:
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "api"})
    return TestClient(create_app(config, InMemoryScoreProvider(records)))


def test_health_reports_the_model_and_the_cohort(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_workers"] == 4
    assert body["model_version"] == "ree-test"
    assert body["as_of"].startswith("2026-01-25")


def test_health_without_scores_is_still_ok(tmp_path: Path) -> None:
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "empty"})
    empty = TestClient(create_app(config, InMemoryScoreProvider([])))
    body = empty.get("/health").json()
    assert body == {"status": "ok", "model_version": None, "n_workers": 0, "as_of": None}
    assert empty.get("/scores").json() == []


def test_the_in_process_probe_answers_exactly_like_the_test_client(
    records: list[CashierScoreRecord], tmp_path: Path
) -> None:
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "probe"})
    app = create_app(config, InMemoryScoreProvider(records))
    reference = TestClient(app)
    requests = [
        ("/health", ""),
        ("/scores", ""),
        ("/scores/W002", ""),
        ("/scores/ranking", "context=peak&min_confidence=0.5"),
        ("/scores/ranking", "context=off"),
    ]
    for path, query in requests:
        status, body = get_json(app, path, query)
        expected = reference.get(
            path, params=dict(pair.split("=") for pair in query.split("&") if pair)
        )
        assert status == expected.status_code == 200
        assert body == expected.json()


def test_the_in_process_probe_reports_errors_as_json(
    records: list[CashierScoreRecord], tmp_path: Path
) -> None:
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "probe"})
    app = create_app(config, InMemoryScoreProvider(records))
    status, body = get_json(app, "/scores/W999")
    assert status == 404 and "W999" in body["detail"]
    status, body = get_json(app, "/scores/ranking", "context=never")
    assert status == 422 and body["detail"]


def test_scores_expose_drivers_model_version_and_the_appeal_field(client: TestClient) -> None:
    listing = client.get("/scores").json()
    assert [item["worker_id"] for item in listing] == ["W001", "W002", "W003", "W004"]
    first = listing[0]
    assert first["drivers"] and first["model_version"] == "ree-test"
    assert (
        first["appeal"]["status"] == "none" and "POST /appeals" in first["appeal"]["how_to_request"]
    )
    single = client.get("/scores/W002")
    assert single.status_code == 200 and single.json()["worker_id"] == "W002"
    assert client.get("/scores/W999").status_code == 404


def test_the_ranking_route_is_not_captured_by_the_worker_route(client: TestClient) -> None:
    response = client.get("/scores/ranking", params={"context": "peak"})
    assert response.status_code == 200
    assert response.json()["context"] == "peak"


def test_ranking_places_unrated_workers_at_the_cohort_average(client: TestClient) -> None:
    ranking = client.get("/scores/ranking", params={"context": "peak"}).json()["ranking"]
    assert [item["rank"] for item in ranking] == [1, 2, 3, 4]
    order = [item["worker_id"] for item in ranking]
    assert order[0] == "W001" and order[-1] == "W003"
    unrated = next(item for item in ranking if item["worker_id"] == "W004")
    average = (70.0 + 55.0 + 40.0) / 3
    assert unrated["unrated"] and unrated["effective_score"] == pytest.approx(average)
    assert unrated["score"] == 80.0
    assert unrated["tier"] == "unrated"
    scores = [item["effective_score"] for item in ranking]
    assert scores == sorted(scores, reverse=True)


def test_min_confidence_turns_low_confidence_scores_into_unrated_placements(
    client: TestClient,
) -> None:
    ranking = client.get(
        "/scores/ranking", params={"context": "off", "min_confidence": 0.9}
    ).json()["ranking"]
    assert all(item["unrated"] for item in ranking)
    assert len({item["effective_score"] for item in ranking}) == 1
    relaxed = client.get("/scores/ranking", params={"context": "off", "min_confidence": 0.5}).json()
    flags = {item["worker_id"]: item["unrated"] for item in relaxed["ranking"]}
    assert flags == {"W001": False, "W002": False, "W003": False, "W004": True}


def test_invalid_queries_are_rejected(client: TestClient) -> None:
    assert client.get("/scores/ranking", params={"context": "night"}).status_code == 422
    assert client.get("/scores/ranking").status_code == 422
    assert (
        client.get("/scores/ranking", params={"context": "peak", "min_confidence": 1.5}).status_code
        == 422
    )


def test_ranking_uses_the_latest_run_up_to_as_of(tmp_path: Path) -> None:
    early = datetime(2026, 1, 11, 20, 0, 0)
    late = datetime(2026, 1, 25, 20, 0, 0)
    runs = [
        _record("W001", 40.0, computed_at=early),
        _record("W002", 60.0, computed_at=early),
        _record("W001", 80.0, computed_at=late),
        _record("W002", 50.0, computed_at=late),
    ]
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "runs"})
    test_client = TestClient(create_app(config, InMemoryScoreProvider(runs)))
    latest = test_client.get("/scores/ranking", params={"context": "peak"}).json()["ranking"]
    assert latest[0]["worker_id"] == "W001"
    earlier = test_client.get(
        "/scores/ranking", params={"context": "peak", "as_of": "2026-01-15T00:00:00"}
    ).json()["ranking"]
    assert earlier[0]["worker_id"] == "W002"
    before_everything = test_client.get(
        "/scores/ranking", params={"context": "peak", "as_of": "2025-01-01T00:00:00"}
    ).json()["ranking"]
    assert before_everything == []


def test_appeals_are_stored_and_shown_on_the_score(client: TestClient) -> None:
    created = client.post("/appeals", json={"worker_id": "W002", "reason": "I covered a new till"})
    assert created.status_code == 201
    body = created.json()
    assert (
        body["appeal_id"] == 1 and body["status"] == "open" and body["model_version"] == "ree-test"
    )
    score = client.get("/scores/W002").json()
    assert score["appeal"] == {
        "status": "open",
        "last_appeal_id": 1,
        "how_to_request": score["appeal"]["how_to_request"],
    }
    assert client.get("/scores/W001").json()["appeal"]["status"] == "none"
    listing = client.get("/appeals", params={"worker_id": "W002"}).json()
    assert [item["appeal_id"] for item in listing] == [1]
    assert client.get("/appeals").json()[0]["reason"] == "I covered a new till"


def test_appeal_requests_are_validated(client: TestClient) -> None:
    assert (
        client.post("/appeals", json={"worker_id": "W999", "reason": "unknown worker"}).status_code
        == 404
    )
    assert client.post("/appeals", json={"worker_id": "W001", "reason": "x"}).status_code == 422
    assert client.post("/appeals", json={"worker_id": "W001"}).status_code == 422
    assert (
        client.post(
            "/appeals", json={"worker_id": "W001", "reason": "valid reason", "extra": 1}
        ).status_code
        == 422
    )


def test_part_two_can_supply_its_own_provider(tmp_path: Path) -> None:
    class FixedProvider:
        def get_ranking(self, context, as_of=None, min_confidence=0.0):
            return InMemoryScoreProvider([_record("W010", 61.0)]).get_ranking(context)

        def get_score(self, worker_id):
            return _record("W010", 61.0) if worker_id == "W010" else None

        def list_scores(self):
            return [_record("W010", 61.0)]

        def latest_as_of(self):
            return datetime(2026, 1, 25, 20, 0, 0)

    provider: ScoreProvider = FixedProvider()
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "fixed"})
    test_client = TestClient(create_app(config, provider))
    assert test_client.get("/scores/W010").status_code == 200
    assert (
        test_client.get("/scores/ranking", params={"context": "peak"}).json()["ranking"][0][
            "worker_id"
        ]
        == "W010"
    )


def test_pipeline_output_flows_through_the_provider(
    artifacts: PipelineArtifacts, tmp_path: Path
) -> None:
    provider = InMemoryScoreProvider(artifacts.records)
    config = Config().with_overrides({"paths.data_dir": str(tmp_path), "paths.store_name": "flow"})
    test_client = TestClient(create_app(config, provider))
    ranking = test_client.get("/scores/ranking", params={"context": "peak"}).json()["ranking"]
    assert len(ranking) == len(artifacts.records)
    assert {item["worker_id"] for item in ranking} == {r.worker_id for r in artifacts.records}
    unrated = [item for item in ranking if item["unrated"]]
    rated = [item for item in ranking if not item["unrated"]]
    assert unrated and rated
    average = sum(item["score"] for item in rated) / len(rated)
    assert all(item["effective_score"] == pytest.approx(average) for item in unrated)


def test_the_committed_openapi_schema_matches_the_application(tmp_path: Path) -> None:
    app = create_app(Config(), InMemoryScoreProvider([]))
    committed = json.loads((ROOT / "docs" / "openapi.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(app.openapi()))
    assert {"/health", "/scores", "/scores/ranking", "/scores/{worker_id}", "/appeals"} <= set(
        committed["paths"]
    )
