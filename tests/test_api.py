"""FastAPI endpoint tests using mocked providers (fully offline)."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeContainer, FakeLLMProvider, FakeUseCases, make_app


@pytest.fixture
def client(repo, llm):
    use_cases = FakeUseCases(repo)
    container = FakeContainer(use_cases=use_cases, llm=llm)
    return make_app(container), use_cases


def test_start_run(client):
    app_client, use_cases = client
    response = app_client.post(
        "/api/runs", json={"topic": "renewable energy", "model": "gemma2", "max_iterations": 2}
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["run_id"] == "run-123"
    assert body["status"] == "completed"
    assert use_cases.started == [("renewable energy", "gemma2", 2)]


def test_start_run_defaults(client):
    app_client, use_cases = client
    response = app_client.post("/api/runs", json={"topic": "deep learning"})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["run_id"] == "run-123"


def test_get_run(client):
    app_client, use_cases = client
    response = app_client.get("/api/runs/run-123")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == "run-123"
    assert body["status"] == "completed"
    assert body["report"]["title"] == "Report"
    assert len(body["steps"]) == 1
    assert body["steps"][0]["agent"] == "Planner"


def test_get_run_not_found(client):
    app_client, _ = client
    response = app_client.get("/api/runs/missing-id")
    assert response.status_code == 404


def test_list_runs(client):
    app_client, use_cases = client
    response = app_client.get("/api/runs")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["runs"][0]["run_id"] == "run-123"


def test_list_models(client):
    app_client, _ = client
    response = app_client.get("/api/models")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "qwen2.5:7b-instruct-q4_k_m" in body["models"]
    assert "gemma2:2b" in body["models"]
