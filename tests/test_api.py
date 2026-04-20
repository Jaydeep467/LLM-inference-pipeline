"""
Tests for the Flask REST API.
Run: pytest tests/ -v
"""

import json
import threading
import pytest

# ── Patch heavy model loading before importing api ──
import unittest.mock as mock

# Stub out the BatchProcessor so tests don't load GPT-2
class _FakeRequest:
    def __init__(self, prompt, **_):
        self.prompt   = prompt
        self.response = f"[mocked response for: {prompt[:30]}]"
        self.error    = None
        self.result   = threading.Event()
        self.result.set()

class _FakeProcessor:
    def submit(self, req): pass

with mock.patch("app.model.load_model", return_value=(mock.MagicMock(), mock.MagicMock())), \
     mock.patch("app.model.BatchProcessor", return_value=_FakeProcessor()):
    from app.api import app as flask_app
    import app.model as model_module
    model_module.get_processor = lambda: _FakeProcessor()


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Health ──────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


# ── Templates ───────────────────────────────────────
def test_list_templates(client):
    r = client.get("/templates")
    assert r.status_code == 200
    data = r.get_json()
    assert "templates" in data
    assert "default" in data["templates"]
    assert "instruct" in data["templates"]


# ── Predict ─────────────────────────────────────────
def test_predict_missing_prompt(client):
    r = client.post("/predict", json={})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_returns_response(client):
    with mock.patch("app.api.get_processor") as gp:
        fake_req_holder = {}

        def fake_submit(req):
            req.response = "Hello world"
            req.result.set()
            fake_req_holder["req"] = req

        proc = mock.MagicMock()
        proc.submit.side_effect = fake_submit
        gp.return_value = proc

        r = client.post("/predict", json={"prompt": "Say hello"})
        assert r.status_code == 200
        data = r.get_json()
        assert "response" in data
        assert data["response"] == "Hello world"


def test_predict_with_template(client):
    with mock.patch("app.api.get_processor") as gp:
        def fake_submit(req):
            req.response = "42"
            req.result.set()

        proc = mock.MagicMock()
        proc.submit.side_effect = fake_submit
        gp.return_value = proc

        r = client.post("/predict", json={
            "prompt": "What is 6x7?",
            "template": "qa",
        })
        assert r.status_code == 200


# ── Batch ────────────────────────────────────────────
def test_batch_missing_prompts(client):
    r = client.post("/batch", json={})
    assert r.status_code == 400


def test_batch_too_large(client):
    r = client.post("/batch", json={"prompts": ["x"] * 20})
    assert r.status_code == 400


def test_batch_returns_results(client):
    with mock.patch("app.api.get_processor") as gp:
        def fake_submit(req):
            req.response = "ok"
            req.result.set()

        proc = mock.MagicMock()
        proc.submit.side_effect = fake_submit
        gp.return_value = proc

        r = client.post("/batch", json={"prompts": ["hello", "world"]})
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 2
        assert len(data["results"]) == 2


# ── Metrics ──────────────────────────────────────────
def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.get_json()
    for key in ["total_requests", "latency_p50_ms", "latency_p99_ms", "requests_per_sec"]:
        assert key in data
