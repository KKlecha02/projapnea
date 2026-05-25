import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_analyze_fields(client):
    with open("data/input.json") as f:
        payload = json.load(f)
    resp = client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    for field in ["apnea_events", "ahi", "severity", "score"]:
        assert field in body


def test_analyze_ahi_tolerance(client):
    with open("data/input.json") as f:
        payload = json.load(f)
    with open("data/expected_output.json") as f:
        expected = json.load(f)
    resp = client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"Accept": "application/json"},
    )
    result = resp.get_json()
    assert abs(result["ahi"] - expected["ahi"]) <= 1.0
