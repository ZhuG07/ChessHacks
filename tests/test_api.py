import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_weights

pytestmark = requires_weights


@pytest.fixture(scope="module")
def client():
    from app.server import app

    with TestClient(app) as c:
        yield c


def test_info(client):
    info = client.get("/api/info").json()
    assert info["network"]["parameters"] > 1_000_000
    assert info["search"]["multipv"] >= 1


def test_position_lists_legal_moves(client):
    pos = client.post("/api/position", json={"moves": []}).json()
    assert pos["turn"] == "w"
    assert len(pos["legal"]) == 20
    assert pos["status"]["over"] is False


def test_rejects_illegal_moves(client):
    r = client.post("/api/position", json={"moves": ["e2e5"]})
    assert r.status_code == 400
    r = client.post("/api/position", json={"moves": ["zz"]})
    assert r.status_code == 400


def test_engine_reply(client):
    r = client.post("/api/move", json={"moves": ["e2e4"], "think_time": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["position"]["turn"] == "w"
    assert body["position"]["history"][0] == "e4"
    a = body["analysis"]
    assert a["best_move"] == body["move"]
    assert 0.0 <= a["score"]["win"] <= 1.0
    assert a["lines"] and a["policy"]


def test_detects_checkmate(client):
    pos = client.post("/api/position", json={"moves": ["f2f3", "e7e5", "g2g4", "d8h4"]}).json()
    assert pos["status"] == {"over": True, "result": "0-1", "reason": "checkmate"}
    assert pos["check"] == "e1"
    assert client.post("/api/move", json={"moves": ["f2f3", "e7e5", "g2g4", "d8h4"]}).status_code == 409


def test_custom_start_fen(client):
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    a = client.post("/api/analyse", json={"start_fen": fen, "think_time": 0.3}).json()["analysis"]
    assert a["best_san"] == "Ra8#"
    assert a["score"]["mate"] == 1


def test_serves_frontend(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
