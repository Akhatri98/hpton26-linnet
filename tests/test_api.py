"""HTTP API: the web UI's contract."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from library.store import LocalJsonStore
from linnet import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    # A copy of data/, so sign-off tests never touch the real files.
    root = tmp_path / "data"
    for sub in ("ingredients", "products", "patients", "extracted"):
        shutil.copytree(Path("data") / sub, root / sub)
    shutil.copy(Path("data") / "groups.json", root / "groups.json")
    monkeypatch.setattr(api, "DATA", root)
    monkeypatch.setattr(api, "store", LocalJsonStore(root))
    return TestClient(api.app)


def _patient(client, pid):
    return client.get(f"/api/patients/{pid}").json()


def test_products_and_patients(client):
    names = {p["name"] for p in client.get("/api/products").json()}
    assert "nimodipine-30mg-cap" in names
    ids = {p["id"] for p in client.get("/api/patients").json()}
    assert "tier2-nimodipine-colestipol" in ids
    assert client.get("/api/patients/..%2Fgroups").status_code == 404


def test_check_tier2_names_the_conflict_with_evidence(client):
    r = client.post("/api/check", json=_patient(client, "tier2-nimodipine-colestipol")).json()
    assert r["outcome"] == "tier2"
    [c] = r["conflicts"]
    assert {x["id"] for x in c["label_rules"]} == {"nimodipine-q4h", "colestipol-other-drugs"}
    assert all(x["source"]["url"].startswith("https://") for x in c["label_rules"])


def test_check_tier0_false_alarm_and_timeline(client):
    r = client.post("/api/check", json=_patient(client, "tier0-levo-cipro-tums")).json()
    assert r["outcome"] == "tier0" and r["baseline"]["false_alarm"] is True
    ev = r["options"][0]["events"]
    assert [e["min"] for e in ev] == sorted(e["min"] for e in ev)
    assert any(e["kind"] == "dose" for e in ev) and r["rules_applied"]


def test_check_rejects_bad_input_readably(client):
    p = _patient(client, "tier0-levo-cipro")
    p["items"][0] = {"product": "nope", "prescribed_count": 1}
    r = client.post("/api/check", json=p)
    assert r.status_code == 422 and "not in the library" in r.text
    p["items"][0] = {"product": "nope"}
    assert client.post("/api/check", json=p).status_code == 422


def test_verify_writes_json_and_round_trips(client):
    r = client.post("/api/library/verify", json={"id": "product:nimodipine-30mg-cap", "verified": True})
    assert r.status_code == 200
    doc = json.loads((api.DATA / "products" / "nimodipine-30mg-cap.json").read_text(encoding="utf-8"))
    assert doc["verified"] is True
    items = {i["id"]: i for i in client.get("/api/library").json()["items"]}
    assert items["product:nimodipine-30mg-cap"]["verified"] is True
    assert client.post("/api/library/verify", json={"id": "nope", "verified": True}).status_code == 404


def test_same_patient_same_answer_every_time(client):
    # Each check runs in a fresh worker process, so earlier solves cannot change the result.
    p = json.loads((Path(__file__).parent / "regimen-state-sensitive.json").read_text(encoding="utf-8"))
    first = client.post("/api/check", json=p).json()
    for _ in range(2):
        assert client.post("/api/check", json=p).json() == first
    assert first["outcome"] == "tier1" and len(first["options"]) == 3


def test_patient_tiers_for_switcher_dots(client, monkeypatch):
    t = client.get("/api/patients/tiers").json()
    assert t["tier2-nimodipine-colestipol"] == "tier2"
    assert t["tier1-duplicate-acetaminophen"] == "tier1"
    assert t["tier0-levo-cipro"] == "tier0"
    nh = [k for k in t if k.startswith("nhanes-")]
    assert sum(t[k] == "tier1" for k in nh) >= 5 and sum(t[k] == "tier2" for k in nh) >= 3
    # Cached until a library or patient file changes.
    calls = []
    monkeypatch.setattr(api.isolated, "outcomes", lambda pts, data: calls.append(1) or t)
    client.get("/api/patients/tiers")
    assert calls == []
    f = api.DATA / "products" / "tums-es-750.json"
    f.write_text(f.read_text(encoding="utf-8") + " ", encoding="utf-8")
    client.get("/api/patients/tiers")
    assert calls == [1]
