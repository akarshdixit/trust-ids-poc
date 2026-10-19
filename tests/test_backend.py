import json
import time

import pytest
from fastapi.testclient import TestClient

from backend import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "RUNS", {})
    monkeypatch.delenv("TRUST_IDS_TOKEN", raising=False)
    return TestClient(server.app)


def _wait(client, run_id, timeout=120, **kw):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}", **kw).json()
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.1)
    raise TimeoutError(run_id)


def _start(client, **body):
    r = client.post("/api/runs", json={"pace_ms": 0, **body})
    assert r.status_code == 202, r.text
    return r.json()["run_id"]


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["auth_required"] is False


def test_console_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "EventSource" in r.text


def test_precomputed_results(client):
    assert "slow_poisoning_log_synth" in client.get("/api/results").json()
    log = client.get("/api/results/slow_poisoning/synth").json()
    assert "gate_full" in log
    assert client.get("/api/results/slow_poisoning/bogus").status_code == 422


def test_run_streams_every_round_then_done(client):
    run_id = _start(client, scenario="slow_poisoning", source="synth", seed=3)
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as r:
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    rounds = [d for n, d in events if n == "round"]
    assert len(rounds) == 80
    assert [d["round"] for d in rounds] == list(range(1, 81))
    assert set(rounds[0]["arms"]) == {"static", "ungated", "gate_traffic_only", "gate_full",
                                      "ewc", "gate_quarantine"}
    assert events[-1][0] == "done"


def test_finished_run_is_persisted_and_reloaded(client, tmp_path, monkeypatch):
    run_id = _start(client, scenario="genuine_drift", source="synth", seed=1)
    run = _wait(client, run_id)
    assert run["status"] == "done"
    assert (tmp_path / f"{run_id}.json").exists()

    monkeypatch.setattr(server, "RUNS", {})        # simulate a restart
    server._load_saved_runs()
    again = client.get(f"/api/runs/{run_id}").json()
    assert again["status"] == "done" and again["events"] == run["events"]


def test_failed_run_reports_error(client, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such dataset")
    monkeypatch.setattr(server, "build_stream", boom)
    run = _wait(client, _start(client, source="real"))
    assert run["status"] == "error" and "no such dataset" in run["error"]


def test_noise_and_feature_drift_options_reach_the_pipeline(client):
    run = _wait(client, _start(client, source="synth", seed=4, feature_drift=True,
                               reputation_noise={"flip_rate": 0.4, "seed": 2}))
    reps = {round(e["arms"]["gate_full"]["reputation"], 3) for e in run["events"]}
    assert len(reps) > 2                           # oracle would give only 1.0 / 0.05


@pytest.mark.parametrize("body", [
    {"gate": {"w_attribution": 0, "w_topology": 0, "w_temporal": 0}},
    {"gate": {"trust_threshold": 1.5}},
    {"source": "nope"},
    {"reputation_noise": {"flip_rate": 2}},
    {"quarantine": {"feed_error": 0.5}},
    {"quarantine": {"quarantine_rounds": -1}},
])
def test_invalid_requests_rejected(client, body):
    assert client.post("/api/runs", json=body).status_code == 422


def test_quarantine_options_reach_the_pipeline(client):
    run = _wait(client, _start(client, source="synth", seed=42,
                               reputation_noise={"flip_rate": 0.4, "sigma": 0.05, "seed": 42},
                               quarantine={"quarantine_rounds": 0}))
    q = [e["arms"]["gate_quarantine"] for e in run["events"]]
    assert len(q) == len(run["events"]) and not any(r["rolled_back"] for r in q)
    off = _wait(client, _start(client, source="synth", seed=1, include_quarantine=False))
    assert "gate_quarantine" not in off["events"][0]["arms"]


def test_describe_reports_quarantine_and_rollback():
    base = {"phase": "slow_poison", "accuracy": 0.5, "drift_active": True,
            "trust_score": 0.6, "reputation": 0.4, "authorized": False}
    entries = {"gate_full": base, "gate_traffic_only": {**base, "authorized": True},
               "gate_quarantine": {**base, "condemned": True, "sprt_llr": -5.1,
                                   "rolled_back": 2}}
    line = server._describe(7, entries)
    assert "VETO" in line and "quarantine=CONDEMNED" in line and "ROLLBACK 2" in line


def test_concurrency_limit(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_ACTIVE_RUNS", 1)
    first = _start(client, source="synth", seed=1, **{"pace_ms": 20})
    r = client.post("/api/runs", json={"source": "synth"})
    assert r.status_code == 429
    _wait(client, first)
    assert client.post("/api/runs", json={"source": "synth", "pace_ms": 0}).status_code == 202


def test_token_auth(client, monkeypatch):
    monkeypatch.setenv("TRUST_IDS_TOKEN", "s3cret")
    assert client.get("/api/health").status_code == 200          # always open
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/runs", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/runs?token=s3cret").status_code == 200


def test_unknown_run_404(client):
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.get("/api/runs/deadbeef/events").status_code == 404


def test_log_stream_replays_run_logs(client):
    run_id = _start(client, source="synth", seed=2)
    _wait(client, run_id)
    with client.stream("GET", "/api/logs/stream?backlog=2000&follow=false") as r:
        body = r.read().decode()
    assert f"trust_ids.run.{run_id}" in body and "done:" in body
