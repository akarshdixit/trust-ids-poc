"""Additions for the review panel: feature catalogue, detector benchmark, other base
learners inside the gate, per-round confusion counts, the live experiment and
live ingestion, and the comparison dashboards."""
import json
import os

import numpy as np
import pytest

from data import features
from data import load_iot23 as io
from experiments import benchmark as bm
from experiments.run_scenarios import ROUND_SIZE, run_scenario
from experiments.streams import build_stream, real_data_available
from models.online_ids import BASE_MODELS, OnlineIDS

needs_nbaiot = pytest.mark.skipif(not real_data_available(), reason="N-BaIoT not downloaded")
needs_iot23 = pytest.mark.skipif(not io.available(), reason="IoT-23 not downloaded")


# --- feature catalogue ----------------------------------------------------------

def test_nbaiot_names_decompose_into_115_documented_features():
    from data.fake_nbaiot import feature_names
    COLUMNS = feature_names()
    parts = [features.parse_nbaiot_name(n) for n in COLUMNS]
    assert len(COLUMNS) == 115 and all(parts)
    counts = {g: sum(p[0] == g for p in parts) for g in features.NBAIOT_GROUPS}
    assert counts == {"MI_dir": 15, "H": 15, "HH": 35, "HH_jit": 15, "HpHp": 35}
    assert {p[1] for p in parts} == set(features.NBAIOT_WINDOWS)
    assert all("unknown" not in features.describe(n) for n in COLUMNS)


def test_jitter_features_are_described_as_time_not_size():
    assert "inter-packet time" in features.describe("HH_jit_L0.1_mean")
    assert "packet size" in features.describe("HH_L0.1_mean")


def test_every_iot23_feature_has_a_meaning():
    assert set(io.feature_names()) == set(features.IOT23_FEATURES)
    rows = features.catalogue(io.feature_names())
    assert all(r[1] == "flow" and r[4] for r in rows)


# --- benchmark protocol -----------------------------------------------------------

def test_temporal_split_never_tests_on_the_past():
    a = np.arange(1000)[:, None].astype(float)
    tr, te = bm._temporal(a, 100, 50)
    assert tr.max() < te.min() and len(tr) == 100 and len(te) == 50
    assert tr.min() == 0 and te.max() == 999            # stride spans each period


def test_metrics_on_a_known_confusion():
    y = [0, 0, 0, 1, 1, 1, 1, 1]
    p = [0, 0, 1, 1, 1, 1, 0, 1]
    m = bm.metrics(y, p)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(4 / 5) and m["recall"] == pytest.approx(4 / 5)
    assert m["fpr"] == pytest.approx(1 / 3)


@pytest.fixture
def toy_split(monkeypatch):
    rng = np.random.default_rng(0)

    def split(root=None, scale=1.0):
        def make(n, shift):
            X = rng.normal(size=(n, 6)) + shift
            X[:, 0] = np.abs(X[:, 0]) * 1e6         # one huge-range feature, like N-BaIoT
            return X
        Xtr = np.vstack([make(400, 0), make(200, 3)])
        Xte = np.vstack([make(80, 0), make(40, 3)])
        ytr = np.r_[np.zeros(400), np.ones(200)].astype(int)
        yte = np.r_[np.zeros(80), np.ones(40)].astype(int)
        return Xtr, ytr, Xte, yte, [f"f{i}" for i in range(6)], {}, {"toy": {}}
    monkeypatch.setitem(bm.SPLITS, "toy", split)
    return "toy"


def test_benchmark_runs_every_model_with_the_same_protocol(toy_split):
    res = bm.run(toy_split, repeats=1, online_train_cap=300, log=lambda *a: None)
    names = {r["model"] for r in res["models"]}
    assert bm.OURS in names and len(names) == 16
    for r in res["models"]:
        assert 0 <= r["f1"] <= 1 and r["single_latency_us"] > 0 and r["model_kb"] > 0
    assert max(r["f1"] for r in res["models"]) > 0.9   # a separable toy problem
    assert res["protocol"]["preprocessing"].endswith("training split")


def test_benchmark_figure_renders(toy_split, tmp_path):
    res = bm.run(toy_split, repeats=1, online_train_cap=300, log=lambda *a: None)
    bm.plot(res, str(tmp_path / "b.png"))
    assert (tmp_path / "b.png").stat().st_size > 10_000


# --- other base learners inside the pipeline -------------------------------------

@pytest.mark.parametrize("base", sorted(BASE_MODELS))
def test_every_base_learner_runs_through_every_arm(base):
    stream, names = build_stream("slow_poisoning", "synth", seed=3)
    log = run_scenario(stream, names, "sp", base_model=base)
    expected = {"static", "ungated", "gate_traffic_only", "gate_full", "gate_quarantine"}
    assert set(log) == expected | ({"ewc"} if base == "lr" else set())
    assert len({len(v) for v in log.values()}) == 1


def test_unknown_base_model_rejected():
    with pytest.raises(ValueError, match="base_model"):
        OnlineIDS(base_model="nope")


def test_reputation_gates_protect_a_different_learner():
    from experiments.sweeps import phase_accuracy
    stream, names = build_stream("slow_poisoning", "synth", seed=42)
    log = run_scenario(stream, names, "sp", include_ewc=False, base_model="arf")
    assert phase_accuracy(log, "gate_full", "recovery_check") == 1.0
    assert phase_accuracy(log, "gate_quarantine", "recovery_check") == 1.0
    assert phase_accuracy(log, "ungated", "recovery_check") < 0.5


def test_confusion_counts_are_opt_in_and_consistent():
    stream, names = build_stream("slow_poisoning", "synth", seed=1)
    plain = run_scenario(stream, names, "sp", include_ewc=False)
    assert "tp" not in plain["static"][0]
    log = run_scenario(stream, names, "sp", include_ewc=False, record_confusion=True)
    for arm, entries in log.items():
        for e in entries:
            assert e["tp"] + e["fp"] + e["tn"] + e["fn"] == ROUND_SIZE
            assert (e["tp"] + e["tn"]) / ROUND_SIZE == pytest.approx(e["round_accuracy"])
    for arm in plain:      # the counts change nothing else
        assert [{k: v for k, v in e.items() if k not in ("tp", "fp", "tn", "fn")}
                for e in log[arm]] == plain[arm]


# --- Zeek featurization for live lines -------------------------------------------

FIELDS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
          "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
          "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
          "tunnel_parents"]
LINE = "1.0\tC1\t10.0.0.1\t1\t10.0.0.2\t23\ttcp\t-\t0.5\t10\t0\tS0\t-\t-\t0\tS\t2\t120\t0\t0\t-"


def test_unlabelled_live_lines_are_featurized():
    df = io.featurize(FIELDS, [LINE.split()])
    assert list(df.columns) == io.feature_names() + ["label", "detail"]
    assert df.label.iat[0] == -1 and df.state_S0.iat[0] == 1 and df.orig_pkts.iat[0] == 2


def test_lines_not_matching_the_header_are_dropped():
    assert io.featurize(FIELDS, [["too", "short"]]).empty


# --- live experiment ---------------------------------------------------------------

@needs_iot23
def test_chronological_stream_is_in_time_order():
    from experiments.live_experiment import chronological_stream
    stream, names, ts = chronological_stream(n_calib=2000, max_flows=3000)
    assert len(stream) == 3000 and np.all(np.diff(ts) >= 0)
    assert {m["source"] for _, _, m in stream[:2000]} == {"calibration"}
    assert {m["source"] for _, _, m in stream[2000:]} == {"live"}


@needs_iot23
def test_live_experiment_paces_rounds_to_the_capture_clock():
    from experiments.live_experiment import LiveExperiment
    fake = {"t": 0.0}
    slept = []

    def clock():
        fake["t"] += 0.001
        return fake["t"]

    def sleep(s):
        slept.append(s)
        fake["t"] += s

    exp = LiveExperiment(speed=1000.0, max_flows=2400)
    summary = exp.run(sleep=sleep, clock=clock)
    assert summary["live_flows"] == 400 and len(exp.rounds) == 20
    assert slept and all(s > 0 for s in slept)
    for a, s in summary["arms"].items():
        assert s["tp"] + s["fp"] + s["tn"] + s["fn"] == 400
    # the replay cannot finish before the last live flow's scaled arrival time
    assert fake["t"] >= (exp.ts[2399] - exp.ts[2000]) / 1000.0
    assert all(r["lag_s"] >= 0 for r in exp.rounds)


@needs_iot23
def test_reputation_gates_survive_live_traffic():
    """Real capture, real order: naive self-learning breaks the detector, the gates do not."""
    from experiments.live_experiment import LiveExperiment
    s = LiveExperiment(speed=0, max_flows=8000).run()["arms"]
    assert s["gate_full"]["accuracy"] > 0.95 and s["gate_quarantine"]["accuracy"] > 0.95
    assert s["ungated"]["accuracy"] < s["gate_full"]["accuracy"] - 0.2


# --- backend live endpoints ----------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import server
    monkeypatch.setattr(server, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "RUNS", {})
    monkeypatch.setattr(server, "LIVE", {})
    monkeypatch.delenv("TRUST_IDS_TOKEN", raising=False)
    return TestClient(server.app)


def test_live_page_served(client):
    r = client.get("/live")
    assert r.status_code == 200 and "Live experiment" in r.text


def test_live_refuses_without_data(client, monkeypatch):
    monkeypatch.setattr(io, "available", lambda root=None: False)
    assert client.post("/api/live", json={"speed": 0}).status_code == 409


@pytest.mark.parametrize("body", [{"speed": -1}, {"max_flows": 10}])
def test_live_request_validated(client, body):
    assert client.post("/api/live", json=body).status_code == 422


@needs_iot23
def test_live_replay_streams_rounds(client):
    r = client.post("/api/live", json={"speed": 0, "max_flows": 2400})
    assert r.status_code == 202
    lid = r.json()["live_id"]
    with client.stream("GET", f"/api/live/{lid}/events") as s:
        body = s.read().decode()
    assert body.count("event: round") == 20 and "event: done" in body
    assert client.get(f"/api/live/{lid}").json()["summary"]["live_flows"] == 400


@needs_iot23
def test_ingest_scores_raw_zeek_lines(client):
    path = os.path.join(io.DEFAULT_ROOT, io.MIRAI, "conn.log.labeled")
    fields, rows = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                fields = line.split()[1:]
            elif not line.startswith("#") and line.strip():
                rows.append(line.rstrip("\n"))
            if len(rows) == 300:
                break
    r = client.post("/api/live/ingest", json={"fields": fields, "lines": rows})
    assert r.status_code == 200
    j = r.json()
    assert len(j["predictions"]) == 300 and j["latency_us_per_flow"] > 0
    assert j["running"]["accuracy_on_labelled"] > 0.8
    unl = client.post("/api/live/ingest", json={"fields": FIELDS, "lines": [LINE]}).json()
    assert "label" not in unl["predictions"][0]


def test_ingest_rejects_garbage(client):
    assert client.post("/api/live/ingest", json={"fields": FIELDS, "lines": []}).status_code == 422


# --- data profile and dashboards -----------------------------------------------------

def test_spread_reports_orders_of_magnitude():
    from experiments.data_profile import spread
    a = np.c_[np.random.default_rng(0).normal(size=100), 1e6 * np.random.default_rng(1).normal(size=100)]
    assert spread(a)["orders_of_magnitude"] == pytest.approx(6, abs=0.3)


def test_stream_layout_lists_every_phase():
    from experiments.data_profile import stream_layout
    lay = stream_layout("synth")
    assert [p["phase"] for p in lay["slow_poisoning"]] == [
        "calibration", "pre_attack", "slow_poison", "post_poison", "recovery_check"]
    assert sum(p["records"] for p in lay["slow_poisoning"]) == 2000


def test_comparison_tabs_render_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"),
                           default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", ["benchmark_real.json", "benchmark_iot23.json",
                                  "sweep_models_real.json", "sweep_models_iot23.json",
                                  "live_experiment_iot23.json"])
def test_committed_comparison_results_are_well_formed(name):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not generated")
    with open(path) as f:
        d = json.load(f)
    if name.startswith("benchmark"):
        assert any(m["model"] == bm.OURS for m in d["models"])
    elif name.startswith("sweep_models"):
        assert {r["base_model"] for r in d} == set(BASE_MODELS)
    else:
        assert d["summary"]["live_flows"] == len(d["rounds"]) * ROUND_SIZE
