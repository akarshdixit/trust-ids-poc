"""Real datasets: N-BaIoT (data/nbaiot/) and IoT-23 (data/iot23/).

Both are gitignored downloads, so every test that reads them skips when the
files are absent. The IoT-23 parser's format handling is also tested on a tiny
hand-written log, which always runs.
"""
import json
import os

import numpy as np
import pytest

from data import load_iot23 as io
from data import load_nbaiot as nb
from experiments.run_scenarios import run_scenario
from experiments.streams import build_stream, real_data_available
from experiments.sweeps import phase_accuracy

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ORIGINAL_ARMS = ("static", "ungated", "gate_traffic_only", "gate_full")

needs_nbaiot = pytest.mark.skipif(not real_data_available(),
                                  reason="N-BaIoT not downloaded to data/nbaiot/")
needs_iot23 = pytest.mark.skipif(not io.available(),
                                 reason="IoT-23 conn logs not downloaded to data/iot23/")


# --- N-BaIoT ------------------------------------------------------------------

@needs_nbaiot
@pytest.mark.parametrize("rel,rows", [
    ("benign_traffic.csv", 49_548),
    ("mirai_attacks/scan.csv", 107_685),
    ("gafgyt_attacks/combo.csv", 59_718),
])
def test_nbaiot_files_match_documented_shape(rel, rows):
    df = nb._read(os.path.join(nb.DEFAULT_ROOT, "Danmini_Doorbell", rel))
    assert df.shape == (rows, 115)
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert list(df.columns) == nb.feature_names()


@pytest.fixture(scope="module")
def real_logs():  # only requested by tests marked needs_nbaiot
    out = {}
    for scenario in ("slow_poisoning", "genuine_drift"):
        stream, names = build_stream(scenario, "real")
        out[scenario] = run_scenario(stream, names, scenario)
    return out


@needs_nbaiot
@pytest.mark.parametrize("scenario", ["slow_poisoning", "genuine_drift"])
def test_real_results_match_committed_logs(real_logs, scenario):
    with open(os.path.join(RESULTS, f"{scenario}_log_real.json")) as f:
        committed = json.load(f)
    fresh = json.loads(json.dumps(real_logs[scenario]))
    for arm in ORIGINAL_ARMS:
        assert fresh[arm] == committed[arm], arm


@needs_nbaiot
def test_real_headline_result(real_logs):
    log = real_logs["slow_poisoning"]
    assert phase_accuracy(log, "gate_full", "recovery_check") == 1.0
    assert phase_accuracy(log, "gate_traffic_only", "recovery_check") == pytest.approx(0.2)
    assert sum(r["authorized"] for r in log["gate_traffic_only"]) == 5
    assert not any(r["authorized"] for r in log["gate_full"])


@needs_nbaiot
def test_real_scenario1_is_a_null_result_without_feature_drift(real_logs):
    assert not any(r["drift_active"] for r in real_logs["genuine_drift"]["static"])


# --- IoT-23 parser on a hand-written log (always runs) ------------------------

HEADER = ("#separator \\x09\n#path\tconn\n"
          "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\t"
          "duration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\t"
          "history\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents{sep}"
          "label{sep}{last}\n")
ROW = ("1.0\tC1\t10.0.0.1\t1\t10.0.0.2\t80\t{proto}\t-\t{dur}\t{ob}\t0\t{state}\t-\t-\t0\t"
       "S\t3\t180\t0\t0\t-{sep}{label}{sep}{detail}\n")


def _write_log(path, sep, last, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = HEADER.format(sep=sep, last=last)
    for proto, dur, ob, state, label, detail in rows:
        body += ROW.format(proto=proto, dur=dur, ob=ob, state=state, label=label,
                           detail=detail, sep=sep)
    path.write_text(body + "#close\t2019\n")
    io._CACHE.clear()
    return str(path)


@pytest.mark.parametrize("sep,last", [("\t", "det_label"), ("   ", "detailed-label")])
def test_iot23_parser_handles_both_header_variants(tmp_path, sep, last):
    path = _write_log(tmp_path / "x" / "conn.log.labeled", sep, last, [
        ("tcp", "3.1", "100", "S0", "Malicious", "C&C"),
        ("udp", "-", "-", "SF", "benign", "-"),
        ("icmp", "0.5", "7", "OTH", "Benign", "-"),
    ])
    df = io.parse_conn_log(path)
    assert list(df.columns) == io.feature_names() + ["label", "detail"]
    assert df["label"].tolist() == [1, 0, 0]                  # case-insensitive
    assert df.loc[1, "duration"] == 0 and df.loc[1, "orig_bytes"] == 0   # "-" -> 0
    assert df.loc[0, "proto_tcp"] == 1 and df.loc[2, "proto_icmp"] == 1
    assert df.loc[1, "state_SF"] == 1 and df.loc[0, "state_SF"] == 0


def test_iot23_parser_rejects_non_zeek_file(tmp_path):
    p = tmp_path / "bad.log"
    p.write_text("not a zeek log\n")
    io._CACHE.clear()
    with pytest.raises(ValueError, match="#fields"):
        io.parse_conn_log(str(p))


def test_iot23_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="conn.log.labeled"):
        io.slow_poisoning_stream(root=str(tmp_path))


# --- IoT-23 on the downloaded captures ----------------------------------------

@needs_iot23
def test_iot23_capture_counts():
    m = io._log(io.MIRAI, None)
    assert len(m) == 23_145 and int(m.label.sum()) == 21_222
    b = io._log(io.BENIGN_B, None)
    assert len(b) == 1_374 and int(b.label.sum()) == 0


@needs_iot23
def test_iot23_streams_have_expected_phases():
    sp, names = build_stream("slow_poisoning", "iot23")
    assert len(sp) == 2000 and len(names) == 21
    assert [m["source"] for _, _, m in sp[-200:]] == ["recovery_check"] * 200
    gd, _ = build_stream("genuine_drift", "iot23")
    assert all(y == 0 for _, y, m in gd if m["source"] != "calib_attack")


NOISY_FEED = {"sigma": 0.05, "flip_rate": 0.3, "seed": 42}


@needs_nbaiot
def test_real_quarantine_survives_a_30pct_wrong_feed():
    stream, names = build_stream("slow_poisoning", "real", seed=42, reputation_noise=NOISY_FEED)
    log = run_scenario(stream, names, "sp", include_ewc=False)
    assert phase_accuracy(log, "gate_full", "recovery_check") < 0.5
    assert phase_accuracy(log, "gate_quarantine", "recovery_check") == 1.0


@needs_iot23
def test_iot23_quarantine_survives_a_30pct_wrong_feed():
    stream, names = build_stream("slow_poisoning", "iot23", seed=42, reputation_noise=NOISY_FEED)
    log = run_scenario(stream, names, "sp", include_ewc=False)
    assert phase_accuracy(log, "gate_full", "recovery_check") < 1.0
    assert phase_accuracy(log, "gate_quarantine", "recovery_check") == 1.0


@needs_iot23
def test_iot23_quarantine_adapts_to_benign_drift_the_one_shot_gate_refuses():
    stream, names = build_stream("genuine_drift", "iot23", seed=42)
    log = run_scenario(stream, names, "gd", include_ewc=False, feature_drift=True)
    assert (phase_accuracy(log, "gate_quarantine", "post_drift")
            > phase_accuracy(log, "gate_full", "post_drift") + 0.05)


@needs_iot23
def test_iot23_hold_must_outlast_the_feed_lag():
    from experiments.sweeps import poisoned_updates_kept
    stream, names = build_stream("slow_poisoning", "iot23", seed=42,
                                 reputation_noise={"lag": 600, "seed": 42})
    kept = {qr: poisoned_updates_kept(
        run_scenario(stream, names, "sp", include_ewc=False,
                     quarantine_params={"quarantine_rounds": qr}), "gate_quarantine", stream)
        for qr in (5, 20)}
    assert kept[5] > 0 and kept[20] == 0


@needs_iot23
def test_iot23_reproduces_the_poisoning_result():
    stream, names = build_stream("slow_poisoning", "iot23")
    log = run_scenario(stream, names, "sp")
    assert phase_accuracy(log, "gate_full", "recovery_check") == 1.0
    assert phase_accuracy(log, "gate_traffic_only", "recovery_check") < 0.5
