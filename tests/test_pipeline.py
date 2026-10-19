"""End-to-end pipeline tests: regression against committed results, the new
arms/triggers, and the shared stream builder."""
import json
import os

import numpy as np
import pytest

from data import load_nbaiot as nb
from data import synth_stream as ss
from experiments.run_scenarios import run_scenario
from experiments.streams import build_stream
from experiments.sweeps import phase_accuracy
from models.online_ids import EWCOnlineIDS

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _roundtrip(log):
    return json.loads(json.dumps(log))


def _committed(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def synth_logs():
    # exactly the order run_scenarios.py --synth uses: poisoning first, then drift
    # continuing the same generator
    ss.reseed(42)
    poison = run_scenario(ss.slow_poisoning_stream(), ss.FEATURE_NAMES, "sp")
    drift = run_scenario(ss.genuine_drift_stream(), ss.FEATURE_NAMES, "gd")
    return poison, drift


def test_synth_slow_poisoning_matches_committed_results(synth_logs):
    assert _roundtrip(synth_logs[0]) == _committed("slow_poisoning_log_synth.json")


def test_synth_genuine_drift_matches_committed_results(synth_logs):
    assert _roundtrip(synth_logs[1]) == _committed("genuine_drift_log_synth.json")


def test_headline_result(synth_logs):
    poison = synth_logs[0]
    assert phase_accuracy(poison, "gate_full", "recovery_check") == 1.0
    assert phase_accuracy(poison, "gate_traffic_only", "recovery_check") <= 0.2
    assert not any(r["authorized"] for r in poison["gate_full"])
    assert any(r["authorized"] for r in poison["gate_traffic_only"])


def test_every_arm_logs_every_round(synth_logs):
    for log in synth_logs:
        lengths = {len(v) for v in log.values()}
        assert len(lengths) == 1
        assert set(log) == {"static", "ungated", "gate_traffic_only", "gate_full", "ewc",
                            "gate_quarantine"}


def test_static_never_authorizes_ungated_always_does_when_drift(synth_logs):
    for log in synth_logs:
        assert not any(r["authorized"] for r in log["static"])
        assert all(r["authorized"] == r["drift_active"] for r in log["ungated"])


def test_on_round_callback_streams_every_round():
    stream, names = build_stream("slow_poisoning", "synth", seed=1)
    seen = []
    log = run_scenario(stream, names, "sp", on_round=lambda i, e: seen.append((i, e)))
    assert [i for i, _ in seen] == list(range(1, len(log["static"]) + 1))
    assert seen[-1][1]["gate_full"] == log["gate_full"][-1]


def test_optional_arms_can_be_switched_off():
    stream, names = build_stream("slow_poisoning", "synth", seed=1)
    assert set(run_scenario(stream, names, "sp", include_ewc=False,
                            include_quarantine=False)) == {
        "static", "ungated", "gate_traffic_only", "gate_full"}


def test_seeded_synth_streams_are_reproducible_and_distinct():
    a, _ = build_stream("slow_poisoning", "synth", seed=5)
    b, _ = build_stream("slow_poisoning", "synth", seed=5)
    c, _ = build_stream("slow_poisoning", "synth", seed=6)
    assert a == b and a != c


def test_reputation_noise_flows_through_build_stream():
    clean, _ = build_stream("slow_poisoning", "synth", seed=2)
    noisy, _ = build_stream("slow_poisoning", "synth", seed=2,
                            reputation_noise={"flip_rate": 0.3, "seed": 1})
    assert [x for x, _, _ in clean] == [x for x, _, _ in noisy]
    assert [m["reputation"] for *_, m in clean] != [m["reputation"] for *_, m in noisy]


def test_heavily_corrupted_reputation_breaks_the_veto():
    """The gate is only as good as its independent channel: at a 50% error rate
    the attack window's mean reputation sits near 0.5, above the 0.3 veto."""
    stream, names = build_stream("slow_poisoning", "synth", seed=42,
                                 reputation_noise={"flip_rate": 0.5, "seed": 42})
    log = run_scenario(stream, names, "sp", include_ewc=False)
    assert phase_accuracy(log, "gate_full", "recovery_check") < 1.0


@pytest.mark.parametrize("bad", [("nope", "synth"), ("slow_poisoning", "nope")])
def test_build_stream_rejects_unknown_names(bad):
    with pytest.raises(ValueError):
        build_stream(*bad)


# --- fake N-BaIoT through the real loader ------------------------------------

def _fake(fake_root, scenario):
    if scenario == "slow_poisoning":
        stream = nb.slow_poisoning_stream(root=fake_root)
    else:
        stream = nb.genuine_drift_stream(
            benign_b_csv=os.path.join(fake_root, "Ennio_Doorbell", "benign_traffic.csv"),
            root=fake_root)
    return stream, nb.feature_names(root=fake_root)


def test_fake_data_reproduces_the_poisoning_result(fake_root):
    log = run_scenario(*_fake(fake_root, "slow_poisoning"), "sp")
    assert phase_accuracy(log, "gate_full", "recovery_check") == 1.0
    assert phase_accuracy(log, "gate_traffic_only", "recovery_check") < 0.5


def test_feature_drift_trigger_fixes_scenario1_null_result(fake_root):
    stream, names = _fake(fake_root, "genuine_drift")
    error_only = run_scenario(stream, names, "gd")
    with_feature = run_scenario(stream, names, "gd", feature_drift=True)
    assert not any(r["drift_active"] for r in error_only["static"])      # the null result
    assert any(r["drift_active"] for r in with_feature["static"])
    # benign drift with clean reputation: the full gate does authorize some of it
    assert any(r["authorized"] for r in with_feature["gate_full"])


# --- EWC ---------------------------------------------------------------------

def _trained_ewc(lam):
    rng = np.random.default_rng(0)
    m = EWCOnlineIDS(ewc_lambda=lam)
    xs = [{"a": float(v), "b": float(-v)} for v in rng.normal(0, 1, 300)]
    for x in xs:
        m.learn_one(x, int(x["a"] > 0))
    m.consolidate(xs)
    return m, xs


def test_ewc_fisher_is_normalised():
    m, _ = _trained_ewc(50)
    vals = list(m.fisher.values()) + [m.fisher_intercept]
    assert max(vals) == pytest.approx(1.0)
    assert min(vals) >= 0


@pytest.mark.parametrize("lam,expect_anchored", [(0.0, False), (100.0, True)])
def test_ewc_anchoring(lam, expect_anchored):
    m, xs = _trained_ewc(lam)
    anchor = dict(m.anchor)
    for x in xs:                                     # adversarial relabelling
        m.learn_one(x, int(x["a"] <= 0))
    drift = max(abs(m._lr._weights[k] - w) for k, w in anchor.items())
    assert (drift < 0.05) == expect_anchored
