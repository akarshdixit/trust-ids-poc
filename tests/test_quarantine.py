"""The quarantine extension (gate/quarantine.py): the sequential reputation
test, provisional updates with rollback, and the gate_quarantine arm."""
import json
import math

import numpy as np
import pytest

from experiments.run_scenarios import advance_evidence, run_scenario
from experiments.streams import build_stream
from experiments.sweeps import phase_accuracy
from gate.quarantine import Quarantine, SequentialReputationTest
from models.online_ids import OnlineIDS

ORIGINAL_ARMS = ("static", "ungated", "gate_traffic_only", "gate_full", "ewc")
BENIGN, ATTACK = 1.0, 0.05


# --- sequential reputation test -------------------------------------------------

def test_wald_bounds_and_step():
    t = SequentialReputationTest(feed_error=0.3, alpha=0.01, beta=0.01)
    assert t.step == pytest.approx(math.log(0.7 / 0.3))
    assert t.upper == pytest.approx(math.log(0.99 / 0.01))
    assert t.lower == pytest.approx(math.log(0.01 / 0.99))
    assert t.llr == 0 and not t.condemned


def test_attack_votes_condemn_after_the_wald_bound():
    t = SequentialReputationTest(feed_error=0.3)
    needed = math.ceil(-t.lower / t.step)
    t.update([ATTACK] * (needed - 1))
    assert not t.condemned
    t.update([ATTACK])
    assert t.condemned and t.llr <= t.lower


def test_benign_evidence_is_clamped_so_it_cannot_be_banked():
    t = SequentialReputationTest(feed_error=0.3)
    t.update([BENIGN] * 1000)
    assert t.llr == t.max_net * t.step <= t.upper
    # a long benign prefix (a lagged feed) still lets an attack be condemned quickly
    t.update([ATTACK] * (t.max_net + math.ceil(-t.lower / t.step)))
    assert t.condemned


def test_condemnation_is_sticky_until_reset():
    t = SequentialReputationTest()
    t.update([ATTACK] * 50)
    t.update([BENIGN] * 50)
    assert t.condemned and t.llr <= t.lower
    t.reset()
    assert not t.condemned and t.llr == 0


def test_vote_threshold_is_strictly_above_half():
    t = SequentialReputationTest()
    t.update([0.5])
    assert t.net == -1
    t.update([0.51])
    assert t.net == 0


@pytest.mark.parametrize("kw", [{"feed_error": 0}, {"feed_error": 0.5}, {"alpha": 0},
                                {"beta": 1.0}])
def test_invalid_test_parameters_rejected(kw):
    with pytest.raises(ValueError):
        SequentialReputationTest(**kw)


def test_error_rates_hold_when_the_feed_assumption_holds():
    """Monte Carlo over 400-record episodes at exactly the assumed 30% feed error."""
    rng = np.random.default_rng(0)
    false_condemn, missed = 0, 0
    for _ in range(500):
        benign, attack = SequentialReputationTest(0.3), SequentialReputationTest(0.3)
        benign.update(np.where(rng.random(400) < 0.3, ATTACK, BENIGN))
        attack.update(np.where(rng.random(400) < 0.3, BENIGN, ATTACK))
        false_condemn += benign.condemned
        missed += not attack.condemned
    assert false_condemn / 500 <= 0.02
    assert missed == 0


# --- provisional updates and rollback -------------------------------------------

def _x(a, b):
    return {"f0": a, "f1": b}


CALIB = [(_x(0.0, 0.1 * i), 0) for i in range(10)] + [(_x(5.0, 0.1 * i), 1) for i in range(10)]
BATCH_A = [(_x(4.0, 0.2), 0)] * 10
BATCH_B = [(_x(1.0, 1.0), 0)] * 10
BATCH_C = [(_x(4.5, 0.5), 0)] * 10
PROBES = [_x(a, b) for a in (0.0, 2.0, 4.0, 5.0) for b in (0.0, 1.0)]


def _model(*batches):
    ids = OnlineIDS(drift_detector=False)
    ids.learn_batch(CALIB)
    for b in batches:
        ids.learn_batch(b)
    return ids


def _probs(ids):
    return [ids.model.predict_proba_one(p).get(1, 0.0) for p in PROBES]


def _provisional(q, ids, state, round_idx, episode, batch, mean):
    q.add(round_idx, episode, batch, mean, ids, state)
    ids.learn_batch(batch)
    return advance_evidence(state, mean)


def test_rollback_reverts_one_episode_and_replays_the_others():
    ids, q = _model(), Quarantine(quarantine_rounds=5)
    s0 = (np.zeros(2), None)
    mean_a, mean_b, mean_c = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 2.0])
    s = _provisional(q, ids, s0, 1, 1, BATCH_A, mean_a)
    s = _provisional(q, ids, s, 2, 2, BATCH_B, mean_b)
    s = _provisional(q, ids, s, 3, 1, BATCH_C, mean_c)

    restored = q.rollback(1, ids, advance_evidence)

    assert _probs(ids) == pytest.approx(_probs(_model(BATCH_B)))
    assert q.n_rolled_back == 2 and [u.episode for u in q.pending] == [2]
    expected = advance_evidence(s0, mean_b)
    assert np.allclose(restored[0], expected[0]) and np.allclose(restored[1], expected[1])
    # the survivor's checkpoint is rewritten, so a later rollback of it is still exact
    q.rollback(2, ids, advance_evidence)
    assert _probs(ids) == pytest.approx(_probs(_model()))


def test_rollback_of_an_episode_with_nothing_pending_is_a_no_op():
    ids, q = _model(BATCH_A), Quarantine()
    before = _probs(ids)
    assert q.rollback(7, ids, advance_evidence) is None
    assert _probs(ids) == before


def test_updates_commit_after_the_quarantine_period():
    ids, q = _model(), Quarantine(quarantine_rounds=5)
    _provisional(q, ids, (np.zeros(2), None), 10, 1, BATCH_A, np.ones(2))
    q.commit_due(14)
    assert len(q.pending) == 1
    q.commit_due(15)
    assert q.pending == [] and q.n_committed == 1
    assert q.rollback(1, ids, advance_evidence) is None       # committed is final


def test_checkpoint_is_a_copy_not_a_reference():
    ids, q = _model(), Quarantine()
    q.add(1, 1, BATCH_A, np.ones(2), ids, (np.zeros(2), None))
    ids.learn_batch(BATCH_A)
    assert q.pending[0].checkpoint is not ids.model
    assert _probs(_model()) == pytest.approx(
        [q.pending[0].checkpoint.predict_proba_one(p).get(1, 0.0) for p in PROBES])


# --- the gate_quarantine arm in the pipeline ------------------------------------

NOISY = {"sigma": 0.05, "flip_rate": 0.3, "seed": 42}


@pytest.fixture(scope="module")
def noisy_synth():
    stream, names = build_stream("slow_poisoning", "synth", seed=42, reputation_noise=NOISY)
    return stream, names, run_scenario(stream, names, "sp")


def test_quarantine_arm_does_not_change_any_other_arm(noisy_synth):
    stream, names, with_q = noisy_synth
    without = run_scenario(stream, names, "sp", include_quarantine=False)
    for arm in ORIGINAL_ARMS:
        assert json.dumps(with_q[arm]) == json.dumps(without[arm]), arm


def test_quarantine_survives_a_feed_the_one_shot_veto_does_not(noisy_synth):
    log = noisy_synth[2]
    assert phase_accuracy(log, "gate_full", "recovery_check") < 0.5
    assert phase_accuracy(log, "gate_quarantine", "recovery_check") == 1.0


def test_quarantine_log_is_consistent(noisy_synth):
    q = noisy_synth[2]["gate_quarantine"]
    assert len(q) == len(noisy_synth[2]["static"])
    assert not any(r["authorized"] for r in q if r["condemned"] or not r["drift_active"])
    assert not any(r["authorized"] for r in q if r["sprt_llr"] <= 0)
    provisional = np.cumsum([r["authorized"] for r in q])
    rolled = np.cumsum([r["rolled_back"] for r in q])
    assert (rolled <= provisional).all()


def test_quarantine_still_adapts_to_benign_drift():
    stream, names = build_stream("genuine_drift", "synth", seed=42)
    log = run_scenario(stream, names, "gd", include_ewc=False)
    assert phase_accuracy(log, "gate_quarantine", "post_drift") >= 0.95
    assert phase_accuracy(log, "static", "post_drift") < 0.5


def test_no_rollback_when_quarantine_period_is_zero():
    stream, names = build_stream("slow_poisoning", "synth", seed=42,
                                 reputation_noise={"sigma": 0.05, "flip_rate": 0.4, "seed": 42})
    log = run_scenario(stream, names, "sp", include_ewc=False,
                       quarantine_params={"quarantine_rounds": 0})
    assert not any(r["rolled_back"] for r in log["gate_quarantine"])


def test_quarantine_params_override_defaults():
    stream, names = build_stream("genuine_drift", "synth", seed=42)
    strict = run_scenario(stream, names, "gd", include_ewc=False,
                          quarantine_params={"provisional_threshold": 1.0})
    assert not any(r["authorized"] for r in strict["gate_quarantine"])


def test_rollback_removes_poison_a_late_feed_let_in():
    """A feed 600 records late reports benign while the attack starts. Without a
    rollback the provisional poisoned updates stay; with the default hold they go."""
    from experiments.sweeps import poisoned_updates_kept
    stream, names = build_stream("slow_poisoning", "synth", seed=42,
                                 reputation_noise={"lag": 600, "seed": 42})
    kept = {qr: poisoned_updates_kept(
        run_scenario(stream, names, "sp", include_ewc=False,
                     quarantine_params={"quarantine_rounds": qr}), "gate_quarantine", stream)
        for qr in (0, 20)}
    assert kept[0] > 0 and kept[20] == 0
