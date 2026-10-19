import numpy as np
import pytest

from data.reputation import (ATTACK_REPUTATION, BENIGN_REPUTATION, ReputationNoise,
                             apply_reputation_noise)


def _stream(n=2000, attack_from=1000):
    out = []
    for i in range(n):
        attack = i >= attack_from
        rep = ATTACK_REPUTATION if attack else BENIGN_REPUTATION
        out.append(({"f": float(i)}, int(attack), {"reputation": rep, "source": "x"}))
    return out


def _reps(stream):
    return np.array([m["reputation"] for _, _, m in stream])


def test_oracle_is_identity():
    s = _stream()
    assert apply_reputation_noise(s, ReputationNoise()) == s


def test_input_not_mutated_and_only_reputation_changes():
    s = _stream()
    before = [(dict(x), y, dict(m)) for x, y, m in s]
    out = apply_reputation_noise(s, ReputationNoise(sigma=0.2, flip_rate=0.3, lag=5))
    assert s == before
    for (x0, y0, m0), (x1, y1, m1) in zip(s, out):
        assert x0 == x1 and y0 == y1 and m0["source"] == m1["source"]
        assert m1["reputation_true"] == m0["reputation"]


def test_values_stay_in_unit_interval():
    r = _reps(apply_reputation_noise(_stream(), ReputationNoise(sigma=0.5, seed=1)))
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_flip_rate_is_respected():
    out = apply_reputation_noise(_stream(), ReputationNoise(flip_rate=0.25, seed=3))
    flipped = np.mean([m["reputation"] != m["reputation_true"] for _, _, m in out])
    assert flipped == pytest.approx(0.25, abs=0.03)
    # a flip swaps to the OTHER class value, nothing in between
    assert set(np.round(_reps(out), 6)) <= {BENIGN_REPUTATION, ATTACK_REPUTATION}


def test_lag_delays_the_feed():
    out = apply_reputation_noise(_stream(attack_from=1000), ReputationNoise(lag=40))
    r = _reps(out)
    assert np.all(r[1000:1040] == BENIGN_REPUTATION)   # feed hasn't caught up yet
    assert np.all(r[1040:] == ATTACK_REPUTATION)


def test_deterministic_for_a_seed():
    noise = ReputationNoise(sigma=0.1, flip_rate=0.1, seed=9)
    assert apply_reputation_noise(_stream(), noise) == apply_reputation_noise(_stream(), noise)


@pytest.mark.parametrize("kw", [{"sigma": -1}, {"flip_rate": 1.5}, {"lag": -2}])
def test_invalid_noise_rejected(kw):
    with pytest.raises(ValueError):
        ReputationNoise(**kw)
