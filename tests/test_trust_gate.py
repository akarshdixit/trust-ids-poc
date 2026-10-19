import pytest

from gate.trust_gate import TrustGate

GOOD = dict(attribution=0.9, topology=0.9, temporal=0.9)   # trust 0.9
WEAK = dict(attribution=0.1, topology=0.1, temporal=0.1)   # trust 0.1


def test_authorizes_good_evidence_with_clean_reputation():
    d = TrustGate().decide(**GOOD, reputation=1.0)
    assert d.authorized
    assert d.trust_score == pytest.approx(0.9)


def test_refuses_weak_evidence():
    d = TrustGate().decide(**WEAK, reputation=1.0)
    assert not d.authorized
    assert "threshold" in d.reason


def test_reputation_veto_overrides_perfect_evidence():
    d = TrustGate().decide(attribution=1, topology=1, temporal=1, reputation=0.05)
    assert not d.authorized
    assert "veto" in d.reason
    assert d.trust_score == pytest.approx(1.0)  # evidence still reported


def test_traffic_only_gate_ignores_reputation():
    d = TrustGate(traffic_only=True).decide(**GOOD, reputation=0.0)
    assert d.authorized


def test_veto_threshold_is_strict_less_than():
    gate = TrustGate(reputation_veto_threshold=0.3)
    assert gate.decide(**GOOD, reputation=0.3).authorized
    assert not gate.decide(**GOOD, reputation=0.2999).authorized


def test_threshold_boundary_is_inclusive():
    gate = TrustGate(trust_threshold=0.5)
    assert gate.decide(attribution=0.5, topology=0.5, temporal=0.5, reputation=1).authorized


def test_default_weights_are_left_bit_identical():
    g = TrustGate()
    assert (g.w_attribution, g.w_topology, g.w_temporal) == (0.4, 0.3, 0.3)


def test_unnormalised_weights_are_rescaled():
    g = TrustGate(w_attribution=2, w_topology=1, w_temporal=1)
    assert g.w_attribution + g.w_topology + g.w_temporal == pytest.approx(1.0)
    assert g.w_attribution == pytest.approx(0.5)
    d = g.decide(attribution=1, topology=1, temporal=1, reputation=1)
    assert d.trust_score == pytest.approx(1.0)  # stays on the [0, 1] scale


@pytest.mark.parametrize("weights", [(0, 0, 0), (-0.1, 0.6, 0.5)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        TrustGate(*weights)
