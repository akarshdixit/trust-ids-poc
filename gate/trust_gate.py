"""
Evidence Authorization Gate — TRUSTED decision boundary.

Combines the three UNTRUSTED traffic-derived evidence scores into a trust
score, then cross-checks against the out-of-band reputation signal, which
can VETO authorization regardless of how good the traffic evidence looks.
This veto is the mechanism's actual research contribution: it is what a
traffic-only gate (e.g. ExpIDS-style drift-magnitude gating) cannot do.
"""
from dataclasses import dataclass


@dataclass
class GateDecision:
    authorized: bool
    trust_score: float
    attribution: float
    topology: float
    temporal: float
    reputation: float
    reason: str


class TrustGate:
    def __init__(self, w_attribution=0.4, w_topology=0.3, w_temporal=0.3,
                 trust_threshold=0.55, reputation_veto_threshold=0.3,
                 traffic_only=False):
        """
        traffic_only=True reproduces an ExpIDS-style / ungated-evidence
        baseline that has NO out-of-band veto — used to show, in the
        Review-2 demo, that traffic-only evidence alone is fooled by slow
        poisoning (attribution + temporal stability can both look fine).
        """
        self.w_attribution = w_attribution
        self.w_topology = w_topology
        self.w_temporal = w_temporal
        self.trust_threshold = trust_threshold
        self.reputation_veto_threshold = reputation_veto_threshold
        self.traffic_only = traffic_only

    def decide(self, attribution, topology, temporal, reputation) -> GateDecision:
        trust_score = (
            self.w_attribution * attribution
            + self.w_topology * topology
            + self.w_temporal * temporal
        )

        if not self.traffic_only and reputation < self.reputation_veto_threshold:
            return GateDecision(
                authorized=False, trust_score=trust_score,
                attribution=attribution, topology=topology, temporal=temporal,
                reputation=reputation,
                reason=f"out-of-band reputation veto ({reputation:.2f} < "
                       f"{self.reputation_veto_threshold})",
            )

        authorized = trust_score >= self.trust_threshold
        reason = (
            f"trust_score {trust_score:.2f} "
            f"{'>=' if authorized else '<'} threshold {self.trust_threshold}"
        )
        return GateDecision(
            authorized=authorized, trust_score=trust_score,
            attribution=attribution, topology=topology, temporal=temporal,
            reputation=reputation, reason=reason,
        )
