"""
Traffic-derived evidence extraction — UNTRUSTED / attacker-influenceable.

These three signals are computed purely from the traffic window being
adapted to, exactly as flagged in the Review-1 architecture. A patient
attacker can shape traffic to make any single one of these look benign;
that is the whole reason the gate also needs the out-of-band signal
(evidence/../gate/trust_gate.py combines them and lets reputation veto).

All three return a score in [0, 1], higher = "looks more trustworthy".
"""
import numpy as np


def attribution_consistency(window_vectors, baseline_vector, reference_direction):
    """
    Does this window's DRIFT DIRECTION (shift away from the last trusted
    baseline) resemble a direction of drift we've already seen authorized
    as genuine (reference_direction, an EMA of past authorized shifts)?

    reference_direction may be None (no authorized drift observed yet) —
    in that case there is nothing to compare against, so we return a
    neutral 0.5 rather than a fabricated score.
    """
    window_mean = np.mean(window_vectors, axis=0)
    shift = window_mean - baseline_vector

    if reference_direction is None:
        return 0.5

    denom = (np.linalg.norm(shift) * np.linalg.norm(reference_direction)) + 1e-9
    cos_sim = float(np.dot(shift, reference_direction) / denom)
    # map cosine similarity [-1, 1] -> [0, 1]
    return max(0.0, min(1.0, (cos_sim + 1) / 2))


def topology_diversity(window_vectors, feature_slice=slice(0, 5)):
    """
    Proxy for "communication topology" when raw IP/port data isn't
    available (as in N-BaIoT). Organic benign IoT usage tends to touch a
    more varied set of behavioural regimes than a narrow, repeated C2
    channel. We approximate this with the normalised entropy of a binned
    subset of features across the window: higher entropy -> more diverse
    -> more consistent with genuine benign behaviour.
    """
    sub = np.asarray(window_vectors)[:, feature_slice]
    bins = np.linspace(sub.min(), sub.max(), 10) if sub.max() > sub.min() else [0, 1]
    hist, _ = np.histogram(sub.flatten(), bins=bins)
    probs = hist / (hist.sum() + 1e-9)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(bins) - 1) if len(bins) > 1 else 1.0
    return float(entropy / (max_entropy + 1e-9))


def temporal_stability(window_vectors, prev_window_mean):
    """
    How smooth/monotonic is the window-to-window change?
    NOTE: this is the signal a stealthy, patient attacker can maximise on
    purpose (that's the point of "slow" poisoning) — so a high stability
    score here is NOT sufficient evidence of trustworthiness on its own.
    It's kept in the evidence vector to demonstrate exactly that limitation
    in the Review-2 demo: traffic-only gating (stability + attribution +
    topology, no reputation) is fooled by slow poisoning; adding the
    out-of-band veto is what catches it.
    """
    window_mean = np.mean(window_vectors, axis=0)
    delta = np.linalg.norm(window_mean - prev_window_mean)
    # smaller delta -> smoother change -> higher "stability" score
    return float(1.0 / (1.0 + delta))
