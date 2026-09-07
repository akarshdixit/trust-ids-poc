"""
Common orchestrator — every arm runs through the identical pipeline so the
comparison is a controlled experiment, exactly as specified in the
Review-1 architecture ("every baseline runs through one orchestrator").

Pipeline per round:
  traffic window -> prequential predict/score (every arm)
                  -> shared ADWIN drift monitor (fed by the STATIC arm's
                     error rate, since it's frozen and so gives a stable,
                     unbiased read on how far the live window has moved
                     from the calibration distribution)
                  -> IF drift was flagged this round: propose an update
                     ("teach the model this window is normal"), then each
                     gated arm's evidence + gate decide whether to
                     authorize it.
                  -> IF no drift flagged: nothing is proposed, nobody learns
                     this round (steady state doesn't need adaptation).

This is the fix for the "ADWIN runs but isn't used for authorization"
critique — drift detection is now the actual trigger for the rest of the
pipeline, for every arm, not a logged-but-unused side computation.

Arms:
  static      - baseline IDS trained once on calibration data, NEVER updates.
  ungated     - authorizes every drift-triggered proposal unconditionally.
  gate_traffic_only - traffic evidence only (attribution+topology+temporal),
                no out-of-band veto. Stands in for ExpIDS-style gating.
  gate_full   - THE PROPOSED MECHANISM: traffic evidence + reputation veto.

Do not read the numbers this script prints as Review-2 RESULTS until it has
been re-run against the real N-BaIoT stream (data/load_nbaiot.py) — on the
synthetic stream, it only proves the pipeline plumbing is correct.
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synth_stream import genuine_drift_stream, slow_poisoning_stream, FEATURE_NAMES
from evidence.extract import attribution_consistency, topology_diversity, temporal_stability
from gate.trust_gate import TrustGate
from models.online_ids import OnlineIDS

ROUND_SIZE = 20
N_CALIB = 400
REFERENCE_EMA = 0.3  # weight on the new shift when updating the trusted drift direction


def vec(x_dict):
    return np.array([x_dict[k] for k in FEATURE_NAMES])


def run_scenario(stream, scenario_name):
    calib = stream[:N_CALIB]
    rest = stream[N_CALIB:]

    arms = {
        "static": OnlineIDS(),
        "ungated": OnlineIDS(),
        "gate_traffic_only": OnlineIDS(),
        "gate_full": OnlineIDS(),
    }
    gates = {
        "gate_traffic_only": TrustGate(traffic_only=True),
        "gate_full": TrustGate(traffic_only=False),
    }

    # --- calibration: every arm learns the same labelled bootstrap set ---
    for x, y, meta in calib:
        for arm in arms.values():
            arm.learn_one(x, y)

    calib_benign_vectors = np.array([vec(x) for x, y, m in calib if y == 0])
    # trusted_baseline: each gated arm's "last known trusted operating point"
    trusted_baseline = {name: calib_benign_vectors.mean(axis=0) for name in gates}
    # reference_direction: EMA of past AUTHORIZED shift vectors; None until
    # the first authorized drift, so attribution has nothing to fake yet
    reference_direction = {name: None for name in gates}
    prev_window_mean = calib_benign_vectors.mean(axis=0)

    log = {name: [] for name in arms}
    round_idx = 0
    DRIFT_COOLDOWN = 15  # rounds an ADWIN-flagged change point stays "active"
    cooldown_remaining = 0

    for i in range(0, len(rest) - ROUND_SIZE + 1, ROUND_SIZE):
        round_records = rest[i:i + ROUND_SIZE]
        round_idx += 1
        true_ys = [y for x, y, meta in round_records]

        # --- prequential predict + score, every arm, every sample ---
        round_acc = {}
        change_point_this_round = False
        for name, arm in arms.items():
            preds = []
            for x, y, meta in round_records:
                p = arm.score_one(x, y)
                preds.append(p)
                if name == "static" and arm.drift_detected():
                    change_point_this_round = True
            round_acc[name] = float(np.mean([p == t for p, t in zip(preds, true_ys)]))

        if change_point_this_round:
            cooldown_remaining = DRIFT_COOLDOWN
        drift_active = cooldown_remaining > 0
        cooldown_remaining = max(0, cooldown_remaining - 1)

        window_vectors = np.array([vec(x) for x, y, m in round_records])
        reputation = float(np.mean([m["reputation"] for x, y, m in round_records]))

        if not drift_active:
            # steady state (or cooldown expired): no update proposed to anyone
            for name, arm in arms.items():
                log[name].append({
                    "round": round_idx, "authorized": False, "drift_active": False,
                    "accuracy": arm.metric.get(), "f1": arm.f1.get(),
                    "round_accuracy": round_acc[name],
                })
            prev_window_mean = window_vectors.mean(axis=0)
            continue

        # drift was signalled -> propose "treat this window as normal"
        proposed_update = [(x, 0) for x, y, meta in round_records]

        for gate_name, gate in gates.items():
            attribution = attribution_consistency(
                window_vectors, trusted_baseline[gate_name], reference_direction[gate_name]
            )
            topology = topology_diversity(window_vectors)
            temporal = temporal_stability(window_vectors, prev_window_mean)
            decision = gate.decide(attribution, topology, temporal, reputation)

            arm = arms[gate_name]
            if decision.authorized:
                arm.learn_batch(proposed_update)
                shift = window_vectors.mean(axis=0) - trusted_baseline[gate_name]
                if reference_direction[gate_name] is None:
                    reference_direction[gate_name] = shift
                else:
                    reference_direction[gate_name] = (
                        (1 - REFERENCE_EMA) * reference_direction[gate_name] + REFERENCE_EMA * shift
                    )
                trusted_baseline[gate_name] = window_vectors.mean(axis=0)

            log[gate_name].append({
                "round": round_idx, "authorized": decision.authorized, "drift_active": True,
                "trust_score": decision.trust_score, "reputation": reputation,
                "attribution": attribution, "topology": topology, "temporal": temporal,
                "reason": decision.reason,
                "accuracy": arm.metric.get(), "f1": arm.f1.get(),
                "round_accuracy": round_acc[gate_name],
            })

        # ungated: drift was signalled -> always authorizes it
        arms["ungated"].learn_batch(proposed_update)
        log["ungated"].append({
            "round": round_idx, "authorized": True, "drift_active": True,
            "accuracy": arms["ungated"].metric.get(), "f1": arms["ungated"].f1.get(),
            "round_accuracy": round_acc["ungated"],
        })

        # static: structurally never authorized
        log["static"].append({
            "round": round_idx, "authorized": False, "drift_active": True,
            "accuracy": arms["static"].metric.get(), "f1": arms["static"].f1.get(),
            "round_accuracy": round_acc["static"],
        })

        prev_window_mean = window_vectors.mean(axis=0)

    return log


def summarize(log, scenario_name):
    print(f"\n=== {scenario_name} — final prequential accuracy / F1 (attack=positive) ===")
    for arm_name, rounds in log.items():
        final = rounds[-1]
        n_auth = sum(1 for r in rounds if r.get("authorized"))
        n_drift = sum(1 for r in rounds if r.get("drift_active"))
        print(f"  {arm_name:20s} acc={final['accuracy']:.3f}  f1={final['f1']:.3f}  "
              f"rounds_authorized={n_auth}/{len(rounds)}  drift_active_rounds={n_drift}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    drift_log = run_scenario(genuine_drift_stream(), "Scenario 1: Genuine Drift")
    poison_log = run_scenario(slow_poisoning_stream(), "Scenario 2: Slow Poisoning")

    summarize(drift_log, "Scenario 1: Genuine Drift")
    summarize(poison_log, "Scenario 2: Slow Poisoning")

    with open(os.path.join(results_dir, "genuine_drift_log.json"), "w") as f:
        json.dump(drift_log, f, indent=2)
    with open(os.path.join(results_dir, "slow_poisoning_log.json"), "w") as f:
        json.dump(poison_log, f, indent=2)

    print(f"\nLogs written to {results_dir}/")
