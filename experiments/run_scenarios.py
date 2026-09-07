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

Drift detection is the actual trigger for the rest of the pipeline, for
every arm — not a logged-but-unused side computation.

Arms:
  static      - baseline IDS trained once on calibration data, NEVER updates.
  ungated     - authorizes every drift-triggered proposal unconditionally.
  gate_traffic_only - traffic evidence only (attribution+topology+temporal),
                no out-of-band veto. Stands in for ExpIDS-style gating.
  gate_full   - THE PROPOSED MECHANISM: traffic evidence + reputation veto.

EVIDENCE IS COMPUTED IN A CALIBRATED FEATURE SPACE (evidence/featurespace.py),
fit on calibration benign data only and then frozen. On raw N-BaIoT values the
geometric evidence signals collapse — temporal_stability returns 1.7e-17 and
five HH_jit_*_variance columns dominate 100% of every cosine — so nothing is
ever authorized and all four arms degenerate to `static`. See that module's
docstring for the measured numbers.

Usage:
  python experiments/run_scenarios.py            # REAL N-BaIoT (Review-2)
  python experiments/run_scenarios.py --synth    # synthetic regression check
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence.extract import attribution_consistency, topology_diversity, temporal_stability
from evidence.featurespace import FeatureSpace
from gate.trust_gate import TrustGate
from models.online_ids import OnlineIDS

ROUND_SIZE = 20
N_CALIB = 400
REFERENCE_EMA = 0.3  # weight on the new shift when updating the trusted drift direction
DRIFT_COOLDOWN = 15  # rounds an ADWIN-flagged change point stays "active"


def run_scenario(stream, feature_names, scenario_name, n_calib=N_CALIB, gate_params=None):
    names = list(feature_names)

    def vec(x_dict):
        return np.array([x_dict[k] for k in names], dtype=float)

    calib = stream[:n_calib]
    rest = stream[n_calib:]

    arms = {
        "static": OnlineIDS(),
        "ungated": OnlineIDS(),
        "gate_traffic_only": OnlineIDS(),
        "gate_full": OnlineIDS(),
    }
    gp = dict(gate_params or {})
    gates = {
        "gate_traffic_only": TrustGate(traffic_only=True, **gp),
        "gate_full": TrustGate(traffic_only=False, **gp),
    }

    # --- calibration: every arm learns the same labelled bootstrap set ---
    for x, y, meta in calib:
        for arm in arms.values():
            arm.learn_one(x, y)

    calib_benign_raw = np.array([vec(x) for x, y, m in calib if y == 0])

    # Trusted frame of reference for ALL evidence, frozen here. Fitting this on
    # live windows would hand the attacker control of the reference frame — the
    # exact influence the gate exists to deny.
    fs = FeatureSpace().fit(calib_benign_raw)
    calib_benign_z = fs.transform(calib_benign_raw)
    baseline_z = calib_benign_z.mean(axis=0)

    trusted_baseline = {name: baseline_z.copy() for name in gates}
    # reference_direction: EMA of past AUTHORIZED shift vectors; None until the
    # first authorized drift, so attribution has nothing to fake yet
    reference_direction = {name: None for name in gates}
    prev_window_mean = baseline_z.copy()

    log = {name: [] for name in arms}
    round_idx = 0
    cooldown_remaining = 0

    for i in range(0, len(rest) - ROUND_SIZE + 1, ROUND_SIZE):
        round_records = rest[i:i + ROUND_SIZE]
        round_idx += 1
        true_ys = [y for x, y, meta in round_records]
        phase = round_records[0][2].get("source", "?")

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

        window_raw = np.array([vec(x) for x, y, m in round_records])
        window_z = fs.transform(window_raw)
        window_mean_z = window_z.mean(axis=0)
        reputation = float(np.mean([m["reputation"] for x, y, m in round_records]))

        if not drift_active:
            # steady state (or cooldown expired): no update proposed to anyone
            for name, arm in arms.items():
                log[name].append({
                    "round": round_idx, "authorized": False, "drift_active": False,
                    "phase": phase, "reputation": reputation,
                    "accuracy": arm.metric.get(), "f1": arm.f1.get(),
                    "round_accuracy": round_acc[name],
                })
            prev_window_mean = window_mean_z
            continue

        # drift was signalled -> propose "treat this window as normal"
        proposed_update = [(x, 0) for x, y, meta in round_records]

        for gate_name, gate in gates.items():
            attribution = attribution_consistency(
                window_z, trusted_baseline[gate_name], reference_direction[gate_name]
            )
            topology = topology_diversity(window_z)
            temporal = temporal_stability(window_z, prev_window_mean)
            decision = gate.decide(attribution, topology, temporal, reputation)

            arm = arms[gate_name]
            if decision.authorized:
                arm.learn_batch(proposed_update)
                shift = window_mean_z - trusted_baseline[gate_name]
                if reference_direction[gate_name] is None:
                    reference_direction[gate_name] = shift
                else:
                    reference_direction[gate_name] = (
                        (1 - REFERENCE_EMA) * reference_direction[gate_name]
                        + REFERENCE_EMA * shift
                    )
                trusted_baseline[gate_name] = window_mean_z.copy()

            log[gate_name].append({
                "round": round_idx, "authorized": decision.authorized, "drift_active": True,
                "phase": phase,
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
            "phase": phase, "reputation": reputation,
            "accuracy": arms["ungated"].metric.get(), "f1": arms["ungated"].f1.get(),
            "round_accuracy": round_acc["ungated"],
        })

        # static: structurally never authorized
        log["static"].append({
            "round": round_idx, "authorized": False, "drift_active": True,
            "phase": phase, "reputation": reputation,
            "accuracy": arms["static"].metric.get(), "f1": arms["static"].f1.get(),
            "round_accuracy": round_acc["static"],
        })

        prev_window_mean = window_mean_z

    return log


def summarize(log, scenario_name, phase_of_interest=None):
    print(f"\n=== {scenario_name} — final prequential accuracy / F1 (attack=positive) ===")
    for arm_name, rounds in log.items():
        final = rounds[-1]
        n_auth = sum(1 for r in rounds if r.get("authorized"))
        n_drift = sum(1 for r in rounds if r.get("drift_active"))
        print(f"  {arm_name:20s} acc={final['accuracy']:.3f}  f1={final['f1']:.3f}  "
              f"rounds_authorized={n_auth}/{len(rounds)}  drift_active_rounds={n_drift}")

    if phase_of_interest:
        print(f"  -- windowed accuracy during '{phase_of_interest}' phase --")
        for arm_name, rounds in log.items():
            sel = [r["round_accuracy"] for r in rounds if r.get("phase") == phase_of_interest]
            if sel:
                print(f"     {arm_name:20s} mean={np.mean(sel):.3f}  n_rounds={len(sel)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true",
                    help="run the synthetic stream instead of real N-BaIoT")
    ap.add_argument("--benign-b", default=None,
                    help="second device's benign_traffic.csv, for Scenario 1")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = "synth" if args.synth else "real"

    if args.synth:
        from data.synth_stream import (genuine_drift_stream, slow_poisoning_stream,
                                       FEATURE_NAMES)
        names = FEATURE_NAMES
        drift_kwargs = {}
    else:
        from data.load_nbaiot import (genuine_drift_stream, slow_poisoning_stream,
                                      feature_names)
        names = feature_names()
        drift_kwargs = {"benign_b_csv": args.benign_b}
        print(f"Real N-BaIoT stream — {len(names)} features")

    print("\nBuilding Scenario 2 (slow poisoning)...")
    poison_log = run_scenario(slow_poisoning_stream(), names, "Scenario 2: Slow Poisoning")
    summarize(poison_log, "Scenario 2: Slow Poisoning", phase_of_interest="recovery_check")
    with open(os.path.join(results_dir, f"slow_poisoning_log_{tag}.json"), "w") as f:
        json.dump(poison_log, f, indent=2)

    print("\nBuilding Scenario 1 (genuine drift)...")
    try:
        drift_log = run_scenario(genuine_drift_stream(**drift_kwargs), names,
                                 "Scenario 1: Genuine Drift")
        summarize(drift_log, "Scenario 1: Genuine Drift", phase_of_interest="post_drift")
        with open(os.path.join(results_dir, f"genuine_drift_log_{tag}.json"), "w") as f:
            json.dump(drift_log, f, indent=2)
    except ValueError as e:
        print(f"  SKIPPED — {e}")

    print(f"\nLogs written to {results_dir}/")


if __name__ == "__main__":
    main()
