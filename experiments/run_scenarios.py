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
  ewc         - authorizes every proposal like `ungated`, but with Elastic
                Weight Consolidation anchoring it to the calibration weights.
                Answers "is the gate's benefit just not-forgetting?".
  gate_quarantine - the extension (gate/quarantine.py): updates that pass the
                traffic evidence are applied provisionally, a sequential test
                keeps weighing the out-of-band channel over the drift episode,
                and a condemned episode's provisional updates are rolled back.

Optional second trigger (feature_drift=True): an ADWIN on each record's
standardised distance from the calibration-benign centre. The error-rate
monitor alone cannot see a benign-to-benign shift the frozen model still
classifies correctly, which is the Scenario-1 null result on real data.

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
from gate.quarantine import Quarantine, SequentialReputationTest
from gate.trust_gate import TrustGate
from models.online_ids import EWCOnlineIDS, OnlineIDS
from river import drift

ROUND_SIZE = 20
N_CALIB = 400
REFERENCE_EMA = 0.3  # weight on the new shift when updating the trusted drift direction
DRIFT_COOLDOWN = 15  # rounds an ADWIN-flagged change point stays "active"
QUARANTINE_DEFAULTS = {
    "provisional_threshold": 0.45,  # one-shot gate's 0.55 minus 0.1: errors are now reversible
    "quarantine_rounds": 20,        # 400 records: must outlast the feed's worst-case lag
    "feed_error": 0.3,              # assumed worst-case fraction of wrong reputation records
    "alpha": 0.01, "beta": 0.01,
}


def advance_evidence(state, window_mean):
    """Evidence bookkeeping after an authorized update: (baseline, reference_direction)."""
    baseline, reference = state
    shift = window_mean - baseline
    if reference is None:
        reference = shift
    else:
        reference = (1 - REFERENCE_EMA) * reference + REFERENCE_EMA * shift
    return window_mean.copy(), reference


def run_scenario(stream, feature_names, scenario_name, n_calib=N_CALIB, gate_params=None,
                 on_round=None, include_ewc=True, ewc_lambda=50.0, feature_drift=False,
                 include_quarantine=True, quarantine_params=None):
    """on_round(round_idx, {arm: log_entry}) is called after every round, if given."""
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
    if include_ewc:
        arms["ewc"] = EWCOnlineIDS(ewc_lambda=ewc_lambda)
    gp = dict(gate_params or {})
    gates = {
        "gate_traffic_only": TrustGate(traffic_only=True, **gp),
        "gate_full": TrustGate(traffic_only=False, **gp),
    }
    if include_quarantine:
        qp = {**QUARANTINE_DEFAULTS, **(quarantine_params or {})}
        arms["gate_quarantine"] = OnlineIDS()
        q_gate = TrustGate(traffic_only=True,
                           **{**gp, "trust_threshold": qp["provisional_threshold"]})
        q_test = SequentialReputationTest(qp["feed_error"], qp["alpha"], qp["beta"])
        quarantine = Quarantine(qp["quarantine_rounds"])

    # --- calibration: every arm learns the same labelled bootstrap set ---
    for x, y, meta in calib:
        for arm in arms.values():
            arm.learn_one(x, y)
    if include_ewc:
        arms["ewc"].consolidate([x for x, y, m in calib])

    calib_benign_raw = np.array([vec(x) for x, y, m in calib if y == 0])

    # Trusted frame of reference for ALL evidence, frozen here. Fitting this on
    # live windows would hand the attacker control of the reference frame — the
    # exact influence the gate exists to deny.
    fs = FeatureSpace().fit(calib_benign_raw)
    calib_benign_z = fs.transform(calib_benign_raw)
    baseline_z = calib_benign_z.mean(axis=0)
    feature_adwin = drift.ADWIN() if feature_drift else None
    calib_centre = baseline_z.copy()
    n_dims = np.sqrt(len(names))

    trusted_baseline = {name: baseline_z.copy() for name in gates}
    # reference_direction: EMA of past AUTHORIZED shift vectors; None until the
    # first authorized drift, so attribution has nothing to fake yet
    reference_direction = {name: None for name in gates}
    prev_window_mean = baseline_z.copy()
    q_state = (baseline_z.copy(), None)  # gate_quarantine's (baseline, reference_direction)

    log = {name: [] for name in arms}
    round_idx = 0
    cooldown_remaining = 0
    episode = 0  # drift episodes: maximal runs of drift-active rounds
    was_active = False

    for i in range(0, len(rest) - ROUND_SIZE + 1, ROUND_SIZE):
        round_records = rest[i:i + ROUND_SIZE]
        round_idx += 1
        true_ys = [y for x, y, meta in round_records]
        phase = round_records[0][2].get("source", "?")

        window_raw = np.array([vec(x) for x, y, m in round_records])
        window_z = fs.transform(window_raw)
        window_mean_z = window_z.mean(axis=0)
        record_reps = [m["reputation"] for x, y, m in round_records]
        reputation = float(np.mean(record_reps))

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

        if feature_adwin is not None:
            for d in np.linalg.norm(window_z - calib_centre, axis=1) / n_dims:
                feature_adwin.update(float(d))
                change_point_this_round = change_point_this_round or feature_adwin.drift_detected

        if change_point_this_round:
            cooldown_remaining = DRIFT_COOLDOWN
        drift_active = cooldown_remaining > 0
        cooldown_remaining = max(0, cooldown_remaining - 1)
        # a new drift run starts a new episode (fresh evidence) only once nothing is
        # held: otherwise held updates would escape the evidence that follows them
        if drift_active and not was_active and not (include_quarantine and quarantine.pending):
            episode += 1
            if include_quarantine:
                q_test.reset()
        was_active = drift_active

        if include_quarantine:
            q_arm = arms["gate_quarantine"]
            q_entry, q_state = _quarantine_round(
                q_arm, q_gate, q_test, quarantine, q_state, round_idx, episode,
                drift_active, record_reps, window_z, window_mean_z, prev_window_mean,
                [(x, 0) for x, y, meta in round_records])
            q_entry.update({"round": round_idx, "drift_active": drift_active, "phase": phase,
                            "reputation": reputation, "accuracy": q_arm.metric.get(),
                            "f1": q_arm.f1.get(),
                            "round_accuracy": round_acc["gate_quarantine"]})
            log["gate_quarantine"].append(q_entry)

        if not drift_active:
            # steady state (or cooldown expired): no update proposed to anyone
            for name, arm in arms.items():
                if name == "gate_quarantine":
                    continue
                log[name].append({
                    "round": round_idx, "authorized": False, "drift_active": False,
                    "phase": phase, "reputation": reputation,
                    "accuracy": arm.metric.get(), "f1": arm.f1.get(),
                    "round_accuracy": round_acc[name],
                })
            prev_window_mean = window_mean_z
            if on_round:
                on_round(round_idx, {name: log[name][-1] for name in arms})
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

        # ungated (and EWC): drift was signalled -> always authorizes it
        for name in ("ungated", "ewc"):
            if name not in arms:
                continue
            arms[name].learn_batch(proposed_update)
            log[name].append({
                "round": round_idx, "authorized": True, "drift_active": True,
                "phase": phase, "reputation": reputation,
                "accuracy": arms[name].metric.get(), "f1": arms[name].f1.get(),
                "round_accuracy": round_acc[name],
            })

        # static: structurally never authorized
        log["static"].append({
            "round": round_idx, "authorized": False, "drift_active": True,
            "phase": phase, "reputation": reputation,
            "accuracy": arms["static"].metric.get(), "f1": arms["static"].f1.get(),
            "round_accuracy": round_acc["static"],
        })

        prev_window_mean = window_mean_z
        if on_round:
            on_round(round_idx, {name: log[name][-1] for name in arms})

    return log


def _quarantine_round(ids, gate, test, quarantine, state, round_idx, episode, drift_active,
                      record_reps, window_z, window_mean_z, prev_window_mean, proposal):
    """One round of the gate_quarantine arm. Returns (log fields, evidence state)."""
    quarantine.commit_due(round_idx)
    has_pending = any(u.episode == episode for u in quarantine.pending)
    if drift_active or has_pending:
        test.update(record_reps)  # evidence keeps arriving while updates are provisional
    reverted = []
    if test.condemned and has_pending:
        before = len(quarantine.reverted_rounds)
        state = quarantine.rollback(episode, ids, advance_evidence)
        reverted = quarantine.reverted_rounds[before:]

    entry = {"authorized": False, "rolled_back": len(reverted), "reverted_rounds": reverted,
             "episode": episode,
             "sprt_llr": test.llr, "condemned": test.condemned,
             "pending": len(quarantine.pending)}
    if not drift_active:
        return entry, state

    baseline, reference = state
    decision = gate.decide(attribution_consistency(window_z, baseline, reference),
                           topology_diversity(window_z),
                           temporal_stability(window_z, prev_window_mean), 1.0)
    if test.condemned:
        reason = f"episode condemned by sequential reputation test (llr {test.llr:.2f})"
    elif test.llr <= 0:
        # admit provisionally only while the out-of-band evidence so far favours benign
        reason = f"held: reputation evidence not yet favourable (llr {test.llr:.2f})"
    elif decision.authorized:
        quarantine.add(round_idx, episode, proposal, window_mean_z, ids, state)
        ids.learn_batch(proposal)
        state = advance_evidence(state, window_mean_z)
        entry["authorized"] = True
        reason = f"provisional: {decision.reason}"
    else:
        reason = decision.reason
    entry.update({"trust_score": decision.trust_score, "attribution": decision.attribution,
                  "topology": decision.topology, "temporal": decision.temporal,
                  "reason": reason})
    return entry, state


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
    from experiments.streams import SOURCES, build_stream

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="real",
                    help="real N-BaIoT (default), generated fake N-BaIoT, or synthetic")
    ap.add_argument("--synth", action="store_true", help="shorthand for --source synth")
    ap.add_argument("--benign-b", default=None,
                    help="second device's benign_traffic.csv, for Scenario 1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--feature-drift", action="store_true",
                    help="also trigger on feature-distance drift, not just error rate")
    ap.add_argument("--rep-sigma", type=float, default=0.0,
                    help="Gaussian noise on the reputation channel")
    ap.add_argument("--rep-flip", type=float, default=0.0,
                    help="per-record probability the reputation feed is wrong")
    ap.add_argument("--rep-lag", type=int, default=0,
                    help="reputation feed delay, in records")
    ap.add_argument("--tag", default=None,
                    help="output suffix; defaults to the source (synth/fake/real)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    source = "synth" if args.synth else args.source
    tag = args.tag or source
    noise = {"sigma": args.rep_sigma, "flip_rate": args.rep_flip, "lag": args.rep_lag}
    opts = {"feature_drift": args.feature_drift}

    def stream_for(scenario):
        return build_stream(scenario, source, seed=args.seed, reputation_noise=noise,
                            benign_b_csv=args.benign_b)

    print("\nBuilding Scenario 2 (slow poisoning)...")
    stream, names = stream_for("slow_poisoning")
    print(f"{source} stream — {len(names)} features, {len(stream)} records")
    poison_log = run_scenario(stream, names, "Scenario 2: Slow Poisoning", **opts)
    summarize(poison_log, "Scenario 2: Slow Poisoning", phase_of_interest="recovery_check")
    with open(os.path.join(results_dir, f"slow_poisoning_log_{tag}.json"), "w") as f:
        json.dump(poison_log, f, indent=2)

    print("\nBuilding Scenario 1 (genuine drift)...")
    try:
        stream, names = stream_for("genuine_drift")
        drift_log = run_scenario(stream, names, "Scenario 1: Genuine Drift", **opts)
        summarize(drift_log, "Scenario 1: Genuine Drift", phase_of_interest="post_drift")
        with open(os.path.join(results_dir, f"genuine_drift_log_{tag}.json"), "w") as f:
            json.dump(drift_log, f, indent=2)
    except ValueError as e:
        print(f"  SKIPPED — {e}")

    print(f"\nLogs written to {results_dir}/")


if __name__ == "__main__":
    main()
