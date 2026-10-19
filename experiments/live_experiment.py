"""
Live experiment: the whole pipeline running in real time on a real capture.

The scenario experiments construct their streams (phases, paced poisoning).
This one does not. It takes the IoT-23 Mirai capture CTU-IoT-Malware-Capture-34-1
(23,145 flows recorded over 24 hours, benign and Mirai C&C / DDoS mixed, with a
DDoS burst of ~13,000 flows at hour 17.5) and replays it IN ITS ORIGINAL TIME
ORDER AND TIMING, compressed by a speed factor, exactly as a network tap would
deliver it:

  * the first n_calib flows in time are the labelled past the detector is
    trained on (calibration); everything after is unseen, live traffic
  * each 20-flow round is released when its last flow's (scaled) timestamp is
    reached on the wall clock; if processing falls behind, the backlog lag is
    measured instead of hidden
  * all six arms run (static, ungated, traffic-only gate, full gate, EWC,
    quarantine) with the feature-drift trigger on, since live traffic has no
    designed phases
  * per round: wall-clock processing time, flows/s, backlog lag, and each arm's
    true/false positives and negatives

Reputation is still simulated from ground truth (as everywhere in the project);
`reputation_noise` degrades it. Labels are used only to SCORE predictions after
the fact (prequential), never to train after calibration.

Usage:
  python experiments/live_experiment.py --speed 600         # 24 h in ~2.4 min
  python experiments/live_experiment.py --speed 0           # as fast as possible
Writes results/live_experiment_iot23.json and .png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

from experiments.run_scenarios import ROUND_SIZE, run_scenario  # noqa: E402

N_CALIB = 2000
ARMS = ["static", "ungated", "gate_traffic_only", "gate_full", "ewc", "gate_quarantine"]


def chronological_stream(n_calib=N_CALIB, max_flows=None, root=None):
    """(stream, feature_names, timestamps): the Mirai capture in time order."""
    from data import load_iot23 as io
    path = os.path.join(root or io.DEFAULT_ROOT, io.MIRAI, "conn.log.labeled")
    df = io.parse_conn_log(path).assign(ts=io.parse_timestamps(path))
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    if max_flows:
        df = df.iloc[:max_flows]
    names = io.feature_names()
    X = df[names].to_numpy(dtype=float)
    stream = []
    for i, (vec, y) in enumerate(zip(X, df.label.to_numpy())):
        rep = io.ATTACK_REPUTATION if y else io.BENIGN_REPUTATION
        stream.append(({k: float(v) for k, v in zip(names, vec)}, int(y),
                       {"reputation": rep, "source": "calibration" if i < n_calib else "live",
                        "detail": df.detail.iat[i]}))
    return stream, names, df.ts.to_numpy()


def summarize_arm(entries):
    c = {k: sum(e.get(k, 0) for e in entries) for k in ("tp", "fp", "tn", "fn")}
    n = sum(c.values())
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {**c, "accuracy": (c["tp"] + c["tn"]) / n if n else float("nan"),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else 0.0,
            "updates_applied": sum(1 for e in entries if e.get("authorized"))}


class LiveExperiment:
    """Replays a chronological stream in (scaled) real time through run_scenario."""

    def __init__(self, speed=600.0, n_calib=N_CALIB, max_flows=None, reputation_noise=None,
                 quarantine_params=None, root=None):
        self.speed, self.n_calib = speed, n_calib
        self.stream, self.names, self.ts = chronological_stream(n_calib, max_flows, root)
        if reputation_noise:
            from data.reputation import ReputationNoise, apply_reputation_noise
            self.stream = apply_reputation_noise(self.stream, ReputationNoise(**reputation_noise))
        self.quarantine_params = quarantine_params
        self.rounds = []

    def run(self, on_event=None, sleep=time.sleep, clock=time.perf_counter):
        t_start = clock()
        ts0 = self.ts[self.n_calib] if len(self.ts) > self.n_calib else self.ts[0]
        last = {"t": clock()}

        def on_round(round_idx, entries):
            now = clock()
            busy = now - last["t"]                      # processing time of this round
            i_last = self.n_calib + round_idx * ROUND_SIZE - 1
            capture_s = float(self.ts[i_last] - ts0)
            due = capture_s / self.speed if self.speed else 0.0
            lag = max(0.0, (now - t_start) - due)       # behind the live arrival time
            ev = {"round": round_idx, "capture_hours": capture_s / 3600,
                  "wall_s": now - t_start, "lag_s": lag,
                  "busy_ms_per_flow": busy / ROUND_SIZE * 1000,
                  "malicious": sum(1 for _, y, _ in
                                   self.stream[i_last - ROUND_SIZE + 1:i_last + 1] if y),
                  "drift_active": entries["static"]["drift_active"],
                  "arms": {a: {k: entries[a].get(k) for k in
                               ("tp", "fp", "tn", "fn", "authorized", "accuracy",
                                "rolled_back", "condemned")} for a in ARMS if a in entries}}
            self.rounds.append(ev)
            if on_event:
                on_event(ev)
            if self.speed:
                wait = due - (clock() - t_start)
                if wait > 0:
                    sleep(wait)
            last["t"] = clock()

        self.log = run_scenario(self.stream, self.names, "live", n_calib=self.n_calib,
                                on_round=on_round, feature_drift=True, record_confusion=True,
                                quarantine_params=self.quarantine_params)
        self.wall_seconds = clock() - t_start
        return self.summary()

    def summary(self):
        # round 1's time includes training on the calibration flows, so it is left out
        busy = np.array([r["busy_ms_per_flow"] for r in self.rounds[1:]] or [0.0])
        n_live = len(self.rounds) * ROUND_SIZE
        live = {a: [e for e in self.log[a] if e.get("phase") == "live"] for a in ARMS
                if a in self.log}
        return {
            "capture": "CTU-IoT-Malware-Capture-34-1 (IoT-23)",
            "speed": self.speed, "calibration_flows": self.n_calib, "live_flows": n_live,
            "live_malicious": int(sum(r["malicious"] for r in self.rounds)),
            "capture_hours": float((self.ts[-1] - self.ts[0]) / 3600),
            "wall_seconds": self.wall_seconds,
            "throughput_flows_per_s": n_live / self.wall_seconds if self.wall_seconds else None,
            "processing_ms_per_flow_all_arms": {
                "p50": float(np.percentile(busy, 50)), "p95": float(np.percentile(busy, 95)),
                "p99": float(np.percentile(busy, 99)), "max": float(busy.max())},
            "max_lag_s": float(max(r["lag_s"] for r in self.rounds)),
            "drift_rounds": int(sum(r["drift_active"] for r in self.rounds)),
            "arms": {a: summarize_arm(es) for a, es in live.items()},
        }


INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLORS = {"static": "#9aa0a6", "ungated": "#d62728", "gate_traffic_only": "#ff7f0e",
          "gate_full": "#2ca02c", "ewc": "#1f77b4", "gate_quarantine": "#9467bd"}
LABELS = {"static": "never learns", "ungated": "learns everything",
          "gate_traffic_only": "traffic-only gate", "gate_full": "full gate (one-shot)",
          "ewc": "EWC", "gate_quarantine": "quarantine gate"}


def plot(result, rounds, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.4), sharex=False)
    x = np.arange(1, len(rounds) + 1)
    ax = axes[0]
    ax.fill_between(x, [r["malicious"] / ROUND_SIZE for r in rounds], color="#d62728",
                    alpha=0.25, step="mid", label="share of malicious flows in the round")
    ax.plot(x, [r["capture_hours"] / max(1e-9, result["capture_hours"]) for r in rounds],
            color=INK, lw=1, label="capture time (fraction of 24 h)")
    drift = [r["drift_active"] for r in rounds]
    ax.fill_between(x, 0, 1, where=drift, color="#79c0ff", alpha=0.18, step="mid",
                    label="drift active (updates proposed)")
    ax.set_title("Live traffic as it arrived (real capture order)", loc="left", fontsize=10,
                 color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))

    ax = axes[1]
    win = 10
    for a in ARMS:
        if a not in rounds[0]["arms"]:
            continue
        acc = np.array([(r["arms"][a]["tp"] + r["arms"][a]["tn"]) / ROUND_SIZE for r in rounds])
        roll = np.convolve(acc, np.ones(win) / win, mode="same")
        ax.plot(x, roll, color=COLORS[a], lw=1.6, label=LABELS[a])
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Accuracy on arriving flows ({win}-round rolling mean)", loc="left",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))

    ax = axes[2]
    # round 1 includes training on the calibration flows; it is not live processing
    ax.plot(x[1:], [r["busy_ms_per_flow"] for r in rounds[1:]], color="#256abf", lw=1,
            label="processing time per flow, all 6 arms (ms)")
    ax2 = ax.twinx()
    ax2.plot(x, [r["lag_s"] for r in rounds], color="#eb6834", lw=1.2,
             label="backlog behind real time (s)")
    ax2.set_ylabel("backlog (s)", fontsize=8, color="#eb6834")
    ax2.tick_params(colors="#eb6834", labelsize=8)
    ax.set_xlabel("round (20 flows)", fontsize=8, color=MUTED)
    ax.set_ylabel("ms per flow", fontsize=8, color="#256abf")
    ax.set_title(f"Real-time cost at {result['speed']:g}x speed: "
                 f"{result['throughput_flows_per_s']:.0f} flows/s sustained, "
                 f"max backlog {result['max_lag_s']:.1f} s", loc="left", fontsize=10,
                 color=INK)
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Live experiment: IoT-23 Mirai capture replayed in real time", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=600.0,
                    help="capture seconds per wall-clock second (0 = as fast as possible)")
    ap.add_argument("--max-flows", type=int, default=None)
    ap.add_argument("--rep-flip", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="output suffix, e.g. maxspeed")
    args = ap.parse_args()
    noise = {"flip_rate": args.rep_flip, "sigma": 0.05, "seed": 42} if args.rep_flip else None
    exp = LiveExperiment(speed=args.speed, max_flows=args.max_flows, reputation_noise=noise)
    print(f"replaying {len(exp.stream)} flows at {args.speed:g}x "
          f"(~{(exp.ts[-1] - exp.ts[exp.n_calib]) / max(args.speed, 1e-9) / 60:.1f} min)")
    summary = exp.run(on_event=lambda e: e["round"] % 100 == 0 and print(
        f"  round {e['round']:4d}  capture {e['capture_hours']:5.2f} h  "
        f"lag {e['lag_s']:5.2f} s  {e['busy_ms_per_flow']:.2f} ms/flow", flush=True))
    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.join(RESULTS, "live_experiment_iot23" + (f"_{args.tag}" if args.tag else ""))
    with open(stem + ".json", "w") as f:
        json.dump({"summary": summary, "rounds": exp.rounds}, f, indent=1)
    print(json.dumps(summary, indent=1))
    plot(summary, exp.rounds, stem + ".png")


if __name__ == "__main__":
    main()
