import json
import os
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")

COLORS = {
    "static": "#999999",
    "ungated": "#d62728",
    "gate_traffic_only": "#ff7f0e",
    "gate_full": "#2ca02c",
}
LABELS = {
    "static": "Static (never updates)",
    "ungated": "Ungated continual learning",
    "gate_traffic_only": "Traffic-only gate (ExpIDS-style)",
    "gate_full": "Proposed: trust gate + reputation veto",
}


def plot_scenario(log_path, title, out_path, phase_lines=None):
    log = json.load(open(log_path))
    fig, ax = plt.subplots(figsize=(10, 5))
    for arm in ["static", "ungated", "gate_traffic_only", "gate_full"]:
        rounds = [r["round"] for r in log[arm]]
        accs = [r["round_accuracy"] for r in log[arm]]
        ax.plot(rounds, accs, label=LABELS[arm], color=COLORS[arm], linewidth=2)
    if phase_lines:
        for x, text in phase_lines:
            ax.axvline(x, color="black", linestyle="--", alpha=0.3)
            ax.text(x, 1.03, text, rotation=0, fontsize=8, ha="center")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Windowed detection accuracy (this round only)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


def plot_authorization_timeline(log_path, out_path):
    log = json.load(open(log_path))["gate_full"]
    log = [r for r in log if "trust_score" in r]  # only rounds where drift was active
    rounds = [r["round"] for r in log]
    trust = [r["trust_score"] for r in log]
    rep = [r["reputation"] for r in log]
    auth = [1 if r["authorized"] else 0 for r in log]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, trust, label="traffic-evidence trust score", color="#1f77b4")
    ax.plot(rounds, rep, label="out-of-band reputation", color="#9467bd")
    ax.fill_between(rounds, 0, 1, where=[a == 0 for a in auth], color="red", alpha=0.08,
                     label="update held / rejected")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Score")
    ax.set_title("Proposed gate: decision trace during slow poisoning")
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


if __name__ == "__main__":
    plot_scenario(
        os.path.join(RESULTS, "genuine_drift_log.json"),
        "Scenario 1 — Genuine drift: static falls behind, all continual learners adapt",
        os.path.join(RESULTS, "scenario1_genuine_drift.png"),
        phase_lines=[(20, "drift starts"), (50, "drift settled")],
    )
    plot_scenario(
        os.path.join(RESULTS, "slow_poisoning_log.json"),
        "Scenario 2 — Slow poisoning: only the trust gate survives it",
        os.path.join(RESULTS, "scenario2_slow_poisoning.png"),
        phase_lines=[(20, "poisoning starts"), (50, "post-poison"), (70, "obvious repeat attack")],
    )
    plot_authorization_timeline(
        os.path.join(RESULTS, "slow_poisoning_log.json"),
        os.path.join(RESULTS, "gate_decision_trace.png"),
    )
