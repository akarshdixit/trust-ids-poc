"""
Robustness sweeps — the analyses a single seed and an oracle reputation
channel cannot answer.

  seeds       repeat both scenarios over N seeds; mean, std and 95% CI per arm
  reputation  degrade the out-of-band channel (flip rate x veto threshold) and
              measure what the full gate loses: attack resistance in Scenario 2,
              adaptation to benign drift in Scenario 1
  ewc         EWC strength lambda vs both scenarios (stability/plasticity)
  quarantine  the quarantine extension vs the one-shot gate under a degraded
              feed, with ablations: no rollback, no lowered threshold, and a
              one-shot gate whose veto sits at the feed's midpoint (0.5)

Usage:
  python experiments/sweeps.py seeds --source synth --n 10
  python experiments/sweeps.py reputation --source synth
  python experiments/sweeps.py ewc --source synth
  python experiments/sweeps.py quarantine --source real

Outputs land in results/sweep_<kind>_<source>.json (+ .png for reputation/ewc).
On --source fake the seed only reshuffles calibration rows (the files are fixed),
so CIs there are narrower than on synth, where each seed is a new world.
"""
import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

from experiments.run_scenarios import N_CALIB, ROUND_SIZE, run_scenario  # noqa: E402
from experiments.streams import SOURCES, build_stream  # noqa: E402

# the metric each scenario is judged on: windowed accuracy in its decisive phase
DECISIVE = {"slow_poisoning": "recovery_check", "genuine_drift": "post_drift"}


def phase_accuracy(log, arm, phase):
    sel = [r["round_accuracy"] for r in log[arm] if r.get("phase") == phase]
    return float(np.mean(sel)) if sel else float("nan")


def poisoned_updates_kept(log, arm, stream, n_calib=N_CALIB):
    """Authorized updates whose window held attack records and were never rolled back."""
    rest = stream[n_calib:]
    reverted = {r for e in log[arm] for r in e.get("reverted_rounds", [])}
    return sum(1 for e in log[arm] if e.get("authorized") and e["round"] not in reverted
               and any(y == 1 for _, y, _ in rest[(e["round"] - 1) * ROUND_SIZE:
                                                  e["round"] * ROUND_SIZE]))


def _ci95(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float("nan")
    from scipy import stats
    return float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))


def sweep_seeds(source, n, feature_drift=False):
    out = {}
    for scenario, phase in DECISIVE.items():
        per_arm = {}
        for seed in range(n):
            stream, names = build_stream(scenario, source, seed=seed)
            log = run_scenario(stream, names, scenario, feature_drift=feature_drift)
            for arm in log:
                per_arm.setdefault(arm, []).append(phase_accuracy(log, arm, phase))
        out[scenario] = {arm: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                               "ci95": _ci95(v), "values": v}
                         for arm, v in per_arm.items()}
        print(f"\n{scenario} — accuracy in '{phase}', {n} seeds ({source})")
        for arm, s in out[scenario].items():
            print(f"  {arm:20s} {s['mean']:.3f} ± {s['ci95']:.3f}  (sd {s['std']:.3f})")
    return out


FLIP_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
VETO_THRESHOLDS = [0.2, 0.3, 0.5, 0.7]


def sweep_reputation(source, sigma=0.05, seed=42):
    """Grid: reputation flip rate x veto threshold, both scenarios, full gate only."""
    rows = []
    for scenario, phase in DECISIVE.items():
        for flip in FLIP_RATES:
            noise = {"sigma": sigma if flip else 0.0, "flip_rate": flip, "seed": seed}
            stream, names = build_stream(scenario, source, seed=seed, reputation_noise=noise)
            for veto in VETO_THRESHOLDS:
                log = run_scenario(stream, names, scenario, include_ewc=False,
                                   gate_params={"reputation_veto_threshold": veto})
                vetoed = sum(1 for r in log["gate_full"] if r.get("drift_active")
                             and "veto" in r.get("reason", ""))
                rows.append({"scenario": scenario, "flip_rate": flip, "sigma": noise["sigma"],
                             "veto_threshold": veto,
                             "gate_full": phase_accuracy(log, "gate_full", phase),
                             "gate_traffic_only": phase_accuracy(log, "gate_traffic_only", phase),
                             "vetoed_rounds": vetoed})
                print(f"  {scenario:15s} flip={flip:.1f} veto<{veto:.1f}  "
                      f"full={rows[-1]['gate_full']:.3f}  vetoed={vetoed}")
    return rows


EWC_LAMBDAS = [0.0, 1.0, 5.0, 20.0, 50.0, 100.0]


def sweep_ewc(source, seed=42):
    rows = []
    for lam in EWC_LAMBDAS:
        row = {"lambda": lam}
        for scenario, phase in DECISIVE.items():
            stream, names = build_stream(scenario, source, seed=seed)
            log = run_scenario(stream, names, scenario, ewc_lambda=lam)
            row[scenario] = phase_accuracy(log, "ewc", phase)
            row[f"{scenario}_gate_full"] = phase_accuracy(log, "gate_full", phase)
        rows.append(row)
        print(f"  lambda={lam:6.1f}  repeat-attack={row['slow_poisoning']:.3f}  "
              f"post-drift={row['genuine_drift']:.3f}")
    return rows


# (label, gate_params, quarantine_params, arm to read). One run_scenario yields
# a gate_full and a gate_quarantine arm, so variants are paired to share runs.
QUARANTINE_RUNS = [
    ({"reputation_veto_threshold": 0.3}, {},
     {"gate_full": "one-shot gate, veto < 0.3", "gate_quarantine": "quarantine (proposed)"}),
    ({"reputation_veto_threshold": 0.5}, {"quarantine_rounds": 0},
     {"gate_full": "one-shot gate, veto < 0.5", "gate_quarantine": "quarantine, no rollback"}),
    ({"reputation_veto_threshold": 0.3}, {"provisional_threshold": 0.55},
     {"gate_quarantine": "quarantine, threshold 0.55"}),
    ({"reputation_veto_threshold": 0.3}, {"quarantine_rounds": 5},
     {"gate_quarantine": "quarantine, 5-round hold"}),
]
QUARANTINE_FEEDS = ([{"flip_rate": f} for f in (0.0, 0.1, 0.2, 0.3, 0.4)]
                    + [{"lag": lag} for lag in (300, 600)])


def sweep_quarantine(source, sigma=0.05, seed=42):
    rows = []
    for scenario, phase in DECISIVE.items():
        for feature_drift in (False, True):
            for feed in QUARANTINE_FEEDS:
                if feature_drift and feed != {"flip_rate": 0.0}:
                    continue  # feature drift is only probed on the oracle feed
                flip = feed.get("flip_rate", 0.0)
                noise = {"sigma": sigma if flip else 0.0, "flip_rate": flip,
                         "lag": feed.get("lag", 0), "seed": seed}
                stream, names = build_stream(scenario, source, seed=seed,
                                             reputation_noise=noise)
                row = {"scenario": scenario, "feature_drift": feature_drift, **noise}
                # ablations change one setting each; the proposed arm is the qp == {} run
                for gp, qp, read in QUARANTINE_RUNS:
                    log = run_scenario(stream, names, scenario, include_ewc=False,
                                       gate_params=gp, quarantine_params=qp,
                                       feature_drift=feature_drift)
                    for arm, label in read.items():
                        row[label] = phase_accuracy(log, arm, phase)
                        row[f"{label} | poisoned updates kept"] = poisoned_updates_kept(
                            log, arm, stream)
                    if not qp:
                        q = log["gate_quarantine"]
                        row["static"] = phase_accuracy(log, "static", phase)
                        row["provisional"] = sum(r["authorized"] for r in q)
                        row["rolled_back"] = sum(r["rolled_back"] for r in q)
                rows.append(row)
                print(f"  {scenario:15s} fd={int(feature_drift)} flip={flip:.1f} "
                      f"lag={noise['lag']:3d}  " + "  ".join(
                          f"{k}={v:.3f}" for k, v in row.items()
                          if isinstance(v, float) and k not in ("sigma", "flip_rate")),
                      flush=True)
                print("      poisoned updates kept: " + "  ".join(
                    f"{k.split(' | ')[0]}={v}" for k, v in row.items() if " | " in k),
                    flush=True)
    return rows


# ------------------------------------------------------------------ figures
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#0d366b"]   # ordinal: low -> high veto
SERIES = {"ewc": "#2a78d6", "gate_full": "#eb6834"}


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_reputation(rows, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    panels = [("slow_poisoning", "Scenario 2 — catches the repeat attack",
               "accuracy, recovery_check"),
              ("genuine_drift", "Scenario 1 — adapts to benign drift",
               "accuracy, post_drift")]
    for ax, (scenario, title, ylabel) in zip(axes, panels):
        for color, veto in zip(BLUE_RAMP, VETO_THRESHOLDS):
            pts = sorted((r["flip_rate"], r["gate_full"]) for r in rows
                         if r["scenario"] == scenario and r["veto_threshold"] == veto)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                    label=f"veto below {veto:.1f}")
        _style(ax, title, ylabel)
        ax.set_xlabel("reputation feed error rate (per record)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.suptitle("Full gate under an imperfect reputation channel", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def plot_ewc(rows, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    lams = [r["lambda"] for r in rows]
    xs = list(range(len(lams)))
    for ax, (scenario, title) in zip(axes, [("slow_poisoning", "Scenario 2 — repeat attack"),
                                            ("genuine_drift", "Scenario 1 — post-drift")]):
        ax.plot(xs, [r[scenario] for r in rows], color=SERIES["ewc"], linewidth=2,
                marker="o", markersize=5, label="EWC (learns everything, anchored)")
        ax.plot(xs, [r[f"{scenario}_gate_full"] for r in rows], color=SERIES["gate_full"],
                linewidth=2, linestyle="--", label="full gate (reference)")
        _style(ax, title, "windowed accuracy")
        ax.set_xticks(xs, [f"{v:g}" for v in lams])
        ax.set_xlabel("EWC strength λ", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.suptitle("EWC trades adaptation for stability; the gate needs neither trade",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


QUARANTINE_SERIES = [  # proposed first; one-shot gates reuse the gate_full hue
    ("quarantine (proposed)", "#9467bd", "-"),
    ("one-shot gate, veto < 0.3", "#eb6834", "--"),
    ("one-shot gate, veto < 0.5", "#eb6834", ":"),
    ("quarantine, no rollback", "#9467bd", ":"),
]


def plot_quarantine(rows, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    panels = [("slow_poisoning", "Scenario 2 — catches the repeat attack",
               "accuracy, recovery_check"),
              ("genuine_drift", "Scenario 1 — adapts to benign drift",
               "accuracy, post_drift")]
    for ax, (scenario, title, ylabel) in zip(axes, panels):
        sel = sorted((r for r in rows if r["scenario"] == scenario and not r["feature_drift"]
                      and r["lag"] == 0), key=lambda r: r["flip_rate"])
        xs = [r["flip_rate"] for r in sel]
        for label, color, style in QUARANTINE_SERIES:
            ax.plot(xs, [r[label] for r in sel], color=color, linestyle=style, linewidth=2,
                    marker="o" if style == "-" else None, markersize=5, label=label)
        _style(ax, title, ylabel)
        ax.set_xlabel("reputation feed error rate (per record)", color=MUTED, fontsize=9)
        ax.axvline(0.3, color=GRID, linewidth=1)
    axes[0].text(0.305, 0.02, "assumed worst-case\nfeed error", color=MUTED, fontsize=7)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.suptitle("Quarantine + sequential evidence vs a one-shot reputation veto",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["seeds", "reputation", "ewc", "quarantine"])
    ap.add_argument("--source", choices=SOURCES, default="synth")
    ap.add_argument("--n", type=int, default=10, help="seeds, for kind=seeds")
    ap.add_argument("--feature-drift", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.join(RESULTS, f"sweep_{args.kind}_{args.source}")
    if args.kind == "seeds":
        data = sweep_seeds(args.source, args.n, args.feature_drift)
    elif args.kind == "reputation":
        data = sweep_reputation(args.source)
        plot_reputation(data, stem + ".png")
    elif args.kind == "quarantine":
        data = sweep_quarantine(args.source)
        plot_quarantine(data, stem + ".png")
    else:
        data = sweep_ewc(args.source)
        plot_ewc(data, stem + ".png")
    with open(stem + ".json", "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", stem + ".json")


if __name__ == "__main__":
    main()
