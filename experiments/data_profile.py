"""
Data processing and feature profile: what the raw data looks like, what each
processing step does to it, and which features carry the signal.

For each real dataset this measures and writes to results/data_profile_<source>.json
(+ .png):
  * raw quality: rows, NaN / Inf counts, negative-valued columns, dynamic range
  * the effect of the processing steps (signed log, then z-score fit on
    training data), as the spread of per-feature standard deviations
  * class balance of the benchmark split and the phase layout of both
    scenario streams
  * feature importance (random forest, impurity-based, on the benchmark's
    training split) with each feature's meaning, and for N-BaIoT the importance
    summed by aggregation and by time window

Usage: python experiments/data_profile.py --source real|iot23
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

from data.features import NBAIOT_GROUPS, NBAIOT_WINDOWS, describe, parse_nbaiot_name  # noqa
from experiments.benchmark import SPLITS, signed_log  # noqa: E402
from experiments.streams import build_stream  # noqa: E402


def raw_quality(source):
    if source == "real":
        from data import load_nbaiot as nb
        files = dict(nb.paths())
        files["ennio_benign"] = os.path.join(nb.DEFAULT_ROOT, "Ennio_Doorbell",
                                             "benign_traffic.csv")
        out = {}
        for key, path in files.items():
            if not os.path.exists(path):
                continue
            a = nb._read(path).to_numpy(dtype=float)
            out[key] = {"rows": int(a.shape[0]), "cols": int(a.shape[1]),
                        "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                        "negative_columns": int((a < 0).any(axis=0).sum()),
                        "max_abs": float(np.abs(a).max())}
        return out
    from data import load_iot23 as io
    out = {}
    for cap in (io.MIRAI, io.BENIGN_B, "CTU-Honeypot-Capture-4-1"):
        path = os.path.join(io.DEFAULT_ROOT, cap, "conn.log.labeled")
        if not os.path.exists(path):
            continue
        df = io.parse_conn_log(path)
        a = df[io.feature_names()].to_numpy(dtype=float)
        out[cap] = {"flows": len(df), "malicious": int(df.label.sum()),
                    "benign": int((df.label == 0).sum()),
                    "attack_types": dict(Counter(d for d, l in zip(df.detail, df.label) if l)),
                    "nan": int(np.isnan(a).sum()), "max_abs": float(np.abs(a).max())}
    return out


def spread(a):
    sd = a.std(axis=0)
    sd = sd[sd > 0]
    return {"std_min": float(sd.min()), "std_median": float(np.median(sd)),
            "std_max": float(sd.max()), "orders_of_magnitude": float(np.log10(sd.max() / sd.min()))}


def stream_layout(source):
    out = {}
    for scenario in ("slow_poisoning", "genuine_drift"):
        stream, _ = build_stream(scenario, source)
        phases = []
        for x, y, m in stream:
            ph = "calibration" if m["source"].startswith("calib") else m["source"]
            if not phases or phases[-1]["phase"] != ph:
                phases.append({"phase": ph, "records": 0, "attack": 0})
            phases[-1]["records"] += 1
            phases[-1]["attack"] += int(y)
        out[scenario] = phases
    return out


def profile(source, scale=1.0, log=print):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte, yte, names, _, provenance = SPLITS[source](None, scale)
    slog = signed_log(Xtr)
    z = StandardScaler().fit(slog).transform(slog)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=1,
                                random_state=42).fit(z, ytr)
    imp = rf.feature_importances_
    order = np.argsort(-imp)
    top = [{"feature": names[i], "importance": float(imp[i]), "meaning": describe(names[i]),
            "benign_mean_z": float(z[ytr == 0, i].mean()),
            "attack_mean_z": float(z[ytr == 1, i].mean())} for i in order[:15]]
    out = {"source": source, "raw_quality": raw_quality(source),
           "split": provenance,
           "class_balance": {"train_benign": int((ytr == 0).sum()),
                             "train_attack": int(ytr.sum()),
                             "test_benign": int((yte == 0).sum()), "test_attack": int(yte.sum())},
           "spread": {"raw": spread(Xtr), "signed_log": spread(slog), "z_score": spread(z)},
           "top_features": top, "stream_layout": stream_layout(source),
           "feature_count": len(names)}
    if source == "real":
        by_group, by_window, grid = Counter(), Counter(), {}
        for n, v in zip(names, imp):
            g, w, _ = parse_nbaiot_name(n)
            by_group[g] += v
            by_window[w] += v
            grid[f"{g}|{w}"] = grid.get(f"{g}|{w}", 0.0) + float(v)
        out["importance_by_group"] = {k: float(v) for k, v in by_group.items()}
        out["importance_by_window"] = {k: float(v) for k, v in by_window.items()}
        out["importance_grid"] = grid
    log(f"{source}: top features " + ", ".join(t["feature"] for t in top[:5]))
    return out


INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def plot(p, path):
    import matplotlib.pyplot as plt
    real = p["source"] == "real"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw={"width_ratios": [0.8, 1.5, 1.1]})
    ax = axes[0]
    steps = ["raw", "signed_log", "z_score"]
    labels = ["raw", "signed log", "signed log\n+ z-score"]
    for i, s in enumerate(steps):
        sp = p["spread"][s]
        ax.plot([i, i], [sp["std_min"], sp["std_max"]], color="#5598e7", lw=6,
                solid_capstyle="round")
        ax.scatter([i], [sp["std_median"]], color=INK, zorder=3, s=18)
        ax.text(i + 0.12, sp["std_max"], f"{sp['orders_of_magnitude']:.1f} decades",
                fontsize=7, color=MUTED, va="center")
    ax.set_yscale("log")
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_xlim(-0.4, 2.9)
    ax.set_title("Per-feature spread after each step", loc="left", fontsize=10, color=INK)
    ax.set_ylabel("feature standard deviation (min-max, dot = median)", fontsize=8,
                  color=MUTED)

    ax = axes[1]
    top = p["top_features"][::-1]
    ax.barh(range(len(top)), [t["importance"] for t in top], color="#256abf", height=0.65)
    ax.set_yticks(range(len(top)), [t["feature"] for t in top], fontsize=7.5)
    ax.set_title("Top 15 features (random-forest importance)", loc="left", fontsize=10,
                 color=INK)
    ax.set_xlabel("share of impurity reduction", fontsize=8, color=MUTED)

    ax = axes[2]
    if real:
        groups, windows = list(NBAIOT_GROUPS), list(NBAIOT_WINDOWS)
        m = np.array([[p["importance_grid"].get(f"{g}|{w}", 0.0) for w in windows]
                      for g in groups])
        im = ax.imshow(m, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(windows)), [f"{w}\n{NBAIOT_WINDOWS[w]}" for w in windows],
                      fontsize=7.5)
        ax.set_yticks(range(len(groups)), groups, fontsize=8)
        for i in range(len(groups)):
            for j in range(len(windows)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() * 0.6 else INK)
        ax.set_title("Importance by aggregation x time window", loc="left", fontsize=10,
                     color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    else:
        top10 = p["top_features"][:8][::-1]
        ys = np.arange(len(top10))
        ax.barh(ys - 0.18, [t["benign_mean_z"] for t in top10], 0.34, color="#2ca02c",
                label="benign")
        ax.barh(ys + 0.18, [t["attack_mean_z"] for t in top10], 0.34, color="#d62728",
                label="attack")
        ax.set_yticks(ys, [t["feature"] for t in top10], fontsize=7.5)
        ax.axvline(0, color=GRID)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Class means of top features (z-score)", loc="left", fontsize=10,
                     color=INK)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED)
    name = "N-BaIoT" if real else "IoT-23"
    fig.suptitle(f"{name}: data processing and features", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SPLITS), default="real")
    args = ap.parse_args()
    p = profile(args.source)
    stem = os.path.join(RESULTS, f"data_profile_{args.source}")
    with open(stem + ".json", "w") as f:
        json.dump(p, f, indent=2)
    print("wrote", stem + ".json")
    plot(p, stem + ".png")


if __name__ == "__main__":
    main()
