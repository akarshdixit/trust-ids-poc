"""
Detector benchmark: how good is the IDS model itself, compared with other
models, on clean held-out traffic?

The scenario experiments (run_scenarios.py) measure what an attacker can do to
a model that keeps learning. This module answers the separate, standard
question a reviewer asks first: trained the usual way, which classifier detects
attacks best, and at what cost?

Protocol (identical for every model):
  * TEMPORAL split per file / capture and per class: the first 70% of rows in
    time order are the training pool, the last 30% the test pool, so a model is
    always tested on traffic recorded after the traffic it learned from.
    Random splits leak neighbouring, nearly identical packet windows.
  * Pools are subsampled with an even stride (spread over the whole period).
  * Preprocessing fit on the TRAINING split only: signed log, then z-score.
  * Extra test: an unseen device's benign traffic (N-BaIoT Ennio_Doorbell),
    reported as its false-positive rate.

Models
  batch (scikit-learn): logistic regression, linear SVM, decision tree, random
    forest, histogram gradient boosting, k-NN, Gaussian naive Bayes, MLP;
    unsupervised on benign only: isolation forest and a deep autoencoder
    (the N-BaIoT paper's method)
  online (River, one pass over the shuffled training split): logistic
    regression, Hoeffding tree, Hoeffding adaptive tree, adaptive random
    forest, Gaussian naive Bayes, and "ours": the exact online LR of the
    scenario experiments, trained on its 400-record calibration set only

Usage:
  python experiments/benchmark.py --source real     # N-BaIoT
  python experiments/benchmark.py --source iot23
Writes results/benchmark_<source>.json and .png.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

TRAIN_FRACTION = 0.7
OURS = "Online LR, 400 calibration rows (ours, as deployed)"


def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _stride(a, n):
    """n rows spread evenly over a (time-ordered) array; all rows if n >= len(a)."""
    if n >= len(a):
        return a
    return a[np.linspace(0, len(a) - 1, n).astype(int)]


def _temporal(a, n_train, n_test):
    cut = int(len(a) * TRAIN_FRACTION)
    return _stride(a[:cut], n_train), _stride(a[cut:], n_test)


# ------------------------------------------------------------------ datasets
def nbaiot_split(root=None, scale=1.0):
    """Danmini_Doorbell benign vs Mirai scan + Gafgyt combo; Ennio benign as unseen device."""
    from data import load_nbaiot as nb
    p = nb.paths(root)
    names = nb.feature_names(root=root)
    parts = {"benign": (p["benign"], 0, 15000, 6000),
             "mirai_scan": (p["mirai_scan"], 1, 7500, 3000),
             "gafgyt_combo": (p["gafgyt_combo"], 1, 7500, 3000)}
    Xtr, ytr, Xte, yte, provenance = [], [], [], [], {}
    for key, (path, label, ntr, nte) in parts.items():
        a = nb._read(path).to_numpy(dtype=float)
        tr, te = _temporal(a, int(ntr * scale), int(nte * scale))
        Xtr.append(tr), Xte.append(te)
        ytr.append(np.full(len(tr), label)), yte.append(np.full(len(te), label))
        provenance[key] = {"rows_in_file": len(a), "train": len(tr), "test": len(te)}
    ennio = os.path.join(root or nb.DEFAULT_ROOT, "Ennio_Doorbell", "benign_traffic.csv")
    extra = {}
    if os.path.exists(ennio):
        e = nb._read(ennio).to_numpy(dtype=float)
        extra["unseen_device_benign"] = _stride(e, int(6000 * scale))
        provenance["ennio_benign (unseen device)"] = {"rows_in_file": len(e),
                                                      "test": len(extra["unseen_device_benign"])}
    return (np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xte), np.concatenate(yte),
            names, extra, provenance)


def iot23_split(root=None, scale=1.0):
    """Mirai capture 34-1 (both classes) + benign honeypots, each in time order."""
    from data import load_iot23 as io
    names = io.feature_names()
    Xtr, ytr, Xte, yte, provenance = [], [], [], [], {}
    caps = [(io.MIRAI, 1, 6000, 3000), (io.MIRAI, 0, 10 ** 6, 10 ** 6),
            (io.BENIGN_B, 0, 10 ** 6, 10 ** 6), ("CTU-Honeypot-Capture-4-1", 0, 10 ** 6, 10 ** 6)]
    for cap, label, ntr, nte in caps:
        path = os.path.join(root or io.DEFAULT_ROOT, cap, "conn.log.labeled")
        if not os.path.exists(path):
            continue
        df = io.parse_conn_log(path).assign(ts=io.parse_timestamps(path))
        df = df[df.label == label].sort_values("ts", kind="stable")
        a = df[names].to_numpy(dtype=float)
        tr, te = _temporal(a, int(ntr * scale), int(nte * scale))
        Xtr.append(tr), Xte.append(te)
        ytr.append(np.full(len(tr), label)), yte.append(np.full(len(te), label))
        provenance[f"{cap} {'malicious' if label else 'benign'}"] = {
            "flows": len(a), "train": len(tr), "test": len(te)}
    return (np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xte), np.concatenate(yte),
            names, {}, provenance)


SPLITS = {"real": nbaiot_split, "iot23": iot23_split}


# ------------------------------------------------------------------ models
def batch_models(seed):
    from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    return {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Linear SVM": LinearSVC(dual="auto", max_iter=5000, random_state=seed),
        "Decision tree": DecisionTreeClassifier(min_samples_leaf=5, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                n_jobs=1, random_state=seed),
        "Gradient boosting (HistGB)": HistGradientBoostingClassifier(random_state=seed),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian naive Bayes": GaussianNB(),
        "MLP (64-32)": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                     max_iter=300, random_state=seed),
        "Isolation forest (benign only)": IsolationForest(n_estimators=200, random_state=seed),
        "Autoencoder (benign only, N-BaIoT paper)": "autoencoder",
    }


class _Autoencoder:
    """Deep autoencoder on benign traffic; anomaly if reconstruction error exceeds
    the 99th percentile of the benign training error (Meidan et al. 2018)."""

    def __init__(self, seed):
        from sklearn.neural_network import MLPRegressor
        self.net = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=200,
                                early_stopping=True, random_state=seed)

    def fit(self, X):
        self.net.fit(X, X)
        self.threshold = np.percentile(self._err(X), 99)
        return self

    def _err(self, X):
        return np.mean((self.net.predict(X) - X) ** 2, axis=1)

    def score(self, X):
        return self._err(X)

    def predict(self, X):
        return (self._err(X) > self.threshold).astype(int)


def online_models():
    from river import ensemble, forest, linear_model, naive_bayes, preprocessing, tree
    return {
        "Online logistic regression (River)":
            lambda: preprocessing.StandardScaler() | linear_model.LogisticRegression(),
        "Hoeffding tree (River)": lambda: tree.HoeffdingTreeClassifier(),
        "Hoeffding adaptive tree (River)": lambda: tree.HoeffdingAdaptiveTreeClassifier(seed=1),
        "Adaptive random forest (River)": lambda: forest.ARFClassifier(n_models=5, seed=1),
        "Gaussian naive Bayes (River)": lambda: naive_bayes.GaussianNB(),
    }


# ------------------------------------------------------------------ metrics
def metrics(y, pred, score=None):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score)
    y, pred = np.asarray(y), np.asarray(pred)
    neg = y == 0
    out = {"accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "mcc": matthews_corrcoef(y, pred),
           "fpr": float(np.mean(pred[neg] == 1)) if neg.any() else float("nan")}
    try:
        out["roc_auc"] = roc_auc_score(y, score if score is not None else pred)
    except ValueError:
        out["roc_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def _single_latency_us(predict_one, rows, n=200):
    ts = []
    for r in rows[:n]:
        t0 = time.perf_counter()
        predict_one(r)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6)


def evaluate_batch(name, model, Xtr, ytr, Xte, yte, extra, seed):
    unsupervised = "benign only" in name
    if model == "autoencoder":
        model = _Autoencoder(seed)
    t0 = time.perf_counter()
    model.fit(Xtr[ytr == 0]) if unsupervised else model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0

    def predict(X):
        if isinstance(model, _Autoencoder):
            return model.predict(X), model.score(X)
        p = model.predict(X)
        if unsupervised:                       # IsolationForest: -1 = anomaly
            return (p == -1).astype(int), -model.decision_function(X)
        if hasattr(model, "predict_proba"):
            return p, model.predict_proba(X)[:, 1]
        return p, model.decision_function(X)

    t0 = time.perf_counter()
    pred, score = predict(Xte)
    batch_us = (time.perf_counter() - t0) / len(Xte) * 1e6
    row = {"model": name, "family": "unsupervised" if unsupervised else "batch",
           **metrics(yte, pred, score), "train_seconds": train_s,
           "batch_latency_us": batch_us,
           "single_latency_us": _single_latency_us(lambda r: predict(r[None, :]), Xte),
           "model_kb": len(pickle.dumps(model)) / 1024}
    if "unseen_device_benign" in extra:
        row["unseen_device_fpr"] = float(np.mean(predict(extra["unseen_device_benign"])[0]))
    return row


def evaluate_online(name, factory, Xtr, ytr, Xte, yte, extra, names, seed, n_train=None):
    to_dict = lambda r: dict(zip(names, map(float, r)))  # noqa: E731
    model = factory()
    order = np.random.default_rng(seed).permutation(len(Xtr))[:n_train]
    t0 = time.perf_counter()
    for i in order:
        model.learn_one(to_dict(Xtr[i]), int(ytr[i]))
    train_s = time.perf_counter() - t0
    dicts = [to_dict(r) for r in Xte]
    t0 = time.perf_counter()
    pred = np.array([int(model.predict_one(d) or 0) for d in dicts])
    online_us = (time.perf_counter() - t0) / len(dicts) * 1e6
    score = np.array([model.predict_proba_one(d).get(1, 0.0) for d in dicts])
    row = {"model": name, "family": "online", **metrics(yte, pred, score),
           "train_seconds": train_s, "train_rows": len(order),
           "batch_latency_us": online_us,
           "single_latency_us": _single_latency_us(model.predict_one, dicts),
           "model_kb": len(pickle.dumps(model)) / 1024}
    if "unseen_device_benign" in extra:
        u = [to_dict(r) for r in extra["unseen_device_benign"]]
        row["unseen_device_fpr"] = float(np.mean([int(model.predict_one(d) or 0) for d in u]))
    return row


def calibration_rows(Xtr, ytr, n=400, attack_fraction=0.3, seed=42):
    """The scenario detector's training budget: 400 rows, 30% attack, shuffled."""
    rng = np.random.default_rng(seed)
    n_att = int(n * attack_fraction)
    b = rng.choice(np.flatnonzero(ytr == 0), n - n_att, replace=False)
    a = rng.choice(np.flatnonzero(ytr == 1), n_att, replace=False)
    idx = np.concatenate([b, a])
    return idx[rng.permutation(len(idx))]


# ------------------------------------------------------------------ runner
def run(source, repeats=3, scale=1.0, online_train_cap=12000, root=None, log=print):
    from sklearn.preprocessing import StandardScaler
    Xtr_raw, ytr, Xte_raw, yte, names, extra_raw, provenance = SPLITS[source](root, scale)
    scaler = StandardScaler().fit(signed_log(Xtr_raw))     # fit on TRAIN only
    prep = lambda a: scaler.transform(signed_log(a))       # noqa: E731
    Xtr, Xte = prep(Xtr_raw), prep(Xte_raw)
    extra = {k: prep(v) for k, v in extra_raw.items()}
    log(f"{source}: train {len(Xtr)} ({int(ytr.sum())} attack), test {len(Xte)} "
        f"({int(yte.sum())} attack), {len(names)} features")

    runs = {}
    for rep in range(repeats):
        seed = 42 + rep
        for name, model in batch_models(seed).items():
            runs.setdefault(name, []).append(
                evaluate_batch(name, model, Xtr, ytr, Xte, yte, extra, seed))
            log(f"  [{rep}] {name:42s} F1={runs[name][-1]['f1']:.4f}")
        for name, factory in online_models().items():   # same preprocessed inputs
            runs.setdefault(name, []).append(evaluate_online(
                name, factory, Xtr, ytr, Xte, yte, extra, names, seed,
                n_train=online_train_cap))
            log(f"  [{rep}] {name:42s} F1={runs[name][-1]['f1']:.4f}")
        # ours exactly as the scenarios run it: raw features, own scaler, 400 rows
        cal = calibration_rows(Xtr_raw, ytr, seed=seed)
        lr = online_models()["Online logistic regression (River)"]
        runs.setdefault(OURS, []).append(evaluate_online(
            OURS, lr, Xtr_raw[cal], ytr[cal], Xte_raw, yte, extra_raw, names, seed))
        log(f"  [{rep}] {OURS:42s} F1={runs[OURS][-1]['f1']:.4f}")

    rows = []
    for name, rs in runs.items():
        row = {"model": name, "family": rs[0]["family"], "repeats": len(rs)}
        for k in rs[0]:
            if isinstance(rs[0][k], float):
                vals = [r[k] for r in rs]
                row[k] = float(np.mean(vals))
                row[k + "_std"] = float(np.std(vals))
        rows.append(row)
    rows.sort(key=lambda r: -r["f1"])
    return {"source": source, "protocol": {
        "split": f"temporal, first {TRAIN_FRACTION:.0%} of each file/class trains",
        "preprocessing": "signed log + z-score fit on the training split",
        "repeats": repeats, "online_train_rows": online_train_cap,
        "train_rows": len(Xtr), "test_rows": len(Xte),
        "train_attack": int(ytr.sum()), "test_attack": int(yte.sum()),
        "features": len(names)}, "provenance": provenance, "models": rows}


# ------------------------------------------------------------------ figure
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
FAMILY_COLOR = {"batch": "#5598e7", "online": "#2ca02c", "unsupervised": "#9aa0a6"}
OURS_COLOR = "#9467bd"


def plot(result, path):
    import matplotlib.pyplot as plt
    rows = result["models"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    order = sorted(rows, key=lambda r: r["f1"])
    ys = np.arange(len(order))
    cols = [OURS_COLOR if r["model"] == OURS else FAMILY_COLOR[r["family"]] for r in order]
    ax.barh(ys, [r["f1"] for r in order], xerr=[r.get("f1_std", 0) for r in order],
            color=cols, height=0.62, error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_yticks(ys, [r["model"] for r in order], fontsize=8)
    for y, r in zip(ys, order):
        ax.text(min(r["f1"], 1) + 0.005, y, f"{r['f1']:.3f}", va="center", fontsize=7,
                color=INK)
    lo = max(0.0, min(r["f1"] for r in rows) - 0.05)
    ax.set_xlim(lo, 1.02)
    ax.set_xlabel("F1 on the later, held-out traffic (attack = positive)", color=MUTED,
                  fontsize=9)
    ax.set_title("Detection quality", loc="left", fontsize=11, color=INK)

    ax = axes[1]
    for r in rows:
        c = OURS_COLOR if r["model"] == OURS else FAMILY_COLOR[r["family"]]
        ax.scatter(r["single_latency_us"], r["f1"], s=30 + 6 * np.sqrt(r["model_kb"]),
                   color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("latency per record, microseconds (log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("F1", color=MUTED, fontsize=9)
    ax.set_title("Efficiency: up and left is better (size = model size)", loc="left",
                 fontsize=11, color=INK)
    for a in axes:
        a.grid(axis="x" if a is axes[0] else "both", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=FAMILY_COLOR["batch"], label="batch (scikit-learn)"),
                            Patch(color=FAMILY_COLOR["online"], label="online (River)"),
                            Patch(color=FAMILY_COLOR["unsupervised"],
                                  label="unsupervised (benign only)"),
                            Patch(color=OURS_COLOR, label="ours, as deployed")],
                   frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    name = {"real": "N-BaIoT", "iot23": "IoT-23"}.get(result["source"], result["source"])
    fig.suptitle(f"Detector benchmark on {name}: temporal train/test split", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SPLITS), default="real")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    result = run(args.source, repeats=args.repeats)
    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.join(RESULTS, f"benchmark_{args.source}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", stem + ".json")
    plot(result, stem + ".png")


if __name__ == "__main__":
    main()
