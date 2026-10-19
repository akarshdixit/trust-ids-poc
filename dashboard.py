"""
Comparison dashboards for app.py: other models vs ours, data and features,
and the live experiment. Everything is read from results/*.json, which the
experiment scripts write:

  results/benchmark_<source>.json      experiments/benchmark.py
  results/sweep_models_<source>.json   experiments/sweeps.py models
  results/data_profile_<source>.json   experiments/data_profile.py
  results/live_experiment_iot23.json   experiments/live_experiment.py
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import charts
from data.features import catalogue

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OURS = "Online LR, 400 calibration rows (ours, as deployed)"
FAMILY = {"batch": "batch (scikit-learn)", "online": "online (River)",
          "unsupervised": "unsupervised (benign only)", "ours": "ours, as deployed"}
FAMILY_COLORS = ["#5598e7", "#2ca02c", "#9aa0a6", "#9467bd"]
DATASET = {"real": "N-BaIoT", "iot23": "IoT-23"}
METRICS = {
    "f1": ("F1 (attack = positive)", True), "accuracy": ("Accuracy", True),
    "balanced_accuracy": ("Balanced accuracy", True), "roc_auc": ("ROC-AUC", True),
    "precision": ("Precision", True), "recall": ("Recall (detection rate)", True),
    "mcc": ("Matthews correlation", True), "fpr": ("False-positive rate", False),
    "unseen_device_fpr": ("False-positive rate, unseen device", False),
    "single_latency_us": ("Latency per record (microseconds)", False),
    "train_seconds": ("Training time (s)", False), "model_kb": ("Model size (KB)", False),
}
DEFENCES = [("static", "never learns"), ("ungated", "learns everything"),
            ("gate_traffic_only", "traffic-only gate"), ("gate_full", "full gate (one-shot)"),
            ("gate_quarantine", "quarantine gate")]


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _missing(what, cmd):
    st.info(f"No {what} yet. Generate it with `{cmd}`.")


# ------------------------------------------------------------------ models
def models_tab(source):
    if source not in DATASET:
        st.info("Model comparison runs on the real datasets: pick N-BaIoT or IoT-23 in the "
                "sidebar.")
        return
    bench = _load(f"benchmark_{source}.json")
    st.markdown(f"### How good is the detector itself? ({DATASET[source]})")
    if bench is None:
        _missing("benchmark", f"python experiments/benchmark.py --source {source}")
    else:
        p = bench["protocol"]
        st.caption(f"{len(bench['models'])} models, same data and preprocessing. {p['split']}; "
                   f"{p['preprocessing']}. Train {p['train_rows']:,} rows "
                   f"({p['train_attack']:,} attack), test {p['test_rows']:,} later rows "
                   f"({p['test_attack']:,} attack). Mean of {p['repeats']} repeats.")
        df = pd.DataFrame(bench["models"])
        df["family"] = [("ours" if m == OURS else f) for m, f in zip(df.model, df.family)]
        df["group"] = df.family.map(FAMILY)
        keys = [k for k in METRICS if k in df]
        metric = st.selectbox("Metric", keys, format_func=lambda k: METRICS[k][0],
                              key=f"bench_metric_{source}")
        label, higher = METRICS[metric]
        d = df.dropna(subset=[metric])
        color = alt.Color("group:N", title=None, scale=alt.Scale(
            domain=list(FAMILY.values()), range=FAMILY_COLORS),
            legend=alt.Legend(orient="bottom"))
        bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X(f"{metric}:Q", title=label,
                    scale=alt.Scale(type="log") if metric in ("single_latency_us",
                                                              "train_seconds", "model_kb")
                    else alt.Scale(zero=False)),
            y=alt.Y("model:N", sort=alt.EncodingSortField(metric,
                                                          order="descending" if higher
                                                          else "ascending"), title=None),
            color=color,
            tooltip=["model", "group", alt.Tooltip(f"{metric}:Q", format=".4f")])
        st.altair_chart(bars.properties(height=28 * len(d) + 40), width="stretch")
        st.caption(("Higher is better." if higher else "Lower is better.")
                   + " Purple is the online logistic regression the trust gate wraps, trained "
                     "exactly as in the scenarios: on 400 calibration rows only.")

        st.markdown("**Efficiency: detection quality vs cost**")
        sc = alt.Chart(df).mark_circle(opacity=0.9, stroke="white", strokeWidth=1.5).encode(
            x=alt.X("single_latency_us:Q", scale=alt.Scale(type="log"),
                    title="latency per record, microseconds (log)"),
            y=alt.Y("f1:Q", scale=alt.Scale(zero=False), title="F1"),
            size=alt.Size("model_kb:Q", title="model size (KB)", scale=alt.Scale(range=[40, 900]),
                          legend=None),
            color=color,
            tooltip=["model", alt.Tooltip("f1:Q", format=".4f"),
                     alt.Tooltip("single_latency_us:Q", format=".1f"),
                     alt.Tooltip("model_kb:Q", format=".0f"),
                     alt.Tooltip("train_seconds:Q", format=".2f")])
        st.altair_chart(sc.properties(height=320), width="stretch")
        st.caption("Up and to the left is better; bubble size is model size.")
        cols = ["model", "group"] + [k for k in ("accuracy", "f1", "precision", "recall",
                                                 "roc_auc", "fpr", "unseen_device_fpr",
                                                 "single_latency_us", "train_seconds",
                                                 "model_kb") if k in df]
        with st.expander("All benchmark numbers"):
            st.dataframe(df[cols].round(4), hide_index=True, width="stretch")

    st.markdown(f"### Which defence survives poisoning, on every model? ({DATASET[source]})")
    sw = _load(f"sweep_models_{source}.json")
    if sw is None:
        _missing("model sweep", f"python experiments/sweeps.py models --source {source}")
        return
    from models.online_ids import BASE_MODEL_NAMES
    rows = []
    for r in sw:
        for scenario, title in (("slow_poisoning", "Repeat attack after poisoning"),
                                ("genuine_drift", "Benign drift (adaptation)")):
            for arm, lab in DEFENCES:
                rows.append({"base model": BASE_MODEL_NAMES[r["base_model"]],
                             "defence": lab, "scenario": title,
                             "accuracy": r[f"{scenario}|{arm}"]})
    d = pd.DataFrame(rows)
    order = [lab for _, lab in DEFENCES]
    palette = [charts.COLORS[a] for a, _ in DEFENCES]
    ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("defence:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="accuracy, decisive phase"),
        color=alt.Color("defence:N", sort=order, scale=alt.Scale(domain=order, range=palette),
                        legend=alt.Legend(orient="bottom", title=None)),
        column=alt.Column("base model:N", title=None, header=alt.Header(labelFontSize=11)),
        row=alt.Row("scenario:N", title=None, header=alt.Header(labelFontSize=11)),
        tooltip=["base model", "defence", "scenario", alt.Tooltip("accuracy:Q", format=".3f")])
    st.altair_chart(ch.properties(width=110, height=150))
    st.caption("Each column is a different learner inside every arm. A good defence is high in "
               "BOTH rows: it refuses the poisoning (top) and still adapts to benign drift "
               "(bottom, feature-drift trigger on).")
    kept = pd.DataFrame([{"base model": BASE_MODEL_NAMES[r["base_model"]],
                          **{lab: r[f"slow_poisoning|{a}|poison_kept"] for a, lab in DEFENCES}}
                         for r in sw])
    with st.expander("Poisoned updates left in each model (lower is better)"):
        st.dataframe(kept, hide_index=True, width="stretch")


# ------------------------------------------------------------------ data
def data_tab(source):
    if source not in DATASET:
        st.info("Data profiles exist for the real datasets: pick N-BaIoT or IoT-23.")
        return
    p = _load(f"data_profile_{source}.json")
    if p is None:
        _missing("data profile", f"python experiments/data_profile.py --source {source}")
        return
    st.markdown(f"### How the data is processed ({DATASET[source]})")
    st.markdown(
        "1. **Load** the files (CSV for N-BaIoT; Zeek `conn.log.labeled` for IoT-23, parsed "
        "into 21 per-flow features).\n"
        "2. **Check** for missing or infinite values and column order.\n"
        "3. **Signed log** `sign(x)·log(1+|x|)`: squeezes values spanning many orders of "
        "magnitude while keeping the sign of correlation features.\n"
        "4. **Z-score** with mean and standard deviation fitted on training (calibration) "
        "data only, then frozen, so live traffic can never move the reference frame.\n"
        "5. **Split in time**: train on earlier traffic, test on later traffic.\n"
        "6. **Scenario streams**: 400 labelled calibration records, then 20-record rounds.")
    q = pd.DataFrame([{"file": k, **v} for k, v in p["raw_quality"].items()])
    if "attack_types" in q:
        q["attack_types"] = q["attack_types"].astype(str)
    st.dataframe(q, hide_index=True, width="stretch")
    sp = pd.DataFrame([{"step": s, **p["spread"][k]} for s, k in
                       (("1 raw", "raw"), ("2 signed log", "signed_log"),
                        ("3 signed log + z-score", "z_score"))])
    st.markdown("**What each processing step does to the scale of the features**")
    base = alt.Chart(sp).encode(y=alt.Y("step:N", title=None))
    st.altair_chart((base.mark_rule(strokeWidth=8, color="#5598e7", opacity=0.7).encode(
        x=alt.X("std_min:Q", scale=alt.Scale(type="log"), title="per-feature standard deviation "
                "(range, log scale)"), x2="std_max:Q",
        tooltip=["step", alt.Tooltip("std_min:Q", format=".3g"),
                 alt.Tooltip("std_max:Q", format=".3g"),
                 alt.Tooltip("orders_of_magnitude:Q", format=".1f")])
        + base.mark_point(color="#1f1f1e", filled=True).encode(x="std_median:Q"))
        .properties(height=140), width="stretch")
    st.caption(f"Raw features span {p['spread']['raw']['orders_of_magnitude']:.1f} orders of "
               f"magnitude; after processing {p['spread']['z_score']['orders_of_magnitude']:.1f}.")

    st.markdown("### Which features matter")
    top = pd.DataFrame(p["top_features"])
    st.altair_chart(alt.Chart(top).mark_bar(color="#256abf", cornerRadiusEnd=3).encode(
        x=alt.X("importance:Q", title="random-forest importance"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", "meaning", alt.Tooltip("importance:Q", format=".3f"),
                 alt.Tooltip("benign_mean_z:Q", format=".2f"),
                 alt.Tooltip("attack_mean_z:Q", format=".2f")])
        .properties(height=24 * len(top) + 30), width="stretch")
    st.dataframe(top[["feature", "meaning", "importance", "benign_mean_z", "attack_mean_z"]]
                 .round(3), hide_index=True, width="stretch")
    if "importance_grid" in p:
        g = pd.DataFrame([{"aggregation": k.split("|")[0], "window": k.split("|")[1], "importance": v}
                          for k, v in p["importance_grid"].items()])
        st.markdown("**Importance by traffic aggregation and time window**")
        st.altair_chart(alt.Chart(g).mark_rect().encode(
            x=alt.X("window:N", sort=["L5", "L3", "L1", "L0.1", "L0.01"],
                    title="decay window (L5 ~100 ms ... L0.01 ~1 min)"),
            y=alt.Y("aggregation:N", title=None),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues"), title=None),
            tooltip=["aggregation", "window", alt.Tooltip("importance:Q", format=".3f")])
            .properties(height=200), width="stretch")
    with st.expander(f"Every feature and what it means ({p['feature_count']})"):
        if source == "real":
            from data import load_nbaiot as nb
            names = nb.feature_names()
        else:
            from data import load_iot23 as io
            names = io.feature_names()
        st.dataframe(pd.DataFrame(catalogue(names), columns=[
            "feature", "aggregation", "window", "statistic", "meaning"]),
            hide_index=True, width="stretch")
    with st.expander("Scenario stream layout"):
        for sc, phases in p["stream_layout"].items():
            st.markdown(f"**{sc}**")
            st.dataframe(pd.DataFrame(phases), hide_index=True, width="stretch")


# ------------------------------------------------------------------ live
def live_tab(backend_url):
    st.markdown("### Live experiment: a real capture, replayed in real time")
    st.markdown(
        f"Start a run on the live dashboard: **[{backend_url}/live]({backend_url}/live)** "
        "(needs the backend). Real Zeek sensors can stream into "
        f"`{backend_url}/api/live/ingest` with `python experiments/live_feed.py <conn.log> "
        "--follow`. Below: the last recorded run.")
    d = _load("live_experiment_iot23.json")
    if d is None:
        _missing("recorded live run", "python experiments/live_experiment.py --speed 600")
        return
    s, rounds = d["summary"], d["rounds"]
    c = st.columns(5)
    c[0].metric("live flows", f"{s['live_flows']:,}")
    c[1].metric("capture covered", f"{s['capture_hours']:.1f} h")
    c[2].metric("replay speed", f"{s['speed']:g}x")
    c[3].metric("sustained throughput", f"{s['throughput_flows_per_s']:,.0f} flows/s")
    c[4].metric("max backlog", f"{s['max_lag_s']:.2f} s")
    st.caption(f"Processing per flow, all six systems together: median "
               f"{s['processing_ms_per_flow_all_arms']['p50']:.2f} ms, 95th percentile "
               f"{s['processing_ms_per_flow_all_arms']['p95']:.2f} ms.")
    rows = []
    for r in rounds:
        for a, v in r["arms"].items():
            rows.append({"round": r["round"], "system": charts.LABELS.get(a, a),
                         "accuracy": (v["tp"] + v["tn"]) / 20})
    df = pd.DataFrame(rows)
    df["rolling"] = df.groupby("system")["accuracy"].transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    order = [charts.LABELS[a] for a in charts.ARMS if charts.LABELS[a] in set(df.system)]
    st.altair_chart(alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("round:Q", title="round (20 live flows)"),
        y=alt.Y("rolling:Q", scale=alt.Scale(domain=[0, 1.02]), title="accuracy, 10-round mean"),
        color=alt.Color("system:N", sort=order, scale=alt.Scale(
            domain=order, range=[charts.COLORS[a] for a in charts.ARMS
                                 if charts.LABELS[a] in order]),
            legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["round", "system", alt.Tooltip("rolling:Q", format=".3f")])
        .properties(height=300), width="stretch")
    t = pd.DataFrame([{"system": charts.LABELS.get(a, a), **{k: v[k] for k in (
        "accuracy", "precision", "recall", "f1", "fpr", "updates_applied")}}
        for a, v in s["arms"].items()])
    st.dataframe(t.round(4), hide_index=True, width="stretch")
