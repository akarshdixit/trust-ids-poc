r"""
Trust-Aware IDS — Review-2 demo frontend.

Run:  .\.venv\Scripts\streamlit.exe run app.py

Loads precomputed logs instantly; the sidebar can re-run the pipeline live with
different gate parameters, which is the point of the demo — you can watch the
reputation veto be the only thing standing between the IDS and a successful
poisoning attack.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full"]
LABELS = {
    "static": "Static (never updates)",
    "ungated": "Ungated continual learning",
    "gate_traffic_only": "Traffic-only gate (ExpIDS-style)",
    "gate_full": "Proposed: trust gate + reputation veto",
}
COLORS = {"static": "#888888", "ungated": "#d62728",
          "gate_traffic_only": "#ff7f0e", "gate_full": "#2ca02c"}

st.set_page_config(page_title="Trust-Aware IDS", layout="wide")


@st.cache_data(show_spinner=False)
def load_log(scenario, tag):
    path = os.path.join(RESULTS, f"{scenario}_log_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Running pipeline on real N-BaIoT (first run reads ~190 MB of CSV)...")
def run_live(scenario, tag, trust_threshold, veto_threshold, w_att, w_topo, w_temp):
    from experiments.run_scenarios import run_scenario
    if tag == "synth":
        from data.synth_stream import (genuine_drift_stream, slow_poisoning_stream,
                                       FEATURE_NAMES)
        names = FEATURE_NAMES
        kw = {}
    else:
        from data.load_nbaiot import (genuine_drift_stream, slow_poisoning_stream,
                                      feature_names)
        names = feature_names()
        kw = {"benign_b_csv": os.path.join(
            BASE, "data", "nbaiot", "Ennio_Doorbell", "benign_traffic.csv")}

    gp = dict(trust_threshold=trust_threshold,
              reputation_veto_threshold=veto_threshold,
              w_attribution=w_att, w_topology=w_topo, w_temporal=w_temp)
    stream = (slow_poisoning_stream() if scenario == "slow_poisoning"
              else genuine_drift_stream(**kw))
    return run_scenario(stream, names, scenario, gate_params=gp)


st.sidebar.title("Configuration")
scenario = st.sidebar.radio(
    "Scenario",
    ["slow_poisoning", "genuine_drift"],
    format_func=lambda s: {"slow_poisoning": "2 — Slow poisoning (adversarial)",
                           "genuine_drift": "1 — Genuine drift (benign)"}[s])
tag = st.sidebar.radio("Data source", ["real", "synth"],
                       format_func=lambda t: {"real": "Real N-BaIoT",
                                              "synth": "Synthetic"}[t])

st.sidebar.divider()
live = st.sidebar.toggle(
    "Live re-run with custom gate", value=False,
    help="Recompute the whole pipeline. Slow on real data (~1 min first time), then cached.")
trust_threshold = st.sidebar.slider("Trust threshold", 0.0, 1.0, 0.55, 0.01,
                                    disabled=not live)
veto_threshold = st.sidebar.slider("Reputation veto threshold", 0.0, 1.0, 0.30, 0.01,
                                   disabled=not live)
w_att = st.sidebar.slider("weight: attribution", 0.0, 1.0, 0.40, 0.05, disabled=not live)
w_topo = st.sidebar.slider("weight: topology", 0.0, 1.0, 0.30, 0.05, disabled=not live)
w_temp = st.sidebar.slider("weight: temporal", 0.0, 1.0, 0.30, 0.05, disabled=not live)

if live:
    log = run_live(scenario, tag, trust_threshold, veto_threshold, w_att, w_topo, w_temp)
else:
    log = load_log(scenario, tag)

st.title("Trust-Aware IDS — authorization gate for continual learning")
st.caption("An IDS that keeps learning can be taught the wrong thing. This gate decides "
           "whether a proposed update is *authorized*, using traffic-derived evidence "
           "cross-checked against an independent out-of-band reputation channel.")

if log is None:
    st.error(f"No log for `{scenario}` / `{tag}`. Run "
             "`python experiments/run_scenarios.py` first, or enable live re-run.")
    st.stop()

n_drift_rounds = sum(1 for r in log["static"] if r.get("drift_active"))

if n_drift_rounds == 0:
    st.warning(
        "**No drift was detected in this scenario, so no update was ever proposed and "
        "all four arms are identical.** This is a real negative result, not a bug: ADWIN "
        "watches the frozen arm's *error rate*, and this benign-to-benign shift never "
        "crosses the decision boundary — the static model still scores ~0.99, so nothing "
        "needed adapting. See the caveats at the bottom.")

st.subheader("Outcome")
recovery = {}
for arm in ARMS:
    sel = [r["round_accuracy"] for r in log[arm] if r.get("phase") == "recovery_check"]
    recovery[arm] = float(np.mean(sel)) if sel else None

cols = st.columns(4)
for col, arm in zip(cols, ARMS):
    final = log[arm][-1]
    n_auth = sum(1 for r in log[arm] if r.get("authorized"))
    with col:
        st.markdown(f"**{LABELS[arm]}**")
        st.metric("Overall accuracy", f"{final['accuracy']:.3f}")
        if recovery[arm] is not None:
            st.metric("Repeat-attack detection", f"{recovery[arm]:.3f}",
                      help="Accuracy on an obvious, full-strength repeat of the attack "
                           "the model was calibrated on — measures lasting damage from "
                           "unauthorized updates.")
        st.caption(f"{n_auth}/{len(log[arm])} rounds authorized")

if scenario == "slow_poisoning" and recovery.get("gate_full") is not None:
    gf, tf = recovery["gate_full"], recovery["gate_traffic_only"]
    if gf > tf:
        st.success(
            f"**The result.** After the poisoning campaign, the traffic-only gate detects "
            f"the obvious repeat attack {tf:.0%} of the time — it was fooled and "
            f"permanently damaged. The full gate, which additionally consults the "
            f"out-of-band reputation channel, still detects it {gf:.0%} of the time. "
            f"Traffic evidence alone was not enough.")

st.subheader("Detection accuracy per round")
df = pd.DataFrame({LABELS[a]: [r["round_accuracy"] for r in log[a]] for a in ARMS},
                  index=[r["round"] for r in log[ARMS[0]]])
df.index.name = "round"
st.line_chart(df, color=[COLORS[a] for a in ARMS], height=340)

phases = []
last = None
for r in log["static"]:
    if r.get("phase") != last:
        phases.append((r["round"], r.get("phase")))
        last = r.get("phase")
st.caption("Phases: " + "  |  ".join(f"round {rd} `{ph}`" for rd, ph in phases))

st.subheader("Gate decision trace")
gate_rounds = [r for r in log["gate_full"] if r.get("drift_active")]
if not gate_rounds:
    st.info("No drift-active rounds — the gate was never asked to decide anything.")
else:
    trace = pd.DataFrame([{
        "round": r["round"],
        "trust score": r.get("trust_score"),
        "reputation": r.get("reputation"),
        "veto threshold": veto_threshold if live else 0.30,
    } for r in gate_rounds]).set_index("round")
    st.line_chart(trace, height=300, color=["#8b1a1a", "#2ca02c", "#bbbbbb"])
    st.caption("When the reputation line sits below the veto threshold, the full gate "
               "refuses regardless of how healthy the trust score looks. That gap is the "
               "mechanism.")

    tbl = pd.DataFrame([{
        "round": r["round"],
        "phase": r.get("phase"),
        "attribution": round(r.get("attribution", float("nan")), 3),
        "topology": round(r.get("topology", float("nan")), 3),
        "temporal": round(r.get("temporal", float("nan")), 3),
        "trust": round(r.get("trust_score", float("nan")), 3),
        "reputation": round(r.get("reputation", float("nan")), 3),
        "traffic-only authorized": next(
            (g["authorized"] for g in log["gate_traffic_only"]
             if g["round"] == r["round"]), None),
        "full gate authorized": r["authorized"],
        "reason (full gate)": r.get("reason", ""),
    } for r in gate_rounds])
    with st.expander(f"Round-by-round decisions ({len(tbl)} drift-active rounds)"):
        st.dataframe(tbl, use_container_width=True, hide_index=True)

st.divider()
with st.expander("What this demonstrates — and what it does not"):
    st.markdown(
        "**The out-of-band reputation signal is simulated.** N-BaIoT carries no "
        "destination-reputation or C2-cost field, so it is keyed here to ground-truth "
        "traffic origin (benign 1.0, attack 0.05). It is *not* a reputation detector — it "
        "derives nothing from the traffic features, R(x) is not a function of x, which is "
        "what makes it a legitimate independent channel in principle. But in this "
        "proof-of-concept it is closer to an oracle than a real threat-intel feed.\n\n"
        "**So the honest claim is:** this demonstrates what an authorization mechanism does "
        "*once it has an independent signal* — not that the gate detects the attacker. "
        "Integrating a real reputation / C2-cost source is Project-II work.\n\n"
        "**Also outstanding:** no hyperparameter tuning (thresholds and weights are untuned "
        "defaults); no EWC/Fisher-freezing baseline; Scenario 1 currently produces a null "
        "result because a benign-to-benign device shift does not degrade the frozen model "
        "enough for error-rate-driven ADWIN to fire.\n\n"
        "**Evidence is computed in a calibrated feature space** (`evidence/featurespace.py`), "
        "fit on calibration benign data only. On raw N-BaIoT values the geometric evidence "
        "signals collapse: `temporal_stability` returns 1.7e-17 and five "
        "`HH_jit_*_variance` columns dominate 100% of every cosine.")
