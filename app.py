r"""
Trust-Aware IDS — Review-2 demo frontend.

Run:  .\.venv\Scripts\streamlit.exe run app.py

Two views:
  LIVE STREAM — replays the traffic stream one round at a time, showing
                classification, drift detection and the gate's authorize/veto
                decision as they happen. This is the demo.
  SUMMARY     — the final numbers, for the report.

Loads precomputed logs instantly; the sidebar can re-run the pipeline live with
different gate parameters, so you can watch the reputation veto be the only
thing standing between the IDS and a successful poisoning attack.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

import charts
import equations

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full"]
LABELS = {
    "static": "Never learns",
    "ungated": "Learns from everything",
    "gate_traffic_only": "Checks traffic only",
    "gate_full": "Our method (traffic + reputation)",
}
SHORT = LABELS
COLORS = {"static": "#888888", "ungated": "#d62728",
          "gate_traffic_only": "#ff7f0e", "gate_full": "#2ca02c"}

EVIDENCE_PLAIN = [
    ("attribution", "Same direction as changes we already approved?",
     "1.0 = yes  ·  0.5 = nothing approved yet  ·  0.0 = opposite direction"),
    ("topology", "Traffic doing lots of different things?",
     "High = looks like a real device  ·  Low = same thing over and over"),
    ("temporal", "Changing smoothly rather than jumping?",
     "High = gradual. A patient attacker gets a high score here for free."),
]

PHASE_HELP = {
    "calib_benign": "training",
    "calib_attack": "training",
    "pre_attack": "Step 1 - normal traffic, nothing happening",
    "slow_poison": "Step 2 - ATTACKER SNEAKING IN, nudging traffic to look normal",
    "post_poison": "Step 3 - the attack now looks completely normal",
    "recovery_check": "Step 4 - AN OBVIOUS ATTACK IS BACK. This step decides everything.",
    "pre_drift": "Step 1 - normal traffic from device A",
    "genuine_drift": "Step 2 - slowly changing to a new normal",
    "post_drift": "Step 3 - settled into the new normal (device B)",
}

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


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Configuration")
scenario = st.sidebar.radio(
    "Scenario", ["slow_poisoning", "genuine_drift"],
    format_func=lambda s: {"slow_poisoning": "2 — Slow poisoning (adversarial)",
                           "genuine_drift": "1 — Genuine drift (benign)"}[s])
tag = st.sidebar.radio("Data source", ["real", "synth"],
                       format_func=lambda t: {"real": "Real N-BaIoT",
                                              "synth": "Synthetic"}[t])

st.sidebar.divider()
live = st.sidebar.toggle(
    "Live re-run with custom gate", value=False,
    help="Recompute the whole pipeline. Slow on real data (~1 min first time), then cached.")
trust_threshold = st.sidebar.slider("Trust threshold", 0.0, 1.0, 0.55, 0.01, disabled=not live)
veto_threshold = st.sidebar.slider("Reputation veto threshold", 0.0, 1.0, 0.30, 0.01, disabled=not live)
w_att = st.sidebar.slider("weight: attribution", 0.0, 1.0, 0.40, 0.05, disabled=not live)
w_topo = st.sidebar.slider("weight: topology", 0.0, 1.0, 0.30, 0.05, disabled=not live)
w_temp = st.sidebar.slider("weight: temporal", 0.0, 1.0, 0.30, 0.05, disabled=not live)

if live:
    log = run_live(scenario, tag, trust_threshold, veto_threshold, w_att, w_topo, w_temp)
else:
    log = load_log(scenario, tag)

THRESH = trust_threshold if live else 0.55
VETO = veto_threshold if live else 0.30

st.title("Trust-Aware IDS — authorization gate for continual learning")

if log is None:
    st.error(f"No log for `{scenario}` / `{tag}`. Run "
             "`python experiments/run_scenarios.py` first, or enable live re-run.")
    st.stop()

N = len(log["static"])
ROUND_SIZE = 20

# reset the playhead whenever the underlying run changes
run_key = f"{scenario}|{tag}|{live}|{THRESH}|{VETO}|{w_att}|{w_topo}|{w_temp}"
if st.session_state.get("run_key") != run_key:
    st.session_state.run_key = run_key
    st.session_state.t = 0
    st.session_state.playing = False

tab_live, tab_summary = st.tabs(["Live stream", "Summary"])

# ================================================================ LIVE STREAM
with tab_live:
    ctl = st.columns([1, 1, 1, 1, 3])
    if ctl[0].button("Play", width='stretch', type="primary"):
        if st.session_state.t >= N - 1:
            st.session_state.t = 0
        st.session_state.playing = True
    if ctl[1].button("Pause", width='stretch'):
        st.session_state.playing = False
    if ctl[2].button("Step", width='stretch'):
        st.session_state.playing = False
        st.session_state.t = min(N - 1, st.session_state.t + 1)
    if ctl[3].button("Reset", width='stretch'):
        st.session_state.playing = False
        st.session_state.t = 0
    speed = ctl[4].select_slider("Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"],
                                 value="1x", label_visibility="collapsed")
    delay = {"0.25x": 0.8, "0.5x": 0.4, "1x": 0.2, "2x": 0.1, "4x": 0.04}[speed]

    t = st.slider("Stream position (round)", 0, N - 1, st.session_state.t,
                  disabled=st.session_state.playing)
    if not st.session_state.playing and t != st.session_state.t:
        st.session_state.t = t
    t = st.session_state.t

    now = {a: log[a][t] for a in ARMS}
    phase = now["static"].get("phase", "?")
    rec_lo = t * ROUND_SIZE
    rec_hi = rec_lo + ROUND_SIZE

    st.progress((t + 1) / N, text=f"round {t + 1} / {N}  ·  records {rec_lo}–{rec_hi}")

    # --- what is on the wire right now ---
    is_attack_phase = phase in ("slow_poison", "post_poison", "recovery_check")
    banner = st.error if is_attack_phase else st.info
    banner(f"**t = round {t + 1}**  ·  phase `{phase}` — "
           f"{PHASE_HELP.get(phase, phase)}  ·  "
           f"reputation of this window: **{now['static'].get('reputation', float('nan')):.2f}**")

    # --- live classification, this window ---
    st.subheader("Is each system spotting the attack right now?")
    cs = st.columns(4)
    for col, arm in zip(cs, ARMS):
        acc = now[arm]["round_accuracy"]
        ncorrect = int(round(acc * ROUND_SIZE))
        with col:
            st.markdown(f"**{SHORT[arm]}**")
            st.progress(acc, text=f"{ncorrect}/{ROUND_SIZE} correct this window")
            st.metric("right so far", f"{now[arm]['accuracy']:.3f}",
                      delta=(f"{now[arm]['accuracy'] - log[arm][t - 1]['accuracy']:+.3f}"
                             if t > 0 else None))

    # --- the gate, right now ---
    st.subheader("Should the system be allowed to learn from this traffic?")
    gnow = now["gate_full"]
    tnow = now["gate_traffic_only"]
    if not gnow.get("drift_active"):
        st.caption("No drift signalled — no update proposed this round. Nobody learns.")
    else:
        gc = st.columns([2, 2, 3])
        with gc[0]:
            st.markdown("**Checks on the traffic** - an attacker can fake all three")
            for k, plain, tip in EVIDENCE_PLAIN:
                v = gnow.get(k)
                if v is not None:
                    st.progress(min(1.0, max(0.0, v)), text=f"{plain}  —  {v:.3f}")
                    st.caption(tip)
            ts = gnow.get("trust_score")
            if ts is not None:
                st.markdown(f"trust score **{ts:.3f}** vs threshold {THRESH:.2f}")
        with gc[1]:
            st.markdown("**Check from outside** - an attacker cannot fake this")
            rep = gnow.get("reputation", float("nan"))
            st.metric("Where is this traffic going?", f"{rep:.2f}")
            st.caption(f"1.0 = known-good destination · 0.05 = known C2 infrastructure. "
                       f"Refuse outright below {VETO:.2f}.")
        with gc[2]:
            st.markdown("**What each one decided**")
            if tnow.get("authorized"):
                st.warning("Traffic-only gate: **AUTHORIZED** — evidence looked fine")
            else:
                st.info("Traffic-only gate: refused")
            if gnow.get("authorized"):
                st.success("Full gate: **AUTHORIZED** — model updates")
            else:
                st.error(f"Full gate: **REFUSED** — {gnow.get('reason', '')}")

    # --- accuracy so far ---
    st.subheader("How often each system has been right")
    st.altair_chart(charts.accuracy_chart(log, upto=t, n_rounds=N, height=340))
    st.caption("Hover anywhere to read every line at that round. Shaded stripes are the "
               "four steps. Triangles mark where a system accepted an update.")

    # --- event feed ---
    st.subheader("Event log")
    events = []
    prev_phase, prev_drift = None, False
    for i in range(t + 1):
        r = log["static"][i]
        g = log["gate_full"][i]
        tr = log["gate_traffic_only"][i]
        ph = r.get("phase")
        if ph != prev_phase:
            events.append((i + 1, "PHASE", f"`{ph}` — {PHASE_HELP.get(ph, ph)}"))
            prev_phase = ph
        if r.get("drift_active") and not prev_drift:
            events.append((i + 1, "DRIFT", "ADWIN change point — updates now being proposed"))
        prev_drift = bool(r.get("drift_active"))
        if g.get("drift_active"):
            if tr.get("authorized") and not g.get("authorized"):
                events.append((i + 1, "VETO",
                               f"traffic-only ACCEPTED this update; full gate refused — "
                               f"{g.get('reason', '')}"))
            elif g.get("authorized"):
                events.append((i + 1, "AUTH", f"full gate authorized — {g.get('reason', '')}"))
    icon = {"PHASE": "•", "DRIFT": "⚡", "VETO": "🛑", "AUTH": "✅"}
    body = "\n".join(f"- `r{rd:>3}` {icon.get(k, '')} **{k}** — {msg}"
                     for rd, k, msg in reversed(events[-14:]))
    st.markdown(body if body else "_no events yet_")

    if st.session_state.playing:
        if t < N - 1:
            time.sleep(delay)
            st.session_state.t += 1
            st.rerun()
        else:
            st.session_state.playing = False
            st.balloons()

# ==================================================================== SUMMARY
with tab_summary:
    n_drift_rounds = sum(1 for r in log["static"] if r.get("drift_active"))
    if n_drift_rounds == 0:
        st.warning(
            "**No drift was detected in this scenario, so no update was ever proposed and "
            "all four arms are identical.** This is a real negative result, not a bug: ADWIN "
            "watches the frozen arm's *error rate*, and this benign-to-benign shift never "
            "crosses the decision boundary — the static model still scores ~0.99, so nothing "
            "needed adapting.")

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
            st.metric("Right overall", f"{final['accuracy']:.3f}")
            if recovery[arm] is not None:
                st.metric("Caught the obvious attack", f"{recovery[arm]:.3f}",
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

    st.subheader("How often each system was right")
    st.altair_chart(charts.accuracy_chart(log, n_rounds=N, height=400))
    st.caption("Hover anywhere to read every line at that round. Shaded stripes are the "
               "four steps. Triangles mark where a system accepted an update - watch "
               "what happens to the orange line shortly after it accepts one.")

    gate_rounds = [r for r in log["gate_full"] if r.get("drift_active")]
    if gate_rounds:
        st.subheader("Traffic score vs reputation, side by side")
        gc = charts.gate_chart(log, THRESH, VETO, height=340)
        if gc is not None:
            st.altair_chart(gc)
            st.caption("Orange crosses are the moments that matter: the traffic-only system said "
                       "YES and ours said NO. Both saw the same healthy traffic score. Only "
                       "the reputation line told them apart.")

        st.subheader("The three traffic checks, separately")
        ec = charts.evidence_chart(log, height=300)
        if ec is not None:
            st.altair_chart(ec)
            st.caption("An attacker can push all three of these up. Added together they make the "
                       "traffic score in the chart above.")

        tbl = pd.DataFrame([{
            "round": r["round"], "phase": r.get("phase"),
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
            st.dataframe(tbl, width='stretch', hide_index=True)

    st.divider()
    equations.render(w_att, w_topo, w_temp, THRESH, VETO)

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
            "**Also outstanding:** no hyperparameter tuning; no EWC/Fisher-freezing baseline; "
            "Scenario 1 currently produces a null result because a benign-to-benign device "
            "shift does not degrade the frozen model enough for error-rate-driven ADWIN to fire.\n\n"
            "**Evidence is computed in a calibrated feature space** (`evidence/featurespace.py`), "
            "fit on calibration benign data only. On raw N-BaIoT values the geometric evidence "
            "signals collapse: `temporal_stability` returns 1.7e-17 and five "
            "`HH_jit_*_variance` columns dominate 100% of every cosine.")
