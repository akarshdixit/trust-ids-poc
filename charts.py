"""
Altair chart builders for the demo frontend.

Altair rather than st.line_chart because the built-in chart gives no real
legend, no axis titles, no phase context and no way to mark which rounds the
gate actually authorized — all of which are the point of these figures.
Altair also inherits Streamlit's light/dark theme, so the figures stay legible
either way.
"""
import altair as alt
import pandas as pd

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full"]
LABELS = {
    "static": "Static — never updates",
    "ungated": "Ungated — accepts every update",
    "gate_traffic_only": "Traffic-only gate (ExpIDS-style)",
    "gate_full": "Proposed — trust gate + reputation veto",
}
COLORS = {
    "static": "#9aa0a6",
    "ungated": "#d62728",
    "gate_traffic_only": "#ff7f0e",
    "gate_full": "#2ca02c",
}
ARM_ORDER = [LABELS[a] for a in ARMS]
ARM_COLORS = [COLORS[a] for a in ARMS]

PHASE_SHORT = {
    "pre_attack": "benign",
    "slow_poison": "poisoning",
    "post_poison": "attack looks benign",
    "recovery_check": "repeat attack",
    "pre_drift": "benign (device A)",
    "genuine_drift": "drifting",
    "post_drift": "benign (device B)",
}


def phase_bands(log, n_rounds):
    """Contiguous [start, end) round ranges per phase, for background shading."""
    rows, start, cur = [], None, None
    for r in log["static"]:
        ph = r.get("phase")
        if ph != cur:
            if cur is not None:
                rows.append({"phase": cur, "start": start, "end": r["round"]})
            cur, start = ph, r["round"]
    if cur is not None:
        rows.append({"phase": cur, "start": start, "end": n_rounds + 1})
    for i, row in enumerate(rows):
        row["mid"] = (row["start"] + row["end"]) / 2
        row["shade"] = i % 2
        row["label"] = PHASE_SHORT.get(row["phase"], row["phase"])
    return pd.DataFrame(rows)


def _band_layers(bands, y_label_pos=1.06):
    shade = alt.Chart(bands).mark_rect(opacity=0.07).encode(
        x=alt.X("start:Q"), x2="end:Q",
        color=alt.Color("shade:N", scale=alt.Scale(domain=[0, 1],
                                                   range=["#7f7f7f", "#000000"]),
                        legend=None),
    )
    text = alt.Chart(bands).mark_text(
        fontSize=10, opacity=0.75, baseline="bottom", fontStyle="italic",
    ).encode(
        x=alt.X("mid:Q"), y=alt.datum(y_label_pos), text="label:N",
    )
    rules = alt.Chart(bands).mark_rule(
        strokeDash=[2, 3], opacity=0.35, strokeWidth=1,
    ).encode(x=alt.X("start:Q"))
    return shade, text, rules


def accuracy_chart(log, upto=None, n_rounds=None, height=400):
    """Windowed detection accuracy per round, all four arms, with phase context."""
    n_rounds = n_rounds or len(log["static"])
    cut = len(log["static"]) if upto is None else upto + 1

    rows = []
    for a in ARMS:
        for r in log[a][:cut]:
            rows.append({
                "round": r["round"],
                "accuracy": r["round_accuracy"],
                "Arm": LABELS[a],
                "phase": r.get("phase", ""),
                "authorized": bool(r.get("authorized")),
                "cumulative": r.get("accuracy"),
            })
    df = pd.DataFrame(rows)
    bands = phase_bands(log, n_rounds)
    shade, text, rules = _band_layers(bands)

    x = alt.X("round:Q",
              title="Adaptation round  (each round = 20 records, test-then-train)",
              scale=alt.Scale(domain=[1, n_rounds], nice=False))
    color = alt.Color("Arm:N",
                      scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLORS),
                      legend=alt.Legend(title="Arm", orient="top", columns=2,
                                        labelLimit=420, symbolStrokeWidth=3))

    line = alt.Chart(df).mark_line(strokeWidth=2.4, interpolate="monotone").encode(
        x=x,
        y=alt.Y("accuracy:Q",
                title="Windowed detection accuracy  (this round only)",
                scale=alt.Scale(domain=[-0.04, 1.04])),
        color=color,
        tooltip=[alt.Tooltip("round:Q", title="round"),
                 alt.Tooltip("Arm:N", title="arm"),
                 alt.Tooltip("accuracy:Q", title="this round", format=".2f"),
                 alt.Tooltip("cumulative:Q", title="running accuracy", format=".3f"),
                 alt.Tooltip("phase:N", title="phase")],
    )

    # mark the rounds where an arm actually accepted an update
    auth = alt.Chart(df[df["authorized"]]).mark_point(
        size=42, filled=True, opacity=0.9, shape="triangle-down",
    ).encode(x=x, y=alt.Y("accuracy:Q"), color=color,
             tooltip=[alt.Tooltip("round:Q", title="round"),
                      alt.Tooltip("Arm:N", title="arm accepted an update here")])

    return (shade + rules + text + line + auth).properties(
        height=height, width="container").resolve_scale(color="independent")


def gate_chart(log, thresh, veto, height=340):
    """Trust score vs reputation against their thresholds, on drift-active rounds."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    tro = {g["round"]: g for g in log["gate_traffic_only"]}

    S_T, S_R = "Trust score  T", "Reputation  R"
    S_TT, S_TR = "Trust threshold  τ_T", "Veto threshold  τ_R"
    order = [S_T, S_R, S_TT, S_TR]
    rng = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    dash = [[1, 0], [1, 0], [5, 4], [5, 4]]

    rows = []
    for r in gr:
        rd = r["round"]
        rows += [
            {"round": rd, "Series": S_T, "value": r.get("trust_score"), "phase": r.get("phase", "")},
            {"round": rd, "Series": S_R, "value": r.get("reputation"), "phase": r.get("phase", "")},
            {"round": rd, "Series": S_TT, "value": thresh, "phase": r.get("phase", "")},
            {"round": rd, "Series": S_TR, "value": veto, "phase": r.get("phase", "")},
        ]
    df = pd.DataFrame(rows)

    x = alt.X("round:Q", title="Adaptation round (drift-active rounds only)")
    lines = alt.Chart(df).mark_line(strokeWidth=2.2).encode(
        x=x,
        y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[-0.04, 1.04])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=order, range=rng),
                        legend=alt.Legend(title=None, orient="top", columns=2,
                                          labelLimit=400)),
        strokeDash=alt.StrokeDash("Series:N",
                                  scale=alt.Scale(domain=order, range=dash),
                                  legend=None),
        tooltip=[alt.Tooltip("round:Q", title="round"),
                 alt.Tooltip("Series:N", title="series"),
                 alt.Tooltip("value:Q", title="value", format=".3f"),
                 alt.Tooltip("phase:N", title="phase")],
    )

    # rounds where traffic-only accepted an update that the full gate vetoed
    div = pd.DataFrame([{
        "round": r["round"],
        "value": r.get("trust_score"),
        "phase": r.get("phase", ""),
        "what": "traffic-only ACCEPTED — full gate vetoed",
    } for r in gr
        if tro.get(r["round"], {}).get("authorized") and not r.get("authorized")])

    layers = [lines]
    if not div.empty:
        layers.append(alt.Chart(div).mark_point(
            size=150, shape="cross", filled=True, color="#ff7f0e",
            strokeWidth=2, opacity=0.95,
        ).encode(x=x, y="value:Q",
                 tooltip=[alt.Tooltip("round:Q", title="round"),
                          alt.Tooltip("what:N", title=""),
                          alt.Tooltip("value:Q", title="trust score", format=".3f"),
                          alt.Tooltip("phase:N", title="phase")]))

    return alt.layer(*layers).properties(height=height, width="container")


def evidence_chart(log, height=300):
    """The three traffic-evidence components over drift-active rounds."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    names = {"attribution": "A — attribution consistency",
             "topology": "P — topology / diversity",
             "temporal": "S — temporal stability"}
    rows = []
    for r in gr:
        for k, lab in names.items():
            if r.get(k) is not None:
                rows.append({"round": r["round"], "Signal": lab, "value": r[k],
                             "phase": r.get("phase", "")})
    df = pd.DataFrame(rows)
    order = list(names.values())
    return alt.Chart(df).mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=alt.X("round:Q", title="Adaptation round (drift-active rounds only)"),
        y=alt.Y("value:Q", title="Evidence score", scale=alt.Scale(domain=[-0.04, 1.04])),
        color=alt.Color("Signal:N",
                        scale=alt.Scale(domain=order,
                                        range=["#4c78a8", "#b279a2", "#e45756"]),
                        legend=alt.Legend(title=None, orient="top", columns=1)),
        tooltip=[alt.Tooltip("round:Q", title="round"),
                 alt.Tooltip("Signal:N", title="signal"),
                 alt.Tooltip("value:Q", title="value", format=".3f"),
                 alt.Tooltip("phase:N", title="phase")],
    ).properties(height=height, width="container")
