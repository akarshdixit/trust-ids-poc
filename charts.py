"""
Altair chart builders for the demo frontend.

Altair rather than st.line_chart because the built-in chart gives no real
legend, no axis titles, no phase context, and no way to mark which rounds the
gate actually authorized - all of which are the point of these figures. Altair
also inherits Streamlit's light/dark theme, so the figures stay legible either
way.

Every chart carries a shared hover: moving the pointer anywhere draws a vertical
rule at the nearest round and shows ONE tooltip listing the x value and every
series' y value at that x, so lines can be compared without hunting for points.
"""
import altair as alt
import pandas as pd

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full"]

# Deliberately plain names - these are read off a screen during a presentation.
LABELS = {
    "static": "Never learns",
    "ungated": "Learns from everything",
    "gate_traffic_only": "Checks traffic only",
    "gate_full": "Our method (traffic + reputation)",
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
    "pre_attack": "1. normal traffic",
    "slow_poison": "2. attacker sneaking in",
    "post_poison": "3. attack now looks normal",
    "recovery_check": "4. obvious attack returns",
    "pre_drift": "1. normal (device A)",
    "genuine_drift": "2. slowly changing",
    "post_drift": "3. normal (device B)",
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


def _bands(bands, y_label=1.06):
    shade = alt.Chart(bands).mark_rect(opacity=0.07).encode(
        x=alt.X("start:Q"), x2="end:Q",
        color=alt.Color("shade:N", scale=alt.Scale(domain=[0, 1],
                                                   range=["#7f7f7f", "#000000"]),
                        legend=None))
    text = alt.Chart(bands).mark_text(
        fontSize=10, opacity=0.8, baseline="bottom", fontStyle="italic",
    ).encode(x=alt.X("mid:Q"), y=alt.datum(y_label), text="label:N")
    rules = alt.Chart(bands).mark_rule(
        strokeDash=[2, 3], opacity=0.35, strokeWidth=1).encode(x=alt.X("start:Q"))
    return shade, rules, text


def _hover_rule(df, x, series_field, value_field, series_order, x_title, fmt=".2f"):
    """Vertical rule + one combined tooltip listing every series at the hovered x."""
    hover = alt.selection_point(fields=["round"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    rule = alt.Chart(df).transform_pivot(
        series_field, value=value_field, groupby=["round"]
    ).mark_rule(color="#9aa0a6", strokeWidth=1).encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.55), alt.value(0)),
        tooltip=[alt.Tooltip("round:Q", title=x_title)] +
                [alt.Tooltip(field=s, type="quantitative", format=fmt)
                 for s in series_order],
    ).add_params(hover)
    return hover, rule


def accuracy_chart(log, upto=None, n_rounds=None, height=400):
    """How often each detector is right, round by round."""
    n_rounds = n_rounds or len(log["static"])
    cut = len(log["static"]) if upto is None else upto + 1

    rows = []
    for a in ARMS:
        for r in log[a][:cut]:
            rows.append({"round": r["round"], "accuracy": r["round_accuracy"],
                         "Detector": LABELS[a], "phase": r.get("phase", ""),
                         "authorized": bool(r.get("authorized"))})
    df = pd.DataFrame(rows)
    shade, rules, text = _bands(phase_bands(log, n_rounds))

    x = alt.X("round:Q", title="Round number  (x)  -  each round is 20 network records",
              scale=alt.Scale(domain=[1, n_rounds], nice=False))
    y = alt.Y("accuracy:Q", title="How often it was right  (y)  -  1.0 = every record",
              scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Detector:N",
                      scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLORS),
                      legend=alt.Legend(title="Detector", orient="top", columns=2,
                                        labelLimit=420, symbolStrokeWidth=3))

    hover, rule = _hover_rule(df, x, "Detector", "accuracy", ARM_ORDER, "Round (x)")

    line = alt.Chart(df).mark_line(strokeWidth=2.4, interpolate="monotone").encode(
        x=x, y=y, color=color)
    dots = alt.Chart(df).mark_point(size=65, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    auth = alt.Chart(df[df["authorized"]]).mark_point(
        size=42, filled=True, opacity=0.9, shape="triangle-down").encode(
        x=x, y=y, color=color,
        tooltip=[alt.Tooltip("round:Q", title="Round"),
                 alt.Tooltip("Detector:N", title="Accepted an update")])

    return alt.layer(shade, rules, text, line, auth, dots, rule).properties(
        height=height, width="container").resolve_scale(color="independent")


def gate_chart(log, thresh, veto, height=340):
    """Traffic score and reputation, against the two cut-off lines."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    tro = {g["round"]: g for g in log["gate_traffic_only"]}

    S_T = "Traffic score (T)"
    S_R = "Reputation (R)"
    S_TT = "Traffic cut-off"
    S_TR = "Reputation cut-off"
    order = [S_T, S_R, S_TT, S_TR]
    rng = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    dash = [[1, 0], [1, 0], [5, 4], [5, 4]]

    rows = []
    for r in gr:
        rd, ph = r["round"], r.get("phase", "")
        rows += [
            {"round": rd, "Series": S_T, "value": r.get("trust_score"), "phase": ph},
            {"round": rd, "Series": S_R, "value": r.get("reputation"), "phase": ph},
            {"round": rd, "Series": S_TT, "value": thresh, "phase": ph},
            {"round": rd, "Series": S_TR, "value": veto, "phase": ph},
        ]
    df = pd.DataFrame(rows)

    x = alt.X("round:Q", title="Round number  (x)")
    y = alt.Y("value:Q", title="Score  (y)  -  0 to 1",
              scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Series:N", scale=alt.Scale(domain=order, range=rng),
                      legend=alt.Legend(title=None, orient="top", columns=2,
                                        labelLimit=400))
    hover, rule = _hover_rule(df, x, "Series", "value", order, "Round (x)", fmt=".3f")

    lines = alt.Chart(df).mark_line(strokeWidth=2.2).encode(
        x=x, y=y, color=color,
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=order, range=dash),
                                  legend=None))
    dots = alt.Chart(df[df["Series"].isin([S_T, S_R])]).mark_point(
        size=65, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))

    div = pd.DataFrame([{
        "round": r["round"], "value": r.get("trust_score"), "phase": r.get("phase", ""),
        "what": "Traffic-only check said YES here - our method said NO",
    } for r in gr if tro.get(r["round"], {}).get("authorized") and not r.get("authorized")])

    layers = [lines, dots]
    if not div.empty:
        layers.append(alt.Chart(div).mark_point(
            size=170, shape="cross", filled=True, color="#ff7f0e",
            strokeWidth=2, opacity=0.95).encode(
            x=x, y="value:Q",
            tooltip=[alt.Tooltip("round:Q", title="Round"),
                     alt.Tooltip("what:N", title=""),
                     alt.Tooltip("value:Q", title="Traffic score", format=".3f")]))
    layers.append(rule)
    return alt.layer(*layers).properties(height=height, width="container")


def evidence_chart(log, height=300):
    """The three traffic checks, plotted separately."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    names = {"attribution": "Same direction as before?",
             "topology": "Varied traffic?",
             "temporal": "Changing smoothly?"}
    rows = []
    for r in gr:
        for k, lab in names.items():
            if r.get(k) is not None:
                rows.append({"round": r["round"], "Check": lab, "value": r[k],
                             "phase": r.get("phase", "")})
    df = pd.DataFrame(rows)
    order = list(names.values())

    x = alt.X("round:Q", title="Round number  (x)")
    y = alt.Y("value:Q", title="Score  (y)  -  higher looks more trustworthy",
              scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Check:N",
                      scale=alt.Scale(domain=order,
                                      range=["#4c78a8", "#b279a2", "#e45756"]),
                      legend=alt.Legend(title=None, orient="top", columns=1))
    hover, rule = _hover_rule(df, x, "Check", "value", order, "Round (x)", fmt=".3f")

    lines = alt.Chart(df).mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=x, y=y, color=color)
    dots = alt.Chart(df).mark_point(size=65, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return alt.layer(lines, dots, rule).properties(height=height, width="container")
