"""
Altair chart builders for the demo frontend.

Altair rather than st.line_chart because the built-in chart gives no real
legend, no axis titles and no phase context - all of which are the point of
these figures. Altair also inherits Streamlit's light/dark theme.

HOVER: each chart has a transparent point layer spanning the data which carries
a `nearest` selection. Hovering draws a vertical rule at the closest round and
shows ONE tooltip listing the round plus every line's value there. The selection
MUST be attached to that point layer - binding it to the rule mark instead gives
it no hit area and it silently never fires.
"""
import altair as alt
import pandas as pd

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full"]

# Deliberately plain names - these get read off a screen during a presentation.
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
    "pre_attack": "1. normal",
    "slow_poison": "2. attacker sneaking in",
    "post_poison": "3. attack looks normal",
    "recovery_check": "4. obvious attack returns",
    "pre_drift": "1. normal (device A)",
    "genuine_drift": "2. slowly changing",
    "post_drift": "3. new normal (device B)",
}


def phase_bands(log, n_rounds):
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
    edges = alt.Chart(bands).mark_rule(
        strokeDash=[2, 3], opacity=0.35, strokeWidth=1).encode(x=alt.X("start:Q"))
    return shade, edges, text


def _hover_layers(df, x, series_field, value_field, series_order, fmt=".2f"):
    """
    Returns (selection, selector_layer, rule_layer).

    selector_layer is a transparent point mark carrying the `nearest` selection -
    this is what actually detects the pointer. rule_layer draws the vertical line
    and holds the combined tooltip, switched on by the same selection.
    """
    sel = alt.selection_point(fields=["round"], nearest=True,
                              on="mouseover", empty=False)
    selectors = alt.Chart(df).mark_point(size=200).encode(
        x=x, opacity=alt.value(0),
    ).add_params(sel)
    rule = alt.Chart(df).transform_pivot(
        series_field, value=value_field, groupby=["round"]
    ).mark_rule(color="#9aa0a6", strokeWidth=1).encode(
        x=x,
        opacity=alt.condition(sel, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("round:Q", title="Round (x)")] +
                [alt.Tooltip(field=s, type="quantitative", format=fmt)
                 for s in series_order],
    )
    return sel, selectors, rule


def accuracy_chart(log, upto=None, n_rounds=None, height=380):
    """How often each system is right, round by round."""
    n_rounds = n_rounds or len(log["static"])
    cut = len(log["static"]) if upto is None else upto + 1

    rows = []
    for a in ARMS:
        for r in log[a][:cut]:
            rows.append({"round": r["round"], "accuracy": r["round_accuracy"],
                         "Detector": LABELS[a], "phase": r.get("phase", "")})
    df = pd.DataFrame(rows)
    shade, edges, text = _bands(phase_bands(log, n_rounds))

    x = alt.X("round:Q", title="Round  (x)",
              scale=alt.Scale(domain=[1, n_rounds], nice=False))
    y = alt.Y("accuracy:Q", title="How often it was right  (y)",
              scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Detector:N",
                      scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLORS),
                      legend=alt.Legend(title=None, orient="top", columns=2,
                                        labelLimit=440, symbolStrokeWidth=3))

    sel, selectors, rule = _hover_layers(df, x, "Detector", "accuracy", ARM_ORDER)

    line = alt.Chart(df).mark_line(strokeWidth=2.6, interpolate="monotone").encode(
        x=x, y=y, color=color)
    # per-point tooltip as well, so hovering a line always shows its own x,y
    dots = alt.Chart(df).mark_point(size=70, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("round:Q", title="Round (x)"),
                 alt.Tooltip("Detector:N", title="Line"),
                 alt.Tooltip("accuracy:Q", title="Value (y)", format=".2f")])

    return alt.layer(shade, edges, text, line, selectors, rule, dots).properties(
        height=height, width="container").resolve_scale(color="independent")


def gate_chart(log, thresh, veto, height=330):
    """Traffic score and reputation against their two cut-offs."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    tro = {g["round"]: g for g in log["gate_traffic_only"]}

    S_T, S_R = "Traffic score", "Reputation"
    S_TT, S_TR = "Traffic cut-off", "Reputation cut-off"
    order = [S_T, S_R, S_TT, S_TR]
    rng = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    dash = [[1, 0], [1, 0], [5, 4], [5, 4]]

    rows = []
    for r in gr:
        rd = r["round"]
        rows += [{"round": rd, "Series": S_T, "value": r.get("trust_score")},
                 {"round": rd, "Series": S_R, "value": r.get("reputation")},
                 {"round": rd, "Series": S_TT, "value": thresh},
                 {"round": rd, "Series": S_TR, "value": veto}]
    df = pd.DataFrame(rows)

    x = alt.X("round:Q", title="Round  (x)")
    y = alt.Y("value:Q", title="Score  (y)", scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Series:N", scale=alt.Scale(domain=order, range=rng),
                      legend=alt.Legend(title=None, orient="top", columns=2))
    sel, selectors, rule = _hover_layers(df, x, "Series", "value", order, fmt=".3f")

    lines = alt.Chart(df).mark_line(strokeWidth=2.4).encode(
        x=x, y=y, color=color,
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=order, range=dash),
                                  legend=None))
    dots = alt.Chart(df[df["Series"].isin([S_T, S_R])]).mark_point(
        size=70, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("round:Q", title="Round (x)"),
                 alt.Tooltip("Series:N", title="Line"),
                 alt.Tooltip("value:Q", title="Value (y)", format=".3f")])

    div = pd.DataFrame([{
        "round": r["round"], "value": r.get("trust_score"),
        "what": "Traffic-only said YES here. Ours said NO.",
    } for r in gr if tro.get(r["round"], {}).get("authorized") and not r.get("authorized")])

    layers = [lines, selectors, rule, dots]
    if not div.empty:
        layers.append(alt.Chart(div).mark_point(
            size=180, shape="cross", filled=True, color="#ff7f0e",
            strokeWidth=2, opacity=0.95).encode(
            x=x, y="value:Q",
            tooltip=[alt.Tooltip("round:Q", title="Round (x)"),
                     alt.Tooltip("what:N", title=""),
                     alt.Tooltip("value:Q", title="Traffic score", format=".3f")]))
    return alt.layer(*layers).properties(height=height, width="container")


def evidence_chart(log, height=290):
    """The three traffic checks, plotted separately."""
    gr = [r for r in log["gate_full"] if r.get("drift_active")]
    if not gr:
        return None
    names = {"attribution": "Same direction?",
             "topology": "Mixed traffic?",
             "temporal": "Smooth change?"}
    rows = [{"round": r["round"], "Check": lab, "value": r[k]}
            for r in gr for k, lab in names.items() if r.get(k) is not None]
    df = pd.DataFrame(rows)
    order = list(names.values())

    x = alt.X("round:Q", title="Round  (x)")
    y = alt.Y("value:Q", title="Score  (y)", scale=alt.Scale(domain=[-0.04, 1.04]))
    color = alt.Color("Check:N",
                      scale=alt.Scale(domain=order,
                                      range=["#4c78a8", "#b279a2", "#e45756"]),
                      legend=alt.Legend(title=None, orient="top", columns=3))
    sel, selectors, rule = _hover_layers(df, x, "Check", "value", order, fmt=".3f")

    lines = alt.Chart(df).mark_line(strokeWidth=2.2, interpolate="monotone").encode(
        x=x, y=y, color=color)
    dots = alt.Chart(df).mark_point(size=70, filled=True).encode(
        x=x, y=y, color=color,
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("round:Q", title="Round (x)"),
                 alt.Tooltip("Check:N", title="Line"),
                 alt.Tooltip("value:Q", title="Value (y)", format=".3f")])
    return alt.layer(lines, selectors, rule, dots).properties(
        height=height, width="container")
