"""
Build report_ids.pdf: the full project report (structure, algorithms,
packages, datasets, experiments, results).

    python docs/build_report.py            # writes report_ids.pdf at the repo root

Result tables are read from results/*.json at build time, so the report always
matches the committed logs. The 5-seed imperfect-feed table (section 8.3) was
measured separately and is written in by hand; it is labelled as such.
"""
import json
import os

import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "report_ids.pdf")

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full", "ewc", "gate_quarantine"]
ARM_NAMES = {
    "static": "Static (never updates)",
    "ungated": "Ungated (learns everything)",
    "gate_traffic_only": "Traffic-only gate",
    "gate_full": "Full gate (one-shot veto)",
    "ewc": "EWC (lambda = 50)",
    "gate_quarantine": "Quarantine gate (new)",
}
SOURCES = [("real", "N-BaIoT"), ("iot23", "IoT-23"), ("synth", "Synthetic")]

INK = colors.HexColor("#1f1f1e")
MUTED = colors.HexColor("#6b6a64")
RULE = colors.HexColor("#d9d8d2")
HEAD_BG = colors.HexColor("#eeede8")
ACCENT = colors.HexColor("#5b3f8c")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=28, textColor=INK),
    "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontSize=12, leading=16,
                               alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=20,
                         textColor=INK, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                         textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14.2,
                           textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bl", parent=ss["BodyText"], fontSize=10, leading=14,
                             textColor=INK, leftIndent=14, bulletIndent=4, spaceAfter=2),
    "caption": ParagraphStyle("c", parent=ss["BodyText"], fontSize=8.5, leading=11,
                              textColor=MUTED, spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8.5, leading=10.5,
                           textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=ss["BodyText"], fontSize=8.5, leading=10.5,
                            textColor=INK, fontName="Helvetica-Bold"),
    "code": ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10.2,
                           textColor=INK, backColor=colors.HexColor("#f6f5f1"),
                           borderPadding=6, leftIndent=4, rightIndent=4, spaceAfter=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def code(text):
    return KeepTogether([Preformatted(text.strip("\n"), S["code"])])


def table(rows, widths, header=True, bold_col0=False):
    data = []
    for i, r in enumerate(rows):
        style = "cellb" if (header and i == 0) else "cell"
        data.append([Paragraph(str(c), S["cellb" if (bold_col0 and j == 0 and i) else style])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    t.setStyle(TableStyle(st))
    return t


def figure(path, caption, width=16.5 * cm):
    from reportlab.lib.utils import ImageReader
    full = os.path.join(BASE, path)
    iw, ih = ImageReader(full).getSize()
    return KeepTogether([Image(full, width=width, height=width * ih / iw),
                         P(caption, "caption")])


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def phase_acc(log, arm, phase):
    sel = [r["round_accuracy"] for r in log.get(arm, []) if r.get("phase") == phase]
    return float(np.mean(sel)) if sel else float("nan")


def fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# ------------------------------------------------------------------ diagram
def pipeline_diagram():
    d = Drawing(17 * cm, 7.6 * cm)
    W, H = 3.0 * cm, 1.3 * cm

    def box(x, y, text, fill="#ffffff", stroke="#5b3f8c"):
        d.add(Rect(x, y, W, H, rx=4, ry=4, fillColor=colors.HexColor(fill),
                   strokeColor=colors.HexColor(stroke), strokeWidth=1))
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            d.add(String(x + W / 2, y + H / 2 + (len(lines) / 2 - i - 0.75) * 9.5, ln,
                         fontName="Helvetica", fontSize=7.6, textAnchor="middle",
                         fillColor=INK))
        return (x, y)

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.9))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        s = 4.5
        d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                       x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)],
                      fillColor=MUTED, strokeColor=MUTED))

    gx = 0.55 * cm
    step = W + 0.55 * cm
    top, bot = 5.6 * cm, 2.3 * cm
    box(gx, top, "IoT traffic stream\n(N-BaIoT / IoT-23)", "#f1eef8")
    box(gx + step, top, "Online IDS\npredict, then score")
    box(gx + 2 * step, top, "ADWIN drift monitor\n(error rate +\nfeature distance)")
    box(gx + 3 * step, top, "Update proposal\n'learn this window\nas benign'")
    box(gx + 4 * step, top, "Frozen feature space\nsigned log + z-score")
    for i in range(4):
        arrow(gx + i * step + W, top + H / 2, gx + (i + 1) * step, top + H / 2)

    box(gx + 4 * step, bot, "Traffic evidence\nattribution, topology,\ntemporal", "#fff7ef",
        "#eb6834")
    arrow(gx + 4 * step + W / 2, top, gx + 4 * step + W / 2, bot + H)
    box(gx + 3 * step, bot, "Trust gate\nT = weighted sum\nthreshold 0.55 / 0.45", "#eef7ee",
        "#2ca02c")
    arrow(gx + 4 * step, bot + H / 2, gx + 3 * step + W, bot + H / 2)
    box(gx + 2 * step, bot, "Quarantine (new)\nSPRT over episode,\ncheckpoint, rollback",
        "#f1eef8")
    arrow(gx + 3 * step, bot + H / 2, gx + 2 * step + W, bot + H / 2)
    box(gx + 1 * step, bot, "Apply update\n(provisional / final)\nor refuse", "#ffffff")
    arrow(gx + 2 * step, bot + H / 2, gx + step + W, bot + H / 2)
    arrow(gx + step + W / 2, bot + H, gx + step + W / 2, top)

    box(gx + 3 * step, 0.15 * cm, "Out-of-band reputation\n(independent of\ntraffic features)",
        "#fdeeee", "#d62728")
    arrow(gx + 3 * step + W / 2, 0.15 * cm + H, gx + 3 * step + W / 2, bot)
    arrow(gx + 3 * step, 0.15 * cm + H / 2, gx + 2 * step + W / 2, bot)
    return d


# ------------------------------------------------------------------ sections
def title_page():
    return [Spacer(1, 4.5 * cm),
            P("Trust-Aware Continual Learning for IoT Intrusion Detection", "title"),
            Spacer(1, 0.4 * cm),
            P("Authorizing drift-triggered model updates with traffic evidence, an "
              "out-of-band reputation signal, and quarantined, reversible updates",
              "subtitle"),
            Spacer(1, 1.2 * cm),
            P("Project report (report_ids)", "subtitle"),
            P("VIT Project-I, Review 2 proof-of-concept", "subtitle"),
            Spacer(1, 0.3 * cm),
            P("Repository: github.com/akarshdixit/trust-ids-poc", "subtitle"),
            Spacer(1, 5 * cm),
            P("Contents: 1 Problem and goal - 2 System structure - 3 Repository layout - "
              "4 Datasets - 5 Algorithms - 6 Experimental setup - 7 Results - "
              "8 Robustness and the quarantine extension - 9 Related work and novelty - "
              "10 Software, packages and backend - 11 Visualizations - 12 Testing - "
              "13 How to run - 14 Limitations and future work - References", "caption"),
            PageBreak()]


def s1_problem():
    return [P("1. Problem and goal", "h1"),
            P("An IoT intrusion detection system (IDS) deployed on a changing network has to "
              "keep learning: devices are added, firmware changes and benign traffic drifts. "
              "Continual learning makes that possible, but it also opens an attack surface. "
              "A patient attacker can shape traffic so that it drifts slowly from attack "
              "towards benign; if the IDS adapts to every detected drift, it gradually learns "
              "that the attack is normal (<i>slow poisoning</i>, the 'boiling frog' attack)."),
            P("The goal of this project is an <b>authorization gate</b> that sits between drift "
              "detection and the model update. Every proposed update must be authorized. The "
              "gate combines two kinds of evidence:"),
            *bullets([
                "<b>Traffic evidence</b> computed from the traffic window itself (attribution "
                "consistency, topology diversity, temporal stability). It is cheap but "
                "attacker-influenceable.",
                "<b>An out-of-band reputation signal</b> (where the traffic goes: destination "
                "reputation or C2 cost), which is independent of the traffic features and "
                "therefore not controlled by the attacker.",
            ]),
            P("This version adds a <b>quarantine extension</b>: the reputation channel is "
              "treated as sequential, fallible evidence about a whole drift episode; updates "
              "are applied provisionally and rolled back when the evidence turns against "
              "them. The research question is whether such a gate keeps the benefits of "
              "continual learning on benign drift while refusing poisoning, including when "
              "the reputation feed is noisy or late."),
            P("<b>Honest scope.</b> No public IoT dataset carries a reputation field, so the "
              "signal is simulated from ground-truth origin (benign 1.0, attack 0.05) and then "
              "degraded on purpose. The claim is what an authorization mechanism achieves "
              "<i>once it has an independent channel</i>, not that the gate detects attackers "
              "by itself.")]


def s2_structure():
    return [P("2. System structure", "h1"),
            pipeline_diagram(),
            P("Figure 1. Pipeline. Every arm sees the same stream and the same drift trigger; "
              "the arms differ only in how they decide whether to apply a proposed update.",
              "caption"),
            P("Per 20-record round:"),
            *bullets([
                "<b>Prequential evaluation.</b> Every arm predicts each record first and is "
                "scored afterwards (test-then-train), so no arm ever sees a label before "
                "predicting it.",
                "<b>Drift trigger.</b> A shared ADWIN detector watches the frozen static arm's "
                "error rate (optionally also each record's distance from the calibration "
                "centre). A change point opens a 15-round drift-active window.",
                "<b>Proposal.</b> Only in drift-active rounds is an update proposed: 'learn this "
                "window as benign'. Outside it nobody learns.",
                "<b>Decision.</b> Each arm's policy (never, always, gate, EWC, quarantine) "
                "decides; authorized updates are applied with River's learn_one.",
            ]),
            P("Six arms run side by side, which makes every comparison a controlled experiment:"),
            table([["Arm", "Policy"],
                   [ARM_NAMES["static"], "Trained on calibration data once, never updates. "
                    "Reference for 'no adaptation'."],
                   [ARM_NAMES["ungated"], "Applies every drift-triggered proposal."],
                   [ARM_NAMES["gate_traffic_only"], "Traffic evidence only, no reputation veto "
                    "(stands in for ExpIDS-style drift gating)."],
                   [ARM_NAMES["gate_full"], "Traffic evidence plus a one-shot reputation veto "
                    "(the Review-2 mechanism)."],
                   [ARM_NAMES["ewc"], "Applies every proposal, with Elastic Weight "
                    "Consolidation anchoring weights to calibration. Tests whether the gate's "
                    "benefit is merely 'not forgetting'."],
                   [ARM_NAMES["gate_quarantine"], "Sequential reputation test over the drift "
                    "episode, provisional updates, rollback (section 5.6)."]],
                  [4.2 * cm, 12.3 * cm], bold_col0=True)]


def s3_layout():
    rows = [["Path", "Role"],
            ["models/online_ids.py", "OnlineIDS (StandardScaler | LogisticRegression, ADWIN) "
             "and EWCOnlineIDS"],
            ["evidence/featurespace.py", "Frozen signed-log + z-score space (trust boundary)"],
            ["evidence/extract.py", "Attribution, topology and temporal evidence"],
            ["gate/trust_gate.py", "One-shot gate: weighted trust score + reputation veto"],
            ["gate/quarantine.py", "SPRT, provisional updates, checkpoints, rollback (new)"],
            ["data/load_nbaiot.py", "N-BaIoT loader and the two scenario streams"],
            ["data/load_iot23.py", "IoT-23 Zeek conn-log parser and streams (new)"],
            ["data/synth_stream.py", "Synthetic 20-feature streams (regression check)"],
            ["data/reputation.py", "Imperfect reputation feed: noise, flips, lag (new)"],
            ["data/fake_nbaiot.py", "Fake N-BaIoT-format files for tests (new)"],
            ["experiments/run_scenarios.py", "Orchestrator; writes results/*_log_&lt;tag&gt;.json"],
            ["experiments/streams.py", "build_stream(): one stream builder for CLI, API, app"],
            ["experiments/sweeps.py", "Seeds, reputation, EWC and quarantine sweeps (new)"],
            ["experiments/make_plots.py", "Static figures per data source"],
            ["backend/server.py, console.html", "FastAPI backend, SSE logs, live console (new)"],
            ["app.py, charts.py, equations.py", "Streamlit demo, Altair charts, equations"],
            ["tests/", "108 pytest tests"],
            ["results/", "Logs (.json) and figures (.png)"],
            ["PROJECT_GUIDE.txt", "Changes, how to run, meaning of every chart"]]
    return [P("3. Repository layout", "h1"), table(rows, [5.4 * cm, 11.1 * cm])]


def s4_datasets():
    return [PageBreak(), P("4. Datasets", "h1"),
            P("4.1 N-BaIoT (primary)", "h2"),
            P("UCI dataset 442 (Meidan et al., 2018): traffic of nine commercial IoT devices "
              "infected with Mirai and BASHLITE (Gafgyt). Each row is a packet window "
              "summarised into 115 statistical features (packet size, jitter, counts, "
              "covariance and correlation over 5 time windows, for host, channel and socket "
              "aggregations), so no packet parsing is needed. Only the files the pipeline "
              "uses were downloaded:"),
            table([["File", "Rows x cols", "Used for"],
                   ["Danmini_Doorbell/benign_traffic.csv", "49,548 x 115",
                    "benign traffic, both scenarios"],
                   ["Danmini_Doorbell/mirai_attacks/scan.csv", "107,685 x 115",
                    "attack traffic, Scenario 2"],
                   ["Danmini_Doorbell/gafgyt_attacks/combo.csv", "59,718 x 115",
                    "sanity check (check_before_real_data.py)"],
                   ["Ennio_Doorbell/benign_traffic.csv", "39,100 x 115",
                    "second device, Scenario 1"]],
                  [6.6 * cm, 2.8 * cm, 7.1 * cm]),
            Spacer(1, 4),
            P("All files: 0 NaN, 0 Inf, identical column order. 20 columns (covariance and "
              "Pearson-correlation features) take negative values, and the dynamic range spans "
              "about 19 orders of magnitude (maximum 5.7e17), which is why evidence is computed "
              "in a calibrated space (section 5.2)."),
            P("4.2 IoT-23 (second, independent dataset)", "h2"),
            P("Stratosphere Laboratory, CTU Prague (Garcia et al., 2020). Only the labelled Zeek "
              "flow logs (conn.log.labeled) were downloaded, not the PCAPs. The parser handles "
              "the format quirks (space-separated last columns, 'det_label' vs "
              "'detailed-label' headers, label case, '-' for missing values)."),
            table([["Capture", "Flows", "Role"],
                   ["CTU-IoT-Malware-Capture-34-1", "23,145 (21,222 Mirai: DDoS + C&C)",
                    "attack, plus device-A benign (1,923 flows)"],
                   ["CTU-Honeypot-Capture-5-1", "1,374 benign (Amazon Echo)",
                    "device B, Scenario 1"],
                   ["CTU-Honeypot-Capture-4-1", "452 benign (Philips HUE)",
                    "parsed, too small to use"]],
                  [5.4 * cm, 5.6 * cm, 5.5 * cm]),
            Spacer(1, 4),
            P("21 features per flow: 8 numeric (duration, originator/responder bytes and "
              "packets, IP bytes, missed bytes), 3 protocol one-hots (tcp, udp, icmp) and 10 "
              "connection-state one-hots (S0, SF, REJ, ...)."),
            P("4.3 Synthetic stream and other sources", "h2"),
            *bullets([
                "<b>Synthetic:</b> 20 Gaussian features (data/synth_stream.py), used as a fast "
                "regression check with known ground truth.",
                "<b>Fake N-BaIoT:</b> generated files with the real column names and dynamic "
                "range, only for tests on machines without the data.",
                "<b>CICIoT2023</b> was considered but not used: UNB requires a registration "
                "form.",
            ]),
            P("4.4 The simulated reputation channel", "h2"),
            P("meta['reputation'] is 1.0 for benign origin and 0.05 for attack origin. "
              "data/reputation.py degrades it in three independently controllable ways: "
              "Gaussian noise (sigma), a per-record flip rate (the feed reports the wrong "
              "verdict, as with a stale or poisoned blocklist) and lag (the verdict for record "
              "i arrives at record i + lag). Features, labels and phases are untouched, so "
              "every comparison against the oracle run stays controlled.")]


def s5_algorithms():
    return [PageBreak(), P("5. Algorithms", "h1"),
            P("5.1 Online IDS and drift detection", "h2"),
            P("The detector is River's pipeline StandardScaler | LogisticRegression, trained "
              "one record at a time with SGD. ADWIN (adaptive windowing, Bifet and Gavalda "
              "2007) keeps a variable-length window of a 0/1 error stream and signals a change "
              "when two sub-windows have significantly different means. The shared monitor "
              "uses the frozen static arm, whose error rate is an unbiased measure of how far "
              "live traffic has moved from calibration. The optional second trigger runs "
              "ADWIN on each record's normalised distance from the calibration centre, "
              "||z - c|| / sqrt(d), which catches benign-to-benign drift that never raises "
              "the error rate."),
            P("5.2 Calibrated feature space (trust boundary)", "h2"),
            code("""
z = (slog(x) - mu_calib) / sd_calib        slog(x) = sign(x) * log(1 + |x|)
mu_calib, sd_calib: fit ONCE on calibration benign records, then frozen
"""),
            P("The evidence signals are geometric, so features must be on comparable scales. "
              "On raw N-BaIoT values temporal stability evaluates to 1.7e-17 and five jitter "
              "variance columns carry 100% of every cosine, so nothing is ever authorized. The "
              "signed log keeps the sign of correlation features. Fitting on calibration data "
              "only means the attacker can never move the reference frame."),
            P("5.3 Traffic evidence", "h2"),
            code("""
shift        = mean(window_z) - trusted_baseline
attribution  = (cos(shift, reference_direction) + 1) / 2      (0.5 if none yet)
reference_direction <- 0.7 * reference_direction + 0.3 * shift  (after authorization)
topology     = H(histogram of 5 features, 10 bins) / log(9)     (normalised entropy)
temporal     = 1 / (1 + ||mean(window_z) - mean(previous_window_z)||)
trust score  T = 0.4 * attribution + 0.3 * topology + 0.3 * temporal
"""),
            P("Attribution asks whether this drift continues a direction that was already "
              "authorized; topology rewards varied, device-like behaviour over a narrow "
              "repeated channel; temporal stability rewards gradual change, which is exactly "
              "what a slow attacker maximises, so it is not sufficient on its own. Weights are "
              "normalised to sum to 1 (negative or all-zero weights are rejected)."),
            P("5.4 One-shot trust gate (Review 2)", "h2"),
            code("""
r = mean reputation of the 20-record window
if r < 0.3:            REFUSE  (out-of-band veto)
elif T >= 0.55:        AUTHORIZE
else:                  REFUSE  (weak traffic evidence)
"""),
            P("5.5 Elastic Weight Consolidation baseline", "h2"),
            P("After calibration the logistic-regression weights w* are frozen and the diagonal "
              "Fisher information F is estimated on calibration inputs (p(1-p) z^2, normalised "
              "to a maximum of 1). After each SGD step every weight is pulled back: "
              "w <- w - min(1, eta * lambda * F) (w - w*). This penalises moving important "
              "weights, the standard defence against catastrophic forgetting. Only the LR "
              "weights are anchored; the scaler keeps adapting."),
            P("5.6 Quarantine extension (new)", "h2"),
            P("<b>Sequential reputation test.</b> Each record's reputation is a vote (benign if "
              "> 0.5). The feed is assumed wrong on at most e = 30% of records, so under "
              "'benign episode' a benign vote has probability 1 - e and under 'attack episode' "
              "probability e. Wald's sequential probability ratio test accumulates the "
              "log-likelihood ratio over the whole drift episode:"),
            code("""
step  = ln((1 - e) / e) = 0.847                 per vote: LLR += step (benign) or -= step
A     = ln((1 - beta) / alpha) = +4.60          upper bound; LLR is clamped at A
B     = ln(beta / (1 - alpha)) = -4.60          lower bound; reaching B condemns the episode
alpha = beta = 0.01                             target false-condemn / miss rates
"""),
            P("Clamping at A stops a long benign prefix (a lagged feed) from banking credit an "
              "attack would first have to burn off. Condemnation is sticky until a new "
              "episode starts."),
            P("<b>Provisional updates, hold and rollback.</b>"),
            code("""
for each round:
    commit updates older than the hold (20 rounds)
    feed this window's votes to the SPRT (if drift-active or updates are held)
    if episode condemned and updates are held:
        restore the checkpoint taken before the episode's earliest held update
        replay held updates of other episodes on top (fresh checkpoints)
    if not drift-active:       nothing is proposed
    elif condemned:            refuse
    elif LLR <= 0:             hold (evidence not yet favourable)
    elif T >= 0.45:            save checkpoint, apply update PROVISIONALLY
    else:                      refuse
a new drift run starts a new episode only when nothing is held
"""),
            P("Because a wrong acceptance is now reversible, the provisional threshold is 0.45, "
              "0.1 below the one-shot gate's 0.55 (set before seeing results). The hold was "
              "first 5 rounds; the 600-record-lag sweep showed that a hold shorter than the "
              "feed's lag lets poison commit before the evidence arrives, so it was raised to "
              "20 rounds (400 records). That is the only setting changed after seeing results, "
              "and the 5-round hold is kept as an ablation.")]


def s6_setup():
    return [PageBreak(), P("6. Experimental setup", "h1"),
            P("Every stream begins with 400 labelled calibration records (30% attack) on which "
              "all arms are trained identically. The rest is processed in 20-record rounds."),
            table([["Scenario", "Phases (records)", "What a good system does"],
                   ["2 - Slow poisoning",
                    "pre-attack benign (400); slow poison: real attack rows interpolated towards "
                    "real benign rows in signed-log space (600); post-poison: attack held in "
                    "the benign region (400); recovery check: raw attack again (200)",
                    "Refuse the poisoning updates, so the obvious repeat attack is still "
                    "detected (recovery-check accuracy 1.0)."],
                   ["1 - Genuine drift",
                    "device A benign (400); gradual blend A to B (600); device B benign (400)",
                    "Accept the benign drift, so accuracy after the shift stays high."]],
                  [3.0 * cm, 7.6 * cm, 5.9 * cm]),
            Spacer(1, 6),
            P("<b>Metrics.</b> Prequential accuracy and F1 (attack = positive), and the windowed "
              "accuracy in each scenario's decisive phase: <i>recovery_check</i> for Scenario 2 "
              "and <i>post_drift</i> for Scenario 1. The sweeps also count <i>poisoned updates "
              "kept</i>: authorized updates whose window contained attack records and that "
              "were never rolled back. F1 is 0 in Scenario 1 because that scenario has no "
              "attack records after calibration; accuracy is the metric there."),
            P("<b>Seeds.</b> Seeds reshuffle calibration rows (N-BaIoT, IoT-23) or regenerate "
              "the world (synthetic). The default run uses the loaders' fixed seed 42.")]


def results_table(scenario, phase, tags):
    head = ["Arm"] + [f"{name}" for _, name in tags]
    rows = [head]
    logs = {t: load(f"{scenario}_log_{t}.json") for t, _ in tags}
    for arm in ARMS:
        rows.append([ARM_NAMES[arm]] + [fmt(phase_acc(logs[t], arm, phase)) for t, _ in tags])
    rows.append(["Drift-active rounds"] + [str(sum(r["drift_active"] for r in logs[t]["static"]))
                                           for t, _ in tags])
    return rows


def s7_results():
    out = [PageBreak(), P("7. Results", "h1"),
           P("7.1 Slow poisoning: accuracy on the obvious repeat attack", "h2"),
           table(results_table("slow_poisoning", "recovery_check", SOURCES),
                 [6.5 * cm, 3.3 * cm, 3.3 * cm, 3.4 * cm], bold_col0=True),
           P("Table 1. Recovery-check accuracy, oracle reputation feed, seed 42.", "caption"),
           P("Learning from every drift (ungated) or trusting traffic evidence alone "
             "(traffic-only) lets the attacker teach the model that the attack is normal: on "
             "real N-BaIoT both detect only 20% of the repeat attack, on IoT-23 0% and 30%. "
             "Both reputation-aware gates refuse every poisoning update and keep 1.000. EWC "
             "also survives here, but at a cost shown in 7.2.")]
    seeds = {t: load(f"sweep_seeds_{t}.json") for t, _ in SOURCES}
    rows = [["Arm"] + [name for _, name in SOURCES]]
    for arm in ARMS:
        rows.append([ARM_NAMES[arm]] + [
            f"{seeds[t]['slow_poisoning'][arm]['mean']:.3f} +/- "
            f"{seeds[t]['slow_poisoning'][arm]['ci95']:.3f}" if arm in seeds[t]["slow_poisoning"]
            else "-" for t, _ in SOURCES])
    out += [table(rows, [6.5 * cm, 3.3 * cm, 3.3 * cm, 3.4 * cm], bold_col0=True),
            P("Table 2. Recovery-check accuracy over 10 seeds, mean +/- 95% CI.", "caption"),
            figure("results/scenario2_slow_poisoning_real.png",
                   "Figure 2. Scenario 2 on real N-BaIoT: windowed accuracy per round. Only the "
                   "final 'obvious repeat attack' phase separates the arms."),
            figure("results/gate_decision_trace_real.png",
                   "Figure 3. One-shot gate decision trace (drift-active rounds): the traffic "
                   "trust score (blue) looks acceptable while reputation (purple) stays low, so "
                   "the veto refuses (red shading).")]

    tags_fd = [("real_featuredrift", "N-BaIoT + FD"), ("iot23", "IoT-23"),
               ("iot23_featuredrift", "IoT-23 + FD"), ("synth", "Synthetic")]
    out += [P("7.2 Genuine drift: accuracy after a benign device shift", "h2"),
            table(results_table("genuine_drift", "post_drift", tags_fd),
                  [5.3 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm], bold_col0=True),
            P("Table 3. Post-drift accuracy (FD = feature-drift trigger on). Without FD, real "
              "N-BaIoT detects no drift at all: a null result, all arms 0.993.", "caption"),
            P("On IoT-23 the frozen static model falls to 0.860 after the device shift, while "
              "the gated learners recover to about 0.98: adaptation is worth having. EWC pays "
              "for its poisoning resistance by not adapting (0.760). With the feature-drift "
              "trigger, the one-shot gate refuses most real benign drift (IoT-23 0.905; on "
              "N-BaIoT its trust score peaks at 0.543, just under 0.55), while the quarantine "
              "gate's lower provisional threshold accepts it (0.975 on IoT-23, 0.995 on "
              "N-BaIoT)."),
            figure("results/scenario1_genuine_drift_iot23.png",
                   "Figure 4. Scenario 1 on IoT-23: static drops after the shift; learners "
                   "that adapt keep high accuracy, EWC does not.")]
    return out


def s8_robustness():
    out = [PageBreak(), P("8. Robustness and the quarantine extension", "h1"),
           P("8.1 One-shot veto under an imperfect feed", "h2"),
           figure("results/sweep_reputation_real.png",
                  "Figure 5. One-shot full gate vs feed error rate, one line per veto "
                  "threshold (real N-BaIoT)."),
           P("The default veto (0.3) holds only up to 10% feed error: at 20% the attack "
             "window's mean reputation crosses 0.3 often enough for poison to get in (0.50 on "
             "N-BaIoT, 0.60 on IoT-23). A 0.5 veto holds to 30-40%; 0.7 holds longer but "
             "starts blocking benign drift. A fixed per-window threshold is fragile."),
           P("8.2 EWC strength", "h2"),
           figure("results/sweep_ewc_iot23.png",
                  "Figure 6. EWC lambda sweep on IoT-23: survives poisoning from lambda = 5 but "
                  "post-drift accuracy falls to 0.76, vs 0.98 for the gate."),
           P("8.3 Quarantine vs one-shot gate", "h2")]
    sq = {t: load(f"sweep_quarantine_{t}.json") for t, _ in SOURCES}

    def cell(t, flip, lag, label, kept=False):
        for r in sq[t]:
            if (r["scenario"] == "slow_poisoning" and not r["feature_drift"]
                    and r["flip_rate"] == flip and r["lag"] == lag):
                return str(r[f"{label} | poisoned updates kept"]) if kept else f"{r[label]:.2f}"
        return "-"

    variants = ["one-shot gate, veto < 0.3", "one-shot gate, veto < 0.5",
                "quarantine (proposed)", "quarantine, no rollback", "quarantine, 5-round hold"]
    short = ["veto<0.3", "veto<0.5", "quarantine", "no rollback", "5-round hold"]
    rows = [["Feed", "Metric"] + short]
    for flip, lag, name in [(0.2, 0, "20% wrong"), (0.3, 0, "30% wrong"), (0.4, 0, "40% wrong"),
                            (0.0, 300, "300 late"), (0.0, 600, "600 late")]:
        if not lag:
            rows.append([name, "repeat-attack acc."] + [
                " / ".join(cell(t, flip, lag, v) for t, _ in SOURCES) for v in variants])
        rows.append([name if lag else "", "poison kept"] + [
            " / ".join(cell(t, flip, lag, v, kept=True) for t, _ in SOURCES) for v in variants])
    out += [table(rows, [1.8 * cm, 2.4 * cm] + [2.46 * cm] * 5),
            P("Table 4. Seed 42; each cell is N-BaIoT / IoT-23 / Synthetic. 'Poison kept' = "
              "authorized updates on attack windows never rolled back. Read from "
              "results/sweep_quarantine_*.json.", "caption"),
            figure("results/sweep_quarantine_iot23.png",
                   "Figure 7. Quarantine sweep on IoT-23 (single seed)."),
            P("Over 5 seeds (measured separately, benign drift with the feature-drift trigger "
              "on the real datasets):"),
            table([["Feed error", "Repeat attack: quarantine vs one-shot (N-BaIoT / IoT-23 / "
                    "synth)", "Benign drift: quarantine vs one-shot"],
                   ["20%", "1.000 / 1.000 / 1.000 vs 0.42 / 0.62 / 0.40",
                    "0.995 / 0.976 / 1.000 vs 0.997 / 0.960 / 1.000"],
                   ["30%", "1.000 / 1.000 / 1.000 vs 0.22 / 0.32 / 0.20",
                    "0.995 / 0.976 / 1.000 vs 0.997 / 0.960 / 1.000"],
                   ["40%", "0.88 / 1.000 / 1.000 vs 0.22 / 0.32 / 0.20",
                    "0.996 / 0.899 / 0.398 vs 0.997 / 0.960 / 1.000"]],
                  [2.0 * cm, 7.4 * cm, 7.1 * cm]),
            P("Table 5. 5-seed means, hand-entered from a separate run.", "caption"),
            P("<b>Reading the ablations.</b> The sequential test is what makes the gate robust "
              "to a <i>wrong</i> feed (zero poison at 20-40% error, where the one-shot veto at "
              "0.3 keeps 2-9 updates and even 0.5 keeps 1-2). Rollback with a hold longer than "
              "the feed lag is what handles a <i>late</i> feed (without rollback 4-14 poisoned "
              "updates stay on IoT-23). The lower provisional threshold is what admits real "
              "benign drift. <b>Limit:</b> above the assumed 30% feed error the SPRT's "
              "guarantee is gone; it starts condemning benign drift (synthetic 0.40) and on "
              "N-BaIoT one seed in five lets the attack through.")]
    return out


def s9_related():
    rows = [["Idea", "Prior work"],
            ["Slow ('boiling frog') poisoning of self-updating detectors",
             "Rubinstein et al., ANTIDOTE (IMC 2009); 'Can't Boil This Frog' (2020)"],
            ["Poisoning that fakes or hides concept drift",
             "Korycki and Krawczyk, adversarial concept drift detection (2020)"],
            ["Drift detection and adaptation for security classifiers",
             "Transcend (USENIX Sec 2017), CADE (USENIX Sec 2021), OWAD (NDSS 2023), "
             "INSOMNIA (2021), ExpIDS (2025)"],
            ["Reject updates a trusted reference contradicts",
             "RONI (Nelson et al. 2008); FLTrust trusted root data (NDSS 2021)"],
            ["Trust-weighted updates and rollback after poisoning",
             "Trust-aware federated IDS (AGAT-FL, 2025); Crab (2024), FedRecover (2023)"],
            ["Sequential tests over IDS feedback",
             "Zhu et al., collaborative IDS with sequential hypothesis testing (2010)"]]
    return [PageBreak(), P("9. Related work and novelty", "h1"),
            P("A targeted (not systematic) literature search found every building block "
              "already published:"),
            table(rows, [6.3 * cm, 10.2 * cm]),
            Spacer(1, 6),
            P("<b>What was not found:</b> a single, non-federated, self-labelling online IoT "
              "IDS that (a) gates its own drift-triggered updates on an out-of-band channel "
              "independent of the traffic features, and (b) treats that channel as sequential, "
              "fallible evidence about a whole drift episode, with explicit assumption-stated "
              "error rates, applying updates provisionally and rolling them back when the "
              "evidence turns. Federated work weighs <i>clients</i>; RONI and FLTrust need "
              "trusted <i>labelled</i> data; drift-adaptation work (OWAD, CADE, Transcend) does "
              "not model an adversary who manufactures the drift."),
            P("The defensible novelty claim is therefore the <b>combination</b> and the "
              "<b>episode-level sequential treatment of the independent channel</b>, not any "
              "single component.")]


def s10_software():
    pk = [["Package", "Version", "Used for"],
          ["Python", "3.12", "runtime (virtual environment .venv)"],
          ["river", "0.26.1", "online logistic regression, StandardScaler, ADWIN, metrics"],
          ["numpy", "2.5.4", "vectors, evidence geometry"],
          ["pandas", "3.0.6", "CSV / Zeek log loading, tables in the app"],
          ["scipy", "1.18.1", "t-distribution confidence intervals in sweeps"],
          ["matplotlib", "3.11.2", "static figures in results/"],
          ["streamlit", "1.66.0", "interactive demo app"],
          ["altair", "6.3.0", "interactive charts in the app"],
          ["fastapi", "0.144.0", "backend REST API and SSE streams"],
          ["uvicorn[standard]", "0.54.0", "ASGI server for the backend"],
          ["httpx", "0.28.1", "app-to-backend client, API tests"],
          ["pytest", "9.1.1", "test suite"],
          ["reportlab", "4.2.5", "building this PDF (docs/build_report.py)"]]
    api = [["Endpoint", "Purpose"],
           ["GET /", "live log console"],
           ["GET /api/health", "status, dataset availability, auth, run limits"],
           ["GET /api/results/{scenario}/{tag}", "precomputed log (real, iot23, synth)"],
           ["POST /api/runs", "start a run: scenario, source, gate, reputation_noise, "
            "quarantine, seed, feature_drift, EWC, pace"],
           ["GET /api/runs, /api/runs/{id}", "list runs / run status and events"],
           ["GET /api/runs/{id}/events", "Server-Sent Events, one per round"],
           ["GET /api/logs/stream", "Server-Sent Events of all server logs"]]
    return [PageBreak(), P("10. Software, packages and backend", "h1"),
            P("10.1 Packages (requirements.txt, pinned)", "h2"),
            table(pk, [3.6 * cm, 2.2 * cm, 10.7 * cm]),
            P("10.2 Backend", "h2"),
            P("backend/server.py is a FastAPI application. Runs execute in worker threads; "
              "every round is logged to the console, logs/backend.log and an in-memory ring "
              "buffer that /api/logs/stream tails. Finished runs are saved to results/runs/ "
              "and reloaded on restart. Requests are validated with Pydantic (weights not all "
              "zero, feed error in (0, 0.5), ...). TRUST_IDS_TOKEN enables API-key auth "
              "(X-API-Key header or ?token=); TRUST_IDS_MAX_RUNS (default 2) caps concurrent "
              "runs and returns HTTP 429 above it. Port 8765 is used because 8000 is taken "
              "on the development machine."),
            table(api, [5.6 * cm, 10.9 * cm])]


def s11_visuals():
    rows = [["Visualization", "What it shows / how to read it"],
            ["scenario2_slow_poisoning_&lt;tag&gt;.png",
             "Windowed accuracy per round in Scenario 2. Judge the last phase (obvious repeat "
             "attack): poisoned systems score low, protected ones 1.0."],
            ["scenario1_genuine_drift_&lt;tag&gt;.png",
             "Same for the benign device shift; adapting systems stay high, static may drop."],
            ["gate_decision_trace_&lt;tag&gt;.png",
             "Trust score (blue) and reputation (purple) on drift-active rounds; red shading = "
             "update held or refused."],
            ["sweep_reputation_&lt;tag&gt;.png",
             "One-shot gate vs feed error, one line per veto threshold (light to dark = low to "
             "high). Left: attack resistance, right: benign adaptation."],
            ["sweep_ewc_&lt;tag&gt;.png",
             "EWC lambda vs both scenarios, gate as dashed reference: EWC trades adaptation "
             "for stability."],
            ["sweep_quarantine_&lt;tag&gt;.png",
             "Quarantine (purple) vs one-shot gates (orange dashed 0.3, dotted 0.5) and the "
             "no-rollback ablation, vs feed error; grey line = assumed 30%."],
            ["App: per-arm columns",
             "Records correct in this window and running accuracy, with change since last "
             "round."],
            ["App: gate panel",
             "During drift: the three traffic checks, trust score vs threshold, reputation, "
             "and the traffic-only vs full gate decisions."],
            ["App: accuracy chart",
             "Running prequential accuracy per arm; stripes = phases; triangles = accepted "
             "updates; hover for values."],
            ["App: gate chart",
             "Traffic score (green) and reputation (red) with dashed cut-offs; an update is "
             "accepted only if both are above their cut-offs."],
            ["App: evidence chart",
             "Attribution ('same direction?'), topology ('mixed traffic?'), temporal ('smooth "
             "change?') separately."],
            ["Live console", "One line per round; yellow VETO, green AUTH, blue DRIFT, purple "
             "ROLLBACK, red errors; quarantine verdict with its LLR."]]
    return [P("11. Visualizations", "h1"),
            P("Colours are consistent everywhere: grey static, red ungated, orange traffic-only, "
              "green full gate, blue EWC, purple quarantine. PROJECT_GUIDE.txt explains each in "
              "more detail."),
            table(rows, [5.2 * cm, 11.3 * cm])]


def s12_tests():
    rows = [["File", "Covers"],
            ["test_trust_gate.py", "veto, thresholds (inclusive / strict), weight normalisation"],
            ["test_quarantine.py", "SPRT bounds, clamping, stickiness, Monte-Carlo error rates; "
             "rollback and replay exactness; commit timing; arm isolation; robustness and lag "
             "behaviour"],
            ["test_pipeline.py", "regression against committed synthetic logs, headline "
             "result, callbacks, seeds, feature drift, EWC anchoring"],
            ["test_real_data.py", "N-BaIoT shapes and exact reproduction of committed results; "
             "IoT-23 parser on hand-written logs; IoT-23 headline and quarantine results"],
            ["test_reputation.py", "noise model: identity, no mutation, bounds, flips, lag"],
            ["test_fake_nbaiot.py", "fake data format, determinism, loader compatibility"],
            ["test_backend.py", "health, SSE streaming, persistence, validation, auth, "
             "concurrency limit, quarantine options, log formatting"]]
    return [P("12. Testing", "h1"),
            P("108 tests, about 30 seconds (python -m pytest). Tests that need the downloaded "
              "datasets skip when they are absent. A mutation check (disabling the veto) makes "
              "the headline tests fail, as it should. Adding the quarantine arm is tested to "
              "leave the other five arms bit-identical."),
            table(rows, [4.0 * cm, 12.5 * cm])]


def s13_run():
    return [P("13. How to run", "h1"),
            code(r"""
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest

.\.venv\Scripts\python.exe experiments\run_scenarios.py --source real
      [--feature-drift] [--rep-flip 0.3 --rep-sigma 0.05] [--rep-lag 300] [--seed N]
.\.venv\Scripts\python.exe experiments\make_plots.py --tag real
.\.venv\Scripts\python.exe experiments\sweeps.py quarantine --source real

.\.venv\Scripts\python.exe -m uvicorn backend.server:app --port 8765   # console :8765
.\.venv\Scripts\streamlit.exe run app.py                               # app     :8501
.\.venv\Scripts\python.exe docs\build_report.py                        # this PDF
"""),
            P("Datasets go in data/nbaiot/ and data/iot23/ (gitignored); download instructions "
              "are in README.md and PROJECT_GUIDE.txt.")]


def s14_limits():
    return [P("14. Limitations and future work", "h1"),
            *bullets([
                "The reputation feed is simulated from ground truth; integrating a real "
                "threat-intelligence or C2-cost source is Project-II work.",
                "The SPRT assumes independent per-record feed errors of at most 30%; above that "
                "it fails. Correlated errors (a whole blocklist going stale) are not modelled.",
                "The quarantine hold was raised from 5 to 20 rounds after seeing the lag sweep; "
                "other thresholds are untuned defaults.",
                "Scenario-2 pacing is constructed by interpolation; every endpoint is a real "
                "record, but the schedule is imposed.",
                "On N-BaIoT, seeds only reshuffle calibration rows, so its confidence intervals "
                "reflect calibration-order sensitivity, not dataset variability.",
                "One device pair per dataset; more devices and CICIoT2023 would strengthen the "
                "generality claim.",
                "Future work: learn the feed's reliability online, model correlated feed errors, "
                "adaptive hold length from measured feed latency, and adaptive attackers who "
                "target the provisional threshold.",
            ]),
            P("References", "h1"),
            *bullets([
                "Meidan et al., N-BaIoT: network-based detection of IoT botnet attacks using deep "
                "autoencoders, IEEE Pervasive Computing, 2018 (UCI dataset 442).",
                "Garcia, Parmisano, Erquiaga, IoT-23: a labeled dataset with malicious and benign "
                "IoT network traffic, Stratosphere Laboratory, 2020.",
                "Bifet and Gavalda, Learning from time-changing data with adaptive windowing "
                "(ADWIN), SDM 2007.",
                "Wald, Sequential tests of statistical hypotheses, Ann. Math. Stat., 1945.",
                "Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks (EWC), "
                "PNAS 2017.",
                "Montiel et al., River: machine learning for streaming data in Python, JMLR 2021.",
                "Rubinstein et al., ANTIDOTE: understanding and defending against poisoning of "
                "anomaly detectors, IMC 2009.",
                "Kravchik et al., Can't boil this frog, arXiv 2002.02741, 2020.",
                "Korycki and Krawczyk, Adversarial concept drift detection under poisoning "
                "attacks, arXiv 2009.09497, 2020.",
                "Jordaney et al., Transcend, USENIX Security 2017; Yang et al., CADE, USENIX "
                "Security 2021; Han et al., OWAD, NDSS 2023.",
                "Kumar, Fok, Thing, ExpIDS: a drift-adaptable NIDS with improved explainability, "
                "arXiv 2509.20767, 2025.",
                "Nelson et al., Exploiting machine learning to subvert your spam filter (RONI), "
                "LEET 2008; Cao et al., FLTrust, NDSS 2021.",
                "Crab: certified recovery from poisoning in federated learning, arXiv 2401.08216, "
                "2024; AGAT-FL, PeerJ Computer Science, 2025.",
                "Zhu et al., A distributed sequential algorithm for collaborative intrusion "
                "detection networks, arXiv 1002.3190, 2010.",
            ])]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Trust-Aware Continual Learning for IoT IDS")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def build(out=OUT):
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="report_ids",
                            author="Trust-IDS project",
                            subject="Trust-aware continual learning for IoT intrusion detection")
    story = (title_page() + s1_problem() + s2_structure() + s3_layout() + s4_datasets()
             + s5_algorithms() + s6_setup() + s7_results() + s8_robustness() + s9_related()
             + s10_software() + s11_visuals() + s12_tests() + s13_run() + s14_limits())
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print("wrote", out)


if __name__ == "__main__":
    build()
