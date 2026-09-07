r"""
The mechanism explained twice: once in plain English for presenting, once
formally for the report and the viva.

Kept in its own module because LaTeX strings and shell escaping do not mix, and
because these definitions must stay in lockstep with the code they describe:
  1    evidence/featurespace.py
  2-4  evidence/extract.py
  5-6  gate/trust_gate.py
  7    experiments/run_scenarios.py  (REFERENCE_EMA)
  8    experiments/run_scenarios.py  (ADWIN + DRIFT_COOLDOWN)
"""
import pandas as pd
import streamlit as st

# symbol, name, plain meaning, range, who controls it
GLOSSARY = [
    ("z", "Calibrated features",
     "Every measurement rescaled so they can be compared to each other",
     "—", "Defender (frozen at calibration)"),
    ("A", "Attribution consistency",
     "Is this change heading the same way as changes we already approved?",
     "0 to 1", "Attacker can shape this"),
    ("P", "Topology / diversity",
     "Is the traffic varied like real usage, or narrow like one C2 channel?",
     "0 to 1", "Attacker can shape this"),
    ("S", "Temporal stability",
     "Is the change smooth and gradual rather than a sudden jump?",
     "0 to 1", "Attacker can shape this"),
    ("T", "Trust score",
     "The three traffic signals above, combined into one verdict",
     "0 to 1", "Attacker can shape this"),
    ("R", "Reputation",
     "Independent verdict on WHERE the traffic is going, not what it looks like",
     "0 to 1", "Attacker cannot shape this"),
    ("tau_T", "Trust threshold",
     "How much traffic evidence we demand before accepting an update",
     "0.55", "Defender sets it"),
    ("tau_R", "Veto threshold",
     "How bad reputation has to be before we refuse outright",
     "0.30", "Defender sets it"),
]


def _card(symbol, title, plain, equations, reading, show_math):
    st.markdown(f"### {symbol} — {title}")
    st.info(f"**In plain English:** {plain}")
    if show_math:
        for eq in equations:
            st.latex(eq)
    st.markdown(reading)
    st.write("")


def render(w_att, w_topo, w_temp, thresh, veto):
    st.subheader("What the system is actually doing")

    st.markdown(
        "**The one-sentence version.** An intrusion detector that keeps learning can be "
        "taught the wrong thing, so before it accepts any update we score the traffic "
        "three ways — but all three are things an attacker can fake, so we also check an "
        "independent signal the attacker does not control, and let that one overrule "
        "everything else.")

    st.markdown("#### Cheat sheet")
    st.table(pd.DataFrame(
        GLOSSARY, columns=["Symbol", "Name", "What it means", "Range", "Who controls it"]
    ).set_index("Symbol"))
    st.caption(
        "The last column is the whole argument. Rows 2-5 are computed from the traffic "
        "being adapted to, so a patient attacker can shape all of them. R is not — which is "
        "why it is allowed to overrule them.")

    st.divider()
    show_math = st.toggle("Show the equations", value=True,
                          help="Turn off for a non-technical audience.")

    left, right = st.columns(2, gap="large")

    with left:
        _card("z", "Calibrated feature space",
              "Put every measurement on the same scale first, so no single one can "
              "shout down the others.",
              [r"z = \frac{\operatorname{sgn}(x)\odot\log(1+|x|) - \mu_{\mathrm{cal}}}"
               r"{\sigma_{\mathrm{cal}}}"],
              "*Why it matters:* raw N-BaIoT features span 19 orders of magnitude. "
              "Without this, five jitter-variance columns account for 100% of every "
              "comparison and the other 110 features are ignored. The scale is fixed "
              "during calibration and never updated — otherwise the attacker could move "
              "the ruler.", show_math)

        _card("A", "Attribution consistency",
              "Is this change heading in the same direction as changes we already "
              "approved as genuine?",
              [r"\Delta_w = \mu_w - b",
               r"A = \operatorname{clip}\!\left(\frac{1}{2}\left(1 + "
               r"\frac{\Delta_w \cdot r}{\lVert\Delta_w\rVert \, \lVert r\rVert}"
               r"\right),\, 0,\, 1\right)"],
              "**1.0** — drifting exactly the way previously approved drift went  \n"
              "**0.5** — nothing to compare against yet (no update approved so far)  \n"
              "**0.0** — drifting the opposite way to everything we have trusted", show_math)

        _card("P", "Topology / diversity",
              "Is this traffic varied, the way genuine device usage is — or narrow and "
              "repetitive, the way a single command-and-control channel is?",
              [r"P = \frac{-\sum_i p_i \ln p_i}{\ln K}, \qquad K = 9"],
              "**High** — many different behaviours, looks like organic usage  \n"
              "**Low** — the same narrow pattern repeating  \n"
              "*Stand-in measure:* N-BaIoT ships no IP or port data, so we approximate "
              "topology using the spread of the traffic features themselves.", show_math)

        _card("S", "Temporal stability",
              "Is the change smooth and gradual, rather than a sudden jump?",
              [r"S = \frac{1}{1 + \lVert \mu_w - \mu_{w-1} \rVert}"],
              "**High** — this window looks much like the last one  \n"
              "**Low** — something changed abruptly  \n"
              "*The catch:* this is exactly the signal a patient attacker maximises on "
              "purpose. Going slowly is free for them. That is why a high S alone proves "
              "nothing — and why the traffic-only baseline exists, to show it being "
              "fooled.", show_math)

    with right:
        _card("T", "Trust score",
              "Roll the three traffic signals into a single number.",
              [r"T = w_A A + w_P P + w_S S",
               rf"T = {w_att:.2f}\,A + {w_topo:.2f}\,P + {w_temp:.2f}\,S"],
              "*The limitation to say out loud:* all three inputs come from the traffic "
              "we are being asked to learn from. No choice of weights fixes that — a "
              "sufficiently patient attacker can raise all three at once. That is the "
              "argument for needing a channel outside the traffic.", show_math)

        _card("R", "Reputation — the out-of-band channel",
              "A verdict on WHERE the traffic is going, arriving from outside the "
              "traffic itself.",
              [r"R(x) \neq f(x)"],
              "**1.0** — destination is known-good  \n"
              "**0.05** — destination is known command-and-control infrastructure  \n"
              "*Why it is independent:* R is not computed from the features x at all. "
              "The attacker can make traffic look however they like and R does not "
              "move.  \n"
              "*Be honest in review:* here R is simulated from ground-truth traffic "
              "origin, because N-BaIoT has no reputation field. It is a stand-in for a "
              "real threat-intel feed, and closer to an oracle than one would be.",
              show_math)

        _card("Decision", "The authorization rule",
              "Refuse outright if reputation is bad. Otherwise accept only if the "
              "traffic evidence clears the bar.",
              [r"\mathrm{auth} = \begin{cases}"
               r"\textbf{false} & \text{if } R < \tau_R \ \ (\text{veto})\\[6pt]"
               r"\mathbb{1}\!\left[\,T \geq \tau_T\,\right] & \text{otherwise}"
               r"\end{cases}",
               rf"\tau_T = {thresh:.2f}, \qquad \tau_R = {veto:.2f}"],
              "The veto line is the **only** difference between the proposed mechanism "
              "and the traffic-only baseline. Everything else about them is identical — "
              "same model, same evidence, same threshold, same data.", show_math)

        _card("Update", "What happens after a decision",
              "An approved update moves the model AND our reference point. A refused "
              "one changes absolutely nothing.",
              [r"r \leftarrow (1-\lambda)\,r + \lambda\,\Delta_w, \qquad b \leftarrow \mu_w",
               r"\lambda = 0.3"],
              "*Why this matters for the result:* a rejected poisoning attempt leaves no "
              "residue at all — not in the model, not in the baseline b, not in the "
              "reference direction r. That is why the gated arm ends up identical to the "
              "frozen baseline rather than merely better than the ungated one.",
              show_math)

        _card("Trigger", "When is an update even proposed?",
              "Only when the frozen detector starts making mistakes — that is what "
              "signals the world has changed.",
              [r"\text{ADWIN}\big(\,\mathbb{1}[\hat{y}^{\text{static}}_t \neq y_t]\,\big)"
               r"\;\Rightarrow\; \text{15-round window}"],
              "ADWIN watches the error stream of the arm that **never updates**, so the "
              "trigger is identical for all four arms and cannot be corrupted by an "
              "arm's own bad updates. Outside a triggered window nothing is proposed and "
              "nobody learns.", show_math)

    st.divider()
    st.markdown("#### How to read the live demo")
    st.markdown(
        "1. **Benign phase** — nothing is happening, all four arms agree.\n"
        "2. **Poisoning phase** — the attacker walks attack traffic slowly toward the "
        "benign region. The traffic evidence starts to look fine, because the attacker "
        "is making it look fine. Reputation stays at 0.05 throughout.\n"
        "3. **Attack looks benign** — the attacker is now fully inside the benign region. "
        "Every arm scores 0 here, including the frozen one. That is not a failure of the "
        "method: these packets are indistinguishable from benign traffic by construction, "
        "so no traffic-based detector could do better. Only R knows.\n"
        "4. **Repeat attack** — an obvious, full-strength repeat of the attack the model "
        "was originally trained on. **This is the phase that decides the result.** Whoever "
        "accepted poisoned updates has forgotten how to catch it; whoever refused them "
        "still catches it every time.")
    st.caption(
        "Present the repeat-attack number as the headline. Overall accuracy is dragged "
        "down for every arm by phase 3, where a perfect detector would also score zero.")
