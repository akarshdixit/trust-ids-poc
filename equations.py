r"""
The mechanism explained twice: plain first, formal second.

The plain version is written for someone in their first year who has not seen
this problem before. The maths sits behind a toggle so it can be switched off
entirely when presenting.

These must stay in lockstep with the code they describe:
  1    evidence/featurespace.py
  2-4  evidence/extract.py
  5-6  gate/trust_gate.py
  7    experiments/run_scenarios.py  (REFERENCE_EMA)
  8    experiments/run_scenarios.py  (ADWIN + DRIFT_COOLDOWN)
"""
import pandas as pd
import streamlit as st

GLOSSARY = [
    ("A", "Same direction?",
     "Is this change going the same way as changes we already said were fine?",
     "Yes - just send traffic that drifts the 'approved' way"),
    ("P", "Mixed traffic?",
     "Is the traffic doing lots of different things (like a real device), or the "
     "same thing over and over (like phoning one bad server)?",
     "Yes - just vary the traffic on purpose"),
    ("S", "Smooth change?",
     "Is the change gradual, or did everything jump at once?",
     "Yes - and it is free. Just go slowly."),
    ("T", "Traffic score",
     "The three answers above, added up into one score out of 1",
     "Yes - it is only made of the three above"),
    ("R", "Reputation",
     "Is this traffic going somewhere known to be bad? Looks at the DESTINATION, "
     "not at what the traffic looks like",
     "NO - this is the whole point"),
]


def _card(title, plain, equations, reading, show_math):
    st.markdown(f"##### {title}")
    st.info(plain)
    if show_math:
        for eq in equations:
            st.latex(eq)
    st.markdown(reading)
    st.write("")


def render(w_att, w_topo, w_temp, thresh, veto):
    st.subheader("What is going on here?")

    st.markdown("""
**The problem.** A normal security system is trained once and then never changes.
That is a problem, because real networks change - a device gets a software update
and starts behaving differently, and the system starts raising false alarms.

**The obvious fix.** Let the system keep learning. Every so often it looks at recent
traffic and says "OK, this is what normal looks like now."

**Why that is dangerous.** An attacker who knows the system keeps learning can abuse
it. Instead of attacking loudly, they attack *very slowly*, nudging their traffic a
tiny bit more normal-looking each time, until the system has been taught that their
attack IS normal. This is called poisoning.

**Our idea.** Before the system is allowed to learn anything new, something has to
approve it. That approver looks at the traffic - but a patient attacker can fake
anything about the traffic. So it *also* checks one thing the attacker cannot fake:
**where the traffic is going**. If the destination is a known-bad server, the update
is refused, no matter how innocent the traffic looks.
""")

    st.markdown("#### The five things we measure")
    st.table(pd.DataFrame(
        GLOSSARY,
        columns=["", "Short name", "What it actually asks", "Can an attacker fake it?"]
    ).set_index(""))
    st.caption(
        "Read the last column. Four of the five can be faked by an attacker who is "
        "willing to be patient, because they are all measured from the traffic itself. "
        "Only R cannot - which is exactly why R is allowed to overrule the rest.")

    st.divider()
    show_math = st.toggle("Show the maths", value=False,
                          help="Off by default. Turn on for the report and the viva.")

    left, right = st.columns(2, gap="large")

    with left:
        _card("Step 0 - put everything on the same scale",
              "Some measurements in this data are millions of times bigger than others. "
              "If we do not fix that, the big ones drown out everything else, like one "
              "person shouting over a whole room.",
              [r"z = \frac{\operatorname{sgn}(x)\odot\log(1+|x|) - \mu_{\mathrm{cal}}}"
               r"{\sigma_{\mathrm{cal}}}"],
              "We work out the scale once, at the start, from traffic we trust - and "
              "then never change it. If we recalculated it as we went, the attacker "
              "could slowly move the goalposts.", show_math)

        _card("A - Same direction?",
              "Is this change heading the same way as changes we already approved?",
              [r"A = \tfrac{1}{2}\left(1 + \cos\theta\right), \qquad "
               r"\theta = \angle(\Delta_w,\ r)"],
              "**1.0** = going exactly the way approved changes went  \n"
              "**0.5** = we have not approved anything yet, so there is nothing to "
              "compare to  \n"
              "**0.0** = going the opposite way", show_math)

        _card("P - Mixed traffic?",
              "Real devices do lots of different things. A machine quietly reporting to "
              "one attacker-controlled server does the same thing over and over.",
              [r"P = \frac{-\sum_i p_i \ln p_i}{\ln K}, \qquad K = 9"],
              "**High** = lots of different behaviour, looks like a real device  \n"
              "**Low** = the same narrow pattern repeating  \n"
              "*Note:* this data does not tell us which addresses were contacted, so we "
              "estimate variety from the traffic measurements instead.", show_math)

        _card("S - Smooth change?",
              "Did things change gradually, or jump all at once? A sudden jump is "
              "suspicious.",
              [r"S = \frac{1}{1 + \lVert \mu_w - \mu_{w-1} \rVert}"],
              "**High** = this batch looks much like the last one  \n"
              "**Low** = something changed suddenly  \n"
              "**The catch:** a patient attacker gets a high score here for free, just "
              "by going slowly. So a high S on its own proves nothing at all.",
              show_math)

    with right:
        _card("T - the traffic score",
              "Add the three answers above together into one number between 0 and 1.",
              [rf"T = {w_att:.2f}\,A + {w_topo:.2f}\,P + {w_temp:.2f}\,S"],
              "**The important limitation:** all three inputs are measured from the very "
              "traffic we are being asked to trust. Changing how we weight them does not "
              "help - a patient attacker can push all three up at once. This is why we "
              "need something from outside the traffic.", show_math)

        _card("R - reputation (the outside check)",
              "Is this traffic going to a destination already known to be bad? This does "
              "not look at what the traffic looks like at all - only where it is headed.",
              [r"R \text{ does not depend on the traffic } x"],
              "**1.0** = destination looks fine  \n"
              "**0.05** = destination is a known attacker server  \n"
              "The attacker can make their traffic look as innocent as they like and this "
              "number does not budge.  \n\n"
              "**Say this honestly in the review:** we do not have a real threat-intel "
              "feed, so R is simulated from whether the record truly came from an attack "
              "file. It stands in for a real feed. This proves what an approval step can "
              "do *once it has an outside signal* - it does not prove we can detect the "
              "attacker.", show_math)

        _card("The decision",
              "If reputation is bad, refuse - full stop. Otherwise, accept only if the "
              "traffic score is high enough.",
              [r"\mathrm{auth} = \begin{cases}"
               r"\textbf{false} & \text{if } R < \tau_R\\[4pt]"
               r"T \geq \tau_T & \text{otherwise}\end{cases}",
               rf"\tau_T = {thresh:.2f}, \qquad \tau_R = {veto:.2f}"],
              "That first line is the **only** difference between our method and the "
              "'checks traffic only' comparison. Same model, same measurements, same "
              "cut-off, same data. One extra check.", show_math)

        _card("What happens afterwards",
              "If an update is approved, the system learns it and we move our reference "
              "point. If it is refused, absolutely nothing changes.",
              [r"r \leftarrow 0.7\,r + 0.3\,\Delta_w, \qquad b \leftarrow \mu_w"],
              "A refused attack leaves no trace at all. That is why our method ends up "
              "performing exactly like the never-learning system on the final test, "
              "instead of being merely a bit better than the careless one.", show_math)

        _card("When do we even ask?",
              "Only when the never-learning system starts getting things wrong. That is "
              "the signal that the world has changed and an update might be needed.",
              [r"\text{ADWIN on } \mathbb{1}[\hat{y}^{\text{static}}_t \neq y_t]"],
              "We watch the system that never updates, so the trigger is the same for "
              "all four and cannot be corrupted by any of them making bad decisions. "
              "The rest of the time, nothing is proposed and nobody learns.", show_math)

    st.divider()
    st.markdown("#### How to read the demo, in four steps")
    st.markdown("""
**Step 1 - normal traffic.** Nothing happening. All four systems agree.

**Step 2 - the attacker sneaks in.** Attack traffic is nudged closer to normal, a
little at a time. The traffic checks start looking fine, because the attacker is
deliberately making them look fine. Reputation stays bad the whole time.

**Step 3 - the attack now looks completely normal.** Every system scores 0 here,
**including the one that never learns**. That is not our method failing: at this
point the attack traffic is genuinely identical to normal traffic, so *nothing* that
only looks at traffic could tell the difference. Only reputation knows.

**Step 4 - an obvious attack comes back.** A loud, full-strength version of the same
attack the system was originally trained to catch. **This is the step that decides
everything.** Any system that accepted the poisoned updates has forgotten how to
catch it. Any system that refused them still catches it every single time.
""")
    st.success(
        "**The one number to quote:** how often each system catches the obvious attack "
        "in Step 4. Do not lead with overall accuracy - Step 3 drags that down for "
        "everyone, including a hypothetical perfect detector.")
