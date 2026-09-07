r"""
The mechanism written out formally, for the report and the viva.

Kept in its own module because LaTeX strings and shell/heredoc escaping do not
mix, and because these should stay in lockstep with the code they describe:
  1  evidence/featurespace.py
  2-4 evidence/extract.py
  5-6 gate/trust_gate.py
  7  experiments/run_scenarios.py (REFERENCE_EMA)
"""
import streamlit as st


def render(w_att, w_topo, w_temp, thresh, veto):
    st.subheader("The mechanism, formally")
    left, right = st.columns(2)

    with left:
        st.markdown("**1 — Calibrated feature space**  (frozen at calibration)")
        st.latex(r"z = \frac{\operatorname{sgn}(x)\odot\log(1+|x|) - \mu_{\mathrm{cal}}}"
                 r"{\sigma_{\mathrm{cal}}}")
        st.caption(
            "Fit on calibration benign rows only, then frozen — refitting on live windows "
            "would hand the attacker the reference frame. Every signal below is computed on "
            "z, never on raw x. On raw N-BaIoT values the geometry collapses: "
            "S = 1.7e-17, and five HH_jit_*_variance columns carry 100% of every cosine.")

        st.markdown("**2 — Attribution consistency**  $A$")
        st.latex(r"\Delta_w = \mu_w - b")
        st.latex(r"A = \operatorname{clip}\!\left(\frac{1}{2}\left(1 + "
                 r"\frac{\Delta_w \cdot r}{\lVert\Delta_w\rVert \, \lVert r\rVert}"
                 r"\right),\, 0,\, 1\right)")
        st.caption(
            "Cosine between this window's shift away from the last trusted baseline b, and "
            "r — an EMA of shift directions already authorized as genuine. Returns a neutral "
            "0.5 while r is undefined (no authorized drift yet), rather than fabricating a "
            "score.")

        st.markdown("**3 — Topology / diversity**  $P$")
        st.latex(r"P = \frac{-\sum_i p_i \ln p_i}{\ln K}, \qquad K = 9")
        st.caption(
            "Normalised entropy of a 9-bin histogram over the first five features. A proxy "
            "for communication-topology diversity, since N-BaIoT ships no IP or port data: "
            "organic benign usage touches more behavioural regimes than a narrow C2 channel.")

        st.markdown("**4 — Temporal stability**  $S$")
        st.latex(r"S = \frac{1}{1 + \lVert \mu_w - \mu_{w-1} \rVert}")
        st.caption(
            "Smoothness of window-to-window change. This is precisely the signal a patient "
            "attacker maximises on purpose — which is why it is not sufficient alone, and "
            "why the traffic-only arm exists as a baseline.")

    with right:
        st.markdown("**5 — Trust score**  $T$")
        st.latex(r"T = w_A A + w_P P + w_S S")
        st.latex(rf"T = {w_att:.2f}\,A + {w_topo:.2f}\,P + {w_temp:.2f}\,S")
        st.caption(
            "A, P and S are all derived from the traffic being adapted to, so all three are "
            "attacker-influenceable. No weighting of them fixes that — which is the argument "
            "for an independent channel.")

        st.markdown("**6 — Authorization rule**")
        st.latex(r"\mathrm{auth} = \begin{cases}"
                 r"\textbf{false} & \text{if } R < \tau_R \ \wedge\ \neg\,\mathrm{traffic\_only}"
                 r"\quad(\text{veto})\\[6pt]"
                 r"\mathbb{1}\!\left[\,T \geq \tau_T\,\right] & \text{otherwise}"
                 r"\end{cases}")
        st.latex(rf"\tau_T = {thresh:.2f}, \qquad \tau_R = {veto:.2f}")
        st.caption(
            "R is the out-of-band reputation — independent of x by construction, R(x) ≠ f(x). "
            "The veto line is the only thing the traffic-only baseline does not have, and it "
            "is what the whole result turns on.")

        st.markdown("**7 — State update, on authorization only**")
        st.latex(r"r \leftarrow (1-\lambda)\,r + \lambda\,\Delta_w, \qquad b \leftarrow \mu_w")
        st.latex(r"\lambda = 0.3")
        st.caption(
            "A refused update changes nothing at all — neither the model, nor the baseline b, "
            "nor the reference direction r. The trusted frame only ever advances through "
            "drift that was authorized, so a rejected poisoning attempt leaves no residue.")

        st.markdown("**8 — Drift trigger**")
        st.latex(r"\text{ADWIN}\big(\,\mathbb{1}[\hat{y}^{\text{static}}_t \neq y_t]\,\big)"
                 r"\;\Rightarrow\; \text{cooldown} = 15 \text{ rounds}")
        st.caption(
            "ADWIN watches the error stream of the FROZEN arm, so the trigger is identical "
            "for all four arms and cannot be corrupted by an arm's own bad updates. Outside "
            "a drift-active window nothing is proposed and nobody learns.")
