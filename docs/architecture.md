# Architecture — Trust-Aware IDS

Paste a block into mermaid.live and export PNG for the slide deck.

Keep labels SHORT. Mermaid sizes a box to its text, so long sentences inside
nodes are what make these diagrams sprawl. Explanations belong in the slide
notes, not in the boxes. Nothing should be left unconnected either — a subgraph
with no edges gets parked in a corner and wrecks the layout.

| Diagram | Use it for |
|---|---|
| 1. Pipeline | Slide 8, Framework/Architecture |
| 2. The decision | The one to actually present — readable from the back |
| 3. Four arms | Slide 11, explaining the comparison |

---

## Diagram 1 — Pipeline

```mermaid
flowchart TB
    SRC["Traffic stream<br/>115 N-BaIoT features"] --> CAL["Calibration<br/>first 400 records"]
    CAL --> FS["Freeze the measuring scale<br/>benign calibration only"]
    FS --> RND["Next round<br/>20 records"]
    RND --> PREQ["Predict first, then score<br/>all four arms"]
    PREQ --> ADWIN{"Frozen arm<br/>making errors?"}

    ADWIN -->|"no"| RND
    ADWIN -->|"yes"| PROP["Propose:<br/>'treat this window as normal'"]

    PROP --> EV["Measure A, P, S"]
    EV --> GATE{"Authorization<br/>gate"}
    OOB["R — reputation<br/>from outside the traffic"] --> GATE

    GATE -->|"R too low"| NO["REFUSE"]
    GATE -->|"T too low"| NO
    GATE -->|"T high enough"| YES["ACCEPT"]

    YES --> UPD["Model learns<br/>b and r move"]
    UPD --> RND
    NO -->|"nothing changes"| RND

    style GATE fill:#2d5016,color:#fff
    style OOB fill:#5c3a1a,color:#fff
    style NO fill:#8b1a1a,color:#fff
    style YES fill:#1a5c1a,color:#fff
    style FS fill:#3a3a5c,color:#fff
```

**Say alongside it:** ADWIN watches the arm that *never* updates, so the trigger
is identical for all four and cannot be corrupted by any arm's own bad updates.
The measuring scale is frozen at calibration so the attacker cannot move the
ruler. A refused update changes nothing at all.

---

## Diagram 2 — The decision (present this one)

```mermaid
flowchart LR
    W["Traffic<br/>window"] --> A["A · same direction<br/>as before?"]
    W --> P["P · varied<br/>traffic?"]
    W --> S["S · smooth<br/>change?"]

    A --> T["Traffic score T"]
    P --> T
    S --> T

    T --> Q{"Reputation R<br/>too low?"}
    D["R · where is it<br/>going?"] --> Q

    Q -->|"yes"| NO["REFUSE<br/>however good T is"]
    Q -->|"no"| Q2{"T high<br/>enough?"}
    Q2 -->|"yes"| YES["ACCEPT"]
    Q2 -->|"no"| NO

    style W fill:#333,color:#fff
    style D fill:#5c3a1a,color:#fff
    style NO fill:#8b1a1a,color:#fff
    style YES fill:#1a5c1a,color:#fff
    style T fill:#2d5016,color:#fff
```

**The line to say:** A, P and S are all measured from the very traffic being
judged, so a patient attacker can raise all three. R is the only input that does
not come from the traffic — which is why it is allowed to overrule the rest.
Delete that one branch and you get the "checks traffic only" baseline, which was
fooled at rounds 40, 72, 73, 74 and 75.

---

## Diagram 3 — Four arms

```mermaid
flowchart LR
    P["Proposed<br/>update"] --> S1["Never learns"]
    P --> S2["Learns from<br/>everything"]
    P --> S3["Checks traffic only"]
    P --> S4["Ours:<br/>traffic + reputation"]

    S1 --> R1["always refuses<br/>1.000"]
    S2 --> R2["always accepts<br/>0.200"]
    S3 --> R3["T only<br/>0.200"]
    S4 --> R4["T + R veto<br/>1.000"]

    style P fill:#333,color:#fff
    style S4 fill:#1a3a5c,color:#fff
    style R1 fill:#1a5c1a,color:#fff
    style R4 fill:#1a5c1a,color:#fff
    style R2 fill:#8b1a1a,color:#fff
    style R3 fill:#8b1a1a,color:#fff
```

Numbers are how often each caught the obvious repeat attack in Step 4, on real
N-BaIoT. Note that "never learns" also scores 1.000 — it is safe but cannot
adapt at all. Ours matches it on safety *while* still being able to accept
legitimate updates.
