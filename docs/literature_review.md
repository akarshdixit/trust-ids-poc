# Literature Review — Review-2 (Slides 5–6)

**VERIFY EVERY ROW BEFORE SUBMITTING.** These are papers I am confident exist and
have described accurately, but volume/issue/page numbers are deliberately
omitted rather than guessed. Look each one up by title, confirm the venue and
year, and fill the exact details from the DOI. Never submit a citation you have
not personally opened.

The template allows a maximum of 2 slides — rows 1–5 on the first, 6–10 on the
second.

The rows are ordered to build one argument: adaptation is necessary → here is
how it is normally done → adaptation is attackable → here is what people do
about it → every one of those defences inspects the same channel the attacker
controls. That last sentence is the research gap.

---

## Slide 5 — Adaptation, and why it is needed

| Paper | Objective | Methodology | Pros / Cons | Findings |
|---|---|---|---|---|
| **Sommer & Paxson**, IEEE S&P 2010 | Ask why machine learning underperforms for network intrusion detection specifically | Critical analysis of the ML-NIDS literature and its evaluation practice | + Names the evaluation traps precisely<br>– Diagnoses, proposes no mechanism | Sets the standard of honesty we hold to: we disclose a null result and a simulated signal rather than hide them |
| **Gama et al.**, ACM Computing Surveys 2014 | Organise how learners cope with data that changes over time | Taxonomy of drift types, detectors and adaptation strategies | + Standard vocabulary for drift<br>– Assumes drift is benign throughout | Frames continual adaptation; the adversarial case sits outside its scope, which is where our work begins |
| **Bifet & Gavaldà**, SIAM SDM 2007 | Detect change in a stream without committing to a fixed window | ADWIN: adaptive windowing with a statistical change test | + Parameter-light, with guarantees<br>– Fires on *error*, not on distribution shift | Our drift trigger. Its error-based nature is exactly why our genuine-drift scenario returned a null result |
| **Montiel et al.**, JMLR 2021 | Provide one library for online / streaming learning in Python | Incremental estimators, drift detectors and prequential metrics | + Prequential evaluation built in<br>– We use only a linear model on top of it | Implements all four arms of our comparison |
| **Meidan et al.**, IEEE Pervasive Computing 2018 | Detect IoT botnet traffic on a per-device basis | Deep autoencoders trained on benign traffic only; 115 streaming features | + Real traffic from genuinely infected commercial devices<br>– No destination, IP or reputation field of any kind | Our dataset. The absence of any reputation field is precisely why our out-of-band signal has to be simulated |

---

## Slide 6 — Attacking the adaptation, and defending it

| Paper | Objective | Methodology | Pros / Cons | Findings |
|---|---|---|---|---|
| **Barreno et al.**, ASIACCS 2006 | Ask what an adversary can do to a learning system | Attack taxonomy: causative vs exploratory, targeted vs indiscriminate | + Defines the causative (training-time) attack class<br>– Predates modern deep learning | Names our threat model exactly: a causative, targeted attack on a deployed learner |
| **Nelson et al.**, USENIX LEET 2008 | Poison a spam filter that retrains itself in deployment | Craft training messages that shift the filter's learned notion of "normal" | + Attacks a real, continually adapting system<br>– Text domain; assumes periodic retraining | The closest prior threat to ours. We move it to network traffic and add an authorization step in front of the update |
| **Biggio et al.**, ICML 2012 | Maximise a classifier's test error by injecting training points | Gradient ascent on the SVM's test loss to construct poison points | + Principled, near-optimal poison<br>– Batch SVM, white-box, injected all at once | The canonical poisoning attack. Ours is *paced slowly over a stream* rather than injected as a batch |
| **Jagielski et al.**, IEEE S&P 2018 | Study poisoning attacks *and* defences for regression | Optimisation-based attacks, plus TRIM, a trimmed-loss defence | + Offers a defence, not just an attack<br>– The defence inspects only the training data itself | The closest defence baseline — and it still judges the data using the data. That limitation is our gap |
| **Kirkpatrick et al.**, PNAS 2017 | Prevent a network forgetting earlier tasks as it learns new ones | Elastic Weight Consolidation: a Fisher-weighted penalty anchoring important weights | + Retains old knowledge under new learning<br>– Preserves whatever was learned, poisoned or not | Our named future baseline. It limits *forgetting*, but it cannot *refuse* a bad update — a different problem from ours |

---

## The research gap that falls out of the table

Every defence above judges a proposed update by inspecting the **same data
channel the attacker controls**. TRIM trims the training data; EWC anchors
weights against the training signal; drift detectors watch the error stream that
the poisoned data itself produces.

None of them consults a signal from **outside** that channel before permitting
the update.

That is the gap this project addresses: an authorization step whose deciding
input is not a function of the traffic being judged.

---

## Optional 11th row, if you have space

| Paper | Objective | Methodology | Pros / Cons | Findings |
|---|---|---|---|---|
| **Arp et al.**, USENIX Security 2022 | Catalogue recurring methodological errors in ML-for-security papers | Survey of common pitfalls — sampling bias, label leakage, inappropriate baselines | + A checklist you can audit your own work against<br>– Prescriptive, not a mechanism | Directly supports our disclosures: the simulated reputation channel and the Scenario-1 null result are both stated rather than buried |

This one is worth adding if a panel member is likely to press you on evaluation
rigour, because it lets you frame your caveats as *following known good practice*
rather than as apologies.

---

## Still to add from your own Review-1 deck

Your code refers to an **ExpIDS-style** traffic-only gating baseline. I do not
have that citation and will not invent one — pull it from your Review-1
literature table and add it as a row. It matters more than any other entry here,
because it is the comparison your headline result is against: it is the arm that
was fooled at rounds 40, 72, 73, 74 and 75.

---

# References (Slide 13) — IEEE format

Numbering matches the table rows above: [1]-[5] are Slide 5, [6]-[10] are Slide 6,
[11] is the optional extra row.

**Confidence note.** Titles, authors, venues and years below I am confident of.
**Page ranges on the conference papers are the fields most likely to need
correcting** — confirm them from the publisher page or DOI. Search the exact
title; every one of these is freely findable.

[1] R. Sommer and V. Paxson, "Outside the closed world: On using machine
learning for network intrusion detection," in *Proc. IEEE Symp. Security and
Privacy (S&P)*, Oakland, CA, USA, 2010, pp. 305-316.

[2] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A
survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4,
art. 44, Mar. 2014.

[3] A. Bifet and R. Gavalda, "Learning from time-changing data with adaptive
windowing," in *Proc. SIAM Int. Conf. Data Mining (SDM)*, Minneapolis, MN, USA,
2007, pp. 443-448.

[4] J. Montiel, M. Halford, S. M. Mastelini, G. Bolmier, R. Sourty, R. Vaysse,
A. Zouitine, H. M. Gomes, J. Read, T. Abdessalem, and A. Bifet, "River: Machine
learning for streaming data in Python," *Journal of Machine Learning Research*,
vol. 22, no. 110, pp. 1-8, 2021.

[5] Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher, A. Shabtai,
and Y. Elovici, "N-BaIoT-Network-based detection of IoT botnet attacks using
deep autoencoders," *IEEE Pervasive Computing*, vol. 17, no. 3, pp. 12-22,
Jul.-Sep. 2018.

[6] M. Barreno, B. Nelson, R. Sears, A. D. Joseph, and J. D. Tygar, "Can machine
learning be secure?," in *Proc. ACM Symp. Information, Computer and
Communications Security (ASIACCS)*, Taipei, Taiwan, 2006, pp. 16-25.

[7] B. Nelson, M. Barreno, F. J. Chi, A. D. Joseph, B. I. P. Rubinstein,
U. Saini, C. Sutton, J. D. Tygar, and K. Xia, "Exploiting machine learning to
subvert your spam filter," in *Proc. USENIX Workshop on Large-Scale Exploits and
Emergent Threats (LEET)*, San Francisco, CA, USA, 2008.

[8] B. Biggio, B. Nelson, and P. Laskov, "Poisoning attacks against support
vector machines," in *Proc. Int. Conf. Machine Learning (ICML)*, Edinburgh, UK,
2012.

[9] M. Jagielski, A. Oprea, B. Biggio, C. Liu, C. Nita-Rotaru, and B. Li,
"Manipulating machine learning: Poisoning attacks and countermeasures for
regression learning," in *Proc. IEEE Symp. Security and Privacy (S&P)*,
San Francisco, CA, USA, 2018, pp. 19-35.

[10] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis,
C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming catastrophic forgetting in
neural networks," *Proc. National Academy of Sciences (PNAS)*, vol. 114, no. 13,
pp. 3521-3526, Mar. 2017.

[11] D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi,
C. Wressnegger, L. Cavallaro, and K. Rieck, "Dos and don'ts of machine learning
in computer security," in *Proc. USENIX Security Symp.*, Boston, MA, USA, 2022.

[12] *<< Your ExpIDS-style traffic-only gating reference - take this from your
Review-1 deck. It is the baseline your headline result is measured against. >>*

---

## Fastest way to verify all of these

Paste each title into Google Scholar or the DOI resolver. For any paper where
the page range disagrees with what is written above, trust the publisher and
correct this file. Author-name diacritics (Zliobaite, Gavalda, Grabska-Barwinska)
are also commonly mangled - restore them if your bibliography style keeps them.
