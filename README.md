# Trust-Aware IDS — Review-2 Proof-of-Concept

Working, tested pipeline for the core research contribution: an
authorization gate that decides whether a proposed continual-learning
update should be trusted, using traffic-derived evidence cross-checked
against an out-of-band reputation signal.

**Status: runs end-to-end on REAL data.** N-BaIoT (the default) reproduces
the committed Review-2 results exactly; IoT-23 replicates the headline on an
independent dataset; the synthetic stream remains as a fast regression check.
A sixth arm, the quarantine gate, extends the mechanism (see "Related work and
what is new here"). 143 tests.

**Documents**

| file | what it is |
|---|---|
| [`report_ids.pdf`](report_ids.pdf) | full report: structure, algorithms, packages, datasets, experiments, results, related work (rebuild: `python docs/build_report.py`) |
| [`PROJECT_GUIDE.txt`](PROJECT_GUIDE.txt) | what changed since the previous version, how to run everything, what every chart means, troubleshooting |
| this README | setup, data sources, results tables, novelty, datasets |

## What's implemented right now (tested)

- `models/online_ids.py` — River-based online logistic-regression IDS with
  ADWIN drift detection, prequential (test-then-train) evaluation.
- `evidence/extract.py` — three traffic-derived evidence signals:
  attribution consistency (cosine similarity between this window's shift
  FROM the last trusted baseline and an EMA of past authorized shift
  directions — not a raw mean-vector comparison), topology/diversity
  proxy, temporal stability.
- `gate/trust_gate.py` — the authorization gate: weighted traffic-evidence
  trust score, with an out-of-band reputation veto.
- `experiments/run_scenarios.py` — common orchestrator. A shared ADWIN
  monitor (fed by the frozen static arm's error rate) is the actual
  trigger for the rest of the pipeline: a detected change point opens a
  15-round "drift-active" window during which updates are proposed and
  each arm's gate decides whether to authorize them; outside that window,
  nothing is proposed to anyone. Runs SIX arms (static / ungated /
  traffic-only gate / proposed full gate / EWC / quarantine gate) through
  both scenarios (genuine drift, slow poisoning + a follow-up "obvious
  repeat attack" check).
- `experiments/make_plots.py` — generates the three demo figures in
  `results/`.

**On the reputation/out-of-band signal — say this, not more:** it is a
simulated stand-in, present because N-BaIoT has no destination-reputation
or C2-cost field at all. It is NOT a reputation *detector* — it does not
derive anything from the traffic features (R(x) ≠ a function of x), which
is what makes it a legitimate independent channel in principle. But in
this proof-of-concept it is keyed directly to ground-truth traffic origin,
so it is closer to an oracle than a real threat-intel feed would be. Don't
claim the gate "detects" the attacker — claim it demonstrates *what an
authorization mechanism does once it has an independent signal*, and name
integrating a real reputation/C2-cost source as Project-II work.

## Quickstart

```bash
pip install -r requirements.txt
python experiments/run_scenarios.py --source real   # prints summary, writes results/*_real.json
python experiments/make_plots.py --tag real         # writes results/*_real.png
```

## Setup, tests, and running everything

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest                       # 143 tests, ~1 min (real-data tests skip if data/ is absent)
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --port 8765
.\.venv\Scripts\streamlit.exe run app.py                     # live re-runs go via the backend
```

**Backend** (`backend/server.py`): http://localhost:8765 is a live console. API:
`POST /api/runs`, `GET /api/runs/{id}/events` (SSE, one event per round),
`GET /api/logs/stream` (SSE, all server logs), `GET /api/results/{scenario}/{tag}`.
Logs also go to `logs/backend.log`; finished runs are saved to `results/runs/`
and reloaded on restart. `TRUST_IDS_TOKEN` turns on API-key auth,
`TRUST_IDS_MAX_RUNS` caps concurrent runs (default 2).

**Data sources** (`experiments/streams.py`, used by CLI, backend and app):
`real` (N-BaIoT, the default), `iot23` (IoT-23, a second real dataset),
`synth` (20 synthetic features). `fake` (generated N-BaIoT-format files,
`data/fake_nbaiot.py`) is kept only for tests and machines without the data;
the UI no longer offers it.

**Pipeline options:** `--source`, `--seed`, `--feature-drift` (second drift
trigger on distance from calibration traffic), `--rep-sigma/--rep-flip/--rep-lag`
(imperfect reputation feed, `data/reputation.py`). A fifth arm, `ewc` (Elastic
Weight Consolidation), runs by default.

**Robustness sweeps** (`experiments/sweeps.py seeds|reputation|ewc|quarantine --source real|iot23|synth`).
Measured on REAL data (N-BaIoT Danmini_Doorbell, and IoT-23 as a second dataset):

| | N-BaIoT | IoT-23 |
|---|---|---|
| repeat attack after poisoning, full gate vs traffic-only (10 seeds) | 1.000 +/- 0.000 vs 0.200 +/- 0.000 | 1.000 +/- 0.000 vs 0.265 +/- 0.032 |
| benign device shift, frozen model -> gated learners (post-drift) | 0.997 -> 0.997 (null result) | 0.845 -> 0.977 |
| full gate vs reputation-feed error rate, veto < 0.3 | holds to 10%, 0.50 at 20% | holds to 10%, 0.60 at 20% |
| same, veto < 0.5 | holds to 30% | holds to 40% |
| EWC: smallest lambda that survives poisoning | 20 | 5 |
| EWC at that lambda, post-drift | 0.993 (no drift to adapt to) | 0.760 (vs gate 0.977) |

So: the headline replicates on an independent dataset; the 0.3 veto is fragile
once the reputation feed is imperfect; EWC can survive poisoning only by giving
up adaptation (visible on IoT-23, where the benign shift actually hurts).

With `--feature-drift`, real N-BaIoT Scenario 1 does trigger (15 drift rounds),
but the full gate authorizes none of it: real benign traffic is jumpy window to
window, so temporal stability is low (median 0.198), attribution sits at its
cold-start 0.5, and the trust score peaks at 0.543 < 0.55.

## Review-panel concerns (added on top; nothing above changed)

| concern | what was added | headline |
|---|---|---|
| 1. no comparison with other models | `experiments/benchmark.py` (16 detectors, same temporal split), `sweeps.py models` (every defence on 5 learners), app tab **Model comparison** | clean detection: best F1 1.000 / 0.993, ours 0.995 / 0.990 (N-BaIoT / IoT-23) at 24 / 5 us per record; under poisoning the reputation gates hold 1.00 in all 8 learner/dataset pairs whose detector works, learning everything is poisoned in 7 |
| 2. how the data was processed | `experiments/data_profile.py`, report section 5, app tab **Data & features** | 0 NaN / 0 Inf; raw N-BaIoT spans 18.5 orders of magnitude, after signed log + z-score 0 |
| 3. what the features are | `data/features.py` (meaning of all 115 + 21 features), importance by aggregation x time window | long windows (~10 s, ~1 min) of per-host packet counts/sizes carry most signal |
| 4. how the models were trained | report section 6.7: every hyperparameter, scenario pipeline and benchmark | 400 labelled calibration rows, then SGD only on authorized updates |
| 5. accuracy too low | per-phase accuracy (report 8.3) | the 0.55 averages in 1,000 records disguised as benign *by design*; clean held-out 0.995, live 0.997 |
| 6. no live experiment | `experiments/live_experiment.py`, backend `/live` dashboard, `POST /api/live/ingest` + `experiments/live_feed.py` for a real Zeek sensor, app tab **Live experiment** | 21,140 real flows replayed in real time: gates 0.997, learning everything 0.134; 13,766 flows/s capacity |

**Detector benchmark** (temporal split: train on the first 70% of each file in
time order, test on the last 30%; signed log + z-score fit on train; 3 repeats):

| model | F1 N-BaIoT | F1 IoT-23 | us / record | KB |
|---|---|---|---|---|
| Random forest | 1.0000 | 0.9922 | 10678 | 110 |
| Gradient boosting (HistGB) | 0.9999 | 0.9926 | 2038 | 431 |
| Logistic regression | 1.0000 | 0.9922 | 230 | 2 |
| MLP (64-32) | 0.9997 | 0.9922 | 249 | 231 |
| k-NN (k=5) | 0.9997 | 0.9921 | 4193 | 27188 |
| Adaptive random forest (River) | 0.9995 | 0.9922 | 32 | 462 |
| Online logistic regression (River) | 0.9994 | 0.9908 | 24 | 7 |
| Online LR, 400 calibration rows (ours, as deployed) | 0.9954 | 0.9898 | 24 | 7 |
| Autoencoder (benign only, N-BaIoT paper) | 0.9931 | 0.5150 | 241 | 408 |
| Isolation forest (benign only) | 0.9601 | 0.9497 | 23501 | 1773 |

Honest reading: as a plain detector ours is **not** the most accurate (several
models reach 1.000 on N-BaIoT); it is within 0.5 points of the best, the
fastest, tiny, and trained on 400 rows only. It also raises false alarms on
36% of an unseen device's benign traffic (best model: 0%). The project's claim
is the gate around the detector, and that holds for every learner tried.

**Live experiment** (IoT-23 Mirai capture, 24 h, original order and timing,
600x; first 2,000 flows train, 21,140 live):

| system | accuracy | recall | FPR | updates |
|---|---|---|---|---|
| static | 0.9971 | 0.9987 | 0.0238 | 0 |
| ungated | 0.1343 | 0.0658 | 0.0000 | 139 |
| gate_traffic_only | 0.2982 | 0.2445 | 0.0238 | 43 |
| gate_full | 0.9971 | 0.9987 | 0.0238 | 8 |
| ewc | 0.3233 | 0.2719 | 0.0277 | 139 |
| gate_quarantine | 0.9971 | 0.9987 | 0.0238 | 6 |

Median processing 0.052 ms per flow for all six
systems; worst backlog 0.72 s during the DDoS burst.

```powershell
.\.venv\Scripts\python.exe experiments\benchmark.py --source real        # or iot23
.\.venv\Scripts\python.exe experiments\sweeps.py models --source real
.\.venv\Scripts\python.exe experiments\data_profile.py --source real
.\.venv\Scripts\python.exe experiments\live_experiment.py --speed 600    # --speed 0 = max throughput
# with the backend running: http://localhost:8765/live, or stream a Zeek sensor's log:
.\.venv\Scripts\python.exe experiments\live_feed.py C:\path\to\conn.log --follow
```

## Related work and what is new here

A targeted search of the literature (not a systematic review) found every
building block of this project already published:

| idea | prior work |
|---|---|
| slow ("boiling frog") poisoning of a self-updating detector | Rubinstein et al., ANTIDOTE (IMC 2009); "Can't Boil This Frog" (arXiv 2002.02741) |
| poisoning that fakes or hides concept drift | Korycki & Krawczyk, adversarial drift detection (arXiv 2009.09497) |
| drift detection / adaptation for security classifiers | Transcend (USENIX Sec 2017), CADE (USENIX Sec 2021), OWAD (NDSS 2023), INSOMNIA, ExpIDS (arXiv 2509.20767) |
| reject an update that a trusted reference contradicts | RONI (Nelson et al. 2008); FLTrust-style trusted root data in federated learning |
| trust-weighted updates, rollback after poisoning | trust-aware federated IDS (e.g. AGAT-FL, PeerJ CS 2025); Crab (arXiv 2401.08216), FedRecover-style recovery |
| sequential hypothesis tests over IDS feedback | Zhu et al., collaborative IDS (arXiv 1002.3190) |

**What we did not find:** a single, non-federated, *self-labelling* online IoT
IDS that (a) gates its own drift-triggered updates on an out-of-band channel
that is independent of the traffic features, and (b) treats that channel as
*sequential, fallible evidence* about a whole drift episode — with explicit,
assumption-stated error rates — applying updates provisionally and rolling
them back when the evidence turns. Federated work weighs *clients*; RONI and
FLTrust need trusted *labelled* data; drift-adaptation work (OWAD, CADE,
Transcend) does not model an adversary who manufactures the drift.

So the defensible novelty claim is the *combination* and the episode-level
sequential treatment of the independent channel, not any single component.

### The extension: quarantine + sequential evidence + rollback (`gate/quarantine.py`)

Arm `gate_quarantine`, run by default alongside the original four arms (which
are bit-identical with or without it — tested):

1. **Sequential reputation test.** Each record's reputation is a vote; a Wald
   SPRT accumulates votes over the drift episode under an explicit feed model
   ("the feed is wrong on at most `feed_error` = 30% of records"), with
   alpha = beta = 0.01. Benign evidence is clamped so a lagged feed cannot bank
   credit; reaching the lower bound condemns the episode for good.
2. **Provisional updates.** An update is applied only while the episode's
   evidence favours benign (LLR > 0) and the traffic trust score clears a
   provisional threshold of 0.45 (the one-shot gate's 0.55 minus 0.1, set
   a priori, not tuned).
3. **Quarantine and rollback.** Each update keeps a model checkpoint for 20
   rounds (400 records). If its episode is condemned in that time, its updates
   are undone (earliest checkpoint restored, other episodes' updates replayed).
   The hold must outlast the feed's worst-case lag. It was first 5 rounds;
   the lag-600 sweep below showed that a hold shorter than the lag lets
   poisoned updates commit before the evidence arrives (14 kept on IoT-23),
   so it was raised to 20. That one setting was changed after seeing results;
   the 5-round hold is kept in the sweep as an ablation.
4. **Episodes merge while anything is held**, so a held update cannot escape
   the evidence that arrives after its drift run ends.

**Results** (`experiments/sweeps.py quarantine`, seed 42; "kept" = poisoned
updates still in the model at the end, i.e. authorized on a window containing
attack records and never rolled back):

| feed | metric | one-shot veto < 0.3 | one-shot veto < 0.5 | **quarantine** | no rollback | 5-round hold |
|---|---|---|---|---|---|---|
| 20% wrong | repeat attack, N-BaIoT / IoT-23 / synth | 0.50 / 0.60 / 0.50 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** | 1.00 all | 1.00 all |
| 30% wrong | repeat attack | 0.30 / 0.50 / 0.30 | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** | 1.00 all | 1.00 all |
| 40% wrong | repeat attack | 0.30 / 0.50 / 0.30 | 0.50 / 1.00 / 0.50 | **1.00 / 1.00 / 1.00** | 1.00 all | 1.00 all |
| 20-40% wrong | poisoned updates kept | 2-9 | 0-2 | **0** | 0 | 0 |
| 300 records late | poisoned updates kept | 0 / 1 / 0 | 0 / 1 / 0 | **0 / 0 / 0** | 0 / 4 / 0 | 0 / 0 / 0 |
| 600 records late | poisoned updates kept | 1 / 1 / 2 | 1 / 1 / 2 | **0 / 0 / 0** | 1 / 14 / 6 | 1 / 14 / 5 |
| oracle, feature-drift trigger | benign drift, post-drift accuracy | 0.993 / 0.905 / 1.00 | same | **0.995 / 0.975 / 1.00** | same as quarantine | same |

10 seeds, oracle feed: quarantine matches the one-shot gate (repeat attack
1.000 +/- 0 on all three; IoT-23 post-drift 0.976 vs 0.977).

What each part buys, read off the ablations:
- **The sequential test** is what makes the gate robust to a wrong feed: it
  keeps zero poison at 20-40% feed error, where the one-shot veto at 0.3 keeps
  2-9 updates and even a 0.5 veto keeps 1-2.
- **Rollback + a hold longer than the feed lag** is what handles a *late*
  feed: without rollback, 4-14 poisoned updates stay on IoT-23.
- **The lowered provisional threshold** is what lets real benign drift in
  under the feature-drift trigger (IoT-23 0.905 -> 0.975); the 0.55-threshold
  ablation loses that gain.

Limits, stated plainly:
- The seed-42 table above flatters the 40% column. Over 5 seeds (benign drift
  with the feature-drift trigger on real data):

  | feed error | repeat attack, quarantine vs one-shot (N-BaIoT / IoT-23 / synth) | benign drift, quarantine vs one-shot |
  |---|---|---|
  | 20% | 1.000 / 1.000 / 1.000 vs 0.42 / 0.62 / 0.40 | 0.995 / 0.976 / 1.000 vs 0.997 / 0.960 / 1.000 |
  | 30% | 1.000 / 1.000 / 1.000 vs 0.22 / 0.32 / 0.20 | 0.995 / 0.976 / 1.000 vs 0.997 / 0.960 / 1.000 |
  | 40% | **0.88** / 1.000 / 1.000 vs 0.22 / 0.32 / 0.20 | 0.996 / **0.899** / **0.398** vs 0.997 / 0.960 / 1.000 |

  Within the assumed 30% feed error the quarantine gate is strictly better.
  Above it the SPRT's guarantee is gone: it starts condemning *benign* drift
  (synth post-drift 0.40, one seed 0.00) and on N-BaIoT one seed in five lets
  the attack through. `feed_error` must be set to a real bound on the feed.
- Cost on the oracle feed: IoT-23 benign post-drift 0.975 vs 0.978.
- The feed is still simulated from ground truth; the SPRT's feed model
  (independent per-record errors) is an assumption a real feed may violate.

## Datasets (downloaded, gitignored)

**N-BaIoT** -> `data/nbaiot/`, from
https://archive.ics.uci.edu/ml/machine-learning-databases/00442/

| file | rows x cols | used for |
|---|---|---|
| `Danmini_Doorbell/benign_traffic.csv` | 49,548 x 115 | benign, both scenarios |
| `Danmini_Doorbell/mirai_attacks/scan.csv` (from `mirai_attacks.rar`) | 107,685 x 115 | attack, Scenario 2 |
| `Danmini_Doorbell/gafgyt_attacks/combo.csv` (from `gafgyt_attacks.rar`) | 59,718 x 115 | `check_before_real_data.py` only |
| `Ennio_Doorbell/benign_traffic.csv` | 39,100 x 115 | second device, Scenario 1 |

All 0 NaN / 0 Inf, identical column order. Only these files are fetched: the
full dataset is ~7 GB unpacked. UCI does not support resumed downloads; if a
`.rar` download stalls, bsdtar (`C:\Windows\System32\tar.exe -xf x.rar scan.csv`)
can still extract any member that precedes the truncation (it CRC-checks each one).

**IoT-23** -> `data/iot23/<capture>/conn.log.labeled`, labelled Zeek flows only
(no PCAPs), from https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/
`<capture>/bro/conn.log.labeled`:

| capture | flows | role |
|---|---|---|
| `CTU-IoT-Malware-Capture-34-1` | 23,145 (21,222 Mirai: DDoS + C&C; 1,923 benign) | attack + device-A benign |
| `CTU-Honeypot-Capture-5-1` | 1,374 benign (Amazon Echo) | device B, Scenario 1 |
| `CTU-Honeypot-Capture-4-1` | 452 benign (Philips HUE) | parsed, too small to use |

**CICIoT2023** is not downloaded: UNB requires a registration form.

## What's still NOT implemented (must decide MUST/SHOULD/CAN-PLAN)

- Real out-of-band reputation feed — currently simulated from ground-truth
  traffic origin (label the honest way in your report: "simulated
  out-of-band signal, proof-of-concept"). Its *degradation* is now modelled
  (`data/reputation.py`) and swept, but it is still keyed to ground truth.
- Any hyperparameter tuning of trust_threshold / evidence weights — current
  values are untuned defaults. The reputation sweep suggests the 0.3 veto is
  too low once the feed is imperfect (0.5 is more robust); not yet adopted,
  because changing it would change the committed real-data results.
- On N-BaIoT, seeds only reshuffle calibration rows (the files are fixed), so
  its CIs measure calibration-order sensitivity, not dataset variability.
