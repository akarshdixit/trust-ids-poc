# Trust-Aware IDS — Review-2 Proof-of-Concept

Working, tested pipeline for the core research contribution: an
authorization gate that decides whether a proposed continual-learning
update should be trusted, using traffic-derived evidence cross-checked
against an out-of-band reputation signal.

**Status: runs end-to-end on a SYNTHETIC stream (`data/synth_stream.py`)
that mimics the N-BaIoT feature schema. This proves the pipeline logic is
correct. It is NOT your Review-2 result. You must re-run it against real
N-BaIoT data (see below) before it goes anywhere near your report or PPT.**

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
  nothing is proposed to anyone. Runs FOUR arms (static / ungated /
  traffic-only gate / proposed full gate) through both scenarios (genuine
  drift, slow poisoning + a follow-up "obvious repeat attack" check).
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

## Quickstart (already run once — this is what produced results/)

```bash
pip install river pandas numpy matplotlib
python experiments/run_scenarios.py   # prints summary, writes results/*.json
python experiments/make_plots.py      # writes results/*.png
```

## Swapping in the real N-BaIoT dataset (DO THIS NEXT)

1. Download ONE device from
   https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
   (`Danmini_Doorbell` is a good pick — has both Mirai and Gafgyt attacks).
2. Use `data/load_nbaiot.py::load_device_stream(...)` to build the stream
   instead of `data/synth_stream.py`. It returns the exact same
   `(features_dict, label, meta_dict)` format, so nothing downstream needs
   to change.
3. For "genuine drift", split `benign_traffic.csv` into two halves (or use
   two same-type devices) instead of an attack file.
4. For "slow poisoning", you need to build the SLOW interpolation yourself
   — real N-BaIoT attack files are not temporally paced to be slow. Either:
   (a) interpolate row-by-row between a benign row and an attack row's
   feature vectors (same technique as `synth_stream.py`, just fed real
   rows), or (b) if time is too short, run the "sudden" version first
   (real attack file introduced abruptly) to get a real result, then
   attempt the slow version as a stretch goal.
5. Re-run `run_scenarios.py` and `make_plots.py`. These numbers — not the
   synthetic ones — are what go in your report/PPT.

## What's still NOT implemented (must decide MUST/SHOULD/CAN-PLAN)

- Real out-of-band reputation feed — currently simulated from ground-truth
  traffic origin (label the honest way in your report: "simulated
  out-of-band signal, proof-of-concept").
- EWC/Fisher-freezing baseline (only static/ungated/traffic-only/full-gate
  implemented — EWC is a stretch goal, see main chat message).
- Any hyperparameter tuning of trust_threshold / reputation_veto_threshold
  / evidence weights — current values are untuned defaults, good enough to
  demonstrate the mechanism, not yet validated.
