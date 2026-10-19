r"""
Trust-Aware IDS — backend API.

Runs the same pipeline as experiments/run_scenarios.py as a background job and
streams every round to clients as it happens (Server-Sent Events), so a
frontend — or just a terminal with curl — can watch classification, drift
detection and gate decisions live instead of loading a finished JSON log.

Run:  .\.venv\Scripts\python.exe -m uvicorn backend.server:app --reload --port 8765

Endpoints:
  GET  /                          live console (HTML) — start a run, watch logs
  GET  /api/health                (never needs a token)
  GET  /api/results               precomputed logs available in results/
  GET  /api/results/{scenario}/{tag}
  POST /api/runs                  start a pipeline run -> {"run_id": ...}
  GET  /api/runs                  every run, including ones from earlier processes
  GET  /api/runs/{run_id}         status + full log so far
  GET  /api/runs/{run_id}/events  SSE: one event per round, then "done"/"error"
  GET  /api/logs/stream           SSE: every server log line, live

Environment:
  TRUST_IDS_TOKEN      if set, every /api/* call except health needs it, as an
                       X-API-Key header or ?token= (EventSource can't send headers)
  TRUST_IDS_MAX_RUNS   concurrent runs allowed (default 2); more -> HTTP 429
  TRUST_IDS_RUNS_DIR   where finished runs are saved (default results/runs/)

Sources: real (default; N-BaIoT in data/nbaiot/, not in git), iot23 (labelled Zeek
flows in data/iot23/, not in git), synth (always). fake (generated N-BaIoT-format
files) is kept for tests and machines without the dataset; the UI no longer offers it.
"""
import asyncio
import collections
import json
import logging
import os
import secrets
import sys
import threading
import time
import traceback
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
RESULTS = os.path.join(BASE, "results")
LOG_DIR = os.path.join(BASE, "logs")
RUNS_DIR = os.environ.get("TRUST_IDS_RUNS_DIR", os.path.join(RESULTS, "runs"))
MAX_ACTIVE_RUNS = int(os.environ.get("TRUST_IDS_MAX_RUNS", "2"))
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

from experiments.run_scenarios import run_scenario  # noqa: E402
from experiments.streams import build_stream, real_data_available  # noqa: E402

ARMS = ["static", "ungated", "gate_traffic_only", "gate_full", "ewc", "gate_quarantine"]

# ------------------------------------------------------------------ logging
# Every log line goes to the console, to logs/backend.log, and into a ring
# buffer that /api/logs/stream tails — that is the "real-time logs" channel.
LOG_RING = collections.deque(maxlen=2000)
_log_seq = 0
_log_lock = threading.Lock()


class _RingHandler(logging.Handler):
    def emit(self, record):
        global _log_seq
        with _log_lock:
            _log_seq += 1
            LOG_RING.append((_log_seq, self.format(record)))


_fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", "%H:%M:%S")
log = logging.getLogger("trust_ids")
log.setLevel(logging.INFO)
if not log.handlers:  # uvicorn --reload / test re-imports must not stack handlers
    for h in (logging.StreamHandler(),
              logging.FileHandler(os.path.join(LOG_DIR, "backend.log"), encoding="utf-8"),
              _RingHandler()):
        h.setFormatter(_fmt)
        log.addHandler(h)
log.propagate = False


# ------------------------------------------------------------------ auth
def require_token(request: Request):
    expected = os.environ.get("TRUST_IDS_TOKEN")
    if not expected:
        return
    given = request.headers.get("x-api-key") or request.query_params.get("token") or ""
    if not secrets.compare_digest(given, expected):
        raise HTTPException(401, "missing or wrong API token")


# ------------------------------------------------------------------ runs
class GateParams(BaseModel):
    trust_threshold: float = Field(0.55, ge=0, le=1)
    reputation_veto_threshold: float = Field(0.30, ge=0, le=1)
    w_attribution: float = Field(0.40, ge=0, le=1)
    w_topology: float = Field(0.30, ge=0, le=1)
    w_temporal: float = Field(0.30, ge=0, le=1)

    @model_validator(mode="after")
    def _weights_not_all_zero(self):
        if self.w_attribution + self.w_topology + self.w_temporal <= 0:
            raise ValueError("gate weights must not all be zero (they are normalised to sum 1)")
        return self


class ReputationNoiseParams(BaseModel):
    sigma: float = Field(0.0, ge=0, le=1)
    flip_rate: float = Field(0.0, ge=0, le=1)
    lag: int = Field(0, ge=0, le=1000)
    seed: int = 0


class QuarantineParams(BaseModel):
    provisional_threshold: float = Field(0.45, ge=0, le=1)
    quarantine_rounds: int = Field(20, ge=0, le=100)
    feed_error: float = Field(0.3, gt=0, lt=0.5)
    alpha: float = Field(0.01, gt=0, lt=1)
    beta: float = Field(0.01, gt=0, lt=1)


class RunRequest(BaseModel):
    scenario: Literal["slow_poisoning", "genuine_drift"] = "slow_poisoning"
    source: Literal["real", "iot23", "synth", "fake"] = "real"
    gate: GateParams = GateParams()
    reputation_noise: ReputationNoiseParams = ReputationNoiseParams()
    seed: int | None = None
    feature_drift: bool = False
    include_ewc: bool = True
    ewc_lambda: float = Field(50.0, ge=0, le=1000)
    include_quarantine: bool = True
    quarantine: QuarantineParams = QuarantineParams()
    pace_ms: int = Field(150, ge=0, le=5000,
                         description="delay per round, so the stream is watchable")


class Run:
    def __init__(self, req: RunRequest, run_id=None):
        self.id = run_id or uuid.uuid4().hex[:8]
        self.req = req
        self.status = "queued"
        self.error = None
        self.events = []          # one dict per round, appended by the worker thread
        self.started = time.time()
        self.finished = None

    def summary(self):
        return {"run_id": self.id, "status": self.status, "error": self.error,
                "request": self.req.model_dump(), "rounds": len(self.events),
                "started": self.started, "finished": self.finished}

    def save(self, status=None):
        """status overrides the saved status, so a run can be persisted BEFORE
        clients are told it finished."""
        path = os.path.join(RUNS_DIR, f"{self.id}.json")
        d = {**self.summary(), "events": self.events}
        if status is not None:
            d["status"] = status
        with open(path + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        run = cls(RunRequest(**d["request"]), run_id=d["run_id"])
        run.status, run.error = d["status"], d["error"]
        run.events, run.started, run.finished = d["events"], d["started"], d["finished"]
        return run


RUNS: dict[str, Run] = {}
_runs_lock = threading.Lock()


def _load_saved_runs():
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith(".json"):
            try:
                run = Run.load(os.path.join(RUNS_DIR, name))
                RUNS[run.id] = run
            except Exception as e:  # one corrupt file must not stop the server
                log.warning("skipping unreadable saved run %s: %s", name, e)


def _describe(round_idx, entries):
    g, t = entries["gate_full"], entries["gate_traffic_only"]
    accs = " ".join(f"{a}={entries[a]['accuracy']:.3f}" for a in ARMS if a in entries)
    q = entries.get("gate_quarantine")
    rollback = (f" ROLLBACK {q['rolled_back']} provisional update(s)"
                if q and q.get("rolled_back") else "")
    if not g.get("drift_active"):
        return f"r{round_idx:>3} {g['phase']:<15} no drift  | {accs}{rollback}"
    verdict = ("VETO (traffic-only accepted)" if t["authorized"] and not g["authorized"]
               else "AUTH" if g["authorized"] else "refused")
    qv = ""
    if q:
        qv = (" quarantine=PROVISIONAL" if q["authorized"]
              else " quarantine=CONDEMNED" if q["condemned"] else " quarantine=held")
        qv += f"(llr={q['sprt_llr']:+.2f})"
    return (f"r{round_idx:>3} {g['phase']:<15} DRIFT trust={g['trust_score']:.3f} "
            f"rep={g['reputation']:.2f} full_gate={verdict}{qv}{rollback} | {accs}")


def _worker(run: Run):
    rl = logging.getLogger(f"trust_ids.run.{run.id}")
    run.status = "running"
    final_status = "error"
    rl.info("start scenario=%s source=%s seed=%s feature_drift=%s noise=%s gate=%s",
            run.req.scenario, run.req.source, run.req.seed, run.req.feature_drift,
            run.req.reputation_noise.model_dump(), run.req.gate.model_dump())
    try:
        stream, names = build_stream(run.req.scenario, run.req.source, seed=run.req.seed,
                                     reputation_noise=run.req.reputation_noise.model_dump())
        rl.info("stream built: %d records, %d features", len(stream), len(names))

        def on_round(round_idx, entries):
            run.events.append({"round": round_idx, "arms": entries})
            rl.info(_describe(round_idx, entries))
            if run.req.pace_ms:
                time.sleep(run.req.pace_ms / 1000)

        final = run_scenario(stream, names, run.req.scenario,
                             gate_params=run.req.gate.model_dump(), on_round=on_round,
                             include_ewc=run.req.include_ewc, ewc_lambda=run.req.ewc_lambda,
                             feature_drift=run.req.feature_drift,
                             include_quarantine=run.req.include_quarantine,
                             quarantine_params=run.req.quarantine.model_dump())
        final_status = "done"
        rl.info("done: " + "  ".join(f"{a}={final[a][-1]['accuracy']:.3f}" for a in final))
    except Exception as e:  # surface the failure to the client instead of dying silently
        final_status, run.error = "error", f"{type(e).__name__}: {e}"
        rl.error("failed: %s\n%s", run.error, traceback.format_exc())
    finally:
        run.finished = time.time()
        try:
            run.save(status=final_status)
        except OSError as e:
            rl.error("could not save run: %s", e)
        run.status = final_status  # only now do pollers and SSE clients see the end


# ------------------------------------------------------------------ app
app = FastAPI(title="Trust-Aware IDS backend")
authed = [Depends(require_token)]


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _get_run(run_id):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return run


@app.get("/")
def console():
    return FileResponse(os.path.join(os.path.dirname(__file__), "console.html"))


@app.get("/api/health")
def health():
    from data.load_iot23 import available as iot23_available
    return {"ok": True, "real_data_available": real_data_available(),
            "iot23_available": iot23_available(),
            "auth_required": bool(os.environ.get("TRUST_IDS_TOKEN")),
            "max_active_runs": MAX_ACTIVE_RUNS,
            "active_runs": sum(r.status in ("queued", "running") for r in RUNS.values())}


@app.get("/api/results", dependencies=authed)
def list_results():
    return sorted(f[:-5] for f in os.listdir(RESULTS) if f.endswith(".json"))


@app.get("/api/results/{scenario}/{tag}", dependencies=authed)
def get_result(scenario: Literal["slow_poisoning", "genuine_drift"],
               tag: Literal["real", "iot23", "synth", "fake"]):
    path = os.path.join(RESULTS, f"{scenario}_log_{tag}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"no precomputed log {os.path.basename(path)}")
    return FileResponse(path, media_type="application/json")


@app.post("/api/runs", status_code=202, dependencies=authed)
def start_run(req: RunRequest):
    with _runs_lock:
        active = sum(r.status in ("queued", "running") for r in RUNS.values())
        if active >= MAX_ACTIVE_RUNS:
            raise HTTPException(429, f"{active} runs already active (limit {MAX_ACTIVE_RUNS})")
        run = Run(req)
        RUNS[run.id] = run
    threading.Thread(target=_worker, args=(run,), daemon=True).start()
    return {"run_id": run.id, "events": f"/api/runs/{run.id}/events"}


@app.get("/api/runs", dependencies=authed)
def list_runs():
    return [r.summary() for r in RUNS.values()]


@app.get("/api/runs/{run_id}", dependencies=authed)
def get_run(run_id: str):
    run = _get_run(run_id)
    return {**run.summary(), "events": run.events}


@app.get("/api/runs/{run_id}/events", dependencies=authed)
async def run_events(run_id: str):
    run = _get_run(run_id)

    async def gen():
        sent = 0
        while True:
            while sent < len(run.events):
                yield _sse("round", run.events[sent])
                sent += 1
            if run.status in ("done", "error") and sent >= len(run.events):
                yield _sse(run.status, run.summary())
                return
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/logs/stream", dependencies=authed)
async def log_stream(backlog: int = 200, follow: bool = True):
    async def gen():
        with _log_lock:
            items = list(LOG_RING)[-backlog:]
        last = items[-1][0] if items else 0
        for _, line in items:
            yield _sse("log", line)
        while follow:
            with _log_lock:
                new = [(s, l) for s, l in LOG_RING if s > last]
            for s, line in new:
                last = s
                yield _sse("log", line)
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ live
# Two live paths:
#  * POST /api/live replays the real IoT-23 Mirai capture in its original time
#    order, at a chosen speed, through the full six-arm pipeline
#    (experiments/live_experiment.py), streamed to /live.
#  * POST /api/live/ingest scores raw Zeek conn.log lines sent by a sensor or by
#    experiments/live_feed.py, with a detector trained on the capture's past.
class LiveRequest(BaseModel):
    speed: float = Field(600.0, ge=0, le=100000,
                         description="capture seconds per wall-clock second; 0 = no pacing")
    max_flows: int | None = Field(None, ge=2100, le=100000)
    reputation_noise: ReputationNoiseParams = ReputationNoiseParams()


class LiveRun:
    def __init__(self, req: LiveRequest):
        self.id = "live-" + uuid.uuid4().hex[:6]
        self.req, self.status, self.error = req, "running", None
        self.events, self.summary = [], None


LIVE: dict[str, LiveRun] = {}


def _live_worker(lr: LiveRun):
    from experiments.live_experiment import LiveExperiment
    ll = logging.getLogger(f"trust_ids.live.{lr.id}")
    try:
        noise = lr.req.reputation_noise.model_dump()
        exp = LiveExperiment(speed=lr.req.speed, max_flows=lr.req.max_flows,
                             reputation_noise=noise if (noise["flip_rate"] or noise["sigma"]
                                                        or noise["lag"]) else None)
        ll.info("live replay: %d flows at %gx", len(exp.stream), lr.req.speed)

        def on_event(ev):
            lr.events.append(ev)
            if ev["round"] % 25 == 0:
                q = ev["arms"].get("gate_quarantine", {})
                ll.info("r%4d capture %5.2fh lag %.2fs %.2f ms/flow malicious %d/20%s",
                        ev["round"], ev["capture_hours"], ev["lag_s"],
                        ev["busy_ms_per_flow"], ev["malicious"],
                        " ROLLBACK" if q.get("rolled_back") else "")

        lr.summary = exp.run(on_event=on_event)
        lr.status = "done"
        ll.info("live done: %.0f flows/s, max lag %.2fs", lr.summary["throughput_flows_per_s"],
                lr.summary["max_lag_s"])
    except Exception as e:  # noqa: BLE001 - reported to the client
        lr.status, lr.error = "error", f"{type(e).__name__}: {e}"
        ll.error("live failed: %s\n%s", lr.error, traceback.format_exc())


@app.get("/live")
def live_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "live.html"))


@app.post("/api/live", status_code=202, dependencies=authed)
def start_live(req: LiveRequest):
    from data.load_iot23 import available as iot23_available
    if not iot23_available():
        raise HTTPException(409, "IoT-23 captures not downloaded to data/iot23/")
    with _runs_lock:
        if any(r.status == "running" for r in LIVE.values()):
            raise HTTPException(429, "a live replay is already running")
        lr = LiveRun(req)
        LIVE[lr.id] = lr
    threading.Thread(target=_live_worker, args=(lr,), daemon=True).start()
    return {"live_id": lr.id, "events": f"/api/live/{lr.id}/events"}


def _get_live(live_id):
    lr = LIVE.get(live_id)
    if lr is None:
        raise HTTPException(404, f"unknown live run {live_id}")
    return lr


@app.get("/api/live/{live_id}", dependencies=authed)
def get_live(live_id: str):
    lr = _get_live(live_id)
    return {"live_id": lr.id, "status": lr.status, "error": lr.error,
            "request": lr.req.model_dump(), "rounds": len(lr.events),
            "summary": lr.summary, "events": lr.events}


@app.get("/api/live/{live_id}/events", dependencies=authed)
async def live_events(live_id: str):
    lr = _get_live(live_id)

    async def gen():
        sent = 0
        while True:
            while sent < len(lr.events):
                yield _sse("round", lr.events[sent])
                sent += 1
            if lr.status != "running" and sent >= len(lr.events):
                yield _sse(lr.status, {"summary": lr.summary, "error": lr.error})
                return
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


class IngestRequest(BaseModel):
    fields: list[str] = Field(..., min_length=5, description="the conn.log #fields header")
    lines: list[str] = Field(..., min_length=1, max_length=5000)


_DETECTOR = {}
_detector_lock = threading.Lock()


def _live_detector():
    """Online LR trained on the first 2,000 flows (in time) of the Mirai capture."""
    with _detector_lock:
        if "ids" not in _DETECTOR:
            from experiments.live_experiment import N_CALIB, chronological_stream
            from models.online_ids import OnlineIDS
            stream, names, _ = chronological_stream(N_CALIB, max_flows=N_CALIB)
            ids = OnlineIDS(drift_detector=False)
            for x, y, _ in stream:
                ids.learn_one(x, y)
            _DETECTOR.update(ids=ids, names=names, seen=0, alerts=0, correct=0, labelled=0)
            log.info("live detector trained on %d calibration flows", len(stream))
        return _DETECTOR


@app.post("/api/live/ingest", dependencies=authed)
def ingest(req: IngestRequest):
    from data.load_iot23 import featurize
    det = _live_detector()
    df = featurize(req.fields, [ln.split() for ln in req.lines])
    if df.empty:
        raise HTTPException(422, "no line matched the #fields header")
    names, ids = det["names"], det["ids"]
    t0 = time.perf_counter()
    out = []
    for vec, label in zip(df[names].to_numpy(dtype=float), df["label"].to_numpy()):
        x = dict(zip(names, map(float, vec)))
        p = ids.model.predict_proba_one(x).get(1, 0.0)
        out.append({"attack": bool(p >= 0.5), "score": round(float(p), 4),
                    **({"label": int(label)} if label >= 0 else {})})
    latency_us = (time.perf_counter() - t0) / len(out) * 1e6
    with _detector_lock:
        det["seen"] += len(out)
        det["alerts"] += sum(o["attack"] for o in out)
        lab = [o for o in out if "label" in o]
        det["labelled"] += len(lab)
        det["correct"] += sum(o["attack"] == bool(o["label"]) for o in lab)
        running = {"flows": det["seen"], "alerts": det["alerts"],
                   "accuracy_on_labelled": (det["correct"] / det["labelled"]
                                            if det["labelled"] else None)}
    log.info("ingest %d flows: %d alerts, %.1f us/flow (total %d)", len(out),
             sum(o["attack"] for o in out), latency_us, running["flows"])
    return {"predictions": out, "latency_us_per_flow": latency_us, "running": running}


_load_saved_runs()
log.info("backend ready: results dir %s, %d saved runs loaded", RESULTS, len(RUNS))
