"""
Feed Zeek conn.log lines to the running backend as they are written.

Point it at the conn.log of a live Zeek sensor (--follow keeps reading new
lines, like `tail -f`), or at a finished log to replay it at a fixed rate:

  python experiments/live_feed.py /opt/zeek/logs/current/conn.log --follow
  python experiments/live_feed.py data/iot23/CTU-Honeypot-Capture-5-1/conn.log.labeled --rate 200

Each batch is POSTed to /api/live/ingest, which featurizes it exactly like the
IoT-23 loader and scores it with a detector trained on the Mirai capture's
first 2,000 flows. If the log is labelled (IoT-23), running accuracy is reported
too. Uses only the standard library, so it runs on a sensor without the venv.
"""
import argparse
import json
import os
import time
import urllib.request


def post(url, body, token=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"X-API-Key": token} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def lines(path, follow, poll=0.5):
    """Yield ('fields', [...]) once, then ('row', line) for every data line."""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                if not follow:
                    return
                time.sleep(poll)
                continue
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                yield "fields", line.split()[1:]
            elif line and not line.startswith("#"):
                yield "row", line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--backend", default=os.environ.get("TRUST_IDS_BACKEND",
                                                        "http://localhost:8765"))
    ap.add_argument("--follow", action="store_true", help="keep reading new lines")
    ap.add_argument("--rate", type=float, default=0,
                    help="lines per second when replaying a finished file (0 = as fast as possible)")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args()
    token = os.environ.get("TRUST_IDS_TOKEN")
    url = args.backend.rstrip("/") + "/api/live/ingest"

    fields, batch, sent, t0 = None, [], 0, time.time()

    def flush():
        nonlocal batch, sent
        if not batch:
            return
        r = post(url, {"fields": fields, "lines": batch}, token)
        sent += len(batch)
        acc = r["running"]["accuracy_on_labelled"]
        print(f"{sent:7d} flows  {sum(p['attack'] for p in r['predictions']):3d}/{len(batch)} alerts  "
              f"{r['latency_us_per_flow']:.0f} us/flow"
              + (f"  running accuracy {acc:.3f}" if acc is not None else ""), flush=True)
        batch = []

    for kind, value in lines(args.path, args.follow):
        if kind == "fields":
            flush()
            fields = value
            continue
        if fields is None:
            continue
        batch.append(value)
        if args.rate:
            time.sleep(1 / args.rate)
        if len(batch) >= args.batch:
            flush()
        if args.max_lines and sent + len(batch) >= args.max_lines:
            break
    flush()
    print(f"done: {sent} flows in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
