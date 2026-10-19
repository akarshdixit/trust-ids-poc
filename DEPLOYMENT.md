# Deploying the Trust-Aware IDS

Two ways to deploy:

1. **Streamlit Community Cloud** (free, hosted, the Streamlit app only): below.
2. **Docker** (your own machine or server, backend + app): further down.

## Streamlit Community Cloud

Community Cloud builds the app straight from GitHub: it reads
`requirements.txt` (Python 3.12, Debian Linux) and runs `app.py`. It does not
use the Dockerfile; the Docker files do not interfere.

### What works there, and how

| feature | on Community Cloud |
|---|---|
| Live stream, Summary, Model comparison, Data & features tabs | yes: precomputed `results/` are in the repository |
| Live re-run on **synthetic** data | yes |
| Live re-run on **IoT-23** | yes: the three flow logs (~3 MB) are downloaded from the dataset's official host on first use (`data/fetch.py`) |
| Live re-run on **N-BaIoT** | no: the dataset is ~300 MB (attack files only inside .rar archives) and is not in the repository; the app says so and shows the precomputed run |
| Live experiment | yes: with no backend, the replay of the real IoT-23 capture runs inside the app ("Run the live replay here") |
| Backend console, `/live` page, ingest API | no: Community Cloud runs only Streamlit; use Docker for these |

Measured with the repository contents only (no datasets, no backend): every
tab renders, the IoT-23 download and re-run work, the in-app live replay
processes all 21,140 flows, and peak memory is about 280 MB (Community Cloud
gives at least 690 MB).

### Steps

1. Sign in at https://share.streamlit.io with a GitHub account that has access
   to `akarshdixit/trust-ids-poc` (the owner's, or a collaborator's).
2. Click **Create app**, then **Yup, I have an app**.
3. Repository `akarshdixit/trust-ids-poc`, branch `main`, main file path `app.py`.
   Optionally choose an App URL, e.g. `trust-ids`.
4. **Advanced settings**: Python version **3.12** (the version the project is
   pinned and tested on). No secrets are needed.
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few
   minutes; later pushes to `main` redeploy automatically.

The app sleeps after 12 hours without visitors; opening the URL wakes it.

## Docker

The application is two services built from **one Docker image**:

| service | what it is | port |
|---|---|---|
| `backend` | FastAPI + uvicorn: REST API, live log console (`/`), live experiment dashboard (`/live`), Zeek ingest (`/api/live/ingest`) | 8765 |
| `app` | Streamlit dashboard (replay, summary, model comparison, data & features, live experiment) | 8501 |

The app calls the backend over a private Docker network (`http://backend:8765`).
Browsers reach each service directly (locally) or through an HTTPS reverse proxy
(public server).

```
 browser ──HTTPS──► Caddy ──► app:8501 (Streamlit) ──► backend:8765 (FastAPI)
                         └──► backend:8765 (/, /live, /api/*)
                                   │
                    data/nbaiot, data/iot23 (read-only mounts)
                    volumes: runs (saved runs), logs
```

## Files

| file | role |
|---|---|
| `Dockerfile` | `python:3.12-slim`, pinned `requirements.txt`, code + precomputed `results/`, non-root user `ids` |
| `.dockerignore` | keeps `.venv`, `.git`, datasets, logs and secrets out of the image |
| `docker-compose.yml` | the two services, health checks, restart policy, memory limits, data mounts, volumes |
| `docker-compose.prod.yml` | public-server overlay: Caddy with automatic HTTPS and a login; services no longer exposed directly |
| `deploy/Caddyfile` | the reverse-proxy config used by the overlay |
| `.env.example` | every setting; copy to `.env` (gitignored) |

## Why it is built this way

* **One image, two commands.** Backend and app share all code and dependencies,
  so one build serves both; compose only changes the command.
* **Datasets are mounted, not baked in.** They are hundreds of MB and their
  licences ask users to download them from the source. The image contains the
  precomputed `results/`, so the dashboards work even without data; live
  re-runs and the live experiment need the mounts.
* **State lives in volumes.** Saved runs (`/app/state/runs`) and logs
  (`/app/logs`) survive container restarts and rebuilds.
* **Non-root, read-only data.** The services run as user `ids` and can only
  write to their state and log directories; datasets are mounted `:ro`.
* **Health checks gate start-up.** The app starts only after the backend's
  `/api/health` answers; both restart automatically (`unless-stopped`).
* **Pinned everything.** Python packages are pinned in `requirements.txt`
  (all 67 resolve to prebuilt manylinux wheels for Python 3.12, so the build
  needs no compilers); the Caddy image is pinned to `2.8.4-alpine`.

## Step 1: prerequisites

* Docker Engine 24+ with Compose v2.24+ (Docker Desktop on Windows/macOS).
* About 2 GB of free RAM for the two containers (each is capped at 1 GB;
  change `BACKEND_MEM` / `APP_MEM` in `.env`).
* The datasets in `data/nbaiot/` and `data/iot23/`, as described in the
  README's "Datasets" section (optional; see above).

## Step 2: configure

```powershell
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(24))"   # paste as TRUST_IDS_TOKEN
```

`TRUST_IDS_TOKEN` turns on the API key: every `/api/*` call except
`/api/health` then needs the header `X-API-Key: <token>` (or `?token=<token>`
in a browser URL, e.g. `http://localhost:8765/live?token=...`). The app sends
it to the backend automatically.

## Step 3: build and run locally

```powershell
docker compose up --build -d      # build the image, start both services
docker compose ps                 # both should become "healthy"
docker compose logs -f backend    # live logs (Ctrl+C to stop following)
```

* App: http://localhost:8501
* Backend console: http://localhost:8765, live experiment: http://localhost:8765/live

Stop with `docker compose down` (add `-v` to also delete saved runs and logs).

## Step 4: verify

```powershell
curl http://localhost:8765/api/health        # real_data_available / iot23_available: true if mounted
docker compose exec backend python -m pytest -q -p no:cacheprovider
```

The test suite runs inside the container exactly as on a laptop; tests that
need datasets skip when the mounts are empty.

## Step 5: deploy on a public server

On a Linux VM (any cloud: AWS EC2, Azure VM, Google Compute Engine,
DigitalOcean; 2 vCPU / 4 GB RAM is comfortable):

1. Install Docker: `curl -fsSL https://get.docker.com | sh`.
2. Point two DNS names at the server's IP, e.g. `ids.example.com` (app) and
   `ids-api.example.com` (backend). Open ports 80 and 443 in the firewall;
   do **not** open 8501 or 8765.
3. Copy the project (`git clone https://github.com/akarshdixit/trust-ids-poc.git`)
   and the datasets (`scp -r data/nbaiot data/iot23 user@server:trust-ids-poc/data/`).
4. In `.env` set `TRUST_IDS_TOKEN`, `APP_DOMAIN`, `API_DOMAIN`,
   `BASIC_AUTH_USER` and the password hash:
   ```bash
   docker run --rm caddy:2.8.4-alpine caddy hash-password --plaintext 'your-password'
   ```
   Paste the hash as `BASIC_AUTH_HASH`, writing every `$` as `$$`.
5. Start with the production overlay:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   ```
   Caddy obtains HTTPS certificates on first request, asks for the login on
   both domains, proxies Streamlit's websockets, and streams the backend's
   Server-Sent Events unbuffered.
6. Update later with `git pull` and the same `up -d --build` command.

## Security checklist

* Set `TRUST_IDS_TOKEN`; without it the API is open to anyone who can reach it.
* Never publish 8501/8765 on a public server; only Caddy (80/443).
* The Streamlit app has no login of its own; the Caddy `basic_auth` provides it.
* `.env` is gitignored and excluded from the image; keep secrets only there.
* `TRUST_IDS_MAX_RUNS` (default 2) caps concurrent runs; each live replay is
  limited to one at a time.

## Without Docker

The same two commands run directly on any machine with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.server:app --host 0.0.0.0 --port 8765
$env:TRUST_IDS_BACKEND="http://localhost:8765"; .\.venv\Scripts\streamlit.exe run app.py --server.address 0.0.0.0
```

For a long-running server, wrap each in a service manager (systemd on Linux,
NSSM or a scheduled task on Windows) and put the same Caddyfile in front.

## What is not containerized

* Live packet capture: the ingest API (`/api/live/ingest`) accepts lines from a
  Zeek sensor, but Zeek itself runs on the monitored network, not in this
  image. Run `experiments/live_feed.py <conn.log> --follow` next to the sensor
  with `TRUST_IDS_BACKEND` pointing at the backend.
* Heavy offline experiments (benchmark, sweeps) work in the container
  (`docker compose exec backend python experiments/benchmark.py --source real`)
  but are meant to be run once; their results are already in `results/`.
