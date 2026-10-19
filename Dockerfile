# Trust-Aware IDS: one image, two services (backend API + Streamlit app).
#
#   docker compose up --build            # see docker-compose.yml and DEPLOYMENT.md
#
# Datasets are NOT baked in (hundreds of MB, and their licences ask you to fetch
# them from the source): mount data/nbaiot and data/iot23 at run time. The
# precomputed results/ ARE in the image, so the app works without the data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TRUST_IDS_RUNS_DIR=/app/state/runs

WORKDIR /app

# dependencies first, so code edits do not reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# unprivileged user; it owns only the directories the services write to
RUN useradd --create-home --uid 1000 ids \
 && mkdir -p /app/logs /app/state/runs /app/data/nbaiot /app/data/iot23 /app/data/fake_nbaiot \
 && chown -R ids:ids /app/logs /app/state /app/data/fake_nbaiot
USER ids

EXPOSE 8765 8501

# default: the backend. The app service in docker-compose.yml overrides this.
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8765", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
