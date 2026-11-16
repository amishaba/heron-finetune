# The live demo: web page + API + the real pipeline in one container (demo/README.md).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FT_RUNS_DIR=/data/runs FT_DEMO_DIR=/data/uploads
WORKDIR /app

# Editable install: the code finds prices.yaml, models.yaml and its templates next to itself.
COPY pyproject.toml ./
COPY ft ft
RUN pip install -e ".[together,demo]"

COPY prices.yaml models.yaml ./
COPY demo demo

EXPOSE 8080
# Hosts like Railway pass the port in $PORT; 8080 otherwise.
CMD ["sh", "-c", "exec uvicorn demo.server:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
