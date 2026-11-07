# sAIfe API for Cloud Run: FastAPI only (the demo UI in web/ is on Vercel).
# Runs from source (PYTHONPATH=src) so the server finds bench/datasets/traps and bench/results next to it.
# The CLI, SDK extras, dashboard/, tests and bench runner stay in the repo, outside the image (.dockerignore).
FROM python:3.12-slim

WORKDIR /app

# Locked from uv.lock: uv export --no-dev --no-hashes --frozen --no-emit-project -o requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY bench/datasets/traps/ bench/datasets/traps/
COPY bench/results/*.json bench/results/

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health', timeout=5)" || exit 1

CMD exec uvicorn saife.server.app:app --host 0.0.0.0 --port ${PORT:-8080}
