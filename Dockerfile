# syntax=docker/dockerfile:1

# ---- builder -----------------------------------------------------------
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime -----------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Running as root inside a container is avoidable, so avoid it.
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
# seed/ sits outside every volume mount point, which is what allows the
# entrypoint to populate an empty volume on a new machine.
COPY --chown=appuser:appuser seed ./seed
COPY --chown=appuser:appuser entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
# One worker, deliberately: the model cache and background training tasks live
# in process memory, so a retrain in one worker would leave others serving a
# stale model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]