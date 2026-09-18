# syntax=docker/dockerfile:1.7
#
# Serving image for the registered classifier. Build through `make docker-build`,
# which first resolves models:/cfpb-complaint-classifier@production into
# build/model/ -- the image holds that immutable copy, not a registry pointer.

ARG PYTHON_IMAGE=python:3.14-slim

# ---- builder: resolve and install the pinned serving deps into a venv --------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv venv /opt/venv
COPY requirements-serve.txt .
# CPU-only torch: the default PyPI wheel on linux/amd64 drags in ~3 GB of CUDA
# libraries a CPU container can never use. Only the lockfile is copied before
# this layer, so code and model changes do not reinstall dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --torch-backend cpu \
        -r requirements-serve.txt

# ---- runtime: interpreter + venv + code + model, nothing else ---------------
FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

# Model before code: it is the largest layer and changes least often.
# Owned by root and read-only to the app user -- the server has no reason to
# modify its own weights.
COPY build/model/ /app/model/
COPY src/ /app/src/
RUN mkdir -p /app/logs && chown app:app /app/logs

ARG GIT_SHA=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_URI=/app/model \
    DEVICE=cpu \
    REQUEST_LOG=/app/logs/requests.jsonl \
    GIT_SHA=${GIT_SHA}

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
