FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY . .
RUN uv sync --frozen --no-dev --no-cache

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app
RUN ln -sfn /app/.venv/bin/python /usr/local/bin/python \
    && ln -sfn /app/.venv/bin/python /usr/local/bin/python3 \
    && python -c \
        "import sys, torch, torchaudio; print(f'python={sys.executable} torch={torch.__version__} torchaudio={torchaudio.__version__}')"

EXPOSE 8020
ENTRYPOINT ["/app/.venv/bin/python", "-m", "server.serve"]
CMD ["--model", "flash", "--host", "0.0.0.0", "--port", "8020"]
