FROM ghcr.io/astral-sh/uv:latest AS uv

FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

COPY --from=uv /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/rocm/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project \
        --prune torch --prune torchaudio --prune torchvision \
        --output-file /tmp/requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . .
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps . \
    && python -c \
        "import sys, perth, torch, torchaudio; assert torch.version.hip; assert torch.version.cuda is None; assert perth.PerthImplicitWatermarker is not None; perth.PerthImplicitWatermarker(); print(f'python={sys.executable} torch={torch.__version__} hip={torch.version.hip} torchaudio={torchaudio.__version__}')"

EXPOSE 8020
ENTRYPOINT ["/opt/venv/bin/python", "-m", "server.serve"]
CMD ["--model", "flash", "--host", "0.0.0.0", "--port", "8020"]
