# syntax=docker/dockerfile:1
# Ming Voice for a Linux host with an NVIDIA or AMD GPU (see "Docker" in README.md). The image holds the code and
# its Python env; the model (~20 GB, downloaded at the first start), your settings, keys, cloned voices and the
# compiled-kernel caches live in the /data volume.
#   docker build -t ming-voice:cuda .                       NVIDIA (CUDA 12.8; RTX 30 series or newer)
#   docker build --build-arg GPU=amd -t ming-voice:rocm .   AMD (ROCm 7.2)
FROM ubuntu:24.04
ARG GPU=nvidia
# The build's version and git revision: reported by the server (start log, /api/health, the realtime
# session.created) so you can tell which image is actually running.
ARG MING_VERSION=dev
ARG MING_GIT_REVISION=dev
LABEL org.opencontainers.image.source=https://github.com/JMS1717/ming-voice \
      org.opencontainers.image.description="Ming Voice: a local, realtime speech-to-speech assistant" \
      org.opencontainers.image.licenses=MIT \
      org.opencontainers.image.version=$MING_VERSION \
      org.opencontainers.image.revision=$MING_GIT_REVISION
# gcc: Triton compiles a small launcher at the first GPU kernel. openssl: the phone-access certificate.
# tini: passes `docker stop` on to the server.
# apt-get update exits 0 even when every source fails to fetch (e.g. transient DNS on the runner),
# leaving apt install with no candidates and a confusing "no installation candidate" error; retry
# until the indexes actually resolve before installing
RUN set -eu; \
    attempt=1; \
    until apt-get update && apt-get install --dry-run -y --no-install-recommends ca-certificates; do \
        attempt=$((attempt + 1)); \
        [ "$attempt" -gt 5 ] && { echo "apt repositories unreachable after 5 attempts"; exit 1; }; \
        echo "apt update could not fetch the package indexes (attempt $((attempt - 1))), retrying in 5s..."; \
        sleep 5; \
    done; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl gcc libc6-dev openssl tini tzdata; \
    rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
# The Python env first: it changes far less often than the code
COPY setup_env.sh requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv MING_GPU=$GPU MING_SKIP_MODEL=1 bash setup_env.sh
COPY . .
ENV MING_GPU=$GPU MING_DATA_DIR=/data MING_MODEL_DIR=/data/model \
    MING_TZ=America/New_York \
    MING_TURN_MODEL=/data/model_extra/smart-turn-v3.2-cpu.onnx \
    MING_VERSION=$MING_VERSION MING_GIT_REVISION=$MING_GIT_REVISION \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache TRITON_CACHE_DIR=/data/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    MIOPEN_USER_DB_PATH=/data/cache/miopen MIOPEN_CUSTOM_CACHE_DIR=/data/cache/miopen \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000 8443 10300
HEALTHCHECK --start-period=20m --interval=1m \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/health" | grep -Eq '"ready": ?true'
ENTRYPOINT ["tini", "-g", "--", "docker/entrypoint.sh"]
