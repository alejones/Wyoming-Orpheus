# Multi-stage build for Wyoming Orpheus TTS service with CUDA support.
# Builder compiles dependencies; runtime copies only what is needed to keep image lean.

# Stage 1: Build environment
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off -DCUDAToolkit_ROOT=/usr/local/cuda"
ENV PYTHONPATH="/app"

# Install build dependencies (python3.12 is default on Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pipenv into the system Python
RUN pip install --break-system-packages pipenv

# Copy Pipfiles and install dependencies into system site-packages
COPY Pipfile* ./
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pipenv install --deploy --system

# Copy the application code
COPY wyoming_orpheus/ ./wyoming_orpheus/

# Stage 2: Runtime environment
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/app"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages and application from builder
COPY --from=builder /usr/lib/python3 /usr/lib/python3
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create directory for model caching
RUN mkdir -p /models

# Default model parameters
ENV MODEL_PATH="orpheus-3b-0.1-ft-q4_K_M.gguf"
ENV VOICE="tara"
ENV N_THREADS=4
ENV N_GPU_LAYERS=0
ENV PORT=10200

EXPOSE 10200

# Write entrypoint script
RUN cat <<'EOF' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh
#!/bin/bash
python3 -m wyoming_orpheus \
  --uri "tcp://0.0.0.0:$PORT" \
  --voice "$VOICE" \
  --n-threads "$N_THREADS" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --model-path "$MODEL_PATH" \
  --model-cache-dir /models \
  "$@"
EOF

ENTRYPOINT ["/app/entrypoint.sh"]
