# Stage 1: Build environment
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off -DCUDAToolkit_ROOT=/usr/local/cuda"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfiles
COPY Pipfile* ./

# Install dependencies using pipenv (without creating a virtualenv inside the container)
RUN pipenv install --deploy --system

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=/usr/local/cuda" pip install --force-reinstall llama-cpp-python --no-cache-dir

# Copy the application code
COPY wyoming_orpheus/ ./wyoming_orpheus/
COPY setup.py .

# Install the package
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime environment
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install only the necessary runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy installed Python packages and application from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /app /app

# Create directory for model caching
RUN mkdir -p /models

# Default model parameters
ENV MODEL_PATH="orpheus-3b-0.1-ft-q4_K_M.gguf"
ENV VOICE="tara"
ENV N_THREADS=4
ENV PORT=10200

# Expose the port
EXPOSE 10200

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
python -m wyoming_orpheus \
  --uri "tcp://0.0.0.0:$PORT" \
  --voice "$VOICE" \
  --n-threads "$N_THREADS" \
  --model-path "$MODEL_PATH" \
  --model-cache-dir /models \
  "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]