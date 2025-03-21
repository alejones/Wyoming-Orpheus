FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" 

WORKDIR /app


# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfiles
COPY Pipfile* ./

# Install dependencies using pipenv (without creating a virtualenv inside the container)
RUN pipenv install --deploy --system

# Copy the application code
COPY wyoming_orpheus/ ./wyoming_orpheus/
COPY setup.py .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directory for model caching
RUN mkdir -p /models

# Set environment variables
ENV PYTHONUNBUFFERED=1

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