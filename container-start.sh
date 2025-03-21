#!/bin/bash
# Simple script to start Wyoming Orpheus TTS in a container environment

# Run the server
python -m wyoming_orpheus \
  --uri "tcp://0.0.0.0:10200" \
  --voice "${VOICE:-tara}" \
  --n-threads "${N_THREADS:-4}" \
  --model-path "${MODEL_PATH:-orpheus-3b-0.1-ft-q4_K_M.gguf}" \
  --model-cache-dir "/models" \
  "$@"
