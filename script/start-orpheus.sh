#!/bin/bash
# Script to start the Wyoming Orpheus TTS server

# Default settings
PORT=10200
VOICE="tara"
THREADS=4
MODEL_PATH="orpheus-3b-0.1-ft-q4_K_M.gguf"
LOG_LEVEL="INFO"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      PORT="$2"
      shift 2
      ;;
    --voice)
      VOICE="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --debug)
      LOG_LEVEL="DEBUG"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set the URI
URI="tcp://0.0.0.0:$PORT"

echo "Starting Wyoming Orpheus TTS server..."
echo "Voice: $VOICE"
echo "Port: $PORT"
echo "Threads: $THREADS"
echo "Model: $MODEL_PATH"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

# Start the server
python -m wyoming_orpheus \
  --uri "$URI" \
  --voice "$VOICE" \
  --n-threads "$THREADS" \
  --model-path "$MODEL_PATH" \
  $([ "$LOG_LEVEL" = "DEBUG" ] && echo "--debug")

# Deactivate virtual environment if activated
if [ -n "$VIRTUAL_ENV" ]; then
  deactivate
fi