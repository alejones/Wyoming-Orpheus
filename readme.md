# Wyoming Orpheus

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for [Orpheus-TTS](https://github.com/hubertsiuzdak/orpheus-tts) using llama.cpp.

## Overview

Wyoming Orpheus provides a Wyoming protocol interface for the Orpheus-TTS system, allowing you to integrate high-quality, emotional text-to-speech into Home Assistant and other compatible systems.

This server uses llama.cpp to run the Orpheus GGUF model locally, generating speech with various voices and emotional expressions without requiring an external API or service.

The system uses the Hugging Face Hub library to automatically download the required model file if it's not found locally.

## Prerequisites

1. Python 3.8 or later
2. Approximately 4GB of disk space for the Orpheus model (will be downloaded automatically)

## Installation

Clone the repository and set up a virtual environment:

```sh
git clone https://github.com/yourusername/wyoming-orpheus.git
cd wyoming-orpheus
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Make scripts executable (Unix/Linux/macOS)
chmod +x script/setup
chmod +x script/run

# Run setup
script/setup
```

## Usage

Run the Wyoming Orpheus server:

```sh
wyoming-orpheus --uri 'tcp://0.0.0.0:10200'
```

The model will be automatically downloaded on first run if not found locally.

If you already have the model file:

```sh
wyoming-orpheus \
  --model-path /path/to/orpheus-3b-0.1-ft-q4_K_M.gguf \
  --uri 'tcp://0.0.0.0:10200'
```

### Command Line Options

#### Model Options:
- `--model-path` - Path to Orpheus GGUF model file (required)
- `--n-threads` - Number of CPU threads to use (default: 4)
- `--n-ctx` - Context size in tokens (default: 2048)

#### Voice and TTS Options:
- `--voice` - Default voice to use (default: tara)
- `--temperature` - Sampling temperature (default: 0.6)
- `--top-p` - Top-p sampling parameter (default: 0.9)
- `--repetition-penalty` - Repetition penalty (default: 1.1)
- `--auto-punctuation` - Characters to use for automatic punctuation (default: .?!)
- `--max-tokens` - Maximum tokens to generate per request (default: 1200)
- `--chunk-max-length` - Maximum text chunk length (default: 400)

#### Server Options:
- `--uri` - Wyoming protocol URI (default: stdio://)
- `--samples-per-chunk` - Audio samples per chunk (default: 1024)

#### Utility Options:
- `--list-voices` - List available voices and exit
- `--debug` - Enable debug logging
- `--version` - Show version and exit

## Available Voices

Orpheus comes with several built-in voices:

- `tara` - Female voice (default)
- `leah` - Female voice
- `jess` - Female voice
- `leo` - Male voice
- `dan` - Male voice
- `mia` - Female voice
- `zac` - Male voice
- `zoe` - Female voice

## Emotion Tags

Orpheus supports adding emotional expressions to text using the following tags:

- `<laugh>` - Add laughter
- `<chuckle>` - Add a chuckle
- `<sigh>` - Add a sigh
- `<cough>` - Add a cough
- `<sniffle>` - Add a sniffle
- `<groan>` - Add a groan
- `<yawn>` - Add a yawn
- `<gasp>` - Add a gasp

Example: "I can't believe you did that! <laugh> That's amazing!"

## Performance Considerations

The Orpheus model requires a reasonable amount of RAM and CPU power:

- At least 4GB of RAM for the quantized model
- Multi-core CPU recommended for faster inference
- GPU acceleration is supported if you build llama-cpp-python with CUDA support

To optimize performance:
- Adjust `--n-threads` to match your CPU core count
- Set an appropriate `--n-ctx` value (higher for more context, but uses more memory)
- Use a GPU-enabled build of llama-cpp-python for faster inference

## Home Assistant Integration

To integrate with Home Assistant:

1. Ensure the Wyoming integration is installed
2. Start the Wyoming Orpheus server
3. Add a new Wyoming TTS in Home Assistant, pointing to the Wyoming Orpheus server
4. Select your preferred voice in the TTS settings

## Docker Deployment

```sh
docker run -it --rm \
  -v /path/to/model:/model \
  -p 10200:10200 \
  yourusername/wyoming-orpheus \
  --model-path /model/orpheus-3b-0.1-ft-q4_K_M.gguf \
  --uri tcp://0.0.0.0:10200
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Orpheus-TTS](https://github.com/hubertsiuzdak/orpheus-tts) for the base TTS system
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the efficient inference engine
- [Wyoming Protocol](https://github.com/rhasspy/wyoming) for the standardized communication protocol
- [SNAC](https://github.com/myshell-ai/OpenVoice) for the audio codec technology