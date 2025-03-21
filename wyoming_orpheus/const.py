"""Constants for the Orpheus TTS Wyoming server using llama.cpp."""

from typing import Dict, Final, List, Set

# Model parameters
MAX_TOKENS: Final = 1200  # Maximum tokens to generate per request
TEMPERATURE: Final = 0.6  # Sampling temperature
TOP_P: Final = 0.9  # Top-p sampling parameter
REPETITION_PENALTY: Final = 1.1  # Repetition penalty (>=1.1 recommended)
SAMPLE_RATE: Final = 24000  # SNAC model uses 24kHz
CHUNK_LIMIT: Final = 400  # Maximum text chunk length for processing

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES: Final[List[str]] = [
    "tara",
    "leah",
    "jess",
    "leo",
    "dan",
    "mia",
    "zac",
    "zoe",
]
DEFAULT_VOICE: Final = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model
START_TOKEN_ID: Final = 128259
END_TOKEN_IDS: Final[List[int]] = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX: Final = "<custom_token_"

# Available emotion tags
EMOTION_TAGS: Final[Set[str]] = {
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
}

# Voice descriptions for info messages
VOICE_DESCRIPTIONS: Final[Dict[str, str]] = {
    "tara": "Female voice",
    "leah": "Female voice",
    "jess": "Female voice",
    "leo": "Male voice",
    "dan": "Male voice",
    "mia": "Female voice",
    "zac": "Male voice",
    "zoe": "Female voice",
}

# Model settings
DEFAULT_THREADS: Final = 4  # Default number of threads for model inference
DEFAULT_CONTEXT_SIZE: Final = 2048  # Default context size in tokens

# Special prompt tokens
AUDIO_START_TOKEN: Final = "<|audio|>"
AUDIO_END_TOKEN: Final = "<|eot_id|>"
