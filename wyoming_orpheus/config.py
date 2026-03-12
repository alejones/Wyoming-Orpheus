"""Pydantic models for Wyoming Orpheus TTS configuration."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Import constants from existing code for default values
from .const import (
    AVAILABLE_VOICES,
    CHUNK_LIMIT,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_SAMPLES_PER_CHUNK,
    DEFAULT_THREADS,
    DEFAULT_VOICE,
    MAX_TOKENS,
    REPETITION_PENALTY,
    SAMPLE_RATE,
    TEMPERATURE,
    TOP_P,
)


class TTSConfig(BaseModel):
    """Configuration for Orpheus TTS speech generation.

    This model includes all parameters for controlling how text is
    synthesized into speech with the Orpheus TTS model.
    """

    voice: str = Field(
        default=DEFAULT_VOICE,
        description="Voice to use for synthesis",
    )

    temperature: float = Field(
        default=TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0-1, higher = more random)",
    )

    top_p: float = Field(
        default=TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter (0-1, lower = more deterministic)",
    )

    max_tokens: int = Field(
        default=MAX_TOKENS,
        gt=0,
        description="Maximum tokens to generate per request",
    )

    repetition_penalty: float = Field(
        default=REPETITION_PENALTY,
        ge=1.0,
        description="Repetition penalty (>=1.1 recommended)",
    )

    chunk_max_length: int = Field(
        default=CHUNK_LIMIT,
        gt=0,
        description="Maximum length of text chunks for processing",
    )

    @field_validator("voice")
    @classmethod
    def voice_must_be_valid(cls, v):
        """Validate that the voice is in the list of available voices."""
        if v not in AVAILABLE_VOICES:
            raise ValueError(
                f"Voice '{v}' not recognized. Available voices: {', '.join(AVAILABLE_VOICES)}"
            )
        return v


class ModelConfig(BaseModel):
    """Configuration for the Orpheus model.

    This model includes all parameters for loading and configuring
    the Orpheus language model.
    """

    model_path: Path = Field(
        description="Path to the Orpheus GGUF model file",
    )

    repo_id: str = Field(
        description="Hugging Face repository ID for model download",
    )

    n_threads: int = Field(
        default=DEFAULT_THREADS,
        gt=0,
        description="Number of threads to use for model inference",
    )

    n_ctx: int = Field(
        default=DEFAULT_CONTEXT_SIZE,
        gt=0,
        description="Context size in tokens",
    )

    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="Number of model layers to offload to GPU (0 = CPU only)",
    )

    verify_model: bool = Field(
        default=False,
        description="Verify model file hash before loading",
    )

    model_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory to store downloaded models",
    )

    force_download: bool = Field(
        default=False,
        description="Force re-download of the model even if it exists",
    )

    no_download: bool = Field(
        default=False,
        description="Do not download the model if not found",
    )

    @model_validator(mode="after")
    def check_download_flags(self):
        """Validate that force_download and no_download are not both True."""
        if self.force_download and self.no_download:
            raise ValueError("force_download and no_download cannot both be True")
        return self


class ServerConfig(BaseModel):
    """Configuration for the Wyoming server.

    This model includes all parameters for the Wyoming server protocol
    and audio processing.
    """

    uri: str = Field(
        default="tcp://0.0.0.0:10200",
        description="unix:// or tcp:// URI for Wyoming protocol",
    )

    samples_per_chunk: int = Field(
        default=DEFAULT_SAMPLES_PER_CHUNK,
        gt=0,
        description="Number of audio samples per chunk",
    )

    sample_rate: int = Field(
        default=SAMPLE_RATE,
        description="Audio sample rate in Hz",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )

    log_format: str = Field(
        default="%(levelname)s: %(message)s",
        description="Format for log messages",
    )

    auto_punctuation: str = Field(
        default=".?!",
        description="Characters to use for automatic punctuation",
    )


class OrpheusConfig(BaseModel):
    """Complete configuration for Wyoming Orpheus TTS.

    This model combines all sub-configurations into a single
    configuration object.
    """

    tts: TTSConfig = Field(default_factory=TTSConfig)
    model: ModelConfig
    server: ServerConfig

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return self.model_dump()
