"""Pydantic models for Wyoming Orpheus TTS configuration."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Import constants from existing code for default values
from .const import (
    AVAILABLE_VOICES,
    CHUNK_LIMIT,
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
        default=4,
        gt=0,
        description="Number of threads to use for model inference",
    )

    n_ctx: int = Field(
        default=2048,
        gt=0,
        description="Context size in tokens",
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
        default="stdio://",
        description="unix:// or tcp:// URI for Wyoming protocol",
    )

    samples_per_chunk: int = Field(
        default=1024,
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

    @classmethod
    def from_args(cls, args):
        """Create a configuration object from argparse namespace.

        Uses default values from constants for any missing parameters.
        """
        # Convert argparse namespace to dictionary
        args_dict = vars(args)

        # Helper function to safely get arguments with defaults
        def get_arg(name: str, default: Any) -> Any:
            value = args_dict.get(name)
            return default if value is None else value

        # Create TTS configuration with proper defaults
        tts_config = TTSConfig(
            voice=get_arg("voice", DEFAULT_VOICE),
            temperature=get_arg("temperature", TEMPERATURE),
            top_p=get_arg("top_p", TOP_P),
            max_tokens=get_arg("max_tokens", MAX_TOKENS),
            repetition_penalty=get_arg("repetition_penalty", REPETITION_PENALTY),
            chunk_max_length=get_arg("chunk_max_length", CHUNK_LIMIT),
        )

        # Extract required model parameters
        model_path = args_dict.get("model_path")
        repo_id = args_dict.get("repo_id")

        # Create model configuration
        model_config = ModelConfig(
            model_path=model_path,
            repo_id=repo_id,
            n_threads=get_arg("n_threads", 4),
            n_ctx=get_arg("n_ctx", 2048),
            verify_model=get_arg("verify_model", False),
            model_cache_dir=args_dict.get("model_cache_dir"),
            force_download=get_arg("force_download", False),
            no_download=get_arg("no_download", False),
        )

        # Create server configuration
        server_config = ServerConfig(
            uri=get_arg("uri", "stdio://"),
            samples_per_chunk=get_arg("samples_per_chunk", 1024),
            sample_rate=SAMPLE_RATE,
            debug=get_arg("debug", False),
            log_format=get_arg("log_format", "%(levelname)s: %(message)s"),
            auto_punctuation=get_arg("auto_punctuation", ".?!"),
        )

        return cls(
            tts=tts_config,
            model=model_config,
            server=server_config,
        )

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return self.model_dump()

    def to_args(self) -> dict[str, Any]:
        """Convert the configuration to a flat dictionary for argparse."""
        # Combine all configs into a single flat dictionary
        result = {}

        # Add TTS config
        result.update(self.tts.model_dump())

        # Add model config
        result.update(self.model.model_dump())

        # Add server config
        result.update(self.server.model_dump())

        return result
