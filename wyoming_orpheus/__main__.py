#!/usr/bin/env python3
"""Main entry point for wyoming-orpheus."""

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import List

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice  # type: ignore
from wyoming.server import AsyncServer  # type: ignore

from . import __version__
from .config import ModelConfig, OrpheusConfig, ServerConfig, TTSConfig
from .const import (
    AVAILABLE_VOICES,
    CHUNK_LIMIT,
    DEFAULT_SAMPLES_PER_CHUNK,
    DEFAULT_VOICE,
    MAX_TOKENS,
    REPETITION_PENALTY,
    SAMPLE_RATE,
    TEMPERATURE,
    TOP_P,
    VOICE_DESCRIPTIONS,
)
from .handler import OrpheusEventHandler
from .model_utils import DEFAULT_MODEL_FILENAME, DEFAULT_REPO_ID
from .orpheus import list_available_voices
from .process import OrpheusModelManager

_LOGGER = logging.getLogger(__name__)


def get_voices() -> List[TtsVoice]:
    """Get available TTS voices."""
    voices = []

    for voice_name in AVAILABLE_VOICES:
        voices.append(
            TtsVoice(
                name=voice_name,
                description=VOICE_DESCRIPTIONS.get(
                    voice_name, f"Orpheus TTS voice: {voice_name}"
                ),
                attribution=Attribution(
                    name="hubert-siuzdak",
                    url="https://github.com/hubertsiuzdak/orpheus-tts",
                ),
                installed=True,
                version=None,
                languages=["en"],  # Currently Orpheus only supports English
            )
        )

    return voices


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with all required arguments."""
    parser = argparse.ArgumentParser(description="Wyoming server for Orpheus TTS")

    # Model and system parameters
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_FILENAME,
        help=f"Path to Orpheus GGUF model file (default: {DEFAULT_MODEL_FILENAME})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="Directory to store downloaded models (uses Hugging Face cache by default)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the model even if it exists",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download the model if not found",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads to use for model inference (default: 4)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context size in tokens (default: 2048)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of model layers to offload to GPU (default: 0, CPU only)",
    )
    parser.add_argument(
        "--verify-model",
        action="store_true",
        help="Verify model file hash before loading",
    )

    # Voice and TTS parameters
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0-1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling parameter (0-1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Repetition penalty (>=1.1 recommended)",
    )
    parser.add_argument(
        "--chunk-max-length",
        type=int,
        help="Maximum length of text chunks for processing",
    )
    parser.add_argument("--auto-punctuation", help="Automatically add punctuation")

    # Wyoming server parameters
    parser.add_argument("--uri", help="unix:// or tcp:// URI for Wyoming protocol")
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        help="Number of audio samples per chunk",
    )

    # Utility parameters
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format",
        default=logging.BASIC_FORMAT,
        help="Format for log messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )

    return parser


def _build_config(args: argparse.Namespace) -> OrpheusConfig:
    """Build OrpheusConfig from a parsed argparse namespace."""

    def _get(name: str, default):
        value = getattr(args, name, None)
        return default if value is None else value

    tts_config = TTSConfig(
        voice=_get("voice", DEFAULT_VOICE),
        temperature=_get("temperature", TEMPERATURE),
        top_p=_get("top_p", TOP_P),
        max_tokens=_get("max_tokens", MAX_TOKENS),
        repetition_penalty=_get("repetition_penalty", REPETITION_PENALTY),
        chunk_max_length=_get("chunk_max_length", CHUNK_LIMIT),
    )
    model_config = ModelConfig(
        model_path=args.model_path,
        repo_id=args.repo_id,
        n_threads=_get("n_threads", 4),
        n_ctx=_get("n_ctx", 2048),
        n_gpu_layers=_get("n_gpu_layers", 0),
        verify_model=_get("verify_model", False),
        model_cache_dir=getattr(args, "model_cache_dir", None),
        force_download=_get("force_download", False),
        no_download=_get("no_download", False),
    )
    server_config = ServerConfig(
        uri=_get("uri", "tcp://0.0.0.0:10200"),
        samples_per_chunk=_get("samples_per_chunk", DEFAULT_SAMPLES_PER_CHUNK),
        sample_rate=SAMPLE_RATE,
        debug=_get("debug", False),
        log_format=_get("log_format", "%(levelname)s: %(message)s"),
        auto_punctuation=_get("auto_punctuation", ".?!"),
    )
    return OrpheusConfig(tts=tts_config, model=model_config, server=server_config)


async def main() -> None:
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up initial logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )

    # Build Pydantic config from CLI args
    try:
        config = _build_config(args)
        _LOGGER.debug(f"Configuration: {config.model_dump()}")
    except Exception as e:
        _LOGGER.error(f"Error in configuration: {e}")
        return

    # List voices and exit if requested
    if args.list_voices:
        list_available_voices()
        return

    # Verify model exists
    if not config.model.model_path.exists() and config.model.no_download:
        _LOGGER.error(
            f"Model file not found: {config.model.model_path} and downloading is disabled"
        )
        return

    # Prepare Wyoming info
    voices = get_voices()
    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="orpheus",
                description="Neural text-to-speech with emotional expressions",
                attribution=Attribution(
                    name="hubert-siuzdak",
                    url="https://github.com/hubertsiuzdak/orpheus-tts",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
                supports_synthesize_streaming=True,
            )
        ],
    )

    # Create a single shared model manager so all connections reuse the same
    # loaded weights rather than each connection loading its own copy.
    model_manager = OrpheusModelManager(config.model)

    # Eagerly download and load the model at startup so the container is ready
    # to serve requests immediately and download failures surface early.
    _LOGGER.info("Loading model at startup...")
    model = await model_manager.get_model()
    if model is None:
        _LOGGER.error("Failed to load model at startup, exiting")
        return

    # Start server
    server = AsyncServer.from_uri(config.server.uri)
    _LOGGER.info(
        f"Starting Wyoming Orpheus server with model {config.model.model_path}"
    )

    await server.run(
        partial(
            OrpheusEventHandler,
            wyoming_info,
            config,
            model_manager,
        )
    )


def run() -> None:
    """Run the program."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
