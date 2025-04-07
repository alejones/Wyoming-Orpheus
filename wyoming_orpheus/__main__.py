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
from .const import AVAILABLE_VOICES, DEFAULT_VOICE, VOICE_DESCRIPTIONS
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


async def main() -> None:
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up initial logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )

    # Convert argparse namespace to Pydantic config
    try:
        config = OrpheusConfig.from_args(args)
        _LOGGER.debug(f"Configuration: {config.dict()}")
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
            )
        ],
    )

    # Create model manager with the config
    model_manager = OrpheusModelManager(config.model)

    # Load model (but don't wait for it to complete)
    asyncio.create_task(model_manager.get_model())

    # Start server
    server = AsyncServer.from_uri(config.server.uri)
    _LOGGER.info(
        f"Starting Wyoming Orpheus server with model {config.model.model_path}"
    )

    # Run server
    await server.run(
        partial(
            OrpheusEventHandler,
            wyoming_info,
            config,
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
