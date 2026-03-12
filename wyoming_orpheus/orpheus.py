"""Orpheus TTS integration for Wyoming using llama.cpp."""

import logging
import re
import sys
from typing import Generator, List

from llama_cpp import Llama  # type: ignore

from .const import (
    AUDIO_END_TOKEN,
    AUDIO_START_TOKEN,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    EMOTION_TAGS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from .decoder import SnacDecoder

_LOGGER = logging.getLogger(__name__)


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Format prompt for Orpheus model with voice prefix and special tokens.

    Args:
        prompt: The text to convert to speech
        voice: The voice to use for synthesis (caller is responsible for validation)

    Returns:
        A formatted prompt string ready for the model
    """
    return f"{AUDIO_START_TOKEN}{voice}: {prompt}{AUDIO_END_TOKEN}"


def chunk_text(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks based on sentence delimiters (., !, ?).
    Each chunk will be at most max_length characters.

    Args:
        text: The input text to chunk
        max_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    # Initialize variables
    chunks: List[str] = []
    current_chunk: str = ""

    # Split the text by sentence delimiters while keeping the delimiters
    sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", text + " ")

    for sentence in sentences:
        # If adding this sentence would exceed max_length, start a new chunk
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def generate_tokens_from_llama(
    llama_model: Llama,
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
) -> Generator[str, None, None]:
    """
    Generate tokens from text using llama.cpp.

    Args:
        llama_model: The loaded llama.cpp model
        prompt: The text to convert to speech
        voice: The voice to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty

    Yields:
        Token strings from the model
    """
    formatted_prompt = format_prompt(prompt, voice)
    _LOGGER.debug(f"Generating speech for: {formatted_prompt}")

    # Create generator with streaming
    generator = llama_model.create_completion(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        stream=True,
    )

    try:
        for output in generator:
            token_text = output["choices"][0]["text"]
            if token_text:
                yield token_text

    except Exception as e:
        _LOGGER.error(f"Error generating tokens: {e}")

    _LOGGER.debug("Token generation complete")


def generate_speech_stream_sync(
    llama_model: Llama,
    snac_decoder: "SnacDecoder",
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    chunk_max_length: int = 400,
) -> Generator[bytes, None, None]:
    """
    Sync generator that yields raw PCM audio bytes as SNAC produces them.

    Token generation and SNAC decoding run in the calling thread. Yields each
    audio batch immediately rather than buffering the full result. No WAV file
    is written.  Each yielded bytes object is 16-bit mono PCM at SAMPLE_RATE Hz.
    """
    chunks = (
        chunk_text(prompt, chunk_max_length)
        if len(prompt) > chunk_max_length
        else [prompt]
    )

    _LOGGER.info("Streaming speech for %d text chunk(s)", len(chunks))

    for i, chunk in enumerate(chunks):
        _LOGGER.debug("Processing chunk %d/%d: %s...", i + 1, len(chunks), chunk[:50])
        token_gen = generate_tokens_from_llama(
            llama_model=llama_model,
            prompt=chunk,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
        yield from snac_decoder.tokens_decoder_sync(token_gen)


def list_available_voices() -> None:
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):", file=sys.stdout)
    for voice in AVAILABLE_VOICES:
        marker = "*" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}", file=sys.stdout)
    print(f"\nDefault voice: {DEFAULT_VOICE}", file=sys.stdout)
    print("\nAvailable emotion tags:", file=sys.stdout)
    print(", ".join(sorted(EMOTION_TAGS)), file=sys.stdout)
