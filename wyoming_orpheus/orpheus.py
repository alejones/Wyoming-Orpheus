"""Orpheus TTS integration for Wyoming using llama.cpp."""
import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

from . import decoder
from .const import (
    AVAILABLE_VOICES,
    CHUNK_LIMIT,
    DEFAULT_VOICE,
    EMOTION_TAGS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    SAMPLE_RATE,
    TEMPERATURE,
    TOP_P,
    AUDIO_START_TOKEN,
    AUDIO_END_TOKEN
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for the TTS system."""

    voice: str = DEFAULT_VOICE
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: int = MAX_TOKENS
    repetition_penalty: float = REPETITION_PENALTY
    chunk_max_length: int = CHUNK_LIMIT


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Format prompt for Orpheus model with voice prefix and special tokens.

    Args:
        prompt: The text to convert to speech
        voice: The voice to use for synthesis

    Returns:
        A formatted prompt string ready for the model
    """
    if voice not in AVAILABLE_VOICES:
        _LOGGER.warning(
            f"Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead."
        )
        voice = DEFAULT_VOICE

    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"

    # Add special token markers
    special_start = AUDIO_START_TOKEN
    special_end = AUDIO_END_TOKEN

    return f"{special_start}{formatted_prompt}{special_end}"


def chunk_text(text: str, max_length: int = CHUNK_LIMIT) -> List[str]:
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
    llama_model,
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


def generate_speech_from_llama(
    llama_model,
    prompt: str,
    voice: str = DEFAULT_VOICE,
    output_file: Optional[Path] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    chunk_max_length: int = CHUNK_LIMIT,
) -> List[bytes]:
    """
    Generate speech from text using Orpheus model via llama.cpp.

    Args:
        llama_model: The loaded llama.cpp model
        prompt: The text to convert to speech
        voice: The voice to use
        output_file: Path to output WAV file (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty
        chunk_max_length: Maximum length of text chunks for processing

    Returns:
        List of audio segments as bytes
    """
    # Initialize the decoder model
    decoder.initialize_model()

    # If prompt is longer than chunk_max_length, split it into chunks
    if len(prompt) > chunk_max_length:
        chunks = chunk_text(prompt, chunk_max_length)
        all_audio_segments: List[bytes] = []

        _LOGGER.info(f"Text split into {len(chunks)} chunks for processing")

        for i, chunk in enumerate(chunks):
            _LOGGER.debug(f"Processing chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")
            
            # Generate tokens and convert to audio
            token_gen = generate_tokens_from_llama(
                llama_model=llama_model,
                prompt=chunk,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
            )
            
            # Collect audio segments
            chunk_segments = list(decoder.tokens_decoder_sync(token_gen))
            all_audio_segments.extend(chunk_segments)

        # Write to WAV file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_file), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                for segment in all_audio_segments:
                    wav_file.writeframes(segment)

        duration = (
            sum([len(segment) // (2 * 1) for segment in all_audio_segments])
            / SAMPLE_RATE
        )
        _LOGGER.info(f"Generated {len(all_audio_segments)} audio segments")
        _LOGGER.info(f"Generated {duration:.2f} seconds of audio")

        return all_audio_segments
    else:
        # Process single chunk
        token_gen = generate_tokens_from_llama(
            llama_model=llama_model,
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        # Convert tokens to audio
        audio_segments = list(decoder.tokens_decoder_sync(token_gen))
        
        # Write to WAV file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_file), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                for segment in audio_segments:
                    wav_file.writeframes(segment)
                    
        return audio_segments


def list_available_voices() -> None:
    """List all available voices with the recommended one marked."""
    _LOGGER.info("Available voices (in order of conversational realism):")
    for voice in AVAILABLE_VOICES:
        marker = "★" if voice == DEFAULT_VOICE else " "
        _LOGGER.info(f"{marker} {voice}")
    _LOGGER.info(f"\nDefault voice: {DEFAULT_VOICE}")

    _LOGGER.info("\nAvailable emotion tags:")
    _LOGGER.info(", ".join(sorted(EMOTION_TAGS)))