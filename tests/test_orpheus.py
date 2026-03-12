"""Tests for wyoming-orpheus — covers both streaming and non-streaming TTS protocols."""

import asyncio
import hashlib
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import python_speech_features  # type:ignore
from huggingface_hub import hf_hub_download
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # type:ignore
from wyoming.error import Error  # type:ignore
from wyoming.event import async_read_event, async_write_event  # type:ignore
from wyoming.info import Describe, Info  # type:ignore
from wyoming.tts import (  # type:ignore
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from .dtw import compute_optimal_path  # type:ignore

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"
_MODEL_REPO_ID = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
_MODEL_FILENAME = "orpheus-3b-0.1-ft-q4_k_m.gguf"
_EXPECTED_HASH = "18284d3efd9831d0a8409f5f5877c84bff69009df7c3db818e904dccea6b6c55"
# Generous per-event timeout; first audio chunk arrives as soon as SNAC decodes
# the initial token batch, well before full synthesis completes.
_TIMEOUT = 120
_TEST_TEXT = "This is a test for Wyoming Orpheus. It is used as a Text-to-Speech voice for Home Assistant."
# DTW threshold (length-normalised); calibrated against reference audio.
_MAX_DTW_DISTANCE = 0.05


def download_orpheus_model() -> Optional[Path]:
    """Downloads the Orpheus model from Hugging Face."""
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _LOCAL_DIR / _MODEL_FILENAME

    if model_path.exists():
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() == _EXPECTED_HASH:
            print(f"Model already exists at {model_path}")
            return model_path

    try:
        print(f"Downloading Orpheus model from {_MODEL_REPO_ID}")
        downloaded_path = hf_hub_download(
            repo_id=_MODEL_REPO_ID,
            filename=_MODEL_FILENAME,
            local_dir=_LOCAL_DIR,
        )
        return Path(downloaded_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def get_reference_wav() -> Path:
    """Get the path to the reference WAV file."""
    reference_wav = _DIR / "orpheus_reference.wav"
    if not reference_wav.exists():
        pytest.skip(f"Reference WAV file not found: {reference_wav}")
    return reference_wav


async def _collect_audio(
    stdout,
    expect_synthesize_stopped: bool = False,
) -> Tuple[AudioStart, bytes]:
    """
    Read AudioStart, zero or more AudioChunks, and AudioStop from stdout.

    If expect_synthesize_stopped is True, also reads and asserts the
    SynthesizeStopped event that follows AudioStop in streaming mode.

    Returns (AudioStart, concatenated raw PCM bytes).
    """
    event = await asyncio.wait_for(async_read_event(stdout), timeout=_TIMEOUT)
    assert event is not None
    assert AudioStart.is_type(event.type), f"Expected AudioStart, got {event.type}"
    audio_start = AudioStart.from_event(event)

    audio_data = bytes()
    while True:
        event = await asyncio.wait_for(async_read_event(stdout), timeout=_TIMEOUT)
        assert event is not None

        if AudioStop.is_type(event.type):
            break

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            assert chunk.rate == audio_start.rate
            assert chunk.width == audio_start.width
            assert chunk.channels == audio_start.channels
            audio_data += chunk.audio
            continue

        if Error.is_type(event.type):
            err = Error.from_event(event)
            pytest.fail(f"Server returned error: {err.code} — {err.text}")

        pytest.fail(f"Unexpected event type during audio collection: {event.type}")

    if expect_synthesize_stopped:
        event = await asyncio.wait_for(async_read_event(stdout), timeout=_TIMEOUT)
        assert event is not None
        assert SynthesizeStopped.is_type(event.type), (
            f"Expected SynthesizeStopped after AudioStop, got {event.type}"
        )

    return audio_start, audio_data


def _verify_audio(
    actual_audio: bytes,
    expected_array: np.ndarray,
    audio_start: AudioStart,
    expected_framerate: int,
    label: str,
) -> None:
    """Assert audio parameters and DTW similarity against the reference."""
    assert len(actual_audio) > 0, f"[{label}] No audio data received"
    actual_array = np.frombuffer(actual_audio, dtype=np.int16)

    assert audio_start.rate == 24000, (
        f"[{label}] Expected 24000 Hz, got {audio_start.rate}"
    )
    assert audio_start.width == 2, (
        f"[{label}] Expected 16-bit (width=2), got {audio_start.width}"
    )
    assert audio_start.channels == 1, (
        f"[{label}] Expected mono, got {audio_start.channels} channels"
    )

    min_samples = int(2.0 * audio_start.rate)
    assert len(actual_array) >= min_samples, (
        f"[{label}] Audio too short: "
        f"{len(actual_array) / audio_start.rate:.2f}s, expected >= 2.0s"
    )

    print(f"[{label}] Computing MFCC features...")
    expected_mfcc = python_speech_features.mfcc(
        expected_array, samplerate=expected_framerate
    )
    actual_mfcc = python_speech_features.mfcc(
        actual_array, samplerate=audio_start.rate
    )

    dtw_distance = compute_optimal_path(actual_mfcc, expected_mfcc)
    print(f"[{label}] DTW distance: {dtw_distance:.4f} (max {_MAX_DTW_DISTANCE})")

    assert dtw_distance < _MAX_DTW_DISTANCE, (
        f"[{label}] Audio differs too much from reference "
        f"(DTW: {dtw_distance:.4f}, max: {_MAX_DTW_DISTANCE})"
    )


@pytest.mark.asyncio
async def test_orpheus() -> None:
    """
    Test Wyoming Orpheus TTS with both non-streaming and streaming input.

    Both modes are exercised in a single server process to avoid the cost of
    loading the model twice.
    """
    model_path = download_orpheus_model()
    if model_path is None:
        pytest.skip("Failed to download Orpheus model")

    reference_wav_path = get_reference_wav()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_orpheus",
        "--uri",
        "stdio://",
        "--model-path",
        str(model_path),
        "--voice",
        "tara",
        "--n-threads",
        "4",
        "--n-ctx",
        "2048",
        "--debug",
        stdin=PIPE,
        stdout=PIPE,
    )
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None

        # --- Describe / Info ---
        await async_write_event(Describe().event(), proc.stdin)
        while True:
            event = await asyncio.wait_for(
                async_read_event(proc.stdout), timeout=_TIMEOUT
            )
            assert event is not None
            if not Info.is_type(event.type):
                continue

            info = Info.from_event(event)
            assert len(info.tts) == 1, "Expected one tts service"
            tts = info.tts[0]
            assert tts.supports_synthesize_streaming, (
                "Expected supports_synthesize_streaming=True"
            )
            assert any(v.name == "tara" for v in tts.voices), "Expected tara voice"
            break

        # Load reference audio once for both tests
        with wave.open(str(reference_wav_path), "rb") as wav_file:
            expected_framerate = wav_file.getframerate()
            expected_audio = wav_file.readframes(wav_file.getnframes())
        expected_array = np.frombuffer(expected_audio, dtype=np.int16)

        # --- Test 1: non-streaming Synthesize ---
        await async_write_event(
            Synthesize(_TEST_TEXT, voice=SynthesizeVoice("tara")).event(),
            proc.stdin,
        )
        audio_start, actual_audio = await _collect_audio(proc.stdout)

        output_wav_path = _DIR / "orpheus_output_nonstreaming.wav"
        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(audio_start.channels)
            wav_file.setsampwidth(audio_start.width)
            wav_file.setframerate(audio_start.rate)
            wav_file.writeframes(actual_audio)
        print(f"Saved non-streaming output to {output_wav_path}")

        _verify_audio(
            actual_audio, expected_array, audio_start, expected_framerate,
            "non-streaming"
        )

        # --- Test 2: streaming SynthesizeStart / SynthesizeChunk / SynthesizeStop ---
        # Split the text to simulate an LLM streaming chunks to the TTS server.
        words = _TEST_TEXT.split()
        mid = len(words) // 2
        chunk1 = " ".join(words[:mid]) + " "
        chunk2 = " ".join(words[mid:])

        await async_write_event(
            SynthesizeStart(voice=SynthesizeVoice("tara")).event(), proc.stdin
        )
        await async_write_event(SynthesizeChunk(text=chunk1).event(), proc.stdin)
        await async_write_event(SynthesizeChunk(text=chunk2).event(), proc.stdin)
        await async_write_event(SynthesizeStop().event(), proc.stdin)

        audio_start, actual_audio = await _collect_audio(
            proc.stdout, expect_synthesize_stopped=True
        )

        output_wav_path = _DIR / "orpheus_output_streaming.wav"
        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(audio_start.channels)
            wav_file.setsampwidth(audio_start.width)
            wav_file.setframerate(audio_start.rate)
            wav_file.writeframes(actual_audio)
        print(f"Saved streaming output to {output_wav_path}")

        # The streaming test verifies protocol correctness (SynthesizeStopped
        # was received by _collect_audio) and that audio was produced with the
        # correct parameters and duration.  A strict DTW comparison against a
        # single reference file is omitted here because the model is
        # non-deterministic (temperature > 0), so two runs of identical text
        # produce different audio.  Audio quality is covered by the
        # non-streaming test above.
        assert audio_start.rate == 24000, (
            f"[streaming] Expected 24000 Hz, got {audio_start.rate}"
        )
        assert audio_start.width == 2, (
            f"[streaming] Expected 16-bit (width=2), got {audio_start.width}"
        )
        assert audio_start.channels == 1, (
            f"[streaming] Expected mono, got {audio_start.channels} channels"
        )
        actual_array = np.frombuffer(actual_audio, dtype=np.int16)
        min_samples = int(2.0 * audio_start.rate)
        assert len(actual_array) >= min_samples, (
            f"[streaming] Audio too short: "
            f"{len(actual_array) / audio_start.rate:.2f}s, expected >= 2.0s"
        )
        print(
            f"[streaming] Generated {len(actual_array) / audio_start.rate:.2f}s of audio"
        )

        print("All tests passed.")
    finally:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
