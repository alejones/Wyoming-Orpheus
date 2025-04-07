"""Tests for wyoming-orpheus"""

import asyncio
import hashlib
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import python_speech_features  # type:ignore
from huggingface_hub import hf_hub_download
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # type:ignore
from wyoming.event import async_read_event, async_write_event  # type:ignore
from wyoming.info import Describe, Info  # type:ignore
from wyoming.tts import Synthesize, SynthesizeVoice  # type:ignore

from tests.dtw import compute_optimal_path  # type:ignore

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"
_MODEL_REPO_ID = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
_MODEL_FILENAME = "orpheus-3b-0.1-ft-q4_k_m.gguf"
_EXPECTED_HASH = "18284d3efd9831d0a8409f5f5877c84bff69009df7c3db818e904dccea6b6c55"
_TIMEOUT = 120  # Longer timeout for model loading
_TEST_TEXT = "This is a test for Wyoming Orpheus. It is used as a Text-to-Speech voice for Home Assistant."


def download_orpheus_model() -> Optional[Path]:
    """Downloads the Orpheus model from Hugging Face."""
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _LOCAL_DIR / _MODEL_FILENAME

    # Check if model already exists with correct hash
    if model_path.exists():
        # Verify hash
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        if sha256_hash.hexdigest() == _EXPECTED_HASH:
            print(f"Model already exists at {model_path}")
            return model_path

    # Download from Hugging Face
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

    # Verify file exists
    if not reference_wav.exists():
        pytest.skip(f"Reference WAV file not found: {reference_wav}")

    return reference_wav


@pytest.mark.asyncio
async def test_orpheus() -> None:
    """Test the Wyoming Orpheus TTS server."""
    model_path = download_orpheus_model()
    if model_path is None:
        pytest.skip("Failed to download Orpheus model")

    # Get reference WAV for comparison
    reference_wav_path = get_reference_wav()

    # Start the Orpheus server process
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

        # Check info
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
            assert len(tts.voices) > 0, "Expected at least one voice"
            voice_model = next((v for v in tts.voices if v.name == "tara"), None)
            assert voice_model is not None, "Expected tara voice"
            break

        # Synthesize text - using the same text as the reference file
        await async_write_event(
            Synthesize(_TEST_TEXT, voice=SynthesizeVoice("tara")).event(),
            proc.stdin,
        )

        # Expect audio start event
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=_TIMEOUT)
        assert event is not None
        assert AudioStart.is_type(event.type), f"Expected AudioStart, got {event.type}"
        audio_start = AudioStart.from_event(event)

        # Load reference audio for comparison
        with wave.open(str(reference_wav_path), "rb") as wav_file:
            expected_framerate = wav_file.getframerate()
            # expected_sampwidth = wav_file.getsampwidth()
            # expected_channels = wav_file.getnchannels()
            expected_audio = wav_file.readframes(wav_file.getnframes())
            expected_array = np.frombuffer(expected_audio, dtype=np.int16)

        # Collect audio chunks
        actual_audio = bytes()
        while True:
            event = await asyncio.wait_for(
                async_read_event(proc.stdout), timeout=_TIMEOUT
            )
            assert event is not None

            if AudioStop.is_type(event.type):
                break

            if AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                assert chunk.rate == audio_start.rate
                assert chunk.width == audio_start.width
                assert chunk.channels == audio_start.channels
                actual_audio += chunk.audio

        # Ensure we got some audio data
        assert len(actual_audio) > 0, "No audio data received"
        actual_array = np.frombuffer(actual_audio, dtype=np.int16)

        # Save the generated audio for comparison/debugging
        output_wav_path = _DIR / "orpheus_output.wav"
        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(audio_start.channels)
            wav_file.setsampwidth(audio_start.width)
            wav_file.setframerate(audio_start.rate)
            wav_file.writeframes(actual_audio)

        print(f"Saved output audio to {output_wav_path}")

        # Check audio parameters match our expectations
        assert audio_start.rate == 24000, (
            f"Expected 24000 Hz sample rate, got {audio_start.rate}"
        )
        assert audio_start.width == 2, (
            f"Expected 16-bit audio (width=2), got {audio_start.width}"
        )
        assert audio_start.channels == 1, (
            f"Expected mono audio, got {audio_start.channels} channels"
        )

        # Check that we got a reasonable amount of audio data
        min_expected_duration = 2.0  # seconds
        min_expected_samples = int(min_expected_duration * audio_start.rate)
        assert len(actual_array) >= min_expected_samples, (
            f"Audio too short, got {len(actual_array) / audio_start.rate:.2f} seconds, "
            f"expected at least {min_expected_duration:.2f} seconds"
        )

        # Compute dynamic time warping (DTW) distance of MFCC features to compare audio similarity
        print("Computing MFCC features for reference and generated audio...")
        expected_mfcc = python_speech_features.mfcc(
            expected_array, samplerate=expected_framerate
        )
        actual_mfcc = python_speech_features.mfcc(
            actual_array, samplerate=audio_start.rate
        )

        # Compute DTW distance - this measures similarity between the two audio files
        # Lower values indicate more similar audio
        dtw_distance = compute_optimal_path(actual_mfcc, expected_mfcc)
        print(f"DTW distance between reference and generated audio: {dtw_distance}")

        # The acceptable threshold depends on your specific use case and quality requirements
        # You may need to adjust this based on empirical testing
        max_acceptable_distance = 15
        assert dtw_distance < max_acceptable_distance, (
            f"Audio differs too much from reference (DTW distance: {dtw_distance}, "
            f"max acceptable: {max_acceptable_distance})"
        )

        print("Test passed successfully!")
    finally:
        # Ensure proper cleanup of subprocess
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
