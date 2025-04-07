"""Event handler for Wyoming clients."""

import argparse
import logging
import math
import tempfile
import time
import wave
from pathlib import Path

from wyoming.audio import AudioChunk, AudioStart, AudioStop  # type: ignore
from wyoming.error import Error  # type: ignore
from wyoming.event import Event  # type: ignore
from wyoming.info import Describe, Info  # type: ignore
from wyoming.server import AsyncEventHandler  # type: ignore
from wyoming.tts import Synthesize  # type: ignore

from .const import AVAILABLE_VOICES, DEFAULT_VOICE, SAMPLE_RATE, VOICE_DESCRIPTIONS
from .orpheus import TTSConfig, generate_speech_from_llama
from .process import OrpheusModelManager

_LOGGER = logging.getLogger(__name__)


class OrpheusEventHandler(AsyncEventHandler):
    """Handle Wyoming events for Orpheus TTS."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_manager: OrpheusModelManager,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the event handler."""
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_manager = model_manager

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True

        try:
            return await self._handle_synthesize(event)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            _LOGGER.exception("Error during synthesis")
            return False

    async def _stream_wav_file(self, wav_path: Path) -> None:
        """Read a WAV file and stream it chunk by chunk."""
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()

            # Send audio start event
            await self.write_event(
                AudioStart(
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event(),
            )

            # Read the entire audio file
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            bytes_per_sample = width * channels
            bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
            num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

            # Split into chunks
            for i in range(num_chunks):
                offset = i * bytes_per_chunk
                chunk = audio_bytes[offset : offset + bytes_per_chunk]
                await self.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event(),
                )

        # Send audio stop event
        await self.write_event(AudioStop().event())
        _LOGGER.debug("Finished streaming WAV file")

    async def _handle_synthesize(self, event: Event) -> bool:
        """Handle a Synthesize event."""
        synthesize = Synthesize.from_event(event)
        _LOGGER.debug(synthesize)

        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        # Add automatic punctuation (important for some voices)
        # TODO is this necessary?
        if self.cli_args.auto_punctuation and text:
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)

        # Get voice name
        voice_name: str = DEFAULT_VOICE
        if synthesize.voice is not None and synthesize.voice.name:
            voice_name = synthesize.voice.name
            if voice_name not in AVAILABLE_VOICES:
                _LOGGER.warning(f"Unknown voice: {voice_name}, using {DEFAULT_VOICE}")
                voice_name = DEFAULT_VOICE

        # Create config for TTS
        config = TTSConfig(
            voice=voice_name,
            temperature=self.cli_args.temperature,
            top_p=self.cli_args.top_p,
            repetition_penalty=self.cli_args.repetition_penalty,
            chunk_max_length=self.cli_args.chunk_max_length,
        )

        # Get the model
        model = await self.model_manager.get_model()
        if model is None:
            await self.write_event(
                Error(
                    text="Failed to load Orpheus model", code="ModelLoadError"
                ).event()
            )
            return False

        # Create temporary directory for the WAV file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate speech
            output_file = Path(temp_dir) / f"{voice_name}_{int(time.time())}.wav"
            audio_segments = generate_speech_from_llama(
                llama_model=model,
                prompt=text,
                voice=config.voice,
                output_file=output_file,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                chunk_max_length=config.chunk_max_length,
            )

            if not audio_segments:
                await self.write_event(
                    Error(
                        text="Failed to generate speech", code="SynthesisError"
                    ).event()
                )
                return False

            # Stream the generated WAV file
            await self._stream_wav_file(output_file)
            _LOGGER.debug("Completed request")

        return True
