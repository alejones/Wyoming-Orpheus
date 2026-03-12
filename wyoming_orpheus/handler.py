"""Event handler for Wyoming clients."""

import asyncio
import logging
import queue as sync_queue
import threading

from wyoming.audio import AudioChunk, AudioStart, AudioStop  # type: ignore
from wyoming.error import Error  # type: ignore
from wyoming.event import Event  # type: ignore
from wyoming.info import Describe, Info  # type: ignore
from wyoming.server import AsyncEventHandler  # type: ignore
from wyoming.tts import (  # type: ignore
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .config import OrpheusConfig
from .const import AVAILABLE_VOICES, DEFAULT_VOICE
from .orpheus import generate_speech_stream_sync
from .process import OrpheusModelManager

_LOGGER = logging.getLogger(__name__)


class OrpheusEventHandler(AsyncEventHandler):
    """Handle Wyoming events for Orpheus TTS."""

    def __init__(
        self,
        wyoming_info: Info,
        config: OrpheusConfig,
        model_manager: OrpheusModelManager,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the event handler with a Pydantic config and shared model manager."""
        super().__init__(*args, **kwargs)

        self.config = config
        self.wyoming_info_event = wyoming_info.event()
        self.model_manager = model_manager

        # State for streaming text input (synthesize-start / chunk / stop)
        self._is_streaming_request: bool = False
        self._streaming_text: str = ""
        self._streaming_voice: str = DEFAULT_VOICE

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if SynthesizeStart.is_type(event.type):
            return await self._handle_synthesize_start(event)

        if SynthesizeChunk.is_type(event.type):
            return self._handle_synthesize_chunk(event)

        if SynthesizeStop.is_type(event.type):
            return await self._handle_synthesize_stop()

        if Synthesize.is_type(event.type):
            # During a streaming session the client also sends a backwards-
            # compatibility Synthesize containing the full text.  Skip it;
            # we already have the text from the chunks.
            if self._is_streaming_request:
                _LOGGER.debug(
                    "Ignoring backwards-compat Synthesize during streaming session"
                )
                return True

            try:
                return await self._handle_synthesize(event)
            except Exception as err:
                await self.write_event(
                    Error(text=str(err), code=err.__class__.__name__).event()
                )
                _LOGGER.exception("Error during synthesis")
                return False

        _LOGGER.warning("Unexpected event: %s", event)
        return True

    # ------------------------------------------------------------------
    # Streaming input handlers
    # ------------------------------------------------------------------

    async def _handle_synthesize_start(self, event: Event) -> bool:
        """Begin accumulating text for a streaming synthesis request."""
        synth_start = SynthesizeStart.from_event(event)
        self._is_streaming_request = True
        self._streaming_text = ""

        voice_name: str = DEFAULT_VOICE
        if synth_start.voice is not None and synth_start.voice.name:
            voice_name = synth_start.voice.name
            if voice_name not in AVAILABLE_VOICES:
                _LOGGER.warning(
                    "Unknown voice: %s, using %s", voice_name, DEFAULT_VOICE
                )
                voice_name = DEFAULT_VOICE
        self._streaming_voice = voice_name

        _LOGGER.debug("Streaming synthesis started, voice=%s", voice_name)
        return True

    def _handle_synthesize_chunk(self, event: Event) -> bool:
        """Accumulate a text chunk for the current streaming request."""
        if not self._is_streaming_request:
            _LOGGER.warning("Received SynthesizeChunk without SynthesizeStart")
            return True

        chunk = SynthesizeChunk.from_event(event)
        self._streaming_text += chunk.text
        _LOGGER.debug("Received text chunk (%d chars)", len(chunk.text))
        return True

    async def _handle_synthesize_stop(self) -> bool:
        """Synthesize the accumulated text and stream audio back."""
        if not self._is_streaming_request:
            _LOGGER.warning("Received SynthesizeStop without SynthesizeStart")
            return True

        text = self._streaming_text
        voice_name = self._streaming_voice

        # Reset streaming state before synthesis so errors don't leave stale state
        self._is_streaming_request = False
        self._streaming_text = ""
        self._streaming_voice = DEFAULT_VOICE

        if not text.strip():
            _LOGGER.warning("No text accumulated for streaming synthesis")
            return True

        text = self._preprocess_text(text)
        _LOGGER.debug("Streaming synthesize: text='%s'", text)

        try:
            return await self._synthesize_and_stream(
                text, voice_name, is_streaming_request=True
            )
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            _LOGGER.exception("Error during streaming synthesis")
            return False

    # ------------------------------------------------------------------
    # Non-streaming (backwards-compatible) handler
    # ------------------------------------------------------------------

    async def _handle_synthesize(self, event: Event) -> bool:
        """Handle a Synthesize event."""
        synthesize = Synthesize.from_event(event)
        _LOGGER.debug(synthesize)

        text = self._preprocess_text(synthesize.text)
        _LOGGER.debug("synthesize: text='%s'", text)

        voice_name: str = DEFAULT_VOICE
        if synthesize.voice is not None and synthesize.voice.name:
            voice_name = synthesize.voice.name
            if voice_name not in AVAILABLE_VOICES:
                _LOGGER.warning(
                    "Unknown voice: %s, using %s", voice_name, DEFAULT_VOICE
                )
                voice_name = DEFAULT_VOICE

        return await self._synthesize_and_stream(
            text, voice_name, is_streaming_request=False
        )

    # ------------------------------------------------------------------
    # Shared synthesis + streaming
    # ------------------------------------------------------------------

    def _preprocess_text(self, raw_text: str) -> str:
        """Join lines and optionally append punctuation."""
        text = " ".join(raw_text.strip().splitlines())
        if self.config.server.auto_punctuation and text:
            if text[-1] not in self.config.server.auto_punctuation:
                text = text + self.config.server.auto_punctuation[0]
        return text

    async def _synthesize_and_stream(
        self,
        text: str,
        voice_name: str,
        is_streaming_request: bool,
    ) -> bool:
        """
        Generate speech and stream audio chunks to the client as they are
        produced by the SNAC decoder.

        Token generation and SNAC decoding run in a background thread so the
        event loop is never blocked.  Audio bytes are delivered to this coroutine
        via an asyncio.Queue as each batch is decoded.

        For streaming-input requests (synthesize-start / chunk / stop), sends
        SynthesizeStopped after the final AudioStop.
        """
        model = await self.model_manager.get_model()
        if model is None:
            await self.write_event(
                Error(
                    text="Failed to load Orpheus model", code="ModelLoadError"
                ).event()
            )
            return False

        snac_decoder = self.model_manager.snac_decoder
        if snac_decoder is None:
            await self.write_event(
                Error(
                    text="SNAC decoder not initialized", code="ModelLoadError"
                ).event()
            )
            return False

        # Use a plain (sync) queue so the producer thread never needs to
        # interact with the event loop.  The consumer side uses run_in_executor
        # so that blocking on q.get() never stalls the event loop.
        audio_q: sync_queue.Queue = sync_queue.Queue()

        def _produce_audio() -> None:
            """Run in a thread: generate tokens and decode to audio, then signal done."""
            try:
                for audio_bytes in generate_speech_stream_sync(
                    llama_model=model,
                    snac_decoder=snac_decoder,
                    prompt=text,
                    voice=voice_name,
                    temperature=self.config.tts.temperature,
                    top_p=self.config.tts.top_p,
                    max_tokens=self.config.tts.max_tokens,
                    repetition_penalty=self.config.tts.repetition_penalty,
                    chunk_max_length=self.config.tts.chunk_max_length,
                ):
                    audio_q.put(audio_bytes)
            except Exception:
                _LOGGER.exception("Error in audio production thread")
            finally:
                audio_q.put(None)  # sentinel — always delivered

        loop = asyncio.get_running_loop()
        rate: int = self.config.server.sample_rate
        width: int = 2  # 16-bit PCM
        channels: int = 1
        bytes_per_chunk: int = width * channels * self.config.server.samples_per_chunk

        await self.write_event(
            AudioStart(rate=rate, width=width, channels=channels).event()
        )

        has_audio = False
        async with self.model_manager.inference_lock:
            thread = threading.Thread(target=_produce_audio, daemon=True)
            thread.start()

            while True:
                # Block a threadpool worker on q.get() so the event loop stays free.
                audio_bytes = await loop.run_in_executor(None, audio_q.get)
                if audio_bytes is None:
                    break
                has_audio = True
                for i in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk = audio_bytes[i : i + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event()
                    )

            thread.join()

        if not has_audio:
            # Send Error before AudioStop so clients reading the event stream
            # see the failure before the collection loop terminates on AudioStop.
            await self.write_event(
                Error(
                    text="Failed to generate speech", code="SynthesisError"
                ).event()
            )

        await self.write_event(AudioStop().event())

        if not has_audio:
            return True  # keep connection alive for future requests

        if is_streaming_request:
            await self.write_event(SynthesizeStopped().event())

        _LOGGER.debug("Completed synthesis request")
        return True
