"""Decoder for Orpheus-TTS."""

import asyncio
import logging
import queue
import threading
from typing import AsyncGenerator, Generator, List, Optional

import numpy as np
import torch
from snac import SNAC

_LOGGER = logging.getLogger(__name__)

# Constants for token processing
CUSTOM_TOKEN_PREFIX = "<custom_token_"
CUSTOM_TOKEN_PREFIX_LEN = 14  # Length of "<custom_token_"
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Audio processing constants
TOKENS_PER_FRAME = 7  # Number of tokens that make up one audio frame
TOKEN_OFFSET = 10  # Offset used in token ID calculation
PROCESSING_BUFFER_SIZE = 28  # Buffer size for processing (4 * TOKENS_PER_FRAME)
MIN_TOKENS_FOR_PROCESSING = 27  # Minimum number of tokens needed before processing

# SNAC model constants
SNAC_OUTPUT_SLICE_START = 2048  # Start index for audio output slice
SNAC_OUTPUT_SLICE_END = 4096  # End index for audio output slice
SNAC_CODEBOOK_SIZE = 4096  # Maximum token value in codebook


class SnacDecoder:
    """Class for decoding Orpheus-TTS tokens to audio."""

    def __init__(self) -> None:
        """Initialize the SNAC model for audio decoding."""
        self.model = None
        self.device = None

        _LOGGER.info("Initializing SNAC model...")
        try:
            self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

            # Check if CUDA is available and set device accordingly
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            _LOGGER.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
        except Exception as e:
            _LOGGER.error(f"Failed to initialize SNAC model: {e}")
            raise

    @staticmethod
    def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
        """
        Convert token string to numeric ID for audio processing.

        Args:
            token_string: The token string to convert
            index: The current token index

        Returns:
            Token ID as integer, or None if conversion failed
        """
        # Strip whitespace
        token_string = token_string.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[CUSTOM_TOKEN_PREFIX_LEN:-1]
                token_id = (
                    int(number_str)
                    - TOKEN_OFFSET
                    - ((index % TOKENS_PER_FRAME) * SNAC_CODEBOOK_SIZE)
                )
                return token_id
            except ValueError:
                return None
        else:
            return None

    def convert_to_audio(self, multiframe: List[int], count: int) -> Optional[bytes]:
        """
        Convert token frames to audio.

        Args:
            multiframe: List of token IDs
            count: Current token count

        Returns:
            Audio data as bytes or None if conversion failed
        """
        if len(multiframe) < TOKENS_PER_FRAME:
            return None

        codes_0 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.device, dtype=torch.int32)

        num_frames = len(multiframe) // TOKENS_PER_FRAME
        frame = multiframe[: num_frames * TOKENS_PER_FRAME]

        for j in range(num_frames):
            i = TOKENS_PER_FRAME * j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor(
                    [frame[i]], device=self.device, dtype=torch.int32
                )
            else:
                codes_0 = torch.cat(
                    [
                        codes_0,
                        torch.tensor([frame[i]], device=self.device, dtype=torch.int32),
                    ]
                )

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor(
                    [frame[i + 1]], device=self.device, dtype=torch.int32
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 1]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_1 = torch.cat(
                    [
                        codes_1,
                        torch.tensor(
                            [frame[i + 4]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )

            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor(
                    [frame[i + 2]], device=self.device, dtype=torch.int32
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
            else:
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 2]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 3]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 5]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )
                codes_2 = torch.cat(
                    [
                        codes_2,
                        torch.tensor(
                            [frame[i + 6]], device=self.device, dtype=torch.int32
                        ),
                    ]
                )

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Check that all tokens are between 0 and SNAC_CODEBOOK_SIZE
        if (
            torch.any(codes[0] < 0)
            or torch.any(codes[0] > SNAC_CODEBOOK_SIZE)
            or torch.any(codes[1] < 0)
            or torch.any(codes[1] > SNAC_CODEBOOK_SIZE)
            or torch.any(codes[2] < 0)
            or torch.any(codes[2] > SNAC_CODEBOOK_SIZE)
        ):
            return None

        try:
            with torch.inference_mode():
                audio_hat = self.model.decode(codes)

            audio_slice = audio_hat[:, :, SNAC_OUTPUT_SLICE_START:SNAC_OUTPUT_SLICE_END]
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            # Convert to 16-bit PCM
            INT16_MAX = 32767  # Maximum value for 16-bit signed integer
            audio_int16 = (audio_np * INT16_MAX).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            return audio_bytes
        except Exception as e:
            _LOGGER.error(f"Error converting tokens to audio: {e}")
            return None

    async def tokens_decoder(
        self, token_gen: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Asynchronous token decoder that converts token stream to audio stream.

        Args:
            token_gen: Asynchronous generator of token strings

        Yields:
            Audio data chunks as bytes
        """
        buffer: List[int] = []
        count: int = 0

        async for token_text in token_gen:
            token = self.turn_token_into_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % TOKENS_PER_FRAME == 0 and count > MIN_TOKENS_FOR_PROCESSING:
                    buffer_to_proc = buffer[-PROCESSING_BUFFER_SIZE:]
                    audio_samples = self.convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

    def tokens_decoder_sync(
        self, syn_token_gen: Generator[str, None, None]
    ) -> Generator[bytes, None, None]:
        """
        Synchronous wrapper for the asynchronous token decoder.

        Args:
            syn_token_gen: Synchronous generator of token strings

        Yields:
            Audio data chunks as bytes
        """
        audio_queue: queue.Queue = queue.Queue()

        # Convert the synchronous token generator into an async generator
        async def async_token_gen() -> AsyncGenerator[str, None]:
            for token in syn_token_gen:
                yield token

        async def async_producer() -> None:
            async for audio_chunk in self.tokens_decoder(async_token_gen()):
                audio_queue.put(audio_chunk)
            audio_queue.put(None)  # Sentinel to indicate completion

        def run_async() -> None:
            asyncio.run(async_producer())

        # Start the async producer in a separate thread
        thread = threading.Thread(target=run_async)
        thread.start()

        # Process audio as it becomes available
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            yield audio

        thread.join()
