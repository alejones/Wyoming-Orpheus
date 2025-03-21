"""Decoder for Orpheus-TTS."""
import asyncio
import logging
import queue
import threading
from typing import AsyncGenerator, Generator, List, Optional

import numpy as np # type: ignore
import torch # type: ignore
from snac import SNAC # type: ignore
 
_LOGGER = logging.getLogger(__name__)

# Constants for token processing
CUSTOM_TOKEN_PREFIX = "<custom_token_"
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Initialize model
model = None
snac_device: str | None = None


def initialize_model() -> None:
    """Initialize the SNAC model."""
    global model, snac_device
    
    if model is not None:
        return
    
    _LOGGER.info("Initializing SNAC model...")
    try:
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        # Check if CUDA is available and set device accordingly
        if torch.cuda.is_available():
            snac_device = "cuda"
        elif torch.backends.mps.is_available():
            snac_device = "mps"
        else:
            snac_device = "cpu"
            
        _LOGGER.info(f"Using device: {snac_device}")
        model = model.to(snac_device)
    except Exception as e:
        _LOGGER.error(f"Failed to initialize SNAC model: {e}")
        raise


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
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None


def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """
    Convert token frames to audio.

    Args:
        multiframe: List of token IDs
        count: Current token count

    Returns:
        Audio data as bytes or None if conversion failed
    """
    global model, snac_device
    
    if model is None:
        initialize_model()
    
    if len(multiframe) < 7:
        return None
    
    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        else:
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
            codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
        
        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
        else:
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
            codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    # Check that all tokens are between 0 and 4096
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    try:
        with torch.inference_mode():
            audio_hat = model.decode(codes)
        
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes
    except Exception as e:
        _LOGGER.error(f"Error converting tokens to audio: {e}")
        return None


async def tokens_decoder(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
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
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples


def tokens_decoder_sync(syn_token_gen: Generator[str, None, None]) -> Generator[bytes, None, None]:
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
        async for audio_chunk in tokens_decoder(async_token_gen()):
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