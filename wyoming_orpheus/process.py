"""Process management for Orpheus TTS using llama.cpp."""

import asyncio
import logging
import os
import time
from typing import Optional

from llama_cpp import Llama

from .config import ModelConfig
from .model_utils import ensure_model_exists, verify_model_file

_LOGGER = logging.getLogger(__name__)


class OrpheusModelManager:
    """Manager for Orpheus TTS model using llama.cpp."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the model manager with a Pydantic ModelConfig."""
        self.config = model_config
        self.model: Optional[Llama] = None
        self.model_path = model_config.model_path
        self.lock = asyncio.Lock()
        self.last_load_attempt = 0.0
        self.load_failed = False

    async def get_model(self) -> Optional[Llama]:
        """Get the loaded llama.cpp model or load if necessary."""
        async with self.lock:
            # If we've already tried and failed to load the model recently,
            # don't keep trying too frequently
            current_time = time.time()
            if self.load_failed and (current_time - self.last_load_attempt < 60):
                return None

            if self.model is None:
                try:
                    # Ensure the model exists, downloading if necessary
                    try:
                        model_path = ensure_model_exists(
                            self.model_path,
                            repo_id=self.config.repo_id,
                            model_cache_dir=self.config.model_cache_dir,
                            force_download=self.config.force_download,
                            no_download=self.config.no_download,
                        )
                        # Update the model path to the actual location
                        self.model_path = model_path
                    except FileNotFoundError as e:
                        _LOGGER.error(f"Model not found: {e}")
                        self.last_load_attempt = current_time
                        self.load_failed = True
                        return None

                    _LOGGER.info(f"Loading Orpheus model from {self.model_path}")

                    # Verify model file if verification is enabled
                    if self.config.verify_model:
                        if not verify_model_file(self.model_path):
                            _LOGGER.warning(
                                "Model verification failed, but continuing with loading"
                            )

                    # Use environment variables to control thread count
                    # llama_cpp library often relies on this env var for thread control
                    if self.config.n_threads > 0 and not os.environ.get(
                        "LLAMA_CPP_N_THREADS"
                    ):
                        os.environ["LLAMA_CPP_N_THREADS"] = str(self.config.n_threads)

                    # Determine context size based on model size
                    context_params = {}
                    if self.config.n_ctx > 0:
                        context_params["n_ctx"] = self.config.n_ctx

                    # Load the model with appropriate parameters
                    self.model = Llama(
                        model_path=str(self.model_path),
                        verbose=False,  # Debug info is handled by our logger
                        **context_params,
                    )
                    _LOGGER.info("Model loaded successfully")
                    self.load_failed = False

                except Exception as e:
                    self.last_load_attempt = current_time
                    self.load_failed = True
                    _LOGGER.error(f"Failed to load Orpheus model: {e}")
                    return None

            return self.model
