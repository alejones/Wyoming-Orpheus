"""Utilities for model downloading and verification using the Hugging Face Hub."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download, scan_cache_dir  # type: ignore
from huggingface_hub.utils import (  # type: ignore
    HfHubHTTPError,
    RepositoryNotFoundError,
)

_LOGGER = logging.getLogger(__name__)

# Default Hugging Face repository and model information
DEFAULT_REPO_ID = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
DEFAULT_MODEL_FILENAME = "orpheus-3b-0.1-ft-q4_K_M.gguf"

# Expected model hashes - update as new models are released
MODEL_HASHES: Dict[str, str] = {
    "orpheus-3b-0.1-ft-q4_k_m.gguf": "44e874b701c348e7ef53f4c9bcfbdd5fe5c7c35c3c60a4eb15a7696c4102663a",
}


def get_file_hash(
    path: Union[str, Path], algorithm: str = "sha256", chunk_size: int = 4096
) -> str:
    """
    Calculate the hash of a file using the specified algorithm.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)
        chunk_size: Size of chunks to read from file

    Returns:
        The hex digest of the file hash
    """
    path = Path(path)
    hasher = hashlib.new(algorithm)

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_model_file(model_path: Union[str, Path]) -> bool:
    """
    Verify a model file against known hashes.

    Args:
        model_path: Path to the model file

    Returns:
        True if the file hash matches the expected hash, False otherwise
    """
    model_path = Path(model_path)
    expected_hash = MODEL_HASHES.get(model_path.name.lower())

    if not expected_hash:
        _LOGGER.warning(f"No known hash for model: {model_path.name}")
        return True

    actual_hash = get_file_hash(model_path)
    if actual_hash != expected_hash:
        _LOGGER.warning(
            f"Model hash mismatch for {model_path.name}. "
            f"Expected: {expected_hash}, actual: {actual_hash}"
        )
        return False

    _LOGGER.info(f"Successfully verified model file: {model_path.name}")
    return True


def find_model_in_cache(filename: str) -> Optional[Path]:
    """
    Search for a model in the Hugging Face cache.

    Args:
        filename: The filename to search for

    Returns:
        Path to the cached file if found, None otherwise
    """
    try:
        cache_info = scan_cache_dir()

        # Search all repositories in cache
        for repo in cache_info.repos:
            for revision in repo.revisions:
                for cached_file in revision.files:
                    if cached_file.filename == filename:
                        return Path(cached_file.file_path)

        return None
    except Exception as e:
        _LOGGER.warning(f"Error scanning cache: {e}")
        return None


def ensure_model_exists(
    model_path: Union[str, Path],
    repo_id: str = DEFAULT_REPO_ID,
    filename: Optional[str] = None,
    model_cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """
    Ensure the model file exists, downloading it from Hugging Face if necessary.

    Args:
        model_path: Path to the model file
        repo_id: Hugging Face repository ID
        filename: Filename of the model (defaults to basename of model_path)
        model_cache_dir: Directory to store downloaded models
        force_download: Force download even if the file exists
        no_download: Do not download the model if not found

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If the model is not found and cannot be downloaded
    """
    model_path = Path(model_path)

    # Use basename if filename not provided
    if filename is None:
        filename = model_path.name

    # If the model path already exists and we're not forcing a download
    if model_path.exists() and not force_download:
        _LOGGER.info(f"Model already exists at {model_path}")
        return model_path

    # If only a filename was provided or the specified path doesn't exist
    if (
        not model_path.parent
        or model_path.parent == Path(".")
        or not model_path.exists()
    ):
        # Check if the model is in the HF cache already
        cached_path = find_model_in_cache(filename)
        if cached_path and cached_path.exists():
            _LOGGER.info(f"Found model in Hugging Face cache: {cached_path}")
            return cached_path

        # If we're not supposed to download, raise an error
        if no_download:
            raise FileNotFoundError(
                f"Model not found at {model_path} and download is disabled"
            )

        # Download from Hugging Face
        try:
            _LOGGER.info(f"Downloading model {filename} from {repo_id}")

            # Set cache directory if specified
            cache_dir = None
            if model_cache_dir:
                cache_dir = str(model_cache_dir)

            # Download the file
            downloaded_str: str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            downloaded_path = Path(downloaded_str)
            _LOGGER.info(f"Model downloaded to {downloaded_path}")

            # If a specific output path was requested, copy the file there
            if model_path != downloaded_path and model_path.parent != Path("."):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _LOGGER.info(f"Copying model to {model_path}")
                model_path.write_bytes(downloaded_path.read_bytes())
                return model_path

            return downloaded_path

        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            _LOGGER.error(f"Error downloading model: {e}")
            raise FileNotFoundError(f"Failed to download model: {e}")

    # If we get here, the model doesn't exist and couldn't be downloaded
    raise FileNotFoundError(
        f"Model not found at {model_path} and could not be downloaded"
    )
