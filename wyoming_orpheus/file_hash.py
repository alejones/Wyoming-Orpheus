"""Utilities for file hashing and verification."""
import hashlib
import logging
from pathlib import Path
from typing import Dict, Union

_LOGGER = logging.getLogger(__name__)

# Expected model hashes - update as new models are released
MODEL_HASHES: Dict[str, str] = {
    "orpheus-3b-0.1-ft-q4_K_M.gguf": "44e874b701c348e7ef53f4c9bcfbdd5fe5c7c35c3c60a4eb15a7696c4102663a",
}


def get_file_hash(path: Union[str, Path], algorithm: str = "sha256", chunk_size: int = 4096) -> str:
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
    expected_hash = MODEL_HASHES.get(model_path.name)
    
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