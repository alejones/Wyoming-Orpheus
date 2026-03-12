"""Utility script to download the Orpheus model and print its SHA-256 hash."""

import hashlib

from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="unsloth/orpheus-3b-0.1-ft-GGUF",
    filename="orpheus-3b-0.1-ft-Q4_K_M.gguf",
)

# Calculate the SHA-256 hash using chunked reads to avoid loading the entire
# file into memory at once.
hasher = hashlib.sha256()
with open(model_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        hasher.update(chunk)
file_hash = hasher.hexdigest()

print(f"SHA-256: {file_hash}")
print("Set _EXPECTED_HASH in tests/test_orpheus.py to this value.")
