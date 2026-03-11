"""Utility script to download the Orpheus model and print its SHA-256 hash."""

import hashlib

from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
    filename="orpheus-3b-0.1-ft-q4_k_m.gguf",
)

# Calculate the SHA-256 hash using chunked reads to avoid loading the entire
# 2.4 GB file into memory at once.
hasher = hashlib.sha256()
with open(model_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        hasher.update(chunk)
file_hash = hasher.hexdigest()

print(f"Calculated hash: {file_hash}")
print(
    "Expected hash:   18284d3efd9831d0a8409f5f5877c84bff69009df7c3db818e904dccea6b6c55"
)
