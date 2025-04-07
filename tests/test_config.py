"""Tests for the Pydantic configuration models."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from wyoming_orpheus.config import ModelConfig, OrpheusConfig, ServerConfig, TTSConfig


def test_tts_config_defaults():
    """Test TTSConfig with default values."""
    config = TTSConfig()
    assert config.voice == "tara"
    assert 0.0 <= config.temperature <= 1.0
    assert 0.0 <= config.top_p <= 1.0
    assert config.max_tokens > 0
    assert config.repetition_penalty >= 1.0
    assert config.chunk_max_length > 0


def test_tts_config_validation():
    """Test TTSConfig validation."""
    # Valid config
    config = TTSConfig(voice="leo", temperature=0.8)
    assert config.voice == "leo"
    assert config.temperature == 0.8

    # Invalid voice
    with pytest.raises(ValidationError):
        TTSConfig(voice="invalid_voice")

    # Invalid temperature (out of range)
    with pytest.raises(ValidationError):
        TTSConfig(temperature=1.5)

    # Invalid top_p (out of range)
    with pytest.raises(ValidationError):
        TTSConfig(top_p=-0.1)

    # Invalid repetition_penalty (too low)
    with pytest.raises(ValidationError):
        TTSConfig(repetition_penalty=0.5)


def test_model_config():
    """Test ModelConfig."""
    # Valid config
    config = ModelConfig(
        model_path=Path("model.gguf"), repo_id="user/repo", n_threads=4
    )
    assert config.model_path == Path("model.gguf")
    assert config.repo_id == "user/repo"
    assert config.n_threads == 4

    # Invalid config (conflicting flags)
    with pytest.raises(ValidationError):
        ModelConfig(
            model_path=Path("model.gguf"),
            repo_id="user/repo",
            force_download=True,
            no_download=True,
        )


def test_server_config():
    """Test ServerConfig."""
    # Valid config
    config = ServerConfig(
        uri="tcp://localhost:10200",
        samples_per_chunk=2048,
        debug=True,
        log_format="%(levelname)s: %(message)s",
    )
    assert config.uri == "tcp://localhost:10200"
    assert config.samples_per_chunk == 2048
    assert config.debug is True

    # Invalid samples_per_chunk (must be positive)
    with pytest.raises(ValidationError):
        ServerConfig(
            uri="tcp://localhost:10200",
            samples_per_chunk=0,
            log_format="%(levelname)s: %(message)s",
        )


def test_orpheus_config():
    """Test the complete OrpheusConfig."""
    # Create sub-configs
    tts_config = TTSConfig(voice="zoe", temperature=0.7)
    model_config = ModelConfig(model_path=Path("model.gguf"), repo_id="user/repo")
    server_config = ServerConfig(
        uri="tcp://localhost:10200", log_format="%(levelname)s: %(message)s"
    )

    # Create complete config
    config = OrpheusConfig(tts=tts_config, model=model_config, server=server_config)

    # Check values
    assert config.tts.voice == "zoe"
    assert config.model.model_path == Path("model.gguf")
    assert config.server.uri == "tcp://localhost:10200"

    # Convert to dict
    config_dict = config.dict()
    assert config_dict["tts"]["voice"] == "zoe"
    assert config_dict["model"]["repo_id"] == "user/repo"


def test_config_serialization():
    """Test serialization and deserialization of config."""
    # Create a config
    config = OrpheusConfig(
        tts=TTSConfig(voice="leo", temperature=0.7),
        model=ModelConfig(model_path=Path("model.gguf"), repo_id="user/repo"),
        server=ServerConfig(
            uri="tcp://localhost:10200", log_format="%(levelname)s: %(message)s"
        ),
    )

    # Serialize to JSON
    config_json = config.json()

    # Deserialize from JSON
    config2 = OrpheusConfig.parse_raw(config_json)

    # Check equality
    assert config.tts.voice == config2.tts.voice
    assert config.model.repo_id == config2.model.repo_id
    assert config.server.uri == config2.server.uri

    # With temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as temp_file:
        # Write to file
        temp_file.write(config_json)
        temp_file.flush()

        # Read from file
        with open(temp_file.name, "r") as f:
            config_data = json.load(f)

        # Parse as config
        config3 = OrpheusConfig.parse_obj(config_data)

        # Check equality
        assert config.tts.voice == config3.tts.voice
        assert config.model.repo_id == config3.model.repo_id
        assert config.server.uri == config3.server.uri


def test_config_from_args():
    """Test creating config from argparse-like namespace."""

    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.model_path = Path("model.gguf")
            self.repo_id = "user/repo"
            self.voice = "leo"
            self.temperature = 0.7
            self.uri = "tcp://localhost:10200"
            self.debug = True
            self.log_format = "%(levelname)s: %(message)s"

    args = MockArgs()

    # Create config from args
    config = OrpheusConfig.from_args(args)

    # Check values
    assert config.tts.voice == "leo"
    assert config.tts.temperature == 0.7
    assert config.model.model_path == Path("model.gguf")
    assert config.model.repo_id == "user/repo"
    assert config.server.uri == "tcp://localhost:10200"
    assert config.server.debug is True
