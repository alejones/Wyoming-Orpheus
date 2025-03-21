"""Wyoming server for Orpheus-TTS."""
from pathlib import Path
from typing import Final

_DIR: Final = Path(__file__).parent
_VERSION_PATH: Final = _DIR / "VERSION"

__version__ = _VERSION_PATH.read_text(encoding="utf-8").strip()

__all__ = ["__version__"]