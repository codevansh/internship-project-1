"""Compatibility entry point. The canonical FastAPI app lives in src.api.main."""

from src.api.main import app

__all__ = ["app"]
