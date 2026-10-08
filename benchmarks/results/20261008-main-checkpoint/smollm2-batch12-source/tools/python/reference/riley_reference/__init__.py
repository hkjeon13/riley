"""Reproducible Hugging Face fixtures for riley parity tests."""

from .constants import (
    DTYPE,
    MAX_CONTEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    RNG_ALGORITHM,
)

__all__ = [
    "DTYPE",
    "MAX_CONTEXT_TOKENS",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_WEIGHTS_SHA256",
    "RNG_ALGORITHM",
]

__version__ = "0.1.0"
