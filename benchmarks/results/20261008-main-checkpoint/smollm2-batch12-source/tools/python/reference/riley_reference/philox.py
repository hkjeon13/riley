"""Portable ``riley.philox4x32-10.v1`` implementation.

The normative prose and known-answer vectors live in ``benchmarks/RNG.md``.
This module deliberately has no NumPy, PyTorch, or platform RNG dependency.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass

from .constants import RNG_ALGORITHM, UINT32_MAX, UINT64_MAX

_M0 = 0xD2511F53
_M1 = 0xCD9E8D57
_W0 = 0x9E3779B9
_W1 = 0xBB67AE85

_SEED_TAG = b"riley.philox4x32-10.v1/seed\0"
_STREAM_TAG = b"riley.philox4x32-10.v1/stream\0"
_DOMAIN_TAG = b"riley.philox4x32-10.v1/domain\0"
_FORK_TAG = b"riley.philox4x32-10.v1/fork\0"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _u32(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= UINT32_MAX:
        raise ValueError(f"{name} must be in [0, 2^32 - 1]")
    return value


def _u64(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in [0, 2^64 - 1]")
    return value


def _bytes(value: str | bytes, *, name: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8", errors="strict")
    if isinstance(value, bytes):
        return value
    raise TypeError(f"{name} must be str or bytes")


def _length_prefixed(value: bytes) -> bytes:
    if len(value) > UINT32_MAX:
        raise ValueError("length-prefixed input exceeds 2^32 - 1 bytes")
    return struct.pack("<I", len(value)) + value


def _mulhilo32(left: int, right: int) -> tuple[int, int]:
    product = left * right
    return (product >> 32) & UINT32_MAX, product & UINT32_MAX


def philox4x32_10(
    counter: Iterable[int], key: Iterable[int]
) -> tuple[int, int, int, int]:
    """Return one Random123-compatible Philox4x32-10 block."""

    counter_words = tuple(counter)
    key_words = tuple(key)
    if len(counter_words) != 4:
        raise ValueError("counter must contain exactly four uint32 words")
    if len(key_words) != 2:
        raise ValueError("key must contain exactly two uint32 words")
    c0, c1, c2, c3 = (
        _u32(word, name=f"counter[{index}]")
        for index, word in enumerate(counter_words)
    )
    k0, k1 = (
        _u32(word, name=f"key[{index}]")
        for index, word in enumerate(key_words)
    )
    for _ in range(10):
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & UINT32_MAX,
            lo1,
            (hi0 ^ c3 ^ k1) & UINT32_MAX,
            lo0,
        )
        k0 = (k0 + _W0) & UINT32_MAX
        k1 = (k1 + _W1) & UINT32_MAX
    return c0, c1, c2, c3


def derive_digest(
    master_seed: int, stream: str | bytes, domain: str | bytes
) -> bytes:
    """Derive the full 32-byte v1 identity from seed, stream, and domain."""

    seed = _u64(master_seed, name="master_seed")
    stream_bytes = _bytes(stream, name="stream")
    domain_bytes = _bytes(domain, name="domain")
    seed_digest = hashlib.sha256(_SEED_TAG + struct.pack("<Q", seed)).digest()
    stream_digest = hashlib.sha256(
        _STREAM_TAG + seed_digest + _length_prefixed(stream_bytes)
    ).digest()
    return hashlib.sha256(
        _DOMAIN_TAG + stream_digest + _length_prefixed(domain_bytes)
    ).digest()


def _identity_words(digest: bytes) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(digest) != 32:
        raise ValueError("derivation digest must contain exactly 32 bytes")
    return struct.unpack("<II", digest[:8]), struct.unpack("<II", digest[8:16])


@dataclass
class Philox4x32:
    """Request-owned state pointing to the next random word."""

    _digest: bytes
    block: int = 0
    word_offset: int = 0
    exhausted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self._digest, bytes) or len(self._digest) != 32:
            raise ValueError("derivation digest must contain exactly 32 bytes")
        _u64(self.block, name="block")
        if (
            isinstance(self.word_offset, bool)
            or not isinstance(self.word_offset, int)
            or not 0 <= self.word_offset <= 3
        ):
            raise ValueError("word_offset must be in [0, 3]")
        if not isinstance(self.exhausted, bool):
            raise TypeError("exhausted must be bool")
        if self.exhausted and not (
            self.block == UINT64_MAX and self.word_offset == 3
        ):
            raise ValueError("exhausted state must use the canonical terminal position")

    @property
    def digest(self) -> bytes:
        return self._digest

    @property
    def key(self) -> tuple[int, int]:
        return _identity_words(self._digest)[0]

    @property
    def nonce(self) -> tuple[int, int]:
        return _identity_words(self._digest)[1]

    def next_u32(self) -> int:
        if self.exhausted:
            raise OverflowError("Philox stream is exhausted")
        counter = (
            self.block & UINT32_MAX,
            (self.block >> 32) & UINT32_MAX,
            self.nonce[0],
            self.nonce[1],
        )
        value = philox4x32_10(counter, self.key)[self.word_offset]
        if self.word_offset < 3:
            self.word_offset += 1
        elif self.block < UINT64_MAX:
            self.block += 1
            self.word_offset = 0
        else:
            self.exhausted = True
        return value

    def uniform_open01(self) -> float:
        return uniform_open01(self.next_u32())

    def snapshot(self) -> dict[str, object]:
        return {
            "algorithm_id": RNG_ALGORITHM,
            "derivation_digest_hex": self._digest.hex(),
            "block": str(self.block),
            "word_offset": self.word_offset,
            "exhausted": self.exhausted,
        }

    @classmethod
    def restore(cls, snapshot: object) -> "Philox4x32":
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "algorithm_id",
            "derivation_digest_hex",
            "block",
            "word_offset",
            "exhausted",
        }:
            raise ValueError("snapshot has unexpected shape")
        if snapshot["algorithm_id"] != RNG_ALGORITHM:
            raise ValueError("snapshot algorithm_id is unsupported")
        digest_hex = snapshot["derivation_digest_hex"]
        if not isinstance(digest_hex, str) or not _HEX64_RE.fullmatch(digest_hex):
            raise ValueError("snapshot digest must be canonical lowercase hex")
        block_text = snapshot["block"]
        if (
            not isinstance(block_text, str)
            or not re.fullmatch(r"0|[1-9][0-9]*", block_text)
        ):
            raise ValueError("snapshot block must be a canonical decimal string")
        block = int(block_text)
        _u64(block, name="snapshot block")
        return cls(
            bytes.fromhex(digest_hex),
            block=block,
            word_offset=snapshot["word_offset"],  # type: ignore[arg-type]
            exhausted=snapshot["exhausted"],  # type: ignore[arg-type]
        )

    def fork(self, label: str | bytes) -> "Philox4x32":
        label_bytes = _bytes(label, name="label")
        child_digest = hashlib.sha256(
            _FORK_TAG + self._digest + _length_prefixed(label_bytes)
        ).digest()
        return Philox4x32(child_digest)


def derive(master_seed: int, stream: str | bytes, domain: str | bytes) -> Philox4x32:
    return Philox4x32(derive_digest(master_seed, stream, domain))


def uniform_open01(word: int) -> float:
    """Map one uint32 exactly to an IEEE-754 binary64 open-interval value."""

    return (float(_u32(word, name="word")) + 0.5) * (2.0**-32)
