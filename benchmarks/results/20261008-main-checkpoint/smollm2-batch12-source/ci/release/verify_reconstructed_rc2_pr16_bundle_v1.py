#!/usr/bin/env python3
"""Verify the reviewed reconstructed RC2 PR16 bundle manifest contract.

This is deliberately a narrow historical verifier, not a general legacy
release-profile selector.  It never imports or executes code from the
historical source archive.  The one accepted manifest is pinned to the
reviewed RC2 source revision, epoch, version, canonical bytes, and the
then-current server-defaults source contract.  Shared tar, checksum, ELF, and
native-dependency checks remain in :mod:`verify_release_bundle`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from release_common import ReleaseContractError
from verify_release_bundle import _verify_bundle


RECONSTRUCTED_RC2_TARGET = "6093006ec2b01b784b01ba278296b676f2dfd03a"
RECONSTRUCTED_RC2_SOURCE_DATE_EPOCH = 1_787_811_743
RECONSTRUCTED_RC2_VERSION = "0.1.0"
RECONSTRUCTED_RC2_ARCHIVE_ROOT = "riley-0.1.0-linux-x86_64-cuda12.8"
RECONSTRUCTED_RC2_SERVER_DEFAULTS_SOURCE_PATH = "crates/riley-server/src/main.rs"
RECONSTRUCTED_RC2_SERVER_DEFAULTS_SOURCE_SHA256 = (
    "1f50fec5b886703fe110c9f0c62560a51193baaaf1d498713c9ba8c17f00d9be"
)
RECONSTRUCTED_RC2_MANIFEST_BYTE_LENGTH = 10_909
RECONSTRUCTED_RC2_MANIFEST_SHA256 = (
    "3da42b3d0bbf1a56ce8768a5cc7bfb175cc969d57c3727ee9b9b0cfd1df6028e"
)


def _reconstructed_rc2_manifest(
    raw: bytes,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Accept only the reviewed historical bytes, never a live template."""

    if (
        len(raw) != RECONSTRUCTED_RC2_MANIFEST_BYTE_LENGTH
        or hashlib.sha256(raw).hexdigest() != RECONSTRUCTED_RC2_MANIFEST_SHA256
    ):
        raise ReleaseContractError("reconstructed RC2 manifest differs from its reviewed pin")
    artifact = manifest.get("artifact")
    defaults = manifest.get("defaults")
    if not isinstance(artifact, dict) or not isinstance(defaults, dict):
        raise ReleaseContractError("reconstructed RC2 manifest has an invalid closed shape")
    if artifact.get("version") != RECONSTRUCTED_RC2_VERSION:
        raise ReleaseContractError("reconstructed RC2 bundle version differs from the reviewed contract")
    if artifact.get("source_revision") != RECONSTRUCTED_RC2_TARGET:
        raise ReleaseContractError("reconstructed RC2 bundle revision differs from the reviewed target")
    if artifact.get("source_date_epoch") != RECONSTRUCTED_RC2_SOURCE_DATE_EPOCH:
        raise ReleaseContractError("reconstructed RC2 bundle epoch differs from the reviewed target")
    if defaults.get("source_contract") != {
        "path": RECONSTRUCTED_RC2_SERVER_DEFAULTS_SOURCE_PATH,
        "sha256": RECONSTRUCTED_RC2_SERVER_DEFAULTS_SOURCE_SHA256,
    }:
        raise ReleaseContractError("reconstructed RC2 source contract differs from the reviewed pin")
    return manifest, RECONSTRUCTED_RC2_ARCHIVE_ROOT


def verify_reconstructed_rc2_pr16_bundle(
    bundle: Path,
    *,
    max_total_bytes: int | None = None,
) -> None:
    """Replay one PR16 bundle under the closed reconstructed-RC2 contract."""

    _verify_bundle(
        bundle,
        max_total_bytes=max_total_bytes,
        manifest_contract=_reconstructed_rc2_manifest,
    )
