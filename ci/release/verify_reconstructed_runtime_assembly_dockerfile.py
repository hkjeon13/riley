#!/usr/bin/env python3
"""Static guard for the source-free reconstructed-runtime assembly recipe.

This verifier deliberately checks only the reviewed Dockerfile source.  It
does not build an image or attest a bundle-to-image relation; the later
arm-specific raw capture receipt must bind that relation from immutable input
and output evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

DOCKERFILE = Path(__file__).with_name("ReconstructedRuntimeAssembly.Dockerfile")
PINNED_RUNTIME = (
    "nvidia/cuda:12.8.1-runtime-ubuntu22.04@"
    "sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd"
)
VERIFY_FROM = f"FROM --platform=linux/amd64 {PINNED_RUNTIME} AS verify-input"
RUNTIME_FROM = f"FROM --platform=linux/amd64 {PINNED_RUNTIME} AS runtime"
REQUIRED_ARGUMENTS = (
    "ARG RILEY_RECONSTRUCTION_ID",
    "ARG RILEY_SOURCE_REVISION",
    "ARG RILEY_SOURCE_ARCHIVE_SHA256",
    "ARG RILEY_REPRO_BUILD_INPUTS_SHA256",
    "ARG RILEY_RELEASE_BINARY_SHA256",
    "ARG RILEY_RELEASE_BUNDLE_SHA256",
    "ARG RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256",
)
INPUT_COPIES = (
    "COPY input/riley /assembly-input/riley",
    "COPY input/riley.tar.gz /assembly-input/riley.tar.gz",
)
INPUT_PREPARATION_RUN = (
    "RUN /usr/bin/chmod 0644 /assembly-input/riley /assembly-input/riley.tar.gz "
    "&& /usr/bin/mkdir -p /opt/riley "
    "&& /usr/bin/chown 65532:65532 /opt/riley"
)
INPUT_STAGE_USER = "USER 65532:65532"
INPUT_VERIFICATION_RUN = (
    'RUN (test "${RILEY_RECONSTRUCTION_ID}" = a || test '
    '"${RILEY_RECONSTRUCTION_ID}" = b) '
    "&& printf '%s\\n' \"${RILEY_SOURCE_REVISION}\" | grep -Ex '[0-9a-f]{40}' >/dev/null "
    "&& printf '%s\\n' \"${RILEY_SOURCE_ARCHIVE_SHA256}\" | grep -Ex '[0-9a-f]{64}' >/dev/null "
    "&& printf '%s\\n' \"${RILEY_REPRO_BUILD_INPUTS_SHA256}\" | grep -Ex '[0-9a-f]{64}' >/dev/null "
    "&& printf '%s\\n' \"${RILEY_RELEASE_BINARY_SHA256}\" | grep -Ex '[0-9a-f]{64}' >/dev/null "
    "&& printf '%s\\n' \"${RILEY_RELEASE_BUNDLE_SHA256}\" | grep -Ex '[0-9a-f]{64}' >/dev/null "
    "&& printf '%s\\n' \"${RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256}\" | grep -Ex '[0-9a-f]{64}' >/dev/null "
    "&& test \"$(sha256sum /assembly-input/riley | cut -d ' ' -f 1)\" = \"${RILEY_RELEASE_BINARY_SHA256}\" "
    "&& test \"$(sha256sum /assembly-input/riley.tar.gz | cut -d ' ' -f 1)\" = \"${RILEY_RELEASE_BUNDLE_SHA256}\" "
    "&& tar --extract --gzip --file /assembly-input/riley.tar.gz "
    "--no-same-owner --no-same-permissions --no-overwrite-dir "
    "--strip-components=1 --directory /opt/riley "
    "&& (cd /opt/riley && sha256sum --strict --check SHA256SUMS) "
    "&& test -z \"$(find /opt/riley -xdev \\( -type l -o -type b -o -type c -o -type p -o -type s \\) -print -quit)\" "
    "&& cmp --silent /assembly-input/riley /opt/riley/bin/riley "
    "&& test -x /opt/riley/bin/riley"
)
LABEL = (
    'LABEL org.riley.reconstructed-runtime-assembly.version="v1" '
    'org.riley.reconstructed-runtime-assembly.reconstruction-id="${RILEY_RECONSTRUCTION_ID}" '
    'org.riley.reconstructed-runtime-assembly.source-revision="${RILEY_SOURCE_REVISION}" '
    'org.riley.reconstructed-runtime-assembly.source-archive-sha256="${RILEY_SOURCE_ARCHIVE_SHA256}" '
    'org.riley.reconstructed-runtime-assembly.repro-build-inputs-sha256="${RILEY_REPRO_BUILD_INPUTS_SHA256}" '
    'org.riley.reconstructed-runtime-assembly.release-binary-sha256="${RILEY_RELEASE_BINARY_SHA256}" '
    'org.riley.reconstructed-runtime-assembly.release-bundle-sha256="${RILEY_RELEASE_BUNDLE_SHA256}" '
    'org.riley.reconstructed-runtime-assembly.recipe-normalized-instructions-sha256="${RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256}"'
)
FINAL_COPY = "COPY --from=verify-input --chown=65532:65532 /opt/riley/ /opt/riley/"
RUNTIME_ENVS = (
    "ENV PATH=/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "ENV NVIDIA_VISIBLE_DEVICES=all",
    "ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility",
)
RUNTIME_BASE_PYTHON_ARTIFACT_REMOVAL = (
    "RUN /usr/bin/rm /usr/share/apport/package-hooks/source_shadow.py "
    "/usr/share/gcc/python/libstdcxx/__init__.py "
    "/usr/share/gcc/python/libstdcxx/v6/__init__.py "
    "/usr/share/gcc/python/libstdcxx/v6/printers.py "
    "/usr/share/gcc/python/libstdcxx/v6/xmethods.py "
    "/usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/"
    "libstdc++.so.6.0.30-gdb.py"
)
FINAL_RUNTIME_CHECK = (
    "RUN export PATH=/usr/bin:/bin "
    "&& test -x /opt/riley/bin/riley "
    "&& test -s /opt/riley/SHA256SUMS "
    "&& test -s /opt/riley/manifest/native-dependencies.txt "
    "&& test -s /opt/riley/manifest/release.json "
    "&& (cd /opt/riley && /usr/bin/sha256sum --strict --check SHA256SUMS) "
    "&& test -z \"$(/usr/bin/find /opt/riley -xdev \\( "
    "-type l -o -type b -o -type c -o -type p -o -type s -o -perm /6000 "
    "-o \\( -type f -links +1 \\) \\) -print -quit)\" "
    "&& for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++; do "
    "if test -e \"/opt/riley/bin/${command}\" || command -v \"${command}\" >/dev/null 2>&1; then "
    "echo \"forbidden runtime executable: ${command}\" >&2; exit 1; "
    "fi; done "
    "&& test ! -e /assembly-input "
    "&& test ! -e /workspace "
    "&& if /usr/bin/find / -xdev -type f \\( "
    "-name '*.py' -o -name '*.pyc' -o -name '*.whl' "
    "-o -name '*.pkl' -o -name '*.pickle' \\) | /usr/bin/grep -q .; then "
    "echo 'forbidden Python artifact in runtime image' >&2; exit 1; fi"
)
FINAL_IDENTITY = (
    "USER 65532:65532",
    "EXPOSE 8080",
    'ENTRYPOINT ["/opt/riley/bin/riley"]',
    'CMD ["--help"]',
)
EXPECTED_NORMALIZED_INSTRUCTION_SHA256 = (
    "d80d657db557f9af62734aebef3527fcf46a0227de1f9ac1cacbbf0c70751114"
)


class RuntimeAssemblyContractError(ValueError):
    """Raised when the reviewed static assembly recipe is altered or unsafe."""


def _instructions(contents: str) -> list[str]:
    """Return Docker logical instructions without comments or continuations."""
    logical: list[str] = []
    current = ""
    for raw_line in contents.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
        else:
            logical.append(current)
            current = ""
    if current:
        raise RuntimeAssemblyContractError(
            "assembly Dockerfile ends with an unterminated continuation"
        )
    return logical


def _fail(message: str) -> None:
    raise RuntimeAssemblyContractError(message)


def verify_reconstructed_runtime_assembly_dockerfile(
    path: Path = DOCKERFILE,
) -> None:
    """Verify the fixed source-free runtime-image assembly recipe."""
    contents = path.read_text(encoding="utf-8")
    if re.search(r"(?im)^\s*#\s*syntax\s*=", contents):
        _fail("assembly Dockerfile must not depend on an external syntax frontend")
    if re.search(r"(?im)^\s*(?:ADD|ONBUILD|HEALTHCHECK|SHELL|STOPSIGNAL)\b", contents):
        _fail("assembly Dockerfile contains an unreviewed Docker instruction")
    if "--mount=" in contents:
        _fail("assembly Dockerfile must not use build mounts, secrets, or SSH forwarding")

    instructions = _instructions(contents)
    normalized = "\n".join(instructions) + "\n"
    normalized_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if normalized_sha256 != EXPECTED_NORMALIZED_INSTRUCTION_SHA256:
        _fail(
            "assembly Dockerfile normalized instruction stream differs from the "
            "reviewed exact order/content"
        )

    from_instructions = [
        instruction
        for instruction in instructions
        if instruction.upper().startswith("FROM ")
    ]
    if from_instructions != [VERIFY_FROM, RUNTIME_FROM]:
        _fail("assembly Dockerfile must have exactly two reviewed pinned runtime stages")
    runtime_index = instructions.index(RUNTIME_FROM)
    verify_stage = instructions[1:runtime_index]
    runtime_stage = instructions[runtime_index + 1 :]
    expected_verify_stage = [
        *REQUIRED_ARGUMENTS,
        *INPUT_COPIES,
        INPUT_PREPARATION_RUN,
        INPUT_STAGE_USER,
        INPUT_VERIFICATION_RUN,
    ]
    if verify_stage != expected_verify_stage:
        _fail(
            "input-verification stage must contain only the closed arguments, "
            "two selected inputs, and exact bundle/binary verification"
        )
    expected_runtime_stage = [
        *REQUIRED_ARGUMENTS,
        LABEL,
        FINAL_COPY,
        RUNTIME_BASE_PYTHON_ARTIFACT_REMOVAL,
        FINAL_RUNTIME_CHECK,
        *RUNTIME_ENVS,
        *FINAL_IDENTITY,
    ]
    if runtime_stage != expected_runtime_stage:
        _fail(
            "final runtime stage must contain only the closed provenance labels, "
            "verified bundle tree, and reviewed Python-free runtime boundary"
        )

    # This makes the policy visible independently of the instruction hash and
    # protects the contract if a reviewer intentionally updates that hash.
    for marker in (
        "RILEY_RECONSTRUCTION_ID",
        "RILEY_SOURCE_REVISION",
        "RILEY_SOURCE_ARCHIVE_SHA256",
        "RILEY_REPRO_BUILD_INPUTS_SHA256",
        "RILEY_RELEASE_BINARY_SHA256",
        "RILEY_RELEASE_BUNDLE_SHA256",
        "RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256",
        "sha256sum /assembly-input/riley",
        "sha256sum /assembly-input/riley.tar.gz",
        "tar --extract --gzip",
        "--no-same-owner --no-same-permissions --no-overwrite-dir --strip-components=1",
        "sha256sum --strict --check SHA256SUMS",
        "-type l -o -type b -o -type c -o -type p -o -type s",
        "-perm /6000",
        "-type f -links +1",
        "cmp --silent /assembly-input/riley /opt/riley/bin/riley",
        "export PATH=/usr/bin:/bin",
        'test -e "/opt/riley/bin/${command}"',
        "test ! -e /assembly-input",
        "test ! -e /workspace",
    ):
        if marker not in normalized:
            _fail(f"assembly Dockerfile lacks required boundary marker: {marker}")


def main() -> int:
    try:
        verify_reconstructed_runtime_assembly_dockerfile()
    except (OSError, RuntimeAssemblyContractError) as error:
        print(f"reconstructed runtime assembly Dockerfile verification failed: {error}", file=os.sys.stderr)
        return 1
    print("reconstructed runtime assembly Dockerfile contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
