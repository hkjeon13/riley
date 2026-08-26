#!/usr/bin/env python3
"""Static guard for the minimal Python-free release Dockerfile contract."""

from __future__ import annotations

import os
import re
from pathlib import Path

from release_common import ReleaseContractError

DOCKERFILE = Path(__file__).with_name("Dockerfile")
PINNED_RUNTIME = (
    "nvidia/cuda:12.8.1-runtime-ubuntu22.04@"
    "sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd"
)
PINNED_RUSTUP_TOOLCHAIN = "1.85.0-x86_64-unknown-linux-gnu"
BUILDER_PACKAGE_INSTALL = (
    "RUN apt-get update && apt-get install -y --no-install-recommends "
    "build-essential ca-certificates cmake pkg-config python3 python3-tomli "
    "&& rm -rf /var/lib/apt/lists/*"
)
BUILDER_PREFLIGHT = (
    "RUN python3 ci/release/run_release_python.py "
    "ci/release/check_release_preflight.py "
    '--source-revision "${RUSTINFER_SOURCE_REVISION}" '
    '--source-date-epoch "${SOURCE_DATE_EPOCH}"'
)
BUILDER_BUNDLE = (
    "RUN mkdir -p /release && python3 ci/release/run_release_python.py "
    "ci/release/build_release_bundle.py --binary target/release/rustinfer "
    "--output /release/rustinfer.tar.gz "
    '--source-revision "${RUSTINFER_SOURCE_REVISION}" '
    '--source-date-epoch "${SOURCE_DATE_EPOCH}" '
    "&& python3 ci/release/run_release_python.py "
    "ci/release/verify_release_bundle.py /release/rustinfer.tar.gz "
    "&& mkdir -p /runtime-root && tar --extract --gzip "
    "--file /release/rustinfer.tar.gz --strip-components=1 "
    "--directory /runtime-root"
)


def _instructions(contents: str) -> list[str]:
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
        raise ReleaseContractError("Dockerfile ends with an unterminated continuation")
    return logical


def verify_dockerfile(path: Path = DOCKERFILE) -> None:
    contents = path.read_text(encoding="utf-8")
    instructions = _instructions(contents)
    from_instructions = [line for line in instructions if line.upper().startswith("FROM ")]
    if len(from_instructions) != 3:
        raise ReleaseContractError("release Dockerfile must have toolchain, builder, and runtime stages")
    if not re.fullmatch(rf"FROM {re.escape(PINNED_RUNTIME)} AS runtime", from_instructions[-1]):
        raise ReleaseContractError("final CUDA runtime image must use the reviewed immutable digest")
    runtime_index = instructions.index(from_instructions[-1])
    builder = instructions[:runtime_index]
    runtime = instructions[runtime_index + 1 :]
    toolchain_environment = f"ENV RUSTUP_TOOLCHAIN={PINNED_RUSTUP_TOOLCHAIN}"
    if builder.count(toolchain_environment) != 1:
        raise ReleaseContractError(
            "release builder must select the reviewed exact rustup toolchain"
        )
    if builder.count(BUILDER_PACKAGE_INSTALL) != 1:
        raise ReleaseContractError(
            "release builder must use the exact reviewed Python 3.10 package install"
        )
    required_helper_instructions = {BUILDER_PREFLIGHT, BUILDER_BUNDLE}
    if not required_helper_instructions <= set(builder):
        raise ReleaseContractError(
            "release builder must execute release helpers through the exact "
            "compatibility wrapper commands"
        )
    builder_text = "\n".join(builder)
    for helper in (
        "check_release_preflight.py",
        "build_release_bundle.py",
        "verify_release_bundle.py",
    ):
        marker = f"python3 ci/release/run_release_python.py ci/release/{helper}"
        if (
            builder_text.count(f"ci/release/{helper}") != 1
            or builder_text.count(marker) != 1
        ):
            raise ReleaseContractError(
                f"release builder must invoke {helper} once through the compatibility wrapper"
            )
    if any("RUSTUP_TOOLCHAIN" in line for line in runtime):
        raise ReleaseContractError(
            "final runtime must not inherit the builder rustup toolchain environment"
        )
    if any(line.upper().startswith(("FROM ", "ADD ")) for line in runtime):
        raise ReleaseContractError("runtime stage contains an unexpected stage or ADD instruction")
    copy_lines = [line for line in runtime if line.upper().startswith("COPY ")]
    if copy_lines != ["COPY --from=builder /runtime-root/ /opt/rustinfer/"]:
        raise ReleaseContractError("runtime stage may copy only the verified builder output")
    if any(
        line.upper().startswith("RUN ") and re.search(r"\b(apt|apt-get|apk|dnf|yum)\b", line)
        for line in runtime
    ):
        raise ReleaseContractError("runtime stage must not install packages")
    required = {
        'USER 65532:65532',
        'ENTRYPOINT ["/opt/rustinfer/bin/rustinfer"]',
        'CMD ["--help"]',
    }
    missing = required - set(runtime)
    if missing:
        raise ReleaseContractError("runtime stage is missing: " + ", ".join(sorted(missing)))
    runtime_text = "\n".join(runtime)
    for marker in (
        "for command in python python3 pip pip3 cargo rustc nvcc cmake make cc c++",
        "test -s /opt/rustinfer/SHA256SUMS",
        "test -s /opt/rustinfer/manifest/native-dependencies.txt",
        "test -s /opt/rustinfer/manifest/release.json",
        "ldd /opt/rustinfer/bin/rustinfer",
        "test ! -e /workspace",
        "find / -xdev -type f",
    ):
        if marker not in runtime_text:
            raise ReleaseContractError(f"runtime stage is missing static contract marker: {marker}")


def main() -> int:
    try:
        verify_dockerfile()
    except (OSError, ReleaseContractError) as error:
        print(f"runtime Dockerfile verification failed: {error}", file=os.sys.stderr)
        return 1
    print("runtime Dockerfile contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
