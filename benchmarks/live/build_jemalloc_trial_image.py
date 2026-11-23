#!/usr/bin/env python3
"""Rebuild and check the W5 jemalloc trial image on a Linux AMD64 Docker host."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
RECIPE = EVIDENCE / "20261121-w5-current-image-allocator"
CONTEXT_HASHES = EVIDENCE / "20261123-w5-dot-allocator" / "context.sha256"
PACKAGE_SHA256 = "144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c"


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def install(source, destination, mode):
    shutil.copyfile(source, destination)
    destination.chmod(mode)


def build(stock_report, package, stub, output):
    stock = json.loads(stock_report.read_bytes())
    parent = stock["image_id"]
    if (stock["schema_version"] != "riley.distribution-image-check.v1"
            or stock["status"] != "passed" or not re.fullmatch(r"sha256:[0-9a-f]{64}", parent)):
        raise ValueError("expected a passing distribution report with a full image ID")
    if not (stock_report.parent / "runtime.sha256").is_file():
        raise ValueError("runtime.sha256 must be beside the stock report")
    if sha256(stub) != stock["cpu_cli_driver_stub_sha256"]:
        raise ValueError("the CUDA driver stub does not match the stock report")
    if sha256(package) != PACKAGE_SHA256:
        raise ValueError("the package does not match the pinned libjemalloc2 SHA-256")

    output.mkdir()
    context = output / "context"
    context.mkdir()
    with tempfile.TemporaryDirectory(dir=output) as extracted:
        subprocess.run(["dpkg-deb", "--extract", package, extracted], check=True)
        install(Path(extracted) / "usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
                context / "libjemalloc.so.2", 0o444)
        install(Path(extracted) / "usr/share/doc/libjemalloc2/copyright",
                context / "jemalloc-copyright", 0o444)
    install(RECIPE / "Dockerfile", context / "Dockerfile", 0o644)
    install(RECIPE / "riley-with-allocator", context / "riley-with-allocator", 0o555)
    shutil.copyfile(CONTEXT_HASHES, output / "context.sha256")
    subprocess.run(["sha256sum", "--strict", "--check", "context.sha256"], cwd=output, check=True)

    actual = subprocess.check_output(["docker", "image", "inspect", "--format",
                                      "{{.Id}} {{.Os}}/{{.Architecture}}", parent], text=True).strip()
    if actual != f"{parent} linux/amd64":
        raise ValueError("the stock image ID or platform differs from the report")
    tag = f"riley-local-allocator-parent:{uuid4().hex}"
    image_id = output / "image.id"
    subprocess.run(["docker", "image", "tag", parent, tag], check=True)
    try:
        with (output / "docker-build.log").open("wb") as log:
            subprocess.run(["docker", "build", "--pull=false", "--network=none", "--no-cache",
                            "--build-arg", f"RILEY_PARENT={tag}",
                            "--build-arg", f"RILEY_PARENT_ID={parent}",
                            "--build-arg", f"JEMALLOC_PACKAGE_SHA256={PACKAGE_SHA256}",
                            "--iidfile", image_id, context], check=True, stdout=log,
                           stderr=subprocess.STDOUT, env={**os.environ, "DOCKER_BUILDKIT": "0"})
    finally:
        subprocess.run(["docker", "image", "rm", tag], stdout=subprocess.DEVNULL, check=True)

    verified = subprocess.check_output([sys.executable, "-B", RECIPE / "verify_allocator_image.py",
                                        parent, image_id.read_text().strip(), context, stock_report,
                                        stub, output / "cpu-output"])
    (output / "verified-image.json").write_bytes(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock_report", type=Path, help="CPU report.json beside runtime.sha256")
    parser.add_argument("package", type=Path, help="pinned libjemalloc2_5.2.1-4ubuntu1_amd64.deb")
    parser.add_argument("stub", type=Path, help="libcuda.so.1 stub used by the stock CPU check")
    parser.add_argument("output", type=Path, help="new output directory under an existing parent")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        build(args.stock_report.absolute(), args.package.resolve(), args.stub.resolve(),
              args.output.resolve())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(args.output / "verified-image.json")


if __name__ == "__main__":
    main()
