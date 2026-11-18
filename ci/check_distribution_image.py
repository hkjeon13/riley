#!/usr/bin/env python3
"""Verify the built CUDA 12.4 distribution image without a GPU or network."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess

from check_serving_source_contract import validate as validate_sources


IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def digest(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def capture(out, name, argv):
    path = out / name
    with path.open("xb") as output:
        subprocess.run(argv, check=True, stdout=output, timeout=90)
    return path.read_text(encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(inspect, image, revision, archive_sha):
    require(len(inspect) == 1 and inspect[0]["Id"] == image, "unexpected image inspection identity")
    document = inspect[0]
    require((document["Os"], document["Architecture"]) == ("linux", "amd64"), "expected a Linux AMD64 image")
    config = document["Config"]
    require(config["User"] == "65532:65532", "distribution must launch as the non-root runtime user")
    require(config["Entrypoint"] == ["/opt/riley/bin/riley-docker"] and config["Cmd"] == ["help"],
            "image entrypoint or default command drifted")
    require("/models" in config["Volumes"] and "8080/tcp" in config["ExposedPorts"],
            "runtime mount or listen port drifted")
    env = dict(value.split("=", 1) for value in config["Env"])
    require(env.get("RILEY_MODEL_DIR") == "/models" and env.get("RILEY_BIND") == "0.0.0.0:8080",
            "runtime environment defaults drifted")
    labels = config["Labels"]
    require(labels["org.opencontainers.image.revision"] == revision
            and labels["io.riley.source-archive-sha256"] == archive_sha,
            "image does not match this source archive")


def read_hashes(source):
    result = {}
    for line in source.splitlines():
        sha, name = line.split("  ", 1)
        require(re.fullmatch("[0-9a-f]{64}", sha) is not None and name not in result,
                f"invalid or duplicate runtime file digest for {name}")
        result[name] = sha
    return result


def check_image(image, revision, source_root, source_archive, out):
    require(IMAGE_ID.fullmatch(image) is not None, "pass an immutable Docker image ID")
    require(re.fullmatch("[0-9a-f]{40}", revision) is not None, "pass a full source revision")
    sources = validate_sources(source_root)
    archive_sha = digest(source_archive)
    inspect = json.loads(capture(out, "image-inspect.json", ["docker", "image", "inspect", image]))
    check_config(inspect, image, revision, archive_sha)
    base = ["docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
    command = [*base, image]
    version = capture(out, "version.txt", [*command, "--version"]).strip()
    require(re.fullmatch(r"riley \S+ \(server=true, cuda=true, cuda_abi=\d+\)", version) is not None,
            "expected a CUDA-enabled Riley server version")
    help_text = capture(out, "serve-help.txt", [*command, "serve", "--help"])
    require("riley serve --model PATH" in help_text and "--inference-profile" in help_text,
            "serve help is unavailable through the image entrypoint")
    checkpoint_help = capture(out, "checkpoint-help.txt", [*command, "riley-checkpoint", "--help"])
    require("riley-checkpoint (offline by default)" in checkpoint_help and "--allow-network" in checkpoint_help,
            "checkpoint executable is unavailable")
    presets = sorted((source_root / "deploy/docker/presets").glob("*.json"))
    preset_list = capture(out, "presets.txt", [*command, "presets"])
    require(sorted(line.split()[0] for line in preset_list.splitlines()) == [file.stem for file in presets],
            "bundled preset list differs from the source archive")

    shell = [*base, "--entrypoint", "/bin/sh", image, "-ec"]
    files = read_hashes(capture(out, "runtime.sha256", [
        *shell,
        'test "$(id -u):$(id -g)" = 65532:65532; '
        'find /opt/riley -type f -exec sha256sum {} +',
    ]))
    expected = {f"/opt/riley/presets/{file.name}": digest(file) for file in presets}
    for name, source in (
        ("/opt/riley/bin/riley-docker", "deploy/docker/entrypoint.sh"),
        ("/opt/riley/licenses/LICENSE", "LICENSE"),
        ("/opt/riley/licenses/flashinfer/LICENSE", "third_party/flashinfer/LICENSE"),
    ):
        expected[name] = digest(source_root / source)
    binaries = {"/opt/riley/bin/riley", "/opt/riley/bin/riley-checkpoint"}
    require(set(files) == set(expected) | binaries, "unexpected files in the Riley runtime tree")
    require(all(files[name] == expected[name] for name in expected), "runtime script, preset or license bytes differ")
    dependencies = capture(out, "runtime-dependencies.txt", [
        *shell, "ldd /opt/riley/bin/riley; ldd /opt/riley/bin/riley-checkpoint",
    ])
    require("not found" not in dependencies and "libcublas" in dependencies and "libcudart" in dependencies,
            "missing expected CUDA runtime dependencies")

    report = {"schema_version": "riley.distribution-image-check.v1", "status": "passed",
              "distribution": sources["distribution"], "source_revision": revision,
              "source_archive_sha256": archive_sha, "image_id": image, "version": version,
              "binaries": {name: files[name] for name in sorted(binaries)},
              "preset_count": len(presets),
              "checks": ["linux-amd64-configuration", "non-root-uid-gid", "offline-cli",
                         "runtime-tree", "native-linkage"]}
    with (out / "report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("revision")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check_image(args.image, args.revision, args.source_root, args.source_archive, args.out)


if __name__ == "__main__":
    main()
