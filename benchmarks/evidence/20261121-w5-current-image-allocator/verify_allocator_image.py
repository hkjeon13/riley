"""Check the packaged allocator through the same CLI and binary as the approved parent."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess


START = "/opt/riley/bin/riley-with-allocator"
LIB = "/opt/riley/lib/libjemalloc.so.2"
NOTICE = "/opt/riley/licenses/jemalloc/copyright"
LIB_SHA = "c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9"
DEB_SHA = "144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c"
BASE_RUN = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
REVIEWED_CONFIG = ("User", "Cmd", "Env", "Volumes", "ExposedPorts", "WorkingDir", "Healthcheck", "StopSignal", "Shell")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    with path.open("rb") as content:
        return file_digest(content, "sha256").hexdigest()


def hashes(output):
    rows = [line.split("  ", 1) for line in output.splitlines()]
    result = {name: value for value, name in rows}
    require(len(result) == len(rows) and all(re.fullmatch("[0-9a-f]{64}", value) for value in result.values()),
            "invalid runtime tree hash")
    return result


def inspect(image):
    require(re.fullmatch("sha256:[0-9a-f]{64}", image), "expected an image ID")
    document, = json.loads(subprocess.check_output(["docker", "image", "inspect", image], timeout=30))
    require((document["Id"], document["Os"], document["Architecture"]) == (image, "linux", "amd64"),
            "wrong inspected image or platform")
    return document


def run(out, name, image, *args, options=()):
    call = subprocess.run([*BASE_RUN, *options, image, *args], capture_output=True, text=True, timeout=60)
    (out / f"{name}.stdout").write_text(call.stdout)
    (out / f"{name}.stderr").write_text(call.stderr)
    call.check_returncode()
    return call.stdout, call.stderr


def image_config(parent, variant):
    reviewed, new = parent["Config"], variant["Config"]
    require(reviewed["Entrypoint"] == ["/opt/riley/bin/riley-docker"] and new["Entrypoint"] == [START]
            and all(reviewed.get(key) == new.get(key) for key in REVIEWED_CONFIG)
            and new["User"] == "65532:65532" and new["Cmd"] == ["help"], "container defaults differ")
    extra = {"io.riley.test.parent-image": parent["Id"], "io.riley.test.jemalloc-package-sha256": DEB_SHA}
    require(new["Labels"] == {**reviewed["Labels"], **extra}
            and not any(entry.split("=")[0] in {"LD_PRELOAD", "MALLOC_CONF", "GLIBC_TUNABLES"}
                        for entry in new["Env"]), "packaging labels or allocator environment differ")
    layers = parent["RootFS"]["Layers"]
    require(variant["RootFS"]["Layers"][:len(layers)] == layers
            and len(variant["RootFS"]["Layers"]) == len(layers) + 3, "unexpected derivative layers")


def tree(out, name, image):
    code = (
        'test "$(id -u):$(id -g)" = 65532:65532; '
        'test "$(command -v riley)" = /opt/riley/bin/riley; '
        'find /opt/riley -type f -exec sha256sum {} +'
    )
    return hashes(run(out, name, image, "-ec", code, options=("--entrypoint", "/bin/sh"))[0])


def verify(parent_id, variant_id, context, stock, stub, out):
    out.mkdir()
    report = json.loads(stock.read_bytes())
    require(report["status"] == "passed" and report["image_id"] == parent_id
            and digest(stub) == report["cpu_cli_driver_stub_sha256"], "stock CPU report or CUDA stub changed")
    image_config(inspect(parent_id), inspect(variant_id))
    parent_files = tree(out, "parent-runtime", parent_id)
    passed_files = hashes((stock.parent / "runtime.sha256").read_text())
    require(parent_files == passed_files and all(parent_files[path] == value
            for path, value in report["binaries"].items()), "approved parent tree changed")
    expected = {START: digest(context / "riley-with-allocator"), LIB: LIB_SHA,
                NOTICE: digest(context / "jemalloc-copyright")}
    require(digest(context / "libjemalloc.so.2") == LIB_SHA
            and not parent_files.keys() & expected.keys()
            and tree(out, "variant-runtime", variant_id) == {**parent_files, **expected},
            "the variant runtime file inventory or allocator differs")

    stub_mount = "/run/riley-ci-driver/libcuda.so.1"
    cli_options = ("--mount", f"type=bind,src={stub},dst={stub_mount},readonly",
                   "--env", "LD_LIBRARY_PATH=/run/riley-ci-driver")
    outputs = {}
    for name, args in (
        ("version", ("--version",)), ("serve-help", ("serve", "--help")),
        ("serve-short-help", ("serve", "-h")), ("default-command", ()), ("presets", ("presets",)),
        ("checkpoint-help", ("riley-checkpoint", "--help")),
    ):
        original, original_errors = run(out, f"parent-{name}", parent_id, *args, options=cli_options)
        observed, errors = run(out, f"variant-{name}", variant_id, *args, options=cli_options)
        require((original, original_errors) == (observed, errors) and observed.strip(),
                f"{name}: CLI output differs")
        outputs[name] = digest(out / f"variant-{name}.stdout")
    require("riley serve --model PATH" in (out / "variant-serve-help.stdout").read_text(),
            "serve entrypoint did not reach Riley")
    binding_options = (*cli_options, "--env", "LD_DEBUG=bindings", "--env", "LD_BIND_NOW=1")
    _, log = run(out, "allocator-bindings", variant_id, "serve", "--help", options=binding_options)
    riley = re.compile(r"binding file (?:/opt/riley/bin/)?riley \[0\] to " + re.escape(LIB)
                       + r" \[0\]: normal symbol .(malloc|calloc|realloc|free).")
    bound = sorted(set(riley.findall(log)))
    require(bound == ["calloc", "free", "malloc", "realloc"], "Riley did not bind the packaged allocator")

    return {"schema_version": "riley.experimental-allocator-image.v1", "status": "passed",
            "parent_image_id": parent_id, "image_id": variant_id, "source_revision": report["source_revision"],
            "source_archive_sha256": report["source_archive_sha256"], "parent_cpu_report_sha256": digest(stock),
            "binaries": report["binaries"], "same_parent_runtime_files": len(parent_files),
            "added_files_sha256": expected, "packaged_riley_allocator_symbols": bound,
            "cli_output_sha256": outputs, "binding_log_sha256": digest(out / "allocator-bindings.stderr")}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("image")
    for name in ("context", "stock", "stub", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = verify(args.parent, args.image, args.context, args.stock, args.stub, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
