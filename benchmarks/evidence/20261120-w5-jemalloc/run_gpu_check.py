"""Run the pinned distribution GPU check with a process-local allocator override."""

from argparse import ArgumentParser
import json
from pathlib import Path
from unittest.mock import patch

import check_distribution_gpu as checker


IMAGE = "sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1"
LIBRARY = "/tmp/riley-w5-libjemalloc.so.2"
LIBRARY_SHA = "c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9"
START = "/tmp/riley-w5-start"
MALLOC_CONF = "abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0"
SHELL_PROOF = r"""
test "$(id -u):$(id -g)" = 65532:65532
test -z "${LD_PRELOAD:-}${MALLOC_CONF:-}${GLIBC_TUNABLES:-}"
test "$(readlink /proc/1/exe)" = /opt/riley/bin/riley
tr '\0' '\n' </proc/1/environ | grep -E '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)='
sha256sum "$1"
awk -v path="$1" '$NF == path { n++ } END { print n+0 }' /proc/1/maps
"""


def mount(source, target):
    checker.require(not set(str(source)).intersection(",\r\n"), "invalid allocator bind path")
    return f"type=bind,source={source},target={target},readonly"


def run(cpu_report, gpu, model_dir, model_sha, out, mode, stage):
    out = out.resolve()
    library = (stage / "libjemalloc.so.2").resolve(strict=True)
    start = (stage / "start_riley.sh").resolve(strict=True)
    checker.require(checker.digest(library) == LIBRARY_SHA, "allocator input hash changed")
    orig_docker, orig_ready = checker.docker, checker.wait_ready
    bindings = ["--entrypoint", START, "--mount", mount(start, START), "--mount", mount(library, LIBRARY)]

    def proof(container, point):
        runtime = checker.inspect_container(container)
        pid = runtime["State"]["Pid"]
        checker.require(runtime["Image"] == IMAGE and runtime["State"]["Running"]
                        and type(pid) is int and pid > 0 and runtime["Path"] == START,
                        "allocator check is on a different process or image")
        fields = orig_docker("exec", container, "/bin/sh", "-ec", SHELL_PROOF, "proof", LIBRARY).splitlines()
        expected = {f"LD_PRELOAD={LIBRARY}", f"MALLOC_CONF={MALLOC_CONF}"}
        checker.require(len(fields) == 4 and set(fields[:2]) == expected
                        and fields[2].split() == [LIBRARY_SHA, LIBRARY] and int(fields[3]) > 0,
                        "allocator was not confined to the target process or did not load")
        measured = {"image_id": IMAGE, "container_id": container, "host_pid": pid,
                    "library": LIBRARY, "library_sha256": LIBRARY_SHA, "malloc_conf": MALLOC_CONF,
                    "library_mapping_present": True, "exec_helper_uses_default_environment": True}
        if point == "after":
            checker.require(measured == json.loads((out.parent / "allocator-before.json").read_bytes()),
                            "allocator binding changed during the test")
        with (out.parent / f"allocator-{point}.json").open("x") as receipt:
            json.dump(measured, receipt, indent=2)
            receipt.write("\n")

    def docker(*args):
        if args[0] == "create":
            checker.require(IMAGE in args, "allocator patch was offered another image")
            return orig_docker("create", *bindings, *args[1:])
        if args[0] == "stop":
            proof(args[-1], "after")
        return orig_docker(*args)

    def wait_ready(container, port):
        report = orig_ready(container, port)
        proof(container, "before")
        return report

    with patch.object(checker, "docker", docker), patch.object(checker, "wait_ready", wait_ready):
        return checker.check_gpu(cpu_report, gpu, model_dir, model_sha, out, True, mixed_soak=mode)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("cpu_report", type=Path)
    parser.add_argument("gpu")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("model_sha")
    parser.add_argument("out", type=Path)
    parser.add_argument("mode", choices=checker.CACHE_BLOCKS)
    args = parser.parse_args()
    raise SystemExit(run(args.cpu_report, args.gpu, args.model_dir, args.model_sha,
                         args.out, args.mode, args.stage))
