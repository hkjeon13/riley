"""Build-only FA3 experiment: export immutable sources; no Python serving path.

Usage: python build_fa3_native_probe.py FLASH_ATTENTION_CHECKOUT NVCC FRESH_OUTPUT
Dependencies must already be fetched. No pip, Torch, network, or GPU execution.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

FA3 = "98eb7998a0eba4047c7a30375522569d4b8efb20"
CUTLASS = "7127592069c2fe01b041e174ba4345ef9b279671"


def export(repo, revision, destination, paths):
    destination.mkdir()
    archive = destination.parent / (destination.name + ".tar")
    with archive.open("wb") as output:
        subprocess.run(["git", "-C", str(repo), "archive", revision, *paths],
                       stdout=output, check=True)
    with tarfile.open(archive) as contents:
        contents.extractall(destination, filter="data")


def prepare(source, output):
    # Never replace an earlier result, nor edit either dependency checkout.
    output.mkdir(parents=True, exist_ok=False)
    export(source, FA3, output / "fa3", ["hopper", "LICENSE", "AUTHORS"])
    export(source / "csrc/cutlass", CUTLASS, output / "cutlass",
           ["include", "tools/util/include", "LICENSE.txt"])
    header = output / "fa3/hopper/cuda_check.h"
    original = header.read_text()
    # Pin the exact source through git archive and require both error sites.
    if original.count("exit(1);") != 2:
        raise ValueError("pinned FA3 error handling no longer matches")
    patched = original.replace("exit(1);", "throw RileyFa3CudaError{status_};", 1)
    patched = patched.replace("exit(1);", "throw RileyFa3CutlassError{status_};", 1)
    declarations = """
struct RileyFa3CudaError { cudaError_t status; };
struct RileyFa3CutlassError { cutlass::Status status; };
"""
    patched = patched.replace("#define CHECK_CUDA(call)", declarations + "\n#define CHECK_CUDA(call)", 1)
    header.write_text(patched)
    return {"fa3_commit": FA3, "cutlass_commit": CUTLASS,
            "original_error_header_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "patched_error_header_sha256": hashlib.sha256(patched.encode()).hexdigest()}


def build(source, nvcc, output):
    source, nvcc, output = source.resolve(), nvcc.resolve(), output.absolute()
    receipt = prepare(source, output)
    probe = Path(__file__).with_name("fa3_native_compile_probe.cu")
    command = [str(nvcc), "-std=c++17", "-O3", "-arch=sm_90a",
               "--expt-relaxed-constexpr", "--expt-extended-lambda", "-Xptxas=-v",
               "-I" + str(output / "fa3/hopper"),
               "-I" + str(output / "cutlass/include"),
               "-I" + str(output / "cutlass/tools/util/include"),
               str(probe), str(output / "fa3/hopper/flash_prepare_scheduler.cu"),
               "-lcuda", "-o", str(output / "fa3-native-probe")]
    receipt.update({"command": command,
               "nvcc": subprocess.check_output([str(nvcc), "--version"], text=True),
               "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
               "scope": "BF16 head64 dense causal and paged ragged prefill/decode native compile/link; host error contract"})
    with (output / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt["build_exit"] = result.returncode
    if result.returncode == 0:
        executable = output / "fa3-native-probe"
        receipt["binary_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
        result = subprocess.run([str(executable)], text=True, capture_output=True)
        receipt["host_contract_exit"] = result.returncode
        (output / "host-contract.stdout").write_text(result.stdout)
        (output / "host-contract.stderr").write_text(result.stderr)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    sys.exit(build(*map(Path, sys.argv[1:])))
