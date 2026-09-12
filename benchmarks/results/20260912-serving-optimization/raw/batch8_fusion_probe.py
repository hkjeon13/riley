#!/usr/bin/env python3
"""Correctness-only synthetic fused RoPE/KV/attention probe: prepare, build, then run.

prepare --source-root SNAPSHOT --build-receipt BUILD.json
        --oracle-source ACCEPTED/graph_numerics.cu
        --candidate-entry enqueue_compiled_packed_decode_rope_attention
        --candidate-source-sha256 SHA256 --output-dir NEW_DIRECTORY
build --manifest NEW_DIRECTORY/fixtures.json --nvcc CUDA13/bin/nvcc
run --manifest NEW_DIRECTORY/fixtures.json

Only run executes GPU work. Inputs are deterministic synthetic BF16 values,
not model activations; layer numbers are independent synthetic seeds. Full accepted precise RoPE and two-warp attention form the oracle. No timing,
performance claim, source modification, or desktop/session control is included.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE = "kernels/src/graph_numerics.cu"
PRECISE = "kernels/src/graph_numerics_precise.cu"
PRECISE_SHA = "fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f"
HELPERS = {"batch7_attention_probe.py": "bf8f8deca087d6db53c6cce9238d05b2d014328b3f0000f864cbf30ebc56e7f1",
           "batch6_projection_probe.py": "e306157697d5584b1f5d59826326f1682e6be57c224108fed68e8a2c3146e3d9"}
EXPECTED_UUID = "9087e4256acab722b8c9cc0423b39fb0"
DRIVER_SHA = "266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4"
ORACLE_SHA256 = "a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e"
PATTERNS = ("bounded_fingerprint", "signed_zero_impulses", "tail_cancellation")
MAPPINGS = ("identity", "reverse", "affine_7_3")
COUNTS = {"cases": 8640, "layers": 30, "positions": 32, "patterns": 3, "mappings": 3}
FLAGS = ("exact_outputs", "finite_outputs", "guards_intact", "inputs_unchanged",
         "oracle_outputs_unchanged", "mapping_invariant", "all_allocations_freed",
         "inactive_kv_unchanged", "current_values_raw_exact")
DEPENDENCIES = (SOURCE, PRECISE, "kernels/src/ffi_internal.hpp", "kernels/include/riley_cuda.h",
                "kernels/CMakeLists.txt", "crates/riley-cuda/build.rs")
RECIPE = {
    "version": "fused-rope-attention-integer-v1",
    "input_kind": "synthetic_only_no_checkpoint_activations_or_trig_tables",
    "generator": "pinned native probe fixture()/sample()",
    "qkv_bytes": 1920, "q_bytes": 1152, "kv_bytes_each": 98304,
    "table_bytes_each": 20480, "metadata_bytes": 116, "allocations_per_case": 12,
    "physical_blocks": 16, "logical_tokens": 160, "query_heads": 9,
    "kv_heads": 3, "head_dim": 64, "guard_bytes_each_side": 256,
    "positions_inclusive": [128, 159], "patterns": list(PATTERNS), "mappings": list(MAPPINGS),
    "metadata_offsets": {"position": 4, "live_blocks": 12, "block_ids": 16, "valid_counts": 80, "sequence_slot": 112},
    "comparison": "all BF16 attention and rotated Q words and full physical K/V; bit exact including signed zeros",
    "current_kv": "poisoned NaN in both independent buffers before oracle or candidate",
    "unused_and_future_slots": "initialized finite and immutable; all slots checked independently",
    "oracle": "full precise RoPE/KV then accepted two-warp attention from preserved source prefix",
}


def helper_evidence():
    paths = {name: evidence(Path(__file__).resolve().parent/name) for name in HELPERS}
    require(all(paths[name]["sha256"] == digest for name, digest in HELPERS.items()), "production flag helper changed")
    return paths


def load_helper(name):
    helper_evidence()
    spec = importlib.util.spec_from_file_location("batch8_"+name, Path(__file__).resolve().parent/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs_unique)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_evidence(value):
    require(set(value) == {"path", "sha256"}, "malformed artifact evidence")
    require(sha(value["path"]) == value["sha256"], "artifact changed: " + value["path"])


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def check_snapshot(build_path):
    build_path = Path(build_path).resolve(strict=True)
    build = read(build_path)
    source = Path(build["source_root"]).resolve(strict=True)
    require(git(source, "rev-parse", "HEAD") == build["source_commit"], "source commit differs from build")
    require(git(source, "rev-parse", "HEAD^") == build["parent_source_commit"], "built parent differs")
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "source snapshot is dirty")
    require(build["source_files"] and len(build["binaries"]) == 2, "incomplete source/binary build pins")
    for name, expected in build["source_files"].items():
        path = source / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(source), "source pin escapes snapshot")
        require(sha(path) == expected, "source file changed: " + name)
    for path, expected in build["binaries"].items():
        require(sha(path) == expected, "serving binary changed: " + path)
    return build


def cases():
    for index in range(COUNTS["cases"]):
        yield {"case_id": index, "layer": index//288, "position": 128+(index//9)%32,
               "pattern": PATTERNS[(index//3)%3], "mapping": MAPPINGS[index%3]}


def case_index():
    return "".join(f"{i}\t{i//288}\t{128+(i//9)%32}\t{(i//3)%3}\t{i%3}\n" for i in range(COUNTS["cases"]))


def source_contract(source, oracle, entry, expected):
    require(entry == "enqueue_compiled_packed_decode_rope_attention", "invalid candidate wrapper identifier")
    require(sha(Path(source)/PRECISE) == PRECISE_SHA, "precise oracle source changed")
    require(entry not in ("enqueue_compiled_attention", "enqueue_compiled_attention_rows", "enqueue_compiled_packed_decode_attention"), "candidate must be a distinct wrapper")
    require(re.fullmatch(r"[0-9a-f]{64}", expected), "explicit candidate source SHA256 required")
    require(sha(oracle) == ORACLE_SHA256, "unknown accepted attention oracle source")
    original, candidate = Path(oracle).read_bytes(), (Path(source)/SOURCE).read_bytes()
    require(hashlib.sha256(candidate).hexdigest() == expected, "candidate source differs from explicit identity")
    require(candidate.startswith(original) and len(candidate)>len(original), "candidate must preserve the complete accepted oracle source prefix")
    added = candidate[len(original):].decode()
    require(re.search(r"cudaError_t\s+"+re.escape(entry)+r"\s*\(", added), "distinct candidate definition is missing from appended source")


def prepare(source_root, build_path, oracle_source, candidate_entry, candidate_source_sha256, output):
    source, build_path = Path(source_root).resolve(strict=True), Path(build_path).resolve(strict=True)
    oracle = Path(oracle_source).resolve(strict=True)
    build = check_snapshot(build_path)
    require(Path(build["source_root"]).resolve() == source, "source root differs from build")
    source_contract(source, oracle, candidate_entry, candidate_source_sha256)
    output = Path(output).resolve()
    require(not output.is_relative_to(source), "probe output must be outside preserved source")
    output.mkdir(parents=True, exist_ok=False)
    index = output/"cases.tsv"
    with index.open("x") as stream:
        stream.write(case_index())
    manifest = {"schema_version": "riley.batch8-fusion-fixtures.v1", **COUNTS,
        "source_root": str(source), "source_commit": build["source_commit"],
        "source_build": evidence(build_path), "source_files": build["source_files"], "binaries": build["binaries"],
        "source_dependencies": {name: evidence(source/name) for name in DEPENDENCIES},
        "oracle_source": evidence(oracle), "candidate_entry": candidate_entry,
        "candidate_source_sha256": candidate_source_sha256,
        "helpers": helper_evidence(), "runner": evidence(__file__), "probe_source": evidence(Path(__file__).with_suffix(".cu")),
        "case_index": evidence(index), "fixture_recipe": RECIPE,
        "gpu_tests_executed": False, "performance_measured": False, "performance_claim_eligible": False}
    write_new(output/"fixtures.json", manifest)
    validate_manifest(output/"fixtures.json")
    return {"fixture_manifest": evidence(output/"fixtures.json"), **COUNTS, "gpu_tests_executed": False}


def validate_manifest(path):
    manifest = read(path)
    require(manifest["helpers"] == helper_evidence(), "helper pins differ")
    require(manifest["schema_version"] == "riley.batch8-fusion-fixtures.v1", "unknown fixture schema")
    require(all(manifest[key] == value for key, value in COUNTS.items()) and manifest["fixture_recipe"] == RECIPE, "synthetic fixture recipe/coverage differs")
    for key in ("source_build", "oracle_source", "runner", "probe_source", "case_index"):
        check_evidence(manifest[key])
    require(manifest["runner"] == evidence(__file__) and manifest["probe_source"] == evidence(Path(__file__).with_suffix(".cu")), "probe helper identity differs")
    build = check_snapshot(manifest["source_build"]["path"])
    require(all(manifest[key] == build[key] for key in ("source_root", "source_commit", "source_files", "binaries")), "source/build identity differs")
    expected = {name: evidence(Path(manifest["source_root"])/name) for name in DEPENDENCIES}
    require(manifest["source_dependencies"] == expected, "source dependencies differ")
    source_contract(manifest["source_root"], manifest["oracle_source"]["path"], manifest["candidate_entry"], manifest["candidate_source_sha256"])
    require(Path(manifest["case_index"]["path"]).read_text() == case_index(), "native case index differs from complete coverage")
    require(manifest["gpu_tests_executed"] is False and manifest["performance_measured"] is False and manifest["performance_claim_eligible"] is False, "fixtures claim execution or performance")
    return manifest


def production_flags(manifest, nvcc):
    fast = load_helper("batch7_attention_probe").production_flags(manifest, nvcc)
    precise = load_helper("batch6_projection_probe").production_flags(manifest, nvcc)
    require(fast["runtime_library_link"] == precise["runtime_library_link"], "TU runtime links differ")
    return {"fast_math": fast, "precise": precise}


def compile_commands(manifest, nvcc, options, directory, runtime):
    kernel, precise, host, binary = (directory/name for name in ("attention.o", "precise.o", "probe.o", "batch8_fusion_probe"))
    arch = [flag for flag in options["fast_math"]["flags"] if "arch=compute_89" in flag]
    return [
        [str(nvcc), *options["fast_math"]["flags"], "-x", "cu", "-c", str(Path(manifest["source_root"])/SOURCE), "-o", str(kernel)],
        [str(nvcc), *options["precise"]["flags"], "-x", "cu", "-c", str(Path(manifest["source_root"])/PRECISE), "-o", str(precise)],
        [str(nvcc), "-std=c++17", "-O2", *arch, "-DRILEY_ATTENTION_CANDIDATE="+manifest["candidate_entry"], "-c", manifest["probe_source"]["path"], "-o", str(host)],
        [str(nvcc), "--cudart=shared", str(kernel), str(precise), str(host), "-L"+str(runtime.parent), "-Xlinker", "-rpath", "-Xlinker", str(runtime.parent), "-o", str(binary)],
    ]


def build_probe(manifest_path, nvcc):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(manifest_path)
    nvcc = Path(nvcc).resolve(strict=True)
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    require(re.search(r"release 13\.", version), "the probe requires the production CUDA13 compiler")
    options = production_flags(manifest, nvcc)
    directory = manifest_path.parent / "build"
    directory.mkdir(exist_ok=False)
    kernel_object, precise_object, host_object, binary = (directory/name for name in ("attention.o", "precise.o", "probe.o", "batch8_fusion_probe"))
    runtime = Path(options["fast_math"]["runtime_library_link"])
    commands = compile_commands(manifest, nvcc, options, directory, runtime)
    log = directory / "compile.log"
    with log.open("x") as stream:
        for command in commands:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=stream, check=True)
    require(validate_manifest(manifest_path) == manifest, "identity changed during compile")
    for tu in options.values():
        for artifact in tu["cmake_artifacts"]:
            check_evidence(artifact)
    receipt = {"schema_version": "riley.batch8-fusion-build.v1", "fixture_manifest": evidence(manifest_path),
        "source_commit": manifest["source_commit"], "compiler": evidence(nvcc), "compiler_version": version,
        "runtime_library": evidence(runtime), "runtime_library_directory": str(runtime.parent),
        "runtime_library_link": str(runtime.absolute()),
        "production_flags": options, "commands": commands, "binary": evidence(binary),
        "kernel_object": evidence(kernel_object), "precise_object": evidence(precise_object), "host_object": evidence(host_object), "log": evidence(log),
        "gpu_tests_executed": False, "performance_measured": False, "performance_claim_eligible": False}
    write_new(manifest_path.parent / "compile.json", receipt)
    return receipt


def validate_compile(path, manifest_path):
    compiled = read(path)
    require(compiled["schema_version"] == "riley.batch8-fusion-build.v1" and compiled["fixture_manifest"] == evidence(manifest_path), "compile receipt differs from fixtures")
    for key in ("compiler", "runtime_library", "binary", "kernel_object", "precise_object", "host_object", "log"):
        check_evidence(compiled[key])
    for tu in compiled["production_flags"].values():
        for artifact in tu["cmake_artifacts"]:
            check_evidence(artifact)
    manifest = read(manifest_path)
    require(compiled["source_commit"] == manifest["source_commit"], "compiled source identity differs")
    options = production_flags(manifest, Path(compiled["compiler"]["path"]))
    require(options == compiled["production_flags"], "compiled production flags differ")
    runtime = Path(compiled["runtime_library_link"])
    require(str(runtime) == options["fast_math"]["runtime_library_link"] and evidence(runtime) == compiled["runtime_library"] and str(runtime.parent) == compiled["runtime_library_directory"], "shared CUDA runtime selection differs")
    require(compiled["commands"] == compile_commands(manifest, Path(compiled["compiler"]["path"]), options, Path(path).parent/"build", runtime), "standalone compile commands differ")
    require(compiled["gpu_tests_executed"] is False and compiled["performance_measured"] is False and compiled["performance_claim_eligible"] is False, "compile receipt claims GPU execution")
    return compiled


def validate_raw(path, manifest):
    rows = [json.loads(line, object_pairs_hook=pairs_unique) for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(rows) == COUNTS["cases"]+2 and all(row.get("schema") == "riley.batch8-fusion-native.v1" for row in rows), "incomplete or unknown native output")
    device, summary = rows[0], rows[-1]
    require(device["kind"] == "device" and device["compute_major"] == 8 and device["compute_minor"] == 9 and device["ordinal"] == 0 and device["runtime_version"] == 13000, "native device/runtime differs")
    require(device["uuid_hex"] == EXPECTED_UUID, "native GPU UUID differs from campaign binding")
    for row, case in zip(rows[1:-1], cases()):
        require(row["kind"] == "case" and row["passed"] is True and all(row[key] == value for key, value in case.items()), "attention case failed or coverage differs")
        require(row["compared_bytes"] == [1152,1152,98304,98304], "incomplete output comparison")
        require(all(row[key] is True for key in FLAGS), "numeric/finite/guard/immutability/map/lifetime check failed")
        require(row["mismatch_words"] == [0,0,0,0] and "first_mismatch" not in row, "fused numeric mismatch")
    require(summary["kind"] == "summary" and summary["passed"] is True and all(summary[key] == value for key, value in COUNTS.items()), "summary success/coverage differs")
    require(summary["device_allocations_created"] == summary["device_allocations_freed"] == 12*COUNTS["cases"], "allocation count differs")
    require(summary["failed_cases"] == summary["live_device_allocations"] == summary["live_device_bytes"] == summary["cleanup_errors"] == 0 and summary["all_allocations_freed"] is True and summary["stream_destroyed"] is True, "cleanup failed")
    require(summary["performance_measured"] is False and summary["performance_claim_eligible"] is False and summary["error"] == "", "native summary error/performance claim")
    return device, summary


def run_probe(manifest_path, driver_library_dir):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(manifest_path)
    compile_path = manifest_path.parent / "compile.json"
    compiled = validate_compile(compile_path, manifest_path)
    command = [compiled["binary"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"]
    output, errors = manifest_path.parent / "native-results.jsonl", manifest_path.parent / "native-stderr.log"
    env = dict(os.environ)
    require(not env.get("LD_PRELOAD"), "LD_PRELOAD would change the runtime")
    driver_dir = Path(driver_library_dir).resolve(strict=True)
    driver = evidence(driver_dir/"libcuda.so.1")
    require(driver["sha256"] == DRIVER_SHA, "private driver differs from verified loaded-kernel runtime")
    libraries = [str(driver_dir), compiled["runtime_library_directory"]]
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    with output.open("x") as stdout, errors.open("x") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
    require(process.returncode == 0, "native attention probe failed; inspect " + str(output))
    require(validate_manifest(manifest_path) == manifest and validate_compile(compile_path, manifest_path) == compiled, "identity changed during GPU probe")
    device, summary = validate_raw(output, manifest)
    check_evidence(driver)
    result = {"schema_version": "riley.batch8-fusion-correctness.v1", "passed": True,
        "gpu_tests_executed": True, **COUNTS, **{key: True for key in FLAGS},
        **{key: manifest[key] for key in ("source_root", "source_commit", "source_files", "source_build", "binaries", "runner", "probe_source", "oracle_source", "candidate_entry", "candidate_source_sha256")},
        "fixture_manifest": evidence(manifest_path), "compile_receipt": evidence(compile_path),
        "probe_binary": compiled["binary"], "raw_results": evidence(output), "stderr": evidence(errors),
        "argv": command, "library_directories": libraries, "explicit_driver_library": driver, "device": device, "allocation_summary": summary,
        "performance_measured": False, "performance_claim_eligible": False,
        "desktop_session_policy": "left untouched; correctness-only execution"}
    write_new(manifest_path.parent / "receipt.json", result)
    return result


def validate_receipt(path, build_path):
    """Read-only qualifier hook: recheck all nested source/artifact evidence."""
    path, build_path = Path(path).resolve(strict=True), Path(build_path).resolve(strict=True)
    result = read(path)
    require(result["schema_version"] == "riley.batch8-fusion-correctness.v1" and result["passed"] is True and result["gpu_tests_executed"] is True, "attention probe did not pass on GPU")
    require(all(result[key] == value for key, value in COUNTS.items()) and all(result[key] is True for key in FLAGS), "attention coverage or correctness flags differ")
    require(result["performance_measured"] is False and result["performance_claim_eligible"] is False, "attention receipt claims performance")
    for key in ("explicit_driver_library", "source_build", "runner", "probe_source", "oracle_source", "fixture_manifest", "compile_receipt", "probe_binary", "raw_results", "stderr"):
        check_evidence(result[key])
    require(result["source_build"] == evidence(build_path), "attention probe used another source build")
    manifest_path = Path(result["fixture_manifest"]["path"])
    manifest = validate_manifest(manifest_path)
    compiled = validate_compile(result["compile_receipt"]["path"], manifest_path)
    device, summary = validate_raw(result["raw_results"]["path"], manifest)
    require(all(result[key] == manifest[key] for key in ("source_root", "source_commit", "source_build", "source_files", "binaries", "runner", "probe_source", "oracle_source", "candidate_entry", "candidate_source_sha256")), "attention result binding differs from fixtures")
    require(result["probe_binary"] == compiled["binary"] and result["device"] == device and result["allocation_summary"] == summary, "attention runtime/binary evidence differs")
    require(result["explicit_driver_library"]["sha256"] == DRIVER_SHA, "private driver hash differs")
    require(len(result["library_directories"]) == 2 and result["library_directories"][1] == compiled["runtime_library_directory"] and evidence(Path(result["library_directories"][0])/"libcuda.so.1") == result["explicit_driver_library"], "private runtime selection differs")
    require(result["argv"] == [compiled["binary"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"], "attention invocation differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    for name in ("source-root", "build-receipt", "oracle-source", "output-dir"):
        create.add_argument("--"+name, type=Path, required=True)
    for name in ("candidate-entry", "candidate-source-sha256"):
        create.add_argument("--"+name, required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--nvcc", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--driver-library-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.build_receipt, args.oracle_source, args.candidate_entry, args.candidate_source_sha256, args.output_dir)
    elif args.command == "build":
        result = build_probe(args.manifest, args.nvcc)
    else:
        result = run_probe(args.manifest, args.driver_library_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
