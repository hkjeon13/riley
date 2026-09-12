#!/usr/bin/env python3
"""Correctness-only synthetic attention probe: prepare, build, then run.

prepare --source-root SNAPSHOT --build-receipt BUILD.json
        --oracle-source ACCEPTED/graph_numerics.cu
        --candidate-entry enqueue_compiled_packed_decode_attention_two_warp
        --candidate-source-sha256 SHA256 --output-dir NEW_DIRECTORY
build --manifest NEW_DIRECTORY/fixtures.json --nvcc CUDA13/bin/nvcc
run --manifest NEW_DIRECTORY/fixtures.json

Only run executes GPU work. Inputs are deterministic synthetic BF16 values,
not model activations; layer numbers are independent synthetic seeds. No timing,
performance claim, source modification, or desktop/session control is included.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE = "kernels/src/graph_numerics.cu"
ORACLE_SHA256 = "d86de0d147fab0e7083095bd0b9515daf73cfe87465dde9c5ead2b6fda744dbf"
PATTERNS = ("bounded_fingerprint", "signed_zero_impulses", "tail_cancellation")
MAPPINGS = ("identity", "reverse", "affine_7_3")
COUNTS = {"cases": 8640, "layers": 30, "positions": 32, "patterns": 3, "mappings": 3}
FLAGS = ("exact_outputs", "finite_outputs", "guards_intact", "inputs_unchanged",
         "oracle_outputs_unchanged", "mapping_invariant", "all_allocations_freed")
DEPENDENCIES = (SOURCE, "kernels/src/ffi_internal.hpp", "kernels/include/riley_cuda.h",
                "kernels/CMakeLists.txt", "crates/riley-cuda/build.rs")
RECIPE = {
    "version": "attention-integer-bf16-v1",
    "input_kind": "synthetic_only_no_checkpoint_activations",
    "generator": "pinned native probe fixture()/sample()",
    "q_bytes": 1152, "kv_bytes_each": 98304, "metadata_bytes": 116,
    "physical_blocks": 16, "logical_tokens": 160, "query_heads": 9,
    "kv_heads": 3, "head_dim": 64, "guard_bytes_each_side": 256,
    "positions_inclusive": [128, 159], "patterns": list(PATTERNS), "mappings": list(MAPPINGS),
    "metadata_offsets": {"position": 4, "live_blocks": 12, "block_ids": 16, "valid_counts": 80, "sequence_slot": 112},
    "comparison": "all 576 BF16 words including signed zeros, both oracles and candidate, plus mapping invariance",
    "unused_and_future_slots": "initialized finite and immutable; all slots checked",
}

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
    require(re.fullmatch(r"enqueue_compiled_[A-Za-z0-9_]+", entry), "invalid candidate wrapper identifier")
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
    manifest = {"schema_version": "riley.batch7-attention-fixtures.v1", **COUNTS,
        "source_root": str(source), "source_commit": build["source_commit"],
        "source_build": evidence(build_path), "source_files": build["source_files"], "binaries": build["binaries"],
        "source_dependencies": {name: evidence(source/name) for name in DEPENDENCIES},
        "oracle_source": evidence(oracle), "candidate_entry": candidate_entry,
        "candidate_source_sha256": candidate_source_sha256,
        "runner": evidence(__file__), "probe_source": evidence(Path(__file__).with_suffix(".cu")),
        "case_index": evidence(index), "fixture_recipe": RECIPE,
        "gpu_tests_executed": False, "performance_measured": False, "performance_claim_eligible": False}
    write_new(output/"fixtures.json", manifest)
    validate_manifest(output/"fixtures.json")
    return {"fixture_manifest": evidence(output/"fixtures.json"), **COUNTS, "gpu_tests_executed": False}


def validate_manifest(path):
    manifest = read(path)
    require(manifest["schema_version"] == "riley.batch7-attention-fixtures.v1", "unknown fixture schema")
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
    """Read actual CMake compile variables; never approximate numerical flags."""
    build = read(manifest["source_build"]["path"])
    target = Path(build["build_environment"]["CARGO_TARGET_DIR"])
    candidates = []
    for cache in sorted(target.glob("release/build/riley-cuda-*/out/cuda-native-build/CMakeCache.txt")):
        # CMake separates entries with blank lines and // documentation. Match
        # each line independently so a key can never absorb either separator.
        entries = {}
        for line in cache.read_text().splitlines():
            match = re.fullmatch(r"([^#/:][^:=]*):[^=]+=(.*)", line)
            if match:
                entries[match[1]] = match[2]
        if Path(entries.get("CMAKE_HOME_DIRECTORY", "/absent")).resolve() != Path(manifest["source_root"]) / "kernels":
            continue
        if Path(entries.get("CMAKE_CUDA_COMPILER", "/absent")).resolve() != nvcc:
            continue
        require(entries.get("CMAKE_BUILD_TYPE") == "Release", "native build was not Release")
        directory = cache.parent / "CMakeFiles/riley_cuda_native.dir"
        flags_path, make_path = directory / "flags.make", directory / "build.make"
        variables = dict(re.findall(r"^(CUDA_[A-Z]+) = (.*)$", flags_path.read_text(), re.MULTILINE))
        require(set(("CUDA_DEFINES", "CUDA_INCLUDES", "CUDA_FLAGS")).issubset(variables), "CMake CUDA flags are incomplete")
        runtime_path = cache.parent / "riley-cuda-cudart-Release.path"
        runtime = Path(runtime_path.read_text().strip())
        require(runtime.is_absolute() and runtime.is_file(), "CMake shared CUDA runtime path is invalid")
        artifacts = [evidence(cache), evidence(flags_path), evidence(make_path), evidence(runtime_path)]
        includes, expanded = shlex.split(variables["CUDA_INCLUDES"]), []
        while includes:
            flag = includes.pop(0)
            if flag == "--options-file":
                require(includes, "missing CMake include response file")
                response = (cache.parent / includes.pop(0)).resolve(strict=True)
                require(response.is_relative_to(cache.parent.resolve()), "CMake response file escapes build directory")
                artifacts.append(evidence(response))
                contents = shlex.split(response.read_text())
                require("--options-file" not in contents and not any(item.startswith("@") for item in contents), "nested response files are unsupported")
                expanded.extend(contents)
            else:
                expanded.append(flag)
        flags = ["-forward-unknown-to-host-compiler", *shlex.split(variables["CUDA_DEFINES"]), *expanded, *shlex.split(variables["CUDA_FLAGS"])]
        require(all(flag in flags for flag in ("-O3", "-DNDEBUG", "-std=c++17", "--objdir-as-tempdir", "-Xcompiler=-fno-exceptions")), "unexpected production CUDA options")
        require(any("arch=compute_89" in flag and "sm_89" in flag for flag in flags), "production source lacks SM89 code generation")
        require(not any("fast_math" in flag or flag.startswith(("--fmad", "--ftz", "--prec-div", "--prec-sqrt")) for flag in flags), "attention source unexpectedly overrides default CUDA arithmetic flags")
        compile_lines = [line.strip() for line in make_path.read_text().splitlines()
                         if "$(CUDA_FLAGS)" in line and " -c " in line and "graph_numerics.cu" in line]
        require(len(compile_lines) == 1 and "$(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS)" in compile_lines[0], "unknown attention-source CMake compile rule")
        tokens = shlex.split(compile_lines[0])
        require(Path(tokens[0]).resolve() == nvcc and tokens[1:5] == ["-forward-unknown-to-host-compiler", "$(CUDA_DEFINES)", "$(CUDA_INCLUDES)", "$(CUDA_FLAGS)"], "unexpected CMake compiler prefix")
        relative_object = "CMakeFiles/riley_cuda_native.dir/src/graph_numerics.cu.o"
        require(tokens[5:] == ["--use_fast_math", "-MD", "-MT", relative_object, "-MF", relative_object+".d", "-x", "cu", "-c", str(Path(manifest["source_root"])/SOURCE), "-o", relative_object], "unexpected per-source attention compile options")
        flags.append("--use_fast_math")
        candidates.append({"flags": flags, "cmake_artifacts": artifacts, "production_compile_rule": compile_lines[0],
                           "original_compile_cwd": str(cache.parent), "response_files_expanded": True,
                           "omitted_dependency_only_flags": ["-MD", "-MT", "-MF"], "runtime_library_link": str(runtime)})
    require(candidates, "no matching frozen production CMake compile flags found")
    signatures = {tuple(item["flags"]) for item in candidates}
    require(len(signatures) == 1, "ambiguous production CUDA flag variants")
    return candidates[0]


def compile_commands(manifest, nvcc, options, directory, runtime):
    kernel_object, host_object, binary = (directory/name for name in ("attention.o", "probe.o", "batch7_attention_probe"))
    arch = [flag for flag in options["flags"] if "arch=compute_89" in flag]
    return [
        [str(nvcc), *options["flags"], "-x", "cu", "-c", str(Path(manifest["source_root"])/SOURCE), "-o", str(kernel_object)],
        [str(nvcc), "-std=c++17", "-O2", *arch, "-DRILEY_ATTENTION_CANDIDATE="+manifest["candidate_entry"], "-c", manifest["probe_source"]["path"], "-o", str(host_object)],
        [str(nvcc), "--cudart=shared", str(kernel_object), str(host_object), "-L"+str(runtime.parent), "-Xlinker", "-rpath", "-Xlinker", str(runtime.parent), "-o", str(binary)],
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
    kernel_object, host_object, binary = (directory/name for name in ("attention.o", "probe.o", "batch7_attention_probe"))
    runtime = Path(options["runtime_library_link"])
    commands = compile_commands(manifest, nvcc, options, directory, runtime)
    log = directory / "compile.log"
    with log.open("x") as stream:
        for command in commands:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=stream, check=True)
    require(validate_manifest(manifest_path) == manifest, "identity changed during compile")
    for artifact in options["cmake_artifacts"]:
        check_evidence(artifact)
    receipt = {"schema_version": "riley.batch7-attention-build.v1", "fixture_manifest": evidence(manifest_path),
        "source_commit": manifest["source_commit"], "compiler": evidence(nvcc), "compiler_version": version,
        "runtime_library": evidence(runtime), "runtime_library_directory": str(runtime.parent),
        "runtime_library_link": str(runtime.absolute()),
        "production_flags": options, "commands": commands, "binary": evidence(binary),
        "kernel_object": evidence(kernel_object), "host_object": evidence(host_object), "log": evidence(log),
        "gpu_tests_executed": False, "performance_measured": False, "performance_claim_eligible": False}
    write_new(manifest_path.parent / "compile.json", receipt)
    return receipt


def validate_compile(path, manifest_path):
    compiled = read(path)
    require(compiled["schema_version"] == "riley.batch7-attention-build.v1" and compiled["fixture_manifest"] == evidence(manifest_path), "compile receipt differs from fixtures")
    for key in ("compiler", "runtime_library", "binary", "kernel_object", "host_object", "log"):
        check_evidence(compiled[key])
    for artifact in compiled["production_flags"]["cmake_artifacts"]:
        check_evidence(artifact)
    manifest = read(manifest_path)
    require(compiled["source_commit"] == manifest["source_commit"], "compiled source identity differs")
    options = production_flags(manifest, Path(compiled["compiler"]["path"]))
    require(options == compiled["production_flags"], "compiled production flags differ")
    runtime = Path(compiled["runtime_library_link"])
    require(str(runtime) == options["runtime_library_link"] and evidence(runtime) == compiled["runtime_library"] and str(runtime.parent) == compiled["runtime_library_directory"], "shared CUDA runtime selection differs")
    require(compiled["commands"] == compile_commands(manifest, Path(compiled["compiler"]["path"]), options, Path(path).parent/"build", runtime), "standalone compile commands differ")
    require(compiled["gpu_tests_executed"] is False and compiled["performance_measured"] is False and compiled["performance_claim_eligible"] is False, "compile receipt claims GPU execution")
    return compiled


def validate_raw(path, manifest):
    rows = [json.loads(line, object_pairs_hook=pairs_unique) for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(rows) == COUNTS["cases"]+2 and all(row.get("schema") == "riley.batch7-attention-native.v1" for row in rows), "incomplete or unknown native output")
    device, summary = rows[0], rows[-1]
    require(device["kind"] == "device" and device["compute_major"] == 8 and device["compute_minor"] == 9 and device["ordinal"] == 0 and device["runtime_version"] == 13000, "native device/runtime differs")
    for row, case in zip(rows[1:-1], cases()):
        require(row["kind"] == "case" and row["passed"] is True and all(row[key] == value for key, value in case.items()), "attention case failed or coverage differs")
        require(row["output_bytes_compared_per_pair"] == 1152 and row["comparison_pairs"] == 2, "incomplete output comparison")
        require(all(row[key] is True for key in FLAGS), "numeric/finite/guard/immutability/map/lifetime check failed")
        require(row["accepted_mismatch_words"] == row["candidate_mismatch_words"] == 0 and "first_mismatch" not in row, "attention numeric mismatch")
    require(summary["kind"] == "summary" and summary["passed"] is True and all(summary[key] == value for key, value in COUNTS.items()), "summary success/coverage differs")
    require(summary["device_allocations_created"] == summary["device_allocations_freed"] == 7*COUNTS["cases"], "allocation count differs")
    require(summary["failed_cases"] == summary["live_device_allocations"] == summary["live_device_bytes"] == summary["cleanup_errors"] == 0 and summary["all_allocations_freed"] is True and summary["stream_destroyed"] is True, "cleanup failed")
    require(summary["performance_measured"] is False and summary["performance_claim_eligible"] is False and summary["error"] == "", "native summary error/performance claim")
    return device, summary


def run_probe(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(manifest_path)
    compile_path = manifest_path.parent / "compile.json"
    compiled = validate_compile(compile_path, manifest_path)
    command = [compiled["binary"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"]
    output, errors = manifest_path.parent / "native-results.jsonl", manifest_path.parent / "native-stderr.log"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = compiled["runtime_library_directory"] + (":"+env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    with output.open("x") as stdout, errors.open("x") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
    require(process.returncode == 0, "native attention probe failed; inspect " + str(output))
    require(validate_manifest(manifest_path) == manifest and validate_compile(compile_path, manifest_path) == compiled, "identity changed during GPU probe")
    device, summary = validate_raw(output, manifest)
    result = {"schema_version": "riley.batch7-attention-correctness.v1", "passed": True,
        "gpu_tests_executed": True, **COUNTS, **{key: True for key in FLAGS},
        **{key: manifest[key] for key in ("source_root", "source_commit", "source_files", "source_build", "binaries", "runner", "probe_source", "oracle_source", "candidate_entry", "candidate_source_sha256")},
        "fixture_manifest": evidence(manifest_path), "compile_receipt": evidence(compile_path),
        "probe_binary": compiled["binary"], "raw_results": evidence(output), "stderr": evidence(errors),
        "argv": command, "device": device, "allocation_summary": summary,
        "performance_measured": False, "performance_claim_eligible": False,
        "desktop_session_policy": "left untouched; correctness-only execution"}
    write_new(manifest_path.parent / "receipt.json", result)
    return result


def validate_receipt(path, build_path):
    """Read-only qualifier hook: recheck all nested source/artifact evidence."""
    path, build_path = Path(path).resolve(strict=True), Path(build_path).resolve(strict=True)
    result = read(path)
    require(result["schema_version"] == "riley.batch7-attention-correctness.v1" and result["passed"] is True and result["gpu_tests_executed"] is True, "attention probe did not pass on GPU")
    require(all(result[key] == value for key, value in COUNTS.items()) and all(result[key] is True for key in FLAGS), "attention coverage or correctness flags differ")
    require(result["performance_measured"] is False and result["performance_claim_eligible"] is False, "attention receipt claims performance")
    for key in ("source_build", "runner", "probe_source", "oracle_source", "fixture_manifest", "compile_receipt", "probe_binary", "raw_results", "stderr"):
        check_evidence(result[key])
    require(result["source_build"] == evidence(build_path), "attention probe used another source build")
    manifest_path = Path(result["fixture_manifest"]["path"])
    manifest = validate_manifest(manifest_path)
    compiled = validate_compile(result["compile_receipt"]["path"], manifest_path)
    device, summary = validate_raw(result["raw_results"]["path"], manifest)
    require(all(result[key] == manifest[key] for key in ("source_root", "source_commit", "source_build", "source_files", "binaries", "runner", "probe_source", "oracle_source", "candidate_entry", "candidate_source_sha256")), "attention result binding differs from fixtures")
    require(result["probe_binary"] == compiled["binary"] and result["device"] == device and result["allocation_summary"] == summary, "attention runtime/binary evidence differs")
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
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.build_receipt, args.oracle_source, args.candidate_entry, args.candidate_source_sha256, args.output_dir)
    elif args.command == "build":
        result = build_probe(args.manifest, args.nvcc)
    else:
        result = run_probe(args.manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
