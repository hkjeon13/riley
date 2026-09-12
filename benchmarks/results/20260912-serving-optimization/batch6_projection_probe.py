#!/usr/bin/env python3
"""Prepare/build/run a correctness-only batch6 projection probe, in separate steps.

prepare --source-root .../batch6-source --build-receipt .../batch6-build.json
        --binding .../native-binding.json --model .../model.safetensors
        --output-dir .../batch6-projection-probe
build   --manifest .../batch6-projection-probe/fixtures.json --nvcc .../bin/nvcc
run     --manifest .../batch6-projection-probe/fixtures.json

All outputs are exclusive. Only `run` launches GPU work. No numerical conversion
of checkpoint weights, performance timing, Blender/session change, or production
source edit is performed. Python dependencies are standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

PRECISE = "kernels/src/graph_numerics_precise.cu"
PRECISE_SHA256 = "fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f"
PROJECTIONS = (
    ("q", "self_attn.q_proj", 576, 576, 192),
    ("k", "self_attn.k_proj", 192, 576, 192),
    ("v", "self_attn.v_proj", 192, 576, 192),
    ("o", "self_attn.o_proj", 576, 576, 128),
    ("gate", "mlp.gate_proj", 1536, 576, 0),
    ("up", "mlp.up_proj", 1536, 576, 0),
    ("down", "mlp.down_proj", 576, 1536, 320),
)
PATTERNS = ("row_fingerprint", "signed_zero_impulses", "bounded_cancellation")
COUNTS = {"cases": 630, "layers": 30, "projections": 7, "patterns": 3, "rows": 128}
FLAGS = ("exact_outputs", "finite_outputs", "guards_intact", "inputs_unchanged",
         "weights_unchanged", "position_unchanged", "all_allocations_freed")
DEPENDENCIES = (PRECISE, "kernels/src/ffi_internal.hpp", "kernels/include/riley_cuda.h",
                "kernels/CMakeLists.txt", "crates/riley-cuda/build.rs")


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
    require(sha(source / PRECISE) == PRECISE_SHA256, "unknown batch6 precise source")
    parent = subprocess.check_output(["git", "-C", str(source), "show", "HEAD^:" + PRECISE])
    require((source / PRECISE).read_bytes().startswith(parent), "original rows128 oracle source was modified")
    return build


def pattern_bytes(name, depth):
    require(name in PATTERNS and depth in (576, 1536), "unsupported input fixture")
    words = []
    for row in range(128):
        pivot = (row*13+7) % depth
        for column in range(depth):
            sign = ((row+column) & 1) << 15
            if name == "row_fingerprint":
                bits = sign | (0x3e00 + ((row*17+column*29) & 127))
            elif name == "signed_zero_impulses":
                bits = sign  # Preserve both +0 and -0 byte patterns.
                if column == pivot:
                    bits = 0x3e80 | row
                elif column == (pivot+17) % depth:
                    bits = 0xbe80 | row
            else:
                # Equal-magnitude adjacent +/- values, bounded below 1.0.
                # Row-dependent mantissas distinguish all 16 MMA fragment rows.
                bits = sign | (0x3f00 + ((row*19+(column//2)*11) & 127))
            words.append(bits)
    return struct.pack("<" + "H"*len(words), *words)


def tensor_header(path):
    with Path(path).open("rb") as stream:
        size_bytes = stream.read(8)
        require(len(size_bytes) == 8, "truncated safetensors prefix")
        length = struct.unpack("<Q", size_bytes)[0]
        require(0 < length <= 16*1024*1024, "safetensors header length is out of bounds")
        raw = stream.read(length)
        require(len(raw) == length, "truncated safetensors header")
    header = json.loads(raw, object_pairs_hook=pairs_unique)
    require(isinstance(header, dict), "safetensors header is not an object")
    return header, length+8


def prepare(source_root, build_path, binding_path, model_path, output):
    source_root, build_path, binding_path = (Path(path).resolve(strict=True) for path in (source_root, build_path, binding_path))
    model_path = Path(model_path).resolve(strict=True)
    if model_path.is_dir():
        model_path /= "model.safetensors"
    output = Path(output).absolute()
    build = check_snapshot(build_path)
    require(build["source_root"] == str(source_root), "explicit source differs from the frozen build")
    binding = read(binding_path)
    require(binding["source"]["correctness_gate_id"] == "g04-vllm-smol-p128-v1", "unexpected numerical reference gate")
    model = evidence(model_path)
    require(model["sha256"] == binding["workload"]["weights_sha256"], "model SHA differs from pinned checkpoint")
    header, data_start = tensor_header(model_path)
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    fixtures = output / "fixtures"
    fixtures.mkdir()
    files, inputs, weights = {}, {}, {}
    for depth in (576, 1536):
        for pattern in PATTERNS:
            path = fixtures / f"x-{pattern}-{depth}.bf16"
            with path.open("xb") as stream:
                stream.write(pattern_bytes(pattern, depth))
            files[str(path)] = sha(path)
            inputs[pattern, depth] = str(path)
    occupied = []
    with model_path.open("rb") as checkpoint:
        for layer in range(30):
            for label, suffix, n, k, interval in PROJECTIONS:
                name = f"model.layers.{layer}.{suffix}.weight"
                tensor = header.get(name)
                require(isinstance(tensor, dict) and tensor.get("dtype") == "BF16" and tensor.get("shape") == [n, k], "checkpoint tensor dtype/shape differs: " + name)
                offsets = tensor.get("data_offsets")
                require(isinstance(offsets, list) and len(offsets) == 2 and all(type(value) is int for value in offsets), "invalid tensor offsets")
                begin, end = offsets
                require(0 <= begin < end and end-begin == n*k*2 and data_start+end <= model_path.stat().st_size, "tensor range out of bounds: " + name)
                require(all(end <= left or begin >= right for left, right in occupied), "overlapping checkpoint projection tensors")
                occupied.append((begin, end))
                checkpoint.seek(data_start+begin)
                raw = checkpoint.read(end-begin)
                require(len(raw) == end-begin, "short checkpoint read")
                path = fixtures / f"w-{layer:02d}-{label}.bf16"
                with path.open("xb") as stream:
                    stream.write(raw)  # Raw BF16 bytes; no dtype conversion.
                files[str(path)] = sha(path)
                weights[layer, label] = {"path": str(path), "tensor": name, "data_offsets": offsets,
                                         "file_offset": data_start+begin, "bytes": len(raw), "sha256": files[str(path)]}
    cases = []
    for layer in range(30):
        for projection, (label, _, n, k, interval) in enumerate(PROJECTIONS):
            for pattern, name in enumerate(PATTERNS):
                cases.append({"case_id": len(cases), "layer": layer, "projection": label,
                    "projection_index": projection, "pattern": name, "pattern_index": pattern,
                    "n": n, "k": k, "interval": interval, "rows": 128,
                    "input": inputs[name, k], "weight": weights[layer, label]})
    index = output / "cases.tsv"
    with index.open("x") as stream:
        for row in cases:
            values = [row["case_id"], row["layer"], row["projection_index"], row["pattern_index"], row["n"], row["k"], row["interval"], row["input"], row["weight"]["path"]]
            require(all("\t" not in str(value) and "\n" not in str(value) for value in values), "fixture path cannot contain TSV control characters")
            stream.write("\t".join(map(str, values)) + "\n")
    manifest = {"schema_version": "riley.batch6-projection-fixtures.v1", **COUNTS,
        "source_root": str(source_root), "source_commit": build["source_commit"],
        "source_build": evidence(build_path), "source_files": build["source_files"], "binaries": build["binaries"],
        "source_dependencies": {name: sha(source_root/name) for name in DEPENDENCIES},
        "runner": evidence(Path(__file__)), "probe_source": evidence(Path(__file__).with_suffix(".cu")),
        "model": model, "reference_binding": evidence(binding_path), "safetensors_data_start": data_start,
        "fixture_files": files, "case_index": evidence(index), "case_records": cases,
        "weights_converted": False, "position": 127, "guard_bytes_each_side": 256,
        "input_patterns": list(PATTERNS), "gpu_tests_executed": False,
        "performance_measured": False, "performance_claim_eligible": False}
    require(check_snapshot(build_path) == build and evidence(model_path) == model, "identity changed while preparing fixtures")
    write_new(output / "fixtures.json", manifest)
    return {"fixture_manifest": evidence(output / "fixtures.json"), **COUNTS, "gpu_tests_executed": False}


def validate_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = read(path)
    require(manifest["schema_version"] == "riley.batch6-projection-fixtures.v1", "unknown fixture schema")
    require(all(manifest[key] == value for key, value in COUNTS.items()), "incomplete fixture coverage")
    for name in ("source_build", "runner", "probe_source", "model", "reference_binding", "case_index"):
        check_evidence(manifest[name])
    require(manifest["runner"] == evidence(Path(__file__)) and manifest["probe_source"] == evidence(Path(__file__).with_suffix(".cu")), "probe helpers differ")
    build = check_snapshot(manifest["source_build"]["path"])
    require(all(manifest[key] == build[key] for key in ("source_root", "source_commit", "source_files", "binaries")), "fixture source/build binding differs")
    for name, digest in manifest["source_dependencies"].items():
        require(name in DEPENDENCIES and sha(Path(manifest["source_root"])/name) == digest, "probe source dependency changed")
    require(set(manifest["source_dependencies"]) == set(DEPENDENCIES), "probe dependencies missing")
    require(len(manifest["fixture_files"]) == 216 and len(manifest["case_records"]) == 630, "fixture file/case count differs")
    for name, digest in manifest["fixture_files"].items():
        require(Path(name).resolve().is_relative_to(path.parent / "fixtures") and sha(name) == digest, "fixture changed or escaped output")
    require(manifest["input_patterns"] == list(PATTERNS), "input patterns differ")
    for depth in (576, 1536):
        for pattern in PATTERNS:
            fixture = path.parent / "fixtures" / f"x-{pattern}-{depth}.bf16"
            require(fixture.read_bytes() == pattern_bytes(pattern, depth), "row-dependent input fixture differs")
    binding = read(manifest["reference_binding"]["path"])
    require(binding["workload"]["weights_sha256"] == manifest["model"]["sha256"], "checkpoint differs from reference binding")
    header, data_start = tensor_header(manifest["model"]["path"])
    require(data_start == manifest["safetensors_data_start"], "safetensors data base differs")
    tsv, checked_weights = [], set()
    with Path(manifest["model"]["path"]).open("rb") as model:
        for index, case in enumerate(manifest["case_records"]):
            layer, projection, pattern = index//21, (index//3)%7, index%3
            label, suffix, n, k, interval = PROJECTIONS[projection]
            expected = {"case_id": index, "layer": layer, "projection_index": projection,
                        "projection": label, "pattern_index": pattern, "pattern": PATTERNS[pattern],
                        "rows": 128, "n": n, "k": k, "interval": interval}
            require(all(case[key] == value for key, value in expected.items()), "fixture case coverage/order differs")
            input_path = path.parent/"fixtures"/f"x-{PATTERNS[pattern]}-{k}.bf16"
            weight_path = path.parent/"fixtures"/f"w-{layer:02d}-{label}.bf16"
            weight = case["weight"]
            tensor_name = f"model.layers.{layer}.{suffix}.weight"
            tensor = header[tensor_name]
            require(tensor["dtype"] == "BF16" and tensor["shape"] == [n, k], "checkpoint tensor contract differs")
            require(case["input"] == str(input_path) and weight["path"] == str(weight_path)
                    and weight["tensor"] == tensor_name and weight["data_offsets"] == tensor["data_offsets"]
                    and weight["file_offset"] == data_start+tensor["data_offsets"][0]
                    and weight["bytes"] == n*k*2 and tensor["data_offsets"][1]-tensor["data_offsets"][0] == n*k*2
                    and weight["sha256"] == manifest["fixture_files"][str(weight_path)], "raw weight fixture binding differs")
            if tensor_name not in checked_weights:
                model.seek(weight["file_offset"])
                require(hashlib.sha256(model.read(weight["bytes"])).hexdigest() == weight["sha256"], "weight fixture is not an exact checkpoint byte slice")
                checked_weights.add(tensor_name)
            tsv.append("\t".join(map(str, [index, layer, projection, pattern, n, k, interval, input_path, weight_path])))
    require(Path(manifest["case_index"]["path"]).read_text() == "\n".join(tsv)+"\n", "native case index differs from pinned fixtures")
    require(manifest["weights_converted"] is False and manifest["position"] == 127 and manifest["guard_bytes_each_side"] == 256, "fixture numeric or guard contract differs")
    require(manifest["gpu_tests_executed"] is False and manifest["performance_measured"] is False and manifest["performance_claim_eligible"] is False, "fixture claims execution")
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
        require(not any("fast_math" in flag or flag.startswith(("--fmad", "--ftz", "--prec-div", "--prec-sqrt")) for flag in flags), "precise source unexpectedly overrides default CUDA arithmetic flags")
        compile_lines = [line.strip() for line in make_path.read_text().splitlines()
                         if "$(CUDA_FLAGS)" in line and " -c " in line and "graph_numerics_precise.cu" in line]
        require(len(compile_lines) == 1 and "$(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS)" in compile_lines[0], "unknown precise-source CMake compile rule")
        tokens = shlex.split(compile_lines[0])
        require(Path(tokens[0]).resolve() == nvcc and tokens[1:5] == ["-forward-unknown-to-host-compiler", "$(CUDA_DEFINES)", "$(CUDA_INCLUDES)", "$(CUDA_FLAGS)"], "unexpected CMake compiler prefix")
        relative_object = "CMakeFiles/riley_cuda_native.dir/src/graph_numerics_precise.cu.o"
        require(tokens[5:] == ["-MD", "-MT", relative_object, "-MF", relative_object+".d", "-x", "cu", "-c", str(Path(manifest["source_root"])/PRECISE), "-o", relative_object], "unexpected per-source precise compile options")
        candidates.append({"flags": flags, "cmake_artifacts": artifacts, "production_compile_rule": compile_lines[0],
                           "original_compile_cwd": str(cache.parent), "response_files_expanded": True,
                           "omitted_dependency_only_flags": ["-MD", "-MT", "-MF"], "runtime_library_link": str(runtime)})
    require(candidates, "no matching frozen production CMake compile flags found")
    signatures = {tuple(item["flags"]) for item in candidates}
    require(len(signatures) == 1, "ambiguous production CUDA flag variants")
    return candidates[0]


def compile_commands(manifest, nvcc, options, directory, runtime):
    kernel_object, host_object, binary = (directory/name for name in ("precise.o", "probe.o", "batch6_projection_probe"))
    arch = [flag for flag in options["flags"] if "arch=compute_89" in flag]
    return [
        [str(nvcc), *options["flags"], "-x", "cu", "-c", str(Path(manifest["source_root"])/PRECISE), "-o", str(kernel_object)],
        [str(nvcc), "-std=c++17", "-O2", *arch, "-c", manifest["probe_source"]["path"], "-o", str(host_object)],
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
    kernel_object, host_object, binary = (directory/name for name in ("precise.o", "probe.o", "batch6_projection_probe"))
    runtime = Path(options["runtime_library_link"])
    commands = compile_commands(manifest, nvcc, options, directory, runtime)
    log = directory / "compile.log"
    with log.open("x") as stream:
        for command in commands:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=stream, check=True)
    require(validate_manifest(manifest_path) == manifest, "identity changed during compile")
    for artifact in options["cmake_artifacts"]:
        check_evidence(artifact)
    receipt = {"schema_version": "riley.batch6-projection-build.v1", "fixture_manifest": evidence(manifest_path),
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
    require(compiled["schema_version"] == "riley.batch6-projection-build.v1" and compiled["fixture_manifest"] == evidence(manifest_path), "compile receipt differs from fixtures")
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
    require(rows and all(row.get("schema") == "riley.batch6-projection-native.v1" for row in rows), "unknown native probe output")
    require(len(rows) == 632 and rows[0]["kind"] == "device" and rows[-1]["kind"] == "summary", "native probe lacks complete device/case/summary records")
    require(rows[0]["compute_major"] == 8 and rows[0]["compute_minor"] == 9 and rows[0]["ordinal"] == 0, "native device differs")
    require(rows[0]["runtime_version"] == 13000, "fixed profile requires CUDA runtime13.0")
    for row, case in zip(rows[1:-1], manifest["case_records"]):
        require(row["kind"] == "case" and row["passed"] is True, "projection case failed")
        for key in ("case_id", "layer", "projection", "pattern", "rows", "n", "k", "interval"):
            require(row[key] == case[key], "native case does not match fixture: " + key)
        require(row["output_bytes_compared"] == 128*case["n"]*2, "incomplete output byte comparison")
        require(all(row[key] is True for key in (*FLAGS, "oracle_output_unchanged")), "projection equality/finite/guard/immutability/lifetime check failed")
        require("first_mismatch" not in row, "unexpected numeric mismatch")
    summary = rows[-1]
    require(summary["passed"] is True and all(summary[key] == value for key, value in COUNTS.items()), "native probe coverage or success differs")
    require(summary["failed_cases"] == 0 and summary["device_allocations_created"] == summary["device_allocations_freed"] == 3150,
            "native allocation count differs")
    require(summary["live_device_allocations"] == summary["live_device_bytes"] == summary["cleanup_errors"] == 0
            and summary["all_allocations_freed"] is True and summary["stream_destroyed"] is True, "native cleanup did not complete")
    require(summary["performance_measured"] is False and summary["performance_claim_eligible"] is False and summary["error"] == "", "native summary reports an error or performance claim")
    return rows[0], summary


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
    require(process.returncode == 0, "native projection probe failed; inspect " + str(output))
    require(validate_manifest(manifest_path) == manifest and validate_compile(compile_path, manifest_path) == compiled, "identity changed during GPU probe")
    device, summary = validate_raw(output, manifest)
    result = {"schema_version": "riley.batch6-projection-correctness.v1", "passed": True,
        "gpu_tests_executed": True, **COUNTS, **{key: True for key in FLAGS},
        **{key: manifest[key] for key in ("source_root", "source_commit", "source_files", "source_build", "binaries", "runner", "probe_source", "model")},
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
    require(result["schema_version"] == "riley.batch6-projection-correctness.v1" and result["passed"] is True and result["gpu_tests_executed"] is True, "projection probe did not pass on GPU")
    require(all(result[key] == value for key, value in COUNTS.items()) and all(result[key] is True for key in FLAGS), "projection coverage or correctness flags differ")
    require(result["performance_measured"] is False and result["performance_claim_eligible"] is False, "projection receipt claims performance")
    for key in ("source_build", "runner", "probe_source", "model", "fixture_manifest", "compile_receipt", "probe_binary", "raw_results", "stderr"):
        check_evidence(result[key])
    require(result["source_build"] == evidence(build_path), "projection probe used another source build")
    manifest_path = Path(result["fixture_manifest"]["path"])
    manifest = validate_manifest(manifest_path)
    compiled = validate_compile(result["compile_receipt"]["path"], manifest_path)
    device, summary = validate_raw(result["raw_results"]["path"], manifest)
    require(all(result[key] == manifest[key] for key in ("source_root", "source_commit", "source_build", "source_files", "binaries", "runner", "probe_source", "model")), "projection result binding differs from fixtures")
    require(result["probe_binary"] == compiled["binary"] and result["device"] == device and result["allocation_summary"] == summary, "projection runtime/binary evidence differs")
    require(result["argv"] == [compiled["binary"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"], "projection invocation differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    for name in ("source-root", "build-receipt", "binding", "model", "output-dir"):
        create.add_argument("--"+name, type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--nvcc", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.source_root, args.build_receipt, args.binding, args.model, args.output_dir)
        elif args.command == "build":
            result = build_probe(args.manifest, args.nvcc)
        else:
            result = run_probe(args.manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
