#!/usr/bin/env python3
"""V2 mixed-FMA RoPE diagnosis; no performance qualification.

prepare is CPU-only. build and inspect invoke the CUDA toolchain, never a GPU.
run alone invokes the GPU executable. Nonmatching arithmetic is a completed
experiment (exit zero), reported through candidate_equal, never fusion_qualified.
The 436 input fixtures are bit-identical to v1; every mismatch bit pair is kept.
Order0 corrects both FMA expressions with scalar RN table conversion. Order1
uses those same expressions and changes only table conversion to packed RN.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
PRECISE = "kernels/src/graph_numerics_precise.cu"
ATTENTION = "kernels/src/graph_numerics.cu"
BUILD_SHA = "4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395"
COMMIT = "1a2be0df01fe49daa4d4db155ad5c44f34ead6df"
SOURCE_SHA = {PRECISE: "fbf8e8cd517abd29502d50c33a7d46fe7738bb7393ffbee7d9b0e532f305d87f",
              ATTENTION: "a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e"}
COUNT = 436
SCHEMA = "riley.rope-arithmetic"
V1_BASIS = {
    "receipt.json": "3d264d8eac757078aafb646f61a306940679b85697b997fc0792439e1fa86201",
    "run/native.jsonl": "4ac563a9166288b5b48ed8681c28ad8d7093f9cc0f52fba3fd3f2ad83cd40b5a",
    "inspection.json": "2243dd45e353f1a8c44702aae4621ac65f2692bf7b8dd843fbf4c87f454e8922",
    "inspect/oracle-rope.sass": "4ce1a5725a741475c69c599699059b3ae8dff2ef7c10e74c6421a7aabed8c433",
    "inspect/production-0-rope.sass": "4ce1a5725a741475c69c599699059b3ae8dff2ef7c10e74c6421a7aabed8c433",
    "inspect/production-1-rope.sass": "4ce1a5725a741475c69c599699059b3ae8dff2ef7c10e74c6421a7aabed8c433",
    "inspect/candidates-rope.sass": "48f6bf36c735739cc1dd92ef1ba88ace0c42c11280c3338f729d7f222aee60df",
    "inspect/oracle.ptx": "fc669c935a52a8a8c0ebf540b612e0975d15ba81e8cb5b1d2d2e0b8a475c250a",
    "inspect/candidates.ptx": "8fcf1ca75b7e2c5a43557401ab32db990d417c1d915c787848eed7d4d58ba81a",
}
# Integer IEEE754 words only: no host float operation can change fixture bits.
POSITIVE_TABLE = (0, 1, 0x7fff, 0x8000, 0x8001, 0xffff, 0x10000,
                  0x7f7fff, 0x7f8000, 0x7f8001, 0x7fffff, 0x800000,
                  0x3d007fff, 0x3d008000, 0x3d008001, 0x3d018000,
                  0x3e807fff, 0x3e808000, 0x3e808001, 0x3effffff,
                  0x3f000000, 0x3f007fff, 0x3f008000, 0x3f008001,
                  0x3f018000, 0x3f400000, 0x3f7f7fff, 0x3f7f8000,
                  0x3f7f8001, 0x3f7fffff, 0x3f800000)
TABLE = POSITIVE_TABLE + tuple(v | 0x80000000 for v in POSITIVE_TABLE)
FINITE = tuple(v for v in range(65536) if v & 0x7f80 != 0x7f80)
EDGES = (0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
         0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x3f7f, 0xbf7f, 0x7f7f, 0xff7f)
RECIPE = {"version": "rope-raw-integer-mixed-fma-v2", "cases": COUNT, "finite_bf16_encodings": 65280,
          "exhaustive_phases": 4, "cases_per_exhaustive_phase": 85, "edge_phases": 3,
          "positions": [128, 159], "qkv_bytes": 1920, "q_bytes": 1152,
          "kv_cache_bytes_each": 98304, "table_bytes_each": 20480, "metadata_bytes": 116,
          "guard_bytes_each_side": 256, "allocations_per_case": 13,
          "input_kind": "synthetic_raw_bits_not_checkpoint_activations_or_actual_trig_table",
          "fixture_bits_identical_to_v1": True,
          "first_output": "fma.rn(a,c,-mul.rn(b,s))",
          "second_output": "fma.rn(a,s,mul.rn(b,c))",
          "oracle": "full_frozen_precise_TU_enqueue_compiled_packed_decode_rope_kv",
          "candidates": ["mixed_fma_scalar_table_rn", "mixed_fma_packed_table_rn"],
          "comparison": "all Q and entire physical K/V caches as uint16; no tolerance",
          "performance_claim_eligible": False, "fusion_qualified": False}


def load_helper(name):
    spec = importlib.util.spec_from_file_location("rope_dependency_"+name, HERE/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention = load_helper("batch7_attention_probe")
precise = load_helper("batch6_projection_probe")
require, sha, evidence = attention.require, attention.sha, attention.evidence
read, write_new, check_evidence = attention.read, attention.write_new, attention.check_evidence


def cases():
    for index in range(COUNT):
        phase = index//85 if index < 340 else 4+(index-340)//32
        batch = index % 85 if index < 340 else (index-340) % 32
        yield index, phase, 128+batch % 32, batch


def qkv_bytes(index, phase, batch):
    if phase < 4:
        stride = (1, 7, 11, 13)[phase]
        values = [FINITE[(stride*(batch*768+i)+phase*9973) % len(FINITE)] for i in range(768)]
    else:
        # Pair opposite halves deterministically; phase4 cancellation, phase5
        # opposite signs, phase6 mixes exponent extremes with signed zeros.
        values = []
        for head in range(12):
            first = [EDGES[(batch*5+head*3+dim) % len(EDGES)] for dim in range(32)]
            if phase == 4:
                second = first
            elif phase == 5:
                second = [v ^ 0x8000 for v in first]
            else:
                second = [EDGES[(batch+head+dim*7+9) % len(EDGES)] for dim in range(32)]
            values.extend(first+second)
    values.extend((index*192+i) & 0xffff for i in range(192))
    return struct.pack("<960H", *values)


def table_bytes(phase, sine):
    values = [TABLE[(position*13+dim*7+phase*11+(19 if sine else 0)) % len(TABLE)]
              for position in range(160) for dim in range(32)]
    return struct.pack("<5120I", *values)


def metadata_bytes(index, position):
    mapping = index % 3
    physical = list(range(16)) if mapping == 0 else list(reversed(range(16))) if mapping == 1 else [(7*i+3)%16 for i in range(16)]
    live = position//16+1
    ids = physical[:live]+[0]*(16-live)
    valid = [16]*(live-1)+[position%16+1]+[0]*(16-live)
    return struct.pack("<4I16I16HI", 0, position, 0, live, *ids, *valid, 0)


def fixture_payloads(directory):
    directory = Path(directory)
    for phase in range(7):
        for name, sine in (("cos", False), ("sin", True)):
            yield directory/f"{name}-{phase}.bin", table_bytes(phase, sine)
    for index, phase, position, batch in cases():
        yield directory/f"qkv-{index:03d}.bin", qkv_bytes(index, phase, batch)
        yield directory/f"metadata-{index:03d}.bin", metadata_bytes(index, position)


def tsv(directory):
    directory = Path(directory)
    return "".join("\t".join(map(str, (index, phase, position,
        directory/f"qkv-{index:03d}.bin", directory/f"cos-{phase}.bin",
        directory/f"sin-{phase}.bin", directory/f"metadata-{index:03d}.bin")))+"\n"
        for index, phase, position, _ in cases())


def source_build(path):
    require(sha(path) == BUILD_SHA, "requires the frozen accepted batch7 build receipt")
    build = attention.check_snapshot(path)
    require(build["source_commit"] == COMMIT, "requires accepted batch7 source commit")
    for name, expected in SOURCE_SHA.items():
        require(sha(Path(build["source_root"])/name) == expected, "frozen source differs: "+name)
    return build


def tool_evidence():
    return {name: evidence(HERE/name) for name in (Path(__file__).name, "rope_arithmetic_probe_v2.cu",
        "rope_arithmetic_candidates_v2.cu", "batch7_attention_probe.py", "batch6_projection_probe.py")}


def basis_evidence():
    result = {}
    for name, expected in V1_BASIS.items():
        recorded = evidence(HERE/"rope-arithmetic-probe"/name)
        require(recorded["sha256"] == expected, "v1 numerical/SASS basis differs: "+name)
        result[name] = recorded
    return result


def prepare(build_path, output):
    build_path = Path(build_path).resolve(strict=True)
    build = source_build(build_path)
    basis = basis_evidence()
    output = Path(output).resolve()
    require(not any(c in str(output) for c in "\t\r\n"), "fixture directory contains TSV delimiters")
    output.mkdir(exist_ok=False)
    directory = output/"fixtures"
    directory.mkdir()
    files = []
    for path, payload in fixture_payloads(directory):
        with path.open("xb") as stream:
            stream.write(payload)
        files.append(evidence(path))
    index = output/"cases.tsv"
    with index.open("x") as stream:
        stream.write(tsv(directory))
    result = {"schema_version": SCHEMA+"-fixtures.v2", "source_build": evidence(build_path),
              "source_root": build["source_root"], "source_commit": COMMIT,
              "recipe": RECIPE, "tools": tool_evidence(), "basis": basis,
              "fixtures": files, "case_index": evidence(index)}
    write_new(output/"fixtures.json", result)
    validate_manifest(output/"fixtures.json")
    return result


def validate_manifest(path):
    path = Path(path).resolve(strict=True)
    manifest = read(path)
    require(manifest["schema_version"] == SCHEMA+"-fixtures.v2" and manifest["recipe"] == RECIPE, "fixture recipe differs")
    check_evidence(manifest["source_build"])
    build = source_build(manifest["source_build"]["path"])
    require(manifest["source_root"] == build["source_root"] and manifest["source_commit"] == COMMIT, "source identity differs")
    require(manifest["tools"] == tool_evidence(), "probe/helper identity changed")
    require(manifest["basis"] == basis_evidence(), "v1 source/disassembly basis changed")
    payloads = list(fixture_payloads(path.parent/"fixtures"))
    require(len(manifest["fixtures"]) == len(payloads), "fixture inventory length differs")
    for recorded, (expected_path, expected_bytes) in zip(manifest["fixtures"], payloads):
        require(recorded == evidence(expected_path), "fixture evidence differs")
        require(expected_path.read_bytes() == expected_bytes, "fixture raw bits differ from recipe")
    require(manifest["case_index"] == evidence(path.parent/"cases.tsv") and (path.parent/"cases.tsv").read_text() == tsv(path.parent/"fixtures"), "case index differs")
    return manifest


def options_for(manifest, nvcc):
    options = {"oracle_precise": precise.production_flags(manifest, nvcc), "candidates_fast_math": attention.production_flags(manifest, nvcc)}
    require(options["candidates_fast_math"]["flags"] == options["oracle_precise"]["flags"]+["--use_fast_math"], "unexpected precise/attention flag difference")
    require(options["oracle_precise"]["runtime_library_link"] == options["candidates_fast_math"]["runtime_library_link"], "runtime link differs between source rules")
    return options


def compile_commands(manifest, nvcc, options, directory):
    directory = Path(directory)
    precise_flags, fast_flags = (options[key]["flags"] for key in ("oracle_precise", "candidates_fast_math"))
    runtime = Path(options["oracle_precise"]["runtime_library_link"])
    arch = [flag for flag in precise_flags if "arch=compute_89" in flag]
    return [[str(nvcc), *precise_flags, "-x", "cu", "-c", str(Path(manifest["source_root"])/PRECISE), "-o", str(directory/"precise.o")],
            [str(nvcc), *fast_flags, "-x", "cu", "-c", manifest["tools"]["rope_arithmetic_candidates_v2.cu"]["path"], "-o", str(directory/"candidates.o")],
            [str(nvcc), "-std=c++17", "-O2", *arch, "-c", manifest["tools"]["rope_arithmetic_probe_v2.cu"]["path"], "-o", str(directory/"host.o")],
            [str(nvcc), "--cudart=shared", str(directory/"precise.o"), str(directory/"candidates.o"), str(directory/"host.o"),
             "-L"+str(runtime.parent), "-Xlinker", "-rpath", "-Xlinker", str(runtime.parent), "-o", str(directory/"rope_arithmetic_probe_v2")]]


def build_probe(manifest_path, nvcc):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(manifest_path)
    nvcc = Path(nvcc).resolve(strict=True)
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    require(re.search(r"release 13\.", version), "requires production CUDA13 compiler")
    options = options_for(manifest, nvcc)
    directory = manifest_path.parent/"build"
    directory.mkdir(exist_ok=False)
    commands = compile_commands(manifest, nvcc, options, directory)
    log = directory/"compile.log"
    with log.open("x") as stream:
        for command in commands:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=stream, check=True)
    require(validate_manifest(manifest_path) == manifest, "fixture/source changed during build")
    require(options_for(manifest, nvcc) == options, "CMake rules changed during build")
    runtime = Path(options["oracle_precise"]["runtime_library_link"])
    result = {"schema_version": SCHEMA+"-build.v2", "fixture_manifest": evidence(manifest_path),
              "compiler": evidence(nvcc), "compiler_version": version, "production_flags": options,
              "runtime_library": evidence(runtime), "runtime_library_link": str(runtime), "commands": commands,
              "artifacts": {name: evidence(directory/name) for name in ("precise.o", "candidates.o", "host.o", "rope_arithmetic_probe_v2", "compile.log")},
              "gpu_tests_executed": False, "performance_claim_eligible": False}
    write_new(manifest_path.parent/"compile.json", result)
    return result


def validate_compile(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = validate_manifest(manifest_path)
    compiled = read(manifest_path.parent/"compile.json")
    require(compiled["schema_version"] == SCHEMA+"-build.v2" and compiled["fixture_manifest"] == evidence(manifest_path), "build manifest differs")
    for key in ("compiler", "runtime_library"):
        check_evidence(compiled[key])
    require(set(compiled["artifacts"]) == {"precise.o", "candidates.o", "host.o", "rope_arithmetic_probe_v2", "compile.log"}, "build artifacts differ")
    for name, artifact in compiled["artifacts"].items():
        require(artifact == evidence(manifest_path.parent/"build"/name), "build artifact differs")
    nvcc = Path(compiled["compiler"]["path"])
    options = options_for(manifest, nvcc)
    require(options == compiled["production_flags"], "actual compile flags changed")
    require(compiled["commands"] == compile_commands(manifest, nvcc, options, manifest_path.parent/"build"), "compile commands differ")
    require(compiled["runtime_library_link"] == options["oracle_precise"]["runtime_library_link"] and evidence(compiled["runtime_library_link"]) == compiled["runtime_library"], "runtime selection differs")
    require(compiled["gpu_tests_executed"] is False and compiled["performance_claim_eligible"] is False, "build claims execution/performance")
    return manifest, compiled


def checked_histogram(rows, expected_count):
    require(isinstance(rows, list), "mismatch bit-pair histogram missing")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"oracle_bits", "candidate_bits", "count"}, "invalid bit-pair fields")
        pair = (row["oracle_bits"], row["candidate_bits"])
        require(all(type(value) is int and 0 <= value <= 65535 for value in pair)
                and pair[0] != pair[1] and pair not in result
                and type(row["count"]) is int and 0 < row["count"] <= expected_count, "invalid mismatch bit-pair count")
        result[pair] = row["count"]
    require(list(result) == sorted(result) and sum(result.values()) == expected_count, "histogram does not cover every mismatch")
    return result


def validate_raw(path):
    records = [json.loads(line, object_pairs_hook=attention.pairs_unique) for line in Path(path).read_text().splitlines()]
    require(len(records) == COUNT+2 and all(r["schema"] == SCHEMA+"-native.v2" for r in records), "native record coverage differs")
    device, summary = records[0], records[-1]
    require(device["kind"] == "device" and device["ordinal"] == 0 and device["compute_major"] == 8 and device["compute_minor"] == 9 and device["runtime_version"] == 13000, "device profile differs")
    require(isinstance(device["name"], str) and re.fullmatch(r"[0-9a-f]{32}", device["uuid_hex"]), "invalid native device identity")
    mismatch_cases, mismatch_words = [0, 0], [0, 0]
    bit_pairs = [{}, {}]
    for record, (index, phase, position, _) in zip(records[1:-1], cases()):
        require(record["kind"] == "case" and (record["case_id"], record["phase"], record["position"], record["mapping"]) == (index, phase, position, index % 3), "native case identity differs")
        for key in ("complete", "guards_intact", "inputs_unchanged", "oracle_outputs_unchanged", "output_contract", "all_allocations_freed"):
            require(record[key] is True, "native case contract failed: "+key)
        require((record["qkv_bytes"], record["q_bytes_compared"], record["kv_bytes_each_compared"]) == (1920, 1152, 98304), "compared extents differ")
        require(len(record["candidates"]) == 2, "candidate coverage differs")
        for order, candidate in enumerate(record["candidates"]):
            regions = candidate["region_mismatches"]
            require(candidate["order"] == order and len(regions) == 3 and all(type(n) is int and 0 <= n <= limit for n, limit in zip(regions, (576, 49152, 49152))), "invalid candidate counts")
            count = sum(regions)
            require(candidate["mismatch_words"] == count and candidate["equal"] is (count == 0), "candidate equality summary differs")
            histogram = checked_histogram(candidate["mismatch_bit_pairs"], count)
            for pair, amount in histogram.items():
                bit_pairs[order][pair] = bit_pairs[order].get(pair, 0)+amount
            first = candidate["first_mismatch"]
            if count:
                region = next(i for i, n in enumerate(regions) if n)
                require(first["region"] == ("q_rotary", "keys", "values")[region] and type(first["word"]) is int and 0 <= first["word"] < (576, 49152, 49152)[region], "invalid first mismatch position")
                require(all(type(first[key]) is int and 0 <= first[key] <= 65535 for key in ("oracle_bits", "candidate_bits")) and first["oracle_bits"] != first["candidate_bits"], "invalid mismatch raw bits")
                require((first["oracle_bits"], first["candidate_bits"]) in histogram, "first mismatch absent from full histogram")
            else:
                require(first is None, "equal result includes mismatch")
            mismatch_cases[order] += count != 0
            mismatch_words[order] += count
    require(summary["kind"] == "summary" and summary["complete"] is True and summary["cases"] == COUNT and summary["error"] == "", "native experiment incomplete")
    require(summary["candidate_mismatch_cases"] == mismatch_cases and summary["candidate_mismatch_words"] == mismatch_words and summary["candidate_equal"] == [n == 0 for n in mismatch_cases], "aggregate mismatch counts differ")
    require(len(summary["candidate_mismatch_bit_pairs"]) == 2, "aggregate histogram coverage differs")
    for order, histogram in enumerate(summary["candidate_mismatch_bit_pairs"]):
        require(checked_histogram(histogram, mismatch_words[order]) == bit_pairs[order], "aggregate bit pairs differ from every case")
    require(summary["device_allocations_created"] == COUNT*13 and summary["device_allocations_freed"] == COUNT*13, "allocation counts differ")
    for key in ("live_device_allocations", "live_device_bytes", "cleanup_errors"):
        require(summary[key] == 0, "native cleanup failed")
    require(summary["all_allocations_freed"] is True and summary["stream_destroyed"] is True, "native cleanup incomplete")
    require(all(summary[key] is False for key in ("fusion_qualified", "performance_measured", "performance_claim_eligible")), "numerical experiment claims unsupported qualification")
    classifications = []
    for order in range(2):
        signed_zero = sum(count for (a, b), count in bit_pairs[order].items() if (a & 0x7fff) == (b & 0x7fff) == 0)
        classifications.append({"candidate": order, "mismatch_words": mismatch_words[order],
            "signed_zero_only_words": signed_zero, "other_mismatch_words": mismatch_words[order]-signed_zero,
            "all_mismatches_signed_zero": signed_zero == mismatch_words[order] if mismatch_words[order] else None})
    return {"device": device, "summary": summary, "mismatch_classification": classifications}


def run_probe(manifest_path, driver_library_dir=None):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, compiled = validate_compile(manifest_path)
    directory = manifest_path.parent/"run"
    directory.mkdir(exist_ok=False)
    runtime = Path(compiled["runtime_library_link"])
    libraries, driver = [str(runtime.parent)], None
    if driver_library_dir:
        driver_dir = Path(driver_library_dir).resolve(strict=True)
        driver = evidence(driver_dir/"libcuda.so.1")
        libraries.insert(0, str(driver_dir))
    command = [compiled["artifacts"]["rope_arithmetic_probe_v2"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"]
    env = os.environ.copy()
    require(not env.get("LD_PRELOAD"), "LD_PRELOAD would change the probe runtime")
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    raw, stderr = directory/"native.jsonl", directory/"stderr.log"
    with raw.open("x") as out, stderr.open("x") as err:
        result = subprocess.run(command, cwd=directory, env=env, stdout=out, stderr=err)
    require(result.returncode == 0, "native probe failed; inspect run/native.jsonl and stderr.log")
    native = validate_raw(raw)
    require(validate_compile(manifest_path) == (manifest, compiled), "source/build changed during run")
    if driver:
        check_evidence(driver)
    receipt = {"schema_version": SCHEMA+"-run.v2", "complete": True, "fixture_manifest": evidence(manifest_path),
               "compile_receipt": evidence(manifest_path.parent/"compile.json"), "command": command,
               "library_directories": libraries, "explicit_driver_library": driver, "raw": evidence(raw), "stderr": evidence(stderr),
               **native, "gpu_tests_executed": True, "fusion_qualified": False, "performance_claim_eligible": False}
    write_new(manifest_path.parent/"receipt.json", receipt)
    return receipt



def validate_receipt(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, compiled = validate_compile(manifest_path)
    receipt = read(manifest_path.parent/"receipt.json")
    require(receipt["schema_version"] == SCHEMA+"-run.v2" and receipt["complete"] is True, "run receipt incomplete")
    require(receipt["fixture_manifest"] == evidence(manifest_path) and receipt["compile_receipt"] == evidence(manifest_path.parent/"compile.json"), "run build/fixture identity differs")
    require(receipt["raw"] == evidence(manifest_path.parent/"run/native.jsonl") and receipt["stderr"] == evidence(manifest_path.parent/"run/stderr.log"), "run output identity differs")
    native = validate_raw(receipt["raw"]["path"])
    require(receipt["device"] == native["device"] and receipt["summary"] == native["summary"], "run summary differs from raw records")
    require(receipt["mismatch_classification"] == native["mismatch_classification"], "raw mismatch classification differs")
    expected = [compiled["artifacts"]["rope_arithmetic_probe_v2"]["path"], "--cases", manifest["case_index"]["path"], "--device", "0"]
    require(receipt["command"] == expected, "run command differs")
    directories = receipt["library_directories"]
    runtime_dir = str(Path(compiled["runtime_library_link"]).parent)
    driver = receipt["explicit_driver_library"]
    if driver is None:
        require(directories == [runtime_dir], "run runtime directory differs")
    else:
        check_evidence(driver)
        require(len(directories) == 2 and directories[1] == runtime_dir and evidence(Path(directories[0])/"libcuda.so.1") == driver, "run explicit driver selection differs")
    require(receipt["gpu_tests_executed"] is True and receipt["fusion_qualified"] is False and receipt["performance_claim_eligible"] is False, "run claim scope differs")
    return receipt


def ptx_flags(flags):
    # PTX emission targets one virtual ISA. Replace only the validated SM89
    # code-generation option; retain every arithmetic/include/host option.
    arch = [flag for flag in flags if "arch=compute_89" in flag]
    require(len(arch) == 1 and arch[0].startswith("--generate-code="), "unknown PTX architecture flag spelling")
    return [flag for flag in flags if flag != arch[0]]+["--gpu-architecture=compute_89"]


def inspect_probe(manifest_path, cuobjdump):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, compiled = validate_compile(manifest_path)
    tool = Path(cuobjdump).resolve(strict=True)
    tool_version = subprocess.check_output([str(tool), "--version"], text=True)
    require(re.search(r"release 13\.", tool_version), "requires CUDA13 cuobjdump")
    directory = manifest_path.parent/"inspect"
    directory.mkdir(exist_ok=False)
    build = read(manifest["source_build"]["path"])
    binaries = sorted(build["binaries"])
    items = [("production-"+str(i), evidence(path)) for i, path in enumerate(binaries)]
    items += [("oracle", compiled["artifacts"]["precise.o"]), ("candidates", compiled["artifacts"]["candidates.o"])]
    records = []
    for name, artifact in items:
        command = [str(tool), "--dump-sass", "--gpu-architecture", "sm_89", artifact["path"]]
        output, err = directory/(name+".sass"), directory/(name+".stderr")
        with output.open("x") as out, err.open("x") as stderr:
            subprocess.run(command, stdout=out, stderr=stderr, check=True)
        text = output.read_text()
        marker = "rope_probe_explicit" if name == "candidates" else "compiled_packed_decode_rope_kv"
        blocks = [block for block in re.split(r"(?=^[ \t]*Function[ \t]*:)", text, flags=re.MULTILINE) if marker in block]
        require(blocks, "RoPE SASS function missing: "+name)
        selected = directory/(name+"-rope.sass")
        with selected.open("x") as stream:
            stream.write("\n".join(blocks))
        records.append({"kind": "actual_binary_sass", "input": artifact, "command": command,
                        "output": evidence(output), "stderr": evidence(err), "selected_rope": evidence(selected)})
    for name, key, source in (("oracle", "oracle_precise", str(Path(manifest["source_root"])/PRECISE)),
                               ("candidates", "candidates_fast_math", manifest["tools"]["rope_arithmetic_candidates_v2.cu"]["path"])):
        output, log = directory/(name+".ptx"), directory/(name+"-ptx.log")
        require(not output.exists(), "PTX output already exists")
        command = [compiled["compiler"]["path"], *ptx_flags(compiled["production_flags"][key]["flags"]), "--ptx", "-x", "cu", source, "-o", str(output)]
        with log.open("x") as stream:
            subprocess.run(command, cwd=directory, stdout=stream, stderr=stream, check=True)
        records.append({"kind": "regenerated_source_ptx_not_frozen_binary", "source": evidence(source), "command": command,
                        "output": evidence(output), "log": evidence(log), "architecture_only_transform": "SM89 generate-code to compute_89 PTX"})
    require(validate_compile(manifest_path) == (manifest, compiled), "build/source changed during inspection")
    result = {"schema_version": SCHEMA+"-inspection.v2", "fixture_manifest": evidence(manifest_path),
              "compile_receipt": evidence(manifest_path.parent/"compile.json"), "cuobjdump": evidence(tool), "cuobjdump_version": tool_version, "artifacts": records,
              "gpu_tests_executed": False, "fusion_qualified": False}
    write_new(manifest_path.parent/"inspection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cli = sub.add_parser("prepare")
    prepare_cli.add_argument("--build-receipt", required=True)
    prepare_cli.add_argument("--output-dir", required=True)
    for name in ("build", "run", "inspect", "validate"):
        child = sub.add_parser(name)
        child.add_argument("--manifest", required=True)
        if name == "build": child.add_argument("--nvcc", required=True)
        if name == "inspect": child.add_argument("--cuobjdump", required=True)
        if name == "run": child.add_argument("--driver-library-dir")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.build_receipt, args.output_dir)
    elif args.command == "build": result = build_probe(args.manifest, args.nvcc)
    elif args.command == "run": result = run_probe(args.manifest, args.driver_library_dir)
    elif args.command == "inspect": result = inspect_probe(args.manifest, args.cuobjdump)
    else:
        manifest = Path(args.manifest).resolve(strict=True)
        if (manifest.parent/"receipt.json").exists(): result = validate_receipt(manifest)
        elif (manifest.parent/"compile.json").exists():
            validate_compile(manifest)
            result = {"fixture_manifest_valid": True, "compile_receipt_valid": True, "gpu_tests_executed": False}
        else:
            validate_manifest(manifest)
            result = {"fixture_manifest_valid": True, "gpu_tests_executed": False}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
