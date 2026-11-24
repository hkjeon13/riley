"""Run the committed FlashInfer workspace layout on the host and audit replay lengths."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import TemporaryDirectory


REVISION = "6ae0e9276cabc2ac6df1d19a0687fea4a8c70c41"
TRACE = Path("benchmarks/evidence/20261124-w3-c32-prefill-tail/raw-results.tar.gz")
PLANNER = "kernels/src/fi_attention.cu"


def git_source(root, revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root, text=True)


def extract(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def fixed_tile_sizes(source, model, cases, compiler):
    heads, kv_heads, dimension, page = model
    pieces = (
        extract(source, "constexpr uint32_t kPage", "constexpr uint32_t kPlannerThreads"),
        extract(source, "uint32_t chunk_tokens(", "uint32_t kv_chunk_tokens("),
        extract(source, "uint64_t max_chunks(", "// Exclusive block-wide sum"),
    )
    cpp = "#include <cstdint>\n#include <iostream>\n" + "".join(pieces)
    cpp += "\nint main() {\n"
    for rows, context, chunk in cases:
        pages = (context + page - 1) // page
        cpp += (
            f"  std::cout << layout_for({rows}, {pages}, {heads}, {heads // kv_heads}, "
            f"{dimension}, chunk_tokens({chunk}, {pages})).bytes << '\\n';\n"
        )
    cpp += "}\n"
    with TemporaryDirectory() as temp:
        output = Path(temp) / "layout"
        subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-x", "c++", "-o", output, "-"],
            input=cpp, text=True, check=True,
        )
        sizes = subprocess.check_output([output], text=True)
    return [int(size) for size in sizes.splitlines()]


def replay_lengths(archive_path):
    instances = {}
    configurations = set()
    groups = {"rag_c8": {}, "short_c32": {}, "holdout": {}}
    with tarfile.open(archive_path) as archive:
        for gpu in (6, 7):
            for arm in ("control", "reserve"):
                path = f"gpu{gpu}/rag/{arm}"
                log = archive.extractfile(f"{path}/server.log").read().decode()
                match = re.search(
                    r"query_heads=(\d+) key_value_heads=(\d+) head_size=(\d+) page_size=(\d+)", log,
                )
                chunk = re.search(r"RILEY_ATTENTION_KV_CHUNK prompt=(\d+) generated=(\d+)", log)
                if "RILEY_NATIVE_ATTENTION mode=fixed-tile-unsplit-v1" not in log or not match or not chunk:
                    raise ValueError(f"missing fixed-tile geometry in {path}")
                if chunk[1] != chunk[2]:
                    raise ValueError(f"different prompt and generated chunk sizes in {path}")
                configurations.add((tuple(map(int, match.groups())), int(chunk[1])))
                plan = json.load(archive.extractfile(f"{path}/plan.json"))
                argv = plan["server"]
                settings = tuple(int(argv[argv.index(opt) + 1]) for opt in (
                    "--batch-token-budget", "--max-sequence-tokens", "--max-active-sequences",
                ))
                instances[path] = settings
                for name, cell in plan["cells"].items():
                    group = groups["rag_c8" if name.startswith("rag-c8") else "short_c32"]
                    output_limit = cell["cell"]["max_tokens"]
                    result = json.load(archive.extractfile(f"{path}/{name}.json"))
                    if (
                        result["length_mismatches"]
                        or len(cell["schedule"]) != cell["requests"]
                        or len(result["measured"]) != cell["requests"]
                    ):
                        raise ValueError(f"incorrect request count for {name}")
                    for entry, observed in zip(cell["schedule"], result["measured"], strict=True):
                        keys = ("request_index", "input_sha256", "prompt_tokens")
                        if any(entry[field] != observed[field] for field in keys):
                            raise ValueError(f"observed prompt differs from schedule in {name}")
                        key = (name, entry["request_index"], entry["input_sha256"])
                        length = entry["prompt_tokens"] + output_limit - 1
                        if group.setdefault(key, length) != length:
                            raise ValueError(f"request length changed for {key}")
                holdout = json.load(archive.extractfile(f"{path}/token-id-holdout.json"))
                for mode, requests in holdout.items():
                    for entry in requests:
                        key = (mode, entry["input_sha256"])
                        if entry["finish"] != "length":
                            raise ValueError("holdout response did not reach its output limit")
                        length = entry["prompt_tokens"] + entry["completion_tokens"] - 1
                        if groups["holdout"].setdefault(key, length) != length:
                            raise ValueError(f"holdout length changed for {key}")
    if len(configurations) != 1 or set(instances.values()) != {(1024, 8192, 32)}:
        raise ValueError("expected one fixed-tile baseline geometry")
    geometry, chunk = configurations.pop()
    report = {
        name: {
            "request_count": len(rows),
            "maximum_kv_tokens": max(rows.values()),
            "requests_above_4096_kv_tokens": sum(length > 4096 for length in rows.values()),
        }
        for name, rows in groups.items()
    }
    return geometry, chunk, len(instances), report


def audit(root, compiler):
    trace = root / TRACE
    registration = json.loads((trace.parent / "registration.json").read_bytes())
    planner = git_source(root, REVISION, PLANNER)
    if planner != git_source(root, registration["source_revision"], PLANNER):
        raise ValueError("trace and audited source use different FlashInfer planners")
    cli = git_source(root, REVISION, "crates/riley-server/src/main.rs")
    owner = git_source(root, REVISION, "crates/riley-runtime/src/llama/executor/owner.rs")
    launch = git_source(root, REVISION, "kernels/src/batch_primitives.cu")
    guards = (
        (cli, "&& (batch_token_budget > 1024"),
        (owner, "NATIVE_D128_TWO_STAGE_MAX_DENSE_ROWS: usize = 1024;"),
        (owner, "NATIVE_D128_TWO_STAGE_MAX_ROW_TOKENS: usize = 256 * 32_768;"),
        (launch, "flashinfer_requested && params->output_row_count <= 1024"),
        (planner, "constexpr uint32_t kPlannerThreads = 1024;"),
        (planner, "output_rows > kPlannerThreads"),
        (planner, "const bool live = thread < active;"),
        (planner, "batch.row_positions[thread]"),
        (planner, "plan_kernel<<<1, kPlannerThreads, 0, stream>>>"),
    )
    if any(expected not in source for source, expected in guards):
        raise ValueError("the audited row or row-token limits changed")
    model, chunk, runs, requests = replay_lengths(trace)
    heads, kv_heads, dimension, page = model
    cases = ((1024, 8192, chunk), (1024, 4096, chunk), (2048, 4096, chunk), (2048, 8192, chunk))
    sizes = fixed_tile_sizes(planner, model, cases, compiler)
    return {
        "source_revision": REVISION,
        "trace_source_revision": registration["source_revision"],
        "trace_sha256": sha256(trace.read_bytes()).hexdigest(),
        "trace_runs": runs,
        "query_heads": heads,
        "key_value_heads": kv_heads,
        "head_dimension": dimension,
        "page_tokens": page,
        "kv_chunk_tokens": chunk,
        "fixed_tile_planner_threads": 1024,
        "maximum_runtime_row_tokens": 256 * 32768,
        "replay_requests": requests,
        "workspace_layouts": [
            {
                "rows": rows,
                "context_tokens": context,
                "row_tokens": rows * context,
                "within_runtime_row_token_product": rows * context <= 256 * 32768,
                "within_runtime_row_limit": rows <= 1024,
                "fixed_tile_partial_state_bytes": size,
                "page_v1_partial_state_bytes": rows * ((context + page - 1) // page) * heads * (dimension + 2) * 4,
            }
            for (rows, context, _), size in zip(cases, sizes, strict=True)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    print(json.dumps(audit(args.repo, args.compiler), indent=2))
