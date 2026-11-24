"""Extract the registered CUDA planner and test its metadata with simulated collectives."""

import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected planner source around {old!r}")
    return source.replace(old, new, 1)


def verify(root, staged, compiler):
    relative = "kernels/src/fi_attention.cu"
    hashes = dict(reversed(line.split("  ", 1)) for line in (staged / "after.sha256").read_text().splitlines())
    files = [relative, "kernels/src/fi_attention.hpp"]
    for name in files:
        if sha256((root / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"planner does not match the staged candidate: {name}")
    body, header = ((root / name).read_text() for name in files)
    start = body.index("template <uint32_t kCapacity>")
    end = body.index("// FlashInfer's paged-prefill work items", start)
    adapted = body[start:end]
    adapted = once(adapted, "__global__ __launch_bounds__(kPlannerThreads) void plan_kernel", "Task plan_kernel")
    adapted = once(adapted, "  uint4* const pad = reinterpret_cast<uint4*>(output + active * row_width);\n", "")
    adapted = once(adapted, "pad[index] = make_uint4(0U, 0U, 0U, 0U);",
                   "std::fill_n(output + active * row_width + 8 * index, 8, uint16_t{0});")
    adapted = adapted.replace("__shared__", "static").replace("__syncthreads();", "co_await Barrier{};")
    adapted = adapted.replace("block_exclusive_sum(", "co_await Scan(")
    start = header.index("struct RowBatch {")
    end = header.index("\n};", start) + len("\n};")
    test = Path(__file__).with_name("replay.cpp")
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "planner.inc").write_text(header[start:end] + "\n" + adapted)
        binary = directory / "replay"
        subprocess.run([compiler, "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-I", directory, "-include", "memory",
                        test, "-o", binary], check=True)
        subprocess.run([binary], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_tree", type=Path)
    parser.add_argument("staged_kit", type=Path)
    parser.add_argument("--compiler", default="clang++")
    args = parser.parse_args()
    verify(args.candidate_tree, args.staged_kit, args.compiler)
