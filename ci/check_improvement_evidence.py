#!/usr/bin/env python3
"""Verify improvement-plan links, evidence digests, and recorded distribution inputs."""

import argparse
from decimal import Decimal
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit


MANIFEST = Path("benchmarks/evidence/20261119-improvement-qualification/manifest.json")
PLANS = Path("docs/improvement-plan")
INLINE_LINK = re.compile(r"\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)\)")
ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
METRIC_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def evidence_entries(value):
    if isinstance(value, dict):
        if "path" in value:
            yield value
        for child in value.values():
            yield from evidence_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_entries(child)


def check_evidence(root, manifest):
    count = 0
    for entry in evidence_entries(manifest):
        name, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError(f"invalid evidence path: {name!r}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid evidence SHA-256: {name}")
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"evidence file missing from repository: {name}")
        with artifact.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"evidence SHA-256 differs: {name}")
        count += 1
    if count == 0:
        raise ValueError("no checked evidence files in manifest")
    return count


def check_distribution_inputs(root, manifest):
    candidate = manifest["current_distribution_candidate"]
    revisions = (candidate["audited_main"], candidate["source_revision"], *candidate.get("follow_up_audited_revisions", []))
    if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
           for revision in revisions):
        raise ValueError("expected full Git revisions for the recorded distribution comparison")
    inputs = candidate["matching_distribution_input_git_objects"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("no recorded distribution inputs")
    expected = {}
    for path, object_id in inputs.items():
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"invalid distribution input path: {path!r}")
        if not isinstance(object_id, str) or not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise ValueError(f"invalid recorded Git object: {path}")
        for revision in revisions:
            expected[f"{revision}:{path}"] = object_id
    result = subprocess.run(["git", "rev-parse", *expected], cwd=root, capture_output=True, text=True)
    entries = result.stdout.splitlines()
    if result.returncode or len(entries) != len(expected):
        raise ValueError(f"unable to resolve the recorded distribution inputs: {result.stderr.strip()}")
    for (expression, object_id), actual in zip(expected.items(), entries, strict=True):
        if actual != object_id:
            raise ValueError(f"recorded distribution input differs: {expression}")
    return len(inputs)


def check_distribution_revision(root, candidate, revision):
    current = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=root, capture_output=True, text=True,
    )
    if current.returncode:
        raise ValueError(f"unable to resolve distribution comparison revision: {revision}")
    commit = current.stdout.strip()
    paths = candidate["matching_distribution_input_git_objects"]
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--name-only", candidate["source_revision"], commit, "--", *paths],
        cwd=root, capture_output=True, text=True,
    )
    if diff.returncode:
        raise ValueError(f"unable to compare distribution inputs at {commit}: {diff.stderr.strip()}")
    if diff.stdout:
        raise ValueError(f"distribution inputs changed at {commit}: {', '.join(diff.stdout.splitlines())}")
    return commit


def check_distribution_serving(root, candidate):
    records = candidate["serving_reports"]
    reports = {role: json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
               for role, entry in records.items()}
    image = candidate["a100_rebuild_gpu_inference"]["image_id"]
    for role, report in reports.items():
        binary_key = "riley_binary_sha256" if role in ("mixed_stage", "tp2") else "binary_sha256"
        if report["image_id"] != image or report[binary_key] != candidate["binary_sha256"]:
            raise ValueError(f"distribution serving report {role} belongs to a different image or binary")

    recheck = reports["mha_recheck"]
    mha = "SmolLM2-1.7B-Instruct"
    if (recheck["model"] != mha or recheck["prior_result_sha256"] != records["extended"]["sha256"]
            or recheck["api_checks_passed"] != recheck["api_checks"] or recheck["api_failed_check_names"]
            or recheck["smoke"]["expect_tools_unsupported"] is not True
            or recheck["smoke"]["counts"]["tools_unsupported"]["unsupported"] <= 0):
        raise ValueError("distribution MHA report does not verify the template rejection and original run")

    smoke_runs = {
        **reports["representative"]["results"],
        "Qwen2.5-14B-Instruct": reports["extended"]["results"]["Qwen2.5-14B-Instruct"],
        mha: recheck,
    }
    if smoke_runs.keys() != {"Qwen2.5-3B-Instruct", "Qwen3-4B", "Qwen2.5-14B-Instruct", mha}:
        raise ValueError("distribution smoke does not cover all four representative checkpoints")
    expected = {}
    for model, run in smoke_runs.items():
        smoke = run["smoke"]
        if (run["status"] != "passed" or run["exit"] != 0 or smoke["status"] != "passed"
                or smoke["failed_checks"] or run["api_checks"] != (39 if model == mha else 40)):
            raise ValueError(f"distribution smoke for {model} did not pass the API and mixed load")
        counts = smoke["counts"]
        completed = sum(category.get("completed", 0) for category in counts.values())
        disconnects = counts["cancel"]["disconnected"]
        detail = f'{smoke["seconds"]:g} s mixed load, {completed} completions, {disconnects} disconnects'
        if model == mha:
            detail += f', {counts["tools_unsupported"]["unsupported"]} expected tool rejections'
        expected[model] = ("1", str(run["api_checks"]), detail)

    mixed = reports["mixed_stage"]
    trials = mixed["mixed"]["trials"]
    if (mixed["api_checks"] != 40 or mixed["mixed"]["status"] != "passed"
            or len(trials) != 3 or any(trial["matching_responses"] != 2 or not trial["witnesses"] for trial in trials)):
        raise ValueError("distribution mixed-stage report lacks the matched mixed GPU iterations")
    iterations = sum(len(trial["witnesses"]) for trial in trials)
    matches = sum(trial["matching_responses"] for trial in trials)
    expected["Qwen2.5-7B-Instruct"] = ("1", str(mixed["api_checks"]),
                                       f"{iterations} committed mixed GPU iterations, {matches} exact responses")

    tp = reports["tp2"]
    holdout = tp["tensor_parallel"]["holdout"]
    modes = holdout["modes"]
    c8, c32 = modes["c8"], modes["c32"]
    if (tp["status"] != "passed" or tp["api_checks"] != 40 or holdout["status"] != "passed"
            or len(set(tp["tensor_parallel"]["gpu_uuids"])) != 2 or modes["c1"]["requests"] != 11
            or c8["exact_matches"] != c8["requests"] or c8["requests"] != 11
            or c32["exact_matches"] != c32["requests"] or c32["requests"] != 32
            or not 1 < c8["observed_peak_active"] <= 8 or not 1 < c32["observed_peak_active"] <= 32):
        raise ValueError("distribution TP=2 report lacks exact long-context matches")
    expected["Qwen2.5-32B-Instruct"] = ("2", str(tp["api_checks"]),
                                      f'{c8["exact_matches"] + c32["exact_matches"]} exact concurrent responses, '
                                      f'peak {c32["observed_peak_active"]} active with 32 clients')

    support = root / "docs/model-support.md"
    image_label = f'A100 image `{image.removeprefix("sha256:")[:12]}` (source `{candidate["source_revision"][:9]}`)'
    if image_label not in support.read_text(encoding="utf-8"):
        raise ValueError("model-support.md: serving image and source differ from the recorded A100 candidate")
    documented = markdown_table(support, ("Checkpoint", "A100 GPUs", "API checks", "Serving exercise"),
                                "## Packaged serving checks")
    if documented != expected:
        rows = sorted(name for name in documented.keys() | expected.keys() if documented.get(name) != expected.get(name))
        raise ValueError(f"model-support.md: packaged serving coverage differs for {', '.join(rows)}")
    return len(expected)


def check_allocator_trial(root, trial, distribution):
    records = trial["evidence"]
    reports = {kind: json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
               for kind, entry in records.items()}
    package = reports["package"]
    soak = reports["soak"]
    accuracy = reports["accuracy"]
    performance = reports["performance"]
    confirmation = reports["rag_confirmation"]
    replay = reports["rag_replay"]
    image = trial["image_id"]
    parent = distribution["a100_rebuild_gpu_inference"]["image_id"]
    if (package["status"] != "passed" or package["image_id"] != image or package["parent_image_id"] != parent
            or package["binaries"]["/opt/riley/bin/riley"] != distribution["binary_sha256"]
            or any(package[key] != distribution[key] for key in ("source_revision", "source_archive_sha256"))):
        raise ValueError("allocator package is not linked to the A100 parent and its binary")
    binding = soak["image_binding"]
    if (binding != accuracy["image_binding"] or binding["status"] != "passed"
            or any(binding[key] != package[key] for key in ("image_id", "parent_image_id", "source_revision",
                                                            "source_archive_sha256", "binaries"))
            or binding["experimental_packaged_cpu_attestation_sha256"] != records["package"]["sha256"]
            or soak["gpu_checker_input_sha256"] != accuracy["gpu_checker_input_sha256"]):
        raise ValueError("allocator package, soak and accuracy attestations differ")
    combined = {**soak, "results": {**soak["results"], "accuracy": accuracy["result"]}}
    encoded = (json.dumps(combined, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if sha256(encoded).hexdigest() != accuracy["combined_report_sha256"]:
        raise ValueError("allocator accuracy is not part of the recorded combined GPU report")

    library = package["added_files_sha256"]["/opt/riley/lib/libjemalloc.so.2"]
    for scope, result in combined["results"].items():
        process = result["process_allocator"]
        if (result["status"] != "passed" or result["api_checks"] != 40 or process["image_id"] != image
                or process["library_sha256"] != library or process["malloc_conf"] != trial["malloc_conf"]):
            raise ValueError(f"allocator {scope} did not use the tested image and settings")
    if set(soak["results"]) != {"cache-on", "cache-off"}:
        raise ValueError("allocator trial needs both cache modes")
    for mode, result in soak["results"].items():
        verified = result["verified_soak"]
        rss = result["mixed_soak"]["rss"]
        if (verified["duration_seconds"] != 1800 or verified["problem_count"] != 0
                or verified["checks"]["rss_growth_second_half_at_most_zero"] is not True
                or rss["rss_growth_kib"] != rss["drain"]["rss_kib"] - rss["baseline"]["rss_kib"]
                or rss["rss_growth_kib"] > 0):
            raise ValueError(f"allocator {mode} lacks a passing 30-minute RSS result")
    counts = {name: suite["verified_requests"] for name, suite in accuracy["result"]["accuracy_suites"].items()}
    if counts != trial["accuracy_requests"]:
        raise ValueError("allocator accuracy coverage differs from the recorded qualification scope")

    order = ["parent-first", "packaged-first", "packaged-second", "parent-second"]
    if any(run["order"] != order or set(run["arms"]) != set(order)
           for run in (performance, confirmation, replay)):
        raise ValueError("allocator timing comparison arms differ")
    server_args = performance["arms"][order[0]]["server"]["server_arguments"]
    for name in order:
        closed = [run["arms"][name] for run in (performance, confirmation)]
        servers = [arm["server"] for arm in closed] + [replay["arms"][name]]
        expected = parent if name.startswith("parent-") else image
        if (any(arm["api_checks"] != 40 or arm["image_id"] != expected for arm in closed)
                or any(server["image_id"] != expected or server["server_arguments"] != server_args
                       for server in servers)):
            raise ValueError(f"allocator {name} timing reports use a different image or serving configuration")
        for server in servers:
            allocator = server["allocator"]
            if expected == parent:
                if allocator != "glibc":
                    raise ValueError(f"allocator {name} parent is not running glibc")
            elif (allocator["image_id"] != image or allocator["library_sha256"] != library
                  or allocator["malloc_conf"] != trial["malloc_conf"]):
                raise ValueError(f"allocator {name} timing reports use different allocator settings")

    if confirmation["prior_closed_loop_summary_sha256"] != records["performance"]["sha256"]:
        raise ValueError("allocator RAG confirmation uses a different prior performance report")
    rag_cells = confirmation["cells"]
    if (set(rag_cells) != {f"rag-c8-seed{seed}" for seed in (1, 2, 3)}
            or any(cell["aligned_samples_per_arm"] != 256 or cell["same_complete_streams"] is not True
                   or set(cell["request_timing"]) != set(order) for cell in rag_cells.values())):
        raise ValueError("allocator RAG confirmation does not cover the three matched 256-request cells")

    original = reports["metrics_observer"]
    warmed = reports["metrics_observer_warmed"]
    for record in (original, warmed):
        observation = record["result"]
        execution = record["execution"]
        raw = (json.dumps(observation, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if (execution["campaign_exit"] != 0 or execution["physical_gpu"] != 6
                or execution["remote_verified_json_sha256"] != sha256(raw).hexdigest()
                or observation["api_checks"] != soak["results"]["cache-on"]["api_checks"]
                or observation["image_id"] != image
                or observation["candidate_binary_sha256"] != distribution["binary_sha256"]):
            raise ValueError("allocator metrics observation is not bound to the completed A100 image run")

    replay_result = warmed["result"]
    if (replay_result["matched_prior_report_sha256"] != original["execution"]["remote_verified_json_sha256"]
            or replay_result["checkpoint_sha256"] != original["result"]["checkpoint_sha256"]
            or replay_result["same_counts_per_phase"] != original["result"]["same_counts_per_phase"]):
        raise ValueError("allocator warmed metrics observation does not match the original run")
    return len(reports)


def prose_lines(path):
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker.startswith(fence) and not line[match.end():].strip():
                fence = None
            continue
        if fence is None:
            yield number, line


def markdown_anchors(path):
    counts = {}
    for _, line in prose_lines(path):
        match = ATX_HEADING.match(line)
        if match:
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
            slug = re.sub(r"[^\w -]", "", text.lower()).replace(" ", "-")
            ordinal = counts.get(slug, 0)
            counts[slug] = ordinal + 1
            yield f"{slug}-{ordinal}" if ordinal else slug


def check_plan_links(root):
    count = 0
    anchors = {}
    for plan in sorted((root / PLANS).glob("*.md")):
        for number, line in prose_lines(plan):
            for href in INLINE_LINK.findall(line):
                destination = urlsplit(href.strip("<>"))
                if destination.scheme or destination.netloc:
                    continue
                target = (plan.parent / unquote(destination.path)).resolve() if destination.path else plan
                location = f"{plan.relative_to(root)}:{number}"
                if not target.is_relative_to(root) or not target.exists():
                    raise ValueError(f"{location}: unresolved local link {href}")
                if destination.fragment and target.suffix.lower() == ".md":
                    if target not in anchors:
                        anchors[target] = set(markdown_anchors(target))
                    if unquote(destination.fragment) not in anchors[target]:
                        raise ValueError(f"{location}: unresolved heading in {href}")
                count += 1
    if count == 0:
        raise ValueError("no local improvement-plan links checked")
    return count


def markdown_table(path, header, section=None):
    lines = iter(prose_lines(path))
    if section is not None and not any(line == section for _, line in lines):
        raise ValueError(f"{path.name}: missing section {section}")
    for _, line in lines:
        if line.startswith("## ") and section is not None:
            raise ValueError(f"{path.name}: missing table in {section}")
        if line.startswith("|") and tuple(cell.strip() for cell in line.strip("|").split("|")) == header:
            break
    else:
        raise ValueError(f"{path.name}: missing table {header[0]}")

    rows = {}
    for number, line in lines:
        if not line.startswith("|"):
            break
        fields = [field.strip() for field in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", field) for field in fields):
            continue
        if len(fields) != len(header) or not fields[0] or fields[0] in rows:
            raise ValueError(f"{path.name}:{number}: malformed or duplicate table row")
        rows[fields[0]] = tuple(value.replace("**", "") for value in fields[1:])
    return rows


def metric_pair(value):
    match = METRIC_PAIR.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid October A100 metric pair: {value!r}")
    riley, vllm = map(Decimal, match.groups())
    if min(riley, vllm) <= 0:
        raise ValueError(f"nonpositive October A100 metric pair: {value!r}")
    return riley, vllm


def check_readme(root):
    readme = root / "README.md"
    with (root / "Cargo.toml").open("rb") as cargo:
        version = tomllib.load(cargo)["workspace"]["package"]["version"]
    status = re.findall(r"^> Status: (\S+) \(\d{4}-\d{2}-\d{2}\)\.",
                        readme.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if status != [version]:
        raise ValueError(f"README.md: Status version must match Cargo workspace version {version}")

    public = markdown_table(readme,
                            ("Model", "Throughput tok/s", "Speed-up", "TTFT p50 ms", "TPOT p50 ms"))
    reference = root / "docs/model-coverage-a100-perf-table.md"
    columns = ("모델", "c1-fixed", "c1-natural", "c8-fixed", "c8-natural",
               "c16-fixed", "c16-natural", "c32-fixed", "c32-natural")
    sections = ("## 처리량 (출력 tok/s, 높을수록 좋음)", "## TTFT p50 (ms, 낮을수록 좋음)",
                "## TPOT p50 (ms, 낮을수록 좋음)")
    sources = [markdown_table(reference, columns, section) for section in sections]
    if len(public) != 24 or any(source.keys() != public.keys() for source in sources):
        raise ValueError("README.md: October A100 C32 fixed table must match all 24 reference models")
    for model, (throughput, ratio, ttft, tpot) in public.items():
        for name, value, source in zip(("throughput", "TTFT", "TPOT"), (throughput, ttft, tpot), sources, strict=True):
            expected = source[model][columns.index("c32-fixed") - 1]
            if value != expected:
                raise ValueError(f"README.md: {model} {name} differs from the October A100 C32 fixed table")
        riley, vllm = metric_pair(throughput)
        if ratio != f"{riley / vllm:.2f}×":
            raise ValueError(f"README.md: {model} speed-up differs from the displayed throughput")

    throughput, ttft, tpot = (
        [[metric_pair(value) for value in row] for row in source.values()] for source in sources
    )
    # The first two cells in each row are the fixed and natural single-client trials.
    c1 = [riley / vllm for row in throughput for riley, vllm in row[:2]]
    throughput_ratios = [riley / vllm for row in throughput for riley, vllm in row[2:]]
    ttft_ratios = [vllm / riley for row in ttft for riley, vllm in row]
    tpot_ratios = [vllm / riley for row in tpot for riley, vllm in row[2:]]
    if min(throughput_ratios) <= 1 or min(ttft_ratios) <= 1:
        raise ValueError("README.md: October A100 summary no longer shows a win in every throughput or TTFT cell")
    if not Decimal("0.97") <= min(c1) <= Decimal("1.03"):
        raise ValueError("README.md: October A100 single-client throughput minimum is outside the stated tie band")

    claims = {
        "**Throughput** at 8–32 concurrent clients": (
            f"higher in all {len(throughput_ratios)} cells — {min(throughput_ratios):.2f}× to "
            f"{max(throughput_ratios):.2f}×"
        ),
        "**Time to first token** (p50)": (
            f"lower in all {len(ttft_ratios)} cells — {min(ttft_ratios):.1f}× to {max(ttft_ratios):.1f}× faster"
        ),
        "**Time per output token** (p50) at 8–32 clients": (
            f"lower in {sum(ratio > 1 for ratio in tpot_ratios)} of {len(tpot_ratios)} cells"
        ),
        "One client": f"within ±3 % (tie) to {max(c1):.1f}× faster on throughput",
    }
    summary = markdown_table(readme, ("", "Riley vs vLLM"))
    for label, expected in claims.items():
        if summary.get(label) != (expected,):
            raise ValueError(f"README.md: October A100 summary differs for {label.replace('**', '')}")
    return len(public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--distribution-revision", metavar="REV",
                        help="also require REV to match the recorded distribution image inputs")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        manifest = json.loads((root / MANIFEST).read_bytes(), object_pairs_hook=unique_object)
        files = check_evidence(root, manifest)
        inputs = check_distribution_inputs(root, manifest)
        allocator_reports = check_allocator_trial(root, manifest["current_allocator_trial"],
                                                  manifest["current_distribution_candidate"])
        serving_models = check_distribution_serving(root, manifest["current_distribution_candidate"])
        links = check_plan_links(root)
        models = check_readme(root)
        revision = None
        if args.distribution_revision is not None:
            revision = check_distribution_revision(root, manifest["current_distribution_candidate"],
                                                   args.distribution_revision)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"improvement evidence check failed: {error}", file=sys.stderr)
        return 1
    print(f"improvement evidence verified: {files} file hashes, {inputs} recorded Git inputs, "
          f"{links} local links, {models} README performance rows and summary, "
          f"{allocator_reports} linked allocator reports, {serving_models} packaged serving rows")
    if revision is not None:
        print(f"distribution source inputs match at {revision} ({inputs} recorded paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
