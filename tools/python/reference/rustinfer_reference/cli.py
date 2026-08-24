"""Command line entry point for fixture generation, validation, and benchmarks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    GOLDEN_GREEDY_MAX_NEW_TOKENS,
    MAX_CONTEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    UINT64_MAX,
)
from .fixture import (
    FixtureError,
    collect_fixture_provenance,
    generate_fixture,
    load_fixture,
    load_prompts,
    utc_now,
    validate_fixture_against_prompts,
    validate_fixture_against_repository,
    write_fixture_exclusive,
)
from .calibration import CalibrationError
from .environment import (
    EnvironmentContractError,
    EnvironmentProbe,
    probe_primary_environment,
    validate_environment_snapshot,
)

BackendFactory = Callable[..., Any]
ProvenanceFactory = Callable[..., dict[str, object]]


def _uint64(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if not 0 <= parsed <= UINT64_MAX:
        raise argparse.ArgumentTypeError("must be in [0, 2^64 - 1]")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _golden_token_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > GOLDEN_GREEDY_MAX_NEW_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be <= {GOLDEN_GREEDY_MAX_NEW_TOKENS} for exact BF16 cache parity"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rustinfer-reference",
        description=(
            f"Pinned {MODEL_ID}@{MODEL_REVISION} external Python reference lane"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate cache-parity golden fixtures"
    )
    generate.add_argument("--prompts", type=Path, required=True, help="prompt JSONL")
    generate.add_argument("--output", type=Path, required=True, help="new fixture path")
    generate.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="clean Git top-level whose exact source bytes are recorded",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=_golden_token_count,
        default=GOLDEN_GREEDY_MAX_NEW_TOKENS,
        metavar="N",
        help=(
            "exact cache-on/off golden window "
            f"(default/max: {GOLDEN_GREEDY_MAX_NEW_TOKENS})"
        ),
    )
    generate.add_argument(
        "--hidden-state-index",
        type=_positive_int,
        default=1,
        metavar="N",
        help="hidden_states tuple index; 1 is the first transformer layer output",
    )
    generate.add_argument("--top-k", type=_positive_int, default=10, metavar="K")
    generate.add_argument("--seed", type=_uint64, default=0)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the immutable revision to be fetched (default: cache-only/offline)",
    )

    validate = subparsers.add_parser(
        "validate", help="validate fixture schema and semantic invariants without PyTorch"
    )
    validate.add_argument("fixture", type=Path)
    validate.add_argument("--prompts", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="emit HF eager end-to-end raw JSONL for one process run"
    )
    benchmark.add_argument("--matrix", type=Path, required=True)
    benchmark.add_argument("--prompts", type=Path, required=True)
    benchmark.add_argument("--result-dir", type=Path, required=True)
    benchmark.add_argument(
        "--run-index",
        type=_positive_int,
        default=1,
        help="independent process-run index (1..matrix independent_runs)",
    )
    benchmark.add_argument(
        "--run-id",
        required=True,
        help="shared ID for every matrix cell in this independent run",
    )
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--allow-download", action="store_true")
    benchmark.add_argument(
        "--warm-state", choices=("cold", "warm"), required=True
    )
    benchmark.add_argument("--concurrency", type=_positive_int, required=True)
    benchmark.add_argument("--prompt-tokens", type=_positive_int, required=True)
    benchmark.add_argument("--output-tokens", type=_positive_int, required=True)

    calibration_produce = subparsers.add_parser(
        "calibrate-produce",
        help="produce one immutable HF FP32 or BF16 oracle manifest plus sidecar",
    )
    calibration_produce.add_argument("--role", choices=("fp32", "bf16"), required=True)
    calibration_produce.add_argument("--prompts", type=Path, required=True)
    calibration_produce.add_argument("--manifest", type=Path, required=True)
    calibration_produce.add_argument("--sidecar", type=Path, required=True)
    calibration_produce.add_argument("--repo-root", type=Path, required=True)
    calibration_produce.add_argument("--device", default="cuda:0")
    calibration_produce.add_argument("--allow-download", action="store_true")

    calibration_manifest = subparsers.add_parser(
        "calibrate-validate-manifest",
        help="verify a calibration manifest, repository bindings, and safetensors sidecar",
    )
    calibration_manifest.add_argument("manifest", type=Path)
    calibration_manifest.add_argument(
        "--expected-kind",
        choices=("fp32-numeric-oracle", "bf16-semantic-oracle", "candidate"),
        required=True,
    )
    calibration_manifest.add_argument("--repo-root", type=Path, required=True)

    oracle_compare = subparsers.add_parser(
        "calibrate-oracles",
        help="recompute HF FP32/BF16 calibration evidence from raw sidecars",
    )
    oracle_compare.add_argument("--fp32-manifest", type=Path, required=True)
    oracle_compare.add_argument("--bf16-manifest", type=Path, required=True)
    oracle_compare.add_argument("--output", type=Path, required=True)
    oracle_compare.add_argument("--repo-root", type=Path, required=True)

    candidate_compare = subparsers.add_parser(
        "calibrate-compare",
        help="gate a native two-variant candidate against FP32/BF16 raw oracles",
    )
    candidate_compare.add_argument("--fp32-manifest", type=Path, required=True)
    candidate_compare.add_argument("--bf16-manifest", type=Path, required=True)
    candidate_compare.add_argument("--oracle-report", type=Path, required=True)
    candidate_compare.add_argument("--candidate-manifest", type=Path, required=True)
    candidate_compare.add_argument("--output", type=Path, required=True)
    candidate_compare.add_argument("--repo-root", type=Path, required=True)

    oracle_validate = subparsers.add_parser(
        "calibrate-validate-oracles",
        help="approve an oracle report only by replaying bound sidecars",
    )
    oracle_validate.add_argument("report", type=Path)
    oracle_validate.add_argument("--fp32-manifest", type=Path, required=True)
    oracle_validate.add_argument("--bf16-manifest", type=Path, required=True)
    oracle_validate.add_argument("--repo-root", type=Path, required=True)

    candidate_validate = subparsers.add_parser(
        "calibrate-validate-report",
        help="approve an E0 report only by replaying all bound raw sidecars",
    )
    candidate_validate.add_argument("report", type=Path)
    candidate_validate.add_argument("--fp32-manifest", type=Path, required=True)
    candidate_validate.add_argument("--bf16-manifest", type=Path, required=True)
    candidate_validate.add_argument("--oracle-report", type=Path, required=True)
    candidate_validate.add_argument("--candidate-manifest", type=Path, required=True)
    candidate_validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def _default_backend_factory(
    *,
    device: str,
    local_files_only: bool,
    enable_observability: bool = False,
):
    # Keeping this import here is intentional: validate and --help have no torch cost.
    from .hf_backend import HuggingFaceBackend

    return HuggingFaceBackend.load(
        device=device,
        local_files_only=local_files_only,
        enable_observability=enable_observability,
    )


def _run_generate(
    args: argparse.Namespace,
    *,
    backend_factory: BackendFactory,
    provenance_factory: ProvenanceFactory,
    environment_probe: EnvironmentProbe,
    now: Callable[[], datetime],
) -> int:
    if args.output.exists():
        raise FixtureError(f"refusing to overwrite existing fixture: {args.output}")
    repo_root = args.repo_root.resolve()
    canonical_prompts = repo_root / "benchmarks/prompts.jsonl"
    if args.prompts.resolve() != canonical_prompts:
        raise FixtureError(
            f"--prompts must be the canonical repository corpus: {canonical_prompts}"
        )
    output = args.output.resolve()
    if output == repo_root or repo_root in output.parents:
        raise FixtureError("fixture staging output must be outside the repository")
    prompts, corpus_sha256 = load_prompts(args.prompts)
    for prompt in prompts:
        if prompt.target_prompt_tokens == 0:
            raise FixtureError(
                f"{prompt.prompt_id}: target_prompt_tokens=0 cannot execute a causal model"
            )
        if (
            prompt.target_prompt_tokens is not None
            and prompt.target_prompt_tokens + args.max_new_tokens > MAX_CONTEXT_TOKENS
        ):
            raise FixtureError(
                f"{prompt.prompt_id}: target prompt plus output exceeds "
                f"{MAX_CONTEXT_TOKENS} tokens"
            )
    try:
        observed_environment = environment_probe()
        validate_environment_snapshot(observed_environment)
    except EnvironmentContractError as error:
        raise FixtureError(f"primary environment preflight failed: {error}") from error
    provenance = provenance_factory(
        repo_root, observed_environment=observed_environment
    )
    recorded_sources = provenance.get("sources")
    recorded_prompts = (
        recorded_sources.get("prompts", {})
        if isinstance(recorded_sources, dict)
        else {}
    )
    if not isinstance(recorded_prompts, dict) or recorded_prompts.get(
        "sha256"
    ) != corpus_sha256:
        raise FixtureError("Git provenance prompt hash differs from the loaded corpus")
    backend = backend_factory(
        device=args.device, local_files_only=not args.allow_download
    )
    fixture = generate_fixture(
        prompts,
        corpus_sha256,
        backend,
        max_new_tokens=args.max_new_tokens,
        hidden_state_index=args.hidden_state_index,
        top_k=args.top_k,
        seed=args.seed,
        provenance=provenance,
        created_at=now(),
    )
    write_fixture_exclusive(args.output, fixture)
    print(f"wrote {len(prompts)} reference cases to {args.output}")
    return 0


def _run_validate(
    args: argparse.Namespace,
    *,
    repository_validator: Callable[[dict[str, object], Path], None],
) -> int:
    repo_root = args.repo_root.resolve()
    canonical_prompts = repo_root / "benchmarks/prompts.jsonl"
    if args.prompts.resolve() != canonical_prompts:
        raise FixtureError(
            f"--prompts must be the canonical repository corpus: {canonical_prompts}"
        )
    fixture = load_fixture(args.fixture)
    prompts, corpus_sha256 = load_prompts(args.prompts)
    validate_fixture_against_prompts(fixture, prompts, corpus_sha256)
    repository_validator(fixture, repo_root)
    print(
        f"valid reference fixture: {args.fixture} "
        f"({len(fixture['cases'])} cases, context={MAX_CONTEXT_TOKENS})"
    )
    return 0


def _run_benchmark(
    args: argparse.Namespace,
    *,
    backend_factory: BackendFactory,
    now: Callable[[], datetime],
) -> int:
    # Benchmark orchestration is pure Python; backend creation remains lazy/timed.
    from .benchmark import run_benchmark

    rows = run_benchmark(
        matrix_path=args.matrix,
        prompts_path=args.prompts,
        result_dir=args.result_dir,
        backend_factory=lambda **kwargs: backend_factory(
            **kwargs, enable_observability=True
        ),
        device=args.device,
        local_files_only=not args.allow_download,
        run_index=args.run_index,
        run_id=args.run_id,
        warm_state_filter=args.warm_state,
        concurrency_filter=args.concurrency,
        prompt_tokens_filter=args.prompt_tokens,
        output_tokens_filter=args.output_tokens,
        now=now,
    )
    print(f"wrote {rows} benchmark trial rows to {args.result_dir / 'raw.jsonl'}")
    return 0


def _run_calibration_produce(
    args: argparse.Namespace,
    *,
    environment_probe: EnvironmentProbe,
    now: Callable[[], datetime],
) -> int:
    from .calibration import BF16_ORACLE_KIND, FP32_ORACLE_KIND
    from .hf_calibration import produce_hf_oracle

    artifact_kind = FP32_ORACLE_KIND if args.role == "fp32" else BF16_ORACLE_KIND
    manifest = produce_hf_oracle(
        artifact_kind=artifact_kind,
        prompts_path=args.prompts,
        manifest_path=args.manifest,
        sidecar_path=args.sidecar,
        repo_root=args.repo_root,
        device=args.device,
        local_files_only=not args.allow_download,
        created_at=now(),
        environment_probe=environment_probe,
    )
    print(
        f"wrote {manifest['artifact_kind']} ({len(manifest['cases'])} cases) "
        f"to {args.manifest}"
    )
    return 0


def _run_calibration_validate_manifest(args: argparse.Namespace) -> int:
    from .calibration import load_calibration_manifest, verify_calibration_artifact

    manifest = load_calibration_manifest(args.manifest)
    tensor_count = verify_calibration_artifact(
        manifest=manifest,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        expected_kind=args.expected_kind,
    )
    print(
        f"valid {manifest['artifact_kind']} calibration artifact: "
        f"{len(manifest['cases'])} cases, {tensor_count} tensors"
    )
    return 0


def _run_calibration_oracles(
    args: argparse.Namespace, *, now: Callable[[], datetime]
) -> int:
    from .calibration import load_calibration_manifest, sha256_file, write_json_exclusive
    from .oracle_calibration import compare_hf_oracles

    if args.output.exists():
        raise CalibrationError(f"refusing to overwrite existing report: {args.output}")
    report = compare_hf_oracles(
        fp32_manifest=load_calibration_manifest(args.fp32_manifest),
        fp32_manifest_path=args.fp32_manifest,
        bf16_manifest=load_calibration_manifest(args.bf16_manifest),
        bf16_manifest_path=args.bf16_manifest,
        repo_root=args.repo_root,
        created_at=now(),
    )
    write_json_exclusive(args.output, report)
    print(
        f"wrote HF oracle calibration report ({report['status']}) to {args.output}; "
        f"raw_sha256={sha256_file(args.output)}; e0_candidate_evidence=false"
    )
    return 0 if report["status"] == "pass" else 1


def _run_calibration_compare(
    args: argparse.Namespace, *, now: Callable[[], datetime]
) -> int:
    from .calibration import (
        compare_calibrations,
        load_calibration_manifest,
        sha256_file,
        write_json_exclusive,
    )
    from .oracle_calibration import load_oracle_calibration_report

    if args.output.exists():
        raise CalibrationError(f"refusing to overwrite existing report: {args.output}")
    report = compare_calibrations(
        fp32_manifest=load_calibration_manifest(args.fp32_manifest),
        fp32_manifest_path=args.fp32_manifest,
        bf16_manifest=load_calibration_manifest(args.bf16_manifest),
        bf16_manifest_path=args.bf16_manifest,
        oracle_calibration_report=load_oracle_calibration_report(args.oracle_report),
        oracle_calibration_report_path=args.oracle_report,
        candidate_manifest=load_calibration_manifest(args.candidate_manifest),
        candidate_manifest_path=args.candidate_manifest,
        repo_root=args.repo_root,
        created_at=now(),
    )
    write_json_exclusive(args.output, report)
    print(
        f"wrote native E0 correctness report ({report['status']}) to {args.output}; "
        f"raw_sha256={sha256_file(args.output)}"
    )
    return 0 if report["status"] == "pass" else 1


def _run_calibration_validate_oracles(args: argparse.Namespace) -> int:
    from .calibration import load_calibration_manifest, sha256_file
    from .oracle_calibration import (
        load_oracle_calibration_report,
        replay_validate_oracle_report,
    )

    report = load_oracle_calibration_report(args.report)
    replay_validate_oracle_report(
        report=report,
        fp32_manifest=load_calibration_manifest(args.fp32_manifest),
        fp32_manifest_path=args.fp32_manifest,
        bf16_manifest=load_calibration_manifest(args.bf16_manifest),
        bf16_manifest_path=args.bf16_manifest,
        repo_root=args.repo_root,
    )
    print(
        f"valid replayed HF oracle calibration report: {args.report}; "
        f"raw_sha256={sha256_file(args.report)}; e0_candidate_evidence=false"
    )
    return 0


def _run_calibration_validate_report(args: argparse.Namespace) -> int:
    from .calibration import (
        load_calibration_manifest,
        load_correctness_report,
        replay_validate_correctness_report,
        sha256_file,
    )
    from .oracle_calibration import load_oracle_calibration_report

    report = load_correctness_report(args.report)
    replay_validate_correctness_report(
        report=report,
        fp32_manifest=load_calibration_manifest(args.fp32_manifest),
        fp32_manifest_path=args.fp32_manifest,
        bf16_manifest=load_calibration_manifest(args.bf16_manifest),
        bf16_manifest_path=args.bf16_manifest,
        oracle_calibration_report=load_oracle_calibration_report(args.oracle_report),
        oracle_calibration_report_path=args.oracle_report,
        candidate_manifest=load_calibration_manifest(args.candidate_manifest),
        candidate_manifest_path=args.candidate_manifest,
        repo_root=args.repo_root,
    )
    print(
        f"valid replayed native E0 correctness report: {args.report}; "
        f"raw_sha256={sha256_file(args.report)}"
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory | None = None,
    provenance_factory: ProvenanceFactory = collect_fixture_provenance,
    repository_validator: Callable[
        [dict[str, object], Path], None
    ] = validate_fixture_against_repository,
    environment_probe: EnvironmentProbe = probe_primary_environment,
    now: Callable[[], datetime] = utc_now,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    factory = backend_factory or _default_backend_factory
    try:
        if args.command == "generate":
            return _run_generate(
                args,
                backend_factory=factory,
                provenance_factory=provenance_factory,
                environment_probe=environment_probe,
                now=now,
            )
        if args.command == "validate":
            return _run_validate(args, repository_validator=repository_validator)
        if args.command == "benchmark":
            return _run_benchmark(args, backend_factory=factory, now=now)
        if args.command == "calibrate-produce":
            return _run_calibration_produce(
                args, environment_probe=environment_probe, now=now
            )
        if args.command == "calibrate-validate-manifest":
            return _run_calibration_validate_manifest(args)
        if args.command == "calibrate-oracles":
            return _run_calibration_oracles(args, now=now)
        if args.command == "calibrate-compare":
            return _run_calibration_compare(args, now=now)
        if args.command == "calibrate-validate-oracles":
            return _run_calibration_validate_oracles(args)
        if args.command == "calibrate-validate-report":
            return _run_calibration_validate_report(args)
        parser.error(f"unsupported command: {args.command}")
    except (FixtureError, CalibrationError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2
