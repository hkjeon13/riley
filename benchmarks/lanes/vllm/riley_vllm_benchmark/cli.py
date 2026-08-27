"""Console entry point for the strict vLLM benchmark adapter."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .adapter import AdapterError, Backend, load_default_backend, run_benchmark


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riley-vllm-benchmark",
        description="Pinned vLLM 0.27.1 single-cell benchmark adapter",
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=_positive_int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warm-state", choices=("cold", "warm"), required=True)
    parser.add_argument("--concurrency", type=_positive_int, required=True)
    parser.add_argument("--prompt-tokens", type=_positive_int, required=True)
    parser.add_argument("--output-tokens", type=_positive_int, required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the exact immutable revision to be fetched (default: cache-only)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: Callable[..., Backend] = load_default_backend,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        row_count = run_benchmark(
            matrix_path=args.matrix,
            prompts_path=args.prompts,
            result_dir=args.result_dir,
            run_index=args.run_index,
            run_id=args.run_id,
            warm_state=args.warm_state,
            concurrency=args.concurrency,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            allow_download=args.allow_download,
            backend_factory=backend_factory,
        )
    except (AdapterError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {row_count} vLLM trial rows to {args.result_dir / 'raw.jsonl'}")
    return 0
