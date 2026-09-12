"""Pinned diagnostic vLLM adapter; invoked only by the explicit measurement runner."""
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    pinned = json.loads((root / 'environment.json').read_text())
    candidate = json.loads((root / 'candidate.json').read_text())
    sys.path.insert(0, str(Path(candidate['source_root']) / 'benchmarks/lanes/vllm'))
    from riley_vllm_benchmark import adapter, cli
    original = adapter._llm_options

    def options(*, max_num_seqs):
        assert max_num_seqs == 1
        result = original(max_num_seqs=1)
        result.update(model=pinned['vllm_checkpoint'], tokenizer=pinned['vllm_checkpoint'],
                      max_model_len=160, max_num_batched_tokens=128,
                      gpu_memory_utilization=0.3, seed=0)
        return result

    # Source-pinned runtime options differ from the legacy 8192-position lane.
    # The model identity and artifact verifier remain unchanged.
    adapter._llm_options = options
    raise SystemExit(cli.main(sys.argv[1:]))


if __name__ == '__main__':
    main()
