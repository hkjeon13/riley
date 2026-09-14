# Short prefill attention on the best ordinary serving path

## Evidence and scope

The speculative attention batch reduced kernel cost but did not beat ordinary split-FFN + rolling decode serving. Prefix cache import uses full pages bounded by `(prompt_len - 1) / 16`, leaving 1–16 inputs on a full hit. Existing ordinary attention processes every query separately below 32 rows. Reuse the exact ordered query-tile arithmetic on this best ordinary path, without introducing draft verification or losing rolling decode.

## Optimization batch

1. Stage-0 counts 2–16 reuse eight-query tiles; count 1 and other stages retain original arithmetic. Keep the wire directory stable, suppress duplicate/excess entries before execution.
2. Connect both small and large split-FFN captures to this dispatch, preserving projection and FFN selection. Other capture profiles remain unchanged.
3. Change the split-FFN catalog identity so cached/prepared state cannot claim the previous implementation identity.

## Validation and status

Native: 768 exact full-buffer cases × two graph replays, small memcheck/racecheck clean. SM89/90a/100a compile, only SM89 executes. Evidence: `benchmarks/results/20260914-short-prefill-native/README.md`.

CUDA model build passes. Extended model test exercises all 1–16 cached tails against unchanged projection-profile logits. Model execution and full-model memcheck both pass, including all cached-tail cases. The first lifecycle attempt failed before model execution because the controller expected the wrong binary receipt filename; Blender restoration succeeded. The corrected attempt uses a separate lifecycle directory.

Completed matched serving scope: frozen best split-FFN + rolling, candidate with the same policies, and vLLM 0.27.1; shared/unique, two reversed orders, C32, 64 warmup + 512 retained per lane, exact references and stop/cancel/recovery. Report throughput plus TTFT/TPOT and P95/P99; this screen is not high-concurrency or latest-vLLM qualification.

Do not claim promotion until real serving improves. Revert this batch to restore previous split-FFN dispatch and catalog identity. Runtime stays Rust → C ABI → CUDA. Multi-GPU/Hopper/Blackwell execution remains future validation.

## Completed screen: reject promotion

12 lanes and 6,912 warmup/retained responses pass verification; 4,608 Riley
responses match frozen references, with 64 stop/cancel/recovery checks each.
Throughput changes −2.24% shared / +1.25% unique. Shared P95/P99 worsen.
Production integration is reverted, with candidate.patch and immutable tested
sources retained in `benchmarks/results/20260914-short-prefill-serving-c32/`.
Final status is correctness passed, performance promotion rejected. No further
short-query micro-variants are selected. Next work is the current-vLLM baseline
refresh in PR-sized plan 15, then a new evidence-selected optimization area.
