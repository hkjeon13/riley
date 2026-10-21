# Command-batch resource ledger CPU check

This exercises the real `command_batch_register_use` function without creating
a CUDA context. Build with CUDA headers/libraries and assertions enabled:

```bash
export TMPDIR=/your/efs/private-build-tmp
mkdir -p "$TMPDIR"
g++ -std=c++17 -O2 -pthread -DRILEY_LEDGER_INDEX_CHECK \
  -DRILEY_CUDA_ENABLE_CUBLAS_GEMM_PROBE=1 \
  -Ikernels/src -Ikernels/include -I/usr/local/cuda-12.4/include \
  benchmarks/tools/host-ledger/check.cpp \
  -L/usr/local/cuda-12.4/lib64 -L/usr/local/cuda-12.4/lib64/stubs \
  -lcuda -lcudart -o "$TMPDIR/check-ledger"
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 \
  "$TMPDIR/check-ledger"
```

The macro enables index-clear/reuse across four generations and full-table
termination checks, in addition to thread/stream ownership, capacity and
acquisition checks. Do not use `-DNDEBUG`: assertions execute the checked calls.
The reported duplicate-registration nanoseconds are a CPU microbenchmark,
not a server throughput measurement. GPU completion/release behavior is
covered separately by the observed GPU contract tests.
