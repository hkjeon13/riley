# Split FFN graph integration — model validation complete; serving pending

The optional split backend keeps original M16 gate/up and down kernels for packed row counts below192 and uses separate M32 kernels at192 or above. The selector reads total rows only after the V7 packet validator accepts the request. Rust → C ABI → CUDA remains the runtime path. The existing unified adaptive backend and defaults are preserved.

Two additional model graphs (full and compact output) use the same retained parent ledger, stream, weight buffers and scratch. Buffered stage lists expand from4 to6, with existing slot-zero borrowing and slot-one memcpy rebinding. The future decode graph remains index3. Extra graph handles participate in drain/destroy and unknown-completion quarantine. CUDA graph/exec internal memory is additional and is not claimed free.

## Current evidence

- CUDA release server build exit0: `build-receipt.json` binds the source archive and server binary.
- Local CPU server tests:73 passed/1 ignored. These do not exercise CUDA.
- Actual SmolLM2 model test `ffn_split_matches_full_model_logits`:1 passed; **4,128,768 BF16 logits bytes exactly match the projection/M16 backend**. It includes191→192→191→193, small/large alternation and cached128-row prefixes, with three teacher-forced output steps per request. Both sessions close and tracked context allocations return to zero.
- Full-model memcheck:1 test passed, `ERROR SUMMARY: 0 errors`. Both model runs exited0 and Blender restoration completed. The seven-member evidence archive was checked for exit status, exact-logit count, sanitizer result, safe member paths and credential patterns; `verification.json` binds its SHA256.

The model test uses full logits and synchronous replay. It does not establish compact/buffered/rolling serving generation parity or throughput. The upcoming C32 matrix compares frozen unified adaptive prior, current M16 control, current split candidate and vLLM in both orders, with the established workload and measurement-client GC policy. It also records launch-to-readiness and readiness RSS/global GPU memory, without claiming isolated graph memory from those measurements.

The backend remains opt-in (`RILEY_PREFILL_FFN_SPLIT=1`, projection pipeline required, unified adaptive excluded). No serving performance result or overall goal qualification is claimed here.
