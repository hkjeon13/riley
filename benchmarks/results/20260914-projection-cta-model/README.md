# Projection CTA real-model gate

The explicit opt-in `RILEY_EXPERIMENT_PROJECTION_CTAS=1` switches only the adaptive pure-decode graph capture to the three separate-CTA projection kernels. Default captures retain the previous adaptive kernels. The flag is recorded in the serving controller launch metadata; the model test requires it explicitly.

`projection_ctas_match_serial_model_and_reclaim_pages` passed on the real SmolLM2 checkpoint. For each active capacity 16 and 32, it runs 32 requests in each of three modes: serial baseline, adaptive dependent decode, and dependent decode with early termination/cancellation. All generated token sequences match the corresponding baseline prefixes, lengths match, settlement checks pass, and allocations return to zero after all six runs. Decode-window counts are 27/26 at capacity 16 and 12/12 at capacity 32.

This test proves the stated workload's greedy sequence parity and resource cleanup, not full-logit bitwise equality or serving performance. Native partial-bit checks and bounded memcheck are recorded in the preceding native screen. No full-model sanitizer run was added.

The release test binary hash, build command and exact source hashes are archived. The serving binary is separately built with server,cuda and verified runtime ticket tests; its receipts are recorded when collected.

Serving candidate build SHA256: `3b2ebf01179305a425ca31ee7dc28245c05843f3597ff9e5c88e4e4d361312e3`. The first serving preflight failed at 16,340,492 KiB MemAvailable, below the unchanged 16 GiB threshold, before any serving server launched. Blender was restored. Subsequent investigation found earlier FFN/projection spools without a `riley-` name prefix. The completed `ffn-split-serving-c32-v1` spool (53 files, 1,148,166,061 bytes) was copied to persistent storage, verified by all-file hashes and fsynced before tmpfs removal. This changed available memory and enabled the second attempt to pass preflight.
