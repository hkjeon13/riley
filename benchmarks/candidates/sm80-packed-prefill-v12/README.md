# SM80 packed prefill v12

Layered patch against immutable v11 E, not clean Git HEAD. Baseline binary SHA256: `2fb223ba679556fc3797fdc745ea5597744352c9de48f71396cf9a324c4e44a4`. Preserve preceding v11 and earlier layers. Three source files changed; files.json records both sides.

Packs adjacent Q/K/V/O prefill token groups under independently qualified native-M128 accumulation families. Q original65..130/merged257..576, K/V original94..128/merged161..512, O original65..128/merged161..576. Unknown extents retain original dispatch. Existing private mode11, cold workspaces, context/environment guards, Graph decode and arithmetic are retained. Actual Qwen Q plan `hf_separate_query` is cold prepared.

Execution and byte gates completed on physical A100 GPU6/7 with original persistent holders. Candidate binary SHA256: `e4a4f4cafe1d005b80e4a0dde9aed249fca3b42f2d721408895e0ce4cdca32c9`. Final root: `/mnt/efs/nlp/psyche/riley-packed-v12b-20261020`.

Five fresh matched repetitions: C16 throughput697.291→696.455tok/s(-0.12%), TTFTp95 94.331→88.822ms(-5.84%); C32 throughput897.190→914.686(+1.95%), TTFTp95 290.554→274.099ms(-5.66%). Stock1125.464/1666.432tok/s. C16 repeat4 TPOTp95/p99 regressed2.556%/4.501%; the all-pair2%tail gate is NOT satisfied. Preserve this candidate as a measured prefill-TTFT improvement; do not claim a general throughput or tail promotion. Original v11 binary remains intact.

523 CPU tests and2464 native/HF BF16 files passed. Actual NVTX selection of all three packed shape families and decode Graph execution was verified. Final raw SSE/identity/resource audits passed. See `docs/qwen-packed-projection-v12-and-vllm-20261020.md` and evidence directory for scope and limits.
