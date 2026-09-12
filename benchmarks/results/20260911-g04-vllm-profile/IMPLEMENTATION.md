# Explicit SmolLM2 P128 numerical profile

The numerical mismatch is reproduced and resolved in the native full graph. Promote it as an explicit, bounded profile rather than silently changing the existing HF/canonical graph contract.

- Keep existing graph APIs/default numerics and canonical tests unchanged.
- Add a versioned CUDA numerical-profile selector and runtime cold configuration.
- Include the selector in registry identity. Require SmolLM2-135M geometry, SM89, CUDA runtime 13.0 and cuBLASLt 13.1.1.
- Admit only a 128-token prompt and at most 32 outputs for this profile. Unsupported graph/eager paths fail closed.
- Remove trace writes, file input, oracle KV and implicit profile redirection. Keep scheduler KV mappings and owned resource lifetimes.
- Preserve the complete existing profile as rollback. Do not describe changed numerics as bitwise equality with the old Riley eager path.
- Validate native token parity, lifecycle/error paths, HTTP CPU/GPU greedy and request bounds, then freeze source/binary provenance. No performance trial until the separate measurement gates pass.
