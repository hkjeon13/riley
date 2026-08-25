# Hugging Face eager reference benchmark

Run `hf-transformers-cache-prime-55a397313acd-01`, independent run index 1; 30 result-schema-v1 rows. Greedy decode ignores EOS and emits the matrix's fixed output length. Each request records u32-le SHA-256 identities for both prompt and generated tokens. Metrics use 10 ms device-wide NVML GPU samples and recursive process-tree CPU-time deltas. Invoke each independent trial in a fresh process and a new result directory.
