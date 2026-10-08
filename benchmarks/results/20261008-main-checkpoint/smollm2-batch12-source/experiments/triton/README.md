# Triton experiments

This directory is an optional prototype environment, not a production dependency.
It is excluded from the Cargo workspace and the release container. Production
crates must not import, link, invoke, or generate code from it, and `triton` is not
a production Cargo feature.

An experiment may compare an idea with a native implementation and emit explicit
JSON, CSV/JSONL, or safetensors evidence. Moving a result into production requires
a separate architecture decision and a CUDA C++/native implementation with its
own correctness and performance gates. PR 02 contains no Triton runtime or JIT.
