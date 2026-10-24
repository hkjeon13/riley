# Vendored FlashInfer headers

Pinned subset of **FlashInfer 0.6.16.post3** (`flashinfer_python-0.6.16.post3`,
`flashinfer/data/` of the PyPI wheel), used only by `kernels/src/fi_attention.cu`
for Riley's batch-invariant D128 attention (`fixed-tile` FlashInfer paged
prefill; see `docs/plan-fixed-split-varlen-attention-20261023.md`).

- `include/flashinfer/`: the FlashInfer headers that translation unit includes
  (Apache License 2.0, `LICENSE`).
- `cccl/libcudacxx/include/`: the libcu++ headers they need (`cuda/cmath`
  and dependencies) from the CCCL copy shipped in the same wheel
  (Apache License 2.0 with LLVM Exceptions; each file carries its SPDX
  header, see https://llvm.org/LICENSE.txt).

The file set is exactly the `nvcc -M` closure of `fi_attention.cu` with CUDA
12.4; `MANIFEST.sha256` lists every file's SHA-256. The files are unmodified.
Only `fi_attention.cu` sees these include paths, so the rest of the kernels
keep the toolkit's own libcu++.

To update: install the new wheel, regenerate the closure with `nvcc -M`, copy
the files with their relative paths, refresh the manifest, and re-run the
attention GPU tests (batch invariance is part of Riley's contract).
