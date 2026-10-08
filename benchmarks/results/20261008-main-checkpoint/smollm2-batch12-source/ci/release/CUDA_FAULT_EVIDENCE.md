# CUDA fault-injection release evidence

The destructive CUDA fault cases run only on an authorized remote GPU host.
`ci/verify_python_free_gpu_runtime.sh` remains Python-free and writes raw logs;
it does not write or approve its own release attestation.

The raw directory is a closed contract. In addition to the existing host and
memory GPU evidence it contains:

- `host-runtime-test-binary`, `memory-test-binary`, and
  `memory-fault-test-binary`, preserving the exact Linux x86-64 test ELF bytes
  that ran on the GPU;
- one `*.sha256` receipt for each preserved test ELF, with the original
  `target/debug/deps/*` path;
- `memory-fault-test-list.txt`, with exactly the parent harness and child test;
- `memory-fault-tests.log`, with one spawn/start/pass/join marker sequence for
  each of the four reviewed cases and a distinct child PID for each case;
- `host-runtime-*`, `memory-*`, and `memory-fault-*` `ldd`, `readelf`, and `nm`
  output whose two-line header binds the original artifact path and ELF digest;
- `release-binary.sha256` for `target/release/riley`; and
- `release-ldd.txt`, `release-readelf.txt`, and `release-nm.txt` for that exact
  production binary.

Every file except `SHA256SUMS` is covered by that manifest. The manifest paths
are sorted and no unlisted file, directory, symlink, device, or oversized file
is accepted. When Compute Sanitizer is enabled, both sanitizer logs are
required and must independently contain zero-error and zero-leak results.

After the remote run, use the CPU-only standard-library checker to bind the
raw directory to the final release artifacts:

```sh
python3 ci/release/check_cuda_fault_evidence.py \
  --evidence-dir /absolute/path/to/gpu-evidence \
  --source-revision "$SOURCE_REVISION" \
  --source-archive /absolute/path/to/source.tar \
  --build-image-id "$CUDA_BUILD_IMAGE_ID" \
  --release-binary /absolute/path/to/riley \
  --release-bundle /absolute/path/to/riley.tar.gz \
  --release-image-id "$RELEASE_IMAGE_ID" \
  --raw-evidence /absolute/path/to/cuda-fault-evidence.tar \
  --report /absolute/path/to/cuda-fault-report.json
```

`--build-image-id` is the immutable CUDA development image that compiled and
ran the fault harness. `--release-image-id` is the separately built minimal
runtime image recorded by the final candidate. They have different roles and
need not be equal. The checker requires the production binary recorded inside
the build-image evidence to be byte-identical to `--release-binary`, verifies
that binary as Linux x86-64 ELF, verifies the release bundle contains the same
binary and source revision, and rejects the test-only native symbol prefix in
both the supplied binary bytes and `nm` evidence.

The three test checksum receipts are not treated as claims. The checker hashes
the preserved ELF bytes, requires each digest and original path to match its
receipt, independently parses ELF64/x86-64 program and dynamic headers, and
cross-checks the direct `DT_NEEDED` inventory against `readelf`. The ordinary
test binaries must omit fault-injection symbols; the fault harness must contain
them. Reviewed libtest names and native success-marker strings must be present
in the corresponding ELF bytes. The list and execution logs must identify the
same original test path, enumerate exactly 8 host-runtime, 5 memory, or 2 fault
tests as applicable, and contain the exact passing summaries and runtime
markers rather than a self-declared `passed` line.

The remote environment is also replayed semantically: it must prove Rust/Cargo
1.85.0, CUDA 12.8, the CUDA-enabled release executable, the immutable build
image, and one visible RTX 4090/sm89 GPU. The two `nvidia-smi` views must agree
on index, canonical UUID, model, memory, and driver. The CUDA device marker is
cross-bound to that GPU, the leak iteration marker to `environment.txt`, and
the allocation marker must report all counters at zero.

The source archive must be the uncompressed output of
`git archive --format=tar HEAD`: its SHA-256 must match `environment.txt` and
its PAX `comment` must equal the supplied full Git revision. The build image ID
must likewise match the immutable `gpu_image_id` captured by the remote runner.

Only after all raw checks pass does the tool create a deterministic tar and a
`riley.release-gate-attestation.v1` report with the exact
`cuda-fault-injection` check set consumed by the final candidate gate. The
attestation's `raw_evidence_sha256` is computed from the newly created tar; a
caller-supplied passing JSON file is never accepted as evidence.

CPU-only regression tests are:

```sh
python3 -m unittest discover -s ci/release -p 'test_cuda_fault_evidence.py' -v
sh -n ci/verify_python_free_gpu_runtime.sh
```
