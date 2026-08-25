# CUDA fault-injection release evidence

The destructive CUDA fault cases run only on an authorized remote GPU host.
`ci/verify_python_free_gpu_runtime.sh` remains Python-free and writes raw logs;
it does not write or approve its own release attestation.

The raw directory is a closed contract. In addition to the existing host and
memory GPU evidence it contains:

- `memory-fault-test-list.txt`, with exactly the parent harness and child test;
- `memory-fault-tests.log`, with one spawn/start/pass/join marker sequence for
  each of the four reviewed cases and a distinct child PID for each case;
- `memory-fault-test-binary.sha256`;
- `release-binary.sha256` for `target/release/rustinfer`; and
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
  --release-binary /absolute/path/to/rustinfer \
  --release-bundle /absolute/path/to/rustinfer.tar.gz \
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

The source archive must be the uncompressed output of
`git archive --format=tar HEAD`: its SHA-256 must match `environment.txt` and
its PAX `comment` must equal the supplied full Git revision. The build image ID
must likewise match the immutable `gpu_image_id` captured by the remote runner.

Only after all raw checks pass does the tool create a deterministic tar and a
`rustinfer.release-gate-attestation.v1` report with the exact
`cuda-fault-injection` check set consumed by the final candidate gate. The
attestation's `raw_evidence_sha256` is computed from the newly created tar; a
caller-supplied passing JSON file is never accepted as evidence.

CPU-only regression tests are:

```sh
python3 -m unittest discover -s ci/release -p 'test_cuda_fault_evidence.py' -v
sh -n ci/verify_python_free_gpu_runtime.sh
```
