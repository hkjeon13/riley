# Distribution Docker image verification (2026-10-29)

Image `deploy/docker/Dockerfile` built on `aws-a100` from source `d41d7dfc0` plus
the files of this commit: `riley-dist:cu124-d41d7dfc`, image id
`sha256:c341cb2da4d8…`, 2.34 GB. `cuobjdump --list-elf` on the image's `riley`
binary lists 18 kernels each for `sm_80`, `sm_86` and `sm_89`.

Verification ran on A100 GPU 6 under the holder lease (`verify_image.py`,
result `verify-report.json`):

- CLI: `help`, `presets`, `riley-checkpoint --help` and `--version` (with GPU) exit 0.
- Fresh download: `serve --preset qwen2.5-0.5b` into a new named volume
  downloaded from the Hub, became ready in 37.5 s and answered; a restart
  reused the volume (no download, ready in 14.9 s). Both stopped with exit 0.
- All 16 presets: the image served each checkpoint with the preset defaults
  (`--inference-profile native-bf16-varlen-v1`, planned weight bytes), and so
  did the measured host binary `riley-mc-b74361d3` of the same source. Two
  greedy prompts, 32 tokens each: outputs identical for 16 of 16 models.

Presets come from `make_presets.py`: every file of each measured checkpoint
manifest matched the Hub at the pinned commit (size; LFS SHA-256 for weights;
git blob id plus local SHA-256 for JSON files).

Not covered: running on `sm_86`/`sm_89` hardware, and gated repositories (all
16 presets are open repositories).
