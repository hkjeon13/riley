# Attempt V1: incomplete — vLLM readiness timeout

The controller terminated with exit1 when shared-p0-vllm did not become ready within the common300-second startup limit. Prior/control/split shared runs completed; no full comparison is available. vLLM loaded weights before termination but had not reached HTTP readiness. Its process exited−9 after the controller shutdown grace elapsed. Blender restoration succeeded.

Read-only observations found the vLLM process in D-state during initialization and host IO PSI some/full around24%/21% after termination. These observations are consistent with delayed initialization but do not identify its single cause. Memory PSI was approximately zero; no memory failure was established.

This archive contains verified, hash-bound failure metadata and controller/client snapshots. Raw responses remain on the remote tmpfs path recorded in the manifest. They were not independently semantically exported, and the three partial throughput estimates are not promoted or merged into a subsequent run.

V2 uses a common600-second readiness limit for every engine. Model, binaries, workload, numerical gates, measured request counts and timed-phase treatment are unchanged. All16 lanes are rerun under that predeclared policy; the original failure remains visible.
