# Hardware counter collection unavailable

Nsight Compute full-set collection of the first four native kernel launches failed with ERR_NVGPUCTRPERM. The process user lacks NVIDIA GPU performance-counter access. A noninteractive sudo check also failed because a password is required. No driver setting was changed. The lifecycle restored all three Blender processes.

All TIME values in ncu.log are invalid for performance comparison: instrumentation was active while counter collection failed. Native CHECK lines do not substitute for hardware counters. No HBM bandwidth, L2 hit rate, achieved occupancy or stall reason was measured in this attempt.

The earlier Nsight Systems per-kernel times and CUDA static resource attributes remain the available evidence. They locate regression in PV but do not prove memory-bandwidth saturation. Do not repeat this identical counter request without an external permission change. Continue work using available kernel timings, CPU traces and native correctness tests; the serving objective is not blocked solely by missing counters.
