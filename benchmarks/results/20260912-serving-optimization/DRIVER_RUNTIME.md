# Private runtime after the host NVIDIA update

The benchmark host changed after Round12. Unattended upgrades ran on 2026-09-12 from 06:43:31 to 06:44:11 KST, replacing NVIDIA user libraries 580.173.02 with 580.178.04. The running kernel module remains 580.173.02. A read-only audit at 10:06 KST recorded default NVML exit 18 and the installed package/library paths in `raw/driver-incident-host-audit.json`. This later failure is separate from Round12's earlier vLLM output-reference failure.

Matching compute and GL libraries were extracted under `/tmp/riley-opt-260912/` from the [official Ubuntu snapshot service](https://snapshot.ubuntu.com/). The helpers verified the Ubuntu archive signature, signed package-index digest, exact package version/architecture, and each downloaded package's size and SHA256. No package installation, kernel-module reload, global loader change or reboot was performed.

| Private runtime | Packages, version 580.173.02-0ubuntu0.22.04.1 | Evidence |
| --- | --- | --- |
| `driver580173-runtime-20260901` | libnvidia-compute-580, nvidia-utils-580 | `raw/driver580173-runtime-20260901/receipt.json` |
| `driver580173-gui-runtime-20260901` | libnvidia-gl-580, libnvidia-cfg1-580, libnvidia-extra-580 | `raw/driver580173-gui-runtime-20260901/receipt.json` |

Regular-file hashes and symlink targets are retained in the receipts. Extracted libraries and package archives remain remote; local evidence retains manifests and verification logs. Explicit child environments choose these directories. Default host NVML still fails; private NVML succeeds and reports the bound RTX 4090 UUID `GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0` and driver 580.173.02.

`raw/driver-runtime-gui-probe-v2/completion.json` proves two new offscreen contexts: EGL and GLX. Both reported the NVIDIA RTX 4090 and 580.173.02, mapped the pinned private vendor libraries, and completed cleanup without creating a window. Blender PIDs 4177821, 4177822 and 4177823 remained unchanged, with their recorded process birth, command, working directory, GUI environment, restoration tag and listening ports verified before and after. These processes retain old libraries mapped before the package replacement.

The first context-check attempt successfully created both contexts but failed its final helper hash check because a string path was passed to a Path-only digest helper. Its outputs and failure remain in `raw/driver-runtime-gui-probe/`. The corrected v2 reran the complete check. The earlier compute-package preparation attempt used a snapshot already containing newer packages; it failed before extraction or GPU execution and is also preserved.

Round13 subsequently paused and restored the three full Blender file/script sessions after the first token measurement stopped. All three relaunched processes loaded the pinned private GL/core and compute libraries; commands, GUI environment, process birth and ports were verified. New PIDs are 2234550/2234607/2234700, ports9876/9911/9887. The receipt is `raw/blender-round13/verified.json`, SHA256 `9bf3b9d644015fec626122271d973e63fcab70f4ee9427320757d009375e1fb0`. The round14 helper pins that exact predecessor snapshot/runtime/receipt and rechecks current maps before any next stop. Only these three authorized successors are in scope.

Correctness diagnostics with Blender retained may use this explicitly recorded runtime. New performance measurements need fresh paired controls, the same GUI-retained 512 MiB idle envelope, no foreign CUDA compute, and a start temperature at or below 48°C. Historical timing receipts retain their original environment and are not relabeled as measurements under the new runtime.
