from pathlib import Path
import subprocess
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
files=['crates/riley-cuda/build.rs','crates/riley-cuda/src/ffi.rs','crates/riley-cuda/src/graph_resources.rs','crates/riley-cuda/tests/v4_shared_recorder_gpu.rs','crates/riley-runtime/src/llama/executor/config.rs','crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-runtime/src/llama/variable_session.rs','crates/riley-scheduler/src/authority.rs','crates/riley-scheduler/src/config.rs','crates/riley-scheduler/src/execution.rs','crates/riley-scheduler/src/scheduler.rs','crates/riley-scheduler/tests/v3_shared_owned_gpu.rs','crates/riley-server/src/engine.rs','crates/riley-server/src/main.rs','kernels/include/riley_cuda.h','kernels/src/decode_shared16_result.cuh','kernels/src/ffi_internal.hpp','kernels/src/gemm.cu','kernels/src/graph_numerics_precise.cu','kernels/src/graph_resources.cu','kernels/src/prefill_shape_model.cuh']
subprocess.run(['git','diff','--check'],cwd=r,check=True)
subprocess.run(['git','add','--',*files],cwd=r,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Connect sixteen-row graph execution through retained sessions and serving'],cwd=r,check=True)
