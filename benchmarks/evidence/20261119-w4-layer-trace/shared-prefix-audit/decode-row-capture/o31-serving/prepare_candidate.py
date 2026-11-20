"""Make an isolated A/B binary with plan 31 in the tested native O-projection geometry."""
from pathlib import Path
import sys


def replace(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f"unexpected source anchor: {anchor.splitlines()[0]}")
    return text.replace(anchor, replacement)


root = Path(sys.argv[1])
dispatch = root / "src/kernels/src/gemm.cu"
code = dispatch.read_text()
if "#include <cstdio>" not in code:
    code = replace(code, "#include <cstdlib>", "#include <cstdio>\n#include <cstdlib>")
anchor = "bool select_anchor_override(RileyCudaGemmPlan* plan) noexcept;"
code = replace(code, anchor, """bool w4_attention_o31_requested(const RileyCudaGemmPlan* plan) noexcept {
  const char* enabled = std::getenv("RILEY_W4_ATTENTION_O31");
  return enabled != nullptr && enabled[0] == '1' && enabled[1] == '\\0' &&
      plan->config.k == 3584 && plan->config.n == 3584 &&
      plan->config.epilogue == RILEY_CUDA_GEMM_EPILOGUE_NONE &&
      plan->config.reserved[0] == 1;
}

RileyCudaStatus choose_w4_attention_o31(
    RileyCudaGemmPlan* plan, RileyCudaErrorInfo* error) noexcept {
  const auto unsupported = [&]() noexcept {
    return set_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RILEY_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RILEY_CUDA_ERROR_STAGE_PREPARE,
                     "prepare isolated W4 O projection",
                     "plan 31 is unavailable for this CUDA configuration");
  };
  constexpr uint64_t kBuckets[] = {
      1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176,
      192, 208, 224, 240, 256,
  };
  bool supported_m = false;
  for (const uint64_t m : kBuckets) {
    supported_m |= plan->config.m == m;
  }
  const auto& info = plan->algorithm_info;
  if (!supported_m || info.compute_capability_major != 8 ||
      info.compute_capability_minor != 0 || info.runtime_version != 12040 ||
      info.cublaslt_version != 120405) {
    return unsupported();
  }
  cublasLtMatmulAlgo_t algorithm{};
  if (cublasLtMatmulAlgoInit(plan->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                              CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                              CUDA_R_16BF, 31, &algorithm) != CUBLAS_STATUS_SUCCESS) {
    return unsupported();
  }
  constexpr cublasLtMatmulAlgoConfigAttributes_t kFields[] = {
      CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  };
  constexpr uint32_t kValues[] = {15, 18, 1, 0, 0, 0};
  for (size_t i = 0; i < sizeof(kValues) / sizeof(kValues[0]); ++i) {
    if (cublasLtMatmulAlgoConfigSetAttribute(&algorithm, kFields[i], &kValues[i],
                                          sizeof(kValues[i])) != CUBLAS_STATUS_SUCCESS) {
      return unsupported();
    }
  }
  cublasLtMatmulHeuristicResult_t checked{};
  if (cublasLtMatmulAlgoCheck(plan->handle, plan->operation, plan->weight_layout,
                             plan->input_layout, plan->output_layout, plan->output_layout,
                             &algorithm, &checked) != CUBLAS_STATUS_SUCCESS ||
      checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize != 0 ||
      !record_algorithm(plan, algorithm, checked.workspaceSize)) {
    return unsupported();
  }
  std::fprintf(stderr,
               "W4_O31_PLAN m=%llu id=%d tile=%u stages=%u split_k=%u reduction=%u workspace=%llu flags=%llu\\n",
               static_cast<unsigned long long>(info.m), info.algorithm_id, info.tile_id,
               info.stages_id, info.split_k, info.reduction_scheme,
               static_cast<unsigned long long>(info.workspace_bytes),
               static_cast<unsigned long long>(info.numerical_implementation_flags));
  return RILEY_CUDA_STATUS_SUCCESS;
}

""" + anchor)
anchor = """    status = anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)"""
code = replace(code, anchor, """    status = w4_attention_o31_requested(plan)
                 ? choose_w4_attention_o31(plan, error)
                 : anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)""")
dispatch.write_text(code)

driver = root / "tools/run_accuracy.py"
s = driver.read_text()
s = replace(s, '''mode = sys.argv[1]
if mode != "riley":
    raise SystemExit("unknown phase")''', '''arm = sys.argv[1]
if arm not in {"baseline", "candidate"}:
    raise SystemExit("unknown arm")
mode = "riley"''')
s = replace(s, "OUT = RUN / mode", "OUT = RUN / arm")
s = replace(s, 'env.pop("RILEY_TIMING", None)', '''env.pop("RILEY_TIMING", None)
env.pop("RILEY_W4_ATTENTION_O31", None)
if arm == "candidate":
    env["RILEY_W4_ATTENTION_O31"] = "1"''')
s = replace(s, 'receipt = {"mode": mode,', 'receipt = {"mode": arm,')
driver.write_text(s)
