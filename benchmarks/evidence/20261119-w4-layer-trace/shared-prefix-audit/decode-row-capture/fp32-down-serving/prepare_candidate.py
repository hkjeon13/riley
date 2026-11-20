"""Make an isolated A/B binary with FP32 partial reduction in the tested down projection."""
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
code = replace(code, "RileyCudaStatus validate_config_for_epilogue(", """bool w4_down_fp32_requested(const RileyCudaGemmConfig& config) noexcept {
  const char* enabled = std::getenv("RILEY_W4_DOWN_FP32");
  return enabled != nullptr && enabled[0] == '1' && enabled[1] == '\\0' &&
      config.k == 18944 && config.n == 3584 &&
      config.epilogue == RILEY_CUDA_GEMM_EPILOGUE_NONE &&
      config.flags == RILEY_CUDA_GEMM_FLAG_ALLOW_COMPUTE_TYPE_SPLIT_K;
}

RileyCudaStatus validate_config_for_epilogue(""")
code = replace(code, """        config->flags != kHfCompatibleBiasAllReductionsFlags)) ||""", """        config->flags != kHfCompatibleBiasAllReductionsFlags) &&
       !w4_down_fp32_requested(*config)) ||""")
anchor = "bool select_anchor_override(RileyCudaGemmPlan* plan) noexcept;"
code = replace(code, anchor, """RileyCudaStatus choose_w4_down_fp32(
    RileyCudaGemmPlan* plan, RileyCudaErrorInfo* error) noexcept {
  const auto unsupported = [&]() noexcept {
    return set_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RILEY_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RILEY_CUDA_ERROR_STAGE_PREPARE,
                     "prepare isolated W4 down projection",
                     "the FP32 split-K variant of plan 21 is unavailable for this CUDA configuration");
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
                              CUDA_R_16BF, 21, &algorithm) != CUBLAS_STATUS_SUCCESS) {
    return unsupported();
  }
  constexpr cublasLtMatmulAlgoConfigAttributes_t kFields[] = {
      CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  };
  constexpr uint32_t kValues[] = {18, 18, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0};
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
      checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize != 43008 * plan->config.m ||
      checked.workspaceSize > plan->config.max_workspace_bytes ||
      !record_algorithm(plan, algorithm, checked.workspaceSize)) {
    return unsupported();
  }
  std::fprintf(stderr,
               "W4_DOWN_FP32_PLAN m=%llu id=%d tile=%u stages=%u split_k=%u reduction=%u workspace=%llu flags=%llu\\n",
               static_cast<unsigned long long>(info.m), info.algorithm_id, info.tile_id,
               info.stages_id, info.split_k, info.reduction_scheme,
               static_cast<unsigned long long>(info.workspace_bytes),
               static_cast<unsigned long long>(info.numerical_implementation_flags));
  return RILEY_CUDA_STATUS_SUCCESS;
}

""" + anchor)
anchor = """    status = anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)"""
code = replace(code, anchor, """    status = w4_down_fp32_requested(plan->config)
                 ? choose_w4_down_fp32(plan, error)
                 : anchor != nullptr
                 ? select_anchored_deterministic_algorithm(plan, anchor, error)""")
dispatch.write_text(code)

gemm = root / "src/crates/riley-cuda/src/gemm.rs"
code = gemm.read_text()
code = replace(code, "        CudaGemmReductionPolicy::StrictNoSplitV1 => false,",
               """        CudaGemmReductionPolicy::W4ComputeTypeSplitKV1 => {
            split_k == 3 && scheme == REDUCTION_SCHEME_COMPUTE_TYPE
        }
        CudaGemmReductionPolicy::StrictNoSplitV1 => false,""")
code = replace(code, "    HfCompatibleBiasSplitKV1,\n}", """    HfCompatibleBiasSplitKV1,
    /// Isolated W4 comparison using Split-K 3 with FP32 workspace partials.
    W4ComputeTypeSplitKV1,
}""")
code = replace(code, '            Self::HfCompatibleBiasSplitKV1 => "hf-compatible-bias-split-k-v1",',
               '            Self::HfCompatibleBiasSplitKV1 => "hf-compatible-bias-split-k-v1",\n'
               '            Self::W4ComputeTypeSplitKV1 => "w4-compute-type-split-k-v1",')
code = replace(code, "            Self::HfCompatibleBiasSplitKV1 => 7,",
               "            Self::HfCompatibleBiasSplitKV1 => 7,\n"
               "            Self::W4ComputeTypeSplitKV1 => 4,")
gemm.write_text(code)

ffi = root / "src/crates/riley-cuda/src/ffi.rs"
code = ffi.read_text()
flag_check = "if flags & !(GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K | GEMM_FLAG_ALLOW_INPLACE_SPLIT_K) != 0 {"
new_check = """if flags != GEMM_FLAG_ALLOW_COMPUTE_TYPE_SPLIT_K
            && flags & !(GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K | GEMM_FLAG_ALLOW_INPLACE_SPLIT_K) != 0
        {"""
start = code.index("impl GemmPlanHandle {")
end = code.index("    fn as_ptr(", start)
handle_methods = code[start:end]
if handle_methods.count(flag_check) != 2:
    raise ValueError("expected both GEMM creation paths")
ffi.write_text(code[:start] + handle_methods.replace(flag_check, new_check) + code[end:])

forward = root / "src/crates/riley-runtime/src/llama/forward.rs"
code = forward.read_text()
code = replace(code, """    let down_plan = prepare(
        sequence,
        hidden,
        intermediate,
        policies.down(),""", """    let down_policy = if hidden == 3584
        && intermediate == 18944
        && std::env::var_os("RILEY_W4_DOWN_FP32").as_deref() == Some(std::ffi::OsStr::new("1"))
    {
        CudaGemmReductionPolicy::W4ComputeTypeSplitKV1
    } else {
        policies.down()
    };
    let down_plan = prepare(
        sequence,
        hidden,
        intermediate,
        down_policy,""")
forward.write_text(code)

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
env.pop("RILEY_W4_DOWN_FP32", None)
if arm == "candidate":
    env["RILEY_W4_DOWN_FP32"] = "1"''')
s = replace(s, 'receipt = {"mode": mode,', 'receipt = {"mode": arm,')
driver.write_text(s)
