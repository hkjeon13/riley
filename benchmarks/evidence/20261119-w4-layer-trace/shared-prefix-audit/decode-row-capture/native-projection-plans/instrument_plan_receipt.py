"""Emit real prepared plans at the pinned layer-zero CUDA calls."""
from pathlib import Path
import sys


def inject(source, anchor, new):
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected source anchor: {anchor.splitlines()[0]}")
    return source.replace(anchor, new + anchor)


stage = Path(sys.argv[1])
target = stage / "src/crates/riley-runtime/src/llama/executor/dispatch.rs"
source = target.read_text()
helper = '''fn trace_projection_plan(
    trace: &Option<(&mut LongTrace, &CudaPinnedHostBuffer)>,
    layer: usize,
    name: &str,
    plan: &PreparedLlamaGemm,
) -> LlamaBatchExecutorResult<()> {
    if layer != 0 {
        return Ok(());
    }
    if let Some((trace, _)) = trace.as_ref() {
        let PreparedLlamaGemm::Canonical(canonical) = plan else {
            return Err(long_trace_error());
        };
        let meta = canonical.algorithm_metadata();
        let (m, n, k) = meta.dimensions();
        let (major, minor) = meta.compute_capability();
        eprintln!(
            "W4_NATIVE_PROJECTION {name} position={} token={} m={m} n={n} k={k} backend={} id={} tile={} stages={} split_k={} reduction={} cta_swizzling={} custom_option={} workspace={} flags={} deterministic={} sm={major}.{minor} cuda={} cublaslt={}",
            trace.position, trace.token_id, meta.backend_id(), meta.algorithm_id(),
            meta.tile_id(), meta.stages_id(), meta.split_k(), meta.reduction_scheme(),
            meta.cta_swizzling(), meta.custom_option(), meta.workspace_bytes(),
            meta.numerical_implementation_flags(), meta.deterministic(),
            meta.runtime_version(), meta.cublaslt_version()
        );
    }
    Ok(())
}

'''
source = inject(source, "fn trace_row<S: CudaExecutionStream + ?Sized>(", helper)
source = inject(
    source,
    "                execute_gemm_with_row_groups(\n                    gemms.output_projection_plan(),\n",
    '''                trace_projection_plan(&long_trace, layer_index, "o_proj", gemms.output_projection_plan())?;
''',
)
source = inject(
    source,
    "                    execute_gemm_with_row_groups_from(\n                        &mut gemms.down,\n",
    '''                    trace_projection_plan(&long_trace, layer_index, "down_proj", &gemms.down)?;
''',
)
target.write_text(source)
