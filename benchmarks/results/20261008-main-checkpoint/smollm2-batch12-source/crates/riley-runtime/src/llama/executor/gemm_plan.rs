//! Cold preparation of anchored GEMM variants for Llama batch shapes.
//!
//! The enclosing executor retains the shared forward owner and all execution
//! resources. This component owns only exact dense-row plans and their
//! matching GEMM handles for optional smaller prepared shapes.

use std::mem;

use riley_cuda::{CudaContext, CudaErrorKind};
use riley_model::LoadedModel;

use super::super::LlamaExecutionPlan;
use super::super::forward::{GemmPlans, LlamaForwardError, PreparedLlamaForward};
use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult};
use super::shape::{LlamaBatchShapePolicy, validate_shape_buckets};

/// One exact dense-row plan and GEMM set sharing the enclosing owner's
/// uploaded weights, maximum-size graph buffers, and paged KV allocations.
pub(in crate::llama) struct PreparedLlamaBatchShape {
    pub(in crate::llama) dense_rows: usize,
    pub(in crate::llama) plan: LlamaExecutionPlan,
    pub(in crate::llama) gemms: GemmPlans,
}

impl PreparedLlamaBatchShape {
    pub(in crate::llama) fn close(self) -> LlamaBatchExecutorResult<()> {
        self.gemms.close().map_err(LlamaBatchExecutorError::Forward)
    }
}

/// Cold-prepares optional exact dense-row variants below the shared maximum.
///
/// The maximum shape is already owned by `forward`. Unsupported anchored
/// smaller variants are intentionally absent so callers fall back to the next
/// prepared shape or that maximum owner.
pub(in crate::llama) fn prepare_shape_variants(
    model: &LoadedModel,
    context: &CudaContext,
    forward: &PreparedLlamaForward,
    shape_policy: LlamaBatchShapePolicy,
    shape_buckets: &[usize],
) -> LlamaBatchExecutorResult<Box<[PreparedLlamaBatchShape]>> {
    if shape_policy == LlamaBatchShapePolicy::FixedMaximum {
        return Ok(Vec::new().into_boxed_slice());
    }
    let maximum_rows = forward.plan.sequence_length();
    validate_shape_buckets(shape_buckets, maximum_rows)?;
    let variant_count = shape_buckets.len() - 1;
    let mut variants: Vec<PreparedLlamaBatchShape> = Vec::new();
    variants.try_reserve_exact(variant_count).map_err(|_| {
        LlamaBatchExecutorError::HostAllocation {
            resource: LlamaBatchExecutorResource::HostWorkspace,
            requested_bytes: u64::try_from(variant_count)
                .unwrap_or(u64::MAX)
                .saturating_mul(mem::size_of::<PreparedLlamaBatchShape>() as u64),
        }
    })?;
    for &dense_rows in &shape_buckets[..variant_count] {
        let (plan, gemms) = match forward.prepare_batch_shape_variant(model, context, dense_rows) {
            Ok(prepared) => prepared,
            Err(error) if is_anchored_gemm_not_supported(&error) => {
                // The maximum shape remains the exact owner. Never substitute
                // an M-specific heuristic when its anchored reduction topology
                // cannot execute; dispatch will use the next available bucket
                // or the fixed maximum plan instead.
                continue;
            }
            Err(error) => {
                for variant in variants {
                    let _ = variant.close();
                }
                return Err(LlamaBatchExecutorError::Forward(error));
            }
        };
        variants.push(PreparedLlamaBatchShape {
            dense_rows,
            plan,
            gemms,
        });
    }
    Ok(variants.into_boxed_slice())
}

fn is_anchored_gemm_not_supported(error: &LlamaForwardError) -> bool {
    matches!(
        error,
        LlamaForwardError::Cuda { source, .. }
            if source.kind() == CudaErrorKind::NotSupported
                && source.operation() == "prepare anchored CUDA GEMM plan"
    )
}
