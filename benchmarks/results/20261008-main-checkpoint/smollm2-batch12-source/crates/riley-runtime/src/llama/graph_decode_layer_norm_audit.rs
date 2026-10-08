//! Cold G02P2 resource-mapping audit. Scratch contents are sampled between
//! iterations, NOT claimed to be the intermediate activation at each layer.
//! Receipts are diagnostic only and cannot promote capture inventory slots.
use super::{PreparedLlamaBatchExecutor, ResidualNormImplementation};
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::PreparedLlamaForward;
use crate::llama::graph_decode_final_norm_owner::{
    C07FinalNormGraphOwner, FinalNormBinding, NormBindingSite,
};
use crate::llama::{ExecutionSite, LlamaOp, PhysicalWeightId};
use riley_cuda::CudaStream;

#[derive(Debug)]
pub(crate) struct LayerNormAuditReceipt {
    pub(crate) site: NormBindingSite,
    pub(crate) hashes: [[u8; 32]; 3],
}

impl PreparedLlamaBatchExecutor {
    /// Audit every standalone input/post-attention norm with the exact uploaded
    /// weight and dispatch scratch allocation. Restore the shared output before
    /// returning. No evidence survives as an admitted graph owner.
    pub(crate) fn audit_c07_layer_norms(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<Vec<LayerNormAuditReceipt>> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() {
            return Err(rejected("executor is poisoned"));
        }
        if self.config.residual_norm_implementation() != ResidualNormImplementation::Separate {
            return Err(rejected(
                "fused residual norm requires separate capture evidence",
            ));
        }
        if !self.output_ready {
            return Err(rejected(
                "audit requires a completed iteration with initialized scratch",
            ));
        }
        let forward = &mut self.owner.forward;
        if forward.plan.sequence_length() != 1 {
            return Err(rejected("initial layer norm audit requires exact M=1"));
        }
        let schedule = layer_norm_schedule(forward)?;
        let mut original = vec![
            0;
            usize::try_from(forward.buffers.hidden_norm.byte_len()).map_err(
                |_| rejected("norm output exceeds host address space")
            )?
        ];
        forward
            .buffers
            .hidden_norm
            .download_to_slice(0, &mut original, &mut forward.io_staging, stream)
            .map_err(|source| {
                crate::llama::forward::poison_for_cuda_error(&mut self.owner.poisoned, &source);
                cuda_error(ExecutionSite::global(LlamaOp::InputNorm), source)
            })?;
        let result = (|| {
            let mut receipts = Vec::with_capacity(schedule.len());
            for (site, weight_id, binding) in schedule {
                let input = match site {
                    NormBindingSite::Input(_) => &mut forward.buffers.hidden_current,
                    NormBindingSite::PostAttention(_) => &mut forward.buffers.hidden_rotary,
                    NormBindingSite::Final => unreachable!("layer-only schedule"),
                };
                let weight = forward
                    .weights
                    .borrow_graph_weight(weight_id)
                    .map_err(|_| rejected("layer norm weight cannot be borrowed"))?;
                let mut graph = C07FinalNormGraphOwner::prepare(
                    stream,
                    input,
                    weight,
                    &mut forward.buffers.hidden_norm,
                    &mut forward.io_staging,
                    binding,
                    &mut self.owner.poisoned,
                )?;
                graph.replay().map_err(|source| {
                    cuda_error(
                        match site {
                            NormBindingSite::Input(layer) => {
                                ExecutionSite::layer(layer, LlamaOp::InputNorm)
                            }
                            NormBindingSite::PostAttention(layer) => {
                                ExecutionSite::layer(layer, LlamaOp::PostAttentionNorm)
                            }
                            NormBindingSite::Final => unreachable!(),
                        },
                        source,
                    )
                })?;
                receipts.push(LayerNormAuditReceipt {
                    site,
                    hashes: graph.parity_hashes(),
                });
                graph.close().map_err(|source| {
                    cuda_error(ExecutionSite::global(LlamaOp::InputNorm), source)
                })?;
            }
            Ok(receipts)
        })();
        // A failed probe may have changed scratch or stream state. Do not try to
        // restore through a potentially poisoned CUDA context or permit reuse.
        if result.is_err() {
            self.owner.poisoned = true;
            return result;
        }
        if let Err(source) = forward.buffers.hidden_norm.upload_from_slice(
            0,
            &original,
            &mut forward.io_staging,
            stream,
        ) {
            self.owner.poisoned = true;
            return Err(cuda_error(
                ExecutionSite::global(LlamaOp::InputNorm),
                source,
            ));
        }
        result
    }
}

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 layer norm audit",
        reason,
    }
}

fn layer_norm_schedule(
    forward: &PreparedLlamaForward,
) -> LlamaBatchExecutorResult<Vec<(NormBindingSite, PhysicalWeightId, FinalNormBinding)>> {
    let profile = forward.rms_norm_profile();
    let hidden = forward.plan.dimensions().hidden_size();
    // Validate the entire schedule before any workspace mutation.
    let mut schedule = Vec::with_capacity(forward.plan.layers().len() * 2);
    for layer in forward.plan.layers() {
        for (site, weight_id, epsilon) in [
            (
                NormBindingSite::Input(layer.index()),
                layer.input_norm_weight(),
                layer.input_norm_epsilon(),
            ),
            (
                NormBindingSite::PostAttention(layer.index()),
                layer.post_attention_norm_weight(),
                layer.post_attention_norm_epsilon(),
            ),
        ] {
            let binding = FinalNormBinding::new(profile, 1, hidden as u64, epsilon, weight_id)
                .map_err(|_| rejected("layer norm profile or geometry has no capture evidence"))?
                .at_site(site);
            let metadata = forward
                .weights
                .physical_metadata(weight_id)
                .ok_or_else(|| rejected("layer norm weight belongs to another owner"))?;
            if metadata.dtype != riley_tensor::DType::BF16
                || metadata.shape != [hidden]
                || metadata.byte_len != hidden as u64 * 2
            {
                return Err(rejected("layer norm weight dtype or shape differs"));
            }
            schedule.push((site, weight_id, binding));
        }
    }
    if schedule.is_empty() {
        return Err(rejected("layer norm schedule is empty"));
    }
    Ok(schedule)
}
