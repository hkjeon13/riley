//! Reserve actual M=1 parents together; no operators or graph admission.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::{
    buffers::{BatchDeviceInput, BatchHostInput},
    error::{LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error},
};
use crate::llama::forward::PreparedLlamaGemm;
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{BorrowedGraphResourceParents, BorrowedGraphResourceReservation, CudaStream};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 resource ledger",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    pub(crate) fn audit_c07_resource_reservation(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<(usize, usize)> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected(
                "reservation audit needs a healthy completed iteration",
            ));
        }
        let owner = &mut self.owner;
        let forward = &mut owner.forward;
        if forward.plan.sequence_length() != 1 {
            return Err(rejected("reservation audit requires M=1"));
        }
        let (BatchDeviceInput::IterationBatch { slab }, BatchHostInput::IterationBatch(host)) =
            (&mut owner.device_input, &mut owner.host.input)
        else {
            return Err(rejected("actual packed parents required"));
        };
        let buffers = &mut forward.buffers;
        let mut devices: Vec<_> = forward.weights.borrow_graph_weight_parents().collect();
        devices.extend([
            &mut buffers.token_ids,
            &mut buffers.hidden_current,
            &mut buffers.hidden_norm,
            &mut buffers.hidden_projection,
            &mut buffers.hidden_rotary,
            &mut buffers.hidden_context,
            &mut buffers.key_raw,
            &mut buffers.value_raw,
            &mut buffers.key_rotary,
            &mut buffers.gate_raw,
            &mut buffers.up_raw,
            &mut buffers.gate_activated,
            &mut buffers.gated_product,
            &mut buffers.rope_cos,
            &mut buffers.rope_sin,
            &mut buffers.logits,
            &mut buffers.embedding_error_scratch,
            &mut owner.key_cache,
            &mut owner.value_cache,
            &mut owner.absolute_rope_cos,
            &mut owner.absolute_rope_sin,
            slab,
        ]);
        devices.extend(buffers.attention_workspace.as_mut());
        devices.extend(buffers.gemm_workspace.as_mut());
        devices.extend(owner.gathered_logits.as_mut());
        devices.extend(owner.greedy_results.as_mut());
        let gemms = &mut forward.gemms;
        let plans = [
            &mut gemms.hidden,
            &mut gemms.key_value,
            &mut gemms.intermediate,
            &mut gemms.down,
            &mut gemms.lm_head,
        ]
        .into_iter()
        .map(|plan| match plan {
            PreparedLlamaGemm::Canonical(plan) => Ok(plan),
            PreparedLlamaGemm::Fixed37(_) => Err(rejected("selected plan backend unsupported")),
        })
        .collect::<LlamaBatchExecutorResult<Vec<_>>>()?;
        let counts = (devices.len(), plans.len());
        let result = BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents {
            stream,
            devices,
            pinned: vec![&mut host.pinned, &mut forward.io_staging],
            plans,
        })
        .and_then(BorrowedGraphResourceReservation::close);
        result.map_err(|source| {
            // Reservation normally rolls back completely. Treat any native
            // failure conservatively; this diagnostic never publishes output.
            self.owner.poisoned = true;
            cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), source)
        })?;
        Ok(counts)
    }
}
