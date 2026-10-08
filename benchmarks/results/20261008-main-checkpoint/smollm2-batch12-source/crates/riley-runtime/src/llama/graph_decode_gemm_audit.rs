//! Cold projection audit on actual plans/weights/scratch, not an activation trace.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::PreparedLlamaGemm;
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{
    BorrowedSelectedGemmGraph, BorrowedSelectedGemmResources, CudaBufferSpan, CudaBufferSpanMut,
    CudaDType, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream, GemmParams,
};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 GEMM audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    /// Cold LM head probe on the actual final-normalized input and logits owner.
    pub(crate) fn audit_c07_lm_head(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<Vec<u8>> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("LM head needs a healthy completed iteration"));
        }
        let f = &mut self.owner.forward;
        if f.plan.sequence_length() != 1 {
            return Err(rejected("LM head audit requires M=1"));
        }
        let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.lm_head else {
            return Err(rejected("LM head backend is unsupported"));
        };
        if plan.algorithm_metadata().split_k() > 1
            || plan.algorithm_metadata().reduction_scheme() != 0
        {
            return Err(rejected("LM head topology is not no-split"));
        }
        let weight = f
            .weights
            .borrow_graph_weight(f.plan.lm_head_weight())
            .map_err(|_| rejected("physical LM head weight unavailable"))?;
        let result = probe(
            BorrowedSelectedGemmResources {
                stream,
                plan,
                input: &mut f.buffers.hidden_norm,
                weight,
                output: &mut f.buffers.logits,
                workspace: f.buffers.gemm_workspace.as_mut(),
            },
            &mut f.io_staging,
        );
        result.map_err(|error| {
            self.owner.poisoned = true;
            match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => {
                    cuda_error(ExecutionSite::global(LlamaOp::LmHead), source)
                }
            }
        })
    }

    /// Preserves the selected plan and actual optional workspace parent.
    #[allow(clippy::too_many_lines)] // Explicit seven mappings mirror dispatch.
    pub(crate) fn audit_c07_gemms(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<usize> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("audit needs a healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1 {
            return Err(rejected("audit requires M=1"));
        }
        let f = &mut self.owner.forward;
        // Preflight every selected plan before modifying any output.
        for plan in [
            &f.gemms.hidden,
            &f.gemms.key_value,
            &f.gemms.intermediate,
            &f.gemms.down,
        ] {
            let PreparedLlamaGemm::Canonical(selected) = plan else {
                return Err(rejected("selected GEMM backend is unsupported"));
            };
            if selected.algorithm_metadata().split_k() > 1
                || selected.algorithm_metadata().reduction_scheme() != 0
            {
                return Err(rejected("selected GEMM topology is not no-split"));
            }
            if plan.workspace_bytes()
                > f.buffers
                    .gemm_workspace
                    .as_ref()
                    .map_or(0, CudaDeviceBuffer::byte_len)
            {
                return Err(rejected("actual shared workspace is too small"));
            }
        }
        let mut count = 0;
        let result = (|| -> Result<(), AuditError> {
            for layer in f.plan.layers() {
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.hidden else {
                        unreachable!()
                    };
                    let weight = f
                        .weights
                        .borrow_graph_weight(layer.query_weight())
                        .map_err(|_| {
                            AuditError::Invalid("physical projection weight unavailable")
                        })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_norm,
                            weight,
                            output: &mut f.buffers.hidden_projection,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.key_value else {
                        unreachable!()
                    };
                    let weight =
                        f.weights
                            .borrow_graph_weight(layer.key_weight())
                            .map_err(|_| {
                                AuditError::Invalid("physical projection weight unavailable")
                            })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_norm,
                            weight,
                            output: &mut f.buffers.key_raw,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.key_value else {
                        unreachable!()
                    };
                    let weight = f
                        .weights
                        .borrow_graph_weight(layer.value_weight())
                        .map_err(|_| {
                            AuditError::Invalid("physical projection weight unavailable")
                        })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_norm,
                            weight,
                            output: &mut f.buffers.value_raw,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.hidden else {
                        unreachable!()
                    };
                    let weight = f
                        .weights
                        .borrow_graph_weight(layer.output_weight())
                        .map_err(|_| {
                            AuditError::Invalid("physical projection weight unavailable")
                        })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_context,
                            weight,
                            output: &mut f.buffers.hidden_projection,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.intermediate else {
                        unreachable!()
                    };
                    let weight =
                        f.weights
                            .borrow_graph_weight(layer.gate_weight())
                            .map_err(|_| {
                                AuditError::Invalid("physical projection weight unavailable")
                            })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_norm,
                            weight,
                            output: &mut f.buffers.gate_raw,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.intermediate else {
                        unreachable!()
                    };
                    let weight =
                        f.weights
                            .borrow_graph_weight(layer.up_weight())
                            .map_err(|_| {
                                AuditError::Invalid("physical projection weight unavailable")
                            })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.hidden_norm,
                            weight,
                            output: &mut f.buffers.up_raw,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
                {
                    let PreparedLlamaGemm::Canonical(plan) = &mut f.gemms.down else {
                        unreachable!()
                    };
                    let weight =
                        f.weights
                            .borrow_graph_weight(layer.down_weight())
                            .map_err(|_| {
                                AuditError::Invalid("physical projection weight unavailable")
                            })?;
                    let workspace = f.buffers.gemm_workspace.as_mut();
                    probe(
                        BorrowedSelectedGemmResources {
                            stream,
                            plan,
                            input: &mut f.buffers.gated_product,
                            weight,
                            output: &mut f.buffers.hidden_current,
                            workspace,
                        },
                        &mut f.io_staging,
                    )?;
                    count += 1;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.owner.poisoned = true;
            return Err(match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => {
                    cuda_error(ExecutionSite::global(LlamaOp::QueryProjection), source)
                }
            });
        }
        Ok(count)
    }
}
fn read(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> Result<Vec<u8>, AuditError> {
    let size = usize::try_from(buffer.byte_len())
        .map_err(|_| AuditError::Invalid("host range overflow"))?;
    let mut bytes = vec![0; size];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}
fn probe(
    resources: BorrowedSelectedGemmResources<'_>,
    staging: &mut CudaPinnedHostBuffer,
) -> Result<Vec<u8>, AuditError> {
    let BorrowedSelectedGemmResources {
        stream,
        plan,
        input,
        weight,
        output,
        mut workspace,
    } = resources;
    let original = read(output, staging, stream)?;
    let input_before = read(input, staging, stream)?;
    let weight_before = read(weight, staging, stream)?;
    let config = plan.config();
    let algorithm = plan.algorithm_metadata();
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    let mut graph = BorrowedSelectedGemmGraph::prepare(BorrowedSelectedGemmResources {
        stream,
        plan,
        input,
        weight,
        output,
        workspace: workspace.as_deref_mut(),
    })?;
    for _ in 0..4 {
        graph.replay()?;
    }
    graph.close()?;
    let captured = read(output, staging, stream)?;
    // Poison the destination to exclude a no-op eager comparison.
    output.upload_from_slice(0, &vec![0xff; original.len()], staging, stream)?;
    plan.execute(
        &mut GemmParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace: if workspace_bytes == 0 {
                None
            } else {
                Some(CudaBufferSpanMut::new(
                    workspace
                        .as_deref_mut()
                        .ok_or(AuditError::Invalid("workspace missing"))?,
                    CudaDType::U8,
                    0,
                    workspace_bytes,
                )?)
            },
        },
        stream,
    )?;
    if plan.config() != config
        || plan.algorithm_metadata() != algorithm
        || captured != read(output, staging, stream)?
        || input_before != read(input, staging, stream)?
        || weight_before != read(weight, staging, stream)?
        || captured
            .chunks_exact(2)
            .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(AuditError::Invalid(
            "GEMM parity, finite output or input/weight preservation failed",
        ));
    }
    output.upload_from_slice(0, &original, staging, stream)?;
    if read(output, staging, stream)? != original {
        return Err(AuditError::Invalid("GEMM scratch restore failed"));
    }
    Ok(captured)
}
#[derive(Debug)]
enum AuditError {
    Cuda(riley_cuda::CudaError),
    Invalid(&'static str),
}
impl From<riley_cuda::CudaError> for AuditError {
    fn from(e: riley_cuda::CudaError) -> Self {
        Self::Cuda(e)
    }
}
