//! Cold pointwise buffer audit; shared scratch is not an in-flight layer trace.
use super::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, PreparedLlamaBatchExecutor,
    ResidualNormImplementation,
};
use riley_cuda::{
    BorrowedGraphResourceParents, BorrowedGraphResourceReservation, BorrowedPointwiseGraph,
    BorrowedPointwiseResources, CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaStream, GatedMultiplyParams, PointwiseGraphOperation as Op,
    ResidualAddParams, SiluParams, gated_multiply, residual_add, silu,
};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 pointwise audit",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    pub(crate) fn audit_c07_swiglu_chain(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("SwiGLU graph needs a healthy completed iteration"));
        }
        let f = &mut self.owner.forward;
        if f.plan.sequence_length() != 1 {
            return Err(rejected("SwiGLU chain requires M=1"));
        }
        let result = probe_chain(
            &mut f.buffers.gate_raw,
            &mut f.buffers.up_raw,
            &mut f.buffers.gate_activated,
            &mut f.buffers.gated_product,
            &mut f.io_staging,
            stream,
        );
        result.map_err(|error| {
            self.owner.poisoned = true;
            match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => crate::llama::executor::error::cuda_error(
                    crate::llama::ExecutionSite::global(crate::llama::LlamaOp::GatedMultiply),
                    source,
                ),
            }
        })
    }

    pub(crate) fn audit_c07_pointwise(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("audit needs a healthy completed iteration"));
        }
        if self.owner.forward.plan.sequence_length() != 1
            || self.config.residual_norm_implementation() != ResidualNormImplementation::Separate
        {
            return Err(rejected(
                "audit requires M=1 and separate residual normalization",
            ));
        }
        let forward = &mut self.owner.forward;
        let hidden = forward.plan.workspace_spec().hidden_buffer_bytes() / 2;
        let intermediate = forward.plan.workspace_spec().intermediate_buffer_bytes() / 2;
        let buffers = &mut forward.buffers;
        let result = (|| -> Result<(), (crate::llama::LlamaOp, AuditError)> {
            probe(
                BorrowedPointwiseResources {
                    stream,
                    input: &mut buffers.gate_raw,
                    other: None,
                    output: &mut buffers.gate_activated,
                },
                Op::Silu,
                intermediate,
                &mut forward.io_staging,
            )
            .map_err(|error| (crate::llama::LlamaOp::Silu, error))?;
            probe(
                BorrowedPointwiseResources {
                    stream,
                    input: &mut buffers.gate_activated,
                    other: Some(&mut buffers.up_raw),
                    output: &mut buffers.gated_product,
                },
                Op::GatedMultiply,
                intermediate,
                &mut forward.io_staging,
            )
            .map_err(|error| (crate::llama::LlamaOp::GatedMultiply, error))?;
            probe(
                BorrowedPointwiseResources {
                    stream,
                    input: &mut buffers.hidden_current,
                    other: Some(&mut buffers.hidden_projection),
                    output: &mut buffers.hidden_rotary,
                },
                Op::ResidualAdd,
                hidden,
                &mut forward.io_staging,
            )
            .map_err(|error| (crate::llama::LlamaOp::AttentionResidual, error))?;
            probe(
                BorrowedPointwiseResources {
                    stream,
                    input: &mut buffers.hidden_rotary,
                    other: Some(&mut buffers.hidden_current),
                    output: &mut buffers.hidden_projection,
                },
                Op::ResidualAdd,
                hidden,
                &mut forward.io_staging,
            )
            .map_err(|error| (crate::llama::LlamaOp::MlpResidual, error))
        })();
        if let Err((site, error)) = result {
            self.owner.poisoned = true;
            return Err(match error {
                AuditError::Invalid(reason) => rejected(reason),
                AuditError::Cuda(source) => crate::llama::executor::error::cuda_error(
                    crate::llama::ExecutionSite::global(site),
                    source,
                ),
            });
        }
        Ok(())
    }
}
fn read(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> Result<Vec<u8>, AuditError> {
    let size = usize::try_from(buffer.byte_len())
        .map_err(|_| AuditError::Invalid("buffer exceeds host address space"))?;
    let mut bytes = vec![0; size];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}
fn probe(
    resources: BorrowedPointwiseResources<'_>,
    op: Op,
    elements: u64,
    staging: &mut CudaPinnedHostBuffer,
) -> Result<(), AuditError> {
    let BorrowedPointwiseResources {
        stream,
        input,
        mut other,
        output,
    } = resources;
    let original = read(output, staging, stream)?;
    let input_before = read(input, staging, stream)?;
    let other_before = other
        .as_deref_mut()
        .map(|b| read(b, staging, stream))
        .transpose()?;
    let mut graph = BorrowedPointwiseGraph::prepare(
        BorrowedPointwiseResources {
            stream,
            input,
            other: other.as_deref_mut(),
            output,
        },
        op,
        elements,
    )?;
    for _ in 0..16 {
        graph.replay()?;
    }
    graph.close()?;
    let captured = read(output, staging, stream)?;
    let bytes = elements * 2;
    let left = CudaBufferSpan::new(input, CudaDType::BF16, 0, bytes)?;
    let right = other
        .as_deref()
        .map(|b| CudaBufferSpan::new(b, CudaDType::BF16, 0, bytes))
        .transpose()?;
    let out = CudaBufferSpanMut::new(output, CudaDType::BF16, 0, bytes)?;
    match op {
        Op::Silu => silu(
            &mut SiluParams {
                input: left,
                output: out,
                element_count: elements,
            },
            stream,
        )?,
        Op::GatedMultiply => gated_multiply(
            &mut GatedMultiplyParams {
                activated_gate: left,
                up: right.expect("binary"),
                output: out,
                element_count: elements,
            },
            stream,
        )?,
        Op::ResidualAdd => residual_add(
            &mut ResidualAddParams {
                left,
                right: right.expect("binary"),
                output: out,
                element_count: elements,
            },
            stream,
        )?,
    }
    let eager = read(output, staging, stream)?;
    let used = usize::try_from(bytes)
        .map_err(|_| AuditError::Invalid("pointwise span exceeds host range"))?;
    if captured != eager
        || read(input, staging, stream)? != input_before
        || other.map(|b| read(b, staging, stream)).transpose()? != other_before
        || captured[used..] != original[used..]
        || captured[..used]
            .chunks_exact(2)
            .any(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(AuditError::Invalid(
            "pointwise parity, finite output or owner preservation failed",
        ));
    }
    output.upload_from_slice(0, &original, staging, stream)?;
    if read(output, staging, stream)? != original {
        return Err(AuditError::Invalid("pointwise output restoration failed"));
    }
    Ok(())
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

// The two eager calls generate a reference from actual retained model inputs.
// This probes the final live scratch snapshot, not every layer's activation trace.
fn probe_chain(
    gate: &mut CudaDeviceBuffer,
    up: &mut CudaDeviceBuffer,
    activated: &mut CudaDeviceBuffer,
    product: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> Result<(), AuditError> {
    let pinned_before = staging.to_vec()?;
    let gate_before = read(gate, staging, stream)?;
    let up_before = read(up, staging, stream)?;
    let activated_before = read(activated, staging, stream)?;
    let product_before = read(product, staging, stream)?;
    let bytes = gate.byte_len();
    silu(
        &mut SiluParams {
            input: CudaBufferSpan::new(gate, CudaDType::BF16, 0, bytes)?,
            output: CudaBufferSpanMut::new(activated, CudaDType::BF16, 0, bytes)?,
            element_count: bytes / 2,
        },
        stream,
    )?;
    gated_multiply(
        &mut GatedMultiplyParams {
            activated_gate: CudaBufferSpan::new(activated, CudaDType::BF16, 0, bytes)?,
            up: CudaBufferSpan::new(up, CudaDType::BF16, 0, bytes)?,
            output: CudaBufferSpanMut::new(product, CudaDType::BF16, 0, bytes)?,
            element_count: bytes / 2,
        },
        stream,
    )?;
    let expected = [
        read(activated, staging, stream)?,
        read(product, staging, stream)?,
    ]
    .concat();
    let payload = [gate_before.clone(), up_before.clone()].concat();
    let zeros = vec![0; payload.len()];
    let mut actual = vec![0; payload.len()];
    staging.write(0, &pinned_before)?;
    let mut graph = BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents {
        stream,
        devices: vec![gate, up, activated, product],
        pinned: vec![staging],
        plans: vec![],
    })?;
    graph.record_swiglu([0, 1, 2, 3], 0)?;
    for _ in 0..16 {
        graph.replay_transfer(&zeros)?;
        graph.read_transfer(&mut actual)?;
        if actual != zeros {
            return Err(AuditError::Invalid("SwiGLU zero payload mismatch"));
        }
        graph.replay_transfer(&payload)?;
        graph.read_transfer(&mut actual)?;
        if actual != expected {
            return Err(AuditError::Invalid("SwiGLU chain differs from eager"));
        }
    }
    graph.close()?;
    let pinned_after = staging.to_vec()?;
    if pinned_after[payload.len() * 2..] != pinned_before[payload.len() * 2..] {
        return Err(AuditError::Invalid("SwiGLU pinned tail changed"));
    }
    if gate_before != read(gate, staging, stream)? || up_before != read(up, staging, stream)? {
        return Err(AuditError::Invalid(
            "SwiGLU final input restoration mismatch",
        ));
    }
    activated.upload_from_slice(0, &activated_before, staging, stream)?;
    product.upload_from_slice(0, &product_before, staging, stream)?;
    if activated_before != read(activated, staging, stream)?
        || product_before != read(product, staging, stream)?
    {
        return Err(AuditError::Invalid("SwiGLU scratch restoration mismatch"));
    }
    staging.write(0, &pinned_before)?;
    Ok(())
}
