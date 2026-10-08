//! Cold MLP-chain audit on real selected plans, weights and scratch parents.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::{PreparedLlamaGemm, execute_gemm};
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{
    BorrowedGraphResourceParents, BorrowedGraphResourceReservation, CudaBufferSpan,
    CudaBufferSpanMut, CudaDType, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream,
    GatedMultiplyParams, ResidualAddParams, RmsNormParams, SiluParams, gated_multiply,
    residual_add, silu,
};

fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "C07 MLP audit",
        reason,
    }
}
fn read(
    b: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> riley_cuda::CudaResult<Vec<u8>> {
    let mut bytes = vec![0; b.byte_len() as usize];
    b.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}
impl PreparedLlamaBatchExecutor {
    #[allow(clippy::too_many_lines)] // Explicit graph/eager mapping must stay reviewable.
    pub(crate) fn audit_c07_mlp_chain(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_c07_mlp_impl(stream, false, false)
    }
    pub(crate) fn audit_c07_norm_mlp_chain(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_c07_mlp_impl(stream, true, false)
    }
    pub(crate) fn audit_c07_layer_tail(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_c07_mlp_impl(stream, true, true)
    }
    #[allow(clippy::too_many_lines)]
    fn audit_c07_mlp_impl(
        &mut self,
        stream: &mut CudaStream,
        include_norm: bool,
        include_tail: bool,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("MLP audit needs a healthy completed iteration"));
        }
        let f = &mut self.owner.forward;
        let norm_profile = f.rms_norm_profile();
        if include_norm
            && !matches!(
                norm_profile,
                crate::llama::forward::LlamaRmsNormProfile::Canonical
                    | crate::llama::forward::LlamaRmsNormProfile::HuggingFaceSmolLm2
            )
        {
            return Err(rejected("norm MLP profile unsupported"));
        }
        if f.plan.sequence_length() != 1 {
            return Err(rejected("MLP audit requires M=1"));
        }
        let layer = f
            .plan
            .layers()
            .last()
            .ok_or_else(|| rejected("model has no layer"))?;
        let projection_id = layer.output_weight();
        if include_tail && layer.output_bias().is_some() {
            return Err(rejected("layer tail projection bias unsupported"));
        }
        let norm_id = layer.post_attention_norm_weight();
        let norm_epsilon = layer.post_attention_norm_epsilon();
        // LlamaExecutionPlan construction rejects biased MLPs (plan.rs).
        let weight_ids = [layer.gate_weight(), layer.up_weight(), layer.down_weight()];
        for id in weight_ids {
            if f.weights.physical_metadata(id).is_none() {
                return Err(rejected("foreign MLP weight"));
            }
        }
        for p in [&f.gemms.intermediate, &f.gemms.down, &f.gemms.hidden] {
            let PreparedLlamaGemm::Canonical(p) = p else {
                return Err(rejected("MLP backend unsupported"));
            };
            if p.algorithm_metadata().split_k() > 1
                || p.algorithm_metadata().reduction_scheme() != 0
            {
                return Err(rejected("MLP selected plan is not no-split"));
            }
        }
        let result = (|| -> LlamaBatchExecutorResult<()> {
            let site = ExecutionSite::global(LlamaOp::GatedMultiply);
            let cuda = |e| cuda_error(site, e);
            let b = &mut f.buffers;
            let pinned = f.io_staging.to_vec().map_err(cuda)?;
            let mut saved = Vec::new();
            for buffer in [
                &mut b.hidden_norm,
                &mut b.hidden_rotary,
                &mut b.gate_raw,
                &mut b.up_raw,
                &mut b.gate_activated,
                &mut b.gated_product,
                &mut b.hidden_current,
                &mut b.hidden_projection,
            ] {
                saved.push(read(buffer, &mut f.io_staging, stream).map_err(cuda)?);
            }
            let hidden_bytes = b.hidden_norm.byte_len();
            let intermediate_bytes = b.gate_raw.byte_len();
            let context = if include_tail {
                let context =
                    read(&mut b.hidden_context, &mut f.io_staging, stream).map_err(cuda)?;
                let weight = f
                    .weights
                    .view_physical(projection_id)
                    .map_err(|_| rejected("layer tail output weight unavailable"))?;
                execute_gemm(
                    &mut f.gemms.hidden,
                    &b.hidden_context,
                    weight.span(),
                    &mut b.hidden_projection,
                    &mut b.gemm_workspace,
                    stream,
                    ExecutionSite::global(LlamaOp::OutputProjection),
                )
                .map_err(LlamaBatchExecutorError::Forward)?;
                residual_add(
                    &mut ResidualAddParams {
                        left: CudaBufferSpan::new(
                            &b.hidden_current,
                            CudaDType::BF16,
                            0,
                            hidden_bytes,
                        )
                        .map_err(cuda)?,
                        right: CudaBufferSpan::new(
                            &b.hidden_projection,
                            CudaDType::BF16,
                            0,
                            hidden_bytes,
                        )
                        .map_err(cuda)?,
                        output: CudaBufferSpanMut::new(
                            &mut b.hidden_rotary,
                            CudaDType::BF16,
                            0,
                            hidden_bytes,
                        )
                        .map_err(cuda)?,
                        element_count: hidden_bytes / 2,
                    },
                    stream,
                )
                .map_err(cuda)?;
                Some(context)
            } else {
                None
            };
            if include_norm {
                let weight = f
                    .weights
                    .view_physical(norm_id)
                    .map_err(|_| rejected("norm MLP weight unavailable"))?;
                crate::llama::forward::execute_profile_rms_norm(
                    norm_profile,
                    &mut RmsNormParams {
                        input: CudaBufferSpan::new(
                            &b.hidden_rotary,
                            CudaDType::BF16,
                            0,
                            hidden_bytes,
                        )
                        .map_err(cuda)?,
                        weight: weight.span(),
                        output: CudaBufferSpanMut::new(
                            &mut b.hidden_norm,
                            CudaDType::BF16,
                            0,
                            hidden_bytes,
                        )
                        .map_err(cuda)?,
                        row_count: 1,
                        hidden_size: hidden_bytes / 2,
                        epsilon: norm_epsilon,
                    },
                    stream,
                )
                .map_err(cuda)?;
            }
            for (id, output, op) in [
                (weight_ids[0], &mut b.gate_raw, LlamaOp::GateProjection),
                (weight_ids[1], &mut b.up_raw, LlamaOp::UpProjection),
            ] {
                let weight = f
                    .weights
                    .view_physical(id)
                    .map_err(|_| rejected("MLP weight span unavailable"))?;
                execute_gemm(
                    &mut f.gemms.intermediate,
                    &b.hidden_norm,
                    weight.span(),
                    output,
                    &mut b.gemm_workspace,
                    stream,
                    ExecutionSite::global(op),
                )
                .map_err(LlamaBatchExecutorError::Forward)?;
            }
            silu(
                &mut SiluParams {
                    input: CudaBufferSpan::new(&b.gate_raw, CudaDType::BF16, 0, intermediate_bytes)
                        .map_err(cuda)?,
                    output: CudaBufferSpanMut::new(
                        &mut b.gate_activated,
                        CudaDType::BF16,
                        0,
                        intermediate_bytes,
                    )
                    .map_err(cuda)?,
                    element_count: intermediate_bytes / 2,
                },
                stream,
            )
            .map_err(cuda)?;
            gated_multiply(
                &mut GatedMultiplyParams {
                    activated_gate: CudaBufferSpan::new(
                        &b.gate_activated,
                        CudaDType::BF16,
                        0,
                        intermediate_bytes,
                    )
                    .map_err(cuda)?,
                    up: CudaBufferSpan::new(&b.up_raw, CudaDType::BF16, 0, intermediate_bytes)
                        .map_err(cuda)?,
                    output: CudaBufferSpanMut::new(
                        &mut b.gated_product,
                        CudaDType::BF16,
                        0,
                        intermediate_bytes,
                    )
                    .map_err(cuda)?,
                    element_count: intermediate_bytes / 2,
                },
                stream,
            )
            .map_err(cuda)?;
            let weight = f
                .weights
                .view_physical(weight_ids[2])
                .map_err(|_| rejected("MLP down weight unavailable"))?;
            execute_gemm(
                &mut f.gemms.down,
                &b.gated_product,
                weight.span(),
                &mut b.hidden_current,
                &mut b.gemm_workspace,
                stream,
                ExecutionSite::global(LlamaOp::DownProjection),
            )
            .map_err(LlamaBatchExecutorError::Forward)?;
            residual_add(
                &mut ResidualAddParams {
                    left: CudaBufferSpan::new(&b.hidden_rotary, CudaDType::BF16, 0, hidden_bytes)
                        .map_err(cuda)?,
                    right: CudaBufferSpan::new(&b.hidden_current, CudaDType::BF16, 0, hidden_bytes)
                        .map_err(cuda)?,
                    output: CudaBufferSpanMut::new(
                        &mut b.hidden_projection,
                        CudaDType::BF16,
                        0,
                        hidden_bytes,
                    )
                    .map_err(cuda)?,
                    element_count: hidden_bytes / 2,
                },
                stream,
            )
            .map_err(cuda)?;
            let expected = [
                read(&mut b.hidden_current, &mut f.io_staging, stream).map_err(cuda)?,
                read(&mut b.hidden_projection, &mut f.io_staging, stream).map_err(cuda)?,
            ]
            .concat();
            let payload = if let Some(context) = context {
                [context, saved[6].clone()].concat()
            } else {
                [saved[0].clone(), saved[1].clone()].concat()
            };
            let zeros = vec![0; payload.len()];
            let mut actual = zeros.clone();
            f.io_staging.write(0, &pinned).map_err(cuda)?;
            let mut parents: Vec<_> = f.weights.borrow_graph_weight_parents().collect();
            let base = parents.len();
            parents.extend([
                &mut b.hidden_norm,
                &mut b.hidden_rotary,
                &mut b.gate_raw,
                &mut b.up_raw,
                &mut b.gate_activated,
                &mut b.gated_product,
                &mut b.hidden_current,
                &mut b.hidden_projection,
            ]);
            let workspace = b.gemm_workspace.as_mut().map(|buffer| {
                let index = parents.len();
                parents.push(buffer);
                index
            });
            let PreparedLlamaGemm::Canonical(intermediate) = &mut f.gemms.intermediate else {
                unreachable!()
            };
            let PreparedLlamaGemm::Canonical(down) = &mut f.gemms.down else {
                unreachable!()
            };
            let PreparedLlamaGemm::Canonical(projection) = &mut f.gemms.hidden else {
                unreachable!()
            };
            let identities = [
                (intermediate.config(), intermediate.algorithm_metadata()),
                (down.config(), down.algorithm_metadata()),
                (projection.config(), projection.algorithm_metadata()),
            ];
            let mut graph =
                BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents {
                    stream,
                    devices: parents,
                    pinned: vec![&mut f.io_staging],
                    plans: vec![intermediate, down, projection],
                })
                .map_err(cuda)?;
            let slots = [
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                weight_ids[0].index(),
                weight_ids[1].index(),
                weight_ids[2].index(),
            ];
            if include_tail {
                graph
                    .record_layer_tail(
                        slots,
                        workspace,
                        [0, 1],
                        0,
                        (
                            norm_id.index(),
                            norm_epsilon,
                            norm_profile
                                == crate::llama::forward::LlamaRmsNormProfile::HuggingFaceSmolLm2,
                        ),
                        (projection_id.index(), 2),
                    )
                    .map_err(cuda)?;
            } else if include_norm {
                graph
                    .record_norm_mlp(
                        slots,
                        workspace,
                        [0, 1],
                        0,
                        (
                            norm_id.index(),
                            norm_epsilon,
                            norm_profile
                                == crate::llama::forward::LlamaRmsNormProfile::HuggingFaceSmolLm2,
                        ),
                    )
                    .map_err(cuda)?;
            } else {
                graph
                    .record_mlp(slots, workspace, [0, 1], 0)
                    .map_err(cuda)?;
            }
            for _ in 0..16 {
                graph.replay_transfer(&zeros).map_err(cuda)?;
                graph.read_transfer(&mut actual).map_err(cuda)?;
                // Negative zero is numerically zero and can arise from GEMM.
                if actual
                    .chunks_exact(2)
                    .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7fff != 0)
                {
                    return Err(rejected("MLP zero replay mismatch"));
                }
                graph.replay_transfer(&payload).map_err(cuda)?;
                graph.read_transfer(&mut actual).map_err(cuda)?;
                if actual != expected {
                    return Err(rejected("MLP graph differs from eager chain"));
                }
            }
            graph.close().map_err(cuda)?;
            if identities
                != [
                    (intermediate.config(), intermediate.algorithm_metadata()),
                    (down.config(), down.algorithm_metadata()),
                    (projection.config(), projection.algorithm_metadata()),
                ]
            {
                return Err(rejected("MLP selected plan changed"));
            }
            if f.io_staging.to_vec().map_err(cuda)?[payload.len() * 2..]
                != pinned[payload.len() * 2..]
            {
                return Err(rejected("MLP staging tail changed"));
            }
            for (buffer, bytes) in [
                &mut b.hidden_norm,
                &mut b.hidden_rotary,
                &mut b.gate_raw,
                &mut b.up_raw,
                &mut b.gate_activated,
                &mut b.gated_product,
                &mut b.hidden_current,
                &mut b.hidden_projection,
            ]
            .into_iter()
            .zip(&saved)
            {
                buffer
                    .upload_from_slice(0, bytes, &mut f.io_staging, stream)
                    .map_err(cuda)?;
                if read(buffer, &mut f.io_staging, stream).map_err(cuda)? != *bytes {
                    return Err(rejected("MLP scratch restore mismatch"));
                }
            }
            f.io_staging.write(0, &pinned).map_err(cuda)?;
            Ok(())
        })();
        if result.is_err() {
            self.owner.poisoned = true;
        }
        result
    }
}
