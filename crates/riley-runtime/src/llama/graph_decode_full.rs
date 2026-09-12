//! Full M=1 model capture on actual executor weights, plans, scratch and KV parents.
#[path = "graph_decode_packed.rs"]
mod packed;
use packed::PackedDecodeParents;
#[path = "graph_decode_multi_parents.rs"]
mod multi_parents;
use multi_parents::MultiDecodeParents;
#[cfg(all(test, feature = "cuda"))]
#[path = "graph_decode_prefill_parity_gpu.rs"]
mod prefill_parity_gpu;
#[cfg(all(test, feature = "cuda"))]
#[path = "graph_decode_stage_parity_gpu.rs"]
mod stage_parity_gpu;
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::{LlamaRmsNormProfile, PreparedLlamaGemm};
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{
    BorrowedGraphResourceParents, BorrowedGraphResourceReservation, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaStream,
};
const PREFILL128_MAGIC: u32 = 0x5031_3238;
const PREFILL128_METADATA_BYTES: usize = 520;
const PREFILL128_SCRATCH_BYTES: [u64; 12] = [
    128 * 1152,
    128 * 1152,
    128 * 1152,
    128 * 1152,
    128 * 1152,
    128 * 384,
    128 * 384,
    128 * 384,
    128 * 3072,
    128 * 3072,
    128 * 2304,
    128 * 3072,
];
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "full decode graph",
        reason,
    }
}
impl PreparedLlamaBatchExecutor {
    // Dedicated metadata/result parents are cold allocations. The exclusive borrow
    // prevents eager dispatch, cache changes or parent release while the graph lives.
    pub(crate) fn prepare_full_decode<'a>(
        &'a mut self,
        stream: &'a mut CudaStream,
        metadata: &'a mut CudaDeviceBuffer,
        result: &'a mut CudaDeviceBuffer,
        staging: &'a mut CudaPinnedHostBuffer,
        capacity: u64,
        publish_logits: bool,
    ) -> LlamaBatchExecutorResult<BorrowedGraphResourceReservation<'a>> {
        self.prepare_full_decode_with_prefill(
            stream,
            metadata,
            result,
            staging,
            capacity,
            publish_logits,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_full_decode_with_prefill<'a>(
        &'a mut self,
        stream: &'a mut CudaStream,
        metadata: &'a mut CudaDeviceBuffer,
        result: &'a mut CudaDeviceBuffer,
        staging: &'a mut CudaPinnedHostBuffer,
        capacity: u64,
        publish_logits: bool,
        prefill: Option<&'a mut Vec<CudaDeviceBuffer>>,
        packed: Option<&'a mut PackedDecodeParents>,
        multi: Option<&'a mut MultiDecodeParents>,
    ) -> LlamaBatchExecutorResult<BorrowedGraphResourceReservation<'a>> {
        if prefill.as_ref().is_some_and(|buffers| buffers.len() != 12)
            || prefill.is_some() != self.config.vllm_smol_p128_batched_prefill()
            || packed.is_some() != prefill.is_some()
            || packed
                .as_ref()
                .is_some_and(|p| p.buffers.len() != 62 || p.plans.len() != 2)
        {
            return Err(rejected("prefill parents differ from configured topology"));
        }
        if multi.is_some() && (packed.is_none() || capacity != 10 || !publish_logits) {
            return Err(rejected(
                "multi catalog requires packed P128, C10 and full logits",
            ));
        }
        if self.owner.poisoned || self.owner.forward.is_poisoned() {
            return Err(rejected("healthy owner required"));
        }
        let owner = &mut self.owner;
        let f = &mut owner.forward;
        let profile = f.rms_norm_profile();
        let vllm_profile = self.config.vllm_smol_p128_graph();
        if f.plan.sequence_length() != 1
            || owner.layout.head_dimension() != 64
            || !matches!(
                profile,
                LlamaRmsNormProfile::Canonical | LlamaRmsNormProfile::HuggingFaceSmolLm2
            )
            || self.config.ragged_attention_reduction_profile()
                != riley_cuda::AttentionReductionProfile::CanonicalV1
        {
            return Err(rejected("unsupported bucket or reduction profile"));
        }
        let mut weights = vec![
            f.plan.embedding_weight().index(),
            f.plan.final_norm_weight().index(),
            f.plan.lm_head_weight().index(),
        ];
        let mut eps = vec![f.plan.final_norm_epsilon()];
        for l in f.plan.layers() {
            if l.query_bias().is_some()
                || l.key_bias().is_some()
                || l.value_bias().is_some()
                || l.output_bias().is_some()
            {
                return Err(rejected("projection bias unsupported"));
            }
            weights.extend(
                [
                    l.input_norm_weight(),
                    l.query_weight(),
                    l.key_weight(),
                    l.value_weight(),
                    l.output_weight(),
                    l.post_attention_norm_weight(),
                    l.gate_weight(),
                    l.up_weight(),
                    l.down_weight(),
                ]
                .map(|w| w.index()),
            );
            eps.extend([l.input_norm_epsilon(), l.post_attention_norm_epsilon()]);
        }
        let geometry = [
            f.plan.layers().len() as u64,
            owner.layout.physical_block_count() as u64,
            capacity,
            f.plan.dimensions().vocabulary_size() as u64,
        ];
        let b = &mut f.buffers;
        let mut devices: Vec<_> = f.weights.borrow_graph_weight_parents().collect();
        let base = devices.len();
        devices.extend([
            &mut b.token_ids,
            &mut b.hidden_current,
            &mut b.hidden_norm,
            &mut b.hidden_projection,
            &mut b.hidden_rotary,
            &mut b.hidden_context,
            &mut b.key_raw,
            &mut b.value_raw,
            &mut b.key_rotary,
            &mut b.gate_raw,
            &mut b.up_raw,
            &mut b.gate_activated,
            &mut b.gated_product,
            &mut b.logits,
            &mut b.embedding_error_scratch,
            &mut owner.key_cache,
            &mut owner.value_cache,
            &mut owner.absolute_rope_cos,
            &mut owner.absolute_rope_sin,
            metadata,
            result,
        ]);
        let workspace = b.gemm_workspace.as_mut().map(|w| {
            let i = devices.len();
            devices.push(w);
            i
        });
        let prefill_indices = prefill.map(|buffers| {
            let first = devices.len();
            devices.extend(buffers.iter_mut());
            std::array::from_fn(|i| first + i)
        });
        let mut plans = [
            &mut f.gemms.hidden,
            &mut f.gemms.key_value,
            &mut f.gemms.intermediate,
            &mut f.gemms.down,
            &mut f.gemms.lm_head,
        ]
        .into_iter()
        .map(|p| match p {
            PreparedLlamaGemm::Canonical(p) => Ok(p),
            _ => Err(rejected("canonical selected plans required")),
        })
        .collect::<LlamaBatchExecutorResult<Vec<_>>>()?;
        let (packed_indices, packed_plan_indices) = match packed {
            Some(packed) => {
                let first_device = devices.len();
                let first_plan = plans.len();
                devices.extend(packed.buffers.iter_mut());
                plans.extend(packed.plans.iter_mut());
                (
                    Some(std::array::from_fn(|i| first_device + i)),
                    Some(std::array::from_fn(|i| first_plan + i)),
                )
            }
            None => (None, None),
        };
        let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e);
        let mut pinned = vec![staging];
        let mut strided = Vec::new();
        let mut multi_indices = Vec::new();
        let mut shared_indices = Vec::new();
        if let Some(multi) = multi {
            let first_shared = plans.len();
            shared_indices.extend((0..multi.shared.len()).map(|i| first_shared + i));
            plans.extend(multi.shared.iter_mut());
            for scratch in &mut multi.scratch {
                let first = devices.len();
                devices.extend(scratch.iter_mut());
                multi_indices.push(std::array::from_fn(|i| {
                    if i < 14 { first + i } else { base + i + 1 }
                }));
            }
            strided.extend(multi.plans.iter_mut());
            pinned.extend(multi.staging.iter_mut());
        }
        let mut graph = BorrowedGraphResourceReservation::reserve_with_strided(
            BorrowedGraphResourceParents {
                stream,
                devices,
                pinned,
                plans,
            },
            strided,
        )
        .map_err(cuda)?;
        graph
            .record_decode_with_profile_prefill_and_packed(
                std::array::from_fn(|i| base + i),
                workspace,
                &weights,
                [0, 1, 2, 3, 4],
                0,
                geometry,
                &eps,
                if vllm_profile {
                    riley_cuda::DecodeNumericalProfile::VllmSmolP128V1
                } else if profile == LlamaRmsNormProfile::HuggingFaceSmolLm2 {
                    riley_cuda::DecodeNumericalProfile::HuggingFaceSmolLm2
                } else {
                    riley_cuda::DecodeNumericalProfile::Canonical
                },
                publish_logits,
                prefill_indices,
                packed_indices,
                packed_plan_indices,
            )
            .map_err(cuda)?;
        if !multi_indices.is_empty() {
            let mut multi_weights = weights.clone();
            let packed = packed_indices.ok_or_else(|| rejected("multi packed indices missing"))?;
            multi_weights.extend_from_slice(&packed[..60]);
            for (index, indices) in multi_indices.iter().enumerate() {
                let shared = !shared_indices.is_empty();
                let p = if shared { [shared_indices[index*3], shared_indices[index*3+1], index*5+2, index*5+3, shared_indices[index*3+2]] } else { std::array::from_fn(|i| index*5+i) };
                for full in [true, false] {
                    let staging = index + if full {1} else {4};
                    if shared { graph.append_shared_multisequence_decode(indices, &multi_weights, &p, staging, [2,4,8][index], geometry[1] as u32, full) }
                    else { graph.append_multisequence_decode(indices, &multi_weights, &p, staging, [2,4,8][index], geometry[1] as u32, full) }.map_err(cuda)?;
                }
            }
        }
        Ok(graph)
    }
}

impl PreparedLlamaBatchExecutor {
    fn full_decode_supported(&self) -> bool {
        use crate::llama::executor::config::{
            RaggedAttentionImplementation, ResidualNormImplementation,
        };
        let f = &self.owner.forward;
        let capacity = self.config.metadata().max_block_entries();
        (!self.config.vllm_smol_p128_graph()
            || (f.rms_norm_profile() == LlamaRmsNormProfile::HuggingFaceSmolLm2
                && f.plan.layers().len() == 30
                && f.plan.dimensions().hidden_size() == 576
                && f.plan.dimensions().intermediate_size() == 1536
                && f.plan.dimensions().vocabulary_size() == 49152
                && f.plan.dimensions().query_heads() == 9
                && f.plan.dimensions().key_value_heads() == 3))
            && f.plan.sequence_length() == 1
            && self.owner.layout.head_dimension() == 64
            && self.config.metadata().max_rows() == 1
            && (self.config.metadata().max_input_tokens() == 1
                || self.config.vllm_smol_p128_batched_prefill())
            && capacity <= 4096
            && self.owner.layout.physical_block_count() <= 4096
            && !f.plan.layers().is_empty()
            && f.plan.layers().len() <= 64
            && self.config.ragged_attention_reduction_profile()
                == riley_cuda::AttentionReductionProfile::CanonicalV1
            && self.config.ragged_attention_implementation()
                == RaggedAttentionImplementation::GroupedHeads
            && self.config.residual_norm_implementation() == ResidualNormImplementation::Separate
            && matches!(
                f.rms_norm_profile(),
                LlamaRmsNormProfile::Canonical | LlamaRmsNormProfile::HuggingFaceSmolLm2
            )
            && f.plan.layers().iter().all(|l| {
                l.query_bias().is_none()
                    && l.key_bias().is_none()
                    && l.value_bias().is_none()
                    && l.output_bias().is_none()
            })
            && [
                &f.gemms.hidden,
                &f.gemms.key_value,
                &f.gemms.intermediate,
                &f.gemms.down,
                &f.gemms.lm_head,
            ]
            .into_iter()
            .all(|p| match p {
                PreparedLlamaGemm::Canonical(p) => {
                    p.algorithm_metadata().split_k() <= 1
                        && p.algorithm_metadata().reduction_scheme() == 0
                        && p.algorithm_metadata().workspace_bytes() == 0
                }
                _ => false,
            })
    }

    /// Consumes this executor to generate a new single-sequence greedy continuation.
    /// Prefills the prompt with the existing eager path, then returns exactly `steps` generated
    /// tokens. The final prompt token and subsequent decode tokens use the retained graph.
    /// No sampling penalties, stop rules, externally supplied prefix, or scheduling is applied.
    /// A supported M=1,D64 canonical bucket captures all layers and reuses one graph.
    /// Other configurations select the established eager path before any graph work.
    /// Execution failures never retry against potentially mutated KV state or return
    /// partial tokens. The default batch executor dispatch is unchanged.
    /// # Errors
    /// Returns invalid request/poison/CUDA errors. `context` and `stream` must belong
    /// to the executor's original device context. No performance timing is collected.
    pub fn generate_greedy_with_decode_graph(
        self,
        context: &riley_cuda::CudaContext,
        stream: &mut CudaStream,
        prompt: &[u32],
        steps: u32,
    ) -> LlamaBatchExecutorResult<Vec<u32>> {
        self.generate_greedy_with_decode_graph_policy(
            context,
            stream,
            prompt,
            steps,
            ExecutionGraphPolicy::Auto,
        )
    }

    /// Selects disabled, automatic fallback, or required full graph generation.
    /// Require rejects unsupported configurations before prefill or capture.
    /// # Errors
    /// Returns validation, required-graph, capture, replay or completion errors.
    pub fn generate_greedy_with_decode_graph_policy(
        mut self,
        context: &riley_cuda::CudaContext,
        stream: &mut CudaStream,
        prompt: &[u32],
        steps: u32,
        policy: ExecutionGraphPolicy,
    ) -> LlamaBatchExecutorResult<Vec<u32>> {
        if self.config.vllm_smol_p128_graph() {
            return Err(rejected(
                "vllm-smol-p128-v1 requires the persistent owned executor",
            ));
        }
        use crate::llama::{LlamaBatchBlockTable, LlamaBatchRow, LlamaBatchRowKind};
        use crate::paged_kv::BLOCK_TABLE_V1_VERSION;
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        let capacity = self.config.metadata().max_block_entries();
        let vocab = self.owner.forward.plan.dimensions().vocabulary_size();
        if steps == 0
            || prompt.is_empty()
            || prompt.len() as u64 + u64::from(steps) - 1 > capacity as u64 * 16
            || prompt.iter().any(|&token| token as usize >= vocab)
            || self.config.metadata().max_output_slots() == 0
        {
            return Err(rejected(
                "invalid token, generation length or output capacity",
            ));
        }
        let use_graph = policy != ExecutionGraphPolicy::Disabled && self.full_decode_supported();
        if !use_graph {
            let decision =
                select_execution_graph(full_decode_request(policy, self.full_decode_supported()))
                    .map_err(|_| rejected("required full decode unsupported"))?;
            if decision.mode() != ExecutionMode::ExactEager {
                return Err(rejected("unexpected pre-capture execution mode"));
            }
        }
        let prepared_signature = if use_graph {
            Some(self.full_decode_signature(stream)?)
        } else {
            None
        };
        let mut generated = Vec::with_capacity(steps as usize);
        let mapping: Vec<u32> = (0..capacity).map(|i| i as u32).collect();
        let first_decode_position = (prompt.len() - 1) as u32;
        // Complete the prefix before capture; graph lifetime then excludes all eager use.
        for (position, &input_token) in prompt[..prompt.len() - 1].iter().enumerate() {
            let pos = position as u32;
            let live = position / 16 + 1;
            let mut valid = vec![16_u16; live];
            valid[live - 1] = (pos % 16 + 1) as u16;
            let input = [input_token];
            let rows = [LlamaBatchRow::new(
                1,
                if pos == 0 {
                    LlamaBatchRowKind::Prefill
                } else {
                    LlamaBatchRowKind::Decode
                },
                &input,
                pos + 1,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &mapping[..live],
                    &valid,
                    pos + 1,
                ),
                Some(0),
            )];
            self.execute(&rows, stream)?;
        }
        let mut token = prompt[prompt.len() - 1];
        if !use_graph {
            for step in 0..steps {
                let pos = first_decode_position + step;
                let live = pos as usize / 16 + 1;
                let mut valid = vec![16_u16; live];
                valid[live - 1] = (pos % 16 + 1) as u16;
                let input = [token];
                let rows = [LlamaBatchRow::new(
                    1,
                    if pos == 0 {
                        LlamaBatchRowKind::Prefill
                    } else {
                        LlamaBatchRowKind::Decode
                    },
                    &input,
                    pos + 1,
                    LlamaBatchBlockTable::new(
                        BLOCK_TABLE_V1_VERSION,
                        &mapping[..live],
                        &valid,
                        pos + 1,
                    ),
                    Some(0),
                )];
                self.execute_greedy(&rows, stream)?;
                let mut output = [0];
                self.download_greedy_tokens(&mut output, stream)?;
                token = output[0];
                generated.push(token);
            }
            self.close()?;
            return Ok(generated);
        }
        let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e);
        let metadata_bytes = ((16 + 6 * capacity + 3) & !3) + 4;
        // Generation publishes only the token and status; diagnostic tests request logits.
        let logits_bytes = 0;
        let transfer = (logits_bytes + 40).max(metadata_bytes);
        let mut staging = context
            .allocate_pinned_host_buffer((transfer * 2) as u64)
            .map_err(cuda)?;
        let mut metadata = context
            .allocate_device_buffer(metadata_bytes as u64)
            .map_err(cuda)?;
        let mut result = context.allocate_device_buffer(8).map_err(cuda)?;
        let mut payload = vec![0; transfer];
        let mut output = vec![0; transfer];
        {
            let graph = self.prepare_full_decode(
                stream,
                &mut metadata,
                &mut result,
                &mut staging,
                capacity as u64,
                false,
            )?;
            let (signature, device_bytes) =
                prepared_signature.ok_or_else(|| rejected("missing prepared signature"))?;
            let mut registered =
                RegisteredFullDecode::new(graph, signature, (transfer * 2) as u64, device_bytes)?;
            for step in 0..steps {
                let pos = first_decode_position + step;
                let live = pos as usize / 16 + 1;
                payload.fill(0);
                payload[..4].copy_from_slice(&token.to_le_bytes());
                payload[4..8].copy_from_slice(&pos.to_le_bytes());
                payload[12..16].copy_from_slice(&(live as u32).to_le_bytes());
                for (i, physical) in mapping[..live].iter().enumerate() {
                    payload[16 + 4 * i..20 + 4 * i].copy_from_slice(&physical.to_le_bytes());
                    let valid = if i + 1 < live {
                        16_u16
                    } else {
                        (pos % 16 + 1) as u16
                    };
                    let o = 16 + 4 * capacity + 2 * i;
                    payload[o..o + 2].copy_from_slice(&valid.to_le_bytes());
                }
                registered.replay(policy, &payload, &mut output)?;
                let read = |offset| {
                    u32::from_le_bytes(
                        output[offset..offset + 4]
                            .try_into()
                            .expect("fixed status layout"),
                    )
                };
                if read(logits_bytes + 4) != 0
                    || read(logits_bytes + 8) != 32
                    || read(logits_bytes + 12) != 0
                    || read(logits_bytes) >= vocab as u32
                {
                    return Err(rejected("completed decode returned invalid output status"));
                }
                token = read(logits_bytes);
                generated.push(token);
            }
            registered.close()?;
        }
        metadata.close().map_err(cuda)?;
        result.close().map_err(cuda)?;
        staging.close().map_err(cuda)?;
        self.close()?;
        Ok(generated)
    }
}

use crate::llama::graph::*;
use crate::llama::graph_registry::{
    GraphEntryFootprint, GraphRegistry, GraphRegistryEntry, GraphRegistryEntryState,
    GraphRegistryLimits, GraphReplayMode, GraphReplaySlot,
};
use crate::llama::graph_registry_dispatch::{
    GraphRegistryDispatchDecision, select_registered_execution_graph,
};
use sha2::{Digest, Sha256};

fn full_decode_request(policy: ExecutionGraphPolicy, supported: bool) -> GraphDispatchRequest {
    GraphDispatchRequest::new(
        policy,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::PureDecode,
            true,
            true,
            true,
            GraphCaptureSafety::new(
                GraphSamplingBackend::GpuGreedy,
                if supported {
                    GraphOperatorCapability::Supported
                } else {
                    GraphOperatorCapability::Unsupported
                },
                supported,
            ),
        ),
        GraphInventoryState::NotPrepared,
    )
}

// This registry cannot escape its retained owner or resolve another owner's slot.
// Never register Prepared until native capture and instantiate have succeeded.
struct RegisteredFullDecode<'a> {
    graph: BorrowedGraphResourceReservation<'a>,
    registry: GraphRegistry<1>,
    signature: GraphSignature,
    poisoned: bool,
}
impl<'a> RegisteredFullDecode<'a> {
    fn new(
        graph: BorrowedGraphResourceReservation<'a>,
        signature: GraphSignature,
        host: u64,
        device: u64,
    ) -> LlamaBatchExecutorResult<Self> {
        let registry = GraphRegistry::try_new(
            GraphRegistryLimits::new(1, 1, 0, host, device),
            &[GraphRegistryEntry::new(
                signature,
                GraphReplayMode::FullGraph,
                GraphReplaySlot::new(0),
                GraphRegistryEntryState::Prepared,
                GraphEntryFootprint::new(host, device),
            )],
        )
        .map_err(|_| rejected("invalid full decode registry accounting"))?;
        Ok(Self {
            graph,
            registry,
            signature,
            poisoned: false,
        })
    }
    fn select(
        &self,
        policy: ExecutionGraphPolicy,
        signature: GraphSignature,
    ) -> LlamaBatchExecutorResult<GraphRegistryDispatchDecision> {
        if self.poisoned {
            return Err(rejected("registered decode owner poisoned"));
        }
        select_registered_execution_graph(
            GraphDispatchRequest::new(
                policy,
                GraphDispatchEligibility::new(
                    GraphWorkloadStage::PureDecode,
                    true,
                    true,
                    true,
                    GraphCaptureSafety::new(
                        GraphSamplingBackend::GpuGreedy,
                        GraphOperatorCapability::Supported,
                        true,
                    ),
                ),
                GraphInventoryState::NotPrepared,
            ),
            signature,
            &self.registry,
        )
        .map_err(|_| rejected("required registered decode unavailable"))
    }
    fn replay(
        &mut self,
        policy: ExecutionGraphPolicy,
        payload: &[u8],
        output: &mut [u8],
    ) -> LlamaBatchExecutorResult<()> {
        match self.select(policy, self.signature)? {
            GraphRegistryDispatchDecision::FullGraph { replay_slot }
                if replay_slot == GraphReplaySlot::new(0) => {}
            _ => return Err(rejected("registered decode slot mismatch")),
        }
        let result = self
            .graph
            .replay_transfer(payload)
            .and_then(|()| self.graph.read_transfer(output));
        if result.is_err() {
            self.poisoned = true;
        }
        result.map_err(|e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e))
    }
    fn close(self) -> LlamaBatchExecutorResult<()> {
        self.graph
            .close()
            .map_err(|e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e))
    }
}

impl PreparedLlamaBatchExecutor {
    // Cold content identity; never use paths, pointers, Debug formatting or truncated
    // algorithm IDs as a model/plan key. Hash actual uploaded weight/table bytes.
    fn full_decode_signature(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<(GraphSignature, u64)> {
        self.full_decode_signature_with_packed(stream, None)
    }

    fn full_decode_signature_with_packed(
        &mut self,
        stream: &mut CudaStream,
        packed: Option<&PackedDecodeParents>,
    ) -> LlamaBatchExecutorResult<(GraphSignature, u64)> {
        if !self.full_decode_supported() {
            return Err(rejected("unsupported registry signature"));
        }
        let owner = &mut self.owner;
        let f = &mut owner.forward;
        let dims = f.plan.dimensions();
        let capacity = self.config.metadata().max_block_entries() as u64;
        let batched_prefill = self.config.vllm_smol_p128_batched_prefill();
        if packed.is_some() && !batched_prefill {
            return Err(rejected("packed decode requires P128 prefill"));
        }
        let implementation = if packed.is_some() {
            0xF108
        } else if batched_prefill {
            0xF103
        } else if self.config.vllm_smol_p128_graph() {
            0xF102
        } else {
            0xF001
        };
        let profile = if self.config.vllm_smol_p128_graph() {
            2_u32
        } else {
            match f.rms_norm_profile() {
                LlamaRmsNormProfile::Canonical => 0_u32,
                LlamaRmsNormProfile::HuggingFaceSmolLm2 => 1,
                _ => unreachable!(),
            }
        };
        let mut digest = Sha256::new();
        digest.update(b"riley.full-decode.owner-content.v1\0");
        digest.update(profile.to_le_bytes());
        digest.update(f.plan.rope_theta().to_bits().to_le_bytes());
        digest.update(f.plan.final_norm_epsilon().to_bits().to_le_bytes());
        for id in [
            f.plan.embedding_weight(),
            f.plan.final_norm_weight(),
            f.plan.lm_head_weight(),
        ] {
            digest.update((id.index() as u64).to_le_bytes());
        }
        for l in f.plan.layers() {
            for id in [
                l.input_norm_weight(),
                l.query_weight(),
                l.key_weight(),
                l.value_weight(),
                l.output_weight(),
                l.post_attention_norm_weight(),
                l.gate_weight(),
                l.up_weight(),
                l.down_weight(),
            ] {
                digest.update((id.index() as u64).to_le_bytes());
            }
            digest.update(l.input_norm_epsilon().to_bits().to_le_bytes());
            digest.update(l.post_attention_norm_epsilon().to_bits().to_le_bytes());
        }
        let mut native_identity = None;
        for p in [
            &f.gemms.hidden,
            &f.gemms.key_value,
            &f.gemms.intermediate,
            &f.gemms.down,
            &f.gemms.lm_head,
        ] {
            let PreparedLlamaGemm::Canonical(p) = p else {
                unreachable!()
            };
            let m = p.algorithm_metadata();
            let identity = (
                m.compute_capability(),
                m.runtime_version(),
                m.cublaslt_version(),
            );
            if native_identity.is_some_and(|previous| previous != identity) {
                return Err(rejected("selected plans have different device identities"));
            }
            native_identity = Some(identity);
            for v in [
                m.backend_id(),
                m.algorithm_id() as u32,
                m.tile_id(),
                m.stages_id(),
                m.split_k(),
                m.reduction_scheme(),
                m.cta_swizzling(),
                m.custom_option(),
                u32::from(m.deterministic()),
            ] {
                digest.update(v.to_le_bytes());
            }
            let (m_rows, n, k) = m.dimensions();
            for v in [
                m_rows,
                n,
                k,
                m.workspace_bytes(),
                m.numerical_implementation_flags(),
            ] {
                digest.update(v.to_le_bytes());
            }
        }
        let mut device_bytes = 0_u64;
        let mut chunk = vec![
            0;
            usize::try_from(f.io_staging.byte_len().min(1024 * 1024))
                .map_err(|_| rejected("staging length overflow"))?
        ];
        if chunk.is_empty() {
            return Err(rejected("content identity requires staging"));
        }
        for (index, buffer) in f
            .weights
            .borrow_graph_weight_parents()
            .chain([&mut owner.absolute_rope_cos, &mut owner.absolute_rope_sin])
            .enumerate()
        {
            digest.update((index as u64).to_le_bytes());
            digest.update(buffer.byte_len().to_le_bytes());
            device_bytes = device_bytes
                .checked_add(buffer.byte_len())
                .ok_or_else(|| rejected("footprint overflow"))?;
            let mut offset = 0;
            while offset < buffer.byte_len() {
                let count = (buffer.byte_len() - offset).min(chunk.len() as u64) as usize;
                buffer
                    .download_to_slice(offset, &mut chunk[..count], &mut f.io_staging, stream)
                    .map_err(|e| {
                        cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e)
                    })?;
                digest.update(&chunk[..count]);
                offset += count as u64;
            }
        }
        let b = &f.buffers;
        for buffer in [
            &b.token_ids,
            &b.hidden_current,
            &b.hidden_norm,
            &b.hidden_projection,
            &b.hidden_rotary,
            &b.hidden_context,
            &b.key_raw,
            &b.value_raw,
            &b.key_rotary,
            &b.gate_raw,
            &b.up_raw,
            &b.gate_activated,
            &b.gated_product,
            &b.logits,
            &b.embedding_error_scratch,
            &owner.key_cache,
            &owner.value_cache,
        ]
        .into_iter()
        .chain(b.gemm_workspace.as_ref())
        {
            device_bytes = device_bytes
                .checked_add(buffer.byte_len())
                .ok_or_else(|| rejected("footprint overflow"))?;
            digest.update(buffer.byte_len().to_le_bytes());
        }
        let legacy_metadata_bytes = ((16 + 6 * capacity + 3) & !3) + 4;
        let metadata_bytes = legacy_metadata_bytes
            + if batched_prefill {
                PREFILL128_METADATA_BYTES as u64
            } else {
                0
            };
        if batched_prefill {
            digest.update(PREFILL128_MAGIC.to_le_bytes());
            for size in PREFILL128_SCRATCH_BYTES {
                digest.update(size.to_le_bytes());
                device_bytes = device_bytes
                    .checked_add(size)
                    .ok_or_else(|| rejected("prefill footprint overflow"))?;
            }
        }
        if let Some(packed) = packed {
            digest.update(b"packed-qkv-gate-up-decode-v1\0");
            digest.update(packed.digest);
            digest.update(packed.device_bytes.to_le_bytes());
            device_bytes = device_bytes
                .checked_add(packed.device_bytes)
                .ok_or_else(|| rejected("packed decode footprint overflow"))?;
        }
        let mut layout = Sha256::new();
        layout.update(b"riley.full-decode.packed-token-position-blocks-slot.v1\0");
        if batched_prefill {
            layout.update(b"prefill128-v1\0");
            layout.update(PREFILL128_MAGIC.to_le_bytes());
        }
        for v in [
            capacity,
            owner.layout.physical_block_count() as u64,
            metadata_bytes,
            8,
            16,
            16 + 4 * capacity,
            legacy_metadata_bytes - 4,
            40,
        ] {
            layout.update(v.to_le_bytes());
        }
        let ((major, minor), runtime, cublas) =
            native_identity.ok_or_else(|| rejected("missing native plan identity"))?;
        if self.config.vllm_smol_p128_graph()
            && (major != 8 || minor != 9 || runtime != 13000 || cublas != 130101)
        {
            return Err(rejected("vllm-smol-p128-v1 environment mismatch"));
        }
        let device = GraphDeviceSignature::new(
            major,
            minor,
            u32::try_from(runtime).map_err(|_| rejected("invalid runtime version"))?,
            u32::try_from(cublas).map_err(|_| rejected("invalid cuBLASLt version"))?,
            riley_cuda::EXPECTED_ABI_VERSION,
        );
        let signature = GraphSignature::new(
            GraphStaticSignature::new(
                GraphModelSignature::new(
                    GraphModelArchitecture::LlamaDecoder,
                    1,
                    GraphRevisionFingerprint::from_bytes(digest.finalize().into()),
                    1,
                ),
                device,
                GraphTensorSignature::new(
                    GraphDataType::BFloat16,
                    GraphDataType::BFloat16,
                    GraphComputeType::Float32,
                ),
                GraphGeometrySignature::new(
                    f.plan.layers().len() as u32,
                    dims.hidden_size() as u32,
                    dims.intermediate_size() as u32,
                    dims.vocabulary_size() as u32,
                    dims.query_heads() as u32,
                    dims.key_value_heads() as u32,
                    dims.head_dimension() as u32,
                ),
                GraphLayoutSignature::new(
                    (capacity * 16) as u32,
                    16,
                    1,
                    GraphMetadataLayoutSignature::new(
                        if batched_prefill { 3 } else { 2 },
                        layout.finalize().into(),
                    ),
                ),
                GraphImplementationSignature::new(
                    GraphImplementationId::new(implementation),
                    GraphImplementationId::new(implementation),
                    GraphImplementationId::new(implementation),
                    GraphImplementationId::new(implementation),
                    GraphGemmPlanSetId::new(if packed.is_some() { 2 } else { 1 }),
                    GraphReductionPolicyId::new(profile),
                ),
            ),
            GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                1,
                GraphSamplingBackend::GpuGreedy,
            ),
        );
        Ok((signature, device_bytes + metadata_bytes + 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
    };
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;
    use riley_cuda::CudaRuntime;
    use riley_model::{LoadLimits, LoadedModel};
    use sha2::{Digest, Sha256};
    #[test]
    #[ignore = "requires CUDA and SmolLM2 checkpoint"]
    fn full_decode_smol_64_tokens() -> Result<(), Box<dyn std::error::Error>> {
        case("RILEY_REAL_CHECKPOINT", 64)
    }
    #[test]
    #[ignore = "requires CUDA canonical fixture"]
    fn full_decode_canonical_64_tokens() -> Result<(), Box<dyn std::error::Error>> {
        case("RILEY_CANONICAL_CHECKPOINT", 64)
    }
    #[test]
    #[ignore = "requires CUDA odd-layer fixture"]
    fn full_decode_odd_64_tokens() -> Result<(), Box<dyn std::error::Error>> {
        case("RILEY_ODD_CHECKPOINT", 64)
    }
    fn case(env: &str, steps: u32) -> Result<(), Box<dyn std::error::Error>> {
        let model = LoadedModel::load(
            std::path::Path::new(&std::env::var_os(env).ok_or("checkpoint absent")?),
            LoadLimits::default(),
        )?;
        let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
        let mut eager_stream = context.create_stream()?;
        let mut stream = context.create_stream()?;
        let capacity = 4_usize;
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, capacity, 1, capacity)?,
            PreparedLlamaForwardConfig::default(),
        )
        .with_grouped_ragged_attention_heads()
        .with_iteration_batch_completion()
        .with_packed_async_metadata();
        let mut baseline =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut eager_stream, config)?;
        let mut candidate =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        let bytes = baseline.owner.forward.buffers.logits.byte_len() as usize;
        let metadata_bytes = ((16 + 6 * capacity + 3) & !3) + 4;
        let transfer = (bytes + 40).max(metadata_bytes);
        let mut staging = context.allocate_pinned_host_buffer((2 * transfer) as u64)?;
        let mut metadata = context.allocate_device_buffer(metadata_bytes as u64)?;
        let mut result = context.allocate_device_buffer(8)?;
        let mut io =
            context.allocate_pinned_host_buffer(candidate.owner.layout.bytes_per_kind())?;
        let zeros = vec![0; candidate.owner.layout.bytes_per_kind() as usize];
        for d in [
            &mut baseline.owner.key_cache,
            &mut baseline.owner.value_cache,
        ] {
            d.upload_from_slice(0, &zeros, &mut io, &mut eager_stream)?;
        }
        for d in [
            &mut candidate.owner.key_cache,
            &mut candidate.owner.value_cache,
        ] {
            d.upload_from_slice(0, &zeros, &mut io, &mut stream)?;
        }
        let (signature, footprint) = candidate.full_decode_signature(&mut stream)?;
        let (same_signature, same_footprint) = baseline.full_decode_signature(&mut eager_stream)?;
        assert_eq!(
            signature, same_signature,
            "stable actual-owner content identity"
        );
        assert_eq!(footprint, same_footprint);
        if env == "RILEY_CANONICAL_CHECKPOINT" {
            let id = candidate.owner.forward.plan.final_norm_weight();
            let weight = candidate.owner.forward.weights.borrow_graph_weight(id)?;
            let mut original = vec![0; weight.byte_len() as usize];
            weight.download_to_slice(0, &mut original, &mut io, &mut stream)?;
            let mut changed = original.clone();
            changed[0] ^= 1;
            weight.upload_from_slice(0, &changed, &mut io, &mut stream)?;
            assert_ne!(
                candidate.full_decode_signature(&mut stream)?.0,
                signature,
                "weight content changes exact key"
            );
            candidate
                .owner
                .forward
                .weights
                .borrow_graph_weight(id)?
                .upload_from_slice(0, &original, &mut io, &mut stream)?;
            candidate.config = PreparedLlamaBatchExecutorConfig::new(
                LlamaBatchMetadataConfig::new(1, 1, 3, 1, 4)?,
                PreparedLlamaForwardConfig::default(),
            )
            .with_grouped_ragged_attention_heads()
            .with_iteration_batch_completion()
            .with_packed_async_metadata();
            assert_ne!(
                candidate.full_decode_signature(&mut stream)?.0,
                signature,
                "metadata capacity changes exact key"
            );
            candidate.config = config;
            assert_eq!(
                candidate.full_decode_signature(&mut stream)?.0,
                signature,
                "restored key"
            );
            println!(
                "G03_REGISTERED_KEY_CHANGE weight_content=true metadata_capacity=true restored=true"
            );
        }
        let stats = context.allocation_stats()?;
        let mut graph = candidate.prepare_full_decode(
            &mut stream,
            &mut metadata,
            &mut result,
            &mut staging,
            capacity as u64,
            true,
        )?;
        let mut output = vec![0; transfer];
        assert!(graph.read_transfer(&mut output).is_err());
        let mapping = [2_u32, 0, 3, 1];
        let mut token = 504_u32;
        let mut evidence = Sha256::new();
        let mut tokens = Vec::new();
        for pos in 0..steps {
            let live = pos as usize / 16 + 1;
            let mut valid = vec![16_u16; live];
            valid[live - 1] = (pos % 16 + 1) as u16;
            let input = [token];
            let rows = [LlamaBatchRow::new(
                1,
                if pos == 0 {
                    LlamaBatchRowKind::Prefill
                } else {
                    LlamaBatchRowKind::Decode
                },
                &input,
                pos + 1,
                LlamaBatchBlockTable::new(
                    BLOCK_TABLE_V1_VERSION,
                    &mapping[..live],
                    &valid,
                    pos + 1,
                ),
                Some(0),
            )];
            baseline.execute(&rows, &mut eager_stream)?;
            let mut expected = vec![0; bytes];
            baseline.download_logits(&mut expected, &mut eager_stream)?;
            let mut payload = vec![0; transfer];
            payload[..4].copy_from_slice(&token.to_le_bytes());
            payload[4..8].copy_from_slice(&pos.to_le_bytes());
            payload[12..16].copy_from_slice(&(live as u32).to_le_bytes());
            for i in 0..live {
                payload[16 + 4 * i..20 + 4 * i].copy_from_slice(&mapping[i].to_le_bytes());
                let o = 16 + 4 * capacity + 2 * i;
                payload[o..o + 2].copy_from_slice(&valid[i].to_le_bytes());
            }
            // Every rejection revokes the previous completed result; corrected input remains usable.
            for offset in [0, 4, 8, 12, 16, ((16 + 6 * capacity + 3) & !3)] {
                let mut bad = payload.clone();
                bad[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
                assert!(graph.replay_transfer(&bad).is_err());
                assert!(graph.read_transfer(&mut output).is_err());
            }
            let mut invalid_prefix = payload.clone();
            invalid_prefix[16 + 4 * capacity..18 + 4 * capacity]
                .copy_from_slice(&0_u16.to_le_bytes());
            assert!(graph.replay_transfer(&invalid_prefix).is_err());
            if live > 1 {
                let mut duplicate = payload.clone();
                duplicate[20..24].copy_from_slice(&mapping[0].to_le_bytes());
                assert!(graph.replay_transfer(&duplicate).is_err());
            }
            if live < capacity {
                let mut padding = payload.clone();
                padding[16 + 4 * live..20 + 4 * live].copy_from_slice(&1_u32.to_le_bytes());
                assert!(graph.replay_transfer(&padding).is_err());
            }
            assert!(graph.replay_transfer(&payload[..transfer - 1]).is_err());
            graph.replay_transfer(&payload)?;
            graph.read_transfer(&mut output)?;
            assert_eq!(
                &output[..bytes],
                &expected,
                "full decode logits {env} position {pos}"
            );
            assert_eq!(
                u32::from_le_bytes(output[bytes + 4..bytes + 8].try_into()?),
                0
            );
            assert_eq!(
                u32::from_le_bytes(output[bytes + 12..bytes + 16].try_into()?),
                0,
                "embedding status"
            );
            let best = expected
                .chunks_exact(2)
                .enumerate()
                .fold((f32::NEG_INFINITY, 0_u32), |best, (i, b)| {
                    let v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
                    if v > best.0 { (v, i as u32) } else { best }
                })
                .1;
            token = u32::from_le_bytes(output[bytes..bytes + 4].try_into()?);
            assert_eq!(token, best);
            tokens.push(token);
            evidence.update(&output[..bytes + 40]);
        }
        graph.close()?;
        assert_eq!(context.allocation_stats()?, stats);
        for (a, b) in [
            (
                &mut baseline.owner.key_cache,
                &mut candidate.owner.key_cache,
            ),
            (
                &mut baseline.owner.value_cache,
                &mut candidate.owner.value_cache,
            ),
        ] {
            let mut x = zeros.clone();
            let mut y = zeros.clone();
            a.download_to_slice(0, &mut x, &mut io, &mut eager_stream)?;
            b.download_to_slice(0, &mut y, &mut io, &mut stream)?;
            assert_eq!(x, y, "all-layer full cache parity");
            evidence.update(&y);
        }
        println!(
            "G03_FULL_DECODE checkpoint={env} steps={steps} blocks=4 all_logits_exact=true all_cache_exact=true sha256={}",
            evidence
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        // A registry key/slot is metadata, never authority to replay a different owner.
        {
            let graph = candidate.prepare_full_decode(
                &mut stream,
                &mut metadata,
                &mut result,
                &mut staging,
                capacity as u64,
                false,
            )?;
            let mut registered =
                RegisteredFullDecode::new(graph, signature, (transfer * 2) as u64, footprint)?;
            assert_eq!(
                registered
                    .select(ExecutionGraphPolicy::Auto, signature)?
                    .mode(),
                ExecutionMode::FullGraph
            );
            assert_eq!(
                registered
                    .select(ExecutionGraphPolicy::Require, signature)?
                    .replay_slot(),
                Some(GraphReplaySlot::new(0))
            );
            assert_eq!(
                registered
                    .select(ExecutionGraphPolicy::Disabled, signature)?
                    .mode(),
                ExecutionMode::ExactEager
            );
            let miss = GraphSignature::new(
                signature.static_signature(),
                GraphIterationSignature::new(
                    GraphWorkloadStage::PureDecode,
                    2,
                    GraphSamplingBackend::GpuGreedy,
                ),
            );
            assert_eq!(
                registered.select(ExecutionGraphPolicy::Auto, miss)?.mode(),
                ExecutionMode::ExactEager
            );
            assert!(
                registered
                    .select(ExecutionGraphPolicy::Require, miss)
                    .is_err()
            );
            let expected_registry = registered.registry.clone();
            registered.registry = GraphRegistry::try_new(
                GraphRegistryLimits::new(1, 1, 0, u64::MAX, u64::MAX),
                &[GraphRegistryEntry::new(
                    signature,
                    GraphReplayMode::FullGraph,
                    GraphReplaySlot::new(1),
                    GraphRegistryEntryState::Prepared,
                    GraphEntryFootprint::new(0, 0),
                )],
            )?;
            assert!(
                registered
                    .replay(ExecutionGraphPolicy::Auto, &[], &mut output)
                    .is_err()
            );
            registered.registry = expected_registry;
            assert!(
                registered
                    .replay(ExecutionGraphPolicy::Auto, &[], &mut output)
                    .is_err()
            );
            assert!(
                registered
                    .select(ExecutionGraphPolicy::Auto, signature)
                    .is_err()
            );
            registered.close()?;
        }
        println!(
            "G03_REGISTERED checkpoint={env} stable_key=true exact_slot=true miss_rejected=true poison_rejected=true signature={}",
            signature
                .fingerprint()
                .as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let generation =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        assert!(generation.full_decode_supported());
        assert_eq!(
            generation.generate_greedy_with_decode_graph(&context, &mut stream, &[504], steps)?,
            tokens
        );
        let required = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        assert_eq!(
            required.generate_greedy_with_decode_graph_policy(
                &context,
                &mut stream,
                &[504],
                steps,
                ExecutionGraphPolicy::Require
            )?,
            tokens
        );
        let disabled = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        assert_eq!(
            disabled.generate_greedy_with_decode_graph_policy(
                &context,
                &mut stream,
                &[504],
                steps,
                ExecutionGraphPolicy::Disabled
            )?,
            tokens
        );
        let legacy = config.with_legacy_ragged_attention_heads();
        let unsupported =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, legacy)?;
        assert!(
            unsupported
                .generate_greedy_with_decode_graph_policy(
                    &context,
                    &mut stream,
                    &[504],
                    steps,
                    ExecutionGraphPolicy::Require
                )
                .is_err()
        );
        println!(
            "G03_REGISTERED_POLICY checkpoint={env} auto=true require=true disabled=true unsupported_require_rejected=true"
        );
        let fallback = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, legacy)?;
        assert!(!fallback.full_decode_supported());
        assert_eq!(
            fallback.generate_greedy_with_decode_graph(&context, &mut stream, &[504], steps)?,
            tokens
        );
        let prompt: Vec<u32> = (0..17).map(|i| if i == 0 { 504 } else { i }).collect();
        let with_prefix =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        let eager_prefix =
            PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, legacy)?;
        let expected_prefix =
            eager_prefix.generate_greedy_with_decode_graph(&context, &mut stream, &prompt, 48)?;
        assert_eq!(
            with_prefix.generate_greedy_with_decode_graph(&context, &mut stream, &prompt, 48)?,
            expected_prefix
        );
        println!(
            "G03_FULL_PREFIX checkpoint={env} prompt_tokens=17 decode_tokens=48 eager_exact=true"
        );
        if env == "RILEY_CANONICAL_CHECKPOINT" {
            let mut bad =
                PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
            let id = bad.owner.forward.plan.final_norm_weight();
            let weight = bad.owner.forward.weights.borrow_graph_weight(id)?;
            let nonfinite = vec![0x7fc0_u16; weight.byte_len() as usize / 2]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            weight.upload_from_slice(0, &nonfinite, &mut io, &mut stream)?;
            let failure = bad
                .generate_greedy_with_decode_graph(&context, &mut stream, &[504], 1)
                .unwrap_err();
            assert!(failure.to_string().contains("invalid output status"));
            println!("G03_FULL_FAILURE nonfinite_output_rejected=true no_partial_tokens=true");
        }
        assert_eq!(
            context.allocation_stats()?,
            stats,
            "generation and failure cleanup"
        );
        println!(
            "G03_FULL_ENTRY checkpoint={env} steps={steps} graph_exact=true eager_fallback_exact=true"
        );
        baseline.close()?;
        candidate.close()?;
        metadata.close()?;
        result.close()?;
        staging.close()?;
        io.close()?;
        stream.close()?;
        eager_stream.close()?;
        context.close()?;
        Ok(())
    }
}

// The owned session retains the original scheduler KV pool and selected plans.
// A single owned registry slot identifies the complete retained graph bundle.
// The fixed-P128 profile validates the scheduler stage against fresh position
// metadata below; native dispatch selects the corresponding stage DAG. Generic
// partial-prefill registry eligibility remains unchanged.
struct FullDecodeParents {
    executor: PreparedLlamaBatchExecutor,
    stream: CudaStream,
    metadata: CudaDeviceBuffer,
    result: CudaDeviceBuffer,
    staging: CudaPinnedHostBuffer,
    prefill: Vec<CudaDeviceBuffer>,
    packed: Option<PackedDecodeParents>,
    multi: Option<MultiDecodeParents>,
}

/// Persistent, thread-confined M=1 graph session on the executor's actual KV pool.
/// The owner exposes no eager access while CUDA graph leases are active.
/// Full logits are retained so ordinary CPU sampling and stop masks remain valid.
pub struct OwnedLlamaDecodeExecutor {
    multi_plan_identity: Vec<u8>,
    graph: riley_cuda::OwnedGraphResourceReservation<FullDecodeParents>,
    registry: GraphRegistry<1>,
    signature: GraphSignature,
    metadata: crate::llama::PreparedLlamaBatchMetadata,
    vocabulary_size: usize,
    maximum_position_count: usize,
    payload: Vec<u8>,
    output: Vec<u8>,
    poisoned: bool,
    output_ready: bool,
    replays: u64,
    vllm_smol_p128_graph: bool,
    batched_prefill: bool,
}
impl PreparedLlamaBatchExecutor {
    /// Whether this exact prepared owner supports a persistent single-row graph.
    #[must_use]
    pub fn supports_owned_decode_graph(&self) -> bool {
        self.full_decode_supported()
    }

    /// Moves all execution resources into a persistent graph. No pointers into
    /// movable Rust values are retained; the CUDA layer owns stable native handles.
    /// # Errors
    /// Rejects unsupported configurations or failed capture before publication.
    pub fn into_owned_decode_graph(
        self,
        context: &riley_cuda::CudaContext,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {
        self.into_owned_decode_graph_with_catalog(context, false)
    }

    // Internal until the scheduler/codec adapter supplies live reservation authority.
    pub(crate) fn into_owned_decode_graph_with_catalog(
        self,
        context: &riley_cuda::CudaContext,
        catalog: bool,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {
        self.into_owned_decode_graph_with_catalog_profile(context, catalog, false)
    }
    pub(crate) fn into_owned_decode_graph_with_catalog_profile(
        mut self, context: &riley_cuda::CudaContext, catalog: bool, shared_rows: bool,
    ) -> LlamaBatchExecutorResult<OwnedLlamaDecodeExecutor> {
        if !self.full_decode_supported() {
            return Err(rejected("unsupported owned graph"));
        }
        let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e);
        let mut stream = context.create_stream().map_err(cuda)?;
        let vllm_smol_p128_graph = self.config.vllm_smol_p128_graph();
        let batched_prefill = self.config.vllm_smol_p128_batched_prefill();
        if catalog && (!batched_prefill || self.config.metadata().max_block_entries() != 10) {
            return Err(rejected("catalog requires packed P128/C10"));
        }
        let multi = if catalog {
            Some(MultiDecodeParents::prepare(context, shared_rows)?)
        } else {
            None
        };
        let packed = if batched_prefill {
            Some(PackedDecodeParents::prepare(
                &mut self,
                context,
                &mut stream,
            )?)
        } else {
            None
        };
        let (signature, device_bytes) =
            self.full_decode_signature_with_packed(&mut stream, packed.as_ref())?;
        // The registry selects the original M1/P128 DAG, but its retained-owner
        // footprint must include every extra catalog allocation as well.
        let multi_device_bytes = multi.as_ref().map_or(0, |m| {
            m.scratch
                .iter()
                .flatten()
                .map(CudaDeviceBuffer::byte_len)
                .sum::<u64>()
        });
        let multi_pinned_bytes = multi.as_ref().map_or(0, |m| {
            m.staging
                .iter()
                .map(CudaPinnedHostBuffer::byte_len)
                .sum::<u64>()
        });
        let device_bytes = device_bytes
            .checked_add(multi_device_bytes)
            .ok_or_else(|| rejected("catalog device footprint overflow"))?;

        let multi_plan_identity = format!("shared-rows-v1:{}:{:?}", shared_rows, multi.as_ref().map(|m| m.shared.iter().map(|p| p.algorithm_metadata()).collect::<Vec<_>>())).into_bytes();
        let vocabulary_size = self.vocabulary_size();
        let maximum_position_count = self.maximum_position_count()?;
        let config = self.config.metadata();
        let capacity = config.max_block_entries();
        let metadata_bytes = ((16 + 6 * capacity + 3) & !3)
            + 4
            + if batched_prefill {
                PREFILL128_METADATA_BYTES
            } else {
                0
            };
        let transfer = (vocabulary_size * 2 + 40).max(metadata_bytes);
        let metadata = crate::llama::PreparedLlamaBatchMetadata::prepare(config)?;
        let parents = FullDecodeParents {
            packed,
            multi,
            executor: self,
            stream,
            metadata: context
                .allocate_device_buffer(metadata_bytes as u64)
                .map_err(cuda)?,
            result: context.allocate_device_buffer(8).map_err(cuda)?,
            staging: context
                .allocate_pinned_host_buffer((transfer * 2) as u64)
                .map_err(cuda)?,
            prefill: if batched_prefill {
                PREFILL128_SCRATCH_BYTES
                    .into_iter()
                    .map(|bytes| context.allocate_device_buffer(bytes).map_err(cuda))
                    .collect::<LlamaBatchExecutorResult<Vec<_>>>()?
            } else {
                Vec::new()
            },
        };
        let mut registry = None;
        let graph = riley_cuda::OwnedGraphResourceReservation::prepare(parents, |p| {
            let graph = p.executor.prepare_full_decode_with_prefill(
                &mut p.stream,
                &mut p.metadata,
                &mut p.result,
                &mut p.staging,
                capacity as u64,
                true,
                if batched_prefill {
                    Some(&mut p.prefill)
                } else {
                    None
                },
                p.packed.as_mut(),
                p.multi.as_mut(),
            )?;
            // Publish only a captured, instantiated exact-owner registry entry.
            let registered = RegisteredFullDecode::new(
                graph,
                signature,
                (transfer * 2) as u64 + multi_pinned_bytes,
                device_bytes,
            )?;
            if registered
                .select(ExecutionGraphPolicy::Require, signature)?
                .mode()
                != ExecutionMode::FullGraph
            {
                return Err(rejected("owned registry slot unavailable"));
            }
            registry = Some(registered.registry);
            Ok(registered.graph)
        })?;
        Ok(OwnedLlamaDecodeExecutor {
            multi_plan_identity,
            graph,
            registry: registry.ok_or_else(|| rejected("registry missing"))?,
            signature,
            metadata,
            vocabulary_size,
            maximum_position_count,
            payload: vec![0; transfer],
            output: vec![0; transfer],
            poisoned: false,
            output_ready: false,
            replays: 0,
            vllm_smol_p128_graph,
            batched_prefill,
        })
    }
}
impl OwnedLlamaDecodeExecutor {
    /// Versioned request bounds, checked before any output is published.
    /// # Errors
    /// Rejects requests outside the explicitly selected numerical profile.
    pub fn validate_request_shape(
        &self,
        prompt_tokens: usize,
        output_tokens: usize,
    ) -> LlamaBatchExecutorResult<()> {
        if self.vllm_smol_p128_graph
            && (prompt_tokens != 128 || output_tokens == 0 || output_tokens > 32)
        {
            return Err(rejected(
                "vllm-smol-p128-v1 requires 128 prompt tokens and 1..32 output tokens",
            ));
        }
        Ok(())
    }
    /// Stable arithmetic identity for request admission and evidence.
    #[must_use]
    pub const fn numerical_profile_id(&self) -> &'static str {
        if self.vllm_smol_p128_graph {
            "vllm-smol-p128-v1"
        } else {
            "existing"
        }
    }

    /// Exact metadata bounds retained from the original executor.
    #[must_use]
    pub fn metadata_config(&self) -> crate::llama::LlamaBatchMetadataConfig {
        self.metadata.config()
    }
    /// Exact vocabulary width.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
    /// Absolute RoPE position capacity.
    #[must_use]
    pub const fn maximum_position_count(&self) -> usize {
        self.maximum_position_count
    }
    /// Successful completed launches on this one retained graph.
    #[must_use]
    pub const fn replay_count(&self) -> u64 {
        self.replays
    }
    /// Exact device argmax from the last successful execution.
    #[must_use]
    pub fn greedy_token(&self) -> LlamaBatchExecutorResult<u32> {
        if self.poisoned || !self.output_ready {
            return Err(LlamaBatchExecutorError::OutputNotReady);
        }
        let offset = self.vocabulary_size * 2;
        Ok(u32::from_le_bytes(
            self.output[offset..offset + 4].try_into().expect("token"),
        ))
    }
    /// Executes one decode token or the configured complete P128 prefill.
    /// No retry or further publication is permitted after an execution error.
    /// # Errors
    /// Rejects invalid metadata before dispatch, or poisoned/failed execution.
    pub fn execute(
        &mut self,
        rows: &[crate::llama::LlamaBatchRow<'_>],
    ) -> LlamaBatchExecutorResult<&[u8]> {
        self.output_ready = false;
        if self.poisoned {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        let capacity = self.metadata.config().max_block_entries();
        let packed = self.metadata.pack(rows)?;
        let input_count = packed.total_input_tokens();
        if packed.row_count() != 1
            || (input_count != 1 && !(self.batched_prefill && input_count == 128))
        {
            return Err(rejected(
                "owned graph input count differs from captured topology",
            ));
        }
        let token = packed.input_token_ids()[input_count - 1];
        let pos = packed.position_ids()[input_count - 1];
        if self.batched_prefill
            && !((input_count == 128
                && pos == 127
                && packed
                    .position_ids()
                    .iter()
                    .enumerate()
                    .all(|(i, &p)| p == i as u32))
                || (input_count == 1 && (128..160).contains(&pos)))
        {
            return Err(rejected(
                "P128 graph requires a complete fresh prompt or one decode token",
            ));
        }
        if self.vllm_smol_p128_graph
            && rows[0].kind()
                != if pos < 128 {
                    crate::llama::LlamaBatchRowKind::Prefill
                } else {
                    crate::llama::LlamaBatchRowKind::Decode
                }
        {
            return Err(rejected("P128 graph stage differs from scheduler row kind"));
        }
        if self.vllm_smol_p128_graph && pos >= 160 {
            return Err(rejected("vllm-smol-p128-v1 position exceeds 159"));
        }
        if packed
            .input_token_ids()
            .iter()
            .any(|&id| id as usize >= self.vocabulary_size)
            || pos as usize >= self.maximum_position_count
        {
            return Err(rejected("owned graph token or position out of bounds"));
        }
        let decision = select_registered_execution_graph(
            full_decode_request(ExecutionGraphPolicy::Require, true),
            self.signature,
            &self.registry,
        )
        .map_err(|_| rejected("owned registry lookup failed"))?;
        if !matches!(decision, GraphRegistryDispatchDecision::FullGraph { replay_slot } if replay_slot == GraphReplaySlot::new(0))
        {
            return Err(rejected("owned registry slot mismatch"));
        }
        self.payload.fill(0);
        self.payload[..4].copy_from_slice(&token.to_le_bytes());
        self.payload[4..8].copy_from_slice(&pos.to_le_bytes());
        self.payload[12..16]
            .copy_from_slice(&(packed.physical_block_ids().len() as u32).to_le_bytes());
        for (i, &physical) in packed.physical_block_ids().iter().enumerate() {
            self.payload[16 + 4 * i..20 + 4 * i].copy_from_slice(&physical.to_le_bytes());
            let offset = 16 + 4 * capacity + 2 * i;
            self.payload[offset..offset + 2]
                .copy_from_slice(&packed.valid_tokens()[i].to_le_bytes());
        }
        if self.batched_prefill {
            let offset = ((16 + 6 * capacity + 3) & !3) + 4;
            self.payload[offset..offset + 4].copy_from_slice(&PREFILL128_MAGIC.to_le_bytes());
            self.payload[offset + 4..offset + 8]
                .copy_from_slice(&(input_count as u32).to_le_bytes());
            if input_count == 128 {
                for (i, &id) in packed.input_token_ids().iter().enumerate() {
                    let at = offset + 8 + 4 * i;
                    self.payload[at..at + 4].copy_from_slice(&id.to_le_bytes());
                }
            }
        }
        self.poisoned = true;
        self.graph
            .replay_transfer(&self.payload)
            .and_then(|()| self.graph.read_transfer(&mut self.output))
            .map_err(|e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e))?;
        let logits = self.vocabulary_size * 2;
        let read = |offset| {
            u32::from_le_bytes(self.output[offset..offset + 4].try_into().expect("status"))
        };
        if read(logits + 4) != 0 || read(logits + 8) != 32 || read(logits + 12) != 0 {
            return Err(rejected("owned graph output status failed"));
        }
        self.poisoned = false;
        self.output_ready = true;
        self.replays = self.replays.saturating_add(1);
        Ok(&self.output[..logits])
    }
    /// Releases graph leases before closing the actual executor and stream.
    /// # Errors
    /// Returns the first native close error; unknown completion is never ignored.
    pub fn close(self) -> LlamaBatchExecutorResult<()> {
        let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion), e);
        let p = self.graph.close().map_err(cuda)?;
        p.executor.close()?;
        p.metadata.close().map_err(cuda)?;
        p.result.close().map_err(cuda)?;
        p.staging.close().map_err(cuda)?;
        for buffer in p.prefill {
            buffer.close().map_err(cuda)?;
        }
        if let Some(packed) = p.packed {
            packed.close()?;
        }
        p.stream.close().map_err(cuda)
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;
    use crate::llama::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
    };
    use riley_model::{LoadLimits, LoadedModel};
    #[test]
    #[ignore = "requires SmolLM2 checkpoint and CUDA"]
    fn owned_graph_smol_p128_o32_reuses_scheduler_block_mappings()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
        let model = LoadedModel::load(std::path::Path::new(&path), LoadLimits::default())?;
        let prompt = model
            .tokenizer()
            .encode(&"Hello".repeat(128), riley_model::EncodeOptions::default())?;
        assert_eq!(prompt.len(), 128);
        let context = riley_cuda::CudaRuntime::initialize()?
            .device(0)?
            .create_context()?;
        let mut eager_stream = context.create_stream()?;
        let mut stream = context.create_stream()?;
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16)?,
            PreparedLlamaForwardConfig::default(),
        )
        .with_grouped_ragged_attention_heads()
        .with_separate_residual_norm()
        .with_iteration_batch_completion()
        .with_packed_async_metadata();
        let candidate = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
        let mut candidate = candidate.into_owned_decode_graph(&context)?;
        let mut expected_tokens = Vec::new();
        for (request, limit) in [159_usize, 23, 159].into_iter().enumerate() {
            // Fresh eager owner is the oracle; graph reuses stale KV allocations
            // with a new, non-identity scheduler block mapping on each request.
            let mut baseline =
                PreparedLlamaBatchExecutor::prepare(&model, &context, &mut eager_stream, config)?;
            let mapping: Vec<u32> = (0..16)
                .map(|i| ((i * 7 + request * 3) % 16) as u32)
                .collect();
            let mut token = 504;
            let mut tokens = Vec::new();
            for position in 0..limit {
                if position < 128 {
                    token = prompt[position];
                }
                let live = position / 16 + 1;
                let mut valid = vec![16; live];
                valid[live - 1] = (position % 16 + 1) as u16;
                let input = [token];
                let rows = [LlamaBatchRow::new(
                    (request + 1) as u64,
                    if position < 128 {
                        LlamaBatchRowKind::Prefill
                    } else {
                        LlamaBatchRowKind::Decode
                    },
                    &input,
                    (position + 1) as u32,
                    LlamaBatchBlockTable::new(
                        crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                        &mapping[..live],
                        &valid,
                        (position + 1) as u32,
                    ),
                    Some(0),
                )];
                baseline.execute(&rows, &mut eager_stream)?;
                let mut expected = vec![0; baseline.vocabulary_size() * 2];
                baseline.download_logits(&mut expected, &mut eager_stream)?;
                assert_eq!(
                    candidate.execute(&rows)?,
                    expected,
                    "request={request} position={position}"
                );
                token = candidate.greedy_token()?;
                if position >= 127 {
                    tokens.push(token);
                }
            }
            if limit == 159 {
                assert_eq!(tokens.len(), 32);
                if expected_tokens.is_empty() {
                    expected_tokens = tokens;
                } else {
                    assert_eq!(tokens, expected_tokens);
                }
            }
            baseline.close()?;
            let invalid = [u32::MAX];
            let invalid_rows = [LlamaBatchRow::new(
                99,
                LlamaBatchRowKind::Prefill,
                &invalid,
                1,
                LlamaBatchBlockTable::new(
                    crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                    &mapping[..1],
                    &[1],
                    1,
                ),
                Some(0),
            )];
            assert!(candidate.execute(&invalid_rows).is_err());
            assert!(
                candidate.greedy_token().is_err(),
                "invalid input must revoke old output"
            );
        }
        assert_eq!(candidate.replay_count(), 341);
        candidate.close()?;
        stream.close()?;
        eager_stream.close()?;
        let stats = context.allocation_stats()?;
        assert_eq!(stats.device_live_allocations(), 0);
        assert_eq!(stats.pinned_host_live_allocations(), 0);
        println!(
            "G04_OWNED p128_o32=true full_logits_exact=true requests=3 cancelled_prefix=23 replays=341 captures=1 zero_allocations=true output_tokens={expected_tokens:?}"
        );
        context.close()?;
        Ok(())
    }
    #[test]
    #[ignore = "requires SmolLM2 checkpoint and CUDA"]
    fn owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
        let model = LoadedModel::load(std::path::Path::new(&path), LoadLimits::default())?;
        let prompt = model
            .tokenizer()
            .encode(&"Hello".repeat(128), riley_model::EncodeOptions::default())?;
        assert_eq!(prompt.len(), 128);
        let context = riley_cuda::CudaRuntime::initialize()?
            .device(0)?
            .create_context()?;
        let mut stream = context.create_stream()?;
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16)?,
            PreparedLlamaForwardConfig::default(),
        )
        .with_grouped_ragged_attention_heads()
        .with_separate_residual_norm()
        .with_iteration_batch_completion()
        .with_packed_async_metadata();
        let candidate = PreparedLlamaBatchExecutor::prepare(
            &model,
            &context,
            &mut stream,
            config.with_vllm_smol_p128_graph(),
        )?;
        let mut candidate = candidate.into_owned_decode_graph(&context)?;
        assert_eq!(candidate.numerical_profile_id(), "vllm-smol-p128-v1");
        assert!(candidate.validate_request_shape(127, 32).is_err());
        assert!(candidate.validate_request_shape(129, 32).is_err());
        assert!(candidate.validate_request_shape(128, 33).is_err());
        candidate.validate_request_shape(128, 32)?;
        let mut expected_tokens = Vec::new();
        for (request, limit) in [159_usize, 23, 150, 159].into_iter().enumerate() {
            // New requests reuse the same retained parents and fresh scheduler mappings.
            let mapping: Vec<u32> = (0..16)
                .map(|i| ((i * 7 + request * 3) % 16) as u32)
                .collect();
            let mut token = 504;
            let mut tokens = Vec::new();
            for position in 0..limit {
                if position < 128 {
                    token = prompt[position];
                }
                let live = position / 16 + 1;
                let mut valid = vec![16; live];
                valid[live - 1] = (position % 16 + 1) as u16;
                let input = [token];
                let rows = [LlamaBatchRow::new(
                    (request + 1) as u64,
                    if position < 128 {
                        LlamaBatchRowKind::Prefill
                    } else {
                        LlamaBatchRowKind::Decode
                    },
                    &input,
                    (position + 1) as u32,
                    LlamaBatchBlockTable::new(
                        crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                        &mapping[..live],
                        &valid,
                        (position + 1) as u32,
                    ),
                    Some(0),
                )];
                let before = context.allocation_stats()?;
                if matches!(position, 0 | 127 | 128 | 158) {
                    let wrong_stage = [LlamaBatchRow::new(
                        (request + 1) as u64,
                        if position < 128 {
                            LlamaBatchRowKind::Decode
                        } else {
                            LlamaBatchRowKind::Prefill
                        },
                        &input,
                        (position + 1) as u32,
                        rows[0].block_table(),
                        Some(0),
                    )];
                    let replays = candidate.replay_count();
                    assert!(candidate.execute(&wrong_stage).is_err());
                    assert!(candidate.greedy_token().is_err());
                    assert_eq!(candidate.replay_count(), replays);
                    assert_eq!(context.allocation_stats()?, before);
                }
                candidate.execute(&rows)?;
                assert_eq!(context.allocation_stats()?, before);
                token = candidate.greedy_token()?;
                if position >= 127 {
                    tokens.push(token);
                }
            }
            let reference = [
                28, 339, 5248, 253, 1838, 3241, 282, 253, 1443, 929, 3156, 28, 198, 198, 504, 808,
                2775, 339, 1277, 288, 1643, 314, 338, 339, 5248, 2045, 288, 1138, 346, 253, 1443,
                282,
            ];
            assert_eq!(tokens.as_slice(), &reference[..tokens.len()]);
            if limit == 159 {
                assert_eq!(tokens.len(), 32);
                if expected_tokens.is_empty() {
                    expected_tokens = tokens;
                } else {
                    assert_eq!(tokens, expected_tokens);
                }
            }
            let invalid = [u32::MAX];
            let invalid_rows = [LlamaBatchRow::new(
                99,
                LlamaBatchRowKind::Prefill,
                &invalid,
                1,
                LlamaBatchBlockTable::new(
                    crate::paged_kv::BLOCK_TABLE_V1_VERSION,
                    &mapping[..1],
                    &[1],
                    1,
                ),
                Some(0),
            )];
            assert!(candidate.execute(&invalid_rows).is_err());
            assert!(
                candidate.greedy_token().is_err(),
                "invalid input must revoke old output"
            );
        }
        assert_eq!(candidate.replay_count(), 491);
        candidate.close()?;
        let mut drop_stream = context.create_stream()?;
        let drop_candidate = PreparedLlamaBatchExecutor::prepare(
            &model,
            &context,
            &mut drop_stream,
            config.with_vllm_smol_p128_graph(),
        )?
        .into_owned_decode_graph(&context)?;
        drop(drop_candidate);
        drop_stream.close()?;
        stream.close()?;
        let stats = context.allocation_stats()?;
        assert_eq!(stats.device_live_allocations(), 0);
        assert_eq!(stats.pinned_host_live_allocations(), 0);
        println!(
            "G04_VLLM_PROFILE p128_o32=true vllm_tokens_exact=true requests=4 cancelled_prefill=23 cancelled_output=23 replays=491 captures=2 shared_owner=true stage_mismatch_rejected=true live_allocation_deltas_zero=true output_tokens={expected_tokens:?}"
        );
        context.close()?;
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "graph_decode_multisequence_model_gpu.rs"]
mod multisequence_model_gpu;

#[path = "graph_decode_multi_session.rs"]
mod multi_session;
pub use multi_session::OwnedLlamaMultiDecodeExecutor;

impl PreparedLlamaBatchExecutor {
    /// Captures V3 using this loader's actual immutable weights, RoPE and KV pool.
    /// Unsupported model geometry/numerics are rejected before capture.
    pub fn prepare_variable_session<'a>(
        &'a mut self, stream:&'a mut CudaStream,
        scratch:&'a mut crate::llama::variable_session::VariableGraphBuffers,
    )->LlamaBatchExecutorResult<crate::llama::variable_session::BorrowedVariableSession<'a>> {
        use sha2::{Digest,Sha256};
        let context=self.maximum_position_count()?.min(4096);
        let physical=self.owner.layout.physical_block_count();
        let f=&mut self.owner.forward;
        let d=f.plan.dimensions();
        if self.owner.poisoned || f.is_poisoned() || f.plan.sequence_length()!=1
            || f.rms_norm_profile()!=LlamaRmsNormProfile::HuggingFaceSmolLm2
            || f.plan.layers().len()!=30 || d.hidden_size()!=576 || d.intermediate_size()!=1536
            || d.vocabulary_size()!=49152 || d.query_heads()!=9 || d.key_value_heads()!=3
            || self.owner.layout.head_dimension()!=64 || context==0 || context>4096 || physical==0 || physical>4096
            || f.plan.rope_theta()!=100000.0 || f.plan.final_norm_epsilon()!=1e-5 {
            return Err(rejected("V3 requires fixed SmolLM2 geometry and numerics"));
        }
        let mut weights=vec![f.plan.embedding_weight().index(),f.plan.final_norm_weight().index(),f.plan.lm_head_weight().index()];
        for l in f.plan.layers() {
            if l.query_bias().is_some() || l.key_bias().is_some() || l.value_bias().is_some() || l.output_bias().is_some()
                || l.input_norm_epsilon()!=1e-5 || l.post_attention_norm_epsilon()!=1e-5 {return Err(rejected("V3 unsupported bias or norm epsilon"));}
            weights.extend([l.input_norm_weight(),l.query_weight(),l.key_weight(),l.value_weight(),l.output_weight(),
                l.post_attention_norm_weight(),l.gate_weight(),l.up_weight(),l.down_weight()].map(|w|w.index()));
        }
        let cuda=|e|cuda_error(ExecutionSite::global(LlamaOp::IterationCompletion),e);
        let mut hash=Sha256::new();hash.update(b"riley.v3.loaded-smol.variable.v1");
        for source in [include_bytes!("../../../../kernels/src/prefill_shape_model.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_projection.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_rope_kv.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_attention.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_pointwise.cuh").as_slice(),
            include_bytes!("../../../../kernels/src/prefill_shape_packet.hpp").as_slice(),
            include_bytes!("../../../../kernels/src/graph_resources.cu").as_slice(),
            include_bytes!("../../../../kernels/src/graph_numerics_precise.cu").as_slice(),
            include_bytes!("multi_descriptor/variable_wire.rs").as_slice()] {hash.update((source.len() as u64).to_le_bytes());hash.update(source);}
        let m=scratch.head.algorithm_metadata();
        let (major,minor)=m.compute_capability();hash.update(major.to_le_bytes());hash.update(minor.to_le_bytes());
        hash.update(m.runtime_version().to_le_bytes());hash.update(m.cublaslt_version().to_le_bytes());
        for v in [m.backend_id(),m.algorithm_id() as u32,m.tile_id(),m.stages_id(),m.split_k(),m.reduction_scheme(),m.cta_swizzling(),m.custom_option(),u32::from(m.deterministic())] {hash.update(v.to_le_bytes());}
        let (mr,n,k)=m.dimensions();for v in [mr,n,k,m.workspace_bytes(),m.numerical_implementation_flags()] {hash.update(v.to_le_bytes());}
        hash.update(scratch.capacity.to_le_bytes());hash.update((physical as u64).to_le_bytes());hash.update((context as u64).to_le_bytes());
        for &w in &weights {hash.update((w as u64).to_le_bytes());}
        let mut chunk=vec![0;f.io_staging.byte_len().min(1024*1024) as usize];
        if chunk.is_empty(){return Err(rejected("V3 needs model staging for content identity"));}
        for buffer in f.weights.borrow_graph_weight_parents().chain([&mut self.owner.absolute_rope_cos,&mut self.owner.absolute_rope_sin]) {
            hash.update(buffer.byte_len().to_le_bytes());let mut offset=0;
            while offset<buffer.byte_len(){let n=(buffer.byte_len()-offset).min(chunk.len() as u64) as usize;
                buffer.download_to_slice(offset,&mut chunk[..n],&mut f.io_staging,stream).map_err(cuda)?;
                hash.update(&chunk[..n]);offset+=n as u64;}
        }
        let mut devices:Vec<_>=f.weights.borrow_graph_weight_parents().collect();let base=devices.len();
        let (rows,tail)=scratch.devices.split_at_mut(12);devices.extend(rows);
        devices.extend([&mut self.owner.key_cache,&mut self.owner.value_cache,&mut self.owner.absolute_rope_cos,&mut self.owner.absolute_rope_sin]);
        devices.extend(tail);
        let mut graph=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream,devices,
            pinned:vec![&mut scratch.staging],plans:vec![&mut scratch.head]}).map_err(cuda)?;
        graph.record_v3_prefill(&std::array::from_fn(|i|base+i),None,&weights,0,0,scratch.capacity,physical as u32).map_err(cuda)?;
        crate::llama::variable_session::BorrowedVariableSession::new(graph,hash.finalize().into(),physical as u32,context as u32)
            .map_err(|_|rejected("V3 session identity rejected"))
    }
}
