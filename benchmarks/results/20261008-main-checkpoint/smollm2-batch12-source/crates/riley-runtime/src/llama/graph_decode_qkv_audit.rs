use crate::llama::LlamaBatchRow;
use crate::llama::executor::buffers::BatchDeviceInput;
use crate::llama::executor::metadata::PackedIterationLayout;
use riley_cuda::{IndexedRopeParams, indexed_rope};
use riley_cuda::{
    PackedBatchHostV1, PackedBatchV1, RaggedPagedAttentionParams, grouped_ragged_paged_attention,
};
// Cold input norm/QKV audit. Completed scratch snapshots are not live layer traces.
use super::PreparedLlamaBatchExecutor;
use crate::llama::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
};
use crate::llama::forward::{
    LlamaRmsNormProfile, PreparedLlamaGemm, execute_gemm, execute_profile_rms_norm,
};
use crate::llama::{ExecutionSite, LlamaOp};
use riley_cuda::{
    BorrowedGraphResourceParents, BorrowedGraphResourceReservation, CudaBufferSpan,
    CudaBufferSpanMut, CudaDType, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream,
    RmsNormParams,
};
fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::InvalidConfiguration {
        field: "norm QKV audit",
        reason,
    }
}
fn read(
    b: &mut CudaDeviceBuffer,
    p: &mut CudaPinnedHostBuffer,
    s: &mut CudaStream,
) -> riley_cuda::CudaResult<Vec<u8>> {
    let mut v = vec![0; b.byte_len() as usize];
    b.download_to_slice(0, &mut v, p, s)?;
    Ok(v)
}
impl PreparedLlamaBatchExecutor {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn audit_c07_norm_qkv(
        &mut self,
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_qkv_impl(None, stream, false, false)
    }
    pub(crate) fn audit_c07_qkv_rope(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_qkv_impl(Some(rows), stream, false, false)
    }
    pub(crate) fn audit_c07_qkv_kv(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_qkv_impl(Some(rows), stream, true, false)
    }
    pub(crate) fn audit_c07_attention_chain(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.audit_qkv_impl(Some(rows), stream, true, true)
    }
    #[allow(clippy::too_many_lines)]
    fn audit_qkv_impl(
        &mut self,
        rows: Option<&[LlamaBatchRow<'_>]>,
        stream: &mut CudaStream,
        include_cache: bool,
        include_attention: bool,
    ) -> LlamaBatchExecutorResult<()> {
        if self.owner.poisoned || self.owner.forward.is_poisoned() || !self.output_ready {
            return Err(rejected("healthy completed iteration required"));
        }
        if include_attention
            && self.config.ragged_attention_reduction_profile()
                != riley_cuda::AttentionReductionProfile::CanonicalV1
        {
            return Err(rejected("canonical attention profile required"));
        }
        let mut attention_fields = None;
        let kv_layout = self.owner.layout;
        let mut cache_geometry = None;
        let rope_position = if let Some(rows) = rows {
            let packed = self.owner.metadata.pack(rows)?;
            if packed.total_input_tokens() != 1 {
                return Err(rejected("one RoPE row required"));
            }
            if include_cache {
                let position = packed.position_ids()[0] as usize;
                let slot = packed.row_sequence_slots()[0] as usize;
                let begin = *packed
                    .block_row_offsets()
                    .get(slot)
                    .ok_or_else(|| rejected("KV sequence absent"))?
                    as usize;
                let index = begin + position / 16;
                let physical = *packed
                    .physical_block_ids()
                    .get(index)
                    .ok_or_else(|| rejected("KV physical block absent"))?;
                let valid = *packed
                    .valid_tokens()
                    .get(index)
                    .ok_or_else(|| rejected("KV valid prefix absent"))?;
                cache_geometry = Some([
                    kv_layout.layer_count() as u64,
                    (kv_layout.layer_count() - 1) as u64,
                    kv_layout.physical_block_count() as u64,
                    u64::from(physical),
                    position as u64 / 16,
                    u64::from(valid),
                ]);
            }
            let layout = PackedIterationLayout::for_batch(&packed, 1)?;
            if include_attention {
                if packed.row_count() != 1
                    || packed.physical_block_ids().len() != 1
                    || packed.position_ids()[0] >= 16
                {
                    return Err(rejected("attention chain requires one first logical block"));
                }
                attention_fields = Some([
                    layout.sequence_block_offsets.offset as u64,
                    layout.physical_block_ids.offset as u64,
                    layout.valid_tokens.offset as u64,
                    layout.row_sequence_slots.offset as u64,
                ]);
            }
            Some((layout.row_positions.offset as u64, packed.position_ids()[0]))
        } else {
            None
        };
        let BatchDeviceInput::IterationBatch { slab } = &mut self.owner.device_input else {
            return Err(rejected("packed metadata parent required"));
        };
        let cos = &mut self.owner.absolute_rope_cos;
        let sin = &mut self.owner.absolute_rope_sin;
        let key_pool = &mut self.owner.key_cache;
        let value_pool = &mut self.owner.value_cache;
        let f = &mut self.owner.forward;
        let profile = f.rms_norm_profile();
        if f.plan.sequence_length() != 1
            || !matches!(
                profile,
                LlamaRmsNormProfile::Canonical | LlamaRmsNormProfile::HuggingFaceSmolLm2
            )
        {
            return Err(rejected("unsupported rows/profile"));
        }
        let layer = f
            .plan
            .layers()
            .last()
            .ok_or_else(|| rejected("layer absent"))?;
        if layer.query_bias().is_some()
            || layer.key_bias().is_some()
            || layer.value_bias().is_some()
        {
            return Err(rejected("biased QKV unsupported"));
        }
        let ids = [
            layer.input_norm_weight(),
            layer.query_weight(),
            layer.key_weight(),
            layer.value_weight(),
        ];
        let epsilon = layer.input_norm_epsilon();
        for plan in [&f.gemms.hidden, &f.gemms.key_value] {
            let PreparedLlamaGemm::Canonical(plan) = plan else {
                return Err(rejected("selected canonical plans required"));
            };
            if plan.algorithm_metadata().split_k() > 1
                || plan.algorithm_metadata().reduction_scheme() != 0
            {
                return Err(rejected("no-split plan required"));
            }
        }
        let result = (|| -> LlamaBatchExecutorResult<()> {
            let cuda = |e| cuda_error(ExecutionSite::global(LlamaOp::QueryProjection), e);
            let b = &mut f.buffers;
            let pinned = f.io_staging.to_vec().map_err(cuda)?;
            let saved_attention = if include_attention {
                Some(read(&mut b.hidden_context, &mut f.io_staging, stream).map_err(cuda)?)
            } else {
                None
            };
            let original_pools = if include_cache {
                Some([
                    read(key_pool, &mut f.io_staging, stream).map_err(cuda)?,
                    read(value_pool, &mut f.io_staging, stream).map_err(cuda)?,
                ])
            } else {
                None
            };
            let mut saved = Vec::new();
            for d in [
                &mut b.hidden_current,
                &mut b.hidden_norm,
                &mut b.hidden_projection,
                &mut b.key_raw,
                &mut b.value_raw,
            ] {
                saved.push(read(d, &mut f.io_staging, stream).map_err(cuda)?);
            }
            let extra_saved = if let Some((offset, position)) = rope_position {
                let metadata = read(slab, &mut f.io_staging, stream).map_err(cuda)?;
                let start =
                    usize::try_from(offset).map_err(|_| rejected("position offset too large"))?;
                if metadata.get(start..start + 4) != Some(position.to_le_bytes().as_slice()) {
                    return Err(rejected("device position differs from host mirror"));
                }
                Some([
                    read(&mut b.hidden_rotary, &mut f.io_staging, stream).map_err(cuda)?,
                    read(&mut b.key_rotary, &mut f.io_staging, stream).map_err(cuda)?,
                    metadata,
                ])
            } else {
                None
            };
            let h = b.hidden_current.byte_len();
            let weight = f
                .weights
                .view_physical(ids[0])
                .map_err(|_| rejected("norm weight absent"))?;
            execute_profile_rms_norm(
                profile,
                &mut RmsNormParams {
                    input: CudaBufferSpan::new(&b.hidden_current, CudaDType::BF16, 0, h)
                        .map_err(cuda)?,
                    weight: weight.span(),
                    output: CudaBufferSpanMut::new(&mut b.hidden_norm, CudaDType::BF16, 0, h)
                        .map_err(cuda)?,
                    row_count: 1,
                    hidden_size: h / 2,
                    epsilon,
                },
                stream,
            )
            .map_err(cuda)?;
            let weight = f
                .weights
                .view_physical(ids[1])
                .map_err(|_| rejected("Q weight absent"))?;
            execute_gemm(
                &mut f.gemms.hidden,
                &b.hidden_norm,
                weight.span(),
                &mut b.hidden_projection,
                &mut b.gemm_workspace,
                stream,
                ExecutionSite::global(LlamaOp::QueryProjection),
            )
            .map_err(LlamaBatchExecutorError::Forward)?;
            for (id, output, op) in [
                (ids[2], &mut b.key_raw, LlamaOp::KeyProjection),
                (ids[3], &mut b.value_raw, LlamaOp::ValueProjection),
            ] {
                let weight = f
                    .weights
                    .view_physical(id)
                    .map_err(|_| rejected("KV weight absent"))?;
                execute_gemm(
                    &mut f.gemms.key_value,
                    &b.hidden_norm,
                    weight.span(),
                    output,
                    &mut b.gemm_workspace,
                    stream,
                    ExecutionSite::global(op),
                )
                .map_err(LlamaBatchExecutorError::Forward)?;
            }
            if let Some((offset, position)) = rope_position {
                if f.plan.dimensions().head_dimension() != 64 {
                    return Err(rejected("D64 RoPE required"));
                }
                for (input, output, heads) in [
                    (
                        &b.hidden_projection,
                        &mut b.hidden_rotary,
                        f.plan.dimensions().query_heads(),
                    ),
                    (
                        &b.key_raw,
                        &mut b.key_rotary,
                        f.plan.dimensions().key_value_heads(),
                    ),
                ] {
                    indexed_rope(
                        &mut IndexedRopeParams {
                            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, input.byte_len())
                                .map_err(cuda)?,
                            output: CudaBufferSpanMut::new(
                                output,
                                CudaDType::BF16,
                                0,
                                input.byte_len(),
                            )
                            .map_err(cuda)?,
                            cos: CudaBufferSpan::new(cos, CudaDType::F32, 0, cos.byte_len())
                                .map_err(cuda)?,
                            sin: CudaBufferSpan::new(sin, CudaDType::F32, 0, sin.byte_len())
                                .map_err(cuda)?,
                            positions: CudaBufferSpan::new(slab, CudaDType::U32, offset, 4)
                                .map_err(cuda)?,
                            positions_host: &[position],
                            head_count: heads as u64,
                            head_size: 64,
                            rotary_dimension: 64,
                            table_position_count: cos.byte_len() / 128,
                        },
                        stream,
                    )
                    .map_err(cuda)?;
                }
            }
            let mut expected = [
                read(&mut b.hidden_projection, &mut f.io_staging, stream).map_err(cuda)?,
                read(&mut b.key_raw, &mut f.io_staging, stream).map_err(cuda)?,
                read(&mut b.value_raw, &mut f.io_staging, stream).map_err(cuda)?,
            ]
            .concat();
            if rope_position.is_some() {
                expected = [
                    read(&mut b.hidden_rotary, &mut f.io_staging, stream).map_err(cuda)?,
                    read(&mut b.key_rotary, &mut f.io_staging, stream).map_err(cuda)?,
                    read(&mut b.value_raw, &mut f.io_staging, stream).map_err(cuda)?,
                ]
                .concat();
            }
            if let (Some(fields), Some(g), Some((offset, position)), Some(original)) = (
                attention_fields,
                cache_geometry,
                rope_position,
                &original_pools,
            ) {
                // Independent CPU scatter builds the history after zero at slot 0,
                // then the original token at the current slot, matching replay order.
                let mut pools = original.clone();
                let k = b.key_raw.byte_len() as usize;
                let layer_offset = kv_layout
                    .layer_byte_offset(g[1] as usize)
                    .ok_or_else(|| rejected("attention layer absent"))?;
                for head in 0..k / 128 {
                    let base =
                        layer_offset as usize + (g[3] as usize * (k / 128) + head) * 16 * 128;
                    for kind in 0..2 {
                        pools[kind][base..base + 128].fill(0);
                        let target = base + position as usize * 128;
                        let source = h as usize + kind * k + head * 128;
                        pools[kind][target..target + 128]
                            .copy_from_slice(&expected[source..source + 128]);
                    }
                }
                for (pool, bytes) in [&mut *key_pool, &mut *value_pool].into_iter().zip(&pools) {
                    pool.upload_from_slice(0, bytes, &mut f.io_staging, stream)
                        .map_err(cuda)?;
                }
                let offsets = [0_u32, 1];
                let physical = [g[3] as u32];
                let valid = [g[5] as u16];
                let slots = [0_u32];
                let positions = [position];
                let host =
                    PackedBatchHostV1::new(&offsets, &physical, &valid, &slots, &positions, g[2])
                        .map_err(cuda)?;
                let view = |offset, len, dtype| {
                    CudaBufferSpan::new(slab, dtype, offset, len).map_err(cuda)
                };
                let batch = PackedBatchV1::new(
                    host,
                    view(fields[0], 8, CudaDType::U32)?,
                    view(fields[1], 4, CudaDType::U32)?,
                    view(fields[2], 2, CudaDType::U16)?,
                    view(fields[3], 4, CudaDType::U32)?,
                    view(offset, 4, CudaDType::U32)?,
                )
                .map_err(cuda)?;
                grouped_ragged_paged_attention(
                    &mut RaggedPagedAttentionParams {
                        query: CudaBufferSpan::new(&b.hidden_rotary, CudaDType::BF16, 0, h)
                            .map_err(cuda)?,
                        key_pool: CudaBufferSpan::new(
                            key_pool,
                            CudaDType::BF16,
                            layer_offset,
                            kv_layout.layer_stride_bytes(),
                        )
                        .map_err(cuda)?,
                        value_pool: CudaBufferSpan::new(
                            value_pool,
                            CudaDType::BF16,
                            layer_offset,
                            kv_layout.layer_stride_bytes(),
                        )
                        .map_err(cuda)?,
                        output: CudaBufferSpanMut::new(
                            &mut b.hidden_context,
                            CudaDType::BF16,
                            0,
                            h,
                        )
                        .map_err(cuda)?,
                        batch,
                        query_head_count: h / 128,
                        key_value_head_count: k as u64 / 128,
                        head_size: 64,
                        output_row_count: 1,
                        scale: 0.125,
                    },
                    stream,
                )
                .map_err(cuda)?;
                let attention =
                    read(&mut b.hidden_context, &mut f.io_staging, stream).map_err(cuda)?;
                if attention
                    .chunks_exact(2)
                    .any(|x| u16::from_le_bytes([x[0], x[1]]) & 0x7f80 == 0x7f80)
                {
                    return Err(rejected("nonfinite attention output"));
                }
                expected[..h as usize].copy_from_slice(&attention);
                // Start the graph from the original cache, not the eager-mutated one.
                for (pool, bytes) in [&mut *key_pool, &mut *value_pool].into_iter().zip(original) {
                    pool.upload_from_slice(0, bytes, &mut f.io_staging, stream)
                        .map_err(cuda)?;
                }
            }
            let table_positions = cos.byte_len() / 128;
            let mut payload = vec![0; expected.len()];
            payload[..saved[0].len()].copy_from_slice(&saved[0]);
            if let Some((_, position)) = rope_position {
                payload[h as usize..h as usize + 4].copy_from_slice(&position.to_le_bytes());
            }
            let mut zeros = vec![0; payload.len()];
            if let Some(g) = cache_geometry {
                zeros[h as usize..h as usize + 4]
                    .copy_from_slice(&((g[4] * 16) as u32).to_le_bytes());
            }
            let mut zero_result = None;
            let mut actual = zeros.clone();
            f.io_staging.write(0, &pinned).map_err(cuda)?;
            let mut parents: Vec<_> = f.weights.borrow_graph_weight_parents().collect();
            let base = parents.len();
            parents.extend([
                &mut b.hidden_current,
                &mut b.hidden_norm,
                &mut b.hidden_projection,
                &mut b.key_raw,
                &mut b.value_raw,
            ]);
            let rope_indices = rope_position.map(|(offset, _)| {
                let start = parents.len();
                parents.extend([
                    &mut b.hidden_rotary,
                    &mut b.key_rotary,
                    &mut *cos,
                    &mut *sin,
                    &mut *slab,
                ]);
                ([start, start + 1, start + 2, start + 3, start + 4], offset)
            });
            let cache_indices = cache_geometry.map(|g| {
                let start = parents.len();
                parents.extend([&mut *key_pool, &mut *value_pool]);
                ([start, start + 1], g)
            });
            let attention_binding = attention_fields.map(|fields| {
                let index = parents.len();
                parents.push(&mut b.hidden_context);
                (index, fields)
            });
            let workspace = b.gemm_workspace.as_mut().map(|d| {
                let i = parents.len();
                parents.push(d);
                i
            });
            let PreparedLlamaGemm::Canonical(q) = &mut f.gemms.hidden else {
                unreachable!()
            };
            let PreparedLlamaGemm::Canonical(kv) = &mut f.gemms.key_value else {
                unreachable!()
            };
            let identities = [
                (q.config(), q.algorithm_metadata()),
                (kv.config(), kv.algorithm_metadata()),
            ];
            let mut graph =
                BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents {
                    stream,
                    devices: parents,
                    pinned: vec![&mut f.io_staging],
                    plans: vec![q, kv],
                })
                .map_err(cuda)?;
            if let Some(rope_indices) = rope_indices {
                if let Some(cache_indices) = cache_indices {
                    if let Some(attention_binding) = attention_binding {
                        graph
                            .record_attention_chain(
                                [
                                    base,
                                    base + 1,
                                    base + 2,
                                    base + 3,
                                    base + 4,
                                    ids[0].index(),
                                    ids[1].index(),
                                    ids[2].index(),
                                    ids[3].index(),
                                ],
                                workspace,
                                [0, 1],
                                0,
                                (epsilon, profile == LlamaRmsNormProfile::HuggingFaceSmolLm2),
                                rope_indices,
                                cache_indices,
                                attention_binding,
                            )
                            .map_err(cuda)?;
                    } else {
                        graph
                            .record_qkv_kv(
                                [
                                    base,
                                    base + 1,
                                    base + 2,
                                    base + 3,
                                    base + 4,
                                    ids[0].index(),
                                    ids[1].index(),
                                    ids[2].index(),
                                    ids[3].index(),
                                ],
                                workspace,
                                [0, 1],
                                0,
                                (epsilon, profile == LlamaRmsNormProfile::HuggingFaceSmolLm2),
                                rope_indices,
                                cache_indices,
                            )
                            .map_err(cuda)?;
                    }
                } else {
                    graph
                        .record_qkv_rope(
                            [
                                base,
                                base + 1,
                                base + 2,
                                base + 3,
                                base + 4,
                                ids[0].index(),
                                ids[1].index(),
                                ids[2].index(),
                                ids[3].index(),
                            ],
                            workspace,
                            [0, 1],
                            0,
                            (epsilon, profile == LlamaRmsNormProfile::HuggingFaceSmolLm2),
                            rope_indices,
                        )
                        .map_err(cuda)?;
                }
                let mut bad = payload.clone();
                bad[h as usize..h as usize + 4].copy_from_slice(&u32::MAX.to_le_bytes());
                if table_positions <= u64::from(u32::MAX) {
                    assert!(graph.replay_transfer(&bad).is_err());
                    assert!(graph.read_transfer(&mut actual).is_err());
                }
            } else {
                graph
                    .record_norm_qkv(
                        [
                            base,
                            base + 1,
                            base + 2,
                            base + 3,
                            base + 4,
                            ids[0].index(),
                            ids[1].index(),
                            ids[2].index(),
                            ids[3].index(),
                        ],
                        workspace,
                        [0, 1],
                        0,
                        (epsilon, profile == LlamaRmsNormProfile::HuggingFaceSmolLm2),
                    )
                    .map_err(cuda)?;
            }
            for _ in 0..16 {
                graph.replay_transfer(&zeros).map_err(cuda)?;
                graph.read_transfer(&mut actual).map_err(cuda)?;
                if actual
                    .chunks_exact(2)
                    .any(|x| u16::from_le_bytes([x[0], x[1]]) & 0x7fff != 0)
                {
                    return Err(rejected("zero QKV mismatch"));
                }
                if include_cache {
                    zero_result = Some(actual.clone());
                }
                graph.replay_transfer(&payload).map_err(cuda)?;
                graph.read_transfer(&mut actual).map_err(cuda)?;
                if actual != expected {
                    return Err(rejected("QKV eager parity mismatch"));
                }
            }
            graph.close().map_err(cuda)?;
            if f.io_staging.to_vec().map_err(cuda)?[payload.len() * 2..]
                != pinned[payload.len() * 2..]
            {
                return Err(rejected("QKV staging tail changed"));
            }

            if let (Some(original), Some(g), Some((_, position))) =
                (&original_pools, cache_geometry, rope_position)
            {
                let mut expected_pools = original.clone();
                let k = b.key_raw.byte_len() as usize;
                for (pos, bytes) in [
                    (
                        g[4] * 16,
                        zero_result
                            .as_ref()
                            .ok_or_else(|| rejected("KV zero receipt absent"))?,
                    ),
                    (u64::from(position), &expected),
                ] {
                    for head in 0..k / 128 {
                        let target = kv_layout
                            .layer_byte_offset(g[1] as usize)
                            .ok_or_else(|| rejected("KV layer offset absent"))?
                            as usize
                            + ((g[3] as usize * (k / 128) + head) * 16 + pos as usize % 16) * 128;
                        for kind in 0..2 {
                            let source = h as usize + kind * k + head * 128;
                            expected_pools[kind][target..target + 128]
                                .copy_from_slice(&bytes[source..source + 128]);
                        }
                    }
                }
                let actual_pools = [
                    read(key_pool, &mut f.io_staging, stream).map_err(cuda)?,
                    read(value_pool, &mut f.io_staging, stream).map_err(cuda)?,
                ];
                if actual_pools != expected_pools {
                    return Err(rejected(
                        "integrated KV differs from CPU scatter or changed another layer",
                    ));
                }
            }

            if identities
                != [
                    (q.config(), q.algorithm_metadata()),
                    (kv.config(), kv.algorithm_metadata()),
                ]
            {
                return Err(rejected("QKV plan changed"));
            }
            for (d, bytes) in [
                &mut b.hidden_current,
                &mut b.hidden_norm,
                &mut b.hidden_projection,
                &mut b.key_raw,
                &mut b.value_raw,
            ]
            .into_iter()
            .zip(&saved)
            {
                d.upload_from_slice(0, bytes, &mut f.io_staging, stream)
                    .map_err(cuda)?;
                if read(d, &mut f.io_staging, stream).map_err(cuda)? != *bytes {
                    return Err(rejected("QKV scratch restore mismatch"));
                }
            }
            if let Some(bytes) = saved_attention {
                b.hidden_context
                    .upload_from_slice(0, &bytes, &mut f.io_staging, stream)
                    .map_err(cuda)?;
                if read(&mut b.hidden_context, &mut f.io_staging, stream).map_err(cuda)? != bytes {
                    return Err(rejected("attention scratch restore mismatch"));
                }
            }
            if let Some(original) = original_pools {
                for (pool, bytes) in [&mut *key_pool, &mut *value_pool]
                    .into_iter()
                    .zip(&original)
                {
                    pool.upload_from_slice(0, bytes, &mut f.io_staging, stream)
                        .map_err(cuda)?;
                    if read(pool, &mut f.io_staging, stream).map_err(cuda)? != *bytes {
                        return Err(rejected("KV parent restore mismatch"));
                    }
                }
            }
            if let Some(extra) = extra_saved {
                for (d, bytes) in [&mut b.hidden_rotary, &mut b.key_rotary, &mut *slab]
                    .into_iter()
                    .zip(&extra)
                {
                    d.upload_from_slice(0, bytes, &mut f.io_staging, stream)
                        .map_err(cuda)?;
                    if read(d, &mut f.io_staging, stream).map_err(cuda)? != *bytes {
                        return Err(rejected("RoPE parent restore mismatch"));
                    }
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
