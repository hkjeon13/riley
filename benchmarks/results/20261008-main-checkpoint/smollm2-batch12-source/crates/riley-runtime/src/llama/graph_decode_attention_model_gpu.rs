//! Actual model-owned attention parity and decode continuation regression.
use super::PreparedLlamaBatchExecutor;
use crate::llama::batch::PreparedLlamaBatchMetadata;
use crate::llama::graph::GraphOperatorCapability;
use crate::llama::graph_decode_capture_inventory::{
    PureDecodeGraphV1CaptureCapabilityInventory, PureDecodeGraphV1CaptureOperation,
};
use crate::llama::graph_decode_exact_device_slab::PureDecodeGraphV1ExactDeviceSlab;
use crate::llama::graph_decode_exact_host_slab::{
    PureDecodeGraphV1ExactHostSlab, PureDecodeGraphV1ExactHostSlabWrite,
};
use crate::llama::graph_decode_layout::{
    PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
};
use crate::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use crate::paged_kv::BLOCK_TABLE_V1_VERSION;
use riley_cuda::{CudaRuntime, PackedBatchHostV1};
use riley_model::{LoadLimits, LoadedModel};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
#[ignore = "requires CUDA and RILEY_REAL_CHECKPOINT; actual model continuation"]
fn c07_model_owned_packed_attention_preserves_logits_tokens_and_kv_continuation() -> TestResult {
    let checkpoint =
        std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("RILEY_REAL_CHECKPOINT is required")?;
    let model = LoadedModel::load(std::path::Path::new(&checkpoint), LoadLimits::default())?;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let bounds = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?;
    let config =
        PreparedLlamaBatchExecutorConfig::new(bounds, PreparedLlamaForwardConfig::default())
            .with_grouped_ragged_attention_heads()
            .with_iteration_batch_completion()
            .with_packed_async_metadata();
    let mut baseline = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
    let mut candidate = PreparedLlamaBatchExecutor::prepare(&model, &context, &mut stream, config)?;
    let kv = candidate.kv_layout();
    let query_heads = candidate.owner.forward.plan.dimensions().query_heads() as u64;
    let layer_index = kv.layer_count() - 1; // Shared query workspace contains the last executed layer.
    let layout =
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(1, 1, 3, 5))?;
    let mut host = PureDecodeGraphV1ExactHostSlab::prepare(layout)?;
    let mut device = PureDecodeGraphV1ExactDeviceSlab::prepare(&context, layout)?;
    let mut staging =
        context.allocate_pinned_host_buffer(kv.bytes_per_kind().max(layout.total_bytes()))?;
    let mut packer = PreparedLlamaBatchMetadata::prepare(bounds)?;
    let mut token = 504_u32;
    let mut predicted = Vec::new();
    for step in 0..4 {
        let tokens = [token];
        let valid = [(step + 1) as u16];
        let rows = [LlamaBatchRow::new(
            1,
            if step == 0 {
                LlamaBatchRowKind::Prefill
            } else {
                LlamaBatchRowKind::Decode
            },
            &tokens,
            step + 1,
            LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &[0], &valid, step + 1),
            Some(0),
        )];
        baseline.execute(&rows, &mut stream)?;
        candidate.execute(&rows, &mut stream)?;
        let mut expected = vec![0; baseline.output_byte_len()?];
        let mut actual = vec![0; candidate.output_byte_len()?];
        baseline.download_logits(&mut expected, &mut stream)?;
        candidate.download_logits(&mut actual, &mut stream)?;
        assert_eq!(actual, expected, "model logits differ at step {step}");
        if step > 0 {
            let packed = packer.pack(&rows)?;
            let source = match host
                .write_exact_v1_leased(&packed, &[1, 2, 3], &[0; 5])
                .map_err(|e| format!("canonical write: {e:?}"))?
            {
                PureDecodeGraphV1ExactHostSlabWrite::Written(source) => source,
                _ => return Err("model decode was not exact M1".into()),
            };
            let batch = PackedBatchHostV1::new(
                packed.block_row_offsets(),
                packed.physical_block_ids(),
                packed.valid_tokens(),
                packed.row_sequence_slots(),
                packed.position_ids(),
                1,
            )?;
            let allocations = context.allocation_stats()?;
            let mut owner = candidate.owner.prepare_c07_attention_graph(
                &mut stream,
                &mut device,
                &mut staging,
                source,
                layout,
                layer_index,
                batch,
            )?;
            let inventory = owner.bind_inventory(
                PureDecodeGraphV1CaptureCapabilityInventory::default(),
                layout,
                kv,
                layer_index,
                query_heads,
            );
            assert_eq!(
                inventory.capability_for(PureDecodeGraphV1CaptureOperation::Attention),
                GraphOperatorCapability::Supported
            );
            assert_eq!(
                inventory.operator_capability(),
                GraphOperatorCapability::Unknown
            );
            for _ in 0..8 {
                owner.replay()?;
            }
            eprintln!(
                "step={step} layer={layer_index} metadata/output SHA256={:02x?}",
                owner.parity_hashes()
            );
            owner.close()?;
            assert_eq!(context.allocation_stats()?, allocations);
            candidate.download_logits(&mut actual, &mut stream)?;
            assert_eq!(actual, expected, "graph changed completed model logits");
        }
        // Compare every initialized KV token of every model layer and KV head.
        // Uninitialized capacity bytes are excluded from cross-allocation parity.
        for (left, right) in [
            (
                &mut baseline.owner.key_cache,
                &mut candidate.owner.key_cache,
            ),
            (
                &mut baseline.owner.value_cache,
                &mut candidate.owner.value_cache,
            ),
        ] {
            let mut left_bytes = vec![0; kv.bytes_per_kind() as usize];
            let mut right_bytes = vec![0; kv.bytes_per_kind() as usize];
            left.download_to_slice(0, &mut left_bytes, &mut staging, &mut stream)?;
            right.download_to_slice(0, &mut right_bytes, &mut staging, &mut stream)?;
            for layer in 0..kv.layer_count() {
                for head in 0..kv.key_value_head_count() {
                    let offset = (kv.layer_byte_offset(layer).unwrap()
                        + head as u64 * kv.head_stride_bytes())
                        as usize;
                    let end = offset + usize::try_from(step + 1)? * 64 * 2;
                    assert_eq!(
                        left_bytes[offset..end],
                        right_bytes[offset..end],
                        "KV continuation step={step} layer={layer} head={head}"
                    );
                }
            }
        }
        let mut best = (f32::NEG_INFINITY, 0_u32);
        for (index, bytes) in expected.chunks_exact(2).enumerate() {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
            if value > best.0 {
                best = (value, index as u32);
            }
        }
        token = best.1;
        predicted.push(token);
    }
    eprintln!("model continuation greedy tokens={predicted:?}");
    baseline.close()?;
    candidate.close()?;
    device.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
