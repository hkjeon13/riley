#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaRuntime,
    CudaStream, PackedBatchHostV1, PackedBatchV1, RaggedPagedAttentionParams,
    grouped_ragged_paged_attention,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const BLOCK_COUNT: u64 = 3;
const ACTIVE_ROW_COUNT: u64 = 5;
const PHYSICAL_BLOCK_COUNT: u64 = 6;
const QUERY_HEAD_COUNT: u64 = 9;
const KEY_VALUE_HEAD_COUNT: u64 = 3;
const HEAD_SIZE: u64 = 64;
const OUTPUT_ROW_COUNT: u64 = 8;
const BLOCK_SIZE: u64 = 16;
const BF16_BYTES: u64 = 2;
const SCALE: f32 = 0.125;

struct GroupedAttentionFixture {
    offsets: Vec<u32>,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
    row_slots: Vec<u32>,
    row_positions: Vec<u32>,
    query: Vec<u8>,
    key_pool: Vec<u8>,
    value_pool: Vec<u8>,
    output_sentinel: Vec<u8>,
}

impl GroupedAttentionFixture {
    fn new() -> Self {
        // This is the production grouped-GQA geometry. Sequence 0 crosses a
        // page boundary at 15/16 through shuffled physical blocks, while the
        // padded output tail exercises M > T zero-fill semantics.
        let offsets = vec![0_u32, 2, 3];
        let block_ids = vec![4_u32, 1, 3];
        let valid_tokens = vec![16_u16, 2, 5];
        let row_slots = vec![0_u32, 1, 0, 1, 0];
        let row_positions = vec![17_u32, 4, 0, 2, 16];
        assert_eq!(u64::try_from(block_ids.len()).unwrap(), BLOCK_COUNT);
        assert_eq!(u64::try_from(row_slots.len()).unwrap(), ACTIVE_ROW_COUNT);

        let query_elements = ACTIVE_ROW_COUNT
            .checked_mul(QUERY_HEAD_COUNT)
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 query element count fits u64");
        let pool_elements = PHYSICAL_BLOCK_COUNT
            .checked_mul(KEY_VALUE_HEAD_COUNT)
            .and_then(|value| value.checked_mul(BLOCK_SIZE))
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 pool element count fits u64");
        let output_elements = OUTPUT_ROW_COUNT
            .checked_mul(QUERY_HEAD_COUNT)
            .and_then(|value| value.checked_mul(HEAD_SIZE))
            .expect("C05-19 output element count fits u64");
        Self {
            offsets,
            block_ids,
            valid_tokens,
            row_slots,
            row_positions,
            query: finite_bf16_pattern(query_elements, 7),
            key_pool: finite_bf16_pattern(pool_elements, 19),
            value_pool: finite_bf16_pattern(pool_elements, 37),
            output_sentinel: vec![0xa5; usize::try_from(output_elements * BF16_BYTES).unwrap()],
        }
    }

    fn host(&self) -> TestResult<PackedBatchHostV1<'_>> {
        Ok(PackedBatchHostV1::new(
            &self.offsets,
            &self.block_ids,
            &self.valid_tokens,
            &self.row_slots,
            &self.row_positions,
            PHYSICAL_BLOCK_COUNT,
        )?)
    }
}

fn finite_bf16_pattern(elements: u64, seed: usize) -> Vec<u8> {
    (0..usize::try_from(elements).expect("C05-19 fixture element count fits usize"))
        .flat_map(|index| {
            // Values in [roughly 0.125, 0.25) are finite normal BF16 values.
            let word = 0x3e00_u16 + u16::try_from((index * seed) % 0x100).unwrap();
            word.to_ne_bytes()
        })
        .collect()
}

fn u32_words_to_ne_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn u16_words_to_ne_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let mut buffer = context.allocate_device_buffer(u64::try_from(bytes.len())?)?;
    buffer.upload_from_slice(0, bytes, staging, stream)?;
    Ok(buffer)
}

fn download(
    buffer: &mut CudaDeviceBuffer,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> TestResult<Vec<u8>> {
    let mut bytes = vec![0_u8; usize::try_from(buffer.byte_len())?];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}

fn bind_batch<'a>(
    host: PackedBatchHostV1<'a>,
    offsets: &'a CudaDeviceBuffer,
    block_ids: &'a CudaDeviceBuffer,
    valid_tokens: &'a CudaDeviceBuffer,
    row_slots: &'a CudaDeviceBuffer,
    row_positions: &'a CudaDeviceBuffer,
) -> TestResult<PackedBatchV1<'a>> {
    Ok(PackedBatchV1::new(
        host,
        CudaBufferSpan::new(offsets, CudaDType::U32, 0, offsets.byte_len())?,
        CudaBufferSpan::new(block_ids, CudaDType::U32, 0, block_ids.byte_len())?,
        CudaBufferSpan::new(valid_tokens, CudaDType::U16, 0, valid_tokens.byte_len())?,
        CudaBufferSpan::new(row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
        CudaBufferSpan::new(row_positions, CudaDType::U32, 0, row_positions.byte_len())?,
    )?)
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

fn check_parent_layer(query_heads: u64, packed: bool) -> TestResult {
    use riley_cuda::{
        AttentionParentLayer, PackedAttentionMetadataLayout, PackedParentAttentionResources,
        ParentAttentionGraph, ParentAttentionResources,
    };
    let mut f = GroupedAttentionFixture::new();
    f.query = finite_bf16_pattern(ACTIVE_ROW_COUNT * query_heads * HEAD_SIZE, 7);
    f.output_sentinel = vec![0xa5; (OUTPUT_ROW_COUNT * query_heads * HEAD_SIZE * 2) as usize];
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer((f.key_pool.len() * 3) as u64)?;
    let mut key_bytes = finite_bf16_pattern((f.key_pool.len() * 3 / 2) as u64, 53);
    let mut value_bytes = finite_bf16_pattern((f.value_pool.len() * 3 / 2) as u64, 71);
    let stride = f.key_pool.len();
    key_bytes[stride..2 * stride].copy_from_slice(&f.key_pool);
    value_bytes[stride..2 * stride].copy_from_slice(&f.value_pool);
    let mut query = upload(&context, &mut stream, &mut staging, &f.query)?;
    let mut key = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let mut value = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let isolated_key = upload(&context, &mut stream, &mut staging, &f.key_pool)?;
    let isolated_value = upload(&context, &mut stream, &mut staging, &f.value_pool)?;
    let mut output = upload(&context, &mut stream, &mut staging, &f.output_sentinel)?;
    let mut eager = upload(&context, &mut stream, &mut staging, &f.output_sentinel)?;
    let mut offsets = upload(
        &context,
        &mut stream,
        &mut staging,
        &u32_words_to_ne_bytes(&f.offsets),
    )?;
    let mut ids = upload(
        &context,
        &mut stream,
        &mut staging,
        &u32_words_to_ne_bytes(&f.block_ids),
    )?;
    let mut valid = upload(
        &context,
        &mut stream,
        &mut staging,
        &u16_words_to_ne_bytes(&f.valid_tokens),
    )?;
    let mut slots = upload(
        &context,
        &mut stream,
        &mut staging,
        &u32_words_to_ne_bytes(&f.row_slots),
    )?;
    let mut positions = upload(
        &context,
        &mut stream,
        &mut staging,
        &u32_words_to_ne_bytes(&f.row_positions),
    )?;
    let mut metadata_slab = context.allocate_device_buffer(160)?;
    let mut payload = vec![0xee; 160];
    for (offset, bytes) in [
        (64, u32_words_to_ne_bytes(&f.offsets)),
        (96, u32_words_to_ne_bytes(&f.block_ids)),
        (128, u16_words_to_ne_bytes(&f.valid_tokens)),
        (32, u32_words_to_ne_bytes(&f.row_slots)),
        (0, u32_words_to_ne_bytes(&f.row_positions)),
    ] {
        payload[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
    let metadata_layout =
        PackedAttentionMetadataLayout::new(160, [(64, 12), (96, 12), (128, 6), (32, 20), (0, 20)])?;
    let eager_len = eager.byte_len();
    grouped_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &isolated_key,
                CudaDType::BF16,
                0,
                isolated_key.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &isolated_value,
                CudaDType::BF16,
                0,
                isolated_value.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut eager, CudaDType::BF16, 0, eager_len)?,
            batch: bind_batch(f.host()?, &offsets, &ids, &valid, &slots, &positions)?,
            query_head_count: query_heads,
            key_value_head_count: KEY_VALUE_HEAD_COUNT,
            head_size: HEAD_SIZE,
            output_row_count: OUTPUT_ROW_COUNT,
            scale: SCALE,
        },
        &mut stream,
    )?;
    stream.synchronize()?;
    let allocations = context.allocation_stats()?;
    // Wrong parent capacity must reject before taking native graph leases.
    let invalid = ParentAttentionGraph::prepare(
        ParentAttentionResources {
            stream: &mut stream,
            query: &mut query,
            key_parent: &mut key,
            value_parent: &mut value,
            output: &mut output,
            sequence_block_offsets: &mut offsets,
            block_ids: &mut ids,
            valid_tokens: &mut valid,
            row_sequence_slots: &mut slots,
            row_positions: &mut positions,
        },
        AttentionParentLayer::new(2, 1, PHYSICAL_BLOCK_COUNT, KEY_VALUE_HEAD_COUNT)?,
        f.host()?,
        query_heads,
        OUTPUT_ROW_COUNT,
        SCALE,
    );
    assert!(invalid.is_err());
    drop(invalid);
    // Context mismatch is rejected before leases; the original buffers remain reusable.
    let foreign_context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut foreign_query = foreign_context.allocate_device_buffer(query.byte_len())?;
    let invalid_context = ParentAttentionGraph::prepare(
        ParentAttentionResources {
            stream: &mut stream,
            query: &mut foreign_query,
            key_parent: &mut key,
            value_parent: &mut value,
            output: &mut output,
            sequence_block_offsets: &mut offsets,
            block_ids: &mut ids,
            valid_tokens: &mut valid,
            row_sequence_slots: &mut slots,
            row_positions: &mut positions,
        },
        AttentionParentLayer::new(3, 1, PHYSICAL_BLOCK_COUNT, KEY_VALUE_HEAD_COUNT)?,
        f.host()?,
        query_heads,
        OUTPUT_ROW_COUNT,
        SCALE,
    );
    assert!(invalid_context.is_err());
    drop(invalid_context);
    foreign_query.close()?;
    foreign_context.close()?;
    for layer_index in [1, 1] {
        let mut graph = if packed {
            ParentAttentionGraph::prepare_packed(
                PackedParentAttentionResources {
                    stream: &mut stream,
                    query: &mut query,
                    key_parent: &mut key,
                    value_parent: &mut value,
                    output: &mut output,
                    metadata_slab: &mut metadata_slab,
                },
                AttentionParentLayer::new(
                    3,
                    layer_index,
                    PHYSICAL_BLOCK_COUNT,
                    KEY_VALUE_HEAD_COUNT,
                )?,
                metadata_layout,
                &payload,
                &mut staging,
                f.host()?,
                query_heads,
                OUTPUT_ROW_COUNT,
                SCALE,
            )?
        } else {
            ParentAttentionGraph::prepare(
                ParentAttentionResources {
                    stream: &mut stream,
                    query: &mut query,
                    key_parent: &mut key,
                    value_parent: &mut value,
                    output: &mut output,
                    sequence_block_offsets: &mut offsets,
                    block_ids: &mut ids,
                    valid_tokens: &mut valid,
                    row_sequence_slots: &mut slots,
                    row_positions: &mut positions,
                },
                AttentionParentLayer::new(
                    3,
                    layer_index,
                    PHYSICAL_BLOCK_COUNT,
                    KEY_VALUE_HEAD_COUNT,
                )?,
                f.host()?,
                query_heads,
                OUTPUT_ROW_COUNT,
                SCALE,
            )?
        };
        for _ in 0..64 {
            graph.replay()?;
        }
        graph.close()?;
        assert_eq!(context.allocation_stats()?, allocations);
        if packed {
            assert_eq!(
                download(&mut metadata_slab, &mut staging, &mut stream)?,
                payload
            );
        }
        assert_eq!(
            download(&mut output, &mut staging, &mut stream)?,
            download(&mut eager, &mut staging, &mut stream)?
        );
        assert_eq!(download(&mut key, &mut staging, &mut stream)?, key_bytes);
        assert_eq!(
            download(&mut value, &mut staging, &mut stream)?,
            value_bytes
        );
    }
    for buffer in [
        metadata_slab,
        positions,
        slots,
        valid,
        ids,
        offsets,
        eager,
        output,
        isolated_value,
        isolated_key,
        value,
        key,
        query,
    ] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    close_context(context)
}

#[test]
fn parent_layer_shared_gqa_replay_and_isolation() -> TestResult {
    check_parent_layer(QUERY_HEAD_COUNT, false)
}

#[test]
fn parent_layer_generic_replay_and_isolation() -> TestResult {
    check_parent_layer(KEY_VALUE_HEAD_COUNT, false)
}

#[test]
fn packed_parent_layer_shared_gqa_replay_and_single_lease_close() -> TestResult {
    check_parent_layer(QUERY_HEAD_COUNT, true)
}

#[test]
fn packed_parent_layer_generic_replay_and_single_lease_close() -> TestResult {
    check_parent_layer(KEY_VALUE_HEAD_COUNT, true)
}
