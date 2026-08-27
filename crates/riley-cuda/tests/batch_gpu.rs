#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    AttentionReductionProfile, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    DecodeAttentionBackend, DecodeAttentionBackendAvailability, DecodeAttentionPreference,
    FIXED37_RAGGED_MAX_LOGICAL_TOKENS, IndexedRopeParams, PACKED_BATCH_BLOCK_SIZE,
    PackedBatchHostV1, PackedBatchV1, PagedDecodeAttentionNoWorkspaceParams,
    PagedDecodeAttentionRequest, PagedKvBlockTableHostV1, PagedKvBlockTableV1,
    PreparedPagedDecodeAttention, RaggedPagedAttentionParams, RaggedPagedKvCacheWriteParams,
    RopeParams, RowGatherParams, fixed37_ragged_paged_attention, indexed_rope,
    ragged_paged_attention, ragged_paged_kv_cache_write, rope, row_gather,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const D: usize = 64;
const BLOCK: usize = 16;
const SCALE: f32 = 0.125;
const FIXED37: AttentionReductionProfile = AttentionReductionProfile::FixedContiguous37BalancedV1;

fn first_context() -> TestResult<Option<(CudaContext, CudaStream)>> {
    let runtime = match CudaRuntime::initialize() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("skipping remote CUDA test: runtime is unavailable: {error}");
            return Ok(None);
        }
    };
    if runtime.device_count() == 0 {
        eprintln!("skipping remote CUDA test: no CUDA device is visible");
        return Ok(None);
    }
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    Ok(Some((context, stream)))
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let mut buffer = context.allocate_device_buffer(u64::try_from(bytes.len())?)?;
    buffer.upload_from_slice(0, bytes, staging, stream)?;
    Ok(buffer)
}

fn download(
    context: &CudaContext,
    stream: &mut CudaStream,
    buffer: &mut CudaDeviceBuffer,
) -> TestResult<Vec<u8>> {
    let mut staging = context.allocate_pinned_host_buffer(buffer.byte_len())?;
    buffer
        .copy_to_pinned_async(0, &mut staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    let bytes = staging.to_vec()?;
    staging.close()?;
    Ok(bytes)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let is_nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
    if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        u16::try_from(bits.wrapping_add(0x7fff + tie) >> 16).unwrap_or(0x7fff)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn round_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_bits(value))
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16_bits(value).to_ne_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bind_batch<'a>(
    host: PackedBatchHostV1<'a>,
    sequence_block_offsets: &'a CudaDeviceBuffer,
    block_ids: &'a CudaDeviceBuffer,
    valid_tokens: &'a CudaDeviceBuffer,
    row_sequence_slots: &'a CudaDeviceBuffer,
    row_positions: &'a CudaDeviceBuffer,
) -> TestResult<PackedBatchV1<'a>> {
    Ok(PackedBatchV1::new(
        host,
        CudaBufferSpan::new(
            sequence_block_offsets,
            CudaDType::U32,
            0,
            sequence_block_offsets.byte_len(),
        )?,
        CudaBufferSpan::new(block_ids, CudaDType::U32, 0, block_ids.byte_len())?,
        CudaBufferSpan::new(valid_tokens, CudaDType::U16, 0, valid_tokens.byte_len())?,
        CudaBufferSpan::new(
            row_sequence_slots,
            CudaDType::U32,
            0,
            row_sequence_slots.byte_len(),
        )?,
        CudaBufferSpan::new(row_positions, CudaDType::U32, 0, row_positions.byte_len())?,
    )?)
}

fn rope_reference(
    input: &[f32],
    positions: &[u32],
    cos: &[f32],
    sin: &[f32],
    head_count: usize,
    head_size: usize,
    rotary_dimension: usize,
) -> Vec<f32> {
    let half = rotary_dimension / 2;
    let mut output = input.to_vec();
    for (row, &position) in positions.iter().enumerate() {
        let table_base = usize::try_from(position).expect("U32 position fits usize") * half;
        for head in 0..head_count {
            let tensor_base = (row * head_count + head) * head_size;
            for pair in 0..half {
                let first_index = tensor_base + pair;
                let second_index = first_index + half;
                let cosine = round_bf16(cos[table_base + pair]);
                let sine = round_bf16(sin[table_base + pair]);
                let first_cosine = round_bf16(input[first_index] * cosine);
                let second_sine = round_bf16(input[second_index] * sine);
                let second_cosine = round_bf16(input[second_index] * cosine);
                let first_sine = round_bf16(input[first_index] * sine);
                output[first_index] = round_bf16(first_cosine - second_sine);
                output[second_index] = round_bf16(second_cosine + first_sine);
            }
        }
    }
    output
}

fn paged_index(
    physical_block: usize,
    head: usize,
    token_in_block: usize,
    depth: usize,
    head_count: usize,
) -> usize {
    (((physical_block * head_count + head) * BLOCK + token_in_block) * D) + depth
}

fn row_cache_index(
    sequence: usize,
    position: usize,
    head: usize,
    depth: usize,
    head_count: usize,
    sequence_block_offsets: &[u32],
    block_ids: &[u32],
) -> usize {
    let block_begin =
        usize::try_from(sequence_block_offsets[sequence]).expect("U32 block offset fits usize");
    let physical = usize::try_from(block_ids[block_begin + position / BLOCK])
        .expect("U32 block id fits usize");
    paged_index(physical, head, position % BLOCK, depth, head_count)
}

fn sequence_length(sequence: usize, sequence_block_offsets: &[u32], valid_tokens: &[u16]) -> usize {
    let start =
        usize::try_from(sequence_block_offsets[sequence]).expect("U32 block offset fits usize");
    let end =
        usize::try_from(sequence_block_offsets[sequence + 1]).expect("U32 block offset fits usize");
    (end - start - 1) * BLOCK + usize::from(valid_tokens[end - 1])
}

fn cpu_ragged_attention(
    query: &[f32],
    key_pool: &[f32],
    value_pool: &[f32],
    sequence_block_offsets: &[u32],
    block_ids: &[u32],
    row_sequence_slots: &[u32],
    row_positions: &[u32],
    query_head_count: usize,
    key_value_head_count: usize,
) -> Vec<f32> {
    let group_size = query_head_count / key_value_head_count;
    let mut output = vec![0.0_f32; row_positions.len() * query_head_count * D];
    for row in 0..row_positions.len() {
        let sequence = usize::try_from(row_sequence_slots[row]).expect("U32 slot fits usize");
        let logical_tokens =
            usize::try_from(row_positions[row]).expect("U32 position fits usize") + 1;
        for query_head in 0..query_head_count {
            let key_value_head = query_head / group_size;
            let query_base = (row * query_head_count + query_head) * D;
            let mut scores = vec![0.0_f32; logical_tokens];
            for (token, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0_f32;
                for depth in 0..D {
                    let cache = row_cache_index(
                        sequence,
                        token,
                        key_value_head,
                        depth,
                        key_value_head_count,
                        sequence_block_offsets,
                        block_ids,
                    );
                    dot = query[query_base + depth].mul_add(key_pool[cache], dot);
                }
                *score = round_bf16(round_bf16(dot) * SCALE);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
            for depth in 0..D {
                let mut numerator = 0.0_f32;
                for (token, score) in scores.iter().enumerate() {
                    let cache = row_cache_index(
                        sequence,
                        token,
                        key_value_head,
                        depth,
                        key_value_head_count,
                        sequence_block_offsets,
                        block_ids,
                    );
                    numerator = ((*score - maximum).exp()).mul_add(value_pool[cache], numerator);
                }
                output[query_base + depth] = round_bf16(numerator * denominator.recip());
            }
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn indexed_rope_shuffled_positions_matches_cpu_and_legacy_consecutive_path() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;

    let row_count = 5_usize;
    let head_count = 2_usize;
    let head_size = 10_usize;
    let rotary_dimension = 8_usize;
    let table_position_count = 13_usize;
    let positions = [11_u32, 2, 8, 0, 5];
    let consecutive_positions = [4_u32, 5, 6, 7, 8];
    let input_values: Vec<f32> = (0..row_count * head_count * head_size)
        .map(|index| (f32::from(u8::try_from((index * 17 + 5) % 41).unwrap_or(0)) - 20.0) / 16.0)
        .collect();
    let mut cos_values = Vec::with_capacity(table_position_count * rotary_dimension / 2);
    let mut sin_values = Vec::with_capacity(table_position_count * rotary_dimension / 2);
    for position in 0..table_position_count {
        for pair in 0..rotary_dimension / 2 {
            let position = f32::from(u16::try_from(position).unwrap_or(0));
            let exponent = f32::from(u16::try_from(pair).unwrap_or(0))
                / f32::from(u16::try_from(rotary_dimension / 2).unwrap_or(1));
            let angle = position / 10_000.0_f32.powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            cos_values.push(cosine);
            sin_values.push(sine);
        }
    }
    let input_bytes = encode_bf16(&input_values);
    let input_exact = decode_bf16(&input_bytes);
    let expected = rope_reference(
        &input_exact,
        &positions,
        &cos_values,
        &sin_values,
        head_count,
        head_size,
        rotary_dimension,
    );
    let sentinel_bytes = encode_bf16(&vec![-91.0; input_exact.len()]);

    let input = upload(&context, &mut stream, &mut staging, &input_bytes)?;
    let cos = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32(&cos_values),
    )?;
    let sin = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_f32(&sin_values),
    )?;
    let positions_device = upload(&context, &mut stream, &mut staging, &encode_u32(&positions))?;
    let consecutive_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&consecutive_positions),
    )?;
    let mut output = upload(&context, &mut stream, &mut staging, &sentinel_bytes)?;
    let mut indexed_consecutive = upload(&context, &mut stream, &mut staging, &sentinel_bytes)?;
    let mut legacy_consecutive = upload(&context, &mut stream, &mut staging, &sentinel_bytes)?;
    let mut validation_output = upload(&context, &mut stream, &mut staging, &sentinel_bytes)?;
    let baseline = context.allocation_stats()?;

    let invalid_positions = [0_u32, u32::try_from(table_position_count)?];
    let validation_output_len = validation_output.byte_len();
    let error = indexed_rope(
        &mut IndexedRopeParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
            cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
            positions: CudaBufferSpan::new(
                &positions_device,
                CudaDType::U32,
                0,
                positions_device.byte_len(),
            )?,
            positions_host: &invalid_positions,
            output: CudaBufferSpanMut::new(
                &mut validation_output,
                CudaDType::BF16,
                0,
                validation_output_len,
            )?,
            head_count: u64::try_from(head_count)?,
            head_size: u64::try_from(head_size)?,
            rotary_dimension: u64::try_from(rotary_dimension)?,
            table_position_count: u64::try_from(table_position_count)?,
        },
        &mut stream,
    )
    .expect_err("host-mirrored OOB position must fail before launch");
    assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    assert_eq!(
        download(&context, &mut stream, &mut validation_output)?,
        sentinel_bytes,
        "indexed RoPE validation mutated output"
    );

    for _ in 0..3 {
        let output_len = output.byte_len();
        indexed_rope(
            &mut IndexedRopeParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
                sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
                positions: CudaBufferSpan::new(
                    &positions_device,
                    CudaDType::U32,
                    0,
                    positions_device.byte_len(),
                )?,
                positions_host: &positions,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                head_count: u64::try_from(head_count)?,
                head_size: u64::try_from(head_size)?,
                rotary_dimension: u64::try_from(rotary_dimension)?,
                table_position_count: u64::try_from(table_position_count)?,
            },
            &mut stream,
        )?;
    }

    let indexed_consecutive_len = indexed_consecutive.byte_len();
    indexed_rope(
        &mut IndexedRopeParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
            cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
            positions: CudaBufferSpan::new(
                &consecutive_positions_device,
                CudaDType::U32,
                0,
                consecutive_positions_device.byte_len(),
            )?,
            positions_host: &consecutive_positions,
            output: CudaBufferSpanMut::new(
                &mut indexed_consecutive,
                CudaDType::BF16,
                0,
                indexed_consecutive_len,
            )?,
            head_count: u64::try_from(head_count)?,
            head_size: u64::try_from(head_size)?,
            rotary_dimension: u64::try_from(rotary_dimension)?,
            table_position_count: u64::try_from(table_position_count)?,
        },
        &mut stream,
    )?;
    let legacy_consecutive_len = legacy_consecutive.byte_len();
    rope(
        &mut RopeParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
            cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
            sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
            output: CudaBufferSpanMut::new(
                &mut legacy_consecutive,
                CudaDType::BF16,
                0,
                legacy_consecutive_len,
            )?,
            token_count: u64::try_from(row_count)?,
            head_count: u64::try_from(head_count)?,
            head_size: u64::try_from(head_size)?,
            rotary_dimension: u64::try_from(rotary_dimension)?,
            table_position_count: u64::try_from(table_position_count)?,
            position_offset: u64::from(consecutive_positions[0]),
        },
        &mut stream,
    )?;

    assert_eq!(
        download(&context, &mut stream, &mut output)?,
        encode_bf16(&expected),
        "shuffled indexed RoPE differs from the staged BF16 CPU reference"
    );
    assert_eq!(
        download(&context, &mut stream, &mut indexed_consecutive)?,
        download(&context, &mut stream, &mut legacy_consecutive)?,
        "indexed consecutive positions differ from the established RoPE path"
    );
    // Exact same-buffer RoPE is intentionally not exposed by CudaBufferSpanMut;
    // indexed-vs-legacy parity is the strongest safe integration comparison.
    assert_eq!(context.allocation_stats()?, baseline);

    validation_output.close()?;
    legacy_consecutive.close()?;
    indexed_consecutive.close()?;
    output.close()?;
    consecutive_positions_device.close()?;
    positions_device.close()?;
    sin.close()?;
    cos.close()?;
    input.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn row_gather_permutation_and_zero_rows_are_exact_and_allocation_free() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let input_row_count = 6_usize;
    let column_count = 7_usize;
    let permutation = [5_u32, 1, 4, 0, 3, 2];
    let duplicate_indices = [2_u32, 2];
    let input_values: Vec<f32> = (0..input_row_count * column_count)
        .map(|index| (f32::from(u8::try_from((index * 13 + 9) % 47).unwrap_or(0)) - 23.0) / 8.0)
        .collect();
    let input_bytes = encode_bf16(&input_values);
    let output_sentinel = encode_bf16(&vec![-101.0; input_values.len()]);
    let validation_sentinel = encode_bf16(&vec![-77.0; duplicate_indices.len() * column_count]);
    let zero_sentinel = encode_bf16(&vec![33.0; column_count]);
    let mut expected = Vec::with_capacity(input_bytes.len());
    for &row in &permutation {
        let row = usize::try_from(row)?;
        let start = row * column_count * 2;
        expected.extend_from_slice(&input_bytes[start..start + column_count * 2]);
    }

    let input = upload(&context, &mut stream, &mut staging, &input_bytes)?;
    let permutation_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&permutation),
    )?;
    let duplicate_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&duplicate_indices),
    )?;
    let zero_indices_device = context.allocate_device_buffer(4)?;
    let mut output = upload(&context, &mut stream, &mut staging, &output_sentinel)?;
    let mut validation_output = upload(&context, &mut stream, &mut staging, &validation_sentinel)?;
    let mut zero_output = upload(&context, &mut stream, &mut staging, &zero_sentinel)?;
    let baseline = context.allocation_stats()?;

    let validation_output_len = validation_output.byte_len();
    let error = row_gather(
        &mut RowGatherParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
            row_indices: CudaBufferSpan::new(
                &duplicate_device,
                CudaDType::U32,
                0,
                duplicate_device.byte_len(),
            )?,
            row_indices_host: &duplicate_indices,
            output: CudaBufferSpanMut::new(
                &mut validation_output,
                CudaDType::BF16,
                0,
                validation_output_len,
            )?,
            input_row_count: u64::try_from(input_row_count)?,
            column_count: u64::try_from(column_count)?,
        },
        &mut stream,
    )
    .expect_err("duplicate gather rows must fail before launch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    assert_eq!(
        download(&context, &mut stream, &mut validation_output)?,
        validation_sentinel,
        "gather validation mutated output"
    );

    for _ in 0..3 {
        let output_len = output.byte_len();
        row_gather(
            &mut RowGatherParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                row_indices: CudaBufferSpan::new(
                    &permutation_device,
                    CudaDType::U32,
                    0,
                    permutation_device.byte_len(),
                )?,
                row_indices_host: &permutation,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                input_row_count: u64::try_from(input_row_count)?,
                column_count: u64::try_from(column_count)?,
            },
            &mut stream,
        )?;
    }

    let empty: [u32; 0] = [];
    row_gather(
        &mut RowGatherParams {
            input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
            row_indices: CudaBufferSpan::new(&zero_indices_device, CudaDType::U32, 0, 0)?,
            row_indices_host: &empty,
            output: CudaBufferSpanMut::new(&mut zero_output, CudaDType::BF16, 0, 0)?,
            input_row_count: u64::try_from(input_row_count)?,
            column_count: u64::try_from(column_count)?,
        },
        &mut stream,
    )?;

    assert_eq!(
        download(&context, &mut stream, &mut output)?,
        expected,
        "BF16 row permutation must be storage-exact"
    );
    assert_eq!(
        download(&context, &mut stream, &mut zero_output)?,
        zero_sentinel,
        "O=0 gather modified the backing allocation"
    );
    assert_eq!(context.allocation_stats()?, baseline);

    zero_output.close()?;
    validation_output.close()?;
    output.close()?;
    zero_indices_device.close()?;
    duplicate_device.close()?;
    permutation_device.close()?;
    input.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn ragged_kv_scatter_crosses_15_16_17_and_preserves_every_untouched_lane() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let sequence_block_offsets = [0_u32, 2, 3];
    let block_ids = [4_u32, 1, 3];
    let valid_tokens = [16_u16, 2, 5];
    let row_sequence_slots = [0_u32, 1, 0, 0, 1];
    let row_positions = [16_u32, 4, 15, 17, 0];
    let physical_block_count = 6_usize;
    let key_value_head_count = 2_usize;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        u64::try_from(physical_block_count)?,
    )?;
    assert_eq!(PACKED_BATCH_BLOCK_SIZE, u64::try_from(BLOCK)?);

    let source_elements = row_positions.len() * key_value_head_count * D;
    let key_source_values: Vec<f32> = (0..source_elements)
        .map(|index| (f32::from(u8::try_from((index * 11 + 3) % 53).unwrap_or(0)) - 26.0) / 32.0)
        .collect();
    let value_source_values: Vec<f32> = (0..source_elements)
        .map(|index| 0.25 + f32::from(u8::try_from((index * 7 + 5) % 31).unwrap_or(0)) / 32.0)
        .collect();
    let key_source_bytes = encode_bf16(&key_source_values);
    let value_source_bytes = encode_bf16(&value_source_values);
    let key_source_exact = decode_bf16(&key_source_bytes);
    let value_source_exact = decode_bf16(&value_source_bytes);
    let sentinel = round_bf16(-113.0);
    let pool_elements = physical_block_count * key_value_head_count * BLOCK * D;
    let pool_sentinel = encode_bf16(&vec![sentinel; pool_elements]);
    let mut expected_key = vec![sentinel; pool_elements];
    let mut expected_value = vec![sentinel; pool_elements];
    for row in 0..row_positions.len() {
        let sequence = usize::try_from(row_sequence_slots[row])?;
        let position = usize::try_from(row_positions[row])?;
        for head in 0..key_value_head_count {
            for depth in 0..D {
                let source = (row * key_value_head_count + head) * D + depth;
                let destination = row_cache_index(
                    sequence,
                    position,
                    head,
                    depth,
                    key_value_head_count,
                    &sequence_block_offsets,
                    &block_ids,
                );
                expected_key[destination] = key_source_exact[source];
                expected_value[destination] = value_source_exact[source];
            }
        }
    }

    let key_source = upload(&context, &mut stream, &mut staging, &key_source_bytes)?;
    let value_source = upload(&context, &mut stream, &mut staging, &value_source_bytes)?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let mut key_pool = upload(&context, &mut stream, &mut staging, &pool_sentinel)?;
    let mut value_pool = upload(&context, &mut stream, &mut staging, &pool_sentinel)?;
    let baseline = context.allocation_stats()?;

    let key_pool_len = key_pool.byte_len();
    let value_pool_len = value_pool.byte_len();
    let error = ragged_paged_kv_cache_write(
        &mut RaggedPagedKvCacheWriteParams {
            key_source: CudaBufferSpan::new(
                &key_source,
                CudaDType::BF16,
                0,
                key_source.byte_len(),
            )?,
            value_source: CudaBufferSpan::new(
                &value_source,
                CudaDType::BF16,
                0,
                value_source.byte_len(),
            )?,
            key_pool: CudaBufferSpanMut::new(&mut key_pool, CudaDType::BF16, 0, key_pool_len)?,
            value_pool: CudaBufferSpanMut::new(
                &mut value_pool,
                CudaDType::BF16,
                0,
                value_pool_len,
            )?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            key_value_head_count: u64::try_from(key_value_head_count)?,
            head_size: 0,
        },
        &mut stream,
    )
    .expect_err("zero head size must fail before either pool is mutated");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    assert_eq!(
        download(&context, &mut stream, &mut key_pool)?,
        pool_sentinel,
        "failed scatter mutated key pool"
    );
    assert_eq!(
        download(&context, &mut stream, &mut value_pool)?,
        pool_sentinel,
        "failed scatter mutated value pool"
    );

    for _ in 0..3 {
        let key_pool_len = key_pool.byte_len();
        let value_pool_len = value_pool.byte_len();
        ragged_paged_kv_cache_write(
            &mut RaggedPagedKvCacheWriteParams {
                key_source: CudaBufferSpan::new(
                    &key_source,
                    CudaDType::BF16,
                    0,
                    key_source.byte_len(),
                )?,
                value_source: CudaBufferSpan::new(
                    &value_source,
                    CudaDType::BF16,
                    0,
                    value_source.byte_len(),
                )?,
                key_pool: CudaBufferSpanMut::new(&mut key_pool, CudaDType::BF16, 0, key_pool_len)?,
                value_pool: CudaBufferSpanMut::new(
                    &mut value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool_len,
                )?,
                batch: bind_batch(
                    host,
                    &offsets_device,
                    &block_ids_device,
                    &valid_tokens_device,
                    &row_slots_device,
                    &row_positions_device,
                )?,
                key_value_head_count: u64::try_from(key_value_head_count)?,
                head_size: u64::try_from(D)?,
            },
            &mut stream,
        )?;
    }

    assert_eq!(
        download(&context, &mut stream, &mut key_pool)?,
        encode_bf16(&expected_key),
        "scatter key destinations or untouched sentinel lanes differ"
    );
    assert_eq!(
        download(&context, &mut stream, &mut value_pool)?,
        encode_bf16(&expected_value),
        "scatter value destinations or untouched sentinel lanes differ"
    );
    assert_eq!(context.allocation_stats()?, baseline);

    value_pool.close()?;
    key_pool.close()?;
    row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    block_ids_device.close()?;
    offsets_device.close()?;
    value_source.close()?;
    key_source.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn ragged_d64_gqa_attention_matches_cpu_for_all_lanes_and_zeroes_tail() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let sequence_block_offsets = [0_u32, 2, 3];
    let block_ids = [4_u32, 1, 3];
    let valid_tokens = [16_u16, 2, 5];
    let row_sequence_slots = [0_u32, 1, 0, 1, 0];
    let row_positions = [17_u32, 4, 0, 2, 16];
    let physical_block_count = 6_usize;
    let query_head_count = 4_usize;
    let key_value_head_count = 2_usize;
    let output_row_count = 8_usize;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        u64::try_from(physical_block_count)?,
    )?;

    let query_values: Vec<f32> = (0..row_positions.len() * query_head_count * D)
        .map(|index| (f32::from(u8::try_from((index * 19 + 7) % 43).unwrap_or(0)) - 21.0) / 64.0)
        .collect();
    let query_bytes = encode_bf16(&query_values);
    let query_exact = decode_bf16(&query_bytes);
    let pool_elements = physical_block_count * key_value_head_count * BLOCK * D;
    let unused_sentinel = -31.0_f32;
    let mut key_values = vec![unused_sentinel; pool_elements];
    let mut value_values = vec![unused_sentinel; pool_elements];
    for sequence in 0..sequence_block_offsets.len() - 1 {
        for position in 0..sequence_length(sequence, &sequence_block_offsets, &valid_tokens) {
            for head in 0..key_value_head_count {
                for depth in 0..D {
                    let cache = row_cache_index(
                        sequence,
                        position,
                        head,
                        depth,
                        key_value_head_count,
                        &sequence_block_offsets,
                        &block_ids,
                    );
                    key_values[cache] = (f32::from(
                        u8::try_from(
                            (sequence * 17 + position * 11 + head * 7 + depth * 5 + 3) % 47,
                        )
                        .unwrap_or(0),
                    ) - 23.0)
                        / 64.0;
                    value_values[cache] = 0.25
                        + f32::from(
                            u8::try_from(
                                (sequence * 13 + position * 7 + head * 11 + depth * 3 + 1) % 31,
                            )
                            .unwrap_or(0),
                        ) / 64.0;
                    assert!(value_values[cache] > 0.0);
                }
            }
        }
    }
    let key_bytes = encode_bf16(&key_values);
    let value_bytes = encode_bf16(&value_values);
    let key_exact = decode_bf16(&key_bytes);
    let value_exact = decode_bf16(&value_bytes);
    let expected = cpu_ragged_attention(
        &query_exact,
        &key_exact,
        &value_exact,
        &sequence_block_offsets,
        &block_ids,
        &row_sequence_slots,
        &row_positions,
        query_head_count,
        key_value_head_count,
    );
    assert!(expected.iter().all(|&value| value > 0.0));

    let query = upload(&context, &mut stream, &mut staging, &query_bytes)?;
    let key_pool = upload(&context, &mut stream, &mut staging, &key_bytes)?;
    let value_pool = upload(&context, &mut stream, &mut staging, &value_bytes)?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let output_sentinel = encode_bf16(&vec![-97.0; output_row_count * query_head_count * D]);
    let mut output = upload(&context, &mut stream, &mut staging, &output_sentinel)?;
    let baseline = context.allocation_stats()?;

    let output_len = output.byte_len();
    let error = ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            query_head_count: u64::try_from(query_head_count)?,
            key_value_head_count: u64::try_from(key_value_head_count)?,
            head_size: u64::try_from(D)?,
            output_row_count: u64::try_from(output_row_count)?,
            scale: 0.0,
        },
        &mut stream,
    )
    .expect_err("non-positive scale must fail before output mutation");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);

    let output_len = output.byte_len();
    let error = ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            query_head_count: u64::try_from(query_head_count)?,
            key_value_head_count: u64::try_from(key_value_head_count)?,
            head_size: u64::try_from(D)?,
            output_row_count: u64::try_from(row_positions.len() - 1)?,
            scale: SCALE,
        },
        &mut stream,
    )
    .expect_err("M<T must fail before output mutation");
    assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
    assert_eq!(
        download(&context, &mut stream, &mut output)?,
        output_sentinel,
        "attention validation mutated output"
    );

    for _ in 0..3 {
        let output_len = output.byte_len();
        ragged_paged_attention(
            &mut RaggedPagedAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                batch: bind_batch(
                    host,
                    &offsets_device,
                    &block_ids_device,
                    &valid_tokens_device,
                    &row_slots_device,
                    &row_positions_device,
                )?,
                query_head_count: u64::try_from(query_head_count)?,
                key_value_head_count: u64::try_from(key_value_head_count)?,
                head_size: u64::try_from(D)?,
                output_row_count: u64::try_from(output_row_count)?,
                scale: SCALE,
            },
            &mut stream,
        )?;
    }

    let actual = decode_bf16(&download(&context, &mut stream, &mut output)?);
    let active_elements = row_positions.len() * query_head_count * D;
    assert_close(
        &actual[..active_elements],
        &expected,
        0.03125,
        "ragged D64 GQA",
    );
    assert!(
        actual[..active_elements].iter().all(|&value| value > 0.0),
        "strictly positive V lanes must produce nonzero positive active outputs"
    );
    assert!(
        actual[active_elements..]
            .iter()
            .all(|value| value.to_bits() == 0.0_f32.to_bits()),
        "every row in [T,M) must be storage-exact zero across all heads and 64 lanes"
    );
    assert_eq!(context.allocation_stats()?, baseline);

    output.close()?;
    row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    block_ids_device.close()?;
    offsets_device.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn fixed37_ragged_boundaries_match_two_pass_paged_siblings_byte_exact() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;

    let maximum_tokens = usize::try_from(FIXED37_RAGGED_MAX_LOGICAL_TOKENS)?;
    let logical_block_count = maximum_tokens.div_ceil(BLOCK);
    let physical_block_count = logical_block_count;
    let sequence_block_offsets = [0_u32, u32::try_from(logical_block_count)?];
    let block_ids: Vec<u32> = (0..logical_block_count)
        .map(|logical| u32::try_from(logical_block_count - 1 - logical))
        .collect::<Result<_, _>>()?;
    let valid_tokens = vec![u16::try_from(BLOCK)?; logical_block_count];
    let row_positions = [37_u32, 0, 8_191, 35, 36];
    let row_sequence_slots = [0_u32; 5];
    let logical_lengths: Vec<usize> = row_positions
        .iter()
        .map(|&position| usize::try_from(position).map(|position| position + 1))
        .collect::<Result<_, _>>()?;
    assert_eq!(logical_lengths, [38, 1, 8_192, 36, 37]);
    let output_row_count = 8_usize;
    let query_head_count = 2_usize;
    let key_value_head_count = 1_usize;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        u64::try_from(physical_block_count)?,
    )?;

    let query_values: Vec<f32> = (0..row_positions.len() * query_head_count * D)
        .map(|index| (f32::from(u8::try_from((index * 29 + 7) % 101).unwrap_or(0)) - 50.0) / 64.0)
        .collect();
    let pool_elements = physical_block_count * key_value_head_count * BLOCK * D;
    let mut key_values = vec![f32::NAN; pool_elements];
    let mut value_values = vec![f32::NAN; pool_elements];
    for token in 0..maximum_tokens {
        for depth in 0..D {
            let cache = row_cache_index(
                0,
                token,
                0,
                depth,
                key_value_head_count,
                &sequence_block_offsets,
                &block_ids,
            );
            key_values[cache] =
                (f32::from(u8::try_from((token * 17 + depth * 11 + 3) % 127).unwrap_or(0)) - 63.0)
                    / 64.0;
            value_values[cache] =
                (f32::from(u8::try_from((token * 13 + depth * 7 + 5) % 113).unwrap_or(0)) - 56.0)
                    / 32.0;
        }
    }

    let query = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&query_values),
    )?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&key_values),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&value_values),
    )?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;

    let sibling_block_ids: Vec<Vec<u32>> = logical_lengths
        .iter()
        .map(|&logical_tokens| block_ids[..logical_tokens.div_ceil(BLOCK)].to_vec())
        .collect();
    let sibling_valid_tokens: Vec<Vec<u16>> = logical_lengths
        .iter()
        .map(|&logical_tokens| {
            let mut valid =
                vec![u16::try_from(BLOCK).unwrap_or(16); logical_tokens.div_ceil(BLOCK)];
            if let Some(last) = valid.last_mut() {
                *last = u16::try_from((logical_tokens - 1) % BLOCK + 1).unwrap_or(16);
            }
            valid
        })
        .collect();
    let mut sibling_id_devices = Vec::with_capacity(row_positions.len());
    let mut sibling_valid_devices = Vec::with_capacity(row_positions.len());
    for row in 0..row_positions.len() {
        sibling_id_devices.push(upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u32(&sibling_block_ids[row]),
        )?);
        sibling_valid_devices.push(upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u16(&sibling_valid_tokens[row]),
        )?);
    }

    let row_bytes = query_head_count * D * 2;
    let fixed_output_sentinel = encode_bf16(&vec![-97.0; output_row_count * query_head_count * D]);
    let mut fixed_output = upload(&context, &mut stream, &mut staging, &fixed_output_sentinel)?;
    let mut sibling_output = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&vec![-71.0; row_positions.len() * query_head_count * D]),
    )?;
    let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
        &context,
        PagedDecodeAttentionRequest::new(
            u64::try_from(maximum_tokens)?,
            u64::try_from(physical_block_count)?,
            u64::try_from(query_head_count)?,
            u64::try_from(key_value_head_count)?,
            u64::try_from(D)?,
            SCALE,
        ),
        DecodeAttentionPreference::Optimized,
        FIXED37,
        DecodeAttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
    assert_eq!(prepared.workspace_bytes(), 0);
    let active = context.allocation_stats()?;

    for _ in 0..3 {
        let fixed_output_len = fixed_output.byte_len();
        fixed37_ragged_paged_attention(
            &mut RaggedPagedAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(
                    &mut fixed_output,
                    CudaDType::BF16,
                    0,
                    fixed_output_len,
                )?,
                batch: bind_batch(
                    host,
                    &offsets_device,
                    &block_ids_device,
                    &valid_tokens_device,
                    &row_slots_device,
                    &row_positions_device,
                )?,
                query_head_count: u64::try_from(query_head_count)?,
                key_value_head_count: u64::try_from(key_value_head_count)?,
                head_size: u64::try_from(D)?,
                output_row_count: u64::try_from(output_row_count)?,
                scale: SCALE,
            },
            &mut stream,
        )?;
        assert_eq!(context.allocation_stats()?, active);
    }

    for row in 0..row_positions.len() {
        let host_table = PagedKvBlockTableHostV1::new(
            &sibling_block_ids[row],
            &sibling_valid_tokens[row],
            u64::try_from(logical_lengths[row])?,
            u64::try_from(physical_block_count)?,
        )?;
        prepared.execute_without_workspace(
            &mut PagedDecodeAttentionNoWorkspaceParams {
                query: CudaBufferSpan::new(
                    &query,
                    CudaDType::BF16,
                    u64::try_from(row * row_bytes)?,
                    u64::try_from(row_bytes)?,
                )?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(
                    &mut sibling_output,
                    CudaDType::BF16,
                    u64::try_from(row * row_bytes)?,
                    u64::try_from(row_bytes)?,
                )?,
                block_table: PagedKvBlockTableV1::new(
                    host_table,
                    CudaBufferSpan::new(
                        &sibling_id_devices[row],
                        CudaDType::U32,
                        0,
                        sibling_id_devices[row].byte_len(),
                    )?,
                    CudaBufferSpan::new(
                        &sibling_valid_devices[row],
                        CudaDType::U16,
                        0,
                        sibling_valid_devices[row].byte_len(),
                    )?,
                )?,
            },
            &mut stream,
        )?;
        assert_eq!(context.allocation_stats()?, active);
    }

    let fixed_bytes = download(&context, &mut stream, &mut fixed_output)?;
    let sibling_bytes = download(&context, &mut stream, &mut sibling_output)?;
    let active_bytes = row_positions.len() * row_bytes;
    assert_eq!(
        &fixed_bytes[..active_bytes],
        sibling_bytes.as_slice(),
        "fixed37 ragged rows T=1/36/37/38/8192 must byte-match their paged two-pass siblings"
    );
    assert!(
        fixed_bytes[active_bytes..].iter().all(|&byte| byte == 0),
        "every BF16 lane in ragged rows [T,M) must be storage-exact +0"
    );
    assert_eq!(context.allocation_stats()?, active);

    drop(prepared);
    sibling_output.close()?;
    fixed_output.close()?;
    for buffer in sibling_valid_devices {
        buffer.close()?;
    }
    for buffer in sibling_id_devices {
        buffer.close()?;
    }
    row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    block_ids_device.close()?;
    offsets_device.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn fixed37_ragged_special_scores_and_zero_times_infinity_are_full_qnan() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 16)?;
    let sequence_block_offsets = [0_u32, 1, 2, 3, 4];
    let block_ids = [3_u32, 1, 0, 2];
    let valid_tokens = [1_u16, 1, 1, 2];
    let row_sequence_slots = [2_u32, 0, 3, 1];
    let row_positions = [0_u32, 0, 1, 0];
    let physical_block_count = 4_usize;
    let output_row_count = 5_usize;
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        u64::try_from(physical_block_count)?,
    )?;

    let mut query_values = vec![0.0_f32; row_positions.len() * D];
    query_values[0] = 1.0;
    query_values[D] = f32::NAN;
    query_values[2 * D] = 1.0;
    query_values[3 * D] = 1.0;
    let pool_elements = physical_block_count * BLOCK * D;
    let mut key_values = vec![0.0_f32; pool_elements];
    let mut value_values = vec![1.0_f32; pool_elements];

    let plus_infinity = row_cache_index(1, 0, 0, 0, 1, &sequence_block_offsets, &block_ids);
    key_values[plus_infinity] = f32::INFINITY;
    let all_negative_infinity = row_cache_index(2, 0, 0, 0, 1, &sequence_block_offsets, &block_ids);
    key_values[all_negative_infinity] = f32::NEG_INFINITY;
    for depth in 0..D {
        let finite_value = row_cache_index(3, 0, 0, depth, 1, &sequence_block_offsets, &block_ids);
        value_values[finite_value] = 2.0;
        let zero_probability_infinity =
            row_cache_index(3, 1, 0, depth, 1, &sequence_block_offsets, &block_ids);
        value_values[zero_probability_infinity] = f32::INFINITY;
    }
    let negative_infinity_score =
        row_cache_index(3, 1, 0, 0, 1, &sequence_block_offsets, &block_ids);
    key_values[negative_infinity_score] = f32::NEG_INFINITY;

    let query = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&query_values),
    )?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&key_values),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&value_values),
    )?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let mut output = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&vec![-101.0; output_row_count * D]),
    )?;
    let active = context.allocation_stats()?;
    let output_len = output.byte_len();
    fixed37_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            query_head_count: 1,
            key_value_head_count: 1,
            head_size: u64::try_from(D)?,
            output_row_count: u64::try_from(output_row_count)?,
            scale: 1.0,
        },
        &mut stream,
    )?;
    assert_eq!(context.allocation_stats()?, active);

    let actual = download(&context, &mut stream, &mut output)?;
    let active_bytes = row_positions.len() * D * 2;
    let qnan_lane = 0x7fff_u16.to_ne_bytes();
    assert!(
        actual[..active_bytes]
            .chunks_exact(2)
            .all(|lane| lane == qnan_lane),
        "NaN, +Inf, all -Inf, and 0*Inf witnesses must each produce a full canonical qNaN row"
    );
    assert!(actual[active_bytes..].iter().all(|&byte| byte == 0));

    output.close()?;
    row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    block_ids_device.close()?;
    offsets_device.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn fixed37_ragged_corrupt_device_metadata_is_full_qnan_without_oob() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut staging = context.allocate_pinned_host_buffer(1 << 16)?;
    let sequence_block_offsets = [0_u32, 1];
    let host_block_ids = [0_u32];
    let corrupt_block_ids = [1_u32];
    let valid_tokens = [1_u16];
    let row_sequence_slots = [0_u32];
    let row_positions = [0_u32];
    let oversized_device_row_positions = [1_u32];
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &host_block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        1,
    )?;
    let query = upload(&context, &mut stream, &mut staging, &encode_bf16(&[1.0; D]))?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[1.0; BLOCK * D]),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[2.0; BLOCK * D]),
    )?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let valid_block_ids_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&host_block_ids),
    )?;
    let corrupt_block_ids_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&corrupt_block_ids),
    )?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let valid_row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let oversized_row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&oversized_device_row_positions),
    )?;
    let output_row_count = 2_usize;
    let output_sentinel = encode_bf16(&vec![-83.0; output_row_count * D]);
    let mut output = upload(&context, &mut stream, &mut staging, &output_sentinel)?;
    let active = context.allocation_stats()?;

    for (case, block_ids_device, row_positions_device) in [
        (
            "out-of-pool block ID",
            &corrupt_block_ids_device,
            &valid_row_positions_device,
        ),
        (
            "device row T exceeds host maximum T",
            &valid_block_ids_device,
            &oversized_row_positions_device,
        ),
    ] {
        output.upload_from_slice(0, &output_sentinel, &mut staging, &mut stream)?;
        for _ in 0..3 {
            let output_len = output.byte_len();
            fixed37_ragged_paged_attention(
                &mut RaggedPagedAttentionParams {
                    query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    key_pool: CudaBufferSpan::new(
                        &key_pool,
                        CudaDType::BF16,
                        0,
                        key_pool.byte_len(),
                    )?,
                    value_pool: CudaBufferSpan::new(
                        &value_pool,
                        CudaDType::BF16,
                        0,
                        value_pool.byte_len(),
                    )?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                    batch: bind_batch(
                        host,
                        &offsets_device,
                        block_ids_device,
                        &valid_tokens_device,
                        &row_slots_device,
                        row_positions_device,
                    )?,
                    query_head_count: 1,
                    key_value_head_count: 1,
                    head_size: u64::try_from(D)?,
                    output_row_count: u64::try_from(output_row_count)?,
                    scale: SCALE,
                },
                &mut stream,
            )?;
            assert_eq!(context.allocation_stats()?, active);
        }

        let actual = download(&context, &mut stream, &mut output)?;
        let active_bytes = D * 2;
        let qnan_lane = 0x7fff_u16.to_ne_bytes();
        assert!(
            actual[..active_bytes]
                .chunks_exact(2)
                .all(|lane| lane == qnan_lane),
            "corrupt metadata case {case} must produce a full canonical qNaN row"
        );
        assert!(
            actual[active_bytes..].iter().all(|&byte| byte == 0),
            "corrupt metadata case {case} must not disturb exact-zero tail rows"
        );
        assert_eq!(context.allocation_stats()?, active);
    }

    output.close()?;
    oversized_row_positions_device.close()?;
    valid_row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    corrupt_block_ids_device.close()?;
    valid_block_ids_device.close()?;
    offsets_device.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the remote CUDA GPU on server-4096"]
fn fixed37_ragged_command_batch_retains_resources_until_finish() -> TestResult {
    let Some((context, mut stream)) = first_context()? else {
        return Ok(());
    };
    let mut other_stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 16)?;
    let sequence_block_offsets = [0_u32, 1];
    let block_ids = [0_u32];
    let valid_tokens = [1_u16];
    let row_sequence_slots = [0_u32];
    let row_positions = [0_u32];
    let host = PackedBatchHostV1::new(
        &sequence_block_offsets,
        &block_ids,
        &valid_tokens,
        &row_sequence_slots,
        &row_positions,
        1,
    )?;
    let query = upload(&context, &mut stream, &mut staging, &encode_bf16(&[1.0; D]))?;
    let key_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[1.0; BLOCK * D]),
    )?;
    let value_pool = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[2.0; BLOCK * D]),
    )?;
    let offsets_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&sequence_block_offsets),
    )?;
    let block_ids_device = upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?;
    let valid_tokens_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u16(&valid_tokens),
    )?;
    let row_slots_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_sequence_slots),
    )?;
    let row_positions_device = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_u32(&row_positions),
    )?;
    let mut output = upload(
        &context,
        &mut stream,
        &mut staging,
        &encode_bf16(&[-59.0; D]),
    )?;
    let active = context.allocation_stats()?;

    let mut command_batch = stream.begin_command_batch()?;
    {
        let mut commands = command_batch.commands();
        let output_len = output.byte_len();
        fixed37_ragged_paged_attention(
            &mut RaggedPagedAttentionParams {
                query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
                value_pool: CudaBufferSpan::new(
                    &value_pool,
                    CudaDType::BF16,
                    0,
                    value_pool.byte_len(),
                )?,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
                batch: bind_batch(
                    host,
                    &offsets_device,
                    &block_ids_device,
                    &valid_tokens_device,
                    &row_slots_device,
                    &row_positions_device,
                )?,
                query_head_count: 1,
                key_value_head_count: 1,
                head_size: u64::try_from(D)?,
                output_row_count: 1,
                scale: SCALE,
            },
            &mut commands,
        )?;
    }

    let output_len = output.byte_len();
    let error = fixed37_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            query_head_count: 1,
            key_value_head_count: 1,
            head_size: u64::try_from(D)?,
            output_row_count: 1,
            scale: SCALE,
        },
        &mut other_stream,
    )
    .expect_err("another stream must not reuse a resource retained by an unfinished batch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    assert_eq!(context.allocation_stats()?, active);

    command_batch.finish()?;
    let output_len = output.byte_len();
    fixed37_ragged_paged_attention(
        &mut RaggedPagedAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(&key_pool, CudaDType::BF16, 0, key_pool.byte_len())?,
            value_pool: CudaBufferSpan::new(
                &value_pool,
                CudaDType::BF16,
                0,
                value_pool.byte_len(),
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: bind_batch(
                host,
                &offsets_device,
                &block_ids_device,
                &valid_tokens_device,
                &row_slots_device,
                &row_positions_device,
            )?,
            query_head_count: 1,
            key_value_head_count: 1,
            head_size: u64::try_from(D)?,
            output_row_count: 1,
            scale: SCALE,
        },
        &mut other_stream,
    )?;
    assert_eq!(context.allocation_stats()?, active);
    assert_eq!(
        download(&context, &mut other_stream, &mut output)?,
        encode_bf16(&[2.0; D]),
        "finish must release the ledger for immediate successful reuse"
    );
    assert_eq!(context.allocation_stats()?, active);

    output.close()?;
    row_positions_device.close()?;
    row_slots_device.close()?;
    valid_tokens_device.close()?;
    block_ids_device.close()?;
    offsets_device.close()?;
    value_pool.close()?;
    key_pool.close()?;
    query.close()?;
    staging.close()?;
    other_stream.close()?;
    stream.close()?;
    close_context(context)
}
