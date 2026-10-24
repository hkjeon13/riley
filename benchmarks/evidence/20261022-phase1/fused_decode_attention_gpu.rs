//! Fused D128 decode attention: with the fused kernel enabled, the
//! tensor-core producer + global-max merge path must write exactly the same
//! output bytes as the two-kernel path, for every row of every arrangement,
//! with and without prompt rows (row kinds and the generated-row list), at
//! short and long contexts, for every head geometry in `GEOMETRIES`.
#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodePartialReductionOrder,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PackedBatchHostV1, PackedBatchV1,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma,
    set_native_d128_fused_decode_min_rows,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Query and key/value head counts of one checked geometry.
#[derive(Clone, Copy, Debug)]
struct Heads {
    query: usize,
    key_value: usize,
}

impl Heads {
    const fn row_width(self) -> usize {
        self.query * HEAD_SIZE
    }
}

/// Qwen2.5-3B first, then groups of 6, 7, 5, 2, 4, 3 and 1 query heads per
/// key/value head.
const GEOMETRIES: [Heads; 8] = [
    Heads {
        query: 16,
        key_value: 2,
    },
    Heads {
        query: 12,
        key_value: 2,
    },
    Heads {
        query: 28,
        key_value: 4,
    },
    Heads {
        query: 40,
        key_value: 8,
    },
    Heads {
        query: 16,
        key_value: 8,
    },
    Heads {
        query: 32,
        key_value: 8,
    },
    Heads {
        query: 24,
        key_value: 8,
    },
    Heads {
        query: 4,
        key_value: 4,
    },
];
const HEAD_SIZE: usize = 128;
const PAGE_SIZE: usize = 16;
const SCALE: f32 = 0.088_388_346;
const SEQUENCES: usize = 6;
/// Output rows launched past the active rows; they must come back as zeros.
const PADDING_ROWS: usize = 3;

/// One logical row: sequence and position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Row {
    sequence: u32,
    position: u32,
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn pattern(count: usize, multiplier: usize, offset: usize, modulus: usize, scale: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|index| {
            let centered = ((index * multiplier + offset) % modulus) as f32 - modulus as f32 / 2.0;
            bf16_bits(centered / scale).to_ne_bytes()
        })
        .collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
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

/// The query of a logical row is a function of the row alone, so the same row
/// carries the same query in every arrangement.
fn row_query(row: Row, row_width: usize) -> Vec<u8> {
    let seed = row.sequence as usize * 7_919 + row.position as usize * 104_729;
    pattern(row_width, 11, seed % 9_973, 29, 9.0)
}

struct Pools {
    heads: Heads,
    pages_per_sequence: usize,
    page_capacity: usize,
    key_pool: CudaDeviceBuffer,
    value_pool: CudaDeviceBuffer,
    offsets: CudaDeviceBuffer,
    ids: CudaDeviceBuffer,
    valid: CudaDeviceBuffer,
    sequence_block_offsets: Vec<u32>,
    block_ids: Vec<u32>,
    valid_tokens: Vec<u16>,
}

/// Runs the attention over `rows` in the given order and returns each active
/// row's output bytes, padding rows included. `prompt_kinds` selects the
/// prompt-row producer with those row kinds (and the generated-row list).
fn run(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    pools: &Pools,
    rows: &[Row],
    prompt_kinds: Option<(&[u32], bool)>,
) -> TestResult<Vec<Vec<u8>>> {
    let row_count = rows.len();
    let row_width = pools.heads.row_width();
    let output_rows = row_count + PADDING_ROWS;
    let slots: Vec<u32> = rows.iter().map(|row| row.sequence).collect();
    let positions: Vec<u32> = rows.iter().map(|row| row.position).collect();
    let host = PackedBatchHostV1::new(
        &pools.sequence_block_offsets,
        &pools.block_ids,
        &pools.valid_tokens,
        &slots,
        &positions,
        u64::try_from(SEQUENCES * pools.pages_per_sequence)?,
    )?;
    let mut query_bytes: Vec<u8> = rows
        .iter()
        .flat_map(|row| row_query(*row, row_width))
        .collect();
    query_bytes.resize(output_rows * row_width * 2, 0);
    let query = upload(context, stream, staging, &query_bytes)?;
    let row_slots = upload(context, stream, staging, &encode_u32(&slots))?;
    // Row kinds follow the positions, then the `[count, rows...]` list of the
    // rows that are not prompt rows.
    let mut position_words = positions.clone();
    let mut generated = None;
    if let Some((kinds, with_list)) = prompt_kinds {
        position_words.extend_from_slice(kinds);
        if with_list {
            let list: Vec<u32> = (0..row_count as u32)
                .filter(|row| kinds[*row as usize] != 1)
                .collect();
            generated = Some((position_words.len() as u64, row_count as u64));
            position_words.push(list.len() as u32);
            position_words.extend_from_slice(&list);
            position_words.resize(position_words.len() + row_count - list.len(), 0);
        }
    }
    let position_buffer = upload(context, stream, staging, &encode_u32(&position_words))?;
    let layout = NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout::with_query_heads(
        u64::try_from(output_rows)?,
        u64::try_from(pools.page_capacity)?,
        u64::try_from(pools.heads.query)?,
    )?;
    let mut partial_states = context.allocate_device_buffer(layout.partial_state_bytes())?;
    let mut reduction_steps = context.allocate_device_buffer(layout.reduction_step_bytes())?;
    let mut reduction_normalizers =
        context.allocate_device_buffer(layout.reduction_normalizer_bytes())?;
    let mut output = upload(
        context,
        stream,
        staging,
        &vec![0x5a_u8; output_rows * row_width * 2],
    )?;
    {
        let (states_len, steps_len, normalizers_len, output_len) = (
            partial_states.byte_len(),
            reduction_steps.byte_len(),
            reduction_normalizers.byte_len(),
            output.byte_len(),
        );
        let mut params = NativeBf16RaggedPagedSplitGqaD128TwoStageParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key_pool: CudaBufferSpan::new(
                &pools.key_pool,
                CudaDType::BF16,
                0,
                pools.key_pool.byte_len(),
            )?,
            value_pool: CudaBufferSpan::new(
                &pools.value_pool,
                CudaDType::BF16,
                0,
                pools.value_pool.byte_len(),
            )?,
            partial_states: CudaBufferSpanMut::new(
                &mut partial_states,
                CudaDType::F32,
                0,
                states_len,
            )?,
            reduction_steps: CudaBufferSpanMut::new(
                &mut reduction_steps,
                CudaDType::F32,
                0,
                steps_len,
            )?,
            reduction_normalizers: CudaBufferSpanMut::new(
                &mut reduction_normalizers,
                CudaDType::F32,
                0,
                normalizers_len,
            )?,
            output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, output_len)?,
            batch: PackedBatchV1::new(
                host,
                CudaBufferSpan::new(&pools.offsets, CudaDType::U32, 0, pools.offsets.byte_len())?,
                CudaBufferSpan::new(&pools.ids, CudaDType::U32, 0, pools.ids.byte_len())?,
                CudaBufferSpan::new(&pools.valid, CudaDType::U16, 0, pools.valid.byte_len())?,
                CudaBufferSpan::new(&row_slots, CudaDType::U32, 0, row_slots.byte_len())?,
                CudaBufferSpan::new(
                    &position_buffer,
                    CudaDType::U32,
                    0,
                    position_buffer.byte_len(),
                )?,
            )?,
            query_head_count: u64::try_from(pools.heads.query)?,
            key_value_head_count: u64::try_from(pools.heads.key_value)?,
            head_size: u64::try_from(HEAD_SIZE)?,
            output_row_count: u64::try_from(output_rows)?,
            partial_state_capacity: u64::try_from(pools.page_capacity)?,
            scale: SCALE,
            reduction_order: DecodePartialReductionOrder::LogicalAscending,
        };
        match prompt_kinds {
            None => {
                native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge(
                    &mut params,
                    stream,
                )?;
            }
            Some(_) => {
                native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma(
                    &mut params,
                    row_count as u64,
                    generated,
                    stream,
                )?;
            }
        }
    }
    stream.synchronize()?;
    let bytes = download(context, stream, &mut output)?;
    for buffer in [
        query,
        row_slots,
        position_buffer,
        partial_states,
        reduction_steps,
        reduction_normalizers,
        output,
    ] {
        buffer.close()?;
    }
    Ok(bytes.chunks_exact(row_width * 2).map(<[u8]>::to_vec).collect())
}

fn rows(sequence: u32, positions: impl IntoIterator<Item = u32>) -> Vec<Row> {
    positions
        .into_iter()
        .map(|position| Row { sequence, position })
        .collect()
}

#[test]
#[ignore = "remote GPU"]
fn fused_decode_matches_the_two_kernel_path_byte_for_byte() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    for heads in GEOMETRIES {
        // Short contexts (capacity 24) and long ones up to the fused kernel's
        // 128-page bound.
        check_geometry(&runtime, heads, 20, 24)?;
        check_geometry(&runtime, heads, 120, 128)?;
    }
    set_native_d128_fused_decode_min_rows(0);
    Ok(())
}

fn check_geometry(
    runtime: &CudaRuntime,
    heads: Heads,
    pages_per_sequence: usize,
    page_capacity: usize,
) -> TestResult {
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(128 << 20)?;

    let physical_blocks = SEQUENCES * pages_per_sequence;
    let pool_elements = physical_blocks * heads.key_value * PAGE_SIZE * HEAD_SIZE;
    let sequence_block_offsets: Vec<u32> = (0..=SEQUENCES)
        .map(|sequence| (sequence * pages_per_sequence) as u32)
        .collect();
    let block_ids: Vec<u32> = (0..physical_blocks)
        .map(|block| ((block * 29 + 5) % physical_blocks) as u32)
        .collect();
    let valid_tokens = vec![PAGE_SIZE as u16; physical_blocks];
    let pools = Pools {
        heads,
        pages_per_sequence,
        page_capacity,
        key_pool: upload(
            &context,
            &mut stream,
            &mut staging,
            &pattern(pool_elements, 5, 3, 31, 10.0),
        )?,
        value_pool: upload(
            &context,
            &mut stream,
            &mut staging,
            &pattern(pool_elements, 13, 9, 37, 24.0),
        )?,
        offsets: upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_u32(&sequence_block_offsets),
        )?,
        ids: upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?,
        valid: upload(
            &context,
            &mut stream,
            &mut staging,
            &valid_tokens
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )?,
        sequence_block_offsets,
        block_ids,
        valid_tokens,
    };

    let last = (pages_per_sequence * PAGE_SIZE - 1) as u32;
    // Prompt runs (kind 1) and single decode rows of every page class: one
    // page, a page boundary on either side, many pages, the last position.
    let mut arrangement = rows(0, 0..40);
    let mut kinds = vec![1_u32; 40];
    for row in rows(1, [37, 15, 16, 127, 128, last]) {
        arrangement.push(row);
        kinds.push(0);
    }
    arrangement.extend(rows(2, 100..131));
    kinds.extend(std::iter::repeat_n(1, 31));
    for row in rows(3, [0, 5, 143, 144, 255, 256, 319, last - 1]) {
        arrangement.push(row);
        kinds.push(0);
    }
    arrangement.extend(rows(4, 280..300));
    kinds.extend(std::iter::repeat_n(1, 20));
    for row in rows(5, [160, 63, 64, 65, 271, last / 2]) {
        arrangement.push(row);
        kinds.push(0);
    }
    let decode_only: Vec<Row> = arrangement
        .iter()
        .zip(&kinds)
        .filter(|(_, kind)| **kind == 0)
        .map(|(row, _)| *row)
        .collect();
    let cases: Vec<(&str, &[Row], Option<(&[u32], bool)>)> = vec![
        ("all-rows-per-page", &arrangement, None),
        ("decode-only", &decode_only, None),
        ("decode-single", &decode_only[..1], None),
        ("prompt-kinds", &arrangement, Some((&kinds, false))),
        ("prompt-kinds-list", &arrangement, Some((&kinds, true))),
    ];
    let mut compared = 0_usize;
    for (name, rows, prompt_kinds) in cases {
        set_native_d128_fused_decode_min_rows(0);
        let two_kernel = run(&context, &mut stream, &mut staging, &pools, rows, prompt_kinds)?;
        set_native_d128_fused_decode_min_rows(1);
        let fused = run(&context, &mut stream, &mut staging, &pools, rows, prompt_kinds)?;
        assert_eq!(two_kernel.len(), fused.len());
        for (index, (left, right)) in two_kernel.iter().zip(&fused).enumerate() {
            assert_eq!(
                left, right,
                "{name} heads={heads:?} pages={pages_per_sequence}: row {index} differs"
            );
            compared += 1;
        }
        let padding = &fused[rows.len()..];
        assert!(
            padding.iter().flatten().all(|byte| *byte == 0),
            "{name}: rows past the active count must be zeros"
        );
    }
    println!(
        "fused-decode query_heads={} key_value_heads={} pages={pages_per_sequence} capacity={page_capacity} identical_rows={compared}",
        heads.query, heads.key_value,
    );

    let Pools {
        key_pool,
        value_pool,
        offsets,
        ids,
        valid,
        ..
    } = pools;
    for buffer in [key_pool, value_pool, offsets, ids, valid] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    context.close()?;
    Ok(())
}
