//! FlashInfer attention mode of the D128 two-stage attention: every row's
//! output bytes (prompt and decode) must not depend on its tile alignment,
//! neighbouring rows, run order, the chunk a prompt arrives in or whether the
//! step also holds prompt rows; padding rows are zeros; and the output stays
//! within rounding distance of Riley's per-page path. Checked for every head
//! geometry in `GEOMETRIES` at short and long contexts.
#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaExecutionStream, CudaPinnedHostBuffer, CudaRuntime, CudaStream, DecodePartialReductionOrder,
    NativeBf16RaggedPagedSplitGqaD128TwoStageParams,
    NativeBf16RaggedPagedSplitGqaD128TwoStageWorkspaceLayout, PackedBatchHostV1, PackedBatchV1,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge,
    native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma,
    begin_step_capture, end_step_capture, launch_step_graph, set_native_d128_attention_flashinfer,
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

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn pattern(count: usize, multiplier: usize, offset: usize, modulus: usize, scale: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|index| {
            let centered = ((index * multiplier + offset) % modulus) as f32 - modulus as f32 / 2.0;
            bf16_bits(centered / scale).to_ne_bytes()
        })
        .collect()
}

// Non-periodic, high-dynamic-range scores expose rounding at softmax maxima
// that the small periodic fixture can miss. Keep each logical row's seed
// independent of its position in the packed batch.
fn random_pattern(count: usize, mut state: u64, scale: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = ((state >> 40) as f32 / 16_777_216.0 - 0.5) * scale;
            bf16_bits(value).to_ne_bytes()
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
fn row_query(row: Row, row_width: usize, random_seed: Option<u64>) -> Vec<u8> {
    let seed = row.sequence as usize * 7_919 + row.position as usize * 104_729;
    if let Some(fixture) = random_seed {
        return random_pattern(row_width, fixture.wrapping_add(seed as u64), 4.0);
    }
    pattern(row_width, 11, seed % 9_973, 29, 9.0)
}

struct Pools {
    heads: Heads,
    random_seed: Option<u64>,
    graph: bool,
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

fn execute<S: CudaExecutionStream + ?Sized>(
    params: &mut NativeBf16RaggedPagedSplitGqaD128TwoStageParams<'_>,
    prompt: Option<(u64, Option<(u64, u64)>)>,
    stream: &mut S,
) -> TestResult {
    if let Some((rows, generated)) = prompt {
        native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge_prompt_mma(
            params, rows, generated, stream,
        )?;
    } else {
        native_bf16_ragged_paged_split_gqa_d128_tensor_core_global_merge(params, stream)?;
    }
    Ok(())
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
        .flat_map(|row| row_query(*row, row_width, pools.random_seed))
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
        let prompt = prompt_kinds.map(|_| (row_count as u64, generated));
        if pools.graph {
            let graph = {
                let mut batch = stream.begin_command_batch()?;
                let graph = {
                    let mut commands = batch.commands();
                    begin_step_capture(&mut commands)?;
                    // Two layers share a plan, and every capture must still
                    // record its first planner even after an eager call.
                    execute(&mut params, prompt, &mut commands)?;
                    execute(&mut params, prompt, &mut commands)?;
                    end_step_capture(&mut commands)?
                };
                batch.finish()?;
                graph
            };
            for _ in 0..2 {
                let mut batch = stream.begin_command_batch()?;
                launch_step_graph(&mut batch.commands(), &graph)?;
                batch.finish()?;
            }
            graph.close()?;
        } else {
            execute(&mut params, prompt, stream)?;
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
fn flashinfer_rows_are_batch_invariant_and_close_to_the_page_path() -> TestResult {
    let runtime = CudaRuntime::initialize()?;
    for heads in GEOMETRIES {
        check_geometry(&runtime, heads, 20, 24, None)?;
        check_geometry(&runtime, heads, 60, 64, None)?;
        check_geometry(&runtime, heads, 60, 64, Some(0x9e37_79b9_7f4a_7c15))?;
    }
    set_native_d128_attention_flashinfer(0);
    Ok(())
}

/// Runs `rows` with their kinds and returns each row's bytes by address and
/// the padding rows.
fn run_kinds(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    pools: &Pools,
    rows: &[(Row, u32)],
    with_list: bool,
) -> TestResult<(std::collections::HashMap<Row, Vec<u8>>, Vec<Vec<u8>>)> {
    let addresses: Vec<Row> = rows.iter().map(|(row, _)| *row).collect();
    let kinds: Vec<u32> = rows.iter().map(|(_, kind)| *kind).collect();
    let output = run(context, stream, staging, pools, &addresses, Some((&kinds, with_list)))?;
    let by_row = addresses.iter().copied().zip(output.iter().cloned()).collect();
    Ok((by_row, output[rows.len()..].to_vec()))
}

#[allow(clippy::too_many_lines)]
fn check_geometry(
    runtime: &CudaRuntime,
    heads: Heads,
    pages_per_sequence: usize,
    page_capacity: usize,
    random_seed: Option<u64>,
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
    let mut pools = Pools {
        heads,
        random_seed,
        graph: false,
        pages_per_sequence,
        page_capacity,
        key_pool: upload(&context, &mut stream, &mut staging, &random_seed.map_or_else(
            || pattern(pool_elements, 5, 3, 31, 10.0),
            |seed| random_pattern(pool_elements, seed, 4.0),
        ))?,
        value_pool: upload(&context, &mut stream, &mut staging, &random_seed.map_or_else(
            || pattern(pool_elements, 13, 9, 37, 24.0),
            |seed| random_pattern(pool_elements, seed ^ 0xd1b5_4a32_d192_ed03, 2.0),
        ))?,
        offsets: upload(&context, &mut stream, &mut staging, &encode_u32(&sequence_block_offsets))?,
        ids: upload(&context, &mut stream, &mut staging, &encode_u32(&block_ids))?,
        valid: upload(
            &context,
            &mut stream,
            &mut staging,
            &valid_tokens.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<u8>>(),
        )?,
        sequence_block_offsets,
        block_ids,
        valid_tokens,
    };

    let last = (pages_per_sequence * PAGE_SIZE - 1) as u32;
    let prompt = |sequence: u32, positions: std::ops::Range<u32>| -> Vec<(Row, u32)> {
        positions.map(|position| (Row { sequence, position }, 1)).collect()
    };
    let decode = |sequence: u32, positions: &[u32]| -> Vec<(Row, u32)> {
        positions.iter().map(|&position| (Row { sequence, position }, 0)).collect()
    };
    // Prompt runs from position 0, across pages and query tiles, a late long
    // run and a short one; decode rows of every page class of other
    // sequences.
    let run_a = prompt(0, 0..70);
    let run_b = prompt(1, 60..101);
    let run_c = prompt(2, last - 40..last + 1);
    let run_d = prompt(3, 15..18);
    let decodes = [decode(4, &[0, 15, 16, 255, last]), decode(5, &[64, 127, 300, last - 1])].concat();
    let mut reference = Vec::new();
    for part in [&run_a, &decodes[..3], &run_b, &run_c, &decodes[3..], &run_d] {
        reference.extend_from_slice(part);
    }

    set_native_d128_attention_flashinfer(0);
    let (page_path, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &reference, true)?;
    set_native_d128_attention_flashinfer(1);
    let (flashinfer, padding) = run_kinds(&context, &mut stream, &mut staging, &pools, &reference, true)?;
    assert!(padding.iter().flatten().all(|byte| *byte == 0), "padding rows must be zeros");
    pools.graph = true;
    let (graph_output, graph_padding) =
        run_kinds(&context, &mut stream, &mut staging, &pools, &reference, true)?;
    assert!(graph_padding.iter().flatten().all(|byte| *byte == 0));
    assert_eq!(graph_output, flashinfer, "captured attention must match eager attention");
    pools.graph = false;

    let mut checked = 0_usize;
    let mut expect = |output: &std::collections::HashMap<Row, Vec<u8>>, rows: &[(Row, u32)], what: &str| {
        for (row, _) in rows {
            assert_eq!(output[row], flashinfer[row], "{row:?}: output depends on {what}");
            checked += 1;
        }
    };
    // Other orders and query-tile alignments, with and without the list.
    for (lead, with_list) in [(1, false), (2, true), (5, false)] {
        let mut rows = decodes[..lead].to_vec();
        for part in [&run_d, &run_c, &decodes[lead..], &run_b, &run_a] {
            rows.extend_from_slice(part);
        }
        let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &rows, with_list)?;
        expect(&output, &rows, "order or alignment");
    }
    // Decode rows alone (a decode-only step), each run alone, and run A in
    // three prefill chunks (each chunk its own call).
    let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &decodes, false)?;
    expect(&output, &decodes, "the step holding prompt rows");
    for run_rows in [&run_a, &run_b, &run_c, &run_d] {
        let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, run_rows, false)?;
        expect(&output, run_rows, "neighbours");
    }
    for chunk in [&run_a[..5], &run_a[5..37], &run_a[37..]] {
        let mut rows = chunk.to_vec();
        rows.extend_from_slice(&decodes[2..4]);
        let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &rows, false)?;
        expect(&output, &rows, "the prefill chunk");
    }
    // A long prompt (sequence 3 from position 20) in serving-like prefill
    // chunks of different sizes, next to decode rows: each split must give the
    // rows the bytes of the whole run.
    let long_end = (last + 1).min(420);
    let long_run = prompt(3, 20..long_end);
    let mut long_checked = 0_usize;
    let (long_whole, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &long_run, false)?;
    for cuts in [vec![256], vec![100, 356], vec![17, 81, 300], vec![64, 128, 192, 256, 320]] {
        let mut begin = 0_usize;
        for end in cuts.iter().copied().chain(std::iter::once(long_run.len())) {
            let end = end.min(long_run.len());
            if end <= begin {
                continue;
            }
            let mut rows = decodes[..2].to_vec();
            rows.extend_from_slice(&long_run[begin..end]);
            let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, &rows, true)?;
            let chunk_rows = &long_run[begin..end];
            for (row, _) in chunk_rows {
                assert_eq!(output[row], long_whole[row], "{row:?}: long prompt output depends on the chunk {cuts:?}");
            }
            expect(&output, &[], "nothing");
            long_checked += chunk_rows.len();
            begin = end;
        }
    }
    // Each decode row as a single-row step.
    for row in &decodes {
        let (output, _) = run_kinds(&context, &mut stream, &mut staging, &pools, std::slice::from_ref(row), false)?;
        expect(&output, std::slice::from_ref(row), "being alone");
    }

    let mut elements = 0_usize;
    let mut identical = 0_usize;
    let mut worst = 0.0_f32;
    let mut reported = 0_usize;
    for (row, kind) in &reference {
        let mut row_worst = 0.0_f32;
        let mut worst_element = 0_usize;
        for (index, (a, b)) in flashinfer[row].chunks_exact(2).zip(page_path[row].chunks_exact(2)).enumerate() {
            let x = bf16_value(u16::from_ne_bytes([a[0], a[1]]));
            let y = bf16_value(u16::from_ne_bytes([b[0], b[1]]));
            if (x - y).abs() > row_worst {
                row_worst = (x - y).abs();
                worst_element = index;
            }
        }
        if row_worst > 0.03125 && reported < 12 {
            println!(
                "MISMATCH heads={heads:?} pages={pages_per_sequence} row={row:?} kind={kind} max={row_worst} at head={} depth={}",
                worst_element / HEAD_SIZE,
                worst_element % HEAD_SIZE
            );
            reported += 1;
        }
        for (a, b) in flashinfer[row].chunks_exact(2).zip(page_path[row].chunks_exact(2)) {
            let x = bf16_value(u16::from_ne_bytes([a[0], a[1]]));
            let y = bf16_value(u16::from_ne_bytes([b[0], b[1]]));
            assert!(x.is_finite() && y.is_finite(), "non-finite attention output");
            worst = worst.max((x - y).abs());
            elements += 1;
            identical += usize::from(a == b);
        }
    }
    assert!(worst <= 0.03125, "the FlashInfer path differs from the page path by {worst}");
    println!(
        "flashinfer query_heads={} key_value_heads={} pages={pages_per_sequence} random_seed={random_seed:?} invariant_rows={checked} long_chunk_rows={long_checked} elements={elements} identical_to_page_path={identical} max_abs_difference={worst}",
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
