//! Remote-only PR13 fixed-M continuous-batch GPU gates.

#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::error::Error;
use std::path::PathBuf;

use rustinfer_cuda::{CudaAllocationStats, CudaContext, CudaRuntime, CudaStream};
use rustinfer_model::{LoadLimits, LoadedModel};
use rustinfer_runtime::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutor, PreparedLlamaBatchExecutorConfig, PreparedLlamaDecode,
    PreparedLlamaDecodeConfig, PreparedLlamaForward, PreparedLlamaForwardConfig,
};
use rustinfer_runtime::paged_kv::{BLOCK_TABLE_V1_VERSION, KV_BLOCK_SIZE};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const TOKENS_A: [u32; 7] = [504, 2_365, 6_354, 16_438, 11_139, 253, 1_890];
const TOKENS_B: [u32; 4] = [504, 2_365, 42, 43];
const BF16_BYTES: usize = 2;
const ONE_GIB: u64 = 1 << 30;

fn checkpoint_path() -> PathBuf {
    std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory")
}

fn checkpoint_load_limits() -> TestResult<LoadLimits> {
    Ok(LoadLimits::default().with_weight_byte_limits(ONE_GIB, ONE_GIB)?)
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());
    Ok((context, stream))
}

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn batch_config(
    max_rows: usize,
    max_input_tokens: usize,
    max_block_entries: usize,
    max_output_slots: usize,
    physical_block_count: usize,
) -> TestResult<PreparedLlamaBatchExecutorConfig> {
    Ok(PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(
            max_rows,
            max_input_tokens,
            max_block_entries,
            max_output_slots,
            physical_block_count,
        )?,
        PreparedLlamaForwardConfig::default(),
    ))
}

fn block_table(length: usize, first_physical: u32) -> TestResult<(Vec<u32>, Vec<u16>)> {
    let block_count = length.div_ceil(KV_BLOCK_SIZE);
    let mut ids = Vec::with_capacity(block_count);
    let mut valid = Vec::with_capacity(block_count);
    for block in 0..block_count {
        ids.push(
            first_physical
                .checked_add(u32::try_from(block)?)
                .ok_or("physical block ID overflow")?,
        );
        let remaining = length - block * KV_BLOCK_SIZE;
        valid.push(u16::try_from(remaining.min(KV_BLOCK_SIZE))?);
    }
    Ok((ids, valid))
}

fn row<'a>(
    tag: u64,
    kind: LlamaBatchRowKind,
    tokens: &'a [u32],
    target_length: usize,
    ids: &'a [u32],
    valid: &'a [u16],
    output_slot: Option<u32>,
) -> TestResult<LlamaBatchRow<'a>> {
    Ok(LlamaBatchRow::new(
        tag,
        kind,
        tokens,
        u32::try_from(target_length)?,
        LlamaBatchBlockTable::new(
            BLOCK_TABLE_V1_VERSION,
            ids,
            valid,
            u32::try_from(target_length)?,
        ),
        output_slot,
    ))
}

fn vocabulary_row_bytes(model: &LoadedModel) -> TestResult<usize> {
    Ok(model
        .spec()
        .embedding()
        .vocabulary_size()
        .checked_mul(BF16_BYTES)
        .ok_or("vocabulary row byte length overflow")?)
}

fn independent_last_logits(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    tokens: &[u32],
) -> TestResult<Vec<u8>> {
    let mut forward = PreparedLlamaForward::prepare(
        model,
        context,
        stream,
        tokens.len(),
        PreparedLlamaForwardConfig::default(),
    )?;
    forward.forward(tokens, stream)?;
    let mut logits = vec![0_u8; vocabulary_row_bytes(model)?];
    forward.download_last_logits(&mut logits, stream)?;
    forward.close()?;
    Ok(logits)
}

fn bf16(value: &[u8]) -> f64 {
    let bits = u32::from(u16::from_ne_bytes([value[0], value[1]])) << 16;
    f64::from(f32::from_bits(bits))
}

fn top1(logits: &[u8]) -> usize {
    logits
        .chunks_exact(BF16_BYTES)
        .enumerate()
        .max_by(|(left_id, left), (right_id, right)| {
            bf16(left)
                .total_cmp(&bf16(right))
                .then_with(|| right_id.cmp(left_id))
        })
        .map_or(0, |(token, _)| token)
}

fn assert_semantic_parity(label: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(top1(actual), top1(expected), "{label} top-1 differs");
    let mut dot = 0.0_f64;
    let mut actual_norm = 0.0_f64;
    let mut expected_norm = 0.0_f64;
    for (actual, expected) in actual
        .chunks_exact(BF16_BYTES)
        .zip(expected.chunks_exact(BF16_BYTES))
    {
        let actual = bf16(actual);
        let expected = bf16(expected);
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    assert!(cosine >= 0.997, "{label} cosine={cosine}");
}

fn assert_report_matches_context(
    report: rustinfer_runtime::llama::PreparedLlamaBatchAllocationReport,
    stats: CudaAllocationStats,
) {
    assert_eq!(report.total_device_bytes(), stats.device_live_bytes());
    assert_eq!(
        report.total_device_allocation_count(),
        stats.device_live_allocations()
    );
    assert_eq!(report.pinned_host_bytes(), stats.pinned_host_live_bytes());
    assert_eq!(
        report.pinned_host_allocation_count(),
        stats.pinned_host_live_allocations()
    );
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn concurrency_one_matches_the_single_request_forward() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let expected = independent_last_logits(&model, &context, &mut stream, &TOKENS_A)?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 8, 8, 1, 32)?,
    )?;
    assert_report_matches_context(batch.allocation_report(), context.allocation_stats()?);
    let (ids, valid) = block_table(TOKENS_A.len(), 0)?;
    let rows = [row(
        1,
        LlamaBatchRowKind::Prefill,
        &TOKENS_A,
        TOKENS_A.len(),
        &ids,
        &valid,
        Some(0),
    )?];
    batch.execute(&rows, &mut stream)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    assert_semantic_parity("concurrency-one", &actual, &expected);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned checkpoint and CUDA GPU on server-4096"]
fn concurrency_one_matches_prepared_decode_for_thirty_two_steps() -> TestResult {
    const DECODE_STEPS: usize = 32;
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let maximum_length = TOKENS_A.len() + DECODE_STEPS;
    let physical_blocks = maximum_length.div_ceil(KV_BLOCK_SIZE);
    let (context, mut stream) = first_context()?;
    let mut reference = PreparedLlamaDecode::prepare(
        &model,
        &context,
        &mut stream,
        TOKENS_A.len(),
        maximum_length,
        PreparedLlamaDecodeConfig::default(),
    )?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 8, physical_blocks, 1, physical_blocks)?,
    )?;

    reference.prefill(&TOKENS_A, &mut stream)?;
    let (prompt_ids, prompt_valid) = block_table(TOKENS_A.len(), 0)?;
    let prompt_rows = [row(
        71,
        LlamaBatchRowKind::Prefill,
        &TOKENS_A,
        TOKENS_A.len(),
        &prompt_ids,
        &prompt_valid,
        Some(0),
    )?];
    batch.execute(&prompt_rows, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut reference_logits = vec![0_u8; row_bytes];
    let mut batch_logits = vec![0_u8; row_bytes];
    reference.download_last_logits(&mut reference_logits, &mut stream)?;
    batch.download_logits(&mut batch_logits, &mut stream)?;
    assert_semantic_parity("decode-step-prefill", &batch_logits, &reference_logits);
    let stable = context.allocation_stats()?;

    for step in 0..DECODE_STEPS {
        let token = u32::try_from(top1(&reference_logits))?;
        reference.decode(token, &mut stream)?;
        let target_length = TOKENS_A.len() + step + 1;
        let (ids, valid) = block_table(target_length, 0)?;
        let tokens = [token];
        let rows = [row(
            71,
            LlamaBatchRowKind::Decode,
            &tokens,
            target_length,
            &ids,
            &valid,
            Some(0),
        )?];
        batch.execute(&rows, &mut stream)?;
        reference.download_last_logits(&mut reference_logits, &mut stream)?;
        batch.download_logits(&mut batch_logits, &mut stream)?;
        assert_semantic_parity(
            &format!("decode-step-{}", step + 1),
            &batch_logits,
            &reference_logits,
        );
        assert_eq!(context.allocation_stats()?, stable, "decode step {step}");
    }

    batch.close()?;
    reference.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn multiple_independent_requests_and_permuted_output_slots_match() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let expected_a = independent_last_logits(&model, &context, &mut stream, &TOKENS_A[..3])?;
    let expected_b = independent_last_logits(&model, &context, &mut stream, &TOKENS_B)?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(2, 8, 8, 2, 32)?,
    )?;
    let (ids_a, valid_a) = block_table(3, 0)?;
    let (ids_b, valid_b) = block_table(TOKENS_B.len(), 1)?;
    let rows = [
        row(
            11,
            LlamaBatchRowKind::Prefill,
            &TOKENS_A[..3],
            3,
            &ids_a,
            &valid_a,
            Some(1),
        )?,
        row(
            22,
            LlamaBatchRowKind::Prefill,
            &TOKENS_B,
            TOKENS_B.len(),
            &ids_b,
            &valid_b,
            Some(0),
        )?,
    ];
    batch.execute(&rows, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    assert_semantic_parity("permuted-slot-b", &actual[..row_bytes], &expected_b);
    assert_semantic_parity("permuted-slot-a", &actual[row_bytes..], &expected_a);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and CUDA GPU on server-4096"]
fn mixed_prefill_chunk_and_decode_match_independent_full_sequences() -> TestResult {
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    let (context, mut stream) = first_context()?;
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(2, 8, 8, 2, 32)?,
    )?;
    let (ids_a3, valid_a3) = block_table(3, 0)?;
    let (ids_b2, valid_b2) = block_table(2, 1)?;
    let initial = [
        row(
            31,
            LlamaBatchRowKind::Prefill,
            &TOKENS_A[..3],
            3,
            &ids_a3,
            &valid_a3,
            Some(0),
        )?,
        row(
            32,
            LlamaBatchRowKind::Prefill,
            &TOKENS_B[..2],
            2,
            &ids_b2,
            &valid_b2,
            Some(1),
        )?,
    ];
    batch.execute(&initial, &mut stream)?;
    let row_bytes = vocabulary_row_bytes(&model)?;
    let mut initial_logits = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut initial_logits, &mut stream)?;
    let decode_token = u32::try_from(top1(&initial_logits[..row_bytes]))?;
    let continuation = &TOKENS_B[2..4];
    let (ids_a4, valid_a4) = block_table(4, 0)?;
    let (ids_b4, valid_b4) = block_table(4, 1)?;
    let mixed = [
        row(
            31,
            LlamaBatchRowKind::Decode,
            std::slice::from_ref(&decode_token),
            4,
            &ids_a4,
            &valid_a4,
            Some(1),
        )?,
        row(
            32,
            LlamaBatchRowKind::Prefill,
            continuation,
            4,
            &ids_b4,
            &valid_b4,
            Some(0),
        )?,
    ];
    batch.execute(&mixed, &mut stream)?;
    let mut actual = vec![0_u8; batch.output_byte_len()?];
    batch.download_logits(&mut actual, &mut stream)?;
    let mut full_a = TOKENS_A[..3].to_vec();
    full_a.push(decode_token);
    let expected_a = independent_last_logits(&model, &context, &mut stream, &full_a)?;
    let expected_b = independent_last_logits(&model, &context, &mut stream, &TOKENS_B)?;
    assert_semantic_parity("mixed-prefill-b", &actual[..row_bytes], &expected_b);
    assert_semantic_parity("mixed-decode-a", &actual[row_bytes..], &expected_a);
    batch.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote 1000-iteration allocation-accounting gate on server-4096"]
fn one_thousand_iterations_do_not_allocate_or_leak() -> TestResult {
    if !env_enabled("RUSTINFER_PR13_LONG_STEPS") {
        eprintln!("pr13-long-gate-skipped env=RUSTINFER_PR13_LONG_STEPS expected=true");
        return Ok(());
    }
    let model = LoadedModel::load(&checkpoint_path(), checkpoint_load_limits()?)?;
    assert!(model.spec().max_sequence_length() >= 1_001);
    let (context, mut stream) = first_context()?;
    let physical_blocks = 1_001_usize.div_ceil(KV_BLOCK_SIZE);
    let mut batch = PreparedLlamaBatchExecutor::prepare(
        &model,
        &context,
        &mut stream,
        batch_config(1, 1, physical_blocks, 1, physical_blocks)?,
    )?;
    let stable = context.allocation_stats()?;
    let mut token = TOKENS_A[0];
    for target_length in 1..=1_001 {
        let (ids, valid) = block_table(target_length, 0)?;
        let kind = if target_length == 1 {
            LlamaBatchRowKind::Prefill
        } else {
            LlamaBatchRowKind::Decode
        };
        let tokens = [token];
        let rows = [row(
            99,
            kind,
            &tokens,
            target_length,
            &ids,
            &valid,
            Some(0),
        )?];
        batch.execute(&rows, &mut stream)?;
        let mut logits = vec![0_u8; batch.output_byte_len()?];
        batch.download_logits(&mut logits, &mut stream)?;
        token = u32::try_from(top1(&logits))?;
        assert_eq!(
            context.allocation_stats()?,
            stable,
            "iteration={target_length}"
        );
    }
    batch.close()?;
    stream.close()?;
    close_context(context)
}
