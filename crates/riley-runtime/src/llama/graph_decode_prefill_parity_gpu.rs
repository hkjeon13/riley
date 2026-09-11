//! Same-process byte parity for P128 prefill and the established M=1 profile2.
//!
//! Declare this as a child of `graph_decode_full` under `cfg(test, feature="cuda")`.
//! These tests are correctness diagnostics, never performance measurements.
use super::*;
use crate::llama::{
    LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
    PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_cuda::CudaContext;
use riley_model::{LoadLimits, LoadedModel};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const PROMPT: usize = 128;
const OUTPUT: usize = 32;
const BLOCKS: usize = 16;
const STATUS_BYTES: usize = 40;
const HELLO_OUTPUT: [u32; OUTPUT] = [
    28, 339, 5248, 253, 1838, 3241, 282, 253, 1443, 929, 3156, 28, 198, 198, 504, 808, 2775, 339,
    1277, 288, 1643, 314, 338, 339, 5248, 2045, 288, 1138, 346, 253, 1443, 282,
];

fn checkpoint() -> TestResult<LoadedModel> {
    let path = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    Ok(LoadedModel::load(
        std::path::Path::new(&path),
        LoadLimits::default(),
    )?)
}

fn prompts(model: &LoadedModel, vocabulary: usize) -> TestResult<[Vec<u32>; 3]> {
    let hello = model.tokenizer().encode(
        &"Hello".repeat(PROMPT),
        riley_model::EncodeOptions::default(),
    )?;
    assert_eq!(hello.len(), PROMPT);
    let vocabulary = u32::try_from(vocabulary)?;
    let diverse_a = (0_u32..128).map(|i| (i * 337 + 11) % vocabulary).collect();
    let diverse_b = (0_u32..128)
        .map(|i| (i * i * 31 + i * 719 + 101) % vocabulary)
        .collect();
    Ok([hello, diverse_a, diverse_b])
}

fn mapping(request: usize) -> Vec<u32> {
    (0..BLOCKS)
        .map(|i| u32::try_from((i * 7 + request * 3) % BLOCKS).expect("bounded mapping"))
        .collect()
}

fn prepare(
    model: &LoadedModel,
    context: &CudaContext,
    stream: &mut CudaStream,
    batched: bool,
) -> TestResult<OwnedLlamaDecodeExecutor> {
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, if batched { PROMPT } else { 1 }, BLOCKS, 1, BLOCKS)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_grouped_ragged_attention_heads()
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_packed_async_metadata()
    .with_vllm_smol_p128_graph();
    assert_eq!(config.vllm_smol_p128_batched_prefill(), batched);
    let mut prepared = PreparedLlamaBatchExecutor::prepare(model, context, stream, config)?;
    let bytes = prepared.owner.layout.bytes_per_kind();
    let zeros = vec![0; usize::try_from(bytes)?];
    let mut io = context.allocate_pinned_host_buffer(bytes)?;
    // Entire physical pools, including padding and unused blocks, are initialized.
    prepared
        .owner
        .key_cache
        .upload_from_slice(0, &zeros, &mut io, stream)?;
    prepared
        .owner
        .value_cache
        .upload_from_slice(0, &zeros, &mut io, stream)?;
    io.close()?;
    let owned = prepared.into_owned_decode_graph(context)?;
    assert_eq!(owned.numerical_profile_id(), "vllm-smol-p128-v1");
    owned.validate_request_shape(PROMPT, OUTPUT)?;
    Ok(owned)
}

fn row<'a>(
    request: usize,
    kind: LlamaBatchRowKind,
    tokens: &'a [u32],
    length: usize,
    physical: &'a [u32],
    valid: &'a [u16],
) -> LlamaBatchRow<'a> {
    let length = u32::try_from(length).expect("bounded request length");
    LlamaBatchRow::new(
        u64::try_from(request + 1).expect("bounded request index"),
        kind,
        tokens,
        length,
        LlamaBatchBlockTable::new(
            crate::paged_kv::BLOCK_TABLE_V1_VERSION,
            &physical[..valid.len()],
            valid,
            length,
        ),
        Some(0),
    )
}

fn valid_tokens(length: usize) -> Vec<u16> {
    let mut valid = vec![16; length.div_ceil(16)];
    *valid.last_mut().expect("positive request length") =
        u16::try_from((length - 1) % 16 + 1).expect("block width");
    valid
}

fn execute(
    owned: &mut OwnedLlamaDecodeExecutor,
    context: &CudaContext,
    request: usize,
    kind: LlamaBatchRowKind,
    tokens: &[u32],
    length: usize,
    physical: &[u32],
) -> TestResult<u32> {
    let valid = valid_tokens(length);
    let rows = [row(request, kind, tokens, length, physical, &valid)];
    let allocations = context.allocation_stats()?;
    let replays = owned.replay_count();
    let logits_bytes = owned.vocabulary_size() * 2;
    assert_eq!(owned.execute(&rows)?.len(), logits_bytes);
    assert_eq!(owned.replay_count(), replays + 1);
    assert_eq!(
        context.allocation_stats()?,
        allocations,
        "hot execution allocated CUDA memory"
    );
    let token = owned.greedy_token()?;
    assert!(usize::try_from(token)? < owned.vocabulary_size());
    Ok(token)
}

fn exact(left: &[u8], right: &[u8], kind: &str, request: usize, position: usize) {
    assert_eq!(left.len(), right.len(), "{kind}: byte lengths differ");
    if let Some(offset) = left.iter().zip(right).position(|(a, b)| a != b) {
        panic!(
            "{kind}: request {request}, position {position}, first mismatching byte {offset}: slow={} fast={}",
            left[offset], right[offset]
        );
    }
}

fn exact_output(
    slow: &OwnedLlamaDecodeExecutor,
    fast: &OwnedLlamaDecodeExecutor,
    request: usize,
    position: usize,
) -> TestResult<u32> {
    let logits = slow.vocabulary_size() * 2;
    assert_eq!(fast.vocabulary_size(), slow.vocabulary_size());
    assert!(slow.output.len() >= logits + STATUS_BYTES);
    assert!(fast.output.len() >= logits + STATUS_BYTES);
    exact(
        &slow.output[..logits],
        &fast.output[..logits],
        "full BF16 logits",
        request,
        position,
    );
    exact(
        &slow.output[logits..logits + STATUS_BYTES],
        &fast.output[logits..logits + STATUS_BYTES],
        "argmax and embedding status",
        request,
        position,
    );
    let status = |offset| {
        u32::from_le_bytes(
            fast.output[offset..offset + 4]
                .try_into()
                .expect("status word"),
        )
    };
    assert_eq!(status(logits + 4), 0, "argmax status");
    assert_eq!(status(logits + 8), 32, "embedding report size");
    assert_eq!(status(logits + 12), 0, "embedding status");
    assert!(
        fast.output[logits + 16..logits + STATUS_BYTES]
            .iter()
            .all(|&v| v == 0)
    );
    assert_eq!(slow.greedy_token()?, fast.greedy_token()?);
    Ok(fast.greedy_token()?)
}

struct CacheBytes {
    keys: Vec<u8>,
    values: Vec<u8>,
}

fn detach_and_snapshot(
    owned: OwnedLlamaDecodeExecutor,
    context: &CudaContext,
    stream: &mut CudaStream,
) -> TestResult<(PreparedLlamaBatchExecutor, CacheBytes)> {
    // Respect the exclusive parent leases: no native handle or raw pointer is
    // exposed while a graph exists. Only completed, detached parents are read.
    let mut parents = owned.graph.close()?;
    let bytes = parents.executor.owner.layout.bytes_per_kind();
    let mut io = context.allocate_pinned_host_buffer(bytes)?;
    let mut keys = vec![0; usize::try_from(bytes)?];
    let mut values = vec![0; keys.len()];
    parents
        .executor
        .owner
        .key_cache
        .download_to_slice(0, &mut keys, &mut io, stream)?;
    parents
        .executor
        .owner
        .value_cache
        .download_to_slice(0, &mut values, &mut io, stream)?;
    io.close()?;
    parents.metadata.close()?;
    parents.result.close()?;
    parents.staging.close()?;
    for buffer in parents.prefill {
        buffer.close()?;
    }
    parents.stream.close()?;
    Ok((parents.executor, CacheBytes { keys, values }))
}

fn exact_cache(slow: &CacheBytes, fast: &CacheBytes, request: usize, position: usize) {
    exact(
        &slow.keys,
        &fast.keys,
        "entire initialized key pool",
        request,
        position,
    );
    exact(
        &slow.values,
        &fast.values,
        "entire initialized value pool",
        request,
        position,
    );
}

fn snapshot_and_reopen(
    slow: OwnedLlamaDecodeExecutor,
    fast: OwnedLlamaDecodeExecutor,
    context: &CudaContext,
    stream: &mut CudaStream,
    request: usize,
    position: usize,
) -> TestResult<(OwnedLlamaDecodeExecutor, OwnedLlamaDecodeExecutor)> {
    let allocations = context.allocation_stats()?;
    let (slow, slow_cache) = detach_and_snapshot(slow, context, stream)?;
    let (fast, fast_cache) = detach_and_snapshot(fast, context, stream)?;
    exact_cache(&slow_cache, &fast_cache, request, position);
    let slow = slow.into_owned_decode_graph(context)?;
    let fast = fast.into_owned_decode_graph(context)?;
    assert_eq!(
        context.allocation_stats()?,
        allocations,
        "snapshot/reopen leaked CUDA memory"
    );
    Ok((slow, fast))
}

fn reject_without_submission(
    fast: &mut OwnedLlamaDecodeExecutor,
    context: &CudaContext,
    request: usize,
    kind: LlamaBatchRowKind,
    tokens: &[u32],
    length: usize,
    physical: &[u32],
) -> TestResult {
    let valid = valid_tokens(length);
    let rows = [row(request, kind, tokens, length, physical, &valid)];
    let allocations = context.allocation_stats()?;
    let replays = fast.replay_count();
    assert!(
        fast.execute(&rows).is_err(),
        "unsupported stage/shape was accepted"
    );
    assert!(
        fast.greedy_token().is_err(),
        "rejected input exposed the previous output"
    );
    assert_eq!(
        fast.replay_count(),
        replays,
        "rejected input submitted a graph"
    );
    assert_eq!(context.allocation_stats()?, allocations);
    Ok(())
}

fn assert_empty(context: &CudaContext) -> TestResult {
    let stats = context.allocation_stats()?;
    assert_eq!(stats.device_live_allocations(), 0);
    assert_eq!(stats.pinned_host_live_allocations(), 0);
    Ok(())
}

#[test]
#[ignore = "requires SM89 CUDA13 and RILEY_REAL_CHECKPOINT; recaptures for full KV snapshots"]
fn batched_prefill_exact_logits_status_and_every_decode_kv() -> TestResult {
    let model = checkpoint()?;
    let context = riley_cuda::CudaRuntime::initialize()?
        .device(0)?
        .create_context()?;
    let mut stream = context.create_stream()?;
    let mut slow = prepare(&model, &context, &mut stream, false)?;
    let mut fast = prepare(&model, &context, &mut stream, true)?;
    let prompts = prompts(&model, slow.vocabulary_size())?;
    let mut snapshots = 0;
    for (request, prompt) in prompts.iter().enumerate() {
        let physical = mapping(request);
        for position in 0..PROMPT {
            execute(
                &mut slow,
                &context,
                request,
                LlamaBatchRowKind::Prefill,
                &prompt[position..position + 1],
                position + 1,
                &physical,
            )?;
        }
        execute(
            &mut fast,
            &context,
            request,
            LlamaBatchRowKind::Prefill,
            prompt,
            PROMPT,
            &physical,
        )?;
        let mut token = exact_output(&slow, &fast, request, PROMPT - 1)?;
        let mut generated = vec![token];
        (slow, fast) = snapshot_and_reopen(slow, fast, &context, &mut stream, request, PROMPT - 1)?;
        snapshots += 1;
        for position in PROMPT..PROMPT + OUTPUT - 1 {
            let input = [token];
            execute(
                &mut slow,
                &context,
                request,
                LlamaBatchRowKind::Decode,
                &input,
                position + 1,
                &physical,
            )?;
            execute(
                &mut fast,
                &context,
                request,
                LlamaBatchRowKind::Decode,
                &input,
                position + 1,
                &physical,
            )?;
            token = exact_output(&slow, &fast, request, position)?;
            generated.push(token);
            (slow, fast) =
                snapshot_and_reopen(slow, fast, &context, &mut stream, request, position)?;
            snapshots += 1;
        }
        assert_eq!(generated.len(), OUTPUT);
        if request == 0 {
            assert_eq!(generated, HELLO_OUTPUT);
        }
    }
    assert_eq!(snapshots, 3 * OUTPUT);
    slow.close()?;
    fast.close()?;
    stream.close()?;
    assert_empty(&context)?;
    context.close()?;
    println!(
        "P128_PREFILL_PARITY prompts=3 full_logits_exact=true argmax_status_exact=true full_initialized_kv_snapshots=96 every_decode_kv_exact=true snapshot_method=close_and_reopen zero_allocations=true performance_claim=false"
    );
    Ok(())
}

#[test]
#[ignore = "requires SM89 CUDA13 and RILEY_REAL_CHECKPOINT"]
fn batched_prefill_retained_reuse_cancel_and_rejected_output_invalidation() -> TestResult {
    let model = checkpoint()?;
    let context = riley_cuda::CudaRuntime::initialize()?
        .device(0)?
        .create_context()?;
    let mut stream = context.create_stream()?;
    let mut slow = prepare(&model, &context, &mut stream, false)?;
    let mut fast = prepare(&model, &context, &mut stream, true)?;
    let prompts = prompts(&model, slow.vocabulary_size())?;
    // Cancellation here means abandoning completed owner work before scheduler
    // publication. HTTP disconnect or in-flight GPU cancellation is not claimed.
    let cases = [(0, 31), (1, 0), (2, 7), (1, 31), (2, 31), (0, 31)];
    let mut expected_hello = None;
    let mut slow_replays = 0;
    let mut fast_replays = 0;
    for (request, (prompt_index, decode_steps)) in cases.into_iter().enumerate() {
        let prompt = &prompts[prompt_index];
        let physical = mapping(request + 5);
        for position in 0..PROMPT {
            execute(
                &mut slow,
                &context,
                request,
                LlamaBatchRowKind::Prefill,
                &prompt[position..position + 1],
                position + 1,
                &physical,
            )?;
            slow_replays += 1;
        }
        execute(
            &mut fast,
            &context,
            request,
            LlamaBatchRowKind::Prefill,
            prompt,
            PROMPT,
            &physical,
        )?;
        fast_replays += 1;
        let mut token = exact_output(&slow, &fast, request, PROMPT - 1)?;
        let mut generated = vec![token];
        // The first rejection revokes valid output; later rejections keep it
        // unavailable. The next valid prefill/decode must reuse the same owner.
        for (kind, input, length) in [
            (LlamaBatchRowKind::Prefill, &prompt[..127], 127),
            (LlamaBatchRowKind::Prefill, &prompt[..1], 1),
            (LlamaBatchRowKind::Prefill, &prompt[..], 129),
            (LlamaBatchRowKind::Decode, &prompt[..], 128),
            (LlamaBatchRowKind::Decode, &prompt[..1], 1),
            (LlamaBatchRowKind::Prefill, &prompt[..1], 129),
            (LlamaBatchRowKind::Decode, &prompt[..1], 161),
        ] {
            reject_without_submission(
                &mut fast, &context, request, kind, input, length, &physical,
            )?;
        }
        for position in PROMPT..PROMPT + decode_steps {
            let input = [token];
            execute(
                &mut slow,
                &context,
                request,
                LlamaBatchRowKind::Decode,
                &input,
                position + 1,
                &physical,
            )?;
            execute(
                &mut fast,
                &context,
                request,
                LlamaBatchRowKind::Decode,
                &input,
                position + 1,
                &physical,
            )?;
            slow_replays += 1;
            fast_replays += 1;
            token = exact_output(&slow, &fast, request, position)?;
            generated.push(token);
        }
        if request == 0 {
            assert_eq!(generated, HELLO_OUTPUT);
            expected_hello = Some(generated.clone());
        }
        if request == 5 {
            assert_eq!(
                Some(&generated),
                expected_hello.as_ref(),
                "abandoned requests contaminated reused KV"
            );
        }
    }
    assert_eq!(slow.replay_count(), slow_replays);
    assert_eq!(fast.replay_count(), fast_replays);
    assert_eq!(slow_replays, 899);
    assert_eq!(fast_replays, 137);
    let (slow, slow_cache) = detach_and_snapshot(slow, &context, &mut stream)?;
    let (fast, fast_cache) = detach_and_snapshot(fast, &context, &mut stream)?;
    exact_cache(&slow_cache, &fast_cache, 5, 158);
    slow.close()?;
    fast.close()?;
    stream.close()?;
    assert_empty(&context)?;
    context.close()?;
    println!(
        "P128_PREFILL_REUSE requests=6 retained_slow_replays=899 retained_fast_replays=137 cancelled_after_prefill=1 cancelled_after_decode=1 invalid_shape_stage_cases=42 raw_outputs_exact=true full_initialized_final_kv_exact=true zero_allocations=true performance_claim=false"
    );
    Ok(())
}
