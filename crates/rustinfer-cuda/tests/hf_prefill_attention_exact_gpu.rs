use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use rustinfer_cuda::{
    AttentionBackend, AttentionBackendAvailability, AttentionMask, AttentionPreference,
    CausalSoftmaxInPlaceParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaPinnedHostBuffer, CudaRuntime, CudaStream, PrefillAttentionParams,
    PrefillAttentionRequest, PreparedPrefillAttention, ScaleCausalMaskInPlaceParams,
    causal_softmax_in_place, scale_causal_mask_in_place,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const P128_S: u64 = 128;
const QH: u64 = 9;
const KVH: u64 = 3;
const D: u64 = 64;
const SCALE: f32 = 0.125;

fn fixture_root() -> TestResult<PathBuf> {
    std::env::var_os("RUSTINFER_HF_FIXTURE_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "RUSTINFER_HF_FIXTURE_ROOT is required",
            )
            .into()
        })
}

fn read_case(root: &Path, case: &str, name: &str) -> TestResult<Vec<u8>> {
    Ok(fs::read(root.join(case).join(name))?)
}

fn first_context() -> TestResult<(CudaContext, CudaStream)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(runtime.device_count() > 0, "runner has no CUDA device");
    let context = runtime.device(0)?.create_context()?;
    let stream = context.create_stream()?;
    Ok((context, stream))
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

fn assert_exact(label: &str, actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let first = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
        .unwrap_or(actual.len().min(expected.len()));
    panic!(
        "{label} differs: actual_bytes={} expected_bytes={} first_byte={first} actual={:?} expected={:?}",
        actual.len(),
        expected.len(),
        actual.get(first),
        expected.get(first),
    );
}

fn bf16_mismatch_summary(actual: &[u8], expected: &[u8]) -> (usize, Option<usize>) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len() % 2, 0);
    let mut mismatches = 0;
    let mut first = None;
    for (index, (actual, expected)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        if actual != expected {
            mismatches += 1;
            first.get_or_insert(index);
        }
    }
    (mismatches, first)
}

fn run_fixture(case: &str, token_count: u64) -> TestResult {
    run_fixture_with_completion(case, token_count, false)
}

fn run_fixture_with_completion(case: &str, token_count: u64, iteration_batch: bool) -> TestResult {
    let root = fixture_root()?;
    assert!(root.is_dir(), "missing fixture root {}", root.display());
    let query_bytes = read_case(&root, case, "layer0_q_rope.bf16")?;
    let key_bytes = read_case(&root, case, "layer0_k_rope.bf16")?;
    let value_bytes = read_case(&root, case, "layer0_v_proj.bf16")?;
    let expected_probabilities = read_case(&root, case, "layer0_attention_probs.bf16")?;
    let expected_context = read_case(&root, case, "layer0_attention_context.bf16")?;
    assert_eq!(
        query_bytes.len(),
        usize::try_from(token_count * QH * D * 2)?
    );
    assert_eq!(key_bytes.len(), usize::try_from(token_count * KVH * D * 2)?);
    assert_eq!(value_bytes.len(), key_bytes.len());
    assert_eq!(
        expected_probabilities.len(),
        usize::try_from(QH * token_count * token_count * 2)?
    );
    assert_eq!(expected_context.len(), query_bytes.len());

    let (context, mut stream) = first_context()?;
    assert_eq!(context.compute_capability(), (8, 9));
    let mut upload_staging =
        context.allocate_pinned_host_buffer(u64::try_from(query_bytes.len())?)?;
    let query = upload(&context, &mut stream, &mut upload_staging, &query_bytes)?;
    let key = upload(&context, &mut stream, &mut upload_staging, &key_bytes)?;
    let value = upload(&context, &mut stream, &mut upload_staging, &value_bytes)?;
    upload_staging.close()?;
    let mut output = context.allocate_device_buffer(u64::try_from(query_bytes.len())?)?;

    let request =
        PrefillAttentionRequest::new(1, token_count, QH, KVH, D, SCALE, AttentionMask::Causal);
    let prepared = PreparedPrefillAttention::select(
        &context,
        request,
        AttentionPreference::HuggingFaceEager,
        AttentionBackendAvailability::linked(),
    )?;
    assert_eq!(prepared.backend(), AttentionBackend::HuggingFaceEager);
    assert_eq!(
        prepared.selection_trace().materialized_score_bytes(),
        QH * token_count * token_count * 2
    );
    assert_eq!(
        prepared.selection_trace().layout_copy_bytes(),
        2 * QH * token_count * D * 2
    );
    let mut workspace = context.allocate_device_buffer(prepared.workspace_bytes())?;
    let before = context.allocation_stats()?;
    {
        let workspace_bytes = workspace.byte_len();
        let mut params = PrefillAttentionParams {
            query: CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
            key: CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
            value: CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
            output: CudaBufferSpanMut::new(
                &mut output,
                CudaDType::BF16,
                0,
                u64::try_from(query_bytes.len())?,
            )?,
            workspace: Some(CudaBufferSpanMut::new(
                &mut workspace,
                CudaDType::BF16,
                0,
                workspace_bytes,
            )?),
        };
        if iteration_batch {
            let mut command_batch = stream.begin_command_batch()?;
            {
                let mut commands = command_batch.commands();
                prepared.execute(&mut params, &mut commands)?;
            }
            command_batch.finish()?;
        } else {
            prepared.execute(&mut params, &mut stream)?;
        }
    }
    assert_eq!(
        context.allocation_stats()?,
        before,
        "hot execution allocated"
    );

    let actual_context = download(&context, &mut stream, &mut output)?;
    let workspace_bytes = download(&context, &mut stream, &mut workspace)?;
    let actual_probabilities = &workspace_bytes[..expected_probabilities.len()];
    let probability_summary = bf16_mismatch_summary(actual_probabilities, &expected_probabilities);
    let context_summary = bf16_mismatch_summary(&actual_context, &expected_context);
    eprintln!(
        "{case}: probability_bf16_mismatches={} first={:?}; context_bf16_mismatches={} first={:?}",
        probability_summary.0, probability_summary.1, context_summary.0, context_summary.1,
    );
    assert_exact(
        &format!("{case} context"),
        &actual_context,
        &expected_context,
    );
    assert_exact(
        &format!("{case} probabilities"),
        actual_probabilities,
        &expected_probabilities,
    );

    prepared.close()?;
    workspace.close()?;
    output.close()?;
    value.close()?;
    key.close()?;
    query.close()?;
    stream.close()?;
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "requires remote HF layer fixtures and an RTX 4090"]
fn hf_eager_p128_00_is_byte_exact() -> TestResult {
    run_fixture("perf-0128-00", P128_S)
}

#[test]
#[ignore = "requires remote HF layer fixtures and an RTX 4090"]
fn hf_eager_p128_05_is_byte_exact() -> TestResult {
    run_fixture("perf-0128-05", P128_S)
}

#[test]
#[ignore = "requires remote HF layer fixtures and an RTX 4090"]
fn hf_eager_command_batch_finish_and_close_are_byte_exact() -> TestResult {
    run_fixture_with_completion("perf-0128-00", P128_S, true)
}

#[test]
#[ignore = "requires remote generated torch fixtures and an RTX 4090"]
fn hf_eager_softmax_dispatch_boundaries_are_byte_exact() -> TestResult {
    for token_count in [1_024, 1_025, 2_048, 2_049] {
        run_fixture(&format!("softmax-{token_count}"), token_count)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires remote torch QK diagnostic fixtures and an RTX 4090"]
fn hf_p128_00_scale_and_softmax_diagnostic() -> TestResult {
    let diagnostic_root = PathBuf::from(
        std::env::var_os("RUSTINFER_HF_QK_DEBUG_ROOT")
            .ok_or("RUSTINFER_HF_QK_DEBUG_ROOT is required")?,
    );
    let qk = fs::read(diagnostic_root.join("qk.bf16"))?;
    let expected_scaled = fs::read(diagnostic_root.join("scaled_masked.bf16"))?;
    let expected_probabilities = read_case(
        &fixture_root()?,
        "perf-0128-00",
        "layer0_attention_probs.bf16",
    )?;
    let (context, mut stream) = first_context()?;
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(qk.len())?)?;
    let mut scores = upload(&context, &mut stream, &mut staging, &qk)?;
    staging.close()?;
    let score_bytes = scores.byte_len();
    let mut scale_params = ScaleCausalMaskInPlaceParams {
        scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: P128_S,
        query_head_count: QH,
        scale: SCALE,
    };
    scale_causal_mask_in_place(&mut scale_params, &mut stream)?;
    let actual_scaled = download(&context, &mut stream, &mut scores)?;
    eprintln!(
        "scaled_masked mismatches: {:?}",
        bf16_mismatch_summary(&actual_scaled, &expected_scaled)
    );
    let mut softmax_params = CausalSoftmaxInPlaceParams {
        scores: CudaBufferSpanMut::new(&mut scores, CudaDType::BF16, 0, score_bytes)?,
        token_count: P128_S,
        query_head_count: QH,
    };
    causal_softmax_in_place(&mut softmax_params, &mut stream)?;
    let actual_probabilities = download(&context, &mut stream, &mut scores)?;
    eprintln!(
        "probability mismatches: {:?}",
        bf16_mismatch_summary(&actual_probabilities, &expected_probabilities)
    );
    Ok(())
}
