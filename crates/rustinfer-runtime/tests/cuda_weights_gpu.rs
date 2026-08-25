#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaRuntime, CudaStream, EmbeddingParams, embedding,
};
use rustinfer_model::{LoadLimits, LoadedModel, WeightSlot};
use rustinfer_runtime::{CudaUploadedWeights, CudaWeightUploadError};

type TestResult = Result<(), Box<dyn Error>>;

fn download(
    context: &CudaContext,
    stream: &mut CudaStream,
    buffer: &mut CudaDeviceBuffer,
) -> TestResult<Vec<u8>> {
    let mut staging = context.allocate_pinned_host_buffer(buffer.byte_len())?;
    buffer
        .copy_to_pinned_async(0, &mut staging, 0, buffer.byte_len(), stream)?
        .synchronize()?;
    let output = staging.to_vec()?;
    staging.close()?;
    Ok(output)
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint and a remote CUDA GPU"]
fn pinned_smollm2_uploads_once_per_physical_tensor() -> TestResult {
    let checkpoint = std::env::var_os("RUSTINFER_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RUSTINFER_REAL_CHECKPOINT must name the remote checkpoint directory");
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())?;

    let mut expected_physical: BTreeMap<(PathBuf, String), u64> = BTreeMap::new();
    let mut expected_order = Vec::new();
    for &slot in model.weights().bindings().keys() {
        let bound = model.weights().view(slot)?;
        let source = bound.source();
        let byte_len = u64::try_from(bound.view().storage().len())?;
        let key = (
            source.shard_path().to_path_buf(),
            source.tensor_name().to_owned(),
        );
        if let Some(previous) = expected_physical.insert(key, byte_len) {
            assert_eq!(previous, byte_len, "alias byte lengths must be identical");
        } else {
            expected_order.push((
                source.shard_path().to_path_buf(),
                source.tensor_name().to_owned(),
            ));
        }
    }
    let expected_bytes = expected_physical
        .values()
        .try_fold(0_u64, |total, &bytes| {
            total.checked_add(bytes).ok_or("physical byte sum overflow")
        })?;

    assert_eq!(model.weights().bindings().len(), 273);
    assert_eq!(expected_physical.len(), 272);

    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    assert!(context.allocation_stats()?.is_zero());

    let foreign_context = device.create_context()?;
    let mut foreign_stream = foreign_context.create_stream()?;
    let failure =
        CudaUploadedWeights::upload(model.weights(), &context, &mut foreign_stream, 4 * 1024)
            .expect_err("a stream owned by another context must fail before copying");
    match failure {
        CudaWeightUploadError::Cuda(error) => {
            assert_eq!(error.kind(), CudaErrorKind::InvalidState);
        }
        other => panic!("expected CUDA context mismatch, got {other}"),
    }
    assert!(
        context.allocation_stats()?.is_zero(),
        "failed upload must release staging and the first partial device allocation"
    );
    assert!(foreign_context.allocation_stats()?.is_zero());
    foreign_stream.close()?;
    foreign_context.close()?;

    let uploaded =
        CudaUploadedWeights::upload(model.weights(), &context, &mut stream, 4 * 1024 * 1024)?;
    assert_eq!(uploaded.physical_tensor_count(), expected_physical.len());
    assert_eq!(uploaded.total_physical_bytes(), expected_bytes);
    assert_eq!(
        uploaded.physical_index(WeightSlot::TokenEmbedding),
        uploaded.physical_index(WeightSlot::LmHead),
        "tied token embedding and LM head must share one device allocation"
    );

    assert_eq!(uploaded.physical_tensors().len(), expected_order.len());
    for (tensor, (expected_path, expected_name)) in
        uploaded.physical_tensors().iter().zip(&expected_order)
    {
        assert_eq!(tensor.source().shard_path(), expected_path);
        assert_eq!(tensor.source().tensor_name(), expected_name);
    }
    let expected_indices: BTreeMap<_, _> = expected_order
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, source)| (source, index))
        .collect();
    for &slot in model.weights().bindings().keys() {
        let host = model.weights().view(slot)?;
        let device = uploaded.view(slot)?;
        let source_key = (
            host.source().shard_path().to_path_buf(),
            host.source().tensor_name().to_owned(),
        );
        assert_eq!(
            uploaded.physical_index(slot),
            expected_indices.get(&source_key).copied()
        );
        assert_eq!(device.slot(), slot);
        assert_eq!(device.tensor().source(), host.source());
        assert_eq!(device.tensor().dtype(), CudaDType::BF16);
        assert_eq!(device.tensor().shape(), host.view().shape().dimensions());
        assert_eq!(
            device.tensor().byte_len(),
            u64::try_from(host.view().storage().len())?
        );
        assert_eq!(device.span().byte_len(), device.tensor().byte_len());
    }

    let embedding_weight = uploaded.view(WeightSlot::TokenEmbedding)?;
    let lm_head = uploaded.view(WeightSlot::LmHead)?;
    assert_eq!(
        embedding_weight.tensor().source(),
        lm_head.tensor().source()
    );
    assert_eq!(
        embedding_weight.span().byte_len(),
        embedding_weight.tensor().byte_len()
    );
    assert_eq!(
        lm_head.span().byte_len(),
        embedding_weight.span().byte_len()
    );

    let hidden_size = model.spec().embedding().hidden_size();
    let embedding_row_bytes = u64::try_from(hidden_size)?
        .checked_mul(2)
        .ok_or("embedding row byte length overflow")?;
    let mut staging = context.allocate_pinned_host_buffer(4 * 1024)?;
    let mut token_ids = context.allocate_device_buffer(4)?;
    token_ids.upload_from_slice(0, &0_u32.to_ne_bytes(), &mut staging, &mut stream)?;
    let mut output = context.allocate_device_buffer(embedding_row_bytes)?;
    let mut error_scratch = context.allocate_device_buffer(32)?;
    let mut params = EmbeddingParams {
        table: embedding_weight.span(),
        token_ids: CudaBufferSpan::new(&token_ids, CudaDType::U32, 0, 4)?,
        output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, embedding_row_bytes)?,
        error_scratch: CudaBufferSpanMut::new(&mut error_scratch, CudaDType::U8, 0, 32)?,
        token_count: 1,
        vocabulary_size: u64::try_from(model.spec().embedding().vocabulary_size())?,
        hidden_size: u64::try_from(hidden_size)?,
    };
    embedding(&mut params, &mut stream)?;
    let uploaded_row = download(&context, &mut stream, &mut output)?;
    let host_embedding = model.weights().view(WeightSlot::TokenEmbedding)?;
    assert_eq!(
        uploaded_row,
        host_embedding.view().storage()[..usize::try_from(embedding_row_bytes)?],
        "the uploaded token-0 embedding row must match checkpoint payload bytes"
    );
    token_ids.close()?;
    output.close()?;
    error_scratch.close()?;
    staging.close()?;

    let live = context.allocation_stats()?;
    assert_eq!(
        live.device_live_allocations(),
        u64::try_from(expected_physical.len())?
    );
    assert_eq!(live.device_live_bytes(), expected_bytes);
    assert_eq!(live.pinned_host_live_allocations(), 0);
    assert_eq!(live.pinned_host_live_bytes(), 0);
    println!(
        "\nrustinfer-runtime-weight-upload physical_tensors={} device_bytes={} staging_bytes={}",
        uploaded.physical_tensor_count(),
        uploaded.total_physical_bytes(),
        4 * 1024 * 1024
    );

    uploaded.close()?;
    assert!(context.allocation_stats()?.is_zero());
    stream.close()?;
    context.close()?;
    Ok(())
}
