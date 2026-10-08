#![allow(clippy::too_many_lines)]

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaErrorKind,
    CudaGemmConfig, CudaPinnedHostBuffer, CudaPreparedGemm, CudaRuntime, CudaStream, GemmParams,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

// This is the canonical decode projection shape used by the roadmap.  It is
// large enough to exercise the actual cuBLASLt path without turning this
// lifecycle/parity test into a throughput benchmark.
const MAX_WORKSPACE_BYTES: u64 = 8 * 1024 * 1024;

fn finite_bf16_pattern(element_count: u64, seed: u64) -> TestResult<Vec<u8>> {
    let byte_len = element_count
        .checked_mul(u64::try_from(std::mem::size_of::<u16>())?)
        .ok_or("C05-21 BF16 fixture byte count overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(byte_len)?)?;
    for index in 0..element_count {
        // Keep every input finite and small.  The varied storage words make a
        // byte-for-byte eager comparison stronger than a single repeated value.
        let word = 0x3e00_u16 + u16::try_from(index.wrapping_mul(31).wrapping_add(seed) % 0x0100)?;
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    Ok(bytes)
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
    buffer: &mut CudaDeviceBuffer,
    staging: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> TestResult<Vec<u8>> {
    let mut bytes = vec![0_u8; usize::try_from(buffer.byte_len())?];
    buffer.download_to_slice(0, &mut bytes, staging, stream)?;
    Ok(bytes)
}

fn execute_eager(
    plan: &mut CudaPreparedGemm,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
) -> TestResult {
    let config = plan.config();
    let workspace_bytes = plan.algorithm_metadata().workspace_bytes();
    assert_eq!(workspace.byte_len(), workspace_bytes);
    let workspace_span = if workspace_bytes == 0 {
        None
    } else {
        Some(CudaBufferSpanMut::new(
            workspace,
            CudaDType::U8,
            0,
            workspace_bytes,
        )?)
    };
    let mut params = GemmParams {
        input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
        weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        workspace: workspace_span,
    };
    plan.execute(&mut params, stream)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU"]
fn selected_gemm_preserves_policy_optional_workspace_and_lifetimes() -> TestResult {
    use riley_cuda::{
        BorrowedGemmGraph, BorrowedGemmResources, BorrowedSelectedGemmGraph,
        BorrowedSelectedGemmResources, CudaGemmReductionPolicy,
    };
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    for policy in [
        CudaGemmReductionPolicy::StrictNoSplitV1,
        CudaGemmReductionPolicy::AllowOutputTypeSplitKV1,
        CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1,
    ] {
        for (m, n, k) in [(1, 576, 576), (1, 192, 576), (1, 1536, 576), (1, 576, 1536)] {
            // The zero-workspace cap makes this an explicit effective-no-split fixture.
            let config = CudaGemmConfig::new(m, n, k, 0)?.with_reduction_policy(policy);
            let mut plan = context.prepare_gemm(config)?;
            assert_eq!(plan.algorithm_metadata().workspace_bytes(), 0);
            assert!(plan.algorithm_metadata().split_k() <= 1);
            let metadata = format!("{:?}", plan.algorithm_metadata());
            let mut input = upload(
                &context,
                &mut stream,
                &mut staging,
                &finite_bf16_pattern(m * k, 7)?,
            )?;
            let mut weight = upload(
                &context,
                &mut stream,
                &mut staging,
                &finite_bf16_pattern(n * k, 19)?,
            )?;
            let mut output = upload(
                &context,
                &mut stream,
                &mut staging,
                &vec![0xff; config.output_bytes() as usize],
            )?;
            let mut empty = context.allocate_device_buffer(0)?;
            let mut parent = upload(&context, &mut stream, &mut staging, &vec![0x5a; 4096])?;
            let mut short_output = context.allocate_device_buffer(config.output_bytes() - 1)?;
            let stats = context.allocation_stats()?;
            let error = BorrowedSelectedGemmGraph::prepare(BorrowedSelectedGemmResources {
                stream: &mut stream,
                plan: &mut plan,
                input: &mut input,
                weight: &mut weight,
                output: &mut short_output,
                workspace: None,
            })
            .err()
            .ok_or("short output accepted")?;
            assert_eq!(error.kind(), CudaErrorKind::OutOfRange);
            if policy != CudaGemmReductionPolicy::StrictNoSplitV1 {
                let error = BorrowedGemmGraph::prepare(BorrowedGemmResources {
                    stream: &mut stream,
                    plan: &mut plan,
                    input: &mut input,
                    weight: &mut weight,
                    output: &mut output,
                    workspace: &mut empty,
                })
                .err()
                .ok_or("legacy strict policy changed")?;
                assert_eq!(error.kind(), CudaErrorKind::NotSupported);
            }
            execute_eager(
                &mut plan,
                &input,
                &weight,
                &mut output,
                &mut empty,
                &mut stream,
            )?;
            let expected = download(&mut output, &mut staging, &mut stream)?;
            assert!(
                expected
                    .chunks_exact(2)
                    .all(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7f80 != 0x7f80)
            );
            for with_parent in [false, true] {
                for explicit_close in [false, true] {
                    output.upload_from_slice(
                        0,
                        &vec![0xff; expected.len()],
                        &mut staging,
                        &mut stream,
                    )?;
                    let mut graph =
                        BorrowedSelectedGemmGraph::prepare(BorrowedSelectedGemmResources {
                            stream: &mut stream,
                            plan: &mut plan,
                            input: &mut input,
                            weight: &mut weight,
                            output: &mut output,
                            workspace: if with_parent { Some(&mut parent) } else { None },
                        })?;
                    for _ in 0..16 {
                        graph.replay()?;
                    }
                    if explicit_close {
                        graph.close()?;
                    } else {
                        drop(graph);
                    }
                    assert_eq!(download(&mut output, &mut staging, &mut stream)?, expected);
                    assert_eq!(
                        download(&mut input, &mut staging, &mut stream)?,
                        finite_bf16_pattern(m * k, 7)?
                    );
                    assert_eq!(
                        download(&mut weight, &mut staging, &mut stream)?,
                        finite_bf16_pattern(n * k, 19)?
                    );
                    assert_eq!(
                        download(&mut parent, &mut staging, &mut stream)?,
                        vec![0x5a; 4096]
                    );
                    assert_eq!(stats, context.allocation_stats()?);
                    assert_eq!(plan.config(), config);
                    assert_eq!(format!("{:?}", plan.algorithm_metadata()), metadata);
                }
            }
            short_output.close()?;
            parent.close()?;
            empty.close()?;
            output.close()?;
            weight.close()?;
            input.close()?;
            plan.close()?;
        }
    }
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
