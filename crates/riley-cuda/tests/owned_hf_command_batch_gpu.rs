#![cfg(feature = "cuda-cublas-gemm-probe")]

use riley_cuda::{
    CublasGemmProbeParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaGemmConfig, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
};
use std::error::Error;
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

fn upload(
    c: &CudaContext,
    s: &mut CudaStream,
    h: &mut CudaPinnedHostBuffer,
    bytes: &[u8],
) -> Result<CudaDeviceBuffer> {
    let mut b = c.allocate_device_buffer(bytes.len() as u64)?;
    b.upload_from_slice(0, bytes, h, s)?;
    Ok(b)
}

#[test]
#[ignore = "A100 GPU 2 or 3: owned cuBLAS batch lifetimes and exact arithmetic"]
fn owned_gemm_batch_matches_integer_oracle_and_retains_leases() -> Result {
    let rt = CudaRuntime::initialize()?;
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut other = c.create_stream()?;
    let mut staging = c.allocate_pinned_host_buffer(65536)?;
    let (m, n, k) = (32u64, 256u64, 128u64);
    let input_bytes: Vec<u8> = (0..m * k)
        .flat_map(|i| if i / k < 13 { 0x3f80u16 } else { 0x4000u16 }.to_le_bytes())
        .collect();
    let weight_bytes: Vec<u8> = (0..n * k)
        .flat_map(|i| [0x3f80u16, 0x4000, 0x3f00, 0xbf80][(i / k % 4) as usize].to_le_bytes())
        .collect();
    let input = upload(&c, &mut stream, &mut staging, &input_bytes)?;
    let weight = upload(&c, &mut stream, &mut staging, &weight_bytes)?;
    let mut output = c.allocate_device_buffer(m * n * 2)?;
    let config = CudaGemmConfig::new(m, n, k, 0)?;
    let mut plan = c.prepare_owned_hf_gemm(config)?;
    let stable = c.allocation_stats()?;
    let slots: Vec<u32> = (0..m).map(|r| u32::from(r >= 13)).collect();
    for grouped in [false, true] {
        let mut batch = stream.begin_command_batch()?;
        {
            let mut commands = batch.commands();
            for _ in 0..3 {
                let mut params = CublasGemmProbeParams {
                    input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, m * k * 2)?,
                    weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, n * k * 2)?,
                    output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, m * n * 2)?,
                };
                if grouped {
                    plan.execute_row_groups(&mut params, &slots, None, &mut commands)?;
                } else {
                    plan.execute(&mut params, &mut commands)?;
                }
            }
        }
        let mut invalid_read = vec![0; (m * n * 2) as usize];
        assert!(
            output
                .download_to_slice(0, &mut invalid_read, &mut staging, &mut other)
                .is_err(),
            "another stream must not read an output leased by an unfinished batch"
        );
        batch.finish()?;
        let mut actual = vec![0; (m * n * 2) as usize];
        output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        for row in 0..m {
            for col in 0..n {
                // Every multiplication and accumulation is an exactly representable integer.
                let expected = (if row < 13 { 128.0f32 } else { 256.0 })
                    * [1.0, 2.0, 0.5, -1.0][(col % 4) as usize];
                let offset = ((row * n + col) * 2) as usize;
                assert_eq!(
                    u16::from_le_bytes([actual[offset], actual[offset + 1]]),
                    (expected.to_bits() >> 16) as u16,
                    "row {row} col {col}, grouped {grouped}"
                );
            }
        }
        assert_eq!(
            c.allocation_stats()?,
            stable,
            "execution must reuse cold allocations"
        );
    }
    // The original constructor must keep rejecting command batches.
    {
        let mut probe = c.prepare_cublas_gemm_probe(config)?;
        let mut batch = stream.begin_command_batch()?;
        {
            let mut commands = batch.commands();
            let mut params = CublasGemmProbeParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, m * k * 2)?,
                weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, n * k * 2)?,
                output: CudaBufferSpanMut::new(&mut output, CudaDType::BF16, 0, m * n * 2)?,
            };
            assert!(
                probe
                    .execute(&mut params, &mut commands)
                    .unwrap_err()
                    .to_string()
                    .contains("rejects active command batches")
            );
        }
        batch.finish()?;
        // The failed probe owner is poisoned; Drop still closes the idle native plan.
    }
    plan.close()?;
    output.close()?;
    weight.close()?;
    input.close()?;
    staging.close()?;
    assert!(c.allocation_stats()?.is_zero());
    other.close()?;
    stream.close()?;
    c.close()?;
    Ok(())
}
