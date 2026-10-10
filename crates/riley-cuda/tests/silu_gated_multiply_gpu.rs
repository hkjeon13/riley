use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, GatedMultiplyParams, SiluParams, gated_multiply,
    silu, silu_gated_multiply,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn first_device() -> TestResult<(CudaRuntime, CudaDevice)> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    Ok((runtime, device))
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
    let output = staging.to_vec()?;
    staging.close()?;
    Ok(output)
}

// Every BF16 gate bit pattern is paired with several up-branch values, so the
// one-kernel path is compared with the two-kernel path over the whole gate
// domain including infinities, NaNs, and subnormals.
#[test]
#[ignore = "remote GPU"]
fn bf16_silu_gated_multiply_matches_separate_kernels_for_every_gate_pattern() -> TestResult {
    const UPS: [u16; 6] = [0x3f80, 0xbf00, 0x3eab, 0x4489, 0x0001, 0x7f80];
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let element_count = 65_536_usize * UPS.len();
    let byte_len = u64::try_from(element_count * 2)?;
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    let baseline = context.allocation_stats()?;

    let mut gate_bytes = Vec::with_capacity(element_count * 2);
    let mut up_bytes = Vec::with_capacity(element_count * 2);
    for up in UPS {
        for gate in 0..=u16::MAX {
            gate_bytes.extend_from_slice(&gate.to_ne_bytes());
            up_bytes.extend_from_slice(&up.to_ne_bytes());
        }
    }
    let gate = upload(&context, &mut stream, &mut staging, &gate_bytes)?;
    let up = upload(&context, &mut stream, &mut staging, &up_bytes)?;
    let mut activated = context.allocate_device_buffer(byte_len)?;
    let mut separate = context.allocate_device_buffer(byte_len)?;
    let mut fused = context.allocate_device_buffer(byte_len)?;
    let elements = u64::try_from(element_count)?;
    {
        let mut params = SiluParams {
            input: CudaBufferSpan::new(&gate, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut activated, CudaDType::BF16, 0, byte_len)?,
            element_count: elements,
        };
        silu(&mut params, &mut stream)?;
    }
    {
        let mut params = GatedMultiplyParams {
            activated_gate: CudaBufferSpan::new(&activated, CudaDType::BF16, 0, byte_len)?,
            up: CudaBufferSpan::new(&up, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut separate, CudaDType::BF16, 0, byte_len)?,
            element_count: elements,
        };
        gated_multiply(&mut params, &mut stream)?;
    }
    {
        let mut params = GatedMultiplyParams {
            activated_gate: CudaBufferSpan::new(&gate, CudaDType::BF16, 0, byte_len)?,
            up: CudaBufferSpan::new(&up, CudaDType::BF16, 0, byte_len)?,
            output: CudaBufferSpanMut::new(&mut fused, CudaDType::BF16, 0, byte_len)?,
            element_count: elements,
        };
        silu_gated_multiply(&mut params, &mut stream)?;
    }
    let expected = download(&context, &mut stream, &mut separate)?;
    let actual = download(&context, &mut stream, &mut fused)?;
    let differing = expected
        .chunks_exact(2)
        .zip(actual.chunks_exact(2))
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(
        differing, 0,
        "fused SiLU multiply differs from the separate kernels"
    );
    println!("silu-gated-multiply elements={element_count} differing={differing}");

    for buffer in [gate, up, activated, separate, fused] {
        buffer.close()?;
    }
    assert_eq!(context.allocation_stats()?, baseline);
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
