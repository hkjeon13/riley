//! The wide BF16 RMS norm kernels for hidden sizes other than 2048 must store
//! the same bytes as the 256-thread kernels they replace.
//!
//! Up to 256 rows the primitives launch the wide form; above 256 rows they
//! launch the 256-thread kernels. Each case therefore runs 300 rows in one
//! call (reference) and again as a 256-row and a 44-row call (wide form) and
//! compares every output bit, for the plain and the residual-fused norm.

use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, ResidualRmsNormParams, RmsNormParams,
    residual_rms_norm, rms_norm,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn encode_bf16(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|value| bf16_bits(value).to_ne_bytes())
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

fn values(count: usize, salt: usize) -> impl Iterator<Item = f32> {
    (0..count).map(move |i| {
        let unit = (((i + salt) * 2_654_435_761_usize) >> 9) % 8191;
        let magnitude = [0.004_f32, 0.09, 0.8, 3.1, 11.0][((i + salt) / 7) % 5];
        (unit as f32 / 4095.0 - 1.0) * magnitude
    })
}

#[test]
#[ignore = "remote GPU"]
fn wide_general_rms_norm_matches_the_256_thread_kernels() -> TestResult {
    const ROWS: usize = 300;
    const WIDE_ROWS: usize = 256;
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 24)?;
    let epsilon = 1.0e-6_f32;

    for hidden in [
        512_usize, 576, 896, 960, 1000, 1024, 1536, 2560, 3072, 3584, 4095, 4096, 4097, 5120, 6000, 8192,
    ] {
        let row_bytes = (hidden * 2) as u64;
        let left = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(values(ROWS * hidden, 3)),
        )?;
        let right = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16(values(ROWS * hidden, 1009)),
        )?;
        let weight = upload(
            &context,
            &mut stream,
            &mut staging,
            &encode_bf16((0..hidden).map(|i| 0.5 + ((i * 31 + 5) % 73) as f32 / 48.0)),
        )?;
        let sentinel = encode_bf16(std::iter::repeat_n(-113.0_f32, ROWS * hidden));
        // [reference, wide] x [plain output, residual output, residual-normalized output]
        let mut outputs = Vec::new();
        for wide in [false, true] {
            let mut plain = upload(&context, &mut stream, &mut staging, &sentinel)?;
            let mut residual = upload(&context, &mut stream, &mut staging, &sentinel)?;
            let mut normalized = upload(&context, &mut stream, &mut staging, &sentinel)?;
            let chunks: &[(usize, usize)] = if wide {
                &[(0, WIDE_ROWS), (WIDE_ROWS, ROWS - WIDE_ROWS)]
            } else {
                &[(0, ROWS)]
            };
            for &(first, rows) in chunks {
                let offset = first as u64 * row_bytes;
                let bytes = rows as u64 * row_bytes;
                rms_norm(
                    &mut RmsNormParams {
                        input: CudaBufferSpan::new(&left, CudaDType::BF16, offset, bytes)?,
                        weight: CudaBufferSpan::new(
                            &weight,
                            CudaDType::BF16,
                            0,
                            weight.byte_len(),
                        )?,
                        output: CudaBufferSpanMut::new(&mut plain, CudaDType::BF16, offset, bytes)?,
                        row_count: rows as u64,
                        hidden_size: hidden as u64,
                        epsilon,
                    },
                    &mut stream,
                )?;
                residual_rms_norm(
                    &mut ResidualRmsNormParams {
                        left: CudaBufferSpan::new(&left, CudaDType::BF16, offset, bytes)?,
                        right: CudaBufferSpan::new(&right, CudaDType::BF16, offset, bytes)?,
                        weight: CudaBufferSpan::new(
                            &weight,
                            CudaDType::BF16,
                            0,
                            weight.byte_len(),
                        )?,
                        residual_output: CudaBufferSpanMut::new(
                            &mut residual,
                            CudaDType::BF16,
                            offset,
                            bytes,
                        )?,
                        normalized_output: CudaBufferSpanMut::new(
                            &mut normalized,
                            CudaDType::BF16,
                            offset,
                            bytes,
                        )?,
                        row_count: rows as u64,
                        hidden_size: hidden as u64,
                        epsilon,
                    },
                    &mut stream,
                )?;
            }
            let mut bytes = Vec::new();
            for buffer in [&mut plain, &mut residual, &mut normalized] {
                bytes.push(download(&context, &mut stream, buffer)?);
            }
            outputs.push(bytes);
            for buffer in [plain, residual, normalized] {
                buffer.close()?;
            }
        }
        for (index, name) in ["norm", "residual", "residual norm"].iter().enumerate() {
            let differing = outputs[0][index]
                .chunks_exact(2)
                .zip(outputs[1][index].chunks_exact(2))
                .filter(|(left, right)| left != right)
                .count();
            let untouched = outputs[1][index]
                .chunks_exact(2)
                .filter(|bits| *bits == bf16_bits(-113.0).to_ne_bytes())
                .count();
            println!(
                "wide-general-rms-norm hidden={hidden} {name} elements={} differing={differing} untouched={untouched}",
                ROWS * hidden
            );
            assert_eq!(
                differing, 0,
                "hidden {hidden} {name} differs from the 256-thread kernel"
            );
            assert_eq!(
                untouched, 0,
                "hidden {hidden} {name} left elements unwritten"
            );
        }
        for buffer in [left, right, weight] {
            buffer.close()?;
        }
    }
    staging.close()?;
    stream.close()?;
    context.synchronize()?;
    context.close()?;
    Ok(())
}
