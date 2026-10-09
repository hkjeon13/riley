use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream, QkvSplitBiasParams,
    RowBiasAddInPlaceParams, qkv_split_bias, row_bias_add_in_place,
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

fn close_context(context: CudaContext) -> TestResult {
    context.synchronize()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let is_nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
    let rounded = if is_nan {
        0x7fff
    } else {
        let tie = (bits >> 16) & 1;
        bits.wrapping_add(0x7fff + tie) >> 16
    };
    u16::try_from(rounded).unwrap_or(0x7fff)
}

fn bf16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn decode_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn reference_row_bias(matrix: &[u16], bias: &[u16], column_count: usize) -> Vec<u16> {
    matrix
        .iter()
        .enumerate()
        .map(|(index, &matrix_bits)| {
            let matrix_value = f32::from_bits(u32::from(matrix_bits) << 16);
            let bias_value = f32::from_bits(u32::from(bias[index % column_count]) << 16);
            f32_to_bf16_bits(matrix_value + bias_value)
        })
        .collect()
}

fn upload(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    bytes: &[u8],
) -> TestResult<CudaDeviceBuffer> {
    let byte_len = u64::try_from(bytes.len())?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;
    if !bytes.is_empty() {
        buffer.upload_from_slice(0, bytes, staging, stream)?;
    }
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

fn run_case(
    context: &CudaContext,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    row_count: u64,
    column_count: u64,
    matrix_bits: &[u16],
    bias_bits: &[u16],
) -> TestResult {
    let expected = if matrix_bits.is_empty() {
        Vec::new()
    } else {
        reference_row_bias(matrix_bits, bias_bits, usize::try_from(column_count)?)
    };
    let mut matrix = upload(context, stream, staging, &bf16_bytes(matrix_bits))?;
    let bias = upload(context, stream, staging, &bf16_bytes(bias_bits))?;
    {
        let matrix_len = matrix.byte_len();
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(&mut matrix, CudaDType::BF16, 0, matrix_len)?,
            bias: CudaBufferSpan::new(&bias, CudaDType::BF16, 0, bias.byte_len())?,
            row_count,
            column_count,
        };
        row_bias_add_in_place(&mut params, stream)?;
    }
    assert_eq!(
        decode_bf16_bits(&download(context, stream, &mut matrix)?),
        expected,
        "row-bias result mismatch for shape [{row_count}, {column_count}]"
    );
    matrix.close()?;
    bias.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn bf16_row_bias_is_exact_in_place_for_odd_and_boundary_shapes() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4_096)?;
    let baseline = context.allocation_stats()?;

    run_case(
        &context,
        &mut stream,
        &mut staging,
        1,
        2,
        &[0x3f80, 0x3f81],
        &[0x3b80, 0x3b80],
    )?;
    run_case(
        &context,
        &mut stream,
        &mut staging,
        1,
        4,
        &[0x7f80, 0xff80, 0x7fc1, 0x3f80],
        &[0x3f80, 0xbf80, 0, 0x7f80],
    )?;

    let odd_matrix: Vec<u16> = (-7_i16..8)
        .map(|value| f32_to_bf16_bits(f32::from(value)))
        .collect();
    let odd_bias: Vec<u16> = [0.5_f32, -1.0, 2.0, 0.25, -0.125]
        .into_iter()
        .map(f32_to_bf16_bits)
        .collect();
    run_case(
        &context,
        &mut stream,
        &mut staging,
        3,
        5,
        &odd_matrix,
        &odd_bias,
    )?;

    let boundary_matrix: Vec<u16> = (0_u16..514)
        .map(|index| f32_to_bf16_bits(f32::from(index % 31) - 15.0))
        .collect();
    let boundary_bias: Vec<u16> = (0_u16..257)
        .map(|index| f32_to_bf16_bits(f32::from(index % 13) * 0.125 - 0.75))
        .collect();
    run_case(
        &context,
        &mut stream,
        &mut staging,
        2,
        257,
        &boundary_matrix,
        &boundary_bias,
    )?;
    run_case(&context, &mut stream, &mut staging, 0, 3, &[], &[0, 0, 0])?;
    assert_eq!(context.allocation_stats()?, baseline);

    staging.close()?;
    stream.close()?;
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn bf16_row_bias_rejects_overflow_capacity_and_foreign_context_before_write() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(64)?;
    let original = bf16_bytes(&[0x3f80, 0x4000]);
    let mut matrix = upload(&context, &mut stream, &mut staging, &original)?;
    let short_bias = upload(&context, &mut stream, &mut staging, &bf16_bytes(&[0]))?;
    {
        let matrix_len = matrix.byte_len();
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(&mut matrix, CudaDType::BF16, 0, matrix_len)?,
            bias: CudaBufferSpan::new(&short_bias, CudaDType::BF16, 0, short_bias.byte_len())?,
            row_count: 1,
            column_count: 2,
        };
        assert_eq!(
            row_bias_add_in_place(&mut params, &mut stream)
                .expect_err("short bias must fail")
                .kind(),
            CudaErrorKind::OutOfRange
        );
    }
    assert_eq!(download(&context, &mut stream, &mut matrix)?, original);
    short_bias.close()?;

    let full_bias = upload(&context, &mut stream, &mut staging, &bf16_bytes(&[0, 0]))?;
    {
        let matrix_len = matrix.byte_len();
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(&mut matrix, CudaDType::BF16, 0, matrix_len)?,
            bias: CudaBufferSpan::new(&full_bias, CudaDType::BF16, 0, full_bias.byte_len())?,
            row_count: u64::MAX,
            column_count: 2,
        };
        assert_eq!(
            row_bias_add_in_place(&mut params, &mut stream)
                .expect_err("shape overflow must fail")
                .kind(),
            CudaErrorKind::OutOfRange
        );
    }
    assert_eq!(download(&context, &mut stream, &mut matrix)?, original);
    full_bias.close()?;

    let foreign_context = device.create_context()?;
    let mut foreign_stream = foreign_context.create_stream()?;
    let mut foreign_staging = foreign_context.allocate_pinned_host_buffer(16)?;
    let foreign_bias = upload(
        &foreign_context,
        &mut foreign_stream,
        &mut foreign_staging,
        &bf16_bytes(&[0, 0]),
    )?;
    {
        let matrix_len = matrix.byte_len();
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(&mut matrix, CudaDType::BF16, 0, matrix_len)?,
            bias: CudaBufferSpan::new(&foreign_bias, CudaDType::BF16, 0, foreign_bias.byte_len())?,
            row_count: 1,
            column_count: 2,
        };
        assert_eq!(
            row_bias_add_in_place(&mut params, &mut stream)
                .expect_err("foreign-context bias must fail")
                .kind(),
            CudaErrorKind::InvalidState
        );
    }
    assert_eq!(download(&context, &mut stream, &mut matrix)?, original);

    foreign_bias.close()?;
    foreign_staging.close()?;
    foreign_stream.close()?;
    close_context(foreign_context)?;
    matrix.close()?;
    staging.close()?;
    stream.close()?;
    close_context(context)
}

/// The merged split must equal, byte for byte, slicing the merged matrix on
/// the host and running the established row-bias add on each slice.
#[test]
#[ignore = "remote GPU"]
fn merged_qkv_split_bias_matches_separate_row_bias_adds() -> TestResult {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 16)?;
    for (rows, query_width, key_value_width) in
        [(1_usize, 1_usize, 1_usize), (5, 7, 3), (33, 64, 8)]
    {
        let merged_width = query_width + 2 * key_value_width;
        let mut state = 0x9e37_79b9_u32 ^ u32::try_from(rows * 131 + merged_width)?;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Finite BF16 patterns of mixed sign and magnitude.
            let bits = u16::try_from(state >> 16)?;
            Ok::<u16, Box<dyn Error>>(if bits & 0x7f80 == 0x7f80 {
                bits & 0xbfff
            } else {
                bits
            })
        };
        let mut merged_bits = Vec::with_capacity(rows * merged_width);
        for _ in 0..rows * merged_width {
            merged_bits.push(next()?);
        }
        // Special values: signed zeros, infinities, and a NaN in each output.
        if rows >= 5 {
            merged_bits[0] = 0x8000;
            merged_bits[merged_width + query_width] = 0x7f80;
            merged_bits[2 * merged_width + query_width + key_value_width] = 0xff80;
            merged_bits[3 * merged_width + 1] = 0x7fc0;
            merged_bits[4 * merged_width + merged_width - 1] = 0x0000;
        }
        let mut bias = |width: usize| -> TestResult<Vec<u16>> {
            let mut values = Vec::with_capacity(width);
            for _ in 0..width {
                values.push(next()?);
            }
            Ok(values)
        };
        let query_bias_bits = bias(query_width)?;
        let key_bias_bits = bias(key_value_width)?;
        let value_bias_bits = bias(key_value_width)?;
        let slice = |begin: usize, width: usize| -> Vec<u16> {
            merged_bits
                .chunks_exact(merged_width)
                .flat_map(|row| row[begin..begin + width].iter().copied())
                .collect()
        };
        let parts = [
            (slice(0, query_width), &query_bias_bits, query_width),
            (
                slice(query_width, key_value_width),
                &key_bias_bits,
                key_value_width,
            ),
            (
                slice(query_width + key_value_width, key_value_width),
                &value_bias_bits,
                key_value_width,
            ),
        ];

        // Established path: three separate in-place row-bias adds.
        let mut separate = Vec::new();
        for (matrix_bits, bias_bits, width) in &parts {
            assert_eq!(
                reference_row_bias(matrix_bits, bias_bits, *width).len(),
                rows * width
            );
            let mut matrix = upload(
                &context,
                &mut stream,
                &mut staging,
                &bf16_bytes(matrix_bits),
            )?;
            let bias = upload(&context, &mut stream, &mut staging, &bf16_bytes(bias_bits))?;
            {
                let matrix_len = matrix.byte_len();
                let mut params = RowBiasAddInPlaceParams {
                    matrix: CudaBufferSpanMut::new(&mut matrix, CudaDType::BF16, 0, matrix_len)?,
                    bias: CudaBufferSpan::new(&bias, CudaDType::BF16, 0, bias.byte_len())?,
                    row_count: u64::try_from(rows)?,
                    column_count: u64::try_from(*width)?,
                };
                row_bias_add_in_place(&mut params, &mut stream)?;
            }
            separate.push(download(&context, &mut stream, &mut matrix)?);
            matrix.close()?;
            bias.close()?;
        }

        let merged = upload(
            &context,
            &mut stream,
            &mut staging,
            &bf16_bytes(&merged_bits),
        )?;
        let query_bias = upload(
            &context,
            &mut stream,
            &mut staging,
            &bf16_bytes(&query_bias_bits),
        )?;
        let key_bias = upload(
            &context,
            &mut stream,
            &mut staging,
            &bf16_bytes(&key_bias_bits),
        )?;
        let value_bias = upload(
            &context,
            &mut stream,
            &mut staging,
            &bf16_bytes(&value_bias_bits),
        )?;
        let sentinel = |elements: usize| bf16_bytes(&vec![0x5a5a_u16; elements]);
        let mut query = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * query_width),
        )?;
        let mut key = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * key_value_width),
        )?;
        let mut value = upload(
            &context,
            &mut stream,
            &mut staging,
            &sentinel(rows * key_value_width),
        )?;
        {
            let (query_len, key_len, value_len) =
                (query.byte_len(), key.byte_len(), value.byte_len());
            let mut params = QkvSplitBiasParams {
                merged: CudaBufferSpan::new(&merged, CudaDType::BF16, 0, merged.byte_len())?,
                query_bias: CudaBufferSpan::new(
                    &query_bias,
                    CudaDType::BF16,
                    0,
                    query_bias.byte_len(),
                )?,
                key_bias: CudaBufferSpan::new(&key_bias, CudaDType::BF16, 0, key_bias.byte_len())?,
                value_bias: CudaBufferSpan::new(
                    &value_bias,
                    CudaDType::BF16,
                    0,
                    value_bias.byte_len(),
                )?,
                query: CudaBufferSpanMut::new(&mut query, CudaDType::BF16, 0, query_len)?,
                key: CudaBufferSpanMut::new(&mut key, CudaDType::BF16, 0, key_len)?,
                value: CudaBufferSpanMut::new(&mut value, CudaDType::BF16, 0, value_len)?,
                row_count: u64::try_from(rows)?,
                query_width: u64::try_from(query_width)?,
                key_value_width: u64::try_from(key_value_width)?,
            };
            qkv_split_bias(&mut params, &mut stream)?;
        }
        let split = [
            download(&context, &mut stream, &mut query)?,
            download(&context, &mut stream, &mut key)?,
            download(&context, &mut stream, &mut value)?,
        ];
        for (index, name) in ["query", "key", "value"].iter().enumerate() {
            assert!(
                split[index] == separate[index],
                "merged split {name} bytes differ from the separate row-bias add for rows={rows} widths=({query_width}, {key_value_width})"
            );
        }

        // Zero-width outputs are rejected before any write.
        {
            let (query_len, key_len, value_len) =
                (query.byte_len(), key.byte_len(), value.byte_len());
            let mut params = QkvSplitBiasParams {
                merged: CudaBufferSpan::new(&merged, CudaDType::BF16, 0, merged.byte_len())?,
                query_bias: CudaBufferSpan::new(
                    &query_bias,
                    CudaDType::BF16,
                    0,
                    query_bias.byte_len(),
                )?,
                key_bias: CudaBufferSpan::new(&key_bias, CudaDType::BF16, 0, key_bias.byte_len())?,
                value_bias: CudaBufferSpan::new(
                    &value_bias,
                    CudaDType::BF16,
                    0,
                    value_bias.byte_len(),
                )?,
                query: CudaBufferSpanMut::new(&mut query, CudaDType::BF16, 0, query_len)?,
                key: CudaBufferSpanMut::new(&mut key, CudaDType::BF16, 0, key_len)?,
                value: CudaBufferSpanMut::new(&mut value, CudaDType::BF16, 0, value_len)?,
                row_count: u64::try_from(rows)?,
                query_width: u64::try_from(query_width)?,
                key_value_width: 0,
            };
            let error = qkv_split_bias(&mut params, &mut stream)
                .expect_err("a zero key/value width must be rejected");
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }
        assert_eq!(download(&context, &mut stream, &mut query)?, split[0]);

        for buffer in [merged, query_bias, key_bias, value_bias, query, key, value] {
            buffer.close()?;
        }
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}
