use std::error::Error;

use riley_cuda::{
    CudaContext, CudaDevice, CudaDeviceBuffer, CudaPendingD2H, CudaPendingH2D,
    CudaPinnedHostBuffer, CudaRuntime,
};

fn assert_send<T: Send>() {}

fn first_device() -> Result<(CudaRuntime, CudaDevice), Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    Ok((runtime, device))
}

fn close_context(context: CudaContext) -> Result<(), Box<dyn Error>> {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn allocation_accounting_returns_to_zero() -> Result<(), Box<dyn Error>> {
    assert_send::<CudaDeviceBuffer>();
    assert_send::<CudaPinnedHostBuffer>();
    assert_send::<CudaPendingH2D<'static>>();
    assert_send::<CudaPendingD2H<'static>>();

    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    assert!(context.allocation_stats()?.is_zero());

    let device_buffer = context.allocate_device_buffer(4_096)?;
    let pinned_buffer = context.allocate_pinned_host_buffer(8_192)?;
    let live = context.allocation_stats()?;
    assert_eq!(live.device_live_bytes(), 4_096);
    assert_eq!(live.device_live_allocations(), 1);
    assert_eq!(live.pinned_host_live_bytes(), 8_192);
    assert_eq!(live.pinned_host_live_allocations(), 1);

    device_buffer.close()?;
    pinned_buffer.close()?;
    let closed = context.allocation_stats()?;
    // Libtest writes the test name before captured output without first ending
    // the line. Keep the evidence marker on its own exact line.
    println!(
        "\nriley-cuda-memory-accounting device_live_bytes={} device_live_allocations={} pinned_host_live_bytes={} pinned_host_live_allocations={}",
        closed.device_live_bytes(),
        closed.device_live_allocations(),
        closed.pinned_host_live_bytes(),
        closed.pinned_host_live_allocations()
    );
    assert!(closed.is_zero());
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn zero_byte_allocations_and_copies_are_logical_noops() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut device_buffer = context.allocate_device_buffer(0)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(0)?;

    let live = context.allocation_stats()?;
    assert_eq!(device_buffer.byte_len(), 0);
    assert_eq!(pinned_buffer.byte_len(), 0);
    assert_eq!(live.device_live_bytes(), 0);
    assert_eq!(live.device_live_allocations(), 1);
    assert_eq!(live.pinned_host_live_bytes(), 0);
    assert_eq!(live.pinned_host_live_allocations(), 1);

    assert!(
        device_buffer
            .copy_from_pinned_async(0, &mut pinned_buffer, 0, 0, &mut stream)?
            .query()?
    );
    device_buffer
        .copy_to_pinned_async(0, &mut pinned_buffer, 0, 0, &mut stream)?
        .synchronize()?;

    device_buffer.close()?;
    pinned_buffer.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn pinned_host_device_round_trip_is_exact() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let pattern: Vec<u8> = (0_u32..262_147)
        .map(|index| {
            u8::try_from(index.wrapping_mul(131) % 251).expect("modulo 251 always fits in one byte")
        })
        .collect();
    let byte_len = u64::try_from(pattern.len())?;
    let mut device_buffer = context.allocate_device_buffer(byte_len)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(byte_len)?;

    pinned_buffer.write(0, &pattern)?;
    device_buffer
        .copy_from_pinned_async(0, &mut pinned_buffer, 0, byte_len, &mut stream)?
        .synchronize()?;
    pinned_buffer.write(0, &vec![0; pattern.len()])?;
    device_buffer
        .copy_to_pinned_async(0, &mut pinned_buffer, 0, byte_len, &mut stream)?
        .synchronize()?;
    assert_eq!(pinned_buffer.to_vec()?, pattern);

    device_buffer.close()?;
    pinned_buffer.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn two_stream_copy_handoff_prevents_early_reuse() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut upload_stream = context.create_stream()?;
    let mut download_stream = context.create_stream()?;
    let pattern = b"explicit stream handoff keeps allocation ownership ordered";
    let byte_len = u64::try_from(pattern.len())?;
    let mut device_buffer = context.allocate_device_buffer(byte_len)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(byte_len)?;

    pinned_buffer.write(0, pattern)?;
    let upload = device_buffer.copy_from_pinned_async(
        0,
        &mut pinned_buffer,
        0,
        byte_len,
        &mut upload_stream,
    )?;
    // The pending value exclusively borrows upload_stream and both buffers;
    // the compile-fail API example proves close/reuse cannot occur here.
    upload.synchronize()?;

    pinned_buffer.write(0, &vec![0; pattern.len()])?;
    device_buffer
        .copy_to_pinned_async(0, &mut pinned_buffer, 0, byte_len, &mut download_stream)?
        .synchronize()?;
    assert_eq!(pinned_buffer.to_vec()?, pattern);

    device_buffer.close()?;
    pinned_buffer.close()?;
    upload_stream.close()?;
    download_stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn copy_ranges_and_context_ownership_are_validated() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let foreign_context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut device_buffer = context.allocate_device_buffer(16)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(16)?;
    let mut foreign_pinned = foreign_context.allocate_pinned_host_buffer(16)?;

    let range_error =
        match device_buffer.copy_from_pinned_async(15, &mut pinned_buffer, 0, 2, &mut stream) {
            Ok(pending) => {
                drop(pending);
                panic!("out-of-range copy must fail before submission");
            }
            Err(error) => error,
        };
    assert_eq!(range_error.kind(), riley_cuda::CudaErrorKind::OutOfRange);

    let ownership_error =
        match device_buffer.copy_from_pinned_async(0, &mut foreign_pinned, 0, 1, &mut stream) {
            Ok(pending) => {
                drop(pending);
                panic!("cross-context copy must fail before submission");
            }
            Err(error) => error,
        };
    assert_eq!(
        ownership_error.kind(),
        riley_cuda::CudaErrorKind::InvalidState
    );

    device_buffer.close()?;
    pinned_buffer.close()?;
    foreign_pinned.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    assert!(foreign_context.allocation_stats()?.is_zero());
    close_context(context)?;
    close_context(foreign_context)
}
