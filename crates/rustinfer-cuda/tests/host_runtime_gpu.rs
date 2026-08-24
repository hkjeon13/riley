use std::error::Error;

use rustinfer_cuda::{
    CudaContext, CudaDevice, CudaErrorKind, CudaErrorStage, CudaEvent, CudaKernel, CudaPendingFill,
    CudaRuntime, CudaStream, DeviceProperties,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

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
fn device_metadata_is_reported() -> Result<(), Box<dyn Error>> {
    assert_send::<CudaRuntime>();
    assert_sync::<CudaRuntime>();
    assert_send::<CudaDevice>();
    assert_sync::<CudaDevice>();
    assert_send::<DeviceProperties>();
    assert_sync::<DeviceProperties>();
    assert_send::<CudaContext>();
    assert_sync::<CudaContext>();
    assert_send::<CudaKernel>();
    assert_sync::<CudaKernel>();
    assert_send::<CudaStream>();
    assert_send::<CudaEvent>();
    assert_send::<CudaPendingFill<'static>>();

    let (runtime, device) = first_device()?;
    let properties = device.properties();
    assert!(!properties.name().is_empty());
    assert!(properties.total_memory_bytes() > 0);
    assert!(properties.compute_capability().0 > 0);
    assert!(properties.multiprocessor_count() > 0);
    assert!(properties.warp_size() > 0);
    assert!(properties.max_threads_per_block() > 0);
    assert!(properties.driver_version() > 0);
    assert!(properties.runtime_version() > 0);
    println!("{}", properties.benchmark_metadata_line());
    let null_error = runtime
        .diagnose_null_device_output()
        .expect_err("null C output pointer must fail");
    assert_eq!(null_error.kind(), CudaErrorKind::InvalidArgument);
    assert_eq!(null_error.stage(), CudaErrorStage::Validation);
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn invalid_device_is_rejected() -> Result<(), Box<dyn Error>> {
    let (runtime, _device) = first_device()?;
    let error = runtime
        .device(runtime.device_count())
        .expect_err("one-past-last device must fail");
    assert_eq!(error.kind(), CudaErrorKind::InvalidDevice);
    assert_eq!(error.stage(), CudaErrorStage::Validation);
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn two_stream_event_ordering_is_explicit() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let kernel = context.kernel();
    let mut producer = context.create_stream()?;
    let mut consumer = context.create_stream()?;
    let mut ready = context.create_event()?;

    let mut pending = kernel.launch_fill(&mut producer, 16_384, 3.5)?;
    pending.record_event(&mut ready)?;
    consumer.wait_event(&ready)?;
    consumer.synchronize()?;
    assert!(ready.query()?);
    let values = pending.finish()?;
    assert!(values.iter().all(|value| *value == 3.5));

    let second_context = device.create_context()?;
    let foreign_event = second_context.create_event()?;
    let mismatch = consumer
        .wait_event(&foreign_event)
        .expect_err("cross-owner event wait must fail");
    assert_eq!(mismatch.kind(), CudaErrorKind::InvalidState);
    foreign_event.close()?;
    close_context(second_context)?;

    ready.close()?;
    producer.close()?;
    consumer.close()?;
    drop(kernel);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn async_fill_is_correct_after_sync() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let kernel = context.kernel();
    let mut stream = context.create_stream()?;
    let values = kernel.launch_fill(&mut stream, 65_537, -7.25)?.finish()?;
    assert_eq!(values.len(), 65_537);
    assert!(values.iter().all(|value| *value == -7.25));
    let overflow = match kernel.launch_fill(&mut stream, u64::MAX, 0.0) {
        Ok(pending) => {
            drop(pending);
            panic!("overflowing element count must fail before allocation");
        }
        Err(error) => error,
    };
    assert_eq!(overflow.kind(), CudaErrorKind::OutOfRange);
    stream.close()?;
    drop(kernel);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn invalid_launch_reports_launch_stage() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let kernel = context.kernel();
    let mut stream = context.create_stream()?;
    let error = kernel
        .diagnose_invalid_launch(&mut stream)
        .expect_err("zero-grid diagnostic launch must fail");
    assert_eq!(error.stage(), CudaErrorStage::Launch);
    assert!(matches!(
        error.kind(),
        CudaErrorKind::InvalidArgument | CudaErrorKind::Runtime
    ));

    let values = kernel.launch_fill(&mut stream, 1_024, 2.0)?.finish()?;
    assert!(values.iter().all(|value| *value == 2.0));
    stream.close()?;
    drop(kernel);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn events_report_positive_elapsed_time() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let kernel = context.kernel();
    let mut stream = context.create_stream()?;
    let mut start = context.create_event()?;
    let mut end = context.create_event()?;

    start.record(&mut stream)?;
    let mut pending = kernel.launch_fill(&mut stream, 16 * 1024 * 1024, 1.0)?;
    pending.record_event(&mut end)?;
    end.synchronize()?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    assert!(elapsed_ms > 0.0, "elapsed CUDA event time was {elapsed_ms}");
    let values = pending.finish()?;
    assert_eq!(values.first(), Some(&1.0));
    assert_eq!(values.last(), Some(&1.0));

    start.close()?;
    end.close()?;
    stream.close()?;
    drop(kernel);
    close_context(context)
}

#[test]
#[ignore = "remote GPU"]
fn repeated_create_drop_has_no_resource_leak() -> Result<(), Box<dyn Error>> {
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let kernel = context.kernel();

    let mut warmup_stream = context.create_stream()?;
    kernel
        .launch_fill(&mut warmup_stream, 4_096, 0.5)?
        .finish()?;
    warmup_stream.close()?;
    context.synchronize()?;
    let (before_free_bytes, total_bytes) = context.memory_info()?;

    let iterations = std::env::var("RUSTINFER_CUDA_LEAK_ITERATIONS")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(128);
    assert!((32..=4_096).contains(&iterations));
    for _ in 0..iterations {
        let iteration_context = device.create_context()?;
        let iteration_kernel = iteration_context.kernel();
        let mut stream = iteration_context.create_stream()?;
        let mut event = iteration_context.create_event()?;
        let values = iteration_kernel
            .launch_fill(&mut stream, 4_096, 0.25)?
            .finish()?;
        assert_eq!(values[0], 0.25);
        event.record(&mut stream)?;
        event.synchronize()?;
        event.close()?;
        stream.close()?;
        drop(iteration_kernel);
        close_context(iteration_context)?;
    }
    context.synchronize()?;
    let (after_free_bytes, after_total_bytes) = context.memory_info()?;
    println!(
        "rustinfer-cuda-leak-smoke iterations={iterations} before_free_bytes={before_free_bytes} after_free_bytes={after_free_bytes}"
    );
    assert_eq!(total_bytes, after_total_bytes);
    const TOLERANCE_BYTES: u64 = 64 * 1024 * 1024;
    assert!(
        after_free_bytes.saturating_add(TOLERANCE_BYTES) >= before_free_bytes,
        "free device memory dropped by more than {TOLERANCE_BYTES} bytes"
    );
    drop(kernel);
    let live_child = context.create_stream()?;
    let error = context
        .close()
        .expect_err("context close must reject a live child stream");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    drop(live_child);
    Ok(())
}
