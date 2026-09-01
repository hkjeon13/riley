use std::error::Error;

use riley_cuda::{
    CudaContext, CudaDevice, CudaErrorKind, CudaGraphCaptureMode, CudaResult, CudaRuntime,
    CudaStream,
};

fn all_f32_bits_equal(values: &[f32], expected: f32) -> bool {
    let expected_bits = expected.to_bits();
    values.iter().all(|value| value.to_bits() == expected_bits)
}

fn first_device() -> Result<CudaDevice, Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    Ok(runtime.device(0)?)
}

fn close_context(context: CudaContext) -> Result<(), Box<dyn Error>> {
    context.synchronize()?;
    context.close()?;
    Ok(())
}

fn assert_invalid_state<T>(result: CudaResult<T>, operation: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{operation} unexpectedly succeeded during capture"));
    assert_eq!(
        error.kind(),
        CudaErrorKind::InvalidState,
        "{operation} must reject before issuing a prohibited CUDA call"
    );
}

fn assert_eager_fill_after_recovery(
    context: &CudaContext,
    stream: &mut CudaStream,
    value: f32,
) -> Result<(), Box<dyn Error>> {
    let kernel = context.kernel();
    let values = kernel.launch_fill(stream, 4_096, value)?.finish()?;
    assert!(all_f32_bits_equal(&values, value));
    drop(kernel);
    Ok(())
}

fn download_f32_buffer(
    context: &CudaContext,
    buffer: &mut riley_cuda::CudaDeviceBuffer,
    stream: &mut CudaStream,
    element_count: u64,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let byte_len = element_count
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 capture output byte length overflow")?;
    let host_len = usize::try_from(byte_len)?;
    let mut staging = context.allocate_pinned_host_buffer(byte_len)?;
    let mut bytes = vec![0_u8; host_len];
    buffer.download_to_slice(0, &mut bytes, &mut staging, stream)?;
    staging.close()?;

    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(values)
}

#[test]
#[ignore = "remote GPU"]
fn fixed_buffer_fill_graph_replays_exactly_and_releases_every_lease() -> Result<(), Box<dyn Error>>
{
    const ELEMENT_COUNT: u64 = 4_096;
    const REPLAYS: usize = 1_000;
    const FINAL_VALUE: f32 = -7.25;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut capture_stream = context.create_stream()?;
    let mut download_stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 capture allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    let captured = {
        let mut capture = capture_stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        // The final node makes every replay's expected output unambiguous.
        capture.enqueue_fill(1.5)?;
        capture.enqueue_fill(-3.0)?;
        capture.enqueue_fill(FINAL_VALUE)?;
        capture.end()?
    };
    let mut exec = captured.instantiate()?;
    for _ in 0..REPLAYS {
        exec.launch()?.finish()?;
    }

    // Safe Rust keeps the capture stream and buffer mutably borrowed until
    // this close succeeds. The native graph exec retains matching raw leases
    // too, so only a known close may return them for this independent D2H.
    exec.close()?;
    let values = download_f32_buffer(&context, &mut buffer, &mut download_stream, ELEMENT_COUNT)?;
    assert_eq!(values.len(), usize::try_from(ELEMENT_COUNT)?);
    assert!(all_f32_bits_equal(&values, FINAL_VALUE));

    buffer.close()?;
    capture_stream.close()?;
    download_stream.close()?;
    assert_eq!(
        context.allocation_stats()?,
        allocation_baseline,
        "successful graph/exec close plus D2H staging close must restore the exact allocation baseline"
    );
    close_context(context)?;
    println!(
        "c05-5-fixed-fill-replay replays={REPLAYS} elements={ELEMENT_COUNT} final_value={FINAL_VALUE} status=passed"
    );
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn captured_graph_close_releases_stream_and_buffer_for_reuse() -> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("captured graph close allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    let captured = {
        let mut capture = stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        capture.enqueue_fill(4.0)?;
        capture.end()?
    };
    // This exercises graph destruction without instantiation. Success must
    // return both the native graph leases and the Rust mutable borrows.
    captured.close()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 4.0)?;

    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-captured-graph-close-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn fixed_buffer_fill_capture_abort_releases_stream_and_buffer_for_reuse()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("f32 abort-recovery allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    for _ in 0..2 {
        stream
            .begin_graph_fill_capture(
                &mut buffer,
                ELEMENT_COUNT,
                CudaGraphCaptureMode::ThreadLocal,
            )?
            .abort()?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, 2.5)?;

    // The buffer comes back from the capture borrow only after native abort
    // has released its active-use lease; explicit close is the observable
    // proof that capture did not strand it busy.
    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-fixed-fill-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn zero_enqueue_fill_end_rejects_before_native_end_and_abort_restores_reuse()
-> Result<(), Box<dyn Error>> {
    const ELEMENT_COUNT: u64 = 256;

    let device = first_device()?;
    let context = device.create_context()?;
    let allocation_baseline = context.allocation_stats()?;
    assert!(allocation_baseline.is_zero());
    let mut stream = context.create_stream()?;
    let byte_len = ELEMENT_COUNT
        .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
        .ok_or("zero-enqueue capture allocation byte length overflow")?;
    let mut buffer = context.allocate_device_buffer(byte_len)?;

    // `end` rejects the missing admitted node in safe Rust, before calling
    // native end. Its consuming error path drops the still-active capture and
    // performs the one-shot abort/recovery instead of leaving either lease
    // stranded.
    let error = {
        let capture = stream.begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        match capture.end() {
            Ok(_) => panic!("zero-enqueue graph capture unexpectedly ended"),
            Err(error) => error,
        }
    };
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);

    // Both the same preallocated buffer and stream must be immediately
    // admissible for a fresh capture after that automatic abort.
    stream
        .begin_graph_fill_capture(
            &mut buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?
        .abort()?;
    assert_eager_fill_after_recovery(&context, &mut stream, -1.5)?;

    buffer.close()?;
    stream.close()?;
    assert_eq!(context.allocation_stats()?, allocation_baseline);
    close_context(context)?;
    println!("c05-5-zero-enqueue-end-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn explicit_abort_restores_stream_for_eager_work() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 3.25)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-explicit-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn drop_abort_restores_stream_for_eager_work() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    {
        let _capture = stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, -7.5)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-drop-abort-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn repeated_abort_releases_stream_and_context_leases() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;

    for _ in 0..8 {
        stream
            .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
            .abort()?;
    }
    assert_eager_fill_after_recovery(&context, &mut stream, 0.125)?;

    stream.close()?;
    close_context(context)?;
    println!("c05-4-repeated-abort-recovery iterations=8 status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn pending_fills_block_capture_until_both_complete() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut first_pending_stream = context.create_stream()?;
    let mut second_pending_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let kernel = context.kernel();

    // Every pending fill reserves the same primary-context admission domain
    // before enqueue. Capture must stay out until all of their native buffers
    // are consumed, not merely until the first stream reports completion.
    let first = kernel.launch_fill(&mut first_pending_stream, 4_096, 6.75)?;
    let second = kernel.launch_fill(&mut second_pending_stream, 4_096, -6.75)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while two pending smoke fills are live",
    );
    assert!(all_f32_bits_equal(&first.finish()?, 6.75));
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while one pending smoke fill remains live",
    );
    assert!(all_f32_bits_equal(&second.finish()?, -6.75));
    drop(kernel);

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    first_pending_stream.close()?;
    second_pending_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-pending-fill-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn same_context_resource_drops_and_closes_are_deferred_until_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut captured_stream = context.create_stream()?;
    let spare_stream = context.create_stream()?;
    let event = context.create_event()?;
    let device_buffer = context.allocate_device_buffer(256)?;
    let pinned_buffer = context.allocate_pinned_host_buffer(256)?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;

    // The owning capture stream stays exclusively borrowed, but independent
    // native owners on the same thread must transfer their close into abort
    // cleanup rather than issue CUDA destruction during capture. Cover both
    // implicit Drop and the explicit close API.
    drop(device_buffer);
    drop(pinned_buffer);
    event.close()?;
    spare_stream.close()?;

    capture.abort()?;
    assert!(
        context.allocation_stats()?.is_zero(),
        "abort must drain every deferred same-context allocation"
    );

    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-deferred-same-context-resources status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn foreign_context_resource_drops_and_closes_survive_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let capture_context = device.create_context()?;
    let mut captured_stream = capture_context.create_stream()?;
    let foreign_context = device.create_context()?;
    let foreign_stream = foreign_context.create_stream()?;
    let foreign_event = foreign_context.create_event()?;
    let foreign_device_buffer = foreign_context.allocate_device_buffer(256)?;
    let foreign_pinned_buffer = foreign_context.allocate_pinned_host_buffer(256)?;

    // Leave the resources as the only Rust owners of their foreign context.
    // Their destruction during capture must retain that Rust lease until the
    // native abort callback has destroyed the queued native owners.
    drop(foreign_context);
    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    drop(foreign_device_buffer);
    foreign_pinned_buffer.close()?;
    foreign_event.close()?;
    foreign_stream.close()?;
    capture.abort()?;

    captured_stream.close()?;
    close_context(capture_context)?;

    // A fresh primary-context lease proves the foreign-owner cleanup did not
    // leave the CUDA runtime in a capture-poisoned or child-leaked state.
    let recovery_context = device.create_context()?;
    let mut recovery_stream = recovery_context.create_stream()?;
    assert_eager_fill_after_recovery(&recovery_context, &mut recovery_stream, 1.75)?;
    recovery_stream.close()?;
    close_context(recovery_context)?;
    println!("c05-4-deferred-foreign-context-resources status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn bare_foreign_context_close_is_deferred_until_abort() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let capture_context = device.create_context()?;
    let mut captured_stream = capture_context.create_stream()?;
    let foreign_context = device.create_context()?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    foreign_context.close()?;
    capture.abort()?;

    captured_stream.close()?;
    close_context(capture_context)?;

    let recovery_context = device.create_context()?;
    let mut recovery_stream = recovery_context.create_stream()?;
    assert_eager_fill_after_recovery(&recovery_context, &mut recovery_stream, -1.75)?;
    recovery_stream.close()?;
    close_context(recovery_context)?;
    println!("c05-4-deferred-foreign-context-close status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn pending_copy_blocks_capture_until_consumed() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut copy_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let mut device_buffer = context.allocate_device_buffer(256)?;
    let mut pinned_buffer = context.allocate_pinned_host_buffer(256)?;
    pinned_buffer.write(0, &[0x5a; 256])?;

    let pending =
        device_buffer.copy_from_pinned_async(0, &mut pinned_buffer, 0, 256, &mut copy_stream)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while a pending pinned-host copy is live",
    );
    pending.synchronize()?;

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    device_buffer.close()?;
    pinned_buffer.close()?;
    copy_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-pending-copy-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn zero_element_pending_fill_blocks_capture_until_consumed() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    let mut pending_stream = context.create_stream()?;
    let mut captured_stream = context.create_stream()?;
    let kernel = context.kernel();

    let pending = kernel.launch_fill(&mut pending_stream, 0, 0.0)?;
    assert_invalid_state(
        captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while a zero-element pending fill is live",
    );
    assert!(
        pending.finish()?.is_empty(),
        "zero-element fill must preserve its empty output contract"
    );
    drop(kernel);

    captured_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;
    pending_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-zero-element-fill-admission-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn same_thread_capture_blocks_context_and_foreign_stream_cuda_work() -> Result<(), Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote GPU runner has no CUDA device"
    );
    let device = runtime.device(0)?;
    let context = device.create_context()?;
    let mut captured_stream = context.create_stream()?;
    let mut foreign_stream = context.create_stream()?;
    let mut third_stream = context.create_stream()?;

    // Nested batches remain supported across streams, but their thread-local
    // count must keep any third-stream capture out until every batch ends.
    let first_batch = foreign_stream.begin_command_batch()?;
    let second_batch = captured_stream.begin_command_batch()?;
    assert_invalid_state(
        third_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while two foreign stream command batches are active",
    );
    second_batch.finish()?;
    assert_invalid_state(
        third_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "capture begin while one foreign stream command batch remains active",
    );
    drop(first_batch);
    third_stream
        .begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?
        .abort()?;

    let capture = captured_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;

    // ThreadLocal capture must reject all potentially unsafe CUDA calls issued
    // from its owning host thread, even when they target a distinct stream.
    assert_invalid_state(CudaRuntime::initialize(), "runtime device-count query");
    assert_invalid_state(runtime.device(0), "device-properties query");
    assert_invalid_state(context.synchronize(), "context synchronize");
    assert_invalid_state(context.memory_info(), "context memory-info query");
    assert_invalid_state(context.create_stream(), "secondary stream create");
    assert_invalid_state(context.create_event(), "event create");
    assert_invalid_state(
        context.allocate_device_buffer(256),
        "device-buffer allocation",
    );
    assert_invalid_state(device.create_context(), "secondary context create");
    assert_invalid_state(
        foreign_stream.begin_command_batch(),
        "foreign stream command-batch begin",
    );
    assert_invalid_state(
        foreign_stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal),
        "foreign stream capture begin",
    );
    let kernel = context.kernel();
    assert_invalid_state(
        kernel.launch_fill(&mut foreign_stream, 256, 1.5),
        "foreign stream eager fill",
    );
    drop(kernel);

    capture.abort()?;
    assert_eager_fill_after_recovery(&context, &mut captured_stream, 4.5)?;
    assert_eager_fill_after_recovery(&context, &mut foreign_stream, -4.5)?;

    third_stream.close()?;
    foreign_stream.close()?;
    captured_stream.close()?;
    close_context(context)?;
    println!("c05-4-thread-local-gate-recovery status=passed");
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn cross_thread_context_controls_are_rejected_while_capturing() -> Result<(), Box<dyn Error>> {
    let device = first_device()?;
    let context = device.create_context()?;
    // A second ABI wrapper retains the same CUDA primary context. It proves
    // the gate is primary-context scoped instead of just another TLS check on
    // the stream-owning wrapper.
    let secondary_context = device.create_context()?;
    let mut stream = context.create_stream()?;

    let capture = stream.begin_graph_capture(CudaGraphCaptureMode::ThreadLocal)?;
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            // Stream creation remains independent across host threads, but a
            // diagnostic pending fill carries a future synchronization/free
            // obligation. Its admission must reject before CUDA enqueue while
            // the primary context is capturing.
            let mut worker_stream = secondary_context
                .create_stream()
                .expect("cross-thread independent stream creation must succeed");
            let kernel = secondary_context.kernel();
            assert_invalid_state(
                kernel.launch_fill(&mut worker_stream, 4_096, -2.25),
                "cross-thread pending smoke-fill launch",
            );
            drop(kernel);
            worker_stream
                .close()
                .expect("cross-thread independent stream close must succeed");

            assert_invalid_state(
                secondary_context.synchronize(),
                "cross-thread context synchronize",
            );
            assert_invalid_state(
                secondary_context.memory_info(),
                "cross-thread context memory-info query",
            );
            assert_invalid_state(
                device.create_context(),
                "cross-thread primary-context retain",
            );
        });
        worker
            .join()
            .expect("cross-thread context-control test worker must not panic");
    });

    capture.abort()?;
    let (_free_bytes, total_bytes) = secondary_context.memory_info()?;
    assert!(
        total_bytes > 0,
        "recovered primary context must be queryable"
    );
    secondary_context.synchronize()?;
    assert_eager_fill_after_recovery(&context, &mut stream, 8.5)?;
    let mut recovery_stream = secondary_context.create_stream()?;
    assert_eager_fill_after_recovery(&secondary_context, &mut recovery_stream, -2.25)?;
    recovery_stream.close()?;

    stream.close()?;
    secondary_context.close()?;
    close_context(context)?;
    println!("c05-4-cross-thread-context-gate-recovery status=passed");
    Ok(())
}
