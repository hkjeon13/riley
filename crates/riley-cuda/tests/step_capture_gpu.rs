use std::error::Error;

use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDevice, CudaDeviceBuffer,
    CudaErrorKind, CudaPinnedHostBuffer, CudaRuntime, CudaStream, ResidualAddParams,
    abort_step_capture, begin_step_capture, end_step_capture, launch_step_graph, residual_add,
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn upload_into(
    buffer: &mut CudaDeviceBuffer,
    stream: &mut CudaStream,
    staging: &mut CudaPinnedHostBuffer,
    values: &[f32],
) -> TestResult {
    buffer.upload_from_slice(0, &f32_bytes(values), staging, stream)?;
    Ok(())
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

/// Two chained residual adds: `sum = left + right`, `out = sum + right`.
fn enqueue<S: riley_cuda::CudaExecutionStream + ?Sized>(
    left: &CudaDeviceBuffer,
    right: &CudaDeviceBuffer,
    sum: &mut CudaDeviceBuffer,
    out: &mut CudaDeviceBuffer,
    elements: u64,
    stream: &mut S,
) -> TestResult {
    let bytes = elements * 4;
    residual_add(
        &mut ResidualAddParams {
            left: CudaBufferSpan::new(left, CudaDType::F32, 0, bytes)?,
            right: CudaBufferSpan::new(right, CudaDType::F32, 0, bytes)?,
            output: CudaBufferSpanMut::new(sum, CudaDType::F32, 0, bytes)?,
            element_count: elements,
        },
        stream,
    )?;
    residual_add(
        &mut ResidualAddParams {
            left: CudaBufferSpan::new(sum, CudaDType::F32, 0, bytes)?,
            right: CudaBufferSpan::new(right, CudaDType::F32, 0, bytes)?,
            output: CudaBufferSpanMut::new(out, CudaDType::F32, 0, bytes)?,
            element_count: elements,
        },
        stream,
    )?;
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn recorded_step_replays_with_new_inputs_and_matches_eager() -> TestResult {
    const ELEMENTS: usize = 1_031;
    let (_runtime, device) = first_device()?;
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(u64::try_from(ELEMENTS * 4)?)?;
    let bytes = u64::try_from(ELEMENTS * 4)?;
    let elements = u64::try_from(ELEMENTS)?;
    let mut left = context.allocate_device_buffer(bytes)?;
    let mut right = context.allocate_device_buffer(bytes)?;
    let mut sum = context.allocate_device_buffer(bytes)?;
    let mut out = context.allocate_device_buffer(bytes)?;
    let mut eager_sum = context.allocate_device_buffer(bytes)?;
    let mut eager_out = context.allocate_device_buffer(bytes)?;

    let first_left: Vec<f32> = (0..ELEMENTS)
        .map(|index| index as f32 * 0.25 - 7.0)
        .collect();
    let first_right: Vec<f32> = (0..ELEMENTS)
        .map(|index| 3.5 - index as f32 * 0.125)
        .collect();
    upload_into(&mut left, &mut stream, &mut staging, &first_left)?;
    upload_into(&mut right, &mut stream, &mut staging, &first_right)?;

    // Capture outside a command batch is rejected before any state changes.
    let error = begin_step_capture(&mut stream).expect_err("capture needs a command batch");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);

    // Record the step; nothing recorded runs until the graph is launched.
    upload_into(&mut out, &mut stream, &mut staging, &vec![-1.0; ELEMENTS])?;
    let graph = {
        let mut batch = stream.begin_command_batch()?;
        let graph = {
            let mut commands = batch.commands();
            begin_step_capture(&mut commands)?;
            enqueue(&left, &right, &mut sum, &mut out, elements, &mut commands)?;
            end_step_capture(&mut commands)?
        };
        batch.finish()?;
        graph
    };
    assert_eq!(
        download(&context, &mut stream, &mut out)?,
        f32_bytes(&vec![-1.0; ELEMENTS]),
        "recording a step must not execute it"
    );

    for round in 0..3 {
        let new_left: Vec<f32> = first_left
            .iter()
            .map(|value| value + round as f32)
            .collect();
        let new_right: Vec<f32> = first_right
            .iter()
            .map(|value| value * (round + 1) as f32)
            .collect();
        upload_into(&mut left, &mut stream, &mut staging, &new_left)?;
        upload_into(&mut right, &mut stream, &mut staging, &new_right)?;
        enqueue(
            &left,
            &right,
            &mut eager_sum,
            &mut eager_out,
            elements,
            &mut stream,
        )?;
        {
            let mut batch = stream.begin_command_batch()?;
            {
                let mut commands = batch.commands();
                launch_step_graph(&mut commands, &graph)?;
            }
            batch.finish()?;
        }
        assert_eq!(
            download(&context, &mut stream, &mut out)?,
            download(&context, &mut stream, &mut eager_out)?,
            "replay {round} differs from the eager result for the same inputs"
        );
    }

    // An aborted capture leaves the batch usable and records nothing.
    {
        let mut batch = stream.begin_command_batch()?;
        {
            let mut commands = batch.commands();
            begin_step_capture(&mut commands)?;
            enqueue(&left, &right, &mut sum, &mut out, elements, &mut commands)?;
            abort_step_capture(&mut commands)?;
            enqueue(&left, &right, &mut sum, &mut out, elements, &mut commands)?;
        }
        batch.finish()?;
    }
    assert_eq!(
        download(&context, &mut stream, &mut out)?,
        download(&context, &mut stream, &mut eager_out)?,
    );

    graph.close()?;
    for buffer in [left, right, sum, out, eager_sum, eager_out] {
        buffer.close()?;
    }
    staging.close()?;
    stream.close()?;
    close_context(context)
}
