use std::error::Error;

use rustinfer_cuda::{
    CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaRuntime, NvidiaDeviceSnapshot,
    NvidiaEnvironmentSnapshot, NvidiaPersistenceMode, diagnose_null_nvidia_environment_output,
    probe_nvidia_environment,
};

fn assert_send_sync<T: Send + Sync>() {}

fn validate_snapshot(snapshot: &NvidiaEnvironmentSnapshot) -> Result<(), Box<dyn Error>> {
    assert!(!snapshot.driver_version().is_empty());
    assert!(!snapshot.cuda_driver_api_version().is_empty());
    assert!(snapshot.cuda_driver_api_version_raw() > 0);
    assert!(snapshot.gpu_count() > 0);
    assert_eq!(
        usize::try_from(snapshot.gpu_count())?,
        snapshot.devices().len()
    );

    let mut process_count = 0_u32;
    for (expected_index, device) in (0_u32..).zip(snapshot.devices()) {
        assert_eq!(device.index(), expected_index);
        assert!(!device.name().is_empty());
        assert!(device.total_memory_bytes() > 0);
        assert!(device.used_memory_bytes() <= device.total_memory_bytes());
        assert!((1..150).contains(&device.temperature_c()));
        assert!(matches!(
            device.persistence_mode(),
            NvidiaPersistenceMode::Disabled | NvidiaPersistenceMode::Enabled
        ));
        assert!(device.power_limit_milliwatts() > 0);
        if let Some(clock_mhz) = device.application_graphics_clock_mhz() {
            assert!(clock_mhz > 0);
        }
        if let Some(clock_mhz) = device.application_memory_clock_mhz() {
            assert!(clock_mhz > 0);
        }
        process_count = process_count
            .checked_add(device.compute_process_count())
            .ok_or("per-device NVML compute-process counts overflowed the public aggregate")?;
    }
    assert_eq!(snapshot.compute_process_count(), process_count);
    Ok(())
}

#[test]
#[ignore = "requires the remote NVIDIA GPU on server-4096"]
fn environment_probe_succeeds_before_cuda_runtime_initialization() -> Result<(), Box<dyn Error>> {
    assert_send_sync::<NvidiaEnvironmentSnapshot>();
    assert_send_sync::<NvidiaDeviceSnapshot>();
    assert_send_sync::<NvidiaPersistenceMode>();

    // The ordering is intentional: the NVML preflight must not depend on a
    // prior CUDA Runtime initialization or retained primary context.
    let snapshot = probe_nvidia_environment()?;
    validate_snapshot(&snapshot)?;

    let runtime = CudaRuntime::initialize()?;
    assert_eq!(runtime.device_count(), snapshot.gpu_count());
    for device_snapshot in snapshot.devices() {
        let cuda_device = runtime.device(device_snapshot.index())?;
        assert_eq!(cuda_device.properties().name(), device_snapshot.name());
        // NVML reports physical framebuffer total while the CUDA Driver API
        // may exclude firmware/driver-reserved memory from allocatable total.
        assert!(
            cuda_device.properties().total_memory_bytes() <= device_snapshot.total_memory_bytes()
        );
        assert_eq!(
            cuda_device.properties().driver_version(),
            snapshot.cuda_driver_api_version_raw()
        );
    }

    println!(
        "rustinfer-cuda-nvml-environment driver_version={} cuda_driver_api_version={} gpu_count={} compute_process_count={}",
        snapshot.driver_version(),
        snapshot.cuda_driver_api_version(),
        snapshot.gpu_count(),
        snapshot.compute_process_count()
    );
    for device in snapshot.devices() {
        println!(
            "rustinfer-cuda-nvml-device index={} name={} total_memory_bytes={} used_memory_bytes={} temperature_c={} persistence_mode={:?} power_limit_milliwatts={} application_graphics_clock_mhz={:?} application_memory_clock_mhz={:?} compute_process_count={}",
            device.index(),
            device.name(),
            device.total_memory_bytes(),
            device.used_memory_bytes(),
            device.temperature_c(),
            device.persistence_mode(),
            device.power_limit_milliwatts(),
            device.application_graphics_clock_mhz(),
            device.application_memory_clock_mhz(),
            device.compute_process_count()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires the remote NVIDIA GPU on server-4096"]
fn validation_error_does_not_poison_repeated_probe() -> Result<(), Box<dyn Error>> {
    let before = probe_nvidia_environment()?;
    let error = diagnose_null_nvidia_environment_output()
        .expect_err("null environment output must fail before NVML initialization");
    assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
    assert_eq!(error.domain(), CudaErrorDomain::Validation);
    assert_eq!(error.stage(), CudaErrorStage::Validation);

    let after = probe_nvidia_environment()?;
    assert_eq!(before.driver_version(), after.driver_version());
    assert_eq!(
        before.cuda_driver_api_version_raw(),
        after.cuda_driver_api_version_raw()
    );
    assert_eq!(before.gpu_count(), after.gpu_count());
    validate_snapshot(&after)
}
