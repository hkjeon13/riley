use crate::error::{CudaError, CudaResult};

#[cfg(feature = "nvml")]
use crate::ffi;

/// Whether this build includes the opt-in in-process NVML probe.
///
/// This is intentionally separate from [`crate::CUDA_ENABLED`] so production
/// CUDA binaries do not acquire an NVML dynamic-library dependency.
pub const NVML_ENABLED: bool = cfg!(feature = "nvml");

/// Normalized NVIDIA persistence-mode state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NvidiaPersistenceMode {
    /// The driver may unload after its final client exits.
    Disabled,
    /// The driver remains loaded after its final client exits.
    Enabled,
}

/// Immutable telemetry for one physical device in an NVML snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvidiaDeviceSnapshot {
    index: u32,
    name: String,
    total_memory_bytes: u64,
    used_memory_bytes: u64,
    temperature_c: u32,
    persistence_mode: NvidiaPersistenceMode,
    power_limit_milliwatts: u32,
    application_graphics_clock_mhz: Option<u32>,
    application_memory_clock_mhz: Option<u32>,
    compute_process_count: u32,
}

impl NvidiaDeviceSnapshot {
    /// Physical NVML device index.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// NVML-reported device name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total framebuffer memory in bytes.
    #[must_use]
    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    /// Allocated framebuffer memory in bytes, excluding the NVML v2
    /// driver/firmware system-reserved amount.
    #[must_use]
    pub const fn used_memory_bytes(&self) -> u64 {
        self.used_memory_bytes
    }

    /// Current GPU temperature in degrees Celsius.
    #[must_use]
    pub const fn temperature_c(&self) -> u32 {
        self.temperature_c
    }

    /// Current driver persistence mode.
    #[must_use]
    pub const fn persistence_mode(&self) -> NvidiaPersistenceMode {
        self.persistence_mode
    }

    /// Current power-management limit in milliwatts.
    #[must_use]
    pub const fn power_limit_milliwatts(&self) -> u32 {
        self.power_limit_milliwatts
    }

    /// Application graphics clock in MHz, or `None` when NVML reports the
    /// field unsupported.
    #[must_use]
    pub const fn application_graphics_clock_mhz(&self) -> Option<u32> {
        self.application_graphics_clock_mhz
    }

    /// Application memory clock in MHz, or `None` when NVML reports the field
    /// unsupported.
    #[must_use]
    pub const fn application_memory_clock_mhz(&self) -> Option<u32> {
        self.application_memory_clock_mhz
    }

    /// Compute-process rows reported for this device at probe time.
    #[must_use]
    pub const fn compute_process_count(&self) -> u32 {
        self.compute_process_count
    }
}

/// Immutable, process-local NVIDIA environment snapshot.
///
/// NVML gathers these fields through multiple synchronous queries, so values
/// may change immediately after return. The probe creates no CUDA context and
/// launches no CUDA work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvidiaEnvironmentSnapshot {
    driver_version: String,
    cuda_driver_api_version: String,
    cuda_driver_api_version_raw: i32,
    gpu_count: u32,
    compute_process_count: u32,
    devices: Box<[NvidiaDeviceSnapshot]>,
}

impl NvidiaEnvironmentSnapshot {
    #[cfg(feature = "nvml")]
    fn from_native(native: ffi::NativeNvidiaEnvironmentSnapshot) -> CudaResult<Self> {
        let gpu_count = u32::try_from(native.devices.len()).map_err(|_| {
            invalid_snapshot("native NVIDIA device count does not fit the public API")
        })?;
        let devices = native
            .devices
            .into_iter()
            .map(|device| {
                let persistence_mode = match device.persistence_mode {
                    0 => NvidiaPersistenceMode::Disabled,
                    1 => NvidiaPersistenceMode::Enabled,
                    value => {
                        return Err(invalid_snapshot(format!(
                            "native NVIDIA persistence mode {value} is unknown"
                        )));
                    }
                };
                Ok(NvidiaDeviceSnapshot {
                    index: device.index,
                    name: device.name,
                    total_memory_bytes: device.total_memory_bytes,
                    used_memory_bytes: device.used_memory_bytes,
                    temperature_c: device.temperature_c,
                    persistence_mode,
                    power_limit_milliwatts: device.power_limit_milliwatts,
                    application_graphics_clock_mhz: device.application_graphics_clock_mhz,
                    application_memory_clock_mhz: device.application_memory_clock_mhz,
                    compute_process_count: device.compute_process_count,
                })
            })
            .collect::<CudaResult<Box<[_]>>>()?;
        Ok(Self {
            driver_version: native.driver_version,
            cuda_driver_api_version: format_cuda_driver_api_version(native.cuda_driver_api_version),
            cuda_driver_api_version_raw: native.cuda_driver_api_version,
            gpu_count,
            compute_process_count: native.compute_process_count,
            devices,
        })
    }

    /// Installed NVIDIA display-driver version string.
    #[must_use]
    pub fn driver_version(&self) -> &str {
        &self.driver_version
    }

    /// CUDA Driver API version in `major.minor` form.
    #[must_use]
    pub fn cuda_driver_api_version(&self) -> &str {
        &self.cuda_driver_api_version
    }

    /// Raw CUDA integer encoding, for example `12080` for CUDA 12.8.
    #[must_use]
    pub const fn cuda_driver_api_version_raw(&self) -> i32 {
        self.cuda_driver_api_version_raw
    }

    /// Number of physical devices reported by NVML.
    #[must_use]
    pub const fn gpu_count(&self) -> u32 {
        self.gpu_count
    }

    /// Sum of per-device compute-process rows at probe time.
    #[must_use]
    pub const fn compute_process_count(&self) -> u32 {
        self.compute_process_count
    }

    /// Devices in ascending physical NVML index order.
    #[must_use]
    pub fn devices(&self) -> &[NvidiaDeviceSnapshot] {
        &self.devices
    }
}

/// Captures the NVIDIA driver and physical-GPU environment in process.
///
/// The opt-in `nvml` feature deliberately controls the dynamic NVML
/// dependency. This function performs no subprocess invocation and creates no
/// CUDA context.
///
/// # Errors
///
/// Returns [`crate::CudaErrorKind::Unavailable`] when built without `nvml`, or
/// fails the whole snapshot on an NVML, cleanup, ABI, or output-contract error.
/// Application clocks are the only soft-optional fields.
pub fn probe_nvidia_environment() -> CudaResult<NvidiaEnvironmentSnapshot> {
    #[cfg(feature = "nvml")]
    {
        let actual_abi = ffi::abi_version();
        if actual_abi != crate::EXPECTED_ABI_VERSION {
            return Err(CudaError::new(
                crate::CudaErrorKind::Internal,
                crate::CudaErrorDomain::Internal,
                crate::CudaErrorStage::Initialize,
                0,
                "probe NVIDIA environment",
                format!(
                    "native ABI mismatch: Rust expects {}, native library reports {actual_abi}",
                    crate::EXPECTED_ABI_VERSION
                ),
            ));
        }
        NvidiaEnvironmentSnapshot::from_native(ffi::nvidia_environment_snapshot()?)
    }
    #[cfg(not(feature = "nvml"))]
    {
        Err(CudaError::nvml_unavailable("probe NVIDIA environment"))
    }
}

/// Exercises the native null-output validation before NVML initialization.
///
/// # Errors
///
/// Returns the expected invalid-argument error in an NVML-enabled build, or an
/// unavailable error when the probe feature is disabled.
#[doc(hidden)]
pub fn diagnose_null_nvidia_environment_output() -> CudaResult<()> {
    #[cfg(feature = "nvml")]
    {
        ffi::diagnose_null_nvidia_environment_snapshot()
    }
    #[cfg(not(feature = "nvml"))]
    {
        Err(CudaError::nvml_unavailable(
            "diagnose null NVIDIA environment output",
        ))
    }
}

#[cfg(any(feature = "nvml", test))]
fn format_cuda_driver_api_version(raw: i32) -> String {
    format!("{}.{}", raw / 1_000, (raw % 1_000) / 10)
}

#[cfg(feature = "nvml")]
fn invalid_snapshot(message: impl Into<String>) -> CudaError {
    CudaError::new(
        crate::CudaErrorKind::Internal,
        crate::CudaErrorDomain::Internal,
        crate::CudaErrorStage::Query,
        0,
        "probe NVIDIA environment",
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::format_cuda_driver_api_version;
    #[cfg(not(feature = "nvml"))]
    use super::{NVML_ENABLED, probe_nvidia_environment};
    #[cfg(not(feature = "nvml"))]
    use crate::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};

    #[test]
    fn cuda_driver_api_version_format_matches_cuda_encoding() {
        assert_eq!(format_cuda_driver_api_version(12_080), "12.8");
        assert_eq!(format_cuda_driver_api_version(13_000), "13.0");
    }

    #[cfg(not(feature = "nvml"))]
    #[test]
    fn feature_off_probe_is_actionable_without_loading_cuda_or_nvml() {
        assert!(!NVML_ENABLED);
        let error = probe_nvidia_environment().expect_err("NVML must remain disabled");
        assert_eq!(error.kind(), CudaErrorKind::Unavailable);
        assert_eq!(error.domain(), CudaErrorDomain::Rust);
        assert_eq!(error.stage(), CudaErrorStage::Initialize);
        assert!(error.message().contains("--features nvml"));
    }
}
