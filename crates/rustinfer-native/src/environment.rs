#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::contract::PRIMARY_ENVIRONMENT_ID;

const EXPECTED_OS_ID: &str = "ubuntu";
const EXPECTED_OS_VERSION: &str = "22.04";
const EXPECTED_KERNEL: &str = "6.8.0-138-generic";
const EXPECTED_MACHINE: &str = "x86_64";
const EXPECTED_CPU: &str = "Intel Core i7-13700K";
const EXPECTED_PHYSICAL_CORES: usize = 16;
const EXPECTED_LOGICAL_THREADS: usize = 24;
const EXPECTED_RAM_BYTES: u64 = 67_185_598_464;
const EXPECTED_GOVERNOR: &str = "powersave";
const EXPECTED_POLICY_COUNT: usize = 24;
const EXPECTED_GPU_NAME: &str = "NVIDIA GeForce RTX 4090";
const EXPECTED_GPU_MEMORY_MIB: u64 = 24_564;
const EXPECTED_DRIVER: &str = "580.173.02";
const EXPECTED_DRIVER_CUDA_API: &str = "13.0";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AcceleratorObservation {
    pub(crate) gpu_count: usize,
    pub(crate) index: u32,
    pub(crate) name: String,
    pub(crate) compute_capability: (u32, u32),
    pub(crate) memory_total_mib: u64,
    pub(crate) driver_version: String,
    pub(crate) driver_cuda_api_version: String,
    pub(crate) persistence_mode: String,
    pub(crate) compute_process_count: usize,
    pub(crate) memory_used_mib: u64,
    pub(crate) temperature_c: u32,
    pub(crate) power_limit_w: f64,
    pub(crate) application_graphics_clock_mhz: Option<u32>,
    pub(crate) application_memory_clock_mhz: Option<u32>,
}

#[derive(Debug)]
pub(crate) enum EnvironmentError {
    Io { path: PathBuf, source: io::Error },
    Invalid(String),
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "cannot read environment fact {}: {source}",
                    path.display()
                )
            }
            Self::Invalid(reason) => write!(formatter, "primary environment differs: {reason}"),
        }
    }
}

impl Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

pub(crate) fn probe_primary_environment(
    accelerator: &AcceleratorObservation,
) -> Result<Value, EnvironmentError> {
    probe_primary_environment_from(
        Path::new("/etc/os-release"),
        Path::new("/proc/cpuinfo"),
        Path::new("/proc/meminfo"),
        Path::new("/proc/sys/kernel/osrelease"),
        Path::new("/sys/devices/system/cpu/cpufreq"),
        Path::new("/run/systemd/timesync/synchronized"),
        accelerator,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn probe_primary_environment_from(
    os_release_path: &Path,
    cpuinfo_path: &Path,
    meminfo_path: &Path,
    kernel_path: &Path,
    cpufreq_path: &Path,
    synchronized_path: &Path,
    accelerator: &AcceleratorObservation,
) -> Result<Value, EnvironmentError> {
    let os_release = parse_os_release(&read_text(os_release_path)?)?;
    let (cpu_model, physical_cores, logical_threads) = parse_cpuinfo(&read_text(cpuinfo_path)?)?;
    let ram_bytes = parse_meminfo(&read_text(meminfo_path)?)?;
    let kernel_release = read_text(kernel_path)?.trim().to_owned();
    let governors = read_governors(cpufreq_path)?;
    let clock_synchronized = synchronized_path.is_file();
    let host_cuda_toolkit_present = path_has_executable("nvcc");

    require_equal(
        "os_id",
        os_release.get("ID").map(String::as_str),
        EXPECTED_OS_ID,
    )?;
    require_equal(
        "os_version_id",
        os_release.get("VERSION_ID").map(String::as_str),
        EXPECTED_OS_VERSION,
    )?;
    require_equal("kernel_release", Some(&kernel_release), EXPECTED_KERNEL)?;
    require_equal("machine", Some(std::env::consts::ARCH), EXPECTED_MACHINE)?;
    require_equal("cpu_model", Some(&cpu_model), EXPECTED_CPU)?;
    require_number(
        "physical_cpu_cores",
        &physical_cores,
        &EXPECTED_PHYSICAL_CORES,
    )?;
    require_number(
        "logical_cpu_threads",
        &logical_threads,
        &EXPECTED_LOGICAL_THREADS,
    )?;
    require_number("ram_bytes", &ram_bytes, &EXPECTED_RAM_BYTES)?;
    if governors.len() != EXPECTED_POLICY_COUNT
        || governors
            .iter()
            .any(|governor| governor != EXPECTED_GOVERNOR)
    {
        return Err(EnvironmentError::Invalid(format!(
            "CPU governors must be {EXPECTED_POLICY_COUNT} copies of {EXPECTED_GOVERNOR:?}"
        )));
    }
    if !clock_synchronized {
        return Err(EnvironmentError::Invalid(
            "systemd clock synchronization marker is absent".to_owned(),
        ));
    }
    if host_cuda_toolkit_present {
        return Err(EnvironmentError::Invalid(
            "host CUDA toolkit must be absent from PATH".to_owned(),
        ));
    }
    validate_accelerator(accelerator)?;

    Ok(json!({
        "schema_version": "1.0.0",
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "host": {
            "os_id": EXPECTED_OS_ID,
            "os_version_id": EXPECTED_OS_VERSION,
            "kernel_release": kernel_release,
            "machine": EXPECTED_MACHINE,
            "cpu_model": cpu_model,
            "physical_cpu_cores": physical_cores,
            "logical_cpu_threads": logical_threads,
            "ram_bytes": ram_bytes,
            "cpu_governor": EXPECTED_GOVERNOR,
            "cpu_governor_policy_count": governors.len(),
            "clock_synchronized": true,
        },
        "accelerator": {
            "gpu_count": accelerator.gpu_count,
            "gpus": [{
                "index": accelerator.index,
                "name": accelerator.name,
                "compute_capability": format!("{}.{}", accelerator.compute_capability.0, accelerator.compute_capability.1),
                "memory_total_mib": accelerator.memory_total_mib,
                "bf16_compute_supported": accelerator.compute_capability.0 >= 8,
            }],
            "nvidia_driver_version": accelerator.driver_version,
            "driver_cuda_api_version": accelerator.driver_cuda_api_version,
            "cuda_driver_available": true,
            "host_cuda_toolkit_present": false,
            "primary_compute_dtype": "bf16",
            "persistence_mode": accelerator.persistence_mode,
            "compute_process_count": accelerator.compute_process_count,
            "memory_used_mib": accelerator.memory_used_mib,
            "temperature_c": accelerator.temperature_c,
            "power_limit_w": accelerator.power_limit_w,
            "application_graphics_clock_mhz": clock_text(accelerator.application_graphics_clock_mhz),
            "application_memory_clock_mhz": clock_text(accelerator.application_memory_clock_mhz),
        },
    }))
}

fn validate_accelerator(observed: &AcceleratorObservation) -> Result<(), EnvironmentError> {
    require_number("gpu_count", &observed.gpu_count, &1_usize)?;
    require_number("gpu.index", &observed.index, &0_u32)?;
    require_equal("gpu.name", Some(&observed.name), EXPECTED_GPU_NAME)?;
    if observed.compute_capability != (8, 9) {
        return Err(EnvironmentError::Invalid(format!(
            "compute capability is {:?}, expected (8, 9)",
            observed.compute_capability
        )));
    }
    require_number(
        "gpu.memory_total_mib",
        &observed.memory_total_mib,
        &EXPECTED_GPU_MEMORY_MIB,
    )?;
    require_equal(
        "nvidia_driver_version",
        Some(&observed.driver_version),
        EXPECTED_DRIVER,
    )?;
    require_equal(
        "driver_cuda_api_version",
        Some(&observed.driver_cuda_api_version),
        EXPECTED_DRIVER_CUDA_API,
    )?;
    require_equal(
        "persistence_mode",
        Some(&observed.persistence_mode),
        "Disabled",
    )?;
    require_number(
        "compute_process_count",
        &observed.compute_process_count,
        &0_usize,
    )?;
    if observed.memory_used_mib > 256 {
        return Err(EnvironmentError::Invalid(format!(
            "GPU memory used is {} MiB, maximum is 256",
            observed.memory_used_mib
        )));
    }
    if observed.temperature_c > 50 {
        return Err(EnvironmentError::Invalid(format!(
            "GPU temperature is {} C, maximum is 50",
            observed.temperature_c
        )));
    }
    if !observed.power_limit_w.is_finite() || (observed.power_limit_w - 450.0).abs() > f64::EPSILON
    {
        return Err(EnvironmentError::Invalid(format!(
            "GPU power limit is {}, expected 450 W",
            observed.power_limit_w
        )));
    }
    if observed.application_graphics_clock_mhz.is_some()
        || observed.application_memory_clock_mhz.is_some()
    {
        return Err(EnvironmentError::Invalid(
            "application clocks must both be unsupported ([N/A])".to_owned(),
        ));
    }
    Ok(())
}

fn read_text(path: &Path) -> Result<String, EnvironmentError> {
    fs::read_to_string(path).map_err(|source| EnvironmentError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_os_release(text: &str) -> Result<BTreeMap<String, String>, EnvironmentError> {
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let raw_value = raw_value.trim();
        let value = if raw_value.len() >= 2
            && ((raw_value.starts_with('"') && raw_value.ends_with('"'))
                || (raw_value.starts_with('\'') && raw_value.ends_with('\'')))
        {
            &raw_value[1..raw_value.len() - 1]
        } else {
            raw_value
        };
        values.insert(key.to_owned(), value.to_owned());
    }
    if values.is_empty() {
        return Err(EnvironmentError::Invalid("os-release is empty".to_owned()));
    }
    Ok(values)
}

fn parse_cpuinfo(text: &str) -> Result<(String, usize, usize), EnvironmentError> {
    let mut models = BTreeSet::new();
    let mut cores = BTreeSet::new();
    let mut logical = 0_usize;
    for block in text.split("\n\n") {
        let fields: BTreeMap<_, _> = block
            .lines()
            .filter_map(|line| line.split_once(':'))
            .map(|(key, value)| (key.trim(), value.trim()))
            .collect();
        if fields.contains_key("processor") {
            logical += 1;
            if let Some(model) = fields.get("model name") {
                models.insert((*model).to_owned());
            }
            if let Some(core_id) = fields.get("core id") {
                cores.insert((fields.get("physical id").copied().unwrap_or("0"), *core_id));
            }
        }
    }
    if models.len() != 1 || cores.is_empty() || logical == 0 {
        return Err(EnvironmentError::Invalid(
            "CPU identity records are incomplete or inconsistent".to_owned(),
        ));
    }
    let raw_model = models.into_iter().next().unwrap_or_default();
    let model = if raw_model.contains("i7-13700K") {
        EXPECTED_CPU.to_owned()
    } else {
        raw_model
    };
    Ok((model, cores.len(), logical))
}

fn parse_meminfo(text: &str) -> Result<u64, EnvironmentError> {
    let line = text
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .ok_or_else(|| EnvironmentError::Invalid("MemTotal is absent".to_owned()))?;
    let mut fields = line.split_ascii_whitespace();
    if fields.next() != Some("MemTotal:") {
        return Err(EnvironmentError::Invalid(
            "MemTotal is malformed".to_owned(),
        ));
    }
    let kib = fields
        .next()
        .ok_or_else(|| EnvironmentError::Invalid("MemTotal value is absent".to_owned()))?
        .parse::<u64>()
        .map_err(|_| EnvironmentError::Invalid("MemTotal is not numeric".to_owned()))?;
    if fields.next() != Some("kB") || fields.next().is_some() {
        return Err(EnvironmentError::Invalid(
            "MemTotal unit is not kB".to_owned(),
        ));
    }
    kib.checked_mul(1_024)
        .ok_or_else(|| EnvironmentError::Invalid("MemTotal overflows".to_owned()))
}

fn read_governors(root: &Path) -> Result<Vec<String>, EnvironmentError> {
    let entries = fs::read_dir(root).map_err(|source| EnvironmentError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    let mut paths = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let index = name.strip_prefix("policy")?.parse::<usize>().ok()?;
            Some((index, entry.path().join("scaling_governor")))
        })
        .collect::<Vec<_>>();
    paths.sort_unstable_by_key(|(index, _)| *index);
    if paths.is_empty() {
        return Err(EnvironmentError::Invalid(
            "no CPU frequency policies are visible".to_owned(),
        ));
    }
    paths
        .into_iter()
        .map(|(_, path)| Ok(read_text(&path)?.trim().to_owned()))
        .collect()
}

fn path_has_executable(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|directory| directory.join(name).is_file())
    })
}

fn require_equal(
    name: &str,
    observed: Option<&str>,
    expected: &str,
) -> Result<(), EnvironmentError> {
    if observed == Some(expected) {
        Ok(())
    } else {
        Err(EnvironmentError::Invalid(format!(
            "{name} is {observed:?}, expected {expected:?}"
        )))
    }
}

fn require_number<T: fmt::Display + PartialEq>(
    name: &str,
    observed: &T,
    expected: &T,
) -> Result<(), EnvironmentError> {
    if observed == expected {
        Ok(())
    } else {
        Err(EnvironmentError::Invalid(format!(
            "{name} is {observed}, expected {expected}"
        )))
    }
}

fn clock_text(clock: Option<u32>) -> Value {
    clock.map_or_else(|| json!("[N/A]"), |value| json!(value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{
        AcceleratorObservation, parse_cpuinfo, parse_meminfo, parse_os_release,
        validate_accelerator,
    };

    fn accelerator() -> AcceleratorObservation {
        AcceleratorObservation {
            gpu_count: 1,
            index: 0,
            name: "NVIDIA GeForce RTX 4090".to_owned(),
            compute_capability: (8, 9),
            memory_total_mib: 24_564,
            driver_version: "580.173.02".to_owned(),
            driver_cuda_api_version: "13.0".to_owned(),
            persistence_mode: "Disabled".to_owned(),
            compute_process_count: 0,
            memory_used_mib: 200,
            temperature_c: 42,
            power_limit_w: 450.0,
            application_graphics_clock_mhz: None,
            application_memory_clock_mhz: None,
        }
    }

    #[test]
    fn parses_host_files_without_commands() {
        let os = parse_os_release("ID=ubuntu\nVERSION_ID=\"22.04\"\n").expect("os");
        assert_eq!(os["VERSION_ID"], "22.04");
        let cpu = parse_cpuinfo(concat!(
            "processor: 0\nphysical id: 0\ncore id: 0\nmodel name: 13th Gen Intel(R) Core(TM) i7-13700K\n\n",
            "processor: 1\nphysical id: 0\ncore id: 1\nmodel name: 13th Gen Intel(R) Core(TM) i7-13700K\n",
        ))
        .expect("cpu");
        assert_eq!(cpu, ("Intel Core i7-13700K".to_owned(), 2, 2));
        assert_eq!(
            parse_meminfo("MemTotal:       65500 kB\n").expect("mem"),
            67_072_000
        );
    }

    #[test]
    fn accelerator_contract_is_fail_closed() {
        validate_accelerator(&accelerator()).expect("valid");
        let mut changed = accelerator();
        changed.compute_process_count = 1;
        assert!(validate_accelerator(&changed).is_err());
        changed = accelerator();
        changed.power_limit_w = 449.0;
        assert!(validate_accelerator(&changed).is_err());
    }
}
