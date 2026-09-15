//! Run one GPU validation command after explicit device preflight.
//! Exit 77 is an unsupported-hardware skip, not success. CUDA errors exit 1.
use riley_cuda::hardware_validation::{
    HardwareRequirement, HardwareValidationDisposition, HardwareValidationPolicy,
};
use riley_cuda::CudaRuntime;
use std::ffi::OsString;
use std::process::{Command, ExitCode};

#[derive(Debug)]
struct Options {
    requirement: HardwareRequirement,
    policy: HardwareValidationPolicy,
    command: Vec<OsString>,
    peer_access: bool,
}

fn parse(args: &[OsString]) -> Result<Options, String> {
    let separator = args
        .iter()
        .position(|arg| arg == "--")
        .ok_or("expected -- followed by a validation command")?;
    let mut requirement = HardwareRequirement {
        architectures: Vec::new(),
        minimum_devices: 1,
        minimum_memory_bytes: 0,
    };
    let mut policy = HardwareValidationPolicy::AllowHardwareSkip;
    let mut i = 0;
    let mut peer_access = false;
    while i < separator {
        let option = args[i].to_str().ok_or("non-UTF8 option")?;
        if option == "--peer-access" {
            peer_access = true;
            i += 1;
            continue;
        }
        if option == "--required" {
            policy = HardwareValidationPolicy::RequireHardware;
            i += 1;
            continue;
        }
        i += 1;
        if i >= separator {
            return Err(format!("missing value for {option}"));
        }
        let value = args[i].to_str().ok_or("non-UTF8 value")?;
        match option {
            "--arch" => {
                let (major, minor) = value.split_once('.').ok_or("--arch expects MAJOR.MINOR")?;
                let major = major
                    .parse::<u32>()
                    .map_err(|_| "invalid architecture major")?;
                let minor = minor
                    .parse::<u32>()
                    .map_err(|_| "invalid architecture minor")?;
                if major == 0 || minor > 9 {
                    return Err("invalid architecture".into());
                }
                requirement.architectures.push((major, minor));
            }
            "--devices" => {
                requirement.minimum_devices = value.parse().map_err(|_| "invalid device count")?;
                if requirement.minimum_devices == 0 {
                    return Err("device count must be positive".into());
                }
            }
            "--memory-bytes" => {
                requirement.minimum_memory_bytes =
                    value.parse().map_err(|_| "invalid memory size")?;
            }
            _ => return Err(format!("unknown option {option}")),
        }
        i += 1;
    }
    if requirement.architectures.is_empty() {
        return Err("at least one --arch is required".into());
    }
    let command = args[separator + 1..].to_vec();
    if command.is_empty() {
        return Err("validation command is empty".into());
    }
    Ok(Options {
        requirement,
        policy,
        command,
        peer_access,
    })
}

fn execute(disposition: HardwareValidationDisposition, command: &[OsString]) -> Result<u8, String> {
    match disposition {
        HardwareValidationDisposition::Skip => Ok(77),
        HardwareValidationDisposition::Fail => Ok(1),
        HardwareValidationDisposition::Ready => {
            let (program, args) = command.split_first().ok_or("empty command")?;
            let status = Command::new(program)
                .args(args)
                .status()
                .map_err(|e| e.to_string())?;
            // A child exit 77 is NOT an eligible hardware skip: preflight already
            // matched. Preserve failure rather than hiding ignored GPU tests.
            if status.success() {
                println!("RILEY_GPU_VALIDATION_COMMAND_RESULT=success");
                Ok(0)
            } else {
                println!("RILEY_GPU_VALIDATION_RESULT=fail child_status={status}");
                Ok(1)
            }
        }
    }
}

fn run() -> Result<u8, String> {
    let options = parse(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
    let runtime = CudaRuntime::initialize().map_err(|error| error.to_string())?;
    let devices = runtime
        .validation_devices()
        .map_err(|error| error.to_string())?;
    // CUDA_VISIBLE_DEVICES defines the selected set and is inherited by the child.
    let mut report = options.requirement.evaluate(&devices, options.policy);
    if options.peer_access && report.disposition == HardwareValidationDisposition::Ready {
        let links = runtime
            .validation_peer_access()
            .map_err(|error| error.to_string())?;
        report.require_full_peer_access(&links, options.policy);
        println!("RILEY_GPU_VALIDATION_PEERS={links:?}");
    }
    println!("RILEY_GPU_VALIDATION_PREFLIGHT={report:?}");
    println!("RILEY_GPU_VALIDATION_COMMAND={:?}", options.command);
    execute(report.disposition, &options.command)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("RILEY_GPU_VALIDATION_RESULT=fail error={error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }
    #[test]
    fn command_arguments_are_preserved_without_a_shell() {
        let options = parse(&args(&[
            "--arch",
            "9.0",
            "--arch",
            "10.0",
            "--devices",
            "2",
            "--required",
            "--",
            "test",
            "two words",
            "$(literal)",
        ]))
        .unwrap();
        assert_eq!(options.command, args(&["test", "two words", "$(literal)"]));
        assert_eq!(options.policy, HardwareValidationPolicy::RequireHardware);
        assert_eq!(options.requirement.architectures, vec![(9, 0), (10, 0)]);
    }
    #[test]
    fn malformed_requirements_fail_before_cuda_probe() {
        for input in [
            vec!["--", "test"],
            vec!["--arch", "9", "--", "test"],
            vec!["--arch", "9.0", "--devices", "0", "--", "test"],
            vec!["--arch", "9.0", "--"],
        ] {
            assert!(parse(&args(&input)).is_err());
        }
    }
    #[test]
    fn unavailable_hardware_never_launches_the_command() {
        let invalid = args(&["/definitely-not-a-validation-executable"]);
        assert_eq!(
            execute(HardwareValidationDisposition::Skip, &invalid).unwrap(),
            77
        );
        assert_eq!(
            execute(HardwareValidationDisposition::Fail, &invalid).unwrap(),
            1
        );
        assert!(execute(HardwareValidationDisposition::Ready, &invalid).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn child_skip_or_failure_does_not_become_success() {
        for code in ["1", "77"] {
            assert_eq!(
                execute(
                    HardwareValidationDisposition::Ready,
                    &args(&["sh", "-c", &format!("exit {code}")])
                )
                .unwrap(),
                1
            );
        }
        assert_eq!(
            execute(
                HardwareValidationDisposition::Ready,
                &args(&["sh", "-c", "exit 0"])
            )
            .unwrap(),
            0
        );
    }
}
