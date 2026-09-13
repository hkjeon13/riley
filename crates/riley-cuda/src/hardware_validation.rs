//! Hardware preflight for GPU validation. Eligibility is not a test pass.
//!
//! This checks explicitly selected devices, not peer topology or instruction
//! support. Those require independent probes before a backend can be selected.
use crate::{CudaResult, CudaRuntime};

/// A CUDA-observed device, or a synthetic device in a CPU policy test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationDevice {
    pub ordinal: u32,
    pub architecture: (u32, u32),
    pub memory_bytes: u64,
}

/// Explicit target architectures avoid assuming forward binary compatibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareRequirement {
    pub architectures: Vec<(u32, u32)>,
    pub minimum_devices: usize,
    pub minimum_memory_bytes: u64,
}

/// Required CI must fail rather than silently skip an unavailable target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareValidationPolicy {
    AllowHardwareSkip,
    RequireHardware,
}

/// Reasons are retained with both requirements and observations in the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HardwareMismatch {
    InvalidRequirement,
    DuplicateDevice(u32),
    DeviceCount { required: usize, observed: usize },
    Architecture { ordinal: u32, observed: (u32, u32) },
    Memory { ordinal: u32, observed: u64 },
    PeerAccess { source: u32, destination: u32 },
    InvalidPeerObservation { source: u32, destination: u32 },
}

/// Ready means the declared preflight matched, never that GPU tests passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareValidationDisposition {
    Ready,
    Skip,
    Fail,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareValidationReport {
    pub requirement: HardwareRequirement,
    pub observed: Vec<ValidationDevice>,
    pub mismatches: Vec<HardwareMismatch>,
    pub disposition: HardwareValidationDisposition,
}

impl CudaRuntime {
    /// Collect real observations. CUDA initialization/probe failures remain errors.
    /// This does not inspect peer access or authorize an optimized kernel.
    ///
    /// # Errors
    /// Propagates CUDA device metadata query failures.
    pub fn validation_devices(&self) -> CudaResult<Vec<ValidationDevice>> {
        self.devices().map(|devices| {
            devices
                .iter()
                .map(|device| {
                    let p = device.properties();
                    ValidationDevice {
                        ordinal: p.ordinal(),
                        architecture: p.compute_capability(),
                        memory_bytes: p.total_memory_bytes(),
                    }
                })
                .collect()
        })
    }
}

impl HardwareRequirement {
    /// Evaluate the actual selected device set. Invalid declarations always fail.
    #[must_use]
    pub fn evaluate(
        &self,
        selected: &[ValidationDevice],
        policy: HardwareValidationPolicy,
    ) -> HardwareValidationReport {
        let mut mismatches = Vec::new();
        let mut invalid = self.minimum_devices == 0
            || self.architectures.is_empty()
            || self
                .architectures
                .iter()
                .any(|&(major, minor)| major == 0 || minor > 9);
        if invalid {
            mismatches.push(HardwareMismatch::InvalidRequirement);
        }
        for (index, device) in selected.iter().enumerate() {
            if selected[..index]
                .iter()
                .any(|previous| previous.ordinal == device.ordinal)
            {
                invalid = true;
                mismatches.push(HardwareMismatch::DuplicateDevice(device.ordinal));
            }
            if !self.architectures.contains(&device.architecture) {
                mismatches.push(HardwareMismatch::Architecture {
                    ordinal: device.ordinal,
                    observed: device.architecture,
                });
            }
            if device.memory_bytes < self.minimum_memory_bytes {
                mismatches.push(HardwareMismatch::Memory {
                    ordinal: device.ordinal,
                    observed: device.memory_bytes,
                });
            }
        }
        if selected.len() < self.minimum_devices {
            mismatches.push(HardwareMismatch::DeviceCount {
                required: self.minimum_devices,
                observed: selected.len(),
            });
        }
        let disposition = if mismatches.is_empty() {
            HardwareValidationDisposition::Ready
        } else if invalid || policy == HardwareValidationPolicy::RequireHardware {
            HardwareValidationDisposition::Fail
        } else {
            HardwareValidationDisposition::Skip
        };
        HardwareValidationReport {
            requirement: self.clone(),
            observed: selected.to_vec(),
            mismatches,
            disposition,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn requirement() -> HardwareRequirement {
        HardwareRequirement {
            architectures: vec![(9, 0)],
            minimum_devices: 2,
            minimum_memory_bytes: 80,
        }
    }
    fn device(ordinal: u32, architecture: (u32, u32)) -> ValidationDevice {
        ValidationDevice {
            ordinal,
            architecture,
            memory_bytes: 80,
        }
    }
    #[test]
    fn absent_target_is_skip_only_in_optional_validation() {
        let selected = [device(0, (8, 9))];
        let optional =
            requirement().evaluate(&selected, HardwareValidationPolicy::AllowHardwareSkip);
        assert_eq!(optional.disposition, HardwareValidationDisposition::Skip);
        assert_eq!(optional.observed, selected);
        assert_eq!(optional.mismatches.len(), 2);
        assert_eq!(
            requirement()
                .evaluate(&selected, HardwareValidationPolicy::RequireHardware)
                .disposition,
            HardwareValidationDisposition::Fail
        );
    }
    #[test]
    fn matching_devices_are_ready_not_passed() {
        let selected = [device(0, (9, 0)), device(1, (9, 0))];
        assert_eq!(
            requirement()
                .evaluate(&selected, HardwareValidationPolicy::RequireHardware)
                .disposition,
            HardwareValidationDisposition::Ready
        );
    }
    #[test]
    fn newer_architecture_is_not_implicitly_compatible() {
        let selected = [device(0, (10, 0)), device(1, (10, 0))];
        assert_eq!(
            requirement()
                .evaluate(&selected, HardwareValidationPolicy::AllowHardwareSkip)
                .mismatches
                .len(),
            2
        );
    }
    #[test]
    fn duplicate_devices_and_invalid_requirements_cannot_skip() {
        let selected = [device(0, (9, 0)), device(0, (9, 0))];
        assert_eq!(
            requirement()
                .evaluate(&selected, HardwareValidationPolicy::AllowHardwareSkip)
                .disposition,
            HardwareValidationDisposition::Fail
        );
        let mut req = requirement();
        req.minimum_devices = 0;
        assert_eq!(
            req.evaluate(&[], HardwareValidationPolicy::AllowHardwareSkip)
                .disposition,
            HardwareValidationDisposition::Fail
        );
    }
    #[test]
    fn memory_shortage_is_reported_for_each_selected_device() {
        let mut selected = [device(0, (9, 0)), device(1, (9, 0))];
        selected[1].memory_bytes = 79;
        assert_eq!(
            requirement()
                .evaluate(&selected, HardwareValidationPolicy::RequireHardware)
                .mismatches,
            vec![HardwareMismatch::Memory {
                ordinal: 1,
                observed: 79
            }]
        );
    }
}

/// One directed CUDA peer query. A true value does not imply `NVLink` bandwidth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPeerAccess {
    pub source: u32,
    pub destination: u32,
    pub accessible: bool,
}

impl CudaRuntime {
    /// Query every ordered pair of the visible set; propagate query errors.
    ///
    /// # Errors
    /// Propagates any CUDA peer query failure without treating it as absence.
    pub fn validation_peer_access(&self) -> CudaResult<Vec<ValidationPeerAccess>> {
        let mut links = Vec::new();
        for source in 0..self.device_count() {
            for destination in 0..self.device_count() {
                if source != destination {
                    links.push(ValidationPeerAccess {
                        source,
                        destination,
                        accessible: self.can_access_peer(source, destination)?,
                    });
                }
            }
        }
        Ok(links)
    }
}

impl HardwareValidationReport {
    /// Require an observed accessible edge for each ordered pair. Missing edges
    /// or duplicate observations are errors, not unsupported hardware. Callers
    /// retain the link observations alongside this report.
    pub fn require_full_peer_access(
        &mut self,
        links: &[ValidationPeerAccess],
        policy: HardwareValidationPolicy,
    ) {
        if self.disposition != HardwareValidationDisposition::Ready {
            return;
        }
        let mut invalid = false;
        for link in links {
            if link.source == link.destination
                || !self.observed.iter().any(|d| d.ordinal == link.source)
                || !self.observed.iter().any(|d| d.ordinal == link.destination)
            {
                invalid = true;
                self.mismatches
                    .push(HardwareMismatch::InvalidPeerObservation {
                        source: link.source,
                        destination: link.destination,
                    });
            }
        }
        if self.observed.len() < 2 {
            self.mismatches.push(HardwareMismatch::DeviceCount {
                required: 2,
                observed: self.observed.len(),
            });
        }
        for source in &self.observed {
            for destination in &self.observed {
                if source.ordinal == destination.ordinal {
                    continue;
                }
                let mut found = links.iter().filter(|link| {
                    link.source == source.ordinal && link.destination == destination.ordinal
                });
                let first = found.next();
                if first.is_none() || found.next().is_some() {
                    invalid = true;
                    self.mismatches
                        .push(HardwareMismatch::InvalidPeerObservation {
                            source: source.ordinal,
                            destination: destination.ordinal,
                        });
                } else if !first.is_some_and(|link| link.accessible) {
                    self.mismatches.push(HardwareMismatch::PeerAccess {
                        source: source.ordinal,
                        destination: destination.ordinal,
                    });
                }
            }
        }
        if !self.mismatches.is_empty() {
            self.disposition = if invalid || policy == HardwareValidationPolicy::RequireHardware {
                HardwareValidationDisposition::Fail
            } else {
                HardwareValidationDisposition::Skip
            };
        }
    }
}

#[cfg(test)]
mod peer_tests {
    use super::*;
    #[test]
    fn directed_and_missing_edges_do_not_qualify() {
        let requirement = HardwareRequirement {
            architectures: vec![(9, 0)],
            minimum_devices: 2,
            minimum_memory_bytes: 0,
        };
        let devices: Vec<_> = (0..2)
            .map(|ordinal| ValidationDevice {
                ordinal,
                architecture: (9, 0),
                memory_bytes: 80,
            })
            .collect();
        let mut links = vec![ValidationPeerAccess {
            source: 0,
            destination: 1,
            accessible: true,
        }];
        let mut report = requirement.evaluate(&devices, HardwareValidationPolicy::RequireHardware);
        report.require_full_peer_access(&links, HardwareValidationPolicy::RequireHardware);
        assert_eq!(
            report.mismatches,
            vec![HardwareMismatch::InvalidPeerObservation {
                source: 1,
                destination: 0
            }]
        );
        links.push(ValidationPeerAccess {
            source: 1,
            destination: 0,
            accessible: true,
        });
        let mut report = requirement.evaluate(&devices, HardwareValidationPolicy::RequireHardware);
        report.require_full_peer_access(&links, HardwareValidationPolicy::RequireHardware);
        assert_eq!(report.disposition, HardwareValidationDisposition::Ready);
        links.push(links[0].clone());
        report.require_full_peer_access(&links, HardwareValidationPolicy::AllowHardwareSkip);
        assert_eq!(report.disposition, HardwareValidationDisposition::Fail);
    }
}
