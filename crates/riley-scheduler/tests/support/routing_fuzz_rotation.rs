//! Test-only seed-band selection for the scheduled C03 CPU routing-fuzz run.
//!
//! Normal PR coverage leaves every variable below unset and therefore keeps the
//! historical 10,000-trace seed windows exactly. The scheduled workflow supplies
//! all three values, records its random base in the workflow output, and assigns
//! one disjoint 10,000-index band to each matrix slot.

use std::env;
use std::ffi::OsString;

pub(crate) const SEED_ROTATION_BASE_ENV: &str = "RILEY_C03_SEED_ROTATION_BASE";
pub(crate) const SEED_ROTATION_SLOT_ENV: &str = "RILEY_C03_SEED_ROTATION_SLOT";
pub(crate) const SEED_ROTATION_SLOT_COUNT_ENV: &str = "RILEY_C03_SEED_ROTATION_SLOT_COUNT";
pub(crate) const SCHEDULED_SLOT_COUNT: u64 = 15;
pub(crate) const TRACES_PER_STREAM: u64 = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SeedRotation {
    base: u64,
    slot: u64,
}

impl SeedRotation {
    pub(crate) const fn baseline() -> Self {
        Self { base: 0, slot: 0 }
    }

    /// Returns the globally selected zero-based trace index for one local lane.
    ///
    /// The scheduled workflow has seven independent 10,000-trace lanes. Each
    /// matrix slot gets the same 10,000-index band in each lane; their seed
    /// factors remain the replay identity. Bounds are checked so malformed CI
    /// values cannot silently wrap into another slot's coverage.
    pub(crate) fn trace_index(
        self,
        local_trace_index: u64,
        traces_per_stream: u64,
    ) -> Result<u64, String> {
        if traces_per_stream == 0 {
            return Err("routing-fuzz traces_per_stream must be nonzero".to_owned());
        }
        if local_trace_index >= traces_per_stream {
            return Err(format!(
                "routing-fuzz local trace index {local_trace_index} is outside 0..{traces_per_stream}"
            ));
        }
        let slot_offset = self
            .slot
            .checked_mul(traces_per_stream)
            .ok_or_else(|| "routing-fuzz slot offset overflows u64".to_owned())?;
        let trace_index = self
            .base
            .checked_add(slot_offset)
            .and_then(|index| index.checked_add(local_trace_index))
            .ok_or_else(|| "routing-fuzz trace index overflows u64".to_owned())?;
        if trace_index == u64::MAX {
            return Err(
                "routing-fuzz trace index leaves no room for the one-based seed index".to_owned(),
            );
        }
        Ok(trace_index)
    }
}

/// Reads the all-or-nothing scheduled rotation configuration.
///
/// All values absent selects the historical baseline. Any partial or malformed
/// configuration is a test harness error instead of falling back to baseline
/// coverage.
pub(crate) fn configured_seed_rotation() -> SeedRotation {
    seed_rotation_from_values(
        env::var_os(SEED_ROTATION_BASE_ENV),
        env::var_os(SEED_ROTATION_SLOT_ENV),
        env::var_os(SEED_ROTATION_SLOT_COUNT_ENV),
    )
    .unwrap_or_else(|error| panic!("invalid C03 routing-fuzz seed rotation: {error}"))
}

/// Resolves a scheduled rotation from explicit environment-like values.
///
/// This pure entry point lets tests validate the contract without changing
/// process-global environment variables.
pub(crate) fn seed_rotation_from_values(
    base: Option<OsString>,
    slot: Option<OsString>,
    slot_count: Option<OsString>,
) -> Result<SeedRotation, String> {
    match (base, slot, slot_count) {
        (None, None, None) => Ok(SeedRotation::baseline()),
        (Some(base), Some(slot), Some(slot_count)) => {
            let base = parse_canonical_u64(SEED_ROTATION_BASE_ENV, base)?;
            let slot = parse_canonical_u64(SEED_ROTATION_SLOT_ENV, slot)?;
            let slot_count = parse_canonical_u64(SEED_ROTATION_SLOT_COUNT_ENV, slot_count)?;
            if slot_count != SCHEDULED_SLOT_COUNT {
                return Err(format!(
                    "{SEED_ROTATION_SLOT_COUNT_ENV} must be {SCHEDULED_SLOT_COUNT}, got {slot_count}"
                ));
            }
            if slot >= slot_count {
                return Err(format!(
                    "{SEED_ROTATION_SLOT_ENV} must be smaller than {SEED_ROTATION_SLOT_COUNT_ENV}"
                ));
            }
            let required_one_based_span = slot_count
                .checked_mul(TRACES_PER_STREAM)
                .expect("fixed scheduled routing-fuzz window fits u64");
            let maximum_base = u64::MAX
                .checked_sub(required_one_based_span)
                .expect("fixed scheduled routing-fuzz window leaves a nonzero base range");
            if base > maximum_base {
                return Err(format!(
                    "{SEED_ROTATION_BASE_ENV} leaves no room for {slot_count} scheduled {TRACES_PER_STREAM}-trace slots"
                ));
            }
            Ok(SeedRotation { base, slot })
        }
        _ => Err(format!(
            "{SEED_ROTATION_BASE_ENV}, {SEED_ROTATION_SLOT_ENV}, and {SEED_ROTATION_SLOT_COUNT_ENV} must be all set or all unset"
        )),
    }
}

fn parse_canonical_u64(variable: &str, value: OsString) -> Result<u64, String> {
    let value = value
        .into_string()
        .map_err(|_| format!("{variable} must be valid UTF-8 canonical decimal"))?;
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(format!(
            "{variable} must be a canonical unsigned decimal u64"
        ));
    }
    value
        .parse::<u64>()
        .map_err(|_| format!("{variable} must be a canonical unsigned decimal u64"))
}
