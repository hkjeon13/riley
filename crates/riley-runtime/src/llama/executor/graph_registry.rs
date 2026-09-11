//! Bounded immutable registry metadata for cold-prepared execution graphs.
//!
//! This C06-2 module deliberately owns no executable graph, device resource,
//! or model state. A replay slot is only a stable logical link that a later
//! C07 owner can resolve after it has validated native lifetime and address
//! invariants. The fixed array gives the hot lookup path a bounded scan with
//! no heap allocation.

use std::error;
use std::fmt;

use super::graph::GraphSignature;

/// Replay shape represented by one cold-prepared graph entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphReplayMode {
    /// A complete pure-decode graph, including the admitted output path.
    FullGraph,
    /// A fixed graph segment around exact eager dynamic boundaries.
    PiecewiseGraph,
}

/// Stable logical identifier for a graph resource owned outside this registry.
///
/// It is intentionally not an address or native handle. A future graph owner
/// is responsible for resolving it and for the resource's destruction order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphReplaySlot(u32);

impl GraphReplaySlot {
    /// Creates one cold-assigned replay-slot identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the stable, non-address replay-slot identifier.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Retained resources attributable to one graph entry.
///
/// Values are accounting metadata supplied by the cold graph owner. They do
/// not describe an allocation address or permit resource ownership here.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphEntryFootprint {
    retained_host_bytes: u64,
    retained_device_bytes: u64,
}

impl GraphEntryFootprint {
    /// Creates fixed retained-resource accounting for one entry.
    #[must_use]
    pub const fn new(retained_host_bytes: u64, retained_device_bytes: u64) -> Self {
        Self {
            retained_host_bytes,
            retained_device_bytes,
        }
    }

    /// Returns the host bytes retained by this entry.
    #[must_use]
    pub const fn retained_host_bytes(self) -> u64 {
        self.retained_host_bytes
    }

    /// Returns the device bytes retained by this entry.
    #[must_use]
    pub const fn retained_device_bytes(self) -> u64 {
        self.retained_device_bytes
    }
}

/// Cold state of a graph entry's future replay resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphRegistryEntryState {
    /// The owner has a replayable resource for the exact key.
    Prepared,
    /// The owner marked the exact resource unreplayable; it remains retained.
    Poisoned,
}

/// One immutable record in a bounded graph registry snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphRegistryEntry {
    signature: GraphSignature,
    replay_mode: GraphReplayMode,
    replay_slot: GraphReplaySlot,
    state: GraphRegistryEntryState,
    footprint: GraphEntryFootprint,
}

impl GraphRegistryEntry {
    /// Creates a cold-validated registry record without resource ownership.
    #[must_use]
    pub const fn new(
        signature: GraphSignature,
        replay_mode: GraphReplayMode,
        replay_slot: GraphReplaySlot,
        state: GraphRegistryEntryState,
        footprint: GraphEntryFootprint,
    ) -> Self {
        Self {
            signature,
            replay_mode,
            replay_slot,
            state,
            footprint,
        }
    }

    /// Returns the full equality key for this entry.
    #[must_use]
    pub const fn signature(self) -> GraphSignature {
        self.signature
    }

    /// Returns the exact replay mode for this entry.
    #[must_use]
    pub const fn replay_mode(self) -> GraphReplayMode {
        self.replay_mode
    }

    /// Returns the non-owning logical replay slot.
    #[must_use]
    pub const fn replay_slot(self) -> GraphReplaySlot {
        self.replay_slot
    }

    /// Returns whether the exact entry is prepared or poisoned.
    #[must_use]
    pub const fn state(self) -> GraphRegistryEntryState {
        self.state
    }

    /// Returns retained-resource accounting for this entry.
    #[must_use]
    pub const fn footprint(self) -> GraphEntryFootprint {
        self.footprint
    }
}

/// Explicit cold limits for one graph registry snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphRegistryLimits {
    maximum_graph_count: usize,
    maximum_full_graph_count: usize,
    maximum_piecewise_graph_count: usize,
    maximum_retained_host_bytes: u64,
    maximum_retained_device_bytes: u64,
}

impl GraphRegistryLimits {
    /// Creates all graph-count, mode-quota, and retained-byte limits.
    #[must_use]
    pub const fn new(
        maximum_graph_count: usize,
        maximum_full_graph_count: usize,
        maximum_piecewise_graph_count: usize,
        maximum_retained_host_bytes: u64,
        maximum_retained_device_bytes: u64,
    ) -> Self {
        Self {
            maximum_graph_count,
            maximum_full_graph_count,
            maximum_piecewise_graph_count,
            maximum_retained_host_bytes,
            maximum_retained_device_bytes,
        }
    }

    /// Creates a registry configuration that disables graph capacity.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::new(0, 0, 0, 0, 0)
    }

    /// Returns the total prepared-or-poisoned entry bound.
    #[must_use]
    pub const fn maximum_graph_count(self) -> usize {
        self.maximum_graph_count
    }

    /// Returns the full-graph entry quota.
    #[must_use]
    pub const fn maximum_full_graph_count(self) -> usize {
        self.maximum_full_graph_count
    }

    /// Returns the piecewise-graph entry quota.
    #[must_use]
    pub const fn maximum_piecewise_graph_count(self) -> usize {
        self.maximum_piecewise_graph_count
    }

    /// Returns the retained host-byte quota.
    #[must_use]
    pub const fn maximum_retained_host_bytes(self) -> u64 {
        self.maximum_retained_host_bytes
    }

    /// Returns the retained device-byte quota.
    #[must_use]
    pub const fn maximum_retained_device_bytes(self) -> u64 {
        self.maximum_retained_device_bytes
    }
}

/// Availability state for all lookups against an immutable registry snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphRegistryAvailability {
    /// The snapshot has cold-prepared capacity and can report exact misses.
    Enabled,
    /// Graph capacity is intentionally unavailable for every lookup.
    CapacityDisabled,
}

/// Aggregate retained resources and quota occupancy for a registry snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphRegistryUsage {
    entry_count: usize,
    full_graph_count: usize,
    piecewise_graph_count: usize,
    retained_host_bytes: u64,
    retained_device_bytes: u64,
}

impl GraphRegistryUsage {
    /// Returns entries retained in this snapshot, including poisoned entries.
    #[must_use]
    pub const fn entry_count(self) -> usize {
        self.entry_count
    }

    /// Returns full-graph entries, including poisoned entries.
    #[must_use]
    pub const fn full_graph_count(self) -> usize {
        self.full_graph_count
    }

    /// Returns piecewise-graph entries, including poisoned entries.
    #[must_use]
    pub const fn piecewise_graph_count(self) -> usize {
        self.piecewise_graph_count
    }

    /// Returns retained host bytes, including poisoned entries.
    #[must_use]
    pub const fn retained_host_bytes(self) -> u64 {
        self.retained_host_bytes
    }

    /// Returns retained device bytes, including poisoned entries.
    #[must_use]
    pub const fn retained_device_bytes(self) -> u64 {
        self.retained_device_bytes
    }
}

/// Typed cold-construction rejection for a bounded graph registry snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphRegistryBuildError {
    /// The configured total cannot fit into the selected fixed storage.
    MaximumGraphCountExceedsStorage {
        /// Requested maximum entry count.
        maximum_graph_count: usize,
        /// Fixed registry storage capacity.
        storage_capacity: usize,
    },
    /// A mode quota exceeds the configured total graph count.
    ModeGraphCountExceedsMaximum {
        /// Mode whose quota is invalid.
        replay_mode: GraphReplayMode,
        /// Requested mode-specific quota.
        maximum_mode_graph_count: usize,
        /// Configured total graph count.
        maximum_graph_count: usize,
    },
    /// A nonzero total capacity admits neither replay mode.
    NoReplayModeCapacity {
        /// Configured total graph count with no usable mode quota.
        maximum_graph_count: usize,
    },
    /// The supplied cold entries exceed the configured total count.
    EntryCountExceedsMaximum {
        /// Number of supplied entries.
        entry_count: usize,
        /// Configured total graph count.
        maximum_graph_count: usize,
    },
    /// The supplied full-graph entries exceed their quota.
    FullGraphCountExceedsMaximum {
        /// Number of supplied full-graph entries.
        full_graph_count: usize,
        /// Configured full-graph quota.
        maximum_full_graph_count: usize,
    },
    /// The supplied piecewise entries exceed their quota.
    PiecewiseGraphCountExceedsMaximum {
        /// Number of supplied piecewise entries.
        piecewise_graph_count: usize,
        /// Configured piecewise-graph quota.
        maximum_piecewise_graph_count: usize,
    },
    /// Retained host-byte accounting overflowed while validating entries.
    RetainedHostBytesOverflow,
    /// Retained device-byte accounting overflowed while validating entries.
    RetainedDeviceBytesOverflow,
    /// The entry set exceeds the retained host-byte limit.
    RetainedHostBytesExceedMaximum {
        /// Configured host-byte quota.
        maximum_retained_host_bytes: u64,
    },
    /// The entry set exceeds the retained device-byte limit.
    RetainedDeviceBytesExceedMaximum {
        /// Configured device-byte quota.
        maximum_retained_device_bytes: u64,
    },
    /// More than one entry has the same exact signature and replay mode.
    DuplicateKey {
        /// First entry with this key.
        existing_index: usize,
        /// Later conflicting entry.
        duplicate_index: usize,
    },
    /// More than one entry claims the same future owner slot.
    DuplicateReplaySlot {
        /// First entry with this slot.
        existing_index: usize,
        /// Later conflicting entry.
        duplicate_index: usize,
    },
}

impl fmt::Display for GraphRegistryBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaximumGraphCountExceedsStorage {
                maximum_graph_count,
                storage_capacity,
            } => write!(
                formatter,
                "execution graph maximum count {maximum_graph_count} exceeds fixed storage {storage_capacity}"
            ),
            Self::ModeGraphCountExceedsMaximum {
                replay_mode,
                maximum_mode_graph_count,
                maximum_graph_count,
            } => write!(
                formatter,
                "execution graph {replay_mode:?} quota {maximum_mode_graph_count} exceeds total maximum {maximum_graph_count}"
            ),
            Self::NoReplayModeCapacity {
                maximum_graph_count,
            } => write!(
                formatter,
                "execution graph maximum count {maximum_graph_count} has no enabled replay mode"
            ),
            Self::EntryCountExceedsMaximum {
                entry_count,
                maximum_graph_count,
            } => write!(
                formatter,
                "execution graph entry count {entry_count} exceeds maximum {maximum_graph_count}"
            ),
            Self::FullGraphCountExceedsMaximum {
                full_graph_count,
                maximum_full_graph_count,
            } => write!(
                formatter,
                "execution full-graph count {full_graph_count} exceeds maximum {maximum_full_graph_count}"
            ),
            Self::PiecewiseGraphCountExceedsMaximum {
                piecewise_graph_count,
                maximum_piecewise_graph_count,
            } => write!(
                formatter,
                "execution piecewise-graph count {piecewise_graph_count} exceeds maximum {maximum_piecewise_graph_count}"
            ),
            Self::RetainedHostBytesOverflow => {
                formatter.write_str("execution graph retained host-byte accounting overflowed")
            }
            Self::RetainedDeviceBytesOverflow => {
                formatter.write_str("execution graph retained device-byte accounting overflowed")
            }
            Self::RetainedHostBytesExceedMaximum {
                maximum_retained_host_bytes,
            } => write!(
                formatter,
                "execution graph retained host bytes exceed maximum {maximum_retained_host_bytes}"
            ),
            Self::RetainedDeviceBytesExceedMaximum {
                maximum_retained_device_bytes,
            } => write!(
                formatter,
                "execution graph retained device bytes exceed maximum {maximum_retained_device_bytes}"
            ),
            Self::DuplicateKey {
                existing_index,
                duplicate_index,
            } => write!(
                formatter,
                "execution graph entries {existing_index} and {duplicate_index} share an exact key"
            ),
            Self::DuplicateReplaySlot {
                existing_index,
                duplicate_index,
            } => write!(
                formatter,
                "execution graph entries {existing_index} and {duplicate_index} share a replay slot"
            ),
        }
    }
}

impl error::Error for GraphRegistryBuildError {}

/// Allocation-free result of an exact graph registry lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphRegistryLookup<'entry> {
    /// Graph capacity is globally unavailable for this snapshot.
    CapacityDisabled,
    /// No entry has the requested full signature and replay mode.
    NotPrepared,
    /// A replayable entry has the requested exact key.
    Prepared(&'entry GraphRegistryEntry),
    /// The requested exact entry is retained but must never be replayed.
    Poisoned(&'entry GraphRegistryEntry),
}

/// Fixed-capacity immutable snapshot of cold-prepared graph metadata.
///
/// `MAX_ENTRIES` is selected by the embedding runtime. Construction validates
/// all limits before publishing a snapshot; lookup only performs a bounded
/// exact scan and does not mutate the inventory or create a graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphRegistry<const MAX_ENTRIES: usize> {
    availability: GraphRegistryAvailability,
    limits: GraphRegistryLimits,
    usage: GraphRegistryUsage,
    entries: [Option<GraphRegistryEntry>; MAX_ENTRIES],
}

impl<const MAX_ENTRIES: usize> GraphRegistry<MAX_ENTRIES> {
    /// Creates an empty registry whose every lookup reports capacity disabled.
    #[must_use]
    pub const fn capacity_disabled() -> Self {
        Self {
            availability: GraphRegistryAvailability::CapacityDisabled,
            limits: GraphRegistryLimits::disabled(),
            usage: GraphRegistryUsage {
                entry_count: 0,
                full_graph_count: 0,
                piecewise_graph_count: 0,
                retained_host_bytes: 0,
                retained_device_bytes: 0,
            },
            entries: [None; MAX_ENTRIES],
        }
    }

    /// Builds one immutable registry snapshot after cold validation.
    ///
    /// The input slice is copied into fixed storage; no capacity is silently
    /// truncated and no executable graph or resource handle is retained.
    ///
    /// # Errors
    ///
    /// Returns a typed error when limits, accounting, exact keys, or replay
    /// slots are invalid for the selected fixed storage capacity.
    pub fn try_new(
        limits: GraphRegistryLimits,
        entries: &[GraphRegistryEntry],
    ) -> Result<Self, GraphRegistryBuildError> {
        if limits.maximum_graph_count > MAX_ENTRIES {
            return Err(GraphRegistryBuildError::MaximumGraphCountExceedsStorage {
                maximum_graph_count: limits.maximum_graph_count,
                storage_capacity: MAX_ENTRIES,
            });
        }
        validate_mode_limit(
            GraphReplayMode::FullGraph,
            limits.maximum_full_graph_count,
            limits.maximum_graph_count,
        )?;
        validate_mode_limit(
            GraphReplayMode::PiecewiseGraph,
            limits.maximum_piecewise_graph_count,
            limits.maximum_graph_count,
        )?;
        if limits.maximum_graph_count > 0
            && limits.maximum_full_graph_count == 0
            && limits.maximum_piecewise_graph_count == 0
        {
            return Err(GraphRegistryBuildError::NoReplayModeCapacity {
                maximum_graph_count: limits.maximum_graph_count,
            });
        }
        if entries.len() > limits.maximum_graph_count {
            return Err(GraphRegistryBuildError::EntryCountExceedsMaximum {
                entry_count: entries.len(),
                maximum_graph_count: limits.maximum_graph_count,
            });
        }
        if limits.maximum_graph_count == 0 {
            return Ok(Self {
                availability: GraphRegistryAvailability::CapacityDisabled,
                limits,
                usage: GraphRegistryUsage::default(),
                entries: [None; MAX_ENTRIES],
            });
        }

        let usage = validate_entries(limits, entries)?;
        let mut fixed_entries = [None; MAX_ENTRIES];
        for (index, entry) in entries.iter().copied().enumerate() {
            fixed_entries[index] = Some(entry);
        }

        Ok(Self {
            availability: GraphRegistryAvailability::Enabled,
            limits,
            usage,
            entries: fixed_entries,
        })
    }

    /// Returns whether this whole snapshot has graph capacity available.
    #[must_use]
    pub const fn availability(&self) -> GraphRegistryAvailability {
        self.availability
    }

    /// Returns the fixed cold limits stored with this snapshot.
    #[must_use]
    pub const fn limits(&self) -> GraphRegistryLimits {
        self.limits
    }

    /// Returns aggregate retained-resource and quota accounting.
    #[must_use]
    pub const fn usage(&self) -> GraphRegistryUsage {
        self.usage
    }

    /// Returns the compile-time maximum number of stored entry records.
    #[must_use]
    pub const fn storage_capacity(&self) -> usize {
        MAX_ENTRIES
    }

    /// Looks up one graph by full signature equality and exact replay mode.
    ///
    /// This hot-path operation has a bounded fixed-array scan. It never uses
    /// a fingerprint as authority, allocates, mutates state, or creates work.
    #[must_use]
    pub fn lookup(
        &self,
        signature: GraphSignature,
        replay_mode: GraphReplayMode,
    ) -> GraphRegistryLookup<'_> {
        if matches!(
            self.availability,
            GraphRegistryAvailability::CapacityDisabled
        ) {
            return GraphRegistryLookup::CapacityDisabled;
        }

        for stored_entry in &self.entries[..self.usage.entry_count] {
            let Some(entry) = stored_entry else {
                continue;
            };
            if entry.signature == signature && entry.replay_mode == replay_mode {
                return match entry.state {
                    GraphRegistryEntryState::Prepared => GraphRegistryLookup::Prepared(entry),
                    GraphRegistryEntryState::Poisoned => GraphRegistryLookup::Poisoned(entry),
                };
            }
        }

        GraphRegistryLookup::NotPrepared
    }
}

const fn validate_mode_limit(
    replay_mode: GraphReplayMode,
    maximum_mode_graph_count: usize,
    maximum_graph_count: usize,
) -> Result<(), GraphRegistryBuildError> {
    if maximum_mode_graph_count > maximum_graph_count {
        return Err(GraphRegistryBuildError::ModeGraphCountExceedsMaximum {
            replay_mode,
            maximum_mode_graph_count,
            maximum_graph_count,
        });
    }
    Ok(())
}

fn validate_entries(
    limits: GraphRegistryLimits,
    entries: &[GraphRegistryEntry],
) -> Result<GraphRegistryUsage, GraphRegistryBuildError> {
    let mut usage = GraphRegistryUsage::default();
    for (index, entry) in entries.iter().copied().enumerate() {
        validate_unique_entry(entries, index, entry)?;
        usage.entry_count += 1;
        match entry.replay_mode {
            GraphReplayMode::FullGraph => {
                usage.full_graph_count += 1;
                if usage.full_graph_count > limits.maximum_full_graph_count {
                    return Err(GraphRegistryBuildError::FullGraphCountExceedsMaximum {
                        full_graph_count: usage.full_graph_count,
                        maximum_full_graph_count: limits.maximum_full_graph_count,
                    });
                }
            }
            GraphReplayMode::PiecewiseGraph => {
                usage.piecewise_graph_count += 1;
                if usage.piecewise_graph_count > limits.maximum_piecewise_graph_count {
                    return Err(GraphRegistryBuildError::PiecewiseGraphCountExceedsMaximum {
                        piecewise_graph_count: usage.piecewise_graph_count,
                        maximum_piecewise_graph_count: limits.maximum_piecewise_graph_count,
                    });
                }
            }
        }
        usage.retained_host_bytes = usage
            .retained_host_bytes
            .checked_add(entry.footprint.retained_host_bytes)
            .ok_or(GraphRegistryBuildError::RetainedHostBytesOverflow)?;
        if usage.retained_host_bytes > limits.maximum_retained_host_bytes {
            return Err(GraphRegistryBuildError::RetainedHostBytesExceedMaximum {
                maximum_retained_host_bytes: limits.maximum_retained_host_bytes,
            });
        }
        usage.retained_device_bytes = usage
            .retained_device_bytes
            .checked_add(entry.footprint.retained_device_bytes)
            .ok_or(GraphRegistryBuildError::RetainedDeviceBytesOverflow)?;
        if usage.retained_device_bytes > limits.maximum_retained_device_bytes {
            return Err(GraphRegistryBuildError::RetainedDeviceBytesExceedMaximum {
                maximum_retained_device_bytes: limits.maximum_retained_device_bytes,
            });
        }
    }
    Ok(usage)
}

fn validate_unique_entry(
    entries: &[GraphRegistryEntry],
    index: usize,
    entry: GraphRegistryEntry,
) -> Result<(), GraphRegistryBuildError> {
    for (previous_index, previous) in entries[..index].iter().copied().enumerate() {
        if previous.signature == entry.signature && previous.replay_mode == entry.replay_mode {
            return Err(GraphRegistryBuildError::DuplicateKey {
                existing_index: previous_index,
                duplicate_index: index,
            });
        }
        if previous.replay_slot == entry.replay_slot {
            return Err(GraphRegistryBuildError::DuplicateReplaySlot {
                existing_index: previous_index,
                duplicate_index: index,
            });
        }
    }
    Ok(())
}
