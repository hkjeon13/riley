//! Immutable prefix identity, local sharing, and detached COW ownership.
//!
//! Device adapters must supply completion evidence before calling completion
//! methods. This layer neither launches CUDA nor treats cancellation as a fence.

use super::*;
use sha2::{Digest, Sha256};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{CowTransferError, CudaPendingCow};

#[cfg(test)]
mod tests;

/// Capacity-independent layout of one physical page across all layers.
/// V1 is BF16, separate K/V, [layer, page, head, token, dimension].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPageLayout {
    layers: usize,
    heads: usize,
    dimension: usize,
    block_size: usize,
    format_version: u16,
}

impl From<KvLayout> for KvPageLayout {
    fn from(layout: KvLayout) -> Self {
        Self {
            layers: layout.layer_count(),
            heads: layout.key_value_head_count(),
            dimension: layout.head_dimension(),
            block_size: KV_BLOCK_SIZE,
            format_version: 1,
        }
    }
}

/// Exact semantic identity. Adapters derive these fingerprints from the loaded
/// model revision/content, arithmetic profile, position encoding (including RoPE
/// parameters), and tensor/pipeline partition. Device ordinal is not identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvIdentity {
    pub model_revision: [u8; 32],
    pub numerical_profile: [u8; 32],
    pub position_encoding: [u8; 32],
    pub partition: [u8; 32],
    pub layout: KvPageLayout,
}

/// Token and position identity bound to the exact execution identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixDescriptor {
    identity: KvIdentity,
    token_digest: [u8; 32],
    start_position: u32,
    tokens: u32,
}

impl PrefixDescriptor {
    /// Builds an unambiguous length-delimited token fingerprint.
    ///
    /// # Errors
    /// Rejects empty, unrepresentable, or overflowing position ranges.
    pub fn new(identity: KvIdentity, start_position: u32, tokens: &[u32]) -> PagedKvResult<Self> {
        if [
            identity.model_revision,
            identity.numerical_profile,
            identity.position_encoding,
            identity.partition,
        ]
        .contains(&[0; 32])
        {
            return Err(invalid(
                "prefix_identity",
                "all execution fingerprints are required",
            ));
        }
        let count =
            u32::try_from(tokens.len()).map_err(|_| invalid("prefix_length", "must fit U32"))?;
        if count == 0 || start_position.checked_add(count).is_none() {
            return Err(invalid("prefix_position", "nonempty range must fit U32"));
        }
        let mut hash = Sha256::new();
        hash.update(b"riley.kv.prefix.tokens.v1\0");
        hash.update(count.to_le_bytes());
        for token in tokens {
            hash.update(token.to_le_bytes());
        }
        Ok(Self {
            identity,
            token_digest: hash.finalize().into(),
            start_position,
            tokens: count,
        })
    }

    #[must_use]
    pub const fn token_count(&self) -> u32 {
        self.tokens
    }

    #[must_use]
    pub fn identity(&self) -> &KvIdentity {
        &self.identity
    }
}

/// A validated descriptor and immutable local pages. Not a response cache.
#[derive(Debug)]
#[must_use = "release the export after all users quiesce"]
pub struct PrefixExport {
    descriptor: PrefixDescriptor,
    lease: KvReadLease,
}

impl PrefixExport {
    #[must_use]
    pub fn descriptor(&self) -> &PrefixDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        self.lease.blocks()
    }
}

fn invalid(field: &'static str, reason: &'static str) -> PagedKvError {
    PagedKvError::InvalidConfiguration { field, reason }
}

impl KvBlockPool {
    /// Captures already-completed KV. The execution adapter is responsible for
    /// binding the descriptor to the actual sequence tokens and loaded model.
    ///
    /// # Errors
    /// Rejects layout mismatch or any invalid/pending source lease.
    pub fn export_prefix(
        &mut self,
        sequence: &SequenceState,
        descriptor: PrefixDescriptor,
    ) -> PagedKvResult<PrefixExport> {
        if descriptor.identity.layout != KvPageLayout::from(self.layout) {
            return Err(invalid("prefix_layout", "does not match source pool"));
        }
        let lease = self.lease_prefix(sequence, descriptor.tokens as usize)?;
        Ok(PrefixExport { descriptor, lease })
    }

    /// Releases a cache/export hold after its readers have completed.
    ///
    /// # Errors
    /// Same ownership validation as `release_read_lease`.
    pub fn release_prefix(&mut self, export: &mut PrefixExport) -> PagedKvResult<()> {
        self.release_read_lease(&mut export.lease)
    }

    fn validate_lease(&self, lease: &KvReadLease) -> PagedKvResult<()> {
        if lease.pool_cookie != self.cookie {
            return Err(PagedKvError::ForeignPool {
                expected_cookie: self.cookie,
                actual_cookie: lease.pool_cookie,
            });
        }
        if lease.blocks().len() != blocks_for_length(lease.logical_length as usize) {
            return Err(invalid("read_lease", "released or incomplete lease"));
        }
        for &block in lease.blocks() {
            self.validate_block_index(block)?;
            let slot = &self.slots[block.physical_index as usize];
            if slot.generation != block.generation {
                return Err(PagedKvError::StaleBlock {
                    physical_index: block.physical_index,
                    handle_generation: block.generation,
                    current_generation: slot.generation,
                });
            }
            if slot.read_leases == 0 {
                return Err(invalid("read_lease", "missing retained reference"));
            }
        }
        Ok(())
    }
}

impl SequenceState {
    /// Shares a matching local prefix into an empty sequence, without copying KV.
    /// Partial last pages remain immutable until COW completes. This cold import
    /// can reserve host ownership storage; the ordinary append path cannot.
    ///
    /// # Errors
    /// Identity, pool, capacity, live-token and ownership validation precede
    /// publication. Allocation failure can grow metadata capacity but changes no
    /// page ownership or sequence state.
    pub fn import_prefix(
        &mut self,
        pool: &mut KvBlockPool,
        export: &PrefixExport,
        expected: &PrefixDescriptor,
    ) -> PagedKvResult<()> {
        self.ensure_mutable(pool)?;
        self.ensure_readable()?;
        if self.logical_length != 0 || self.allocated_block_count != 0 {
            return Err(invalid("prefix_target", "must be empty"));
        }
        if expected != &export.descriptor
            || expected.identity.layout != KvPageLayout::from(pool.layout)
        {
            return Err(invalid(
                "prefix_identity",
                "descriptor does not match destination",
            ));
        }
        if expected.tokens > self.maximum_logical_length {
            return Err(PagedKvError::CapacityExceeded {
                requested_tokens: expected.tokens as usize,
                maximum_tokens: self.maximum_logical_length as usize,
            });
        }
        pool.validate_lease(&export.lease)?;
        for block in export.blocks() {
            let slot = &mut pool.slots[block.physical_index as usize];
            if slot.owner.is_some() {
                let previous_capacity = slot.shared_owners.capacity();
                slot.shared_owners
                    .try_reserve(1)
                    .map_err(|_| PagedKvError::HostAllocation {
                        resource: "shared_page_owners",
                        requested_elements: slot.shared_owners.len() + 1,
                    })?;
                let delta = (slot.shared_owners.capacity() - previous_capacity)
                    * mem::size_of::<SequenceId>();
                pool.shared_owner_metadata_bytes = pool
                    .shared_owner_metadata_bytes
                    .saturating_add(delta as u64);
            }
        }
        for (index, &block) in export.blocks().iter().enumerate() {
            let slot = &mut pool.slots[block.physical_index as usize];
            if slot.owner.is_none() {
                slot.owner = Some(self.sequence_id);
            } else {
                slot.shared_owners.push(self.sequence_id);
            }
            self.block_ids[index] = Some(block);
            self.physical_block_ids[index] = block.physical_index;
        }
        self.allocated_block_count = export.blocks().len();
        self.fill_valid_tokens(expected.tokens as usize, self.allocated_block_count);
        self.logical_length = expected.tokens;
        Ok(())
    }
}

/// Result after actual device completion, including a cancelled/failed copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CopyOutcome {
    Published,
    Discarded,
    AlreadyCompleted,
}

#[derive(Debug)]
struct CopyState {
    source: KvReadLease,
    staging: SequenceState,
    staging_reservation: SequenceReservation,
    target_reservation: SequenceReservation,
    logical_index: usize,
    valid_tokens: u16,
    layout: KvLayout,
}

/// Detached COW operation. Holds source pages and a separately owned destination
/// until completion, even if the requesting sequence is reset or abandoned.
/// Dropping a live ticket conservatively leaks its holds until pool teardown.
#[must_use = "complete or discard only after device work has quiesced"]
#[derive(Debug)]
pub struct CopyOnWrite {
    state: Option<CopyState>,
}

/// One contiguous head slice in a separate K or V allocation. Both offsets
/// are relative to that allocation, not to a combined K+V buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCopyRegion {
    pub value: bool,
    pub source_offset: u64,
    pub destination_offset: u64,
    pub bytes: u64,
}

impl CopyOnWrite {
    /// Source, destination and token count for a transport adapter. The adapter
    /// must copy the valid range for every K/V head and layer before publication.
    #[must_use]
    pub fn copy_range(&self) -> Option<(BlockId, BlockId, u16)> {
        let state = self.state.as_ref()?;
        Some((
            state.source.blocks()[state.logical_index],
            state.staging.block_ids[0].expect("prepared staging page"),
            state.valid_tokens,
        ))
    }

    /// Allocation-free enumeration of every valid K/V slice. Unused tokens
    /// are excluded, so the copy never reads uninitialized tail elements.
    pub fn copy_regions(&self) -> impl Iterator<Item = KvCopyRegion> + '_ {
        self.state.iter().flat_map(|state| {
            let (source, destination, valid) = self.copy_range().expect("live copy");
            let layout = state.layout;
            [false, true].into_iter().flat_map(move |value| {
                (0..layout.layer_count()).flat_map(move |layer| {
                    (0..layout.key_value_head_count()).map(move |head| KvCopyRegion {
                        value,
                        source_offset: layout
                            .element_byte_offset(layer, source.physical_index as usize, head, 0)
                            .expect("validated source layout"),
                        destination_offset: layout
                            .element_byte_offset(
                                layer,
                                destination.physical_index as usize,
                                head,
                                0,
                            )
                            .expect("validated destination layout"),
                        bytes: u64::from(valid) * layout.head_dimension() as u64 * BF16_BYTES,
                    })
                })
            })
        })
    }
}

impl SequenceState {
    /// Prepares an isolated copy of the last shared/leased partial page.
    /// Returns None for empty, full-page, or already-exclusive tails.
    /// Device writes are not submitted here and no copied page is yet published.
    ///
    /// # Errors
    /// Rejects pending/poisoned state, OOM, or invalid ownership without modifying
    /// the target sequence. A returned ticket blocks target table publication.
    pub fn begin_copy_on_write(
        &mut self,
        pool: &mut KvBlockPool,
    ) -> PagedKvResult<Option<CopyOnWrite>> {
        self.ensure_mutable(pool)?;
        self.ensure_readable()?;
        let valid_tokens = (self.logical_length as usize % KV_BLOCK_SIZE) as u16;
        if valid_tokens == 0 {
            return Ok(None);
        }
        let index = self.allocated_block_count - 1;
        let block = self.allocated_block(index)?;
        pool.validate_block(block, self.sequence_id)?;
        let slot = &pool.slots[block.physical_index as usize];
        if slot.read_leases == 0 && slot.shared_owners.is_empty() {
            return Ok(None);
        }
        // Reserve staging first so OOM does not pin the source or block target.
        let mut staging = pool.create_sequence(KV_BLOCK_SIZE)?;
        let staging_reservation = staging.reserve_to(pool, valid_tokens as usize)?;
        let mut source = match pool.lease_prefix(self, self.logical_length as usize) {
            Ok(source) => source,
            Err(error) => {
                staging.reset(pool)?;
                return Err(error);
            }
        };
        let target_reservation = match self.reserve_to(pool, self.logical_length as usize) {
            Ok(reservation) => reservation,
            Err(error) => {
                staging.reset(pool)?;
                pool.release_read_lease(&mut source)?;
                return Err(error);
            }
        };
        Ok(Some(CopyOnWrite {
            state: Some(CopyState {
                source,
                staging,
                staging_reservation,
                target_reservation,
                logical_index: index,
                valid_tokens,
                layout: pool.layout,
            }),
        }))
    }

    /// Completes after ALL copy writes have quiesced. `succeeded` means every
    /// requested byte was copied successfully, not merely that enqueue succeeded.
    /// A late completion after reset/new reservation discards its private page
    /// without changing the new sequence transaction.
    ///
    /// # Errors
    /// Foreign pool/sequence or corrupt private ownership preserves the ticket.
    pub fn complete_copy_on_write(
        &mut self,
        pool: &mut KvBlockPool,
        ticket: &mut CopyOnWrite,
        succeeded: bool,
    ) -> PagedKvResult<CopyOutcome> {
        self.ensure_pool(pool)?;
        let Some(state) = ticket.state.as_ref() else {
            return Ok(CopyOutcome::AlreadyCompleted);
        };
        if state.target_reservation.sequence_id != self.sequence_id {
            return Err(PagedKvError::ReservationMismatch);
        }
        pool.validate_copy(state)?;
        let still_current =
            !self.poisoned && self.validate_reservation(&state.target_reservation).is_ok();
        if !succeeded || !still_current {
            if still_current {
                self.pending = None;
            }
            return pool.discard_completed_copy(ticket);
        }
        let old = self.allocated_block(state.logical_index)?;
        if old != state.source.blocks()[state.logical_index] {
            return Err(PagedKvError::ReservationMismatch);
        }
        pool.validate_block(old, self.sequence_id)?;
        let mut state = ticket.state.take().expect("validated ticket");
        let new = state.staging.allocated_block(0)?;
        // All fallible ownership checks precede this publication transition.
        pool.release_block(old, self.sequence_id)?;
        pool.slots[new.physical_index as usize].owner = Some(self.sequence_id);
        self.block_ids[state.logical_index] = Some(new);
        self.physical_block_ids[state.logical_index] = new.physical_index;
        self.pending = None;
        pool.release_read_lease(&mut state.source)?;
        // Staging ownership moved to the target. Its host table and reservation
        // are discarded without releasing the transferred physical page.
        Ok(CopyOutcome::Published)
    }
}

impl KvBlockPool {
    fn validate_copy(&self, state: &CopyState) -> PagedKvResult<()> {
        self.validate_lease(&state.source)?;
        state.staging.ensure_pool(self)?;
        state
            .staging
            .validate_reservation(&state.staging_reservation)?;
        let destination = state.staging.allocated_block(0)?;
        self.validate_block(destination, state.staging.sequence_id)?;
        let slot = &self.slots[destination.physical_index as usize];
        if !slot.shared_owners.is_empty() || slot.read_leases != 0 {
            return Err(invalid("copy_destination", "must remain exclusive"));
        }
        Ok(())
    }

    /// Discards a failed/cancelled/orphan copy after device quiescence. This can
    /// run after the target was consumed by `abandon_for_reclaim`. A still-live
    /// target must also reset or complete its pending transaction.
    ///
    /// # Errors
    /// Rejects invalid pool/ownership before releasing any holds.
    pub fn discard_completed_copy(
        &mut self,
        ticket: &mut CopyOnWrite,
    ) -> PagedKvResult<CopyOutcome> {
        let Some(state) = ticket.state.as_ref() else {
            return Ok(CopyOutcome::AlreadyCompleted);
        };
        self.validate_copy(state)?;
        let mut state = ticket.state.take().expect("validated ticket");
        state.staging.reset(self)?;
        self.release_read_lease(&mut state.source)?;
        Ok(CopyOutcome::Discarded)
    }
}
