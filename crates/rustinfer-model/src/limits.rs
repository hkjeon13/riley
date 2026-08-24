use crate::{ModelError, ModelResult};

/// Explicit allocation and complexity bounds for untrusted model artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadLimits {
    config_bytes: u64,
    tokenizer_bytes: u64,
    index_bytes: u64,
    manifest_bytes: u64,
    safetensors_header_bytes: u64,
    shards: usize,
    tensors: usize,
    tensor_rank: usize,
    shard_bytes: u64,
    total_weight_bytes: u64,
    vocabulary_entries: usize,
    merges: usize,
    added_tokens: usize,
}

impl LoadLimits {
    /// Conservative production defaults large enough for the PR05 reference model.
    #[must_use]
    pub const fn production() -> Self {
        Self {
            config_bytes: 1024 * 1024,
            tokenizer_bytes: 64 * 1024 * 1024,
            index_bytes: 16 * 1024 * 1024,
            manifest_bytes: 4 * 1024 * 1024,
            safetensors_header_bytes: 16 * 1024 * 1024,
            shards: 256,
            tensors: 100_000,
            tensor_rank: 16,
            shard_bytes: 512 * 1024 * 1024,
            total_weight_bytes: 512 * 1024 * 1024,
            vocabulary_entries: 1_000_000,
            merges: 2_000_000,
            added_tokens: 65_536,
        }
    }

    /// Returns the maximum `config.json` size.
    #[must_use]
    pub const fn config_bytes(self) -> u64 {
        self.config_bytes
    }

    /// Returns the maximum `tokenizer.json` size.
    #[must_use]
    pub const fn tokenizer_bytes(self) -> u64 {
        self.tokenizer_bytes
    }

    /// Returns the maximum shard-index size.
    #[must_use]
    pub const fn index_bytes(self) -> u64 {
        self.index_bytes
    }

    /// Returns the maximum provenance-manifest size.
    #[must_use]
    pub const fn manifest_bytes(self) -> u64 {
        self.manifest_bytes
    }

    /// Returns the maximum safetensors JSON header size.
    #[must_use]
    pub const fn safetensors_header_bytes(self) -> u64 {
        self.safetensors_header_bytes
    }

    /// Returns the maximum number of checkpoint shards.
    #[must_use]
    pub const fn shards(self) -> usize {
        self.shards
    }

    /// Returns the maximum number of serialized tensors.
    #[must_use]
    pub const fn tensors(self) -> usize {
        self.tensors
    }

    /// Returns the maximum tensor rank.
    #[must_use]
    pub const fn tensor_rank(self) -> usize {
        self.tensor_rank
    }

    /// Returns the maximum bytes in one checkpoint shard.
    #[must_use]
    pub const fn shard_bytes(self) -> u64 {
        self.shard_bytes
    }

    /// Returns the maximum bytes across all checkpoint shards.
    #[must_use]
    pub const fn total_weight_bytes(self) -> u64 {
        self.total_weight_bytes
    }

    /// Returns the maximum tokenizer vocabulary size.
    #[must_use]
    pub const fn vocabulary_entries(self) -> usize {
        self.vocabulary_entries
    }

    /// Returns the maximum BPE merge count.
    #[must_use]
    pub const fn merges(self) -> usize {
        self.merges
    }

    /// Returns the maximum added-token count.
    #[must_use]
    pub const fn added_tokens(self) -> usize {
        self.added_tokens
    }

    /// Returns limits with an explicit resident weight-memory budget.
    ///
    /// PR05 retains every validated shard in memory. Raising this budget is an
    /// explicit operational decision until a later file-backed loader replaces
    /// the owned-read implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when either bound is zero, one shard could exceed the
    /// total resident budget, or the total cannot be represented by `usize`.
    pub fn with_weight_byte_limits(
        mut self,
        shard_bytes: u64,
        total_weight_bytes: u64,
    ) -> ModelResult<Self> {
        if shard_bytes == 0 || total_weight_bytes == 0 || shard_bytes > total_weight_bytes {
            return Err(ModelError::InvalidConfig {
                field: "load_limits.weight_bytes".to_owned(),
                reason:
                    "bounds must be positive and shard_bytes must not exceed total_weight_bytes"
                        .to_owned(),
            });
        }
        usize::try_from(total_weight_bytes).map_err(|_| ModelError::NumericOverflow {
            field: "load_limits.total_weight_bytes".to_owned(),
        })?;
        self.shard_bytes = shard_bytes;
        self.total_weight_bytes = total_weight_bytes;
        Ok(self)
    }
}

impl Default for LoadLimits {
    fn default() -> Self {
        Self::production()
    }
}

#[cfg(test)]
mod tests {
    use super::LoadLimits;

    #[test]
    fn weight_budget_is_checked_and_explicit() {
        let limits = LoadLimits::default()
            .with_weight_byte_limits(1024, 2048)
            .unwrap();
        assert_eq!(limits.shard_bytes(), 1024);
        assert_eq!(limits.total_weight_bytes(), 2048);
        assert!(
            LoadLimits::default()
                .with_weight_byte_limits(2048, 1024)
                .is_err()
        );
        assert!(
            LoadLimits::default()
                .with_weight_byte_limits(0, 1024)
                .is_err()
        );
    }
}
