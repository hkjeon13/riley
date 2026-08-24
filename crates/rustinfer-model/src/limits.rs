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
            shard_bytes: 32 * 1024 * 1024 * 1024,
            total_weight_bytes: 1024 * 1024 * 1024 * 1024,
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
}

impl Default for LoadLimits {
    fn default() -> Self {
        Self::production()
    }
}
