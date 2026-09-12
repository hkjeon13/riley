//! V4 sixteen-row contract. This does not enable sixteen-row serving by itself.
pub use super::variable_wire::{Row, RowResult, encode_into, validate, validate_packet, validate_result, validate_batch_result, HEADER_BYTES, ROW_BYTES, RESULT_BYTES};
pub type Expectation=super::variable_wire::Expectation<16>;
pub type Layout=super::variable_wire::Layout<16>;
pub const MAGIC:u32=Layout::MAGIC;
pub const RESULT_MAGIC:u32=Layout::RESULT_MAGIC;
pub const VERSION:u32=Layout::VERSION;
pub const TOKENS_OFFSET:usize=Layout::TOKENS_OFFSET;
pub const REQUEST_BYTES:usize=Layout::REQUEST_BYTES;
pub const BATCH_RESULT_BYTES:usize=Layout::BATCH_RESULT_BYTES;
pub const OUTPUT_OFFSET:usize=Layout::OUTPUT_OFFSET;
pub const STAGING_BYTES:usize=Layout::STAGING_BYTES;
