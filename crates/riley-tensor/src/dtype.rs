use std::fmt;

/// The scalar representation of each logical tensor element.
///
/// Metadata operations preserve this value exactly. They never reinterpret or
/// convert the backing bytes to another dtype.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point, 16-bit.
    BF16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

impl DType {
    /// Returns the number of storage bytes per scalar element.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns the stable lowercase name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
