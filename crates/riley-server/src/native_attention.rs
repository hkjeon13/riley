//! Explicit selection of separately qualified native attention numerics.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NativeAttentionMode {
    Page,
    FixedTileUnsplit,
}

impl NativeAttentionMode {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "page-v1" => Ok(Self::Page),
            "fixed-tile-unsplit-v1" => Ok(Self::FixedTileUnsplit),
            _ => Err("--native-attention-mode requires page-v1 or fixed-tile-unsplit-v1".into()),
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) const fn cli_id(self) -> &'static str {
        match self {
            Self::Page => "page-v1",
            Self::FixedTileUnsplit => "fixed-tile-unsplit-v1",
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) const fn setting(self) -> u64 {
        match self {
            Self::Page => 0,
            Self::FixedTileUnsplit => 1,
        }
    }
}
