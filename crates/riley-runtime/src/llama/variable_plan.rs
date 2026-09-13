//! Cold identity for the fixed variable-serving backend's execution mode.
//! Model contents, device/GEMM metadata and kernel sources are hashed by the caller.
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug)]
pub(crate) struct VariablePlanSignature {
    pub rows: usize,
    pub shared: bool,
    pub compact: bool,
    pub packed: bool,
    pub mixed: bool,
    pub native_abi: u32,
}

impl VariablePlanSignature {
    pub fn bind(self, hash: &mut Sha256) -> Result<(), &'static str> {
        if !matches!(self.rows, 8 | 16 | 32)
            || (self.rows >= 16 && !self.shared)
            || (self.compact && (!self.shared || self.rows < 16))
            || (self.packed && (!self.shared || self.rows != 32))
            || (self.mixed && !self.packed)
            || self.native_abi == 0
        {
            return Err("unsupported variable backend plan");
        }
        hash.update(b"riley.variable-plan.bf16-hnd-page16.ordered.v1\0");
        hash.update(self.native_abi.to_le_bytes());
        hash.update((self.rows as u32).to_le_bytes());
        hash.update([
            u8::from(self.shared),
            u8::from(self.compact),
            u8::from(self.packed),
            u8::from(self.mixed),
        ]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn signature() -> VariablePlanSignature {
        VariablePlanSignature {
            rows: 32,
            shared: true,
            compact: false,
            packed: false,
            mixed: false,
            native_abi: 1,
        }
    }
    fn digest(plan: VariablePlanSignature) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"same-model-device-kernel-catalog");
        plan.bind(&mut hash).unwrap();
        hash.finalize().into()
    }
    #[test]
    fn different_execution_modes_do_not_alias_the_same_catalog() {
        let decode = signature();
        let packed = VariablePlanSignature {
            packed: true,
            ..decode
        };
        let mixed = VariablePlanSignature {
            mixed: true,
            ..packed
        };
        let compact = VariablePlanSignature {
            compact: true,
            ..mixed
        };
        let plans = [
            decode,
            packed,
            mixed,
            compact,
            VariablePlanSignature {
                native_abi: 2,
                ..decode
            },
        ];
        for (i, a) in plans.iter().enumerate() {
            for b in &plans[i + 1..] {
                assert_ne!(digest(*a), digest(*b));
            }
        }
        assert_eq!(digest(decode), digest(decode));
    }
    #[test]
    fn invalid_mode_does_not_mutate_catalog_hash() {
        let base = signature();
        for plan in [
            VariablePlanSignature {
                mixed: true,
                ..base
            },
            VariablePlanSignature {
                rows: 16,
                packed: true,
                ..base
            },
            VariablePlanSignature {
                shared: false,
                ..base
            },
            VariablePlanSignature {
                native_abi: 0,
                ..base
            },
        ] {
            let mut hash = Sha256::new();
            hash.update(b"unchanged");
            let before = hash.clone().finalize();
            assert!(plan.bind(&mut hash).is_err());
            assert_eq!(before, hash.finalize());
        }
    }
}
