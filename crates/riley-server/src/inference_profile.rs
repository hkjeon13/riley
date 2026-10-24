//! Expand opt-in serving profiles before normal option validation.
use std::{collections::BTreeSet, ffi::OsString};

/// Defaults of the measured native BF16 route. Explicit options take priority;
/// the ordinary parser and model capability checks still validate the result.
const NATIVE_BF16: &[(&str, &str)] = &[
    ("--max-sequence-tokens", "1024"),
    ("--batch-token-budget", "256"),
    ("--prefill-chunk-tokens", "256"),
    ("--gemm-reduction-policy", "first-heuristic-split-k-v1"),
    (
        "--decode-attention-backend",
        "native-bf16-paged-split-gqa-d128-two-stage",
    ),
    ("--execution-graph-policy", "disabled"),
    ("--reduction-profile", "canonical-v1"),
    ("--projection-bias-backend", "merged-qkv-strict-staged-v1"),
    ("--batch-shape-policy", "power-of-two"),
    ("--execution-completion", "iteration-batch"),
    ("--metadata-transport", "synchronous"),
    ("--residual-rmsnorm", "fused"),
    (
        "--batch-shape-buckets",
        "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256",
    ),
    ("--decode-attention-merge", "global-max-v1"),
    ("--decode-step-graph", "capture-v1"),
    ("--mlp-activation", "fused-silu-multiply-v1"),
    ("--mlp-projection", "merged-gate-up-v1"),
    ("--query-key-rope", "fused-split-v1"),
    ("--gemm-algorithm-selection", "bit-equivalent-table-v1"),
    ("--mlp-residual-norm", "fused-next-norm-v1"),
    ("--lm-head-rows", "output-rows-v1"),
    ("--kv-cache-write", "fused-split-v1"),
    ("--decode-attention-producer", "tensor-core-v1"),
    ("--prefill-plateau-rows", "128,192"),
    ("--prefill-plateau-min-decode-rows", "6"),
    ("--sampling-backend", "gpu-greedy"),
    ("--event-publication", "after-submit-v1"),
    ("--startup-warmup", "step-shapes-v1"),
    ("--step-uploads", "staged-batch-v1"),
    ("--prompt-attention", "dense-v1"),
];

pub(super) fn expand(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<Vec<OsString>, String> {
    let mut source = arguments.into_iter();
    let Some(command) = source.next() else {
        return Ok(Vec::new());
    };
    let mut result = vec![command.clone()];
    if command != "serve" {
        result.extend(source);
        return Ok(result);
    }
    let mut selected = None;
    let mut explicit = BTreeSet::new();
    while let Some(flag) = source.next() {
        if flag == "--inference-profile" {
            if selected.is_some() {
                return Err("--inference-profile may occur only once".into());
            }
            let value = source
                .next()
                .ok_or("--inference-profile requires native-bf16-v1 or native-bf16-varlen-v1")?;
            selected = Some(match value.to_str() {
                Some("native-bf16-v1") => false,
                Some("native-bf16-varlen-v1") => true,
                _ => {
                    return Err(
                        "--inference-profile requires native-bf16-v1 or native-bf16-varlen-v1"
                            .into(),
                    );
                }
            });
            continue;
        }
        if let Some(name) = flag.to_str() {
            explicit.insert(name.to_owned());
        }
        let no_value = matches!(flag.to_str(), Some("--shutdown-on-stdin" | "--help" | "-h"));
        result.push(flag);
        // Preserve values verbatim, including non-UTF8 paths and values that
        // happen to look like option names. Unknown/missing options remain
        // errors in the ordinary parser rather than being reinterpreted here.
        if !no_value {
            if let Some(value) = source.next() {
                result.push(value);
            }
        }
    }
    if let Some(varlen) =
        selected.filter(|_| !explicit.contains("--help") && !explicit.contains("-h"))
    {
        for &(flag, value) in NATIVE_BF16 {
            if !explicit.contains(flag) {
                result.extend([OsString::from(flag), OsString::from(value)]);
            }
        }
        if !explicit.contains("--native-attention-mode") {
            result.extend([
                OsString::from("--native-attention-mode"),
                OsString::from(if varlen {
                    "fixed-tile-unsplit-v1"
                } else {
                    "page-v1"
                }),
            ]);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_defaults_match_the_published_phase0_argv() {
        let expanded =
            expand(["serve", "--inference-profile", "native-bf16-v1"].map(OsString::from)).unwrap();
        let expected: Vec<String> = serde_json::from_str(include_str!(
            "../../../benchmarks/evidence/20261022-phase0/native-extra.json"
        ))
        .unwrap();
        for pair in expected.chunks_exact(2) {
            let at = expanded
                .iter()
                .position(|v| v == pair[0].as_str())
                .unwrap_or_else(|| panic!("missing published option {}", pair[0]));
            assert_eq!(expanded[at + 1], OsString::from(&pair[1]), "{}", pair[0]);
        }
    }
    #[test]
    fn profile_leaves_explicit_values_and_duplicates_for_normal_validation() {
        let a = expand(
            [
                "serve",
                "--inference-profile",
                "native-bf16-v1",
                "--batch-token-budget",
                "128",
                "--batch-token-budget",
                "64",
            ]
            .map(OsString::from),
        )
        .unwrap();
        assert_eq!(a.iter().filter(|v| *v == "--batch-token-budget").count(), 2);
        assert_eq!(
            &a[1..5],
            &["--batch-token-budget", "128", "--batch-token-budget", "64"].map(OsString::from)
        );
    }
    #[test]
    fn option_like_paths_and_unselected_commands_are_preserved() {
        let a = ["serve", "--model", "--inference-profile"].map(OsString::from);
        assert_eq!(expand(a.clone()).unwrap(), a);
        let a = ["--help"].map(OsString::from);
        assert_eq!(expand(a.clone()).unwrap(), a);
        assert!(expand(["serve", "--inference-profile"].map(OsString::from)).is_err());
        assert!(expand(["serve", "--inference-profile", "unknown"].map(OsString::from)).is_err());
        assert!(
            expand(
                [
                    "serve",
                    "--inference-profile",
                    "native-bf16-v1",
                    "--inference-profile",
                    "native-bf16-v1"
                ]
                .map(OsString::from)
            )
            .is_err()
        );
    }
}
