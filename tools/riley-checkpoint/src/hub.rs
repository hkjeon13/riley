//! Explicit online transport; never linked into Riley serving or feature-off CLI.
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};

use riley_model::{CONFIG_FILENAME, ModelConfig, TOKENIZER_FILENAME};
use sha2::{Digest, Sha256};

use crate::{
    Plan, Result, TRANSPORT, create_directory, hex_digest, invalid, lower_hex, publish_with,
    regular_file, safe_segment,
};

pub fn check_network_permission(allow_network: bool) -> Result<()> {
    if !allow_network {
        return Err(invalid("download requires --allow-network"));
    }
    if let Some(value) = std::env::var_os("HF_HUB_OFFLINE") {
        let disabled = matches!(
            value.to_str().map(str::to_ascii_lowercase).as_deref(),
            Some("0" | "false" | "no" | "off" | "")
        );
        if !disabled {
            return Err(invalid("HF_HUB_OFFLINE forbids download"));
        }
    }
    Ok(())
}

/// Download ONLY the pinned allowlist into a fresh private transport cache.
/// Tokens are supplied explicitly; no ambient Hub cache/config/token is used.
pub fn download(
    plan: &Plan,
    destination: &Path,
    allow_network: bool,
    token: Option<String>,
) -> Result<()> {
    check_network_permission(allow_network)?;
    if plan.provenance().converter_revision().is_some() {
        return Err(invalid(
            "Hub transport cannot claim an offline converter revision",
        ));
    }
    let repo = Repo::with_revision(
        plan.provenance().source_model().to_owned(),
        RepoType::Model,
        plan.provenance().source_revision().to_owned(),
    );
    let folder = repo.folder_name();
    let mut api: Option<ApiRepo> = None;
    let mut token = token;
    publish_with(plan, destination, |root, entry| {
        let cache = root.join(TRANSPORT);
        if api.is_none() {
            create_directory(&cache)?;
            let client = ApiBuilder::from_cache(Cache::new(cache.clone()))
                .with_token(token.take())
                .with_progress(false)
                .with_retries(0)
                .build()
                .map_err(|_| invalid("Hub client initialization failed"))?;
            api = Some(client.repo(repo.clone()));
        }
        let filename = entry
            .path()
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 payload name"))?;
        let pointer = api
            .as_ref()
            .ok_or_else(|| invalid("Hub client missing"))?
            .download(filename)
            // Do not put signed redirect URLs or authorization headers in errors.
            .map_err(|_| invalid(format!("Hub download failed: {filename}")))?;
        let repo_root = cache.join(&folder);
        let expected = repo_root
            .join("snapshots")
            .join(plan.provenance().source_revision())
            .join(filename);
        cache_blob(&repo_root, &expected, &pointer)
    })
}

fn cache_blob(repo_root: &Path, expected: &Path, returned: &Path) -> Result<PathBuf> {
    if returned != expected {
        return Err(invalid("Hub returned a different revision/path"));
    }
    // hf-hub 0.4.3 creates snapshot symlinks on Unix. They are transport
    // metadata, NOT checkpoint files. Decode only its exact private-cache
    // pointer grammar, then open the independently derived regular blob path.
    // Never canonicalize/follow an arbitrary caller-provided cache symlink.
    let metadata = fs::symlink_metadata(returned)?;
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(returned)?;
        let text = target
            .to_str()
            .ok_or_else(|| invalid("non-UTF-8 cache pointer"))?;
        let hash = text
            .strip_prefix("../../blobs/")
            .ok_or_else(|| invalid("unsafe cache pointer"))?;
        if !(lower_hex(hash, 40) || lower_hex(hash, 64)) {
            return Err(invalid("unsafe blob identity"));
        }
        let blob = repo_root.join("blobs").join(hash);
        regular_file(&blob)?;
        Ok(blob)
    } else {
        regular_file(returned)?;
        Ok(returned.to_owned())
    }
}

const HUB: &str = "https://huggingface.co";
const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JSON_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Writes a plan for `repository` at `revision` (branch, tag or commit) from
/// the Hub's own metadata at the resolved commit. Weights are pinned by the
/// Hub's LFS size and SHA-256; JSON files are downloaded, must match the Hub's
/// size and git blob id, and are then pinned by their SHA-256. The model must
/// pass Riley's config parser before any weight is downloaded.
pub fn plan(repository: &str, revision: &str, token: Option<String>) -> Result<Vec<u8>> {
    check_repository(repository)?;
    if revision.is_empty()
        || revision.contains("..")
        || !revision
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
    {
        return Err(invalid("revision must be a branch, tag or commit name"));
    }
    let agent = ureq::AgentBuilder::new().build();
    let get = |url: &str, limit: u64| -> Result<Vec<u8>> {
        let mut request = agent.get(url);
        if let Some(token) = token.as_deref() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        // Keep URLs with tokens or signed redirects out of error text.
        let response = request.call().map_err(|error| match error {
            ureq::Error::Status(401 | 403, _) => {
                invalid("Hub denied access; gated or private repositories need --token-env")
            }
            ureq::Error::Status(404, _) => {
                invalid("repository, revision or file not found on the Hub")
            }
            ureq::Error::Status(code, _) => invalid(format!("Hub request failed with HTTP {code}")),
            ureq::Error::Transport(_) => invalid("Hub request failed (network)"),
        })?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(limit + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit {
            return Err(invalid("Hub response exceeds size limit"));
        }
        Ok(bytes)
    };
    let metadata = get(
        &format!(
            "{HUB}/api/models/{repository}/revision/{}?blobs=true",
            revision.replace('/', "%2F")
        ),
        MAX_METADATA_BYTES,
    )?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&metadata).map_err(|_| invalid("malformed Hub metadata"))?;
    build_plan(repository, &metadata, |commit, path| {
        get(
            &format!("{HUB}/{repository}/resolve/{commit}/{path}"),
            MAX_JSON_PAYLOAD_BYTES,
        )
    })
}

fn check_repository(repository: &str) -> Result<()> {
    let parts: Vec<_> = repository.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|part| !safe_segment(part)) {
        return Err(invalid("repository must be a Hub id such as Qwen/Qwen3-4B"));
    }
    Ok(())
}

struct HubFile {
    size: u64,
    blob_id: String,
    lfs_sha256: Option<String>,
}

/// Network-free core of [`plan`]: `fetch(commit, path)` returns a small file.
fn build_plan(
    repository: &str,
    metadata: &serde_json::Value,
    mut fetch: impl FnMut(&str, &str) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let commit = metadata["sha"]
        .as_str()
        .filter(|sha| lower_hex(sha, 40))
        .ok_or_else(|| invalid("Hub metadata has no commit id"))?;
    let mut files = std::collections::BTreeMap::new();
    for sibling in metadata["siblings"]
        .as_array()
        .ok_or_else(|| invalid("Hub metadata has no file list"))?
    {
        let (Some(name), Some(size), Some(blob_id)) = (
            sibling["rfilename"].as_str(),
            sibling["size"].as_u64(),
            sibling["blobId"].as_str(),
        ) else {
            return Err(invalid("Hub metadata lacks file sizes or blob ids"));
        };
        let lfs_sha256 = match &sibling["lfs"] {
            serde_json::Value::Null => None,
            lfs => Some(
                lfs["sha256"]
                    .as_str()
                    .filter(|sha| lower_hex(sha, 64))
                    .filter(|_| lfs["size"].as_u64() == Some(size))
                    .ok_or_else(|| invalid(format!("Hub LFS metadata is malformed: {name}")))?
                    .to_owned(),
            ),
        };
        files.insert(
            name.to_owned(),
            HubFile {
                size,
                blob_id: blob_id.to_owned(),
                lfs_sha256,
            },
        );
    }

    let mut assertions = Vec::new();
    // Downloads a non-LFS file whose bytes must match the Hub's git blob id.
    let mut pin = |name: &str,
                   assertions: &mut Vec<serde_json::Value>|
     -> Result<Option<Vec<u8>>> {
        let file = files
            .get(name)
            .ok_or_else(|| invalid(format!("the repository has no {name}")))?;
        let (sha256, bytes) = match &file.lfs_sha256 {
            Some(sha256) => (sha256.clone(), None),
            None => {
                let bytes = fetch(commit, name)?;
                let mut blob = sha1::Sha1::new();
                blob.update(format!("blob {}\0", bytes.len()).as_bytes());
                blob.update(&bytes);
                if bytes.len() as u64 != file.size
                    || hex_digest(blob.finalize().as_ref()) != file.blob_id
                {
                    return Err(invalid(format!(
                        "{name} does not match the Hub's git blob id"
                    )));
                }
                (hex_digest(Sha256::digest(&bytes).as_ref()), Some(bytes))
            }
        };
        assertions.push(serde_json::json!({"path": name, "bytes": file.size, "sha256": sha256}));
        Ok(bytes)
    };

    let config = pin(CONFIG_FILENAME, &mut assertions)?
        .ok_or_else(|| invalid("config.json is stored in LFS; Riley cannot preflight it"))?;
    let model = ModelConfig::from_json_slice(&config)
        .map_err(|error| invalid(format!("model is not supported by Riley: {error}")))?;
    let dtype = model.to_model_spec().dtype().name();
    if !files.contains_key(TOKENIZER_FILENAME) {
        return Err(invalid(
            "the repository has no tokenizer.json; Riley does not read other tokenizer formats",
        ));
    }
    pin(TOKENIZER_FILENAME, &mut assertions)?;
    for optional in [
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        if files.contains_key(optional) {
            pin(optional, &mut assertions)?;
        }
    }
    if files.contains_key("model.safetensors") {
        pin("model.safetensors", &mut assertions)?;
    } else if files.contains_key("model.safetensors.index.json") {
        let index = pin("model.safetensors.index.json", &mut assertions)?
            .ok_or_else(|| invalid("model.safetensors.index.json is stored in LFS"))?;
        let index: serde_json::Value = serde_json::from_slice(&index)
            .map_err(|_| invalid("malformed model.safetensors.index.json"))?;
        let shards: std::collections::BTreeSet<&str> = index["weight_map"]
            .as_object()
            .ok_or_else(|| invalid("model.safetensors.index.json has no weight_map"))?
            .values()
            .map(|shard| {
                shard
                    .as_str()
                    .ok_or_else(|| invalid("weight_map entries must be names"))
            })
            .collect::<Result<_>>()?;
        for shard in shards {
            pin(shard, &mut assertions)?;
        }
    } else {
        return Err(invalid(
            "the repository has no safetensors weights; Riley does not read .bin or GGUF files",
        ));
    }

    let bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "format": "riley-checkpoint-v1",
        "source_model": repository,
        "source_revision": commit,
        "converter_revision": null,
        "transforms": [],
        "dtype": dtype,
        "files": assertions,
    }))?;
    // The same strict manifest checks as a hand-written plan.
    Plan::from_bytes(bytes.clone())?;
    Ok(bytes)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::tests::Temp;
    use std::os::unix::fs::symlink;

    #[test]
    fn private_cache_pointer_materializes_only_a_regular_blob() {
        let temp = Temp::new();
        let repo = temp.0.join("models--fixture--tiny");
        let revision = "1".repeat(40);
        let hash = "a".repeat(64);
        fs::create_dir_all(repo.join("snapshots").join(&revision)).unwrap();
        fs::create_dir_all(repo.join("blobs")).unwrap();
        let blob = repo.join("blobs").join(&hash);
        fs::write(&blob, b"payload").unwrap();
        let pointer = repo.join("snapshots").join(&revision).join("config.json");
        symlink(format!("../../blobs/{hash}"), &pointer).unwrap();
        assert_eq!(cache_blob(&repo, &pointer, &pointer).unwrap(), blob);
        assert!(cache_blob(&repo, &pointer.with_file_name("other.json"), &pointer).is_err());
        fs::remove_file(&blob).unwrap();
        symlink(temp.0.join("outside"), &blob).unwrap();
        assert!(cache_blob(&repo, &pointer, &pointer).is_err());
        fs::remove_file(&pointer).unwrap();
        symlink("../../../../etc/passwd", &pointer).unwrap();
        assert!(cache_blob(&repo, &pointer, &pointer).is_err());
    }

    #[test]
    fn absent_network_consent_is_rejected() {
        assert!(check_network_permission(false).is_err());
    }

    fn git_blob(bytes: &[u8]) -> String {
        let mut blob = sha1::Sha1::new();
        blob.update(format!("blob {}\0", bytes.len()).as_bytes());
        blob.update(bytes);
        hex_digest(blob.finalize().as_ref())
    }

    /// Hub metadata for the fixture files: weights in LFS, JSON files in git.
    fn metadata(root: &Path, lfs: &[&str]) -> serde_json::Value {
        let mut siblings = Vec::new();
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            let bytes = fs::read(entry.path()).unwrap();
            let lfs = lfs.contains(&name.as_str()).then(|| {
                serde_json::json!({"sha256": hex_digest(Sha256::digest(&bytes).as_ref()),
                    "size": bytes.len(), "pointerSize": 134})
            });
            siblings.push(serde_json::json!({"rfilename": name, "size": bytes.len(),
                "blobId": git_blob(&bytes), "lfs": lfs}));
        }
        serde_json::json!({"sha": "2".repeat(40), "siblings": siblings})
    }

    fn fetch_from(root: &Path) -> impl FnMut(&str, &str) -> Result<Vec<u8>> + '_ {
        move |commit, path| {
            assert_eq!(commit, "2".repeat(40));
            assert_ne!(path, "model.safetensors", "LFS weights must not be fetched");
            Ok(fs::read(root.join(path))?)
        }
    }

    #[test]
    fn hub_plan_pins_lfs_and_git_files_and_passes_the_real_loader() {
        let source = Temp::new();
        let parent = Temp::new();
        crate::tests::fixture(&source.0);
        fs::write(source.0.join("README.md"), b"ignored").unwrap();
        let metadata = metadata(&source.0, &["model.safetensors"]);
        let bytes = build_plan("fixture/tiny-llama", &metadata, fetch_from(&source.0)).unwrap();
        let plan = Plan::from_bytes(bytes).unwrap();
        assert_eq!(plan.provenance().source_revision(), "2".repeat(40));
        assert_eq!(plan.provenance().files().len(), 3);
        crate::prepare(&plan, &source.0, &parent.0.join("ready")).unwrap();
    }

    #[test]
    fn hub_plan_rejects_mismatched_blobs_unsupported_models_and_missing_weights() {
        let source = Temp::new();
        crate::tests::fixture(&source.0);
        let good = metadata(&source.0, &["model.safetensors"]);

        let mut tampered = good.clone();
        for sibling in tampered["siblings"].as_array_mut().unwrap() {
            if sibling["rfilename"] == "tokenizer.json" {
                sibling["blobId"] = serde_json::json!("0".repeat(40));
            }
        }
        let error = build_plan("fixture/tiny-llama", &tampered, fetch_from(&source.0)).unwrap_err();
        assert!(error.to_string().contains("git blob id"), "{error}");

        let mut no_weights = good;
        no_weights["siblings"]
            .as_array_mut()
            .unwrap()
            .retain(|sibling| sibling["rfilename"] != "model.safetensors");
        let error =
            build_plan("fixture/tiny-llama", &no_weights, fetch_from(&source.0)).unwrap_err();
        assert!(error.to_string().contains("no safetensors"), "{error}");

        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(source.0.join("config.json")).unwrap()).unwrap();
        config["model_type"] = serde_json::json!("gemma2");
        fs::write(
            source.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let unsupported = metadata(&source.0, &["model.safetensors"]);
        let error =
            build_plan("fixture/tiny-llama", &unsupported, fetch_from(&source.0)).unwrap_err();
        assert!(
            error.to_string().contains("not supported by Riley"),
            "{error}"
        );
    }

    #[test]
    fn hub_plan_rejects_unsafe_repository_and_revision_names() {
        assert!(plan("../etc", "main", None).is_err());
        assert!(plan("owner/model", "..", None).is_err());
        assert!(plan("owner/model", "a b", None).is_err());
    }
}
