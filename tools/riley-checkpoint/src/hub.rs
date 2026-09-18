//! Explicit online transport; never linked into Riley serving or feature-off CLI.
use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};

use crate::{
    Plan, Result, TRANSPORT, create_directory, invalid, lower_hex, publish_with, regular_file,
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
}
