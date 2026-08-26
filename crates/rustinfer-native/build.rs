use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "src/git.rs"]
mod git;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=RUSTINFER_SOURCE_REVISION");

    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let revision = if cuda && profile == "release" {
        if std::env::var("TARGET").as_deref() != Ok("x86_64-unknown-linux-gnu") {
            return Err(
                "native calibration release builds require x86_64-unknown-linux-gnu".into(),
            );
        }
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()?;
        if repository.join(".git").exists() {
            for path in git_watch_paths(&repository)? {
                println!("cargo:rerun-if-changed={}", path.display());
            }
            git::require_clean_repository(&repository)?.revision
        } else {
            archive_revision()?
        }
    } else {
        "unverified-non-release".to_owned()
    };
    println!("cargo:rustc-env=RUSTINFER_NATIVE_BUILD_GIT_REVISION={revision}");
    println!(
        "cargo:rustc-env=RUSTINFER_NATIVE_BUILD_PROFILE={}",
        if cuda && profile == "release" {
            "release-cuda-linux-x86_64"
        } else {
            "unverified-non-release"
        }
    );
    Ok(())
}

fn archive_revision() -> Result<String, Box<dyn Error>> {
    let revision = std::env::var("RUSTINFER_SOURCE_REVISION")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("archive build requires a lowercase 40-hex RUSTINFER_SOURCE_REVISION".into());
    }
    Ok(revision)
}

fn git_watch_paths(repository: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let marker = repository.join(".git");
    let git_dir = if marker.is_dir() {
        marker.clone()
    } else {
        let contents = fs::read_to_string(&marker)?;
        let relative = contents
            .trim()
            .strip_prefix("gitdir: ")
            .ok_or("Git indirection file is malformed")?;
        let path = PathBuf::from(relative);
        if path.is_absolute() {
            path
        } else {
            repository.join(path)
        }
    };
    let common_dir = match fs::read_to_string(git_dir.join("commondir")) {
        Ok(contents) => {
            let path = PathBuf::from(contents.trim());
            if path.is_absolute() {
                path
            } else {
                git_dir.join(path)
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => git_dir.clone(),
        Err(error) => return Err(error.into()),
    };
    let head_path = git_dir.join("HEAD");
    let head = fs::read_to_string(&head_path)?;
    let reference = head
        .trim()
        .strip_prefix("ref: ")
        .ok_or("Git HEAD is not symbolic")?;
    Ok(vec![
        marker,
        head_path,
        git_dir.join("index"),
        git_dir.join(reference),
        common_dir.join(reference),
    ])
}
