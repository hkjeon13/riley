//! Subprocess-free Git provenance checks for native calibration evidence.
//!
//! This module intentionally implements the small SHA-1 and index subset that
//! the evidence producer needs. It does not consult Git configuration, invoke
//! hooks, or launch `git`, so provenance collection cannot execute repository
//! controlled programs.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

const EMPTY_STATUS_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const INDEX_SIGNATURE: &[u8; 4] = b"DIRC";
const CACHE_TREE_SIGNATURE: &[u8; 4] = b"TREE";
const INDEX_CHECKSUM_BYTES: usize = 20;
const INDEX_ENTRY_FIXED_BYTES: usize = 62;
const MAX_INDEX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_REF_BYTES: u64 = 4096;

/// Git bindings recorded for a repository proven clean without a subprocess.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CleanGitProvenance {
    /// Lowercase SHA-1 object name resolved from `HEAD`.
    pub revision: String,
    /// SHA-256 of the empty porcelain status required by the evidence contract.
    pub status_sha256: String,
}

/// Failure to prove that a repository is a clean SHA-1 Git checkout.
#[derive(Debug)]
pub(crate) enum GitProvenanceError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    InvalidRepository(String),
    UnsupportedRepository(String),
    DirtyRepository(String),
}

impl fmt::Display for GitProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(formatter, "cannot {operation} {}: {source}", path.display()),
            Self::InvalidRepository(message) => {
                write!(formatter, "invalid Git repository: {message}")
            }
            Self::UnsupportedRepository(message) => {
                write!(formatter, "unsupported Git repository: {message}")
            }
            Self::DirtyRepository(message) => write!(formatter, "dirty Git repository: {message}"),
        }
    }
}

impl Error for GitProvenanceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidRepository(_)
            | Self::UnsupportedRepository(_)
            | Self::DirtyRepository(_) => None,
        }
    }
}

/// Resolves `HEAD` and proves that the index, tracked files, and untracked-file
/// inventory are clean without invoking Git or any other subprocess.
///
/// `HEAD` must be symbolic and resolve through a loose ref whose modification
/// time is not older than the index. The accepted index format is Git index v2
/// or v3 with a complete, valid root `TREE` cache-tree extension. Untracked
/// files are rejected except below the root `.git`, root `target`, and
/// `benchmarks/results` paths.
///
/// # Errors
///
/// Returns an error for malformed or unsupported repository metadata, any
/// staged/unstaged/deleted tracked entry, or any non-exempt untracked file.
pub(crate) fn require_clean_repository(
    root: &Path,
) -> Result<CleanGitProvenance, GitProvenanceError> {
    let root = canonical_directory(root, "resolve repository root")?;
    let layout = resolve_git_layout(&root)?;
    let head = resolve_head(&layout)?;
    let index_path = layout.git_dir.join("index");
    let index_bytes = read_limited(&index_path, MAX_INDEX_BYTES, "read Git index")?;
    let index_modified = modified_time(&index_path, "read Git index modification time")?;
    if index_modified > head.ref_modified {
        return Err(dirty(
            "Git index is newer than the loose HEAD ref; staged changes cannot be excluded",
        ));
    }
    let index = parse_index(&index_bytes)?;

    verify_tracked_files(&root, &index.entries)?;
    let computed_tree = compute_index_tree(&index.entries)?;
    validate_cache_tree(&index.cache_tree, &computed_tree)?;
    reject_untracked_files(&root, &index.entries)?;

    Ok(CleanGitProvenance {
        revision: head.revision,
        status_sha256: EMPTY_STATUS_SHA256.to_owned(),
    })
}

#[derive(Debug)]
struct GitLayout {
    git_dir: PathBuf,
    common_dir: PathBuf,
}

fn canonical_directory(
    path: &Path,
    operation: &'static str,
) -> Result<PathBuf, GitProvenanceError> {
    let canonical = fs::canonicalize(path).map_err(|source| io_error(operation, path, source))?;
    let metadata = fs::metadata(&canonical)
        .map_err(|source| io_error("inspect directory", &canonical, source))?;
    if !metadata.is_dir() {
        return Err(invalid(format!("{} is not a directory", path.display())));
    }
    Ok(canonical)
}

fn resolve_git_layout(root: &Path) -> Result<GitLayout, GitProvenanceError> {
    let dot_git = root.join(".git");
    let metadata = fs::symlink_metadata(&dot_git)
        .map_err(|source| io_error("inspect .git", &dot_git, source))?;
    let git_dir = if metadata.is_dir() {
        fs::canonicalize(&dot_git)
            .map_err(|source| io_error("resolve .git directory", &dot_git, source))?
    } else if metadata.is_file() {
        let contents = read_limited(&dot_git, MAX_REF_BYTES, "read .git indirection")?;
        let text = parse_single_line(&contents, ".git indirection")?;
        let value = text
            .strip_prefix("gitdir: ")
            .ok_or_else(|| invalid(".git file must begin with `gitdir: `"))?;
        if value.is_empty() {
            return Err(invalid(".git indirection has an empty target"));
        }
        let candidate = Path::new(value);
        let candidate = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            root.join(candidate)
        };
        canonical_directory(&candidate, "resolve .git indirection")?
    } else {
        return Err(invalid(".git is neither a directory nor a gitdir file"));
    };

    let common_dir_path = git_dir.join("commondir");
    let common_dir = match fs::symlink_metadata(&common_dir_path) {
        Ok(metadata) if metadata.is_file() => {
            let contents = read_limited(
                &common_dir_path,
                MAX_REF_BYTES,
                "read Git common-directory indirection",
            )?;
            let value = parse_single_line(&contents, "Git commondir")?;
            if value.is_empty() {
                return Err(invalid("Git commondir is empty"));
            }
            let candidate = Path::new(value);
            let candidate = if candidate.is_absolute() {
                candidate.to_path_buf()
            } else {
                git_dir.join(candidate)
            };
            canonical_directory(&candidate, "resolve Git common directory")?
        }
        Ok(_) => return Err(invalid("Git commondir is not a regular file")),
        Err(source) if source.kind() == io::ErrorKind::NotFound => git_dir.clone(),
        Err(source) => {
            return Err(io_error(
                "inspect Git common-directory indirection",
                &common_dir_path,
                source,
            ));
        }
    };

    Ok(GitLayout {
        git_dir,
        common_dir,
    })
}

#[derive(Debug)]
struct ResolvedHead {
    revision: String,
    ref_modified: std::time::SystemTime,
}

fn resolve_head(layout: &GitLayout) -> Result<ResolvedHead, GitProvenanceError> {
    let head_path = layout.git_dir.join("HEAD");
    let contents = read_limited(&head_path, MAX_REF_BYTES, "read Git HEAD")?;
    let head = parse_single_line(&contents, "Git HEAD")?;
    let reference = head.strip_prefix("ref: ").ok_or_else(|| {
        unsupported("detached HEAD is not accepted; a loose symbolic HEAD ref is required")
    })?;
    validate_reference_name(reference)?;
    let (revision, ref_modified) = resolve_loose_reference(layout, reference)?;
    validate_revision(&revision)?;
    Ok(ResolvedHead {
        revision,
        ref_modified,
    })
}

fn resolve_loose_reference(
    layout: &GitLayout,
    reference: &str,
) -> Result<(String, std::time::SystemTime), GitProvenanceError> {
    let relative = reference_path(reference)?;
    for base in [&layout.git_dir, &layout.common_dir] {
        let path = base.join(&relative);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                let contents = read_limited(&path, MAX_REF_BYTES, "read loose Git ref")?;
                let modified = metadata.modified().map_err(|source| {
                    io_error("read loose Git ref modification time", &path, source)
                })?;
                return Ok((
                    parse_single_line(&contents, "loose Git ref")?.to_owned(),
                    modified,
                ));
            }
            Ok(_) => return Err(invalid(format!("loose ref {reference:?} is not a file"))),
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(io_error("inspect loose Git ref", &path, source)),
        }
        if layout.git_dir == layout.common_dir {
            break;
        }
    }
    Err(unsupported(format!(
        "HEAD ref {reference:?} is not loose; packed or missing refs are not accepted"
    )))
}

fn validate_reference_name(reference: &str) -> Result<(), GitProvenanceError> {
    if !reference.starts_with("refs/")
        || reference.ends_with('/')
        || reference.contains("//")
        || reference.contains("..")
        || reference.contains("@{")
        || reference
            .bytes()
            .any(|byte| byte <= b' ' || b"~^:?*[\\".contains(&byte))
    {
        return Err(invalid("HEAD contains an unsafe symbolic ref name"));
    }
    for component in reference.split('/') {
        if component.is_empty()
            || component == "."
            || component == ".."
            || component.ends_with('.')
            || component.as_bytes().ends_with(b".lock")
        {
            return Err(invalid("HEAD contains an unsafe symbolic ref component"));
        }
    }
    Ok(())
}

fn reference_path(reference: &str) -> Result<PathBuf, GitProvenanceError> {
    let path = Path::new(reference);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid("symbolic ref is not a safe relative path"));
    }
    Ok(path.to_path_buf())
}

fn validate_revision(revision: &str) -> Result<(), GitProvenanceError> {
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(unsupported(
            "HEAD must resolve to one lowercase 40-hex SHA-1 object name",
        ));
    }
    Ok(())
}

fn parse_single_line<'a>(bytes: &'a [u8], label: &str) -> Result<&'a str, GitProvenanceError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| invalid(format!("{label} is not valid UTF-8/ASCII")))?;
    let text = text.strip_suffix('\n').unwrap_or(text);
    let text = text.strip_suffix('\r').unwrap_or(text);
    if text.contains(['\n', '\r']) {
        return Err(invalid(format!("{label} must contain exactly one line")));
    }
    Ok(text)
}

#[derive(Clone, Debug)]
struct IndexEntry {
    path: Vec<u8>,
    mode: u32,
    object_id: [u8; 20],
}

#[derive(Debug)]
struct ParsedIndex {
    entries: Vec<IndexEntry>,
    cache_tree: Vec<u8>,
}

#[allow(clippy::too_many_lines)]
fn parse_index(bytes: &[u8]) -> Result<ParsedIndex, GitProvenanceError> {
    if bytes.len() < 12 + INDEX_CHECKSUM_BYTES {
        return Err(invalid("Git index is truncated"));
    }
    let checksum_start = bytes.len() - INDEX_CHECKSUM_BYTES;
    let expected_checksum = array_20(&bytes[checksum_start..])?;
    if Sha1::digest(&bytes[..checksum_start]) != expected_checksum {
        return Err(invalid("Git index checksum differs"));
    }
    if &bytes[..4] != INDEX_SIGNATURE {
        return Err(invalid("Git index signature differs"));
    }
    let version = read_u32(bytes, 4)?;
    if !matches!(version, 2 | 3) {
        return Err(unsupported(format!(
            "Git index version {version}; only v2/v3 are accepted"
        )));
    }
    let entry_count = usize::try_from(read_u32(bytes, 8)?)
        .map_err(|_| invalid("Git index entry count does not fit usize"))?;
    if entry_count > checksum_start.saturating_sub(12) / INDEX_ENTRY_FIXED_BYTES {
        return Err(invalid("Git index entry count exceeds file size"));
    }

    let mut cursor = 12_usize;
    let mut entries = Vec::new();
    entries
        .try_reserve(entry_count)
        .map_err(|_| invalid("cannot reserve Git index entries"))?;
    let mut previous_path: Option<Vec<u8>> = None;
    for _ in 0..entry_count {
        let entry_start = cursor;
        let fixed_end = cursor
            .checked_add(INDEX_ENTRY_FIXED_BYTES)
            .ok_or_else(|| invalid("Git index entry offset overflow"))?;
        if fixed_end > checksum_start {
            return Err(invalid("Git index entry is truncated"));
        }
        let mode = read_u32(bytes, cursor + 24)?;
        validate_index_mode(mode)?;
        let object_id = array_20(&bytes[cursor + 40..cursor + 60])?;
        if object_id == [0; 20] {
            return Err(unsupported("Git index contains a null object name"));
        }
        let flags = read_u16(bytes, cursor + 60)?;
        if flags & 0x3000 != 0 {
            return Err(dirty("Git index contains non-zero merge stage entries"));
        }
        cursor = fixed_end;
        if flags & 0x4000 != 0 {
            if version != 3 {
                return Err(unsupported(
                    "Git index v2 entry unexpectedly uses extended flags",
                ));
            }
            let extended = read_u16(bytes, cursor)?;
            cursor = cursor
                .checked_add(2)
                .ok_or_else(|| invalid("Git index extended-flag offset overflow"))?;
            if extended != 0 {
                return Err(unsupported(
                    "Git index skip-worktree/intent-to-add flags are not accepted",
                ));
            }
        }
        let nul_offset = bytes[cursor..checksum_start]
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| invalid("Git index path is not NUL terminated"))?;
        let path_end = cursor
            .checked_add(nul_offset)
            .ok_or_else(|| invalid("Git index path offset overflow"))?;
        let path = bytes[cursor..path_end].to_vec();
        validate_index_path(&path)?;
        let encoded_length = usize::from(flags & 0x0fff);
        if encoded_length != 0x0fff && encoded_length != path.len() {
            return Err(invalid("Git index pathname length flag differs"));
        }
        if let Some(previous) = &previous_path {
            match previous.as_slice().cmp(&path) {
                Ordering::Less => {}
                Ordering::Equal => return Err(invalid("Git index contains a duplicate path")),
                Ordering::Greater => return Err(invalid("Git index paths are not sorted")),
            }
        }
        previous_path = Some(path.clone());

        let unpadded_end = path_end
            .checked_add(1)
            .ok_or_else(|| invalid("Git index entry end overflow"))?;
        let relative_end = unpadded_end
            .checked_sub(entry_start)
            .ok_or_else(|| invalid("Git index entry alignment underflow"))?;
        let padded_length = relative_end
            .checked_add(7)
            .ok_or_else(|| invalid("Git index entry alignment overflow"))?
            & !7;
        cursor = entry_start
            .checked_add(padded_length)
            .ok_or_else(|| invalid("Git index entry alignment overflow"))?;
        if cursor > checksum_start || bytes[unpadded_end..cursor].iter().any(|byte| *byte != 0) {
            return Err(invalid("Git index entry padding is malformed"));
        }
        entries.push(IndexEntry {
            path,
            mode,
            object_id,
        });
    }

    let mut cache_tree = None;
    while cursor < checksum_start {
        let header_end = cursor
            .checked_add(8)
            .ok_or_else(|| invalid("Git index extension offset overflow"))?;
        if header_end > checksum_start {
            return Err(invalid("Git index extension header is truncated"));
        }
        let signature = &bytes[cursor..cursor + 4];
        let extension_size = usize::try_from(read_u32(bytes, cursor + 4)?)
            .map_err(|_| invalid("Git index extension size does not fit usize"))?;
        let payload_end = header_end
            .checked_add(extension_size)
            .ok_or_else(|| invalid("Git index extension length overflow"))?;
        if payload_end > checksum_start {
            return Err(invalid("Git index extension payload is truncated"));
        }
        if signature == CACHE_TREE_SIGNATURE {
            if cache_tree
                .replace(bytes[header_end..payload_end].to_vec())
                .is_some()
            {
                return Err(invalid("Git index contains duplicate TREE extensions"));
            }
        } else if signature == b"link" || signature == b"sdir" {
            return Err(unsupported(
                "split-index and sparse-index extensions are not accepted",
            ));
        } else if signature[0].is_ascii_lowercase() {
            return Err(unsupported(format!(
                "unknown mandatory Git index extension {:?}",
                String::from_utf8_lossy(signature)
            )));
        }
        cursor = payload_end;
    }
    let cache_tree = cache_tree.ok_or_else(|| {
        dirty("Git index has no valid root TREE cache; staged state cannot be excluded")
    })?;
    Ok(ParsedIndex {
        entries,
        cache_tree,
    })
}

fn validate_index_mode(mode: u32) -> Result<(), GitProvenanceError> {
    if matches!(mode, 0o100_644 | 0o100_755 | 0o120_000 | 0o160_000) {
        Ok(())
    } else {
        Err(unsupported(format!(
            "Git index mode {mode:o} is not a canonical file mode"
        )))
    }
}

fn validate_index_path(path: &[u8]) -> Result<(), GitProvenanceError> {
    if path.is_empty()
        || path.starts_with(b"/")
        || path.ends_with(b"/")
        || path
            .split(|byte| *byte == b'/')
            .any(|component| component.is_empty() || component == b"." || component == b"..")
    {
        return Err(invalid("Git index contains an unsafe pathname"));
    }
    if path.split(|byte| *byte == b'/').next() == Some(b".git") {
        return Err(invalid("Git index attempts to track repository metadata"));
    }
    Ok(())
}

fn verify_tracked_files(root: &Path, entries: &[IndexEntry]) -> Result<(), GitProvenanceError> {
    for entry in entries {
        let relative = bytes_to_relative_path(&entry.path)?;
        let path = root.join(relative);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return Err(dirty(format!(
                    "tracked path {} was deleted",
                    path.display()
                )));
            }
            Err(source) => return Err(io_error("inspect tracked path", &path, source)),
        };
        let actual = hash_worktree_entry(&path, &metadata, entry.mode)?;
        if actual != entry.object_id {
            return Err(dirty(format!(
                "tracked path {} differs from the Git index",
                path.display()
            )));
        }
    }
    Ok(())
}

fn hash_worktree_entry(
    path: &Path,
    metadata: &Metadata,
    mode: u32,
) -> Result<[u8; 20], GitProvenanceError> {
    match mode {
        0o100_644 | 0o100_755 => {
            if !metadata.file_type().is_file() {
                return Err(dirty(format!(
                    "tracked regular file {} changed type",
                    path.display()
                )));
            }
            verify_executable_mode(path, metadata, mode)?;
            hash_regular_blob(path, metadata.len())
        }
        0o120_000 => {
            if !metadata.file_type().is_symlink() {
                return Err(dirty(format!(
                    "tracked symlink {} changed type",
                    path.display()
                )));
            }
            let target = fs::read_link(path)
                .map_err(|source| io_error("read tracked symlink", path, source))?;
            let bytes = os_str_bytes(target.as_os_str())?;
            Ok(hash_git_object(b"blob", &bytes))
        }
        0o160_000 => Err(unsupported(format!(
            "tracked gitlink {} cannot be proven clean without recursing into another repository",
            path.display()
        ))),
        _ => Err(unsupported("unrecognized Git index file mode")),
    }
}

#[cfg(unix)]
fn verify_executable_mode(
    path: &Path,
    metadata: &Metadata,
    index_mode: u32,
) -> Result<(), GitProvenanceError> {
    use std::os::unix::fs::PermissionsExt;

    let executable = metadata.permissions().mode() & 0o100 != 0;
    if executable != (index_mode == 0o100_755) {
        return Err(dirty(format!(
            "tracked path {} has a different executable mode",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_executable_mode(
    _path: &Path,
    _metadata: &Metadata,
    _index_mode: u32,
) -> Result<(), GitProvenanceError> {
    Ok(())
}

fn hash_regular_blob(path: &Path, expected_len: u64) -> Result<[u8; 20], GitProvenanceError> {
    let mut file =
        File::open(path).map_err(|source| io_error("open tracked file", path, source))?;
    let opened_metadata = file
        .metadata()
        .map_err(|source| io_error("inspect opened tracked file", path, source))?;
    if !opened_metadata.is_file() || opened_metadata.len() != expected_len {
        return Err(dirty(format!(
            "tracked path {} changed while being inspected",
            path.display()
        )));
    }
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {expected_len}\0").as_bytes());
    let mut total = 0_u64;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io_error("hash tracked file", path, source))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| invalid("read length overflow"))?)
            .ok_or_else(|| invalid("tracked file length overflow"))?;
        hasher.update(&buffer[..read]);
    }
    if total != expected_len {
        return Err(dirty(format!(
            "tracked path {} changed while being hashed",
            path.display()
        )));
    }
    Ok(hasher.finalize())
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)]
fn os_str_bytes(value: &OsStr) -> Result<Vec<u8>, GitProvenanceError> {
    use std::os::unix::ffi::OsStrExt;

    Ok(value.as_bytes().to_vec())
}

#[cfg(not(unix))]
fn os_str_bytes(value: &OsStr) -> Result<Vec<u8>, GitProvenanceError> {
    value
        .to_str()
        .map(str::as_bytes)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| unsupported("non-UTF-8 repository paths require Unix"))
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)]
fn bytes_to_os_string(bytes: &[u8]) -> Result<OsString, GitProvenanceError> {
    use std::os::unix::ffi::OsStringExt;

    Ok(OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn bytes_to_os_string(bytes: &[u8]) -> Result<OsString, GitProvenanceError> {
    let value = std::str::from_utf8(bytes)
        .map_err(|_| unsupported("non-UTF-8 repository paths require Unix"))?;
    Ok(OsString::from(value))
}

fn bytes_to_relative_path(bytes: &[u8]) -> Result<PathBuf, GitProvenanceError> {
    let mut path = PathBuf::new();
    for component in bytes.split(|byte| *byte == b'/') {
        path.push(bytes_to_os_string(component)?);
    }
    Ok(path)
}

#[derive(Clone, Debug)]
enum TreeEntry {
    File { mode: u32, object_id: [u8; 20] },
    Directory(TreeNode),
}

#[derive(Clone, Debug, Default)]
struct TreeNode {
    entries: BTreeMap<Vec<u8>, TreeEntry>,
}

#[derive(Clone, Debug)]
struct ComputedTree {
    object_id: [u8; 20],
    entry_count: usize,
    directories: BTreeMap<Vec<u8>, ComputedTree>,
}

fn compute_index_tree(entries: &[IndexEntry]) -> Result<ComputedTree, GitProvenanceError> {
    let mut root = TreeNode::default();
    for entry in entries {
        let components: Vec<&[u8]> = entry.path.split(|byte| *byte == b'/').collect();
        insert_tree_entry(&mut root, &components, entry)?;
    }
    Ok(compute_tree(&root))
}

fn insert_tree_entry(
    node: &mut TreeNode,
    components: &[&[u8]],
    entry: &IndexEntry,
) -> Result<(), GitProvenanceError> {
    let (name, rest) = components
        .split_first()
        .ok_or_else(|| invalid("Git index path has no components"))?;
    if rest.is_empty() {
        if node
            .entries
            .insert(
                name.to_vec(),
                TreeEntry::File {
                    mode: entry.mode,
                    object_id: entry.object_id,
                },
            )
            .is_some()
        {
            return Err(invalid("Git index tree contains a duplicate entry"));
        }
        return Ok(());
    }
    let child = node
        .entries
        .entry(name.to_vec())
        .or_insert_with(|| TreeEntry::Directory(TreeNode::default()));
    match child {
        TreeEntry::Directory(directory) => insert_tree_entry(directory, rest, entry),
        TreeEntry::File { .. } => Err(invalid("Git index path collides with a file prefix")),
    }
}

fn compute_tree(node: &TreeNode) -> ComputedTree {
    let mut children: Vec<(&[u8], u32, [u8; 20], bool)> = Vec::with_capacity(node.entries.len());
    let mut directories = BTreeMap::new();
    let mut entry_count = 0_usize;
    for (name, entry) in &node.entries {
        match entry {
            TreeEntry::File { mode, object_id } => {
                entry_count += 1;
                children.push((name, *mode, *object_id, false));
            }
            TreeEntry::Directory(directory) => {
                let computed = compute_tree(directory);
                entry_count += computed.entry_count;
                children.push((name, 0o040_000, computed.object_id, true));
                directories.insert(name.clone(), computed);
            }
        }
    }
    children.sort_by(|left, right| git_tree_name_cmp(left.0, left.3, right.0, right.3));
    let mut contents = Vec::new();
    for (name, mode, object_id, _) in children {
        contents.extend_from_slice(format!("{mode:o} ").as_bytes());
        contents.extend_from_slice(name);
        contents.push(0);
        contents.extend_from_slice(&object_id);
    }
    ComputedTree {
        object_id: hash_git_object(b"tree", &contents),
        entry_count,
        directories,
    }
}

fn git_tree_name_cmp(left: &[u8], left_tree: bool, right: &[u8], right_tree: bool) -> Ordering {
    let common = left.len().min(right.len());
    match left[..common].cmp(&right[..common]) {
        Ordering::Equal => {
            let left_suffix = if left.len() == common {
                u8::from(left_tree) * b'/'
            } else {
                left[common]
            };
            let right_suffix = if right.len() == common {
                u8::from(right_tree) * b'/'
            } else {
                right[common]
            };
            left_suffix.cmp(&right_suffix)
        }
        ordering => ordering,
    }
}

fn validate_cache_tree(
    payload: &[u8],
    expected_root: &ComputedTree,
) -> Result<(), GitProvenanceError> {
    let mut cursor = 0_usize;
    parse_cache_tree_node(payload, &mut cursor, b"", expected_root)?;
    if cursor != payload.len() {
        return Err(invalid("TREE extension has trailing records"));
    }
    Ok(())
}

fn parse_cache_tree_node(
    payload: &[u8],
    cursor: &mut usize,
    expected_name: &[u8],
    expected: &ComputedTree,
) -> Result<(), GitProvenanceError> {
    let name_end = payload[*cursor..]
        .iter()
        .position(|byte| *byte == 0)
        .and_then(|offset| cursor.checked_add(offset))
        .ok_or_else(|| invalid("TREE extension node name is truncated"))?;
    if &payload[*cursor..name_end] != expected_name {
        return Err(invalid("TREE extension subtree order/name differs"));
    }
    *cursor = name_end
        .checked_add(1)
        .ok_or_else(|| invalid("TREE extension node offset overflow"))?;
    let header_end = payload[*cursor..]
        .iter()
        .position(|byte| *byte == b'\n')
        .and_then(|offset| cursor.checked_add(offset))
        .ok_or_else(|| invalid("TREE extension node header is truncated"))?;
    let header = std::str::from_utf8(&payload[*cursor..header_end])
        .map_err(|_| invalid("TREE extension node header is not ASCII"))?;
    let (entry_count, subtree_count) = header
        .split_once(' ')
        .ok_or_else(|| invalid("TREE extension node header is malformed"))?;
    let entry_count = entry_count
        .parse::<i64>()
        .map_err(|_| invalid("TREE extension entry count is malformed"))?;
    if entry_count < 0 {
        return Err(dirty(
            "Git index root/subtree cache is invalidated by staged changes",
        ));
    }
    let entry_count = usize::try_from(entry_count)
        .map_err(|_| invalid("TREE extension entry count does not fit usize"))?;
    let subtree_count = subtree_count
        .parse::<usize>()
        .map_err(|_| invalid("TREE extension subtree count is malformed"))?;
    *cursor = header_end
        .checked_add(1)
        .ok_or_else(|| invalid("TREE extension header offset overflow"))?;
    let object_end = cursor
        .checked_add(20)
        .ok_or_else(|| invalid("TREE extension object offset overflow"))?;
    if object_end > payload.len() {
        return Err(invalid("TREE extension object name is truncated"));
    }
    let object_id = array_20(&payload[*cursor..object_end])?;
    *cursor = object_end;

    if entry_count != expected.entry_count
        || subtree_count != expected.directories.len()
        || object_id != expected.object_id
    {
        return Err(dirty(
            "Git index TREE cache differs from the recursively computed index tree",
        ));
    }
    for (name, directory) in &expected.directories {
        parse_cache_tree_node(payload, cursor, name, directory)?;
    }
    Ok(())
}

fn reject_untracked_files(root: &Path, entries: &[IndexEntry]) -> Result<(), GitProvenanceError> {
    let tracked: HashSet<Vec<u8>> = entries.iter().map(|entry| entry.path.clone()).collect();
    walk_worktree(root, &[], &tracked)
}

fn walk_worktree(
    directory: &Path,
    prefix: &[u8],
    tracked: &HashSet<Vec<u8>>,
) -> Result<(), GitProvenanceError> {
    let listing = fs::read_dir(directory)
        .map_err(|source| io_error("enumerate repository directory", directory, source))?;
    for item in listing {
        let item = item.map_err(|source| io_error("read repository entry", directory, source))?;
        let name = os_str_bytes(&item.file_name())?;
        let relative = join_git_path(prefix, &name);
        if is_untracked_exception(&relative) {
            continue;
        }
        let file_type = item
            .file_type()
            .map_err(|source| io_error("inspect repository entry type", &item.path(), source))?;
        if file_type.is_dir() {
            walk_worktree(&item.path(), &relative, tracked)?;
        } else if !tracked.contains(&relative) {
            return Err(dirty(format!(
                "untracked path {} is not an allowed evidence output",
                item.path().display()
            )));
        }
    }
    Ok(())
}

fn join_git_path(prefix: &[u8], name: &[u8]) -> Vec<u8> {
    let mut path = Vec::with_capacity(prefix.len() + usize::from(!prefix.is_empty()) + name.len());
    path.extend_from_slice(prefix);
    if !prefix.is_empty() {
        path.push(b'/');
    }
    path.extend_from_slice(name);
    path
}

fn is_untracked_exception(path: &[u8]) -> bool {
    path == b".git"
        || path.starts_with(b".git/")
        || path == b"target"
        || path.starts_with(b"target/")
        || path == b"benchmarks/results"
        || path.starts_with(b"benchmarks/results/")
}

fn hash_git_object(kind: &[u8], contents: &[u8]) -> [u8; 20] {
    let mut hasher = Sha1::new();
    hasher.update(kind);
    hasher.update(b" ");
    hasher.update(contents.len().to_string().as_bytes());
    hasher.update(&[0]);
    hasher.update(contents);
    hasher.finalize()
}

fn modified_time(
    path: &Path,
    operation: &'static str,
) -> Result<std::time::SystemTime, GitProvenanceError> {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(|source| io_error(operation, path, source))
}

fn read_limited(
    path: &Path,
    maximum: u64,
    operation: &'static str,
) -> Result<Vec<u8>, GitProvenanceError> {
    let metadata = fs::metadata(path).map_err(|source| io_error(operation, path, source))?;
    if !metadata.is_file() {
        return Err(invalid(format!("{} is not a regular file", path.display())));
    }
    if metadata.len() > maximum {
        return Err(unsupported(format!(
            "{} exceeds the reviewed {maximum}-byte limit",
            path.display()
        )));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| unsupported("metadata file length does not fit usize"))?;
    let file = File::open(path).map_err(|source| io_error(operation, path, source))?;
    let mut contents = Vec::new();
    contents
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("cannot reserve metadata file buffer"))?;
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut contents)
        .map_err(|source| io_error(operation, path, source))?;
    if u64::try_from(contents.len()).unwrap_or(u64::MAX) > maximum {
        return Err(unsupported(format!(
            "{} grew beyond the reviewed {maximum}-byte limit",
            path.display()
        )));
    }
    Ok(contents)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, GitProvenanceError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| invalid("binary metadata offset overflow"))?;
    let raw: [u8; 2] = bytes
        .get(offset..end)
        .ok_or_else(|| invalid("binary metadata is truncated"))?
        .try_into()
        .map_err(|_| invalid("binary u16 field has the wrong length"))?;
    Ok(u16::from_be_bytes(raw))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, GitProvenanceError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| invalid("binary metadata offset overflow"))?;
    let raw: [u8; 4] = bytes
        .get(offset..end)
        .ok_or_else(|| invalid("binary metadata is truncated"))?
        .try_into()
        .map_err(|_| invalid("binary u32 field has the wrong length"))?;
    Ok(u32::from_be_bytes(raw))
}

fn array_20(bytes: &[u8]) -> Result<[u8; 20], GitProvenanceError> {
    bytes
        .try_into()
        .map_err(|_| invalid("SHA-1 object name has the wrong length"))
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> GitProvenanceError {
    GitProvenanceError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

fn invalid(message: impl Into<String>) -> GitProvenanceError {
    GitProvenanceError::InvalidRepository(message.into())
}

fn unsupported(message: impl Into<String>) -> GitProvenanceError {
    GitProvenanceError::UnsupportedRepository(message.into())
}

fn dirty(message: impl Into<String>) -> GitProvenanceError {
    GitProvenanceError::DirtyRepository(message.into())
}

#[derive(Clone, Debug)]
struct Sha1 {
    state: [u32; 5],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Sha1 {
    const fn new() -> Self {
        Self {
            state: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    fn digest(bytes: &[u8]) -> [u8; 20] {
        let mut hasher = Self::new();
        hasher.update(bytes);
        hasher.finalize()
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.length = self
            .length
            .wrapping_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        if self.buffered != 0 {
            let take = (64 - self.buffered).min(bytes.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&bytes[..take]);
            self.buffered += take;
            bytes = &bytes[take..];
            if self.buffered < 64 {
                return;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffered = 0;
        }
        while bytes.len() >= 64 {
            let block: &[u8; 64] = bytes[..64].try_into().expect("64-byte SHA-1 block");
            self.compress(block);
            bytes = &bytes[64..];
        }
        self.buffer[..bytes.len()].copy_from_slice(bytes);
        self.buffered = bytes.len();
    }

    fn finalize(mut self) -> [u8; 20] {
        let bit_length = self.length.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut digest = [0_u8; 20];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    #[allow(clippy::many_single_char_names)]
    fn compress(&mut self, block: &[u8; 64]) {
        let mut schedule = [0_u32; 80];
        for (slot, chunk) in schedule[..16].iter_mut().zip(block.chunks_exact(4)) {
            *slot = u32::from_be_bytes(chunk.try_into().expect("four-byte SHA-1 word"));
        }
        for index in 16..80 {
            schedule[index] = (schedule[index - 3]
                ^ schedule[index - 8]
                ^ schedule[index - 14]
                ^ schedule[index - 16])
                .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for (index, word) in schedule.into_iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    struct TestRepository {
        root: PathBuf,
    }

    impl TestRepository {
        fn new() -> Self {
            let unique = NEXT_TEMP.fetch_add(1, AtomicOrdering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "rustinfer-native-git-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("create test repository");
            fs::create_dir(root.join(".git")).expect("create .git");
            fs::create_dir_all(root.join(".git/refs/heads")).expect("create refs");
            fs::write(root.join(".git/HEAD"), b"ref: refs/heads/main\n").expect("write HEAD");
            fs::write(root.join(".git/refs/heads/main"), format!("{REVISION}\n"))
                .expect("write branch ref");
            Self { root }
        }

        fn write_file(&self, path: &str, contents: &[u8]) {
            let path = self.root.join(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create tracked parent");
            }
            fs::write(path, contents).expect("write tracked file");
        }

        fn write_index(&self, entries: &[TestIndexEntry], cached_entries: &[TestIndexEntry]) {
            self.write_index_without_ref_update(entries, cached_entries);
            self.touch_loose_ref();
        }

        fn write_index_without_ref_update(
            &self,
            entries: &[TestIndexEntry],
            cached_entries: &[TestIndexEntry],
        ) {
            fs::write(
                self.root.join(".git/index"),
                encode_index(entries, cached_entries),
            )
            .expect("write index");
        }

        fn touch_loose_ref(&self) {
            fs::write(
                self.root.join(".git/refs/heads/main"),
                format!("{REVISION}\n"),
            )
            .expect("refresh loose branch ref");
        }
    }

    impl Drop for TestRepository {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[derive(Clone)]
    struct TestIndexEntry {
        path: &'static str,
        mode: u32,
        contents: &'static [u8],
    }

    fn encode_index(entries: &[TestIndexEntry], cached_entries: &[TestIndexEntry]) -> Vec<u8> {
        let mut ordered = entries.to_vec();
        ordered.sort_by_key(|entry| entry.path);
        let mut bytes = Vec::from(*INDEX_SIGNATURE);
        bytes.extend_from_slice(&2_u32.to_be_bytes());
        bytes.extend_from_slice(
            &u32::try_from(ordered.len())
                .expect("test entry count fits u32")
                .to_be_bytes(),
        );
        for entry in &ordered {
            let start = bytes.len();
            bytes.extend_from_slice(&[0; 24]);
            bytes.extend_from_slice(&entry.mode.to_be_bytes());
            bytes.extend_from_slice(&[0; 12]);
            bytes.extend_from_slice(&hash_git_object(b"blob", entry.contents));
            let path = entry.path.as_bytes();
            let flags = u16::try_from(path.len().min(0x0fff)).expect("test path length fits u16");
            bytes.extend_from_slice(&flags.to_be_bytes());
            bytes.extend_from_slice(path);
            bytes.push(0);
            while (bytes.len() - start) % 8 != 0 {
                bytes.push(0);
            }
        }
        let cached: Vec<IndexEntry> = cached_entries
            .iter()
            .map(|entry| IndexEntry {
                path: entry.path.as_bytes().to_vec(),
                mode: entry.mode,
                object_id: hash_git_object(b"blob", entry.contents),
            })
            .collect();
        let tree = compute_index_tree(&cached).expect("compute test tree");
        let mut payload = Vec::new();
        encode_cache_tree_node(&mut payload, b"", &tree);
        bytes.extend_from_slice(CACHE_TREE_SIGNATURE);
        bytes.extend_from_slice(
            &u32::try_from(payload.len())
                .expect("test TREE length fits u32")
                .to_be_bytes(),
        );
        bytes.extend_from_slice(&payload);
        let checksum = Sha1::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn encode_cache_tree_node(output: &mut Vec<u8>, name: &[u8], tree: &ComputedTree) {
        output.extend_from_slice(name);
        output.push(0);
        output.extend_from_slice(
            format!("{} {}\n", tree.entry_count, tree.directories.len()).as_bytes(),
        );
        output.extend_from_slice(&tree.object_id);
        for (child_name, child) in &tree.directories {
            encode_cache_tree_node(output, child_name, child);
        }
    }

    fn tracked_files() -> [TestIndexEntry; 2] {
        [
            TestIndexEntry {
                path: "README.md",
                mode: 0o100_644,
                contents: b"clean repository\n",
            },
            TestIndexEntry {
                path: "src/lib.rs",
                mode: 0o100_644,
                contents: b"pub fn clean() {}\n",
            },
        ]
    }

    fn populate(repository: &TestRepository, entries: &[TestIndexEntry]) {
        for entry in entries {
            repository.write_file(entry.path, entry.contents);
        }
        repository.write_index(entries, entries);
    }

    fn replace_index_version(bytes: &mut [u8], version: u32) {
        bytes[4..8].copy_from_slice(&version.to_be_bytes());
        let checksum_start = bytes.len() - INDEX_CHECKSUM_BYTES;
        let checksum = Sha1::digest(&bytes[..checksum_start]);
        bytes[checksum_start..].copy_from_slice(&checksum);
    }

    #[test]
    fn sha1_matches_standard_vectors_and_git_blob_format() {
        assert_eq!(
            hex(&Sha1::digest(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            hex(&Sha1::digest(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex(&hash_git_object(b"blob", b"test\n")),
            "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"
        );
    }

    #[test]
    fn accepts_clean_v2_index_and_only_reviewed_untracked_exceptions() {
        let repository = TestRepository::new();
        let entries = tracked_files();
        populate(&repository, &entries);
        repository.write_file("target/debug/output", b"ignored");
        repository.write_file("benchmarks/results/evidence.json", b"ignored");

        let provenance = require_clean_repository(&repository.root).expect("clean repository");
        assert_eq!(provenance.revision, REVISION);
        assert_eq!(provenance.status_sha256, EMPTY_STATUS_SHA256);
    }

    #[test]
    fn rejects_detached_head_and_packed_branch_ref() {
        let detached = TestRepository::new();
        let entries = tracked_files();
        populate(&detached, &entries);
        fs::write(detached.root.join(".git/HEAD"), format!("{REVISION}\n"))
            .expect("write detached HEAD");
        assert!(matches!(
            require_clean_repository(&detached.root),
            Err(GitProvenanceError::UnsupportedRepository(_))
        ));

        let packed = TestRepository::new();
        populate(&packed, &entries);
        fs::remove_file(packed.root.join(".git/refs/heads/main")).expect("remove loose ref");
        fs::write(
            packed.root.join(".git/packed-refs"),
            format!("# pack-refs with: peeled fully-peeled sorted\n{REVISION} refs/heads/main\n"),
        )
        .expect("write packed refs");
        assert!(matches!(
            require_clean_repository(&packed.root),
            Err(GitProvenanceError::UnsupportedRepository(_))
        ));
    }

    #[test]
    fn accepts_v3_index_and_relative_gitdir_indirection() {
        let repository = TestRepository::new();
        let entries = tracked_files();
        for entry in &entries {
            repository.write_file(entry.path, entry.contents);
        }
        let mut index = encode_index(&entries, &entries);
        replace_index_version(&mut index, 3);
        fs::write(repository.root.join(".git/index"), index).expect("write v3 index");
        repository.touch_loose_ref();

        fs::create_dir(repository.root.join("target")).expect("create target");
        fs::rename(
            repository.root.join(".git"),
            repository.root.join("target/git-metadata"),
        )
        .expect("move git metadata");
        fs::write(
            repository.root.join(".git"),
            b"gitdir: target/git-metadata\n",
        )
        .expect("write gitdir indirection");

        let provenance =
            require_clean_repository(&repository.root).expect("clean v3 gitdir repository");
        assert_eq!(provenance.revision, REVISION);
    }

    #[test]
    fn rejects_modified_deleted_and_untracked_paths() {
        let modified = TestRepository::new();
        let entries = tracked_files();
        populate(&modified, &entries);
        modified.write_file("README.md", b"modified\n");
        assert!(matches!(
            require_clean_repository(&modified.root),
            Err(GitProvenanceError::DirtyRepository(_))
        ));

        let deleted = TestRepository::new();
        populate(&deleted, &entries);
        fs::remove_file(deleted.root.join("README.md")).expect("delete tracked file");
        assert!(matches!(
            require_clean_repository(&deleted.root),
            Err(GitProvenanceError::DirtyRepository(_))
        ));

        let untracked = TestRepository::new();
        populate(&untracked, &entries);
        untracked.write_file("artifacts/candidate.json", b"untracked\n");
        assert!(matches!(
            require_clean_repository(&untracked.root),
            Err(GitProvenanceError::DirtyRepository(_))
        ));
    }

    #[test]
    fn rejects_add_and_write_tree_indexes_newer_than_loose_head_ref() {
        let added = TestRepository::new();
        let clean_entries = tracked_files();
        populate(&added, &clean_entries);
        let mut staged_entries = clean_entries.clone();
        staged_entries[0].contents = b"staged contents\n";
        added.write_file(staged_entries[0].path, staged_entries[0].contents);
        std::thread::sleep(std::time::Duration::from_millis(20));
        added.write_index_without_ref_update(&staged_entries, &clean_entries);
        assert!(matches!(
            require_clean_repository(&added.root),
            Err(GitProvenanceError::DirtyRepository(_))
        ));

        let written_tree = TestRepository::new();
        populate(&written_tree, &clean_entries);
        written_tree.write_file(staged_entries[0].path, staged_entries[0].contents);
        std::thread::sleep(std::time::Duration::from_millis(20));
        written_tree.write_index_without_ref_update(&staged_entries, &staged_entries);
        let error = require_clean_repository(&written_tree.root)
            .expect_err("write-tree-style staged index must be rejected");
        assert!(error.to_string().contains("index is newer"));
    }

    #[test]
    fn rejects_noncanonical_revision() {
        let clean_entries = tracked_files();
        let uppercase = TestRepository::new();
        populate(&uppercase, &clean_entries);
        fs::write(
            uppercase.root.join(".git/refs/heads/main"),
            format!("{}\n", REVISION.to_ascii_uppercase()),
        )
        .expect("write uppercase ref");
        assert!(matches!(
            require_clean_repository(&uppercase.root),
            Err(GitProvenanceError::UnsupportedRepository(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn verifies_symlink_contents_and_executable_mode() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let repository = TestRepository::new();
        repository.write_file("script.sh", b"#!/bin/sh\nexit 0\n");
        let mut permissions = fs::metadata(repository.root.join("script.sh"))
            .expect("stat script")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(repository.root.join("script.sh"), permissions)
            .expect("make script executable");
        symlink("script.sh", repository.root.join("script-link")).expect("create symlink");
        let entries = [
            TestIndexEntry {
                path: "script-link",
                mode: 0o120_000,
                contents: b"script.sh",
            },
            TestIndexEntry {
                path: "script.sh",
                mode: 0o100_755,
                contents: b"#!/bin/sh\nexit 0\n",
            },
        ];
        repository.write_index(&entries, &entries);
        require_clean_repository(&repository.root).expect("clean mode and symlink");

        fs::remove_file(repository.root.join("script-link")).expect("remove symlink");
        symlink("missing", repository.root.join("script-link")).expect("replace symlink");
        assert!(matches!(
            require_clean_repository(&repository.root),
            Err(GitProvenanceError::DirtyRepository(_))
        ));
    }

    fn hex(bytes: &[u8]) -> String {
        use std::fmt::Write;

        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(&mut output, "{byte:02x}").expect("write hex");
        }
        output
    }
}
