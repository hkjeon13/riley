use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_CUDA_ARCHITECTURES: &str = "89";

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    if let Err(error) = build_native_cuda() {
        eprintln!("error: rustinfer-cuda native build failed: {error}");
        std::process::exit(1);
    }
}

fn build_native_cuda() -> Result<(), String> {
    for variable in [
        "CUDAToolkit_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVCC",
        "CUDACXX",
        "CMAKE",
        "RUSTINFER_CUDA_ARCHITECTURES",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let manifest_dir = required_path("CARGO_MANIFEST_DIR")?;
    let kernels_dir = manifest_dir.join("../../kernels");
    let cmake_lists = kernels_dir.join("CMakeLists.txt");
    if !cmake_lists.is_file() {
        return Err(format!(
            "native source root is incomplete: expected {}",
            cmake_lists.display()
        ));
    }

    for source in [
        cmake_lists,
        kernels_dir.join("include/rustinfer_cuda.h"),
        kernels_dir.join("src/version.cu"),
    ] {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    let toolkit = discover_cuda_toolkit()?;
    let cmake = discover_executable("CMAKE", "cmake")?;
    let architectures = cuda_architectures()?;
    let profile = cmake_profile();
    let out_dir = required_path("OUT_DIR")?;
    let build_dir = out_dir.join("cuda-native-build");
    let install_dir = out_dir.join("cuda-native-install");

    println!(
        "cargo:warning=rustinfer-cuda: CUDA toolkit={} nvcc={}",
        toolkit.root.display(),
        toolkit.nvcc.display()
    );
    println!(
        "cargo:warning=rustinfer-cuda: CUDA AOT architectures={architectures} (compile targets only; no runtime device detection)"
    );

    let mut configure = Command::new(&cmake);
    configure
        .arg("-S")
        .arg(&kernels_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={profile}"))
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={architectures}"))
        .arg(format!("-DCUDAToolkit_ROOT={}", toolkit.root.display()))
        .arg(format!("-DCMAKE_CUDA_COMPILER={}", toolkit.nvcc.display()));
    run(&mut configure, "configure the native CUDA library")?;

    let cudart_link_dir = discover_dynamic_cudart(&build_dir, profile, &toolkit)?;

    let mut build = Command::new(&cmake);
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile)
        .arg("--target")
        .arg("rustinfer_cuda_native")
        .arg("--parallel");
    run(&mut build, "compile the native CUDA library")?;

    let mut install = Command::new(&cmake);
    install
        .arg("--install")
        .arg(&build_dir)
        .arg("--config")
        .arg(profile);
    run(&mut install, "install the native CUDA library")?;

    let native_lib_dir = install_dir.join("lib");
    let installed_library = native_lib_dir.join(static_library_filename());
    if !installed_library.is_file() {
        return Err(format!(
            "CMake completed without the expected static library {}; inspect the CMake install output",
            installed_library.display()
        ));
    }

    println!(
        "cargo:rustc-link-search=native={}",
        native_lib_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cudart_link_dir.display()
    );
    println!("cargo:rustc-link-lib=static=rustinfer_cuda_native");
    // nvcc emits fatbinary registration calls even for the PR 02 host-only
    // `.cu` translation unit. Use the toolkit's shared CUDA Runtime both to
    // satisfy those symbols and to preserve the runtime strategy needed by
    // later host-runtime PRs. The release environment must provide cudart.
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}

struct CudaToolkit {
    root: PathBuf,
    nvcc: PathBuf,
}

fn discover_dynamic_cudart(
    build_dir: &Path,
    profile: &str,
    toolkit: &CudaToolkit,
) -> Result<PathBuf, String> {
    let metadata = build_dir.join(format!("rustinfer-cuda-cudart-{profile}.path"));
    let contents = fs::read_to_string(&metadata).map_err(|error| {
        format!(
            "CMake did not produce CUDA Runtime link metadata at {}: {error}; ensure the selected toolkit includes the cudart development library",
            metadata.display()
        )
    })?;
    let linker_path = contents.trim();
    if linker_path.is_empty() || linker_path.lines().count() != 1 {
        return Err(format!(
            "invalid CUDA Runtime link metadata in {}: expected one non-empty path",
            metadata.display()
        ));
    }

    let linker_path = PathBuf::from(linker_path);
    if !linker_path.is_absolute() || !linker_path.is_file() {
        return Err(format!(
            "CMake selected CUDA Runtime linker file {}, but it is not an absolute existing file",
            linker_path.display()
        ));
    }

    let canonical_linker = linker_path.canonicalize().map_err(|error| {
        format!(
            "cannot resolve CUDA Runtime linker file {}: {error}",
            linker_path.display()
        )
    })?;
    if linker_path
        .components()
        .any(|component| component.as_os_str() == "stubs")
    {
        return Err(format!(
            "CMake selected CUDA Runtime linker file {} from a stubs directory; select the real shared CUDA Runtime development library",
            linker_path.display()
        ));
    }
    if !canonical_linker.starts_with(&toolkit.root) {
        return Err(format!(
            "CMake selected CUDA Runtime {} outside the nvcc toolkit root {}; clear the CMake cache and select one CUDA toolkit",
            canonical_linker.display(),
            toolkit.root.display()
        ));
    }

    let link_dir = linker_path.parent().ok_or_else(|| {
        format!(
            "CUDA Runtime linker file {} has no parent directory",
            linker_path.display()
        )
    })?;
    let expected_linker = link_dir.join(dynamic_cudart_filename());
    if !expected_linker.is_file() {
        return Err(format!(
            "CUDA Runtime development linker file {} is missing; install the complete CUDA toolkit rather than a runtime-only package",
            expected_linker.display()
        ));
    }

    println!(
        "cargo:warning=rustinfer-cuda: CUDA Runtime strategy=shared linker={}",
        expected_linker.display()
    );
    Ok(link_dir.to_path_buf())
}

fn discover_cuda_toolkit() -> Result<CudaToolkit, String> {
    let mut configured_roots = Vec::new();
    for variable in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = nonempty_env(variable)? {
            let root = canonical_directory(Path::new(&value), variable)?;
            configured_roots.push((variable, root));
        }
    }

    if let Some((first_name, first_root)) = configured_roots.first() {
        for (name, root) in configured_roots.iter().skip(1) {
            if root != first_root {
                return Err(format!(
                    "conflicting CUDA toolkit roots: {first_name}={} but {name}={}; set them to one toolkit",
                    first_root.display(),
                    root.display()
                ));
            }
        }

        let nvcc = nvcc_under(first_root).ok_or_else(|| {
            format!(
                "{first_name}={} does not contain bin/{}; point it at a complete CUDA toolkit root",
                first_root.display(),
                nvcc_filename()
            )
        })?;
        validate_explicit_compiler_matches(&nvcc)?;
        return Ok(CudaToolkit {
            root: first_root.clone(),
            nvcc,
        });
    }

    let nvcc = if let Some(value) = nonempty_env("NVCC")? {
        resolve_executable(&value).ok_or_else(|| {
            format!(
                "NVCC={} is not an executable file or discoverable on PATH",
                Path::new(&value).display()
            )
        })?
    } else if let Some(value) = nonempty_env("CUDACXX")? {
        resolve_executable(&value).ok_or_else(|| {
            format!(
                "CUDACXX={} is not an executable file or discoverable on PATH",
                Path::new(&value).display()
            )
        })?
    } else {
        find_on_path(nvcc_filename()).ok_or_else(|| {
            "CUDA feature requested but nvcc was not found; set CUDAToolkit_ROOT (preferred), CUDA_HOME, NVCC, or add nvcc to PATH".to_owned()
        })?
    };

    let nvcc = canonical_file(&nvcc, "nvcc")?;
    let root = nvcc
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| format!("cannot derive a CUDA toolkit root from {}", nvcc.display()))?
        .to_path_buf();
    Ok(CudaToolkit { root, nvcc })
}

fn validate_explicit_compiler_matches(discovered_nvcc: &Path) -> Result<(), String> {
    for variable in ["NVCC", "CUDACXX"] {
        if let Some(value) = nonempty_env(variable)? {
            let explicit = resolve_executable(&value).ok_or_else(|| {
                format!(
                    "{variable}={} is not an executable file or discoverable on PATH",
                    Path::new(&value).display()
                )
            })?;
            let explicit = canonical_file(&explicit, variable)?;
            let discovered = canonical_file(discovered_nvcc, "toolkit nvcc")?;
            if explicit != discovered {
                return Err(format!(
                    "conflicting CUDA compilers: toolkit selects {} but {variable} selects {}; use one toolkit",
                    discovered.display(),
                    explicit.display()
                ));
            }
        }
    }
    Ok(())
}

fn cuda_architectures() -> Result<String, String> {
    let raw = match nonempty_env("RUSTINFER_CUDA_ARCHITECTURES")? {
        Some(value) => value
            .into_string()
            .map_err(|_| "RUSTINFER_CUDA_ARCHITECTURES must be valid UTF-8".to_owned())?,
        None => DEFAULT_CUDA_ARCHITECTURES.to_owned(),
    };
    let normalized = raw.replace(',', ";");
    let tokens: Vec<_> = normalized.split(';').map(str::trim).collect();
    if tokens.is_empty() || tokens.iter().any(|token| token.is_empty()) {
        return Err(
            "RUSTINFER_CUDA_ARCHITECTURES must be a non-empty comma- or semicolon-separated architecture list such as 80;89"
                .to_owned(),
        );
    }
    for token in &tokens {
        let numeric = token
            .strip_suffix("-real")
            .or_else(|| token.strip_suffix("-virtual"))
            .unwrap_or(token);
        if numeric.is_empty() || !numeric.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "invalid CUDA architecture {token:?}; use numeric AOT targets such as 80, 89, 90-real, or 90-virtual (runtime-dependent 'native' is intentionally unsupported)"
            ));
        }
    }
    Ok(tokens.join(";"))
}

fn discover_executable(variable: &str, fallback: &str) -> Result<PathBuf, String> {
    if let Some(value) = nonempty_env(variable)? {
        return resolve_executable(&value).ok_or_else(|| {
            format!(
                "{variable}={} is not an executable file or discoverable on PATH",
                Path::new(&value).display()
            )
        });
    }
    find_on_path(fallback).ok_or_else(|| {
        format!(
            "required build tool {fallback:?} was not found; install CMake or set {variable} to its executable"
        )
    })
}

fn nonempty_env(variable: &str) -> Result<Option<OsString>, String> {
    match env::var_os(variable) {
        Some(value) if value.is_empty() => Err(format!(
            "{variable} is set but empty; unset it or provide a valid value"
        )),
        value => Ok(value),
    }
}

fn required_path(variable: &str) -> Result<PathBuf, String> {
    nonempty_env(variable)?.map(PathBuf::from).ok_or_else(|| {
        format!("Cargo did not provide the required build environment variable {variable}")
    })
}

fn canonical_directory(path: &Path, variable: &str) -> Result<PathBuf, String> {
    if !path.is_dir() {
        return Err(format!("{variable}={} is not a directory", path.display()));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot resolve {variable}={}: {error}", path.display()))
}

fn canonical_file(path: &Path, description: &str) -> Result<PathBuf, String> {
    if !path.is_file() {
        return Err(format!("{description}={} is not a file", path.display()));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", path.display()))
}

fn resolve_executable(value: &OsStr) -> Option<PathBuf> {
    let path = PathBuf::from(value);
    if path.components().count() > 1 || path.is_absolute() {
        path.is_file().then_some(path)
    } else {
        find_on_path(value)
    }
}

fn find_on_path(program: impl AsRef<OsStr>) -> Option<PathBuf> {
    let program = program.as_ref();
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .flat_map(|directory| executable_candidates(&directory, program))
            .find(|candidate| candidate.is_file())
    })
}

fn executable_candidates(directory: &Path, program: &OsStr) -> Vec<PathBuf> {
    let direct = directory.join(program);
    #[cfg(windows)]
    {
        if direct.extension().is_some() {
            return vec![direct];
        }
        let extensions =
            env::var_os("PATHEXT").unwrap_or_else(|| OsString::from(".COM;.EXE;.BAT;.CMD"));
        extensions
            .to_string_lossy()
            .split(';')
            .filter(|extension| !extension.is_empty())
            .map(|extension| directory.join(format!("{}{}", program.to_string_lossy(), extension)))
            .collect()
    }
    #[cfg(not(windows))]
    {
        vec![direct]
    }
}

fn nvcc_under(toolkit_root: &Path) -> Option<PathBuf> {
    let candidate = toolkit_root.join("bin").join(nvcc_filename());
    candidate.is_file().then_some(candidate)
}

fn nvcc_filename() -> &'static str {
    if cfg!(windows) { "nvcc.exe" } else { "nvcc" }
}

fn cmake_profile() -> &'static str {
    match env::var("PROFILE").as_deref() {
        Ok("release" | "bench") => "Release",
        _ => "Debug",
    }
}

fn static_library_filename() -> &'static str {
    if cfg!(windows) {
        "rustinfer_cuda_native.lib"
    } else {
        "librustinfer_cuda_native.a"
    }
}

fn dynamic_cudart_filename() -> &'static str {
    if cfg!(windows) {
        "cudart.lib"
    } else if cfg!(target_os = "macos") {
        "libcudart.dylib"
    } else {
        "libcudart.so"
    }
}

fn run(command: &mut Command, action: &str) -> Result<(), String> {
    let rendered = format!("{command:?}");
    let status = command
        .status()
        .map_err(|error| format!("could not {action} with {rendered}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "could not {action}: {rendered} exited with {status}"
        ))
    }
}
