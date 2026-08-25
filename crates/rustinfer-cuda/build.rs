use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_CUDA_ARCHITECTURES: &str = "89";

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTINFER_CUDA_ARCHITECTURES");
    let architectures = match cuda_architectures() {
        Ok(architectures) => architectures,
        Err(error) => {
            eprintln!("error: rustinfer-cuda architecture configuration failed: {error}");
            std::process::exit(1);
        }
    };
    println!("cargo:rustc-env=RUSTINFER_CUDA_COMPILED_ARCHITECTURES={architectures}");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    if let Err(error) = build_native_cuda(&architectures) {
        eprintln!("error: rustinfer-cuda native build failed: {error}");
        std::process::exit(1);
    }
}

fn build_native_cuda(architectures: &str) -> Result<(), String> {
    for variable in [
        "CUDAToolkit_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVCC",
        "CUDACXX",
        "CMAKE",
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

    emit_native_rerun_inputs(&kernels_dir, cmake_lists);

    let toolkit = discover_cuda_toolkit()?;
    let cmake = discover_executable("CMAKE", "cmake")?;
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

    let cublaslt_link_dir = discover_dynamic_cublaslt(&build_dir, profile, &toolkit)?;
    let cudart_link_dir = discover_dynamic_cudart(&build_dir, profile, &toolkit)?;
    let cuda_driver_link_dir = discover_cuda_driver(&build_dir, profile)?;

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
        cublaslt_link_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cudart_link_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_driver_link_dir.display()
    );
    println!("cargo:rustc-link-lib=static=rustinfer_cuda_native");
    // The static adapter archive retains cuBLASLt calls. Forward the exact
    // toolkit-selected shared library after the archive so its symbols are
    // resolved without accidentally accepting a driver stub or another CUDA
    // installation found earlier on the host search path.
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    // nvcc emits fatbinary registration calls for the AOT CUDA translation
    // units. Use the selected toolkit's shared CUDA Runtime to satisfy those
    // symbols; the release environment must provide cudart.
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    Ok(())
}

fn emit_native_rerun_inputs(kernels_dir: &Path, cmake_lists: PathBuf) {
    for source in [
        cmake_lists,
        kernels_dir.join("include/rustinfer_cuda.h"),
        kernels_dir.join("src/ffi_internal.hpp"),
        kernels_dir.join("src/attention_online.cu"),
        kernels_dir.join("src/attention_online.hpp"),
        kernels_dir.join("src/attention_reference.cu"),
        kernels_dir.join("src/decode_attention.cu"),
        kernels_dir.join("src/gemm.cu"),
        kernels_dir.join("src/host_runtime.cu"),
        kernels_dir.join("src/memory.cu"),
        kernels_dir.join("src/primitives.cu"),
        kernels_dir.join("src/smoke_fill.cu"),
        kernels_dir.join("src/version.cu"),
        kernels_dir.join("tests/abi_layout.c"),
    ] {
        println!("cargo:rerun-if-changed={}", source.display());
    }
}

fn discover_dynamic_cublaslt(
    build_dir: &Path,
    profile: &str,
    toolkit: &CudaToolkit,
) -> Result<PathBuf, String> {
    let metadata = build_dir.join(format!("rustinfer-cuda-cublaslt-{profile}.path"));
    let contents = fs::read_to_string(&metadata).map_err(|error| {
        format!(
            "CMake did not produce cuBLASLt link metadata at {}: {error}; ensure the selected toolkit includes the shared cuBLASLt development library",
            metadata.display()
        )
    })?;
    let linker_path = contents.trim();
    if linker_path.is_empty() || linker_path.lines().count() != 1 {
        return Err(format!(
            "invalid cuBLASLt link metadata in {}: expected one non-empty path",
            metadata.display()
        ));
    }

    let linker_path = PathBuf::from(linker_path);
    if !linker_path.is_absolute() || !linker_path.is_file() {
        return Err(format!(
            "CMake selected cuBLASLt linker file {}, but it is not an absolute existing file",
            linker_path.display()
        ));
    }
    if !is_dynamic_cublaslt_path(&linker_path) {
        return Err(format!(
            "CMake selected {}, which is not a shared cuBLASLt linker file; static cuBLASLt is unsupported by this Cargo link contract",
            linker_path.display()
        ));
    }
    if linker_path
        .components()
        .any(|component| component.as_os_str() == "stubs")
    {
        return Err(format!(
            "CMake selected cuBLASLt linker file {} from a stubs directory; select the real shared cuBLASLt library",
            linker_path.display()
        ));
    }

    let canonical_linker = linker_path.canonicalize().map_err(|error| {
        format!(
            "cannot resolve cuBLASLt linker file {}: {error}",
            linker_path.display()
        )
    })?;
    if canonical_linker
        .components()
        .any(|component| component.as_os_str() == "stubs")
    {
        return Err(format!(
            "CMake selected cuBLASLt linker file {} that resolves through a stubs directory",
            linker_path.display()
        ));
    }
    if !canonical_linker.starts_with(&toolkit.root) {
        return Err(format!(
            "CMake selected cuBLASLt {} outside the nvcc toolkit root {}; clear the CMake cache and select one CUDA toolkit",
            canonical_linker.display(),
            toolkit.root.display()
        ));
    }

    let link_dir = linker_path.parent().ok_or_else(|| {
        format!(
            "cuBLASLt linker file {} has no parent directory",
            linker_path.display()
        )
    })?;
    let expected_linker = link_dir.join(dynamic_cublaslt_filename());
    if !expected_linker.is_file() {
        return Err(format!(
            "cuBLASLt shared development linker file {} is missing; install the complete CUDA toolkit rather than a static- or runtime-only package",
            expected_linker.display()
        ));
    }
    validate_cublaslt_development_linker(&expected_linker, &canonical_linker, toolkit)?;

    println!(
        "cargo:warning=rustinfer-cuda: cuBLASLt strategy=shared linker={}",
        expected_linker.display()
    );
    Ok(link_dir.to_path_buf())
}

fn validate_cublaslt_development_linker(
    linker: &Path,
    cmake_selection: &Path,
    toolkit: &CudaToolkit,
) -> Result<(), String> {
    let canonical = linker.canonicalize().map_err(|error| {
        format!(
            "cannot resolve cuBLASLt development linker file {}: {error}",
            linker.display()
        )
    })?;
    if canonical
        .components()
        .any(|component| component.as_os_str() == "stubs")
        || !canonical.starts_with(&toolkit.root)
    {
        return Err(format!(
            "cuBLASLt development linker {} resolves outside the selected real toolkit {}",
            linker.display(),
            toolkit.root.display()
        ));
    }
    if canonical != cmake_selection {
        return Err(format!(
            "cuBLASLt development linker {} resolves to {}, but CMake selected {}; refusing mixed libraries",
            linker.display(),
            canonical.display(),
            cmake_selection.display()
        ));
    }
    Ok(())
}

fn discover_cuda_driver(build_dir: &Path, profile: &str) -> Result<PathBuf, String> {
    let metadata = build_dir.join(format!("rustinfer-cuda-driver-{profile}.path"));
    let contents = fs::read_to_string(&metadata).map_err(|error| {
        format!(
            "CMake did not produce CUDA Driver link metadata at {}: {error}; install the CUDA driver development linker or toolkit stubs",
            metadata.display()
        )
    })?;
    let linker_path = contents.trim();
    if linker_path.is_empty() || linker_path.lines().count() != 1 {
        return Err(format!(
            "invalid CUDA Driver link metadata in {}: expected one non-empty path",
            metadata.display()
        ));
    }
    let linker_path = PathBuf::from(linker_path);
    if !linker_path.is_absolute() || !linker_path.is_file() {
        return Err(format!(
            "CMake selected CUDA Driver linker file {}, but it is not an absolute existing file",
            linker_path.display()
        ));
    }
    let link_dir = linker_path.parent().ok_or_else(|| {
        format!(
            "CUDA Driver linker file {} has no parent directory",
            linker_path.display()
        )
    })?;
    let expected_linker = link_dir.join(dynamic_cuda_driver_filename());
    if !expected_linker.is_file() {
        return Err(format!(
            "CUDA Driver development linker file {} is missing; install the NVIDIA driver development package or complete toolkit stubs",
            expected_linker.display()
        ));
    }
    println!(
        "cargo:warning=rustinfer-cuda: CUDA Driver linker={}",
        expected_linker.display()
    );
    Ok(link_dir.to_path_buf())
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
    let mut canonical = Vec::with_capacity(tokens.len());
    for token in tokens {
        let (numeric, suffix) = if let Some(numeric) = token.strip_suffix("-real") {
            (numeric, "-real")
        } else if let Some(numeric) = token.strip_suffix("-virtual") {
            (numeric, "-virtual")
        } else {
            (token, "")
        };
        let architecture = numeric.parse::<u32>().map_err(|_| {
            format!(
                "invalid CUDA architecture {token:?}; use numeric targets such as 80, 89, 90-real, or 90-virtual (runtime-dependent 'native' is intentionally unsupported)"
            )
        })?;
        if architecture < 10 {
            return Err(format!(
                "invalid CUDA architecture {token:?}; the target must encode a major and minor compute capability"
            ));
        }
        let canonical_token = format!("{architecture}{suffix}");
        if !canonical.contains(&canonical_token) {
            canonical.push(canonical_token);
        }
    }
    Ok(canonical.join(";"))
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

fn dynamic_cublaslt_filename() -> &'static str {
    if cfg!(windows) {
        "cublasLt.lib"
    } else if cfg!(target_os = "macos") {
        "libcublasLt.dylib"
    } else {
        "libcublasLt.so"
    }
}

fn is_dynamic_cublaslt_path(path: &Path) -> bool {
    let Some(filename) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    if cfg!(windows) {
        filename.eq_ignore_ascii_case("cublasLt.lib")
    } else if cfg!(target_os = "macos") {
        filename.starts_with("libcublasLt")
            && path
                .extension()
                .and_then(OsStr::to_str)
                .is_some_and(|extension| extension.eq_ignore_ascii_case("dylib"))
    } else {
        filename == "libcublasLt.so" || filename.starts_with("libcublasLt.so.")
    }
}

fn dynamic_cuda_driver_filename() -> &'static str {
    if cfg!(windows) {
        "cuda.lib"
    } else if cfg!(target_os = "macos") {
        "libcuda.dylib"
    } else {
        "libcuda.so"
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
