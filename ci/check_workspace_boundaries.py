#!/usr/bin/env python3
"""Fail closed when the PR 02 production workspace boundary drifts.

This checker deliberately uses only the Python standard library. Python is a
CI inspection tool here; it is not invoked by Cargo or shipped with a
production artifact. The Python-free CUDA image performs its own shell-only
artifact checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any


EXPECTED_CRATES = {
    "crates/rustinfer-core": "rustinfer-core",
    "crates/rustinfer-cuda": "rustinfer-cuda",
    "crates/rustinfer-tensor": "rustinfer-tensor",
    "crates/rustinfer-model": "rustinfer-model",
    "crates/rustinfer-runtime": "rustinfer-runtime",
    "crates/rustinfer-scheduler": "rustinfer-scheduler",
    "crates/rustinfer-server": "rustinfer-server",
}

EXPECTED_INTERNAL_DEPENDENCIES = {
    "rustinfer-core": set(),
    "rustinfer-cuda": {"rustinfer-core"},
    "rustinfer-tensor": {"rustinfer-core", "rustinfer-cuda"},
    "rustinfer-model": {"rustinfer-core", "rustinfer-tensor"},
    "rustinfer-runtime": {
        "rustinfer-core",
        "rustinfer-cuda",
        "rustinfer-model",
        "rustinfer-tensor",
    },
    "rustinfer-scheduler": {"rustinfer-core", "rustinfer-runtime"},
    "rustinfer-server": {"rustinfer-core", "rustinfer-scheduler"},
}

EXPECTED_FEATURES = {
    "rustinfer-core": {"default": []},
    "rustinfer-cuda": {"cuda": [], "default": []},
    "rustinfer-tensor": {
        "cuda": ["dep:rustinfer-cuda", "rustinfer-cuda/cuda"],
        "default": [],
    },
    "rustinfer-model": {"default": []},
    "rustinfer-runtime": {
        "cuda": [
            "dep:rustinfer-cuda",
            "rustinfer-cuda/cuda",
            "rustinfer-tensor/cuda",
        ],
        "default": [],
    },
    "rustinfer-scheduler": {
        "cuda": ["rustinfer-runtime/cuda"],
        "default": [],
    },
    "rustinfer-server": {
        "bench": [],
        "cuda": ["rustinfer-scheduler/cuda"],
        "default": [],
        "experimental": [],
        "server": [],
    },
}

EXPECTED_OPTIONAL_DEPENDENCIES = {
    "rustinfer-core": set(),
    "rustinfer-cuda": set(),
    "rustinfer-tensor": {"rustinfer-cuda"},
    "rustinfer-model": set(),
    "rustinfer-runtime": {"rustinfer-cuda"},
    "rustinfer-scheduler": set(),
    "rustinfer-server": set(),
}

EXPECTED_PROFILES = {
    "dev": {"debug": 2, "overflow-checks": True, "panic": "unwind"},
    "test": {"debug": 2, "overflow-checks": True},
    "release": {
        "opt-level": 3,
        "debug": "line-tables-only",
        "lto": "thin",
        "codegen-units": 1,
        "overflow-checks": False,
        "panic": "abort",
        "strip": "none",
    },
}

EXPECTED_LINTS = {
    "rust": {"unsafe_code": "deny"},
    "clippy": {
        "all": {"level": "warn", "priority": -1},
        "pedantic": {"level": "warn", "priority": -1},
        "module_name_repetitions": "allow",
    },
}

EXPECTED_DEFAULT_MEMBERS = ["crates/rustinfer-server"]
EXPECTED_EXCLUDES = ["tools/python", "tools/native", "experiments/triton"]
FORBIDDEN_PRODUCTION_FEATURES = {"python", "pytorch", "torch", "transformers", "triton"}
FORBIDDEN_BUILD_COMMAND = re.compile(
    r"Command\s*::\s*new\s*\(\s*[br#]*[\"'](?:python(?:3(?:\.\d+)*)?|triton)[\"']",
    re.IGNORECASE,
)


class BoundaryError(RuntimeError):
    """A production workspace invariant was violated."""


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BoundaryError(f"cannot read TOML {path}: {error}") from error


def normalized_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise BoundaryError(f"path escapes repository root: {path}") from error


def cargo_metadata(root: Path, locked: bool) -> dict[str, Any]:
    command = ["cargo", "metadata", "--format-version", "1", "--no-deps"]
    if locked:
        command.append("--locked")
    process = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise BoundaryError(f"{' '.join(command)} failed: {detail}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise BoundaryError(f"cargo metadata emitted invalid JSON: {error}") from error


def validate_root_manifest(root: Path) -> None:
    manifest = load_toml(root / "Cargo.toml")
    workspace = manifest.get("workspace", {})

    if workspace.get("resolver") != "3":
        raise BoundaryError("workspace.resolver must remain 3")

    members = workspace.get("members")
    if members != list(EXPECTED_CRATES):
        raise BoundaryError(
            "workspace.members must be the seven ordered production crates; "
            f"expected {list(EXPECTED_CRATES)!r}, found {members!r}"
        )
    if workspace.get("default-members") != EXPECTED_DEFAULT_MEMBERS:
        raise BoundaryError(
            "workspace.default-members must make rustinfer-server the root build owner"
        )
    if workspace.get("exclude") != EXPECTED_EXCLUDES:
        raise BoundaryError(
            "workspace.exclude must be exactly tools/python, tools/native, and "
            "experiments/triton"
        )

    package_policy = workspace.get("package", {})
    expected_package_policy = {
        "version": "0.1.0",
        "edition": "2024",
        "rust-version": "1.85",
        "publish": False,
    }
    for key, expected in expected_package_policy.items():
        if package_policy.get(key) != expected:
            raise BoundaryError(
                f"workspace.package.{key} must be {expected!r}, found "
                f"{package_policy.get(key)!r}"
            )
    if package_policy.get("publish") is not False:
        raise BoundaryError("workspace.package.publish must remain false")
    if "license" in package_policy or "license-file" in package_policy:
        raise BoundaryError(
            "PR 02 does not declare a repository license; do not add license metadata "
            "without an explicit repository licensing decision"
        )

    if workspace.get("lints") != EXPECTED_LINTS:
        raise BoundaryError(
            "workspace lint policy drifted; unsafe_code=deny and the shared Clippy "
            "policy are mandatory"
        )
    if manifest.get("profile") != EXPECTED_PROFILES:
        raise BoundaryError("debug/test/release profile policy drifted")

    toolchain = load_toml(root / "rust-toolchain.toml")
    if toolchain != {
        "toolchain": {
            "channel": "1.85.0",
            "profile": "minimal",
            "components": ["clippy", "rustfmt"],
        }
    }:
        raise BoundaryError(
            "rust-toolchain.toml must pin 1.85.0 with minimal, clippy, and rustfmt"
        )

    rustfmt = load_toml(root / "rustfmt.toml")
    if rustfmt != {
        "edition": "2024",
        "max_width": 100,
        "newline_style": "Unix",
        "use_small_heuristics": "Default",
    }:
        raise BoundaryError("rustfmt.toml policy drifted")


def validate_package_manifests(root: Path) -> None:
    for relative_dir, expected_name in EXPECTED_CRATES.items():
        manifest_path = root / relative_dir / "Cargo.toml"
        manifest = load_toml(manifest_path)
        package = manifest.get("package", {})
        if package.get("name") != expected_name:
            raise BoundaryError(
                f"{manifest_path}: expected package name {expected_name!r}"
            )
        if package.get("publish") != {"workspace": True}:
            raise BoundaryError(
                f"{manifest_path}: publish must inherit workspace publish=false"
            )
        if "license" in package or "license-file" in package:
            raise BoundaryError(
                f"{manifest_path}: package license metadata requires a repository "
                "licensing decision"
            )
        description = package.get("description")
        if not isinstance(description, str) or not description.strip():
            raise BoundaryError(f"{manifest_path}: package description is required")


def package_by_name(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise BoundaryError("cargo metadata is missing packages")
    result: dict[str, dict[str, Any]] = {}
    for package in packages:
        name = package.get("name")
        if not isinstance(name, str) or name in result:
            raise BoundaryError(f"invalid or duplicate package name in metadata: {name!r}")
        result[name] = package
    return result


def validate_members(root: Path, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages = package_by_name(metadata)
    expected_names = set(EXPECTED_CRATES.values())
    workspace_ids = metadata.get("workspace_members")
    if not isinstance(workspace_ids, list):
        raise BoundaryError("cargo metadata is missing workspace_members")

    id_to_package = {package.get("id"): package for package in packages.values()}
    try:
        members = [id_to_package[package_id] for package_id in workspace_ids]
    except KeyError as error:
        raise BoundaryError(f"unknown workspace package id: {error.args[0]}") from error

    member_names = {package["name"] for package in members}
    if member_names != expected_names or len(members) != len(expected_names):
        raise BoundaryError(
            f"workspace package set drifted: expected {sorted(expected_names)}, "
            f"found {sorted(member_names)}"
        )

    expected_by_name = {name: path for path, name in EXPECTED_CRATES.items()}
    for package in members:
        name = package["name"]
        manifest_dir = Path(package["manifest_path"]).resolve().parent
        relative_dir = normalized_relative(manifest_dir, root)
        if relative_dir != expected_by_name[name]:
            raise BoundaryError(
                f"{name}: expected at {expected_by_name[name]}, found {relative_dir}"
            )
        if package.get("publish") != []:
            raise BoundaryError(f"{name}: effective publish policy must be false")
        if package.get("edition") != "2024" or package.get("rust_version") != "1.85":
            raise BoundaryError(f"{name}: effective edition/rust-version policy drifted")
        if package.get("license") is not None or package.get("license_file") is not None:
            raise BoundaryError(
                f"{name}: effective license metadata requires a repository licensing decision"
            )

    default_ids = metadata.get("workspace_default_members")
    default_names = {
        id_to_package[package_id]["name"] for package_id in default_ids or []
    }
    if default_names != {"rustinfer-server"}:
        raise BoundaryError(
            "effective default workspace member must be rustinfer-server"
        )
    return {package["name"]: package for package in members}


def validate_dependencies(root: Path, packages: dict[str, dict[str, Any]]) -> None:
    production_names = set(packages)
    graph: dict[str, set[str]] = {}

    for name, package in packages.items():
        actual_internal: set[str] = set()
        for dependency in package.get("dependencies", []):
            dependency_name = dependency.get("name")
            if dependency_name not in production_names:
                raise BoundaryError(
                    f"{name}: third-party Cargo dependency {dependency_name!r} is not "
                    "permitted by the PR 02 zero-third-party license policy"
                )
            dependency_path_raw = dependency.get("path")
            if not isinstance(dependency_path_raw, str):
                raise BoundaryError(f"{name} -> {dependency_name}: must be a path dependency")
            if dependency.get("source") is not None:
                raise BoundaryError(f"{name} -> {dependency_name}: registry source is forbidden")
            if dependency.get("rename") is not None:
                raise BoundaryError(f"{name} -> {dependency_name}: dependency renames are forbidden")
            if dependency.get("kind") is not None or dependency.get("target") is not None:
                raise BoundaryError(
                    f"{name} -> {dependency_name}: must be an unconditional normal dependency"
                )
            if dependency.get("features") != [] or not dependency.get(
                "uses_default_features"
            ):
                raise BoundaryError(
                    f"{name} -> {dependency_name}: dependency-level feature policy drifted"
                )
            expected_optional = dependency_name in EXPECTED_OPTIONAL_DEPENDENCIES[name]
            if dependency.get("optional") is not expected_optional:
                raise BoundaryError(
                    f"{name} -> {dependency_name}: optional must be {expected_optional}"
                )
            dependency_path = Path(dependency_path_raw)
            relative = normalized_relative(dependency_path, root)
            expected_path = next(
                path for path, candidate in EXPECTED_CRATES.items() if candidate == dependency_name
            )
            if relative != expected_path:
                raise BoundaryError(
                    f"{name} -> {dependency_name}: path must resolve to {expected_path}, "
                    f"found {relative}"
                )
            actual_internal.add(dependency_name)

        if actual_internal != EXPECTED_INTERNAL_DEPENDENCIES[name]:
            raise BoundaryError(
                f"{name}: dependency boundary drifted; expected "
                f"{sorted(EXPECTED_INTERNAL_DEPENDENCIES[name])}, found "
                f"{sorted(actual_internal)}"
            )
        graph[name] = actual_internal

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise BoundaryError(f"production dependency cycle reaches {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for package_name in graph:
        visit(package_name)


def validate_features(packages: dict[str, dict[str, Any]]) -> None:
    for name, package in packages.items():
        features = package.get("features")
        if not isinstance(features, dict):
            raise BoundaryError(f"{name}: cargo metadata is missing features")
        if features != EXPECTED_FEATURES[name]:
            raise BoundaryError(
                f"{name}: feature propagation drifted; expected "
                f"{EXPECTED_FEATURES[name]!r}, found {features!r}"
            )
        forbidden = {feature.lower() for feature in features} & FORBIDDEN_PRODUCTION_FEATURES
        if forbidden:
            raise BoundaryError(
                f"{name}: forbidden production feature(s): {sorted(forbidden)}"
            )
        default = features.get("default")
        if default != []:
            raise BoundaryError(f"{name}: default features must remain empty")

    server_targets = packages["rustinfer-server"].get("targets", [])
    server_bins = [target for target in server_targets if "bin" in target.get("kind", [])]
    if len(server_bins) != 1:
        raise BoundaryError("rustinfer-server must own exactly one production binary")
    binary = server_bins[0]
    if binary.get("name") != "rustinfer" or binary.get("required-features") != ["server"]:
        raise BoundaryError(
            "the rustinfer binary must be owned by rustinfer-server and require only `server`"
        )
    for name, package in packages.items():
        if name == "rustinfer-server":
            continue
        if any("bin" in target.get("kind", []) for target in package.get("targets", [])):
            raise BoundaryError(f"{name}: only rustinfer-server may own a binary")


def validate_build_scripts(root: Path) -> None:
    for relative_dir in EXPECTED_CRATES:
        build_script = root / relative_dir / "build.rs"
        if not build_script.exists():
            continue
        source = build_script.read_text(encoding="utf-8")
        if FORBIDDEN_BUILD_COMMAND.search(source):
            raise BoundaryError(
                f"{build_script}: production build scripts may not invoke Python or Triton"
            )


def validate_lockfile(root: Path, packages: dict[str, dict[str, Any]]) -> None:
    lockfile = root / "Cargo.lock"
    lock = load_toml(lockfile)
    locked_packages = lock.get("package")
    if not isinstance(locked_packages, list):
        raise BoundaryError("Cargo.lock does not contain a package list")
    locked_names = [package.get("name") for package in locked_packages]
    expected_names = set(packages)
    if set(locked_names) != expected_names or len(locked_names) != len(expected_names):
        extra = sorted(str(name) for name in set(locked_names) - expected_names)
        missing = sorted(expected_names - set(locked_names))
        raise BoundaryError(
            "Cargo.lock violates the PR 02 zero-third-party policy; "
            f"unexpected={extra}, missing={missing}"
        )
    for package in locked_packages:
        if "source" in package or "checksum" in package:
            raise BoundaryError(
                f"Cargo.lock package {package.get('name')!r} unexpectedly has an external source"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of ci/)",
    )
    parser.add_argument(
        "--locked",
        action="store_true",
        help="pass --locked to cargo metadata and validate Cargo.lock",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    try:
        validate_root_manifest(root)
        validate_package_manifests(root)
        metadata = cargo_metadata(root, args.locked)
        packages = validate_members(root, metadata)
        validate_dependencies(root, packages)
        validate_features(packages)
        validate_build_scripts(root)
        if args.locked:
            validate_lockfile(root, packages)
    except BoundaryError as error:
        print(f"workspace boundary check failed: {error}", file=sys.stderr)
        return 1

    print("workspace boundary check passed")
    print("  production crates: 7")
    print("  excluded tool/research roots: tools/python, tools/native, experiments/triton")
    print("  third-party Cargo packages requiring license review: 0")
    print("  production Python/Triton build invocations: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
