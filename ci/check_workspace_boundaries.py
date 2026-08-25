#!/usr/bin/env python3
"""Fail closed when the production workspace or dependency allowlist drifts.

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
    "rustinfer-server": {
        "rustinfer-core",
        "rustinfer-model",
        "rustinfer-runtime",
        "rustinfer-scheduler",
    },
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
        "bench": ["dep:serde", "dep:serde_json"],
        "cuda": ["rustinfer-runtime/cuda", "rustinfer-scheduler/cuda"],
        "default": [],
        "experimental": [],
        "server": ["dep:serde", "dep:serde_json"],
    },
}

EXPECTED_OPTIONAL_DEPENDENCIES = {
    "rustinfer-core": set(),
    "rustinfer-cuda": set(),
    "rustinfer-tensor": {"rustinfer-cuda"},
    "rustinfer-model": set(),
    "rustinfer-runtime": {"rustinfer-cuda"},
    "rustinfer-scheduler": set(),
    "rustinfer-server": {"serde", "serde_json"},
}

EXPECTED_EXTERNAL_DIRECT_DEPENDENCIES = {
    ("rustinfer-model", "serde"): {
        "version": "1.0.228",
        "default_features": True,
        "features": ["derive"],
    },
    ("rustinfer-model", "serde_json"): {
        "version": "1.0.145",
        "default_features": True,
        "features": [],
    },
    ("rustinfer-model", "sha2"): {
        "version": "0.11.0",
        "default_features": False,
        "features": [],
    },
    ("rustinfer-model", "unicode-normalization"): {
        "version": "0.1.25",
        "default_features": True,
        "features": [],
    },
    ("rustinfer-runtime", "sha2"): {
        "version": "0.11.0",
        "default_features": False,
        "features": [],
    },
    ("rustinfer-server", "serde"): {
        "version": "1.0.228",
        "default_features": True,
        "features": ["derive"],
    },
    ("rustinfer-server", "serde_json"): {
        "version": "1.0.145",
        "default_features": True,
        "features": [],
    },
}

EXPECTED_DEV_DEPENDENCIES = {
    ("rustinfer-scheduler", "rustinfer-model"): {
        "req": "*",
        "default_features": True,
        "features": [],
        "path": "crates/rustinfer-model",
    },
    ("rustinfer-runtime", "rustinfer-cuda"): {
        "req": "*",
        "default_features": True,
        "features": [],
        "path": "crates/rustinfer-cuda",
    },
    ("rustinfer-runtime", "serde_json"): {
        "req": "=1.0.145",
        "default_features": True,
        "features": [],
        "path": None,
    },
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
DEPENDENCY_POLICY_PATH = Path("ci/approved_cargo_dependencies.toml")
DEPENDENCY_POLICY_KEYS = {
    "format",
    "registry_source",
    "workspace_msrv",
    "direct_dependencies",
    "packages",
}
DIRECT_DEPENDENCY_KEYS = {
    "owner",
    "name",
    "version",
    "default_features",
    "features",
}
APPROVED_PACKAGE_KEYS = {
    "name",
    "version",
    "source",
    "checksum",
    "license",
    "rust_version",
    "dependencies",
}
PACKAGE_ID = re.compile(r"(?P<name>[A-Za-z0-9_-]+)@(?P<version>\d+\.\d+\.\d+)")
SHA256 = re.compile(r"[0-9a-f]{64}")
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


def parse_rust_version(value: object, context: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value):
        raise BoundaryError(f"{context}: invalid Rust version {value!r}")
    components = tuple(int(component) for component in value.split("."))
    if len(components) == 2:
        return components + (0,)
    return components


def load_dependency_policy(root: Path) -> dict[str, Any]:
    policy_path = root / DEPENDENCY_POLICY_PATH
    policy = load_toml(policy_path)
    if set(policy) != DEPENDENCY_POLICY_KEYS:
        raise BoundaryError(
            f"{policy_path}: expected top-level keys "
            f"{sorted(DEPENDENCY_POLICY_KEYS)}, found {sorted(policy)}"
        )
    if policy["format"] != 1:
        raise BoundaryError(f"{policy_path}: unsupported dependency policy format")
    registry_source = policy["registry_source"]
    if registry_source != "registry+https://github.com/rust-lang/crates.io-index":
        raise BoundaryError(f"{policy_path}: only the crates.io registry is permitted")
    workspace_msrv = parse_rust_version(policy["workspace_msrv"], policy_path.as_posix())
    if workspace_msrv != (1, 85, 0):
        raise BoundaryError(f"{policy_path}: workspace_msrv must remain 1.85")

    direct_dependencies = policy["direct_dependencies"]
    if not isinstance(direct_dependencies, list):
        raise BoundaryError(f"{policy_path}: direct_dependencies must be an array")
    direct_keys: set[tuple[str, str]] = set()
    for dependency in direct_dependencies:
        if not isinstance(dependency, dict) or set(dependency) != DIRECT_DEPENDENCY_KEYS:
            raise BoundaryError(
                f"{policy_path}: every direct dependency must have exactly "
                f"{sorted(DIRECT_DEPENDENCY_KEYS)}"
            )
        owner = dependency["owner"]
        name = dependency["name"]
        version = dependency["version"]
        features = dependency["features"]
        if not all(isinstance(value, str) and value for value in (owner, name, version)):
            raise BoundaryError(f"{policy_path}: invalid direct dependency identity")
        if owner not in EXPECTED_CRATES.values():
            raise BoundaryError(f"{policy_path}: unknown dependency owner {owner!r}")
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise BoundaryError(f"{policy_path}: {name}: version must be exact x.y.z")
        if type(dependency["default_features"]) is not bool:
            raise BoundaryError(f"{policy_path}: {name}: default_features must be boolean")
        if (
            not isinstance(features, list)
            or not all(isinstance(feature, str) and feature for feature in features)
            or features != sorted(set(features))
        ):
            raise BoundaryError(f"{policy_path}: {name}: features must be sorted and unique")
        key = (owner, name)
        if key in direct_keys:
            raise BoundaryError(f"{policy_path}: duplicate direct dependency {owner} -> {name}")
        direct_keys.add(key)

    if direct_keys != set(EXPECTED_EXTERNAL_DIRECT_DEPENDENCIES):
        raise BoundaryError(
            f"{policy_path}: direct external dependencies must be exactly "
            "rustinfer-model -> serde, serde_json, sha2, and unicode-normalization plus "
            "rustinfer-runtime -> sha2 plus rustinfer-server -> serde and serde_json"
        )
    for dependency in direct_dependencies:
        expected = EXPECTED_EXTERNAL_DIRECT_DEPENDENCIES[
            (dependency["owner"], dependency["name"])
        ]
        actual = {
            key: dependency[key]
            for key in ("version", "default_features", "features")
        }
        if actual != expected:
            raise BoundaryError(
                f"{policy_path}: {dependency['name']}: direct dependency policy drifted; "
                f"expected {expected!r}, found {actual!r}"
            )

    approved_packages = policy["packages"]
    if not isinstance(approved_packages, list):
        raise BoundaryError(f"{policy_path}: packages must be an array")
    package_ids: set[str] = set()
    package_names: set[str] = set()
    for package in approved_packages:
        if not isinstance(package, dict) or set(package) != APPROVED_PACKAGE_KEYS:
            raise BoundaryError(
                f"{policy_path}: every package must have exactly "
                f"{sorted(APPROVED_PACKAGE_KEYS)}"
            )
        name = package["name"]
        version = package["version"]
        identity = f"{name}@{version}"
        if not isinstance(name, str) or PACKAGE_ID.fullmatch(identity) is None:
            raise BoundaryError(f"{policy_path}: invalid package identity {identity!r}")
        if identity in package_ids or name in package_names:
            raise BoundaryError(
                f"{policy_path}: duplicate identity or multi-version package {identity}"
            )
        package_ids.add(identity)
        package_names.add(name)
        if package["source"] != registry_source:
            raise BoundaryError(f"{policy_path}: {identity}: non-crates.io source")
        if not isinstance(package["checksum"], str) or SHA256.fullmatch(
            package["checksum"]
        ) is None:
            raise BoundaryError(f"{policy_path}: {identity}: invalid SHA-256 checksum")
        if not isinstance(package["license"], str) or not package["license"].strip():
            raise BoundaryError(f"{policy_path}: {identity}: license is required")
        rust_version = package["rust_version"]
        package_msrv = (
            (0, 0, 0)
            if rust_version == "unspecified"
            else parse_rust_version(rust_version, f"{policy_path}: {identity}")
        )
        if package_msrv > workspace_msrv:
            raise BoundaryError(
                f"{policy_path}: {identity}: MSRV {package['rust_version']} exceeds 1.85"
            )
        dependencies = package["dependencies"]
        if (
            not isinstance(dependencies, list)
            or not all(
                isinstance(dependency, str) and PACKAGE_ID.fullmatch(dependency)
                for dependency in dependencies
            )
            or dependencies != sorted(set(dependencies))
        ):
            raise BoundaryError(
                f"{policy_path}: {identity}: dependencies must be sorted unique name@version IDs"
            )

    unknown_edges = sorted(
        dependency
        for package in approved_packages
        for dependency in package["dependencies"]
        if dependency not in package_ids
    )
    if unknown_edges:
        raise BoundaryError(
            f"{policy_path}: dependency edges reference unapproved packages {unknown_edges}"
        )
    approved_names = {package["name"] for package in approved_packages}
    missing_direct = sorted(
        dependency["name"]
        for dependency in direct_dependencies
        if dependency["name"] not in approved_names
    )
    if missing_direct:
        raise BoundaryError(
            f"{policy_path}: direct dependencies missing from package closure {missing_direct}"
        )
    return policy


def normalized_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise BoundaryError(f"path escapes repository root: {path}") from error


def cargo_metadata(root: Path, locked: bool, *, no_deps: bool) -> dict[str, Any]:
    command = ["cargo", "metadata", "--format-version", "1"]
    if no_deps:
        command.append("--no-deps")
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


def validate_dependencies(
    root: Path, packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> None:
    production_names = set(packages)
    graph: dict[str, set[str]] = {}
    approved_direct = {
        (dependency["owner"], dependency["name"]): dependency
        for dependency in policy["direct_dependencies"]
    }

    for name, package in packages.items():
        actual_internal: set[str] = set()
        actual_external: set[tuple[str, str]] = set()
        actual_dev: set[tuple[str, str]] = set()
        for dependency in package.get("dependencies", []):
            if dependency.get("kind") == "dev":
                dependency_name = dependency.get("name")
                key = (name, dependency_name)
                expected = EXPECTED_DEV_DEPENDENCIES.get(key)
                if expected is None:
                    raise BoundaryError(
                        f"{name}: development Cargo dependency {dependency_name!r} "
                        "is not in the exact dev-dependency allowlist"
                    )
                if dependency.get("req") != expected["req"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: development requirement must be "
                        f"{expected['req']!r}"
                    )
                if dependency.get("rename") is not None or dependency.get("target") is not None:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: development dependency must be "
                        "unrenamed and unconditional"
                    )
                if dependency.get("optional") is not False:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: development dependency cannot be optional"
                    )
                if dependency.get("uses_default_features") is not expected["default_features"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: development default-feature policy drifted"
                    )
                if dependency.get("features") != expected["features"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: development features must be "
                        f"{expected['features']!r}"
                    )
                expected_path = expected["path"]
                if expected_path is None:
                    if dependency.get("source") != policy["registry_source"]:
                        raise BoundaryError(
                            f"{name} -> {dependency_name}: development dependency must use "
                            "the approved crates.io source"
                        )
                    if dependency.get("path") is not None:
                        raise BoundaryError(
                            f"{name} -> {dependency_name}: registry development dependency "
                            "cannot have a path"
                        )
                else:
                    if dependency.get("source") is not None:
                        raise BoundaryError(
                            f"{name} -> {dependency_name}: workspace development dependency "
                            "cannot use a registry source"
                        )
                    dependency_path_raw = dependency.get("path")
                    if not isinstance(dependency_path_raw, str) or normalized_relative(
                        Path(dependency_path_raw), root
                    ) != expected_path:
                        raise BoundaryError(
                            f"{name} -> {dependency_name}: development path must resolve to "
                            f"{expected_path}"
                        )
                actual_dev.add(key)
                continue
            dependency_name = dependency.get("name")
            if dependency_name not in production_names:
                approved = approved_direct.get((name, dependency_name))
                if approved is None:
                    raise BoundaryError(
                        f"{name}: external Cargo dependency {dependency_name!r} is not "
                        f"approved in {DEPENDENCY_POLICY_PATH}"
                    )
                if dependency.get("source") != policy["registry_source"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: only the approved crates.io source "
                        "is permitted; git dependencies are forbidden"
                    )
                if dependency.get("req") != f"={approved['version']}":
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: requirement must be the exact version "
                        f"={approved['version']}"
                    )
                if dependency.get("rename") is not None:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: dependency renames are forbidden"
                    )
                if dependency.get("kind") is not None or dependency.get("target") is not None:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: must be an unconditional normal dependency"
                    )
                expected_optional = (
                    dependency_name in EXPECTED_OPTIONAL_DEPENDENCIES[name]
                )
                if dependency.get("optional") is not expected_optional:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: optional must be {expected_optional}"
                    )
                if dependency.get("uses_default_features") is not approved["default_features"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: default-feature policy drifted"
                    )
                if dependency.get("features") != approved["features"]:
                    raise BoundaryError(
                        f"{name} -> {dependency_name}: expected features "
                        f"{approved['features']!r}, found {dependency.get('features')!r}"
                    )
                actual_external.add((name, dependency_name))
                continue
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
        expected_external = {
            key for key in approved_direct if key[0] == name
        }
        if actual_external != expected_external:
            raise BoundaryError(
                f"{name}: approved external dependency boundary drifted; expected "
                f"{sorted(dependency for _, dependency in expected_external)}, found "
                f"{sorted(dependency for _, dependency in actual_external)}"
            )
        expected_dev = {key for key in EXPECTED_DEV_DEPENDENCIES if key[0] == name}
        if actual_dev != expected_dev:
            raise BoundaryError(
                f"{name}: development dependency boundary drifted; expected "
                f"{sorted(dependency for _, dependency in expected_dev)}, found "
                f"{sorted(dependency for _, dependency in actual_dev)}"
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
    server_bins = {
        target.get("name"): target
        for target in server_targets
        if "bin" in target.get("kind", [])
    }
    if set(server_bins) != {"rustinfer", "rustinfer-profile"}:
        raise BoundaryError(
            "rustinfer-server must own exactly the production and native-profile binaries"
        )
    production_binary = server_bins["rustinfer"]
    if production_binary.get("required-features") != ["server"]:
        raise BoundaryError(
            "the rustinfer binary must be owned by rustinfer-server and require only `server`"
        )
    profile_binary = server_bins["rustinfer-profile"]
    if profile_binary.get("required-features") != ["bench", "cuda"]:
        raise BoundaryError(
            "the rustinfer-profile binary must require exactly `bench` and `cuda`"
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


def lock_dependency_name(reference: object, context: str) -> tuple[str, str | None]:
    if not isinstance(reference, str):
        raise BoundaryError(f"{context}: Cargo.lock dependency must be a string")
    components = reference.split()
    if not components or len(components) > 3:
        raise BoundaryError(f"{context}: malformed Cargo.lock dependency {reference!r}")
    version = components[1] if len(components) >= 2 else None
    return components[0], version


def validate_lockfile(
    root: Path, packages: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> None:
    lockfile = root / "Cargo.lock"
    lock = load_toml(lockfile)
    locked_packages = lock.get("package")
    if not isinstance(locked_packages, list):
        raise BoundaryError("Cargo.lock does not contain a package list")
    locked_by_name: dict[str, dict[str, Any]] = {}
    for package in locked_packages:
        name = package.get("name")
        if not isinstance(name, str) or name in locked_by_name:
            raise BoundaryError(
                f"Cargo.lock has an invalid, duplicate, or multi-version package {name!r}"
            )
        locked_by_name[name] = package

    workspace_names = set(packages)
    approved_by_name = {package["name"]: package for package in policy["packages"]}
    expected_names = workspace_names | set(approved_by_name)
    locked_names = set(locked_by_name)
    if locked_names != expected_names:
        extra = sorted(locked_names - expected_names)
        missing = sorted(expected_names - locked_names)
        raise BoundaryError(
            "Cargo.lock violates the reviewed dependency closure; "
            f"unexpected={extra}, missing={missing}"
        )

    for name in workspace_names:
        package = locked_by_name[name]
        if package.get("version") != "0.1.0":
            raise BoundaryError(f"Cargo.lock workspace package {name!r} version drifted")
        if "source" in package or "checksum" in package:
            raise BoundaryError(
                f"Cargo.lock workspace package {name!r} unexpectedly has an external source"
            )

    for name, approved in approved_by_name.items():
        package = locked_by_name[name]
        identity = f"{name}@{package.get('version')}"
        for field in ("version", "source", "checksum"):
            if package.get(field) != approved[field]:
                raise BoundaryError(
                    f"Cargo.lock {identity}: {field} must be {approved[field]!r}, "
                    f"found {package.get(field)!r}"
                )
        actual_dependencies: list[str] = []
        for reference in package.get("dependencies", []):
            dependency_name, reference_version = lock_dependency_name(
                reference, f"Cargo.lock {identity}"
            )
            dependency = locked_by_name.get(dependency_name)
            if dependency is None:
                raise BoundaryError(
                    f"Cargo.lock {identity}: unresolved dependency {reference!r}"
                )
            dependency_version = dependency.get("version")
            if reference_version is not None and reference_version != dependency_version:
                raise BoundaryError(
                    f"Cargo.lock {identity}: dependency version mismatch for {reference!r}"
                )
            if dependency_name in workspace_names:
                raise BoundaryError(
                    f"Cargo.lock {identity}: registry package may not depend on workspace "
                    f"package {dependency_name}"
                )
            actual_dependencies.append(f"{dependency_name}@{dependency_version}")
        if sorted(actual_dependencies) != approved["dependencies"]:
            raise BoundaryError(
                f"Cargo.lock {identity}: dependency edges drifted; expected "
                f"{approved['dependencies']!r}, found {sorted(actual_dependencies)!r}"
            )

    reachable: set[str] = set()
    pending = [dependency["name"] for dependency in policy["direct_dependencies"]]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(
            dependency.rsplit("@", 1)[0]
            for dependency in approved_by_name[name]["dependencies"]
        )
    if reachable != set(approved_by_name):
        raise BoundaryError(
            "approved Cargo package closure contains unreachable packages: "
            f"{sorted(set(approved_by_name) - reachable)}"
        )


def validate_external_metadata(metadata: dict[str, Any], policy: dict[str, Any]) -> None:
    workspace_names = set(EXPECTED_CRATES.values())
    external_packages = [
        package
        for package in metadata.get("packages", [])
        if package.get("name") not in workspace_names
    ]
    approved_by_identity = {
        (package["name"], package["version"]): package for package in policy["packages"]
    }
    actual_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for package in external_packages:
        identity = (package.get("name"), package.get("version"))
        if not all(isinstance(component, str) for component in identity):
            raise BoundaryError(f"cargo metadata has invalid external identity {identity!r}")
        if identity in actual_by_identity:
            raise BoundaryError(f"cargo metadata has duplicate external package {identity!r}")
        actual_by_identity[identity] = package
    if set(actual_by_identity) != set(approved_by_identity):
        unexpected = sorted(set(actual_by_identity) - set(approved_by_identity))
        missing = sorted(set(approved_by_identity) - set(actual_by_identity))
        raise BoundaryError(
            "resolved Cargo metadata violates the approved package closure; "
            f"unexpected={unexpected}, missing={missing}"
        )

    for identity, package in actual_by_identity.items():
        approved = approved_by_identity[identity]
        display = f"{identity[0]}@{identity[1]}"
        if package.get("source") != approved["source"]:
            raise BoundaryError(
                f"resolved {display}: source must be {approved['source']!r}; git is forbidden"
            )
        if package.get("license") != approved["license"]:
            raise BoundaryError(
                f"resolved {display}: license metadata drifted; expected "
                f"{approved['license']!r}, found {package.get('license')!r}"
            )
        expected_rust_version = (
            None if approved["rust_version"] == "unspecified" else approved["rust_version"]
        )
        if package.get("rust_version") != expected_rust_version:
            raise BoundaryError(
                f"resolved {display}: MSRV metadata drifted; expected "
                f"{expected_rust_version!r}, found {package.get('rust_version')!r}"
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
        dependency_policy = load_dependency_policy(root)
        validate_root_manifest(root)
        validate_package_manifests(root)
        metadata = cargo_metadata(root, args.locked, no_deps=True)
        packages = validate_members(root, metadata)
        validate_dependencies(root, packages, dependency_policy)
        validate_features(packages)
        validate_build_scripts(root)
        if args.locked:
            validate_lockfile(root, packages, dependency_policy)
            resolved_metadata = cargo_metadata(root, True, no_deps=False)
            validate_external_metadata(resolved_metadata, dependency_policy)
    except BoundaryError as error:
        print(f"workspace boundary check failed: {error}", file=sys.stderr)
        return 1

    print("workspace boundary check passed")
    print("  production crates: 7")
    print("  excluded tool/research roots: tools/python, tools/native, experiments/triton")
    print(
        "  approved direct third-party Cargo dependencies: "
        f"{len(dependency_policy['direct_dependencies'])} "
        "(rustinfer-model, rustinfer-runtime, and rustinfer-server)"
    )
    print(
        "  approved third-party Cargo package closure: "
        f"{len(dependency_policy['packages'])}"
    )
    if args.locked:
        print("  resolved source/checksum/license/MSRV/dependency edges: exact allowlist")
    print("  production Python/Triton build invocations: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
