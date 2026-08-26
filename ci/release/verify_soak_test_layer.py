#!/usr/bin/env python3
"""Static guards for the Python-free PR16 reliability-soak test layer."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from release_common import ReleaseContractError
from verify_runtime_dockerfile import _instructions


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = Path(__file__).with_name("ReliabilitySoak.Dockerfile")
RUNNER = ROOT / "ci/run_remote_release_soak.sh"
DRIVER = ROOT / "ci/run_release_soak.sh"
EXPECTED_REMOTE_RUNNER_SHA256 = (
    "c329733c37ab370bd04d1959ab7bd46e74cb89fb042a38638a23f24cfab64268"
)
EXPECTED_RELEASE_DRIVER_SHA256 = (
    "c1080c1939f199bf3e8d5dc1503d9149a5dbf9ddf15279ffc5f6bc37623f688b"
)
EXPECTED_REVIEWED_HOST_TOOLS = (
    "bash|/usr/bin/bash|59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4",
    "basename|/usr/bin/basename|3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22",
    "cat|/bin/cat|210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba",
    "chmod|/usr/bin/chmod|e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8",
    "cmp|/usr/bin/cmp|b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791",
    "cp|/usr/bin/cp|8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2",
    "date|/usr/bin/date|08b85d43067bcd15edb0882d5372a8b5635e211f76b62ccc4d575f2ed4920e18",
    "dirname|/usr/bin/dirname|674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94",
    "docker|/usr/bin/docker|29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6",
    "env|/usr/bin/env|85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0",
    "find|/usr/bin/find|791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1",
    "git|/usr/bin/git|587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a",
    "grep|/usr/bin/grep|73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96",
    "hostname|/usr/bin/hostname|d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e",
    "jq|/usr/bin/jq|858a84f22b39317f13a57b4b91e535925c1b4f819d9bb2864361df4ad6acb00f",
    "mawk|/usr/bin/mawk|dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e",
    "mkdir|/usr/bin/mkdir|bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b",
    "nvidia-smi|/usr/bin/nvidia-smi|22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be",
    "od|/usr/bin/od|8831c6be1e0b0a7c8c01e2f939b03d8d1d144e238c6b8e0a5d9d1a8c367ac910",
    "python3|/usr/bin/python3.10|7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
    "sha256sum|/usr/bin/sha256sum|7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3",
    "sort|/usr/bin/sort|0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78",
    "stat|/usr/bin/stat|9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3",
    "tail|/usr/bin/tail|d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0",
    "tar|/usr/bin/tar|fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2",
    "tee|/usr/bin/tee|eb219ccfbdad53064135a4101d4f56f0d9e5f7f1cd20c032b29e3604264cf79b",
    "tr|/usr/bin/tr|24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c",
    "wc|/usr/bin/wc|504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8",
)
EXPECTED_HOST_TOOL_BIN_VARIABLES = {
    "bash": "BASH_BIN",
    "basename": "BASENAME_BIN",
    "cat": "CAT_BIN",
    "chmod": "CHMOD_BIN",
    "cmp": "CMP_BIN",
    "cp": "CP_BIN",
    "date": "DATE_BIN",
    "dirname": "DIRNAME_BIN",
    "docker": "DOCKER_BIN",
    "env": "ENV_BIN",
    "find": "FIND_BIN",
    "git": "GIT_BIN",
    "grep": "GREP_BIN",
    "hostname": "HOSTNAME_BIN",
    "jq": "JQ_BIN",
    "mawk": "MAWK_BIN",
    "mkdir": "MKDIR_BIN",
    "nvidia-smi": "NVIDIA_SMI_BIN",
    "od": "OD_BIN",
    "python3": "PYTHON_BIN",
    "sha256sum": "SHA256SUM_BIN",
    "sort": "SORT_BIN",
    "stat": "STAT_BIN",
    "tail": "TAIL_BIN",
    "tar": "TAR_BIN",
    "tee": "TEE_BIN",
    "tr": "TR_BIN",
    "wc": "WC_BIN",
}
EXPECTED_DIRECT_HOST_TOOLS = {"bash", "cat", "env", "python3"}
EXPECTED_DIRECT_HOST_TOOL_INVOCATIONS = {
    "bash": ("#!/usr/bin/bash\n", 'os.execve(\n        "/usr/bin/bash",'),
    "cat": ("/bin/cat >&2 <<'EOF'",),
    "env": ("/usr/bin/env -i PATH=/usr/bin:/bin", "exec /usr/bin/env -i"),
    "python3": (
        "/usr/bin/python3.10 -I -S -c '",
        '"$PYTHON_BIN" -I -S - "$1" "$2"',
    ),
}
EXPECTED_WRAPPED_HOST_TOOLS = tuple(
    name
    for name in EXPECTED_HOST_TOOL_BIN_VARIABLES
    if name not in EXPECTED_DIRECT_HOST_TOOLS
)
EXPECTED_ABSOLUTE_HOST_TOOL_PATHS = {
    record.split("|")[1] for record in EXPECTED_REVIEWED_HOST_TOOLS
}

BASE_ARGUMENT = "ARG RUSTINFER_RELEASE_IMAGE_REF"
IDENTITY_ARGUMENT = "ARG RUSTINFER_RELEASE_IMAGE_ID"
BASE_INSTRUCTION = (
    "FROM ${RUSTINFER_RELEASE_IMAGE_REF} AS reliability-soak-test-layer"
)
APT_INSTALL_MARKER = (
    "apt-get install -y --no-install-recommends --no-upgrade "
    "bash coreutils curl findutils gawk grep jq procps util-linux"
)
EXPECTED_COPIES = (
    "COPY ci/run_release_soak.sh "
    "/opt/rustinfer-soak/ci/run_release_soak.sh",
    "COPY benchmarks/soak/reliability-soak-v1.json "
    "/opt/rustinfer-soak/benchmarks/soak/reliability-soak-v1.json",
)
EXPECTED_NORMALIZED_INSTRUCTION_SHA256 = (
    "dcd8ca61a97fab028d81d9875c17c4d3b017df3ab6e6cb15337249f840b3e1f1"
)
EXPECTED_LABEL_BINDINGS = (
    'org.rustinfer.reliability-soak.release-image-id="${RUSTINFER_RELEASE_IMAGE_ID}"',
    'org.rustinfer.reliability-soak.source-revision="${RUSTINFER_SOURCE_REVISION}"',
    'org.rustinfer.reliability-soak.source-archive-sha256="${RUSTINFER_SOURCE_ARCHIVE_SHA256}"',
    'org.rustinfer.reliability-soak.release-binary-sha256="${RUSTINFER_RELEASE_BINARY_SHA256}"',
)


def _fail(message: str) -> None:
    raise ReleaseContractError(message)


def verify_soak_dockerfile(path: Path = DOCKERFILE) -> None:
    contents = path.read_text(encoding="utf-8")
    if re.search(r"(?im)^\s*#\s*syntax\s*=", contents):
        _fail("soak Dockerfile must not depend on an external syntax frontend")
    instructions = _instructions(contents)
    normalized = "\n".join(instructions) + "\n"
    normalized_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if normalized_sha256 != EXPECTED_NORMALIZED_INSTRUCTION_SHA256:
        _fail(
            "soak Dockerfile normalized instruction stream differs from the "
            "reviewed exact order/content"
        )
    if instructions.count(BASE_ARGUMENT) != 1:
        _fail("soak Dockerfile must declare exactly one required base-reference argument")
    if instructions.count(IDENTITY_ARGUMENT) != 1:
        _fail("soak Dockerfile must declare the immutable release-image identity")
    if any(
        instruction.startswith(f"{argument}=")
        for instruction in instructions
        for argument in (BASE_ARGUMENT, IDENTITY_ARGUMENT)
    ):
        _fail("soak Dockerfile must not provide a mutable default release image")
    from_instructions = [
        instruction
        for instruction in instructions
        if instruction.upper().startswith("FROM ")
    ]
    if from_instructions != [BASE_INSTRUCTION]:
        _fail("soak Dockerfile must derive one stage from the supplied image ID")
    if any(instruction.upper().startswith("ADD ") for instruction in instructions):
        _fail("soak Dockerfile must not use ADD")

    copies = tuple(
        instruction
        for instruction in instructions
        if instruction.upper().startswith("COPY ")
    )
    if copies != EXPECTED_COPIES:
        _fail("soak layer may copy only the driver and canonical manifest template")
    docker_text = "\n".join(instructions)
    if docker_text.count(APT_INSTALL_MARKER) != 1:
        _fail("soak layer must install exactly the reviewed observation utilities")

    label_lines = [
        instruction
        for instruction in instructions
        if instruction.upper().startswith("LABEL ")
    ]
    if len(label_lines) != 1:
        _fail("soak Dockerfile must have one closed provenance label instruction")
    for binding in EXPECTED_LABEL_BINDINGS:
        if label_lines[0].count(binding) != 1:
            _fail(f"soak Dockerfile label is missing exact binding: {binding}")

    required = {
        "ARG RUSTINFER_SOURCE_REVISION",
        "ARG RUSTINFER_SOURCE_ARCHIVE_SHA256",
        "ARG RUSTINFER_RELEASE_BINARY_SHA256",
        "USER 65532:65532",
        'ENTRYPOINT ["/opt/rustinfer-soak/ci/run_release_soak.sh"]',
        "CMD []",
    }
    missing = required - set(instructions)
    if missing:
        _fail("soak Dockerfile is missing: " + ", ".join(sorted(missing)))
    users = [line for line in instructions if line.upper().startswith("USER ")]
    entrypoints = [
        line for line in instructions if line.upper().startswith("ENTRYPOINT ")
    ]
    commands = [line for line in instructions if line.upper().startswith("CMD ")]
    final_runtime = (
        "USER 65532:65532",
        'ENTRYPOINT ["/opt/rustinfer-soak/ci/run_release_soak.sh"]',
        "CMD []",
    )
    if users != ["USER 0:0", final_runtime[0]]:
        _fail("soak Dockerfile USER transitions must be root-build then production")
    if entrypoints != [final_runtime[1]] or commands != [final_runtime[2]]:
        _fail("soak Dockerfile must have one final ENTRYPOINT and one final CMD")
    if tuple(instructions[-3:]) != final_runtime:
        _fail("soak Dockerfile must end with the exact production runtime identity")

    for marker in (
        "sha256sum /opt/rustinfer/bin/rustinfer",
        "for command_name in python python3 pip pip3 cargo rustc nvcc cmake make cc c++",
        "for required_command in bash jq curl sha256sum awk ps flock readlink find sort grep wc date env",
        "find / -xdev -type f",
    ):
        if marker not in docker_text:
            _fail(f"soak Dockerfile lacks runtime boundary marker: {marker}")
    if re.search(r"\b(python3?|pip3?)\b", APT_INSTALL_MARKER):
        _fail("soak observation package set must not contain Python")
    closure_markers = (
        "closure_before=/opt/rustinfer-soak/release-runtime-closure.tsv",
        "capture_runtime_closure \"$closure_before\"",
        APT_INSTALL_MARKER,
        "capture_runtime_closure \"$closure_after\"",
        "cmp --silent \"$closure_before\" \"$closure_after\"",
        "chmod 0444 \"$closure_before\"",
    )
    offset = 0
    for marker in closure_markers:
        position = docker_text.find(marker, offset)
        if position < 0:
            _fail(f"soak Dockerfile lacks ordered runtime-closure marker: {marker}")
        offset = position + len(marker)
    for marker in (
        "LC_ALL=C ldd \"$release_binary_path\"",
        "test \"$dependency_name\" != linux-vdso.so.1",
        "if test \"$dependency_resolution\" = not",
        'test "$dependency_name" = libcuda.so.1',
        "unresolved_count=$((unresolved_count + 1))",
        "printf '%s\\tNOT_FOUND\\t-\\t-\\n'",
        "dependency_target=$(readlink -f -- \"$dependency_path\")",
        "test -f \"$dependency_target\" && test ! -L \"$dependency_target\"",
        "dependency_sha256=$(sha256sum \"$dependency_target\")",
        "printf '%s\\t%s\\t%s\\t%s\\n'",
        '"$dependency_sha256" >>"$closure_unsorted"',
        'test "$unresolved_count" -eq 1',
        "LC_ALL=C sort -u \"$closure_unsorted\" >\"$closure_output\"",
    ):
        if marker not in docker_text:
            _fail(f"soak Dockerfile lacks runtime-closure invariant: {marker}")


def _shell_code(contents: str) -> str:
    return "\n".join(
        line for line in contents.splitlines() if not line.lstrip().startswith("#")
    )


def _require_once(code: str, marker: str, subject: str) -> None:
    count = code.count(marker)
    if count != 1:
        _fail(f"{subject} must contain exactly one marker ({count} found): {marker}")


def _require_in_order(code: str, markers: tuple[str, ...], subject: str) -> None:
    offset = 0
    for marker in markers:
        position = code.find(marker, offset)
        if position < 0:
            _fail(f"{subject} lacks ordered marker: {marker}")
        offset = position + len(marker)


def _mask_noncommand_shell_regions(contents: str) -> str:
    """Keep shell command positions while blanking quoted programs and arrays."""
    masked_lines: list[str] = []
    in_single_quote = False
    heredoc_delimiter: str | None = None
    in_array_literal = False
    previous_line_continues = False
    for line in contents.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped) :]
        logical_continuation = previous_line_continues or in_single_quote
        previous_line_continues = stripped.rstrip().endswith("\\")
        if heredoc_delimiter is not None:
            masked_lines.append(" " * len(stripped) + newline)
            if stripped == heredoc_delimiter:
                heredoc_delimiter = None
            continue
        if in_array_literal:
            masked_lines.append(" " * len(stripped) + newline)
            if re.fullmatch(r"\s*\)\s*", stripped):
                in_array_literal = False
            continue
        if not in_single_quote and re.fullmatch(
            r"\s*[A-Za-z_][A-Za-z0-9_]*=\(\s*", stripped
        ):
            masked_lines.append(" " * len(stripped) + newline)
            in_array_literal = True
            continue

        heredoc_match = re.search(
            r"<<-?['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", stripped
        )
        characters = list(stripped)
        for index, character in enumerate(stripped):
            if character == "'":
                in_single_quote = not in_single_quote
                characters[index] = " "
            elif in_single_quote:
                characters[index] = " "
        masked = "".join(characters)
        for pattern in (r"\(\(.*?\)\)", r"\[\[.*?\]\]"):
            masked = re.sub(pattern, lambda match: " " * len(match.group(0)), masked)
        masked_lines.append(("@" if logical_continuation else "") + masked + newline)
        if heredoc_match is not None:
            heredoc_delimiter = heredoc_match.group(1)
    if in_single_quote or heredoc_delimiter is not None or in_array_literal:
        _fail("remote soak runner has an unterminated quoted program, heredoc, or array")
    return "".join(masked_lines)


def _static_external_command_names(
    contents: str,
    reviewed_records: dict[str, tuple[str, str]],
) -> set[str]:
    masked = _mask_noncommand_shell_regions(_shell_code(contents))
    functions = set(
        re.findall(r"(?m)^([a-z][a-z0-9_-]*)\(\)[ \t]*\{", masked)
    )
    shell_only = {
        "break",
        "case",
        "cd",
        "compgen",
        "continue",
        "declare",
        "do",
        "done",
        "echo",
        "elif",
        "else",
        "esac",
        "exec",
        "exit",
        "export",
        "fi",
        "for",
        "hash",
        "if",
        "in",
        "local",
        "printf",
        "pwd",
        "read",
        "readonly",
        "return",
        "set",
        "shift",
        "test",
        "then",
        "trap",
        "true",
        "umask",
        "unset",
        "until",
        "wait",
        "while",
    }
    reverse_variables = {
        variable: name for name, variable in EXPECTED_HOST_TOOL_BIN_VARIABLES.items()
    }
    reverse_paths = {path: name for name, (path, _digest) in reviewed_records.items()}
    command_positions = re.compile(
        r"(?m)(?:^[ \t]*|[ \t]+(?:&&|\|\|)[ \t]+|[ \t]+\|[ \t]+|"
        r"\{[ \t]+|\$\([ \t]*|[<>]\([ \t]*)"
        r"(?P<command>\"?\$[A-Z][A-Z0-9_]*_BIN\"?|"
        r"/(?:usr/)?bin/[A-Za-z0-9._+-]+|[a-z][a-z0-9_-]*)"
    )
    external: set[str] = set(EXPECTED_DIRECT_HOST_TOOLS)
    unknown: set[str] = set()
    for match in command_positions.finditer(masked):
        token = match.group("command")
        line_prefix = masked[masked.rfind("\n", 0, match.start()) + 1 : match.start()]
        if not line_prefix.strip() and re.match(
            r"[ \t]*[A-Za-z_][A-Za-z0-9_]*=", masked[match.start() :]
        ):
            continue
        if token in shell_only or token in functions:
            continue
        if token in reviewed_records:
            external.add(token)
            continue
        if token in reverse_paths:
            external.add(reverse_paths[token])
            continue
        variable_match = re.fullmatch(
            r'"?\$([A-Za-z_][A-Za-z0-9_]*)"?', token
        )
        if variable_match is not None and variable_match.group(1) in reverse_variables:
            external.add(reverse_variables[variable_match.group(1)])
            continue
        unknown.add(token)
    if unknown:
        _fail(
            "remote soak runner has unreviewed command-position tokens: "
            + ", ".join(sorted(unknown))
        )
    return external


def verify_remote_soak_runner(path: Path = RUNNER) -> None:
    contents = path.read_text(encoding="utf-8")
    if hashlib.sha256(contents.encode("utf-8")).hexdigest() != EXPECTED_REMOTE_RUNNER_SHA256:
        _fail("remote soak runner differs from the reviewed exact source digest")
    code = _shell_code(contents)
    if not contents.startswith("#!/usr/bin/bash\n"):
        _fail("remote soak launcher must use the absolute reviewed Bash shebang")
    reviewed_block = re.search(
        r"(?ms)^reviewed_tools=\(\n(?P<body>.*?)^\)\n",
        code,
    )
    if reviewed_block is None:
        _fail("remote soak runner lacks the reviewed host-tool inventory")
    reviewed_tools = tuple(
        re.findall(r"(?m)^\s+'([^']+)'$", reviewed_block.group("body"))
    )
    if reviewed_tools != EXPECTED_REVIEWED_HOST_TOOLS:
        _fail(
            "remote soak reviewed host-tool inventory is not exact: "
            f"{reviewed_tools}"
        )
    reviewed_records: dict[str, tuple[str, str]] = {}
    for reviewed_tool in reviewed_tools:
        name, tool_path, digest = reviewed_tool.split("|")
        if name in reviewed_records:
            _fail(f"remote soak reviewed host-tool name is duplicated: {name}")
        reviewed_records[name] = (tool_path, digest)

    expected_names = set(EXPECTED_HOST_TOOL_BIN_VARIABLES)
    if set(reviewed_records) != expected_names:
        _fail("remote soak executable host-tool set differs from its inventory")
    declared_bin_paths = dict(
        re.findall(
            r"(?m)^readonly ([A-Z][A-Z0-9_]*_BIN)=(/(?:usr/)?bin/[A-Za-z0-9._+-]+)$",
            code,
        )
    )
    expected_bin_paths = {
        variable: reviewed_records[name][0]
        for name, variable in EXPECTED_HOST_TOOL_BIN_VARIABLES.items()
    }
    if declared_bin_paths != expected_bin_paths:
        _fail(
            "remote soak absolute dispatch variables differ from the reviewed "
            f"host-tool inventory: {declared_bin_paths}"
        )

    wrapper_dispatch = dict(
        re.findall(
            r'(?m)^([a-z][a-z0-9-]*)\(\) \{ "\$([A-Z][A-Z0-9_]*_BIN)" "\$@"; \}$',
            code,
        )
    )
    expected_wrapper_dispatch = {
        name: EXPECTED_HOST_TOOL_BIN_VARIABLES[name]
        for name in EXPECTED_WRAPPED_HOST_TOOLS
    }
    if wrapper_dispatch != expected_wrapper_dispatch:
        _fail(
            "remote soak reviewed wrappers do not exactly dispatch every "
            f"non-bootstrap host tool: {wrapper_dispatch}"
        )
    readonly_function_lines = re.findall(r"(?m)^readonly -f (.+)$", code)
    expected_readonly_functions = " ".join(EXPECTED_WRAPPED_HOST_TOOLS)
    if readonly_function_lines != [expected_readonly_functions]:
        _fail("remote soak reviewed host-tool wrappers must all be readonly")
    wrapper_lock_marker = f"readonly -f {expected_readonly_functions}"
    runtime_dispatch = code.split(wrapper_lock_marker, 1)[1]
    for name in EXPECTED_WRAPPED_HOST_TOOLS:
        if re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?=[ \t\\\n])",
            runtime_dispatch,
        ) is None:
            _fail(f"remote soak reviewed host-tool inventory contains an unused command: {name}")
    if set(wrapper_dispatch) | EXPECTED_DIRECT_HOST_TOOLS != expected_names:
        _fail("remote soak executable host-tool set is not closed by direct calls and wrappers")
    if _static_external_command_names(contents, reviewed_records) != expected_names:
        _fail("remote soak statically used external command set differs from its inventory")
    if set(EXPECTED_DIRECT_HOST_TOOL_INVOCATIONS) != EXPECTED_DIRECT_HOST_TOOLS:
        _fail("remote soak direct host-tool invocation contract is incomplete")
    for name, markers in EXPECTED_DIRECT_HOST_TOOL_INVOCATIONS.items():
        if any(marker not in contents for marker in markers):
            _fail(f"remote soak direct host tool is not invoked by absolute path: {name}")
    if contents.count("/usr/bin/python3.10 -I -S -c '") != 2:
        _fail("both remote soak bootstrap Python calls must use -I -S")
    if contents.count('"$PYTHON_BIN" -I -S - "$1" "$2"') != 1:
        _fail("the remote soak snapshot Python call must use -I -S exactly once")
    for name, (tool_path, _digest) in reviewed_records.items():
        if re.search(
            rf"\bcommand\s+(?:--\s+)?(?:{re.escape(name)}|{re.escape(tool_path)})\b",
            code,
        ):
            _fail(f"remote soak host tool bypasses its reviewed absolute dispatch: {name}")
    find_exec_targets = set(re.findall(r'\s-exec\s+("[^\"]+"|[^\s;]+)', code))
    if find_exec_targets != {'"$CHMOD_BIN"'}:
        _fail(
            "remote soak find -exec targets must use only the reviewed absolute "
            f"chmod path: {sorted(find_exec_targets)}"
        )
    absolute_host_tools = set(
        re.findall(
            r"(?<![A-Za-z0-9._+-])/(?:usr/)?bin/[A-Za-z0-9._+-]+",
            code,
        )
    )
    if absolute_host_tools != EXPECTED_ABSOLUTE_HOST_TOOL_PATHS:
        _fail(
            "remote soak absolute host-tool inventory is not exact: "
            f"{sorted(absolute_host_tools)}"
        )
    validation_markers = (
        'test -f "$tool_path" && test ! -L "$tool_path" && test -x "$tool_path"',
        'test "$("$STAT_BIN" -c \'%u\' -- "$tool_path")" = 0',
        'tool_mode=$("$STAT_BIN" -c \'%a\' -- "$tool_path")',
        '(( (8#$tool_mode & 8#022) == 0 ))',
        'test "$("$SHA256SUM_BIN" "$tool_path" | "$MAWK_BIN" \'{print $1}\')" = "$tool_sha256"',
    )
    for marker in validation_markers:
        if code.count(marker) != 1:
            _fail(
                "every reviewed host tool must be a root-owned, non-writable, "
                f"non-symlink executable with its exact digest: {marker}"
            )
    if re.search(r"(?:/usr/bin/)?\bawk\b", code):
        _fail("remote soak runner must use the reviewed /usr/bin/mawk, never awk")
    required_markers = (
        "DESIGNATED_HOSTNAME=psyche-MS-7D91",
        "DESIGNATED_GPU_UUID=GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0",
        'lock_path = "/var/tmp/rustinfer-server-4096-gpu-evidence.lock"',
        "os.O_NOFOLLOW",
        "os.O_NONBLOCK",
        'getattr(os, "O_CLOEXEC", 0)',
        "metadata = os.fstat(descriptor)",
        "metadata.st_nlink != 1",
        "metadata.st_uid != os.getuid()",
        "stat.S_IMODE(metadata.st_mode) != 0o600",
        "fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "child = os.fork()",
        "os.close(descriptor)",
        "os.waitpid(child, 0)",
        "signal.signal(forwarded_signal, forward)",
        '"RUSTINFER_SOAK_SUPERVISOR_PID": str(os.getpid())',
        '"RUSTINFER_SOAK_SUPERVISOR_LOCK_FD": str(descriptor)',
        "PR_SET_PDEATHSIG = 1",
        "libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM",
        "if os.getppid() != supervisor_pid:",
        'test "$PPID" = "$RUSTINFER_SOAK_SUPERVISOR_PID"',
        'f"/proc/{supervisor_pid}/exe"',
        'f"/proc/{supervisor_pid}/fd/{descriptor}"',
        'f"/proc/{supervisor_pid}/fdinfo/{descriptor}"',
        "FLOCK\\s+ADVISORY\\s+WRITE",
        'exec /usr/bin/env -i',
        'PATH=/usr/bin:/bin',
        '/usr/bin/python3.10',
        '/usr/bin/python3.10 -I -S -c',
        '"$PYTHON_BIN" -I -S - "$1" "$2"',
        '7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86',
        'readonly MAWK_BIN=/usr/bin/mawk',
        'readonly STAT_BIN=/usr/bin/stat',
        '"$STAT_BIN" -c \'%u\' -- "$tool_path"',
        '"$SHA256SUM_BIN" "$tool_path" | "$MAWK_BIN" \'{print $1}\'',
        'readonly -f basename chmod cmp cp date dirname docker find git grep hostname jq mawk mkdir nvidia-smi od sha256sum sort stat tail tar tee tr wc',
        'export DOCKER_CONFIG="$docker_config"',
        'BASH_FUNC_*|LD_*|GIT_*|DOCKER_*|BUILDX_*',
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
        'source_archive="$output_dir/source-archive.tar"',
        'snapshot_regular_file "$source_archive_input" "$source_archive"',
        "os.O_EXCL",
        'tar --extract --to-stdout --file "$source_archive"',
        'echo "source archive must be an uncompressed tar stream"',
        'test "$archive_ustar_magic" = 757374617200',
        'tar --list --file "$source_archive" >/dev/null',
        "git get-tar-commit-id",
        'test "$resolved_release_image_id" = "$release_image_id"',
        'test "$(sha256_file "$release_binary")" = "$expected_release_binary_sha256"',
        'write_model_manifest "$model_snapshot" "$model_snapshot_pre_manifest"',
        'test "$(sha256_file "$model_snapshot_pre_manifest")" = "$expected_model_tree_sha256"',
        "--expected-correctness-golden-sha256",
        "--expected-native-correctness-report-sha256",
        'test "$(sha256_file "$correctness_golden")" = "$expected_correctness_golden_sha256"',
        'test "$(sha256_file "$native_correctness_report")" = "$expected_native_correctness_report_sha256"',
        '.correctness_report_sha256 == $native_report_sha256',
        '.bindings.candidate_git_revision == $source_revision',
        '(.bindings.candidate_executable_sha256',
        'type == "string" and test("^[0-9a-f]{64}$"))',
        "test \"$(jq -er '.golden.generated_sha256' \"$materialized_manifest\")\" = \"$golden_generated_sha256\"",
        "test \"$(jq -er '.golden.provenance_sha256' \"$materialized_manifest\")\" = \"$expected_native_correctness_report_sha256\"",
        'runtime_receipts="$output_dir/runtime-receipts"',
        'mkdir -m 0700 "$runtime_receipts"',
        'model_snapshot="$output_dir/model-snapshot"',
        'cp --recursive --no-preserve=mode,ownership,timestamps -- "$model_dir_input/." "$model_snapshot/"',
        'find "$model_snapshot" -type f -exec "$CHMOD_BIN" 0444 {} +',
        'find "$model_snapshot" -type d -exec "$CHMOD_BIN" 0555 {} +',
        'write_model_manifest "$model_snapshot" "$model_snapshot_pre_manifest"',
        'verify_input_snapshots immediate-pre-start',
        'verify_input_snapshots post',
        'build_context="$output_dir/test-layer-build-context"',
        'context_members=(',
        'tar --extract',
        '--directory "$build_context"',
        'cmp --silent "$build_context_pre_manifest" "$build_context_immediate_manifest"',
        'docker image inspect "$release_image_id" >"$runtime_receipts/release-image-inspect.json"',
        'release image has an unsafe or unexpected inherited build environment',
        '$name == "BASH_ENV"',
        '$name == "LD_PRELOAD"',
        '$name == "LD_AUDIT"',
        '$name == "HOME"',
        '$name == "CURL_HOME"',
        '$name == "XDG_CONFIG_HOME"',
        '$name | startswith("BASH_FUNC_")',
        '/opt/rustinfer/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        '$environment.NVIDIA_VISIBLE_DEVICES == "all"',
        '$environment.NVIDIA_DRIVER_CAPABILITIES == "compute,utility"',
        '--arg expected_ld_library_path "/usr/local/cuda/lib64"',
        '$environment.LD_LIBRARY_PATH == $expected_ld_library_path',
        "DOCKER_BUILDKIT=1 docker build",
        '--file "$build_context/ci/release/ReliabilitySoak.Dockerfile"',
        'base_image_tag="rustinfer-soak-release-base:${resolved_revision}-${release_image_id#sha256:}"',
        'docker image tag "$release_image_id" "$base_image_tag"',
        'test "$resolved_base_image_id" = "$release_image_id"',
        'test "$post_build_base_image_id" = "$release_image_id"',
        "--build-arg \"RUSTINFER_RELEASE_IMAGE_REF=${base_image_tag}\"",
        "--build-arg \"RUSTINFER_RELEASE_IMAGE_ID=${release_image_id}\"",
        '"$build_context" 2>&1 | tee "$output_dir/test-layer-build.log"',
        '$test_layer[0:($release | length)] == $release',
        'docker image inspect "$test_image_id" >"$runtime_receipts/test-layer-image-inspect.json"',
        'and .[0].Config.User == "65532:65532"',
        'and .[0].Config.WorkingDir == $working_directory',
        'and .[0].Config.Labels == $expected_labels',
        'and .[0].Config.Entrypoint == ["/opt/rustinfer-soak/ci/run_release_soak.sh"]',
        'and .[0].Config.Cmd == []',
        'expected_container_environment=$(jq -cn',
        '--user 65532:65532',
        '--network none',
        '--pid host',
        '--read-only',
        '--cap-drop ALL',
        '--security-opt no-new-privileges',
        '--tmpfs /tmp:rw,nosuid,nodev,noexec,size=67108864',
        '--gpus "device=${DESIGNATED_GPU_UUID}"',
        'source=${model_snapshot},destination=/model,readonly',
        'source=${materialized_manifest_copy},destination=/run-input/reliability-soak-v1.json,readonly',
        'source=${container_evidence},destination=/evidence',
        'mkdir -m 0777 "$container_evidence"',
        'RUSTINFER_SOAK_OUTPUT=/evidence/run',
        'RUSTINFER_SOAK_FINAL_METRICS_JSON=/evidence/final-metrics.json',
        '$container.State.Status == $status',
        '$container.State.OOMKilled == false',
        '$container.State.Error == ""',
        '$container.State.Pid == 0',
        '$container.Config.Healthcheck == {Test:["NONE"]}',
        '$container.Path == "/opt/rustinfer-soak/ci/run_release_soak.sh"',
        '$container.Args == []',
        'def docker_timestamp:',
        '>= 26100',
        '$container.RestartCount == 0',
        '$container.HostConfig.NetworkMode == "none"',
        '$container.HostConfig.PidMode == "host"',
        '$container.HostConfig.IpcMode == "private"',
        '$container.HostConfig.UTSMode == ""',
        '$container.HostConfig.UsernsMode == ""',
        '$container.HostConfig.CgroupnsMode == "private"',
        '$container.HostConfig.Runtime == "runc"',
        '$container.HostConfig.Devices == []',
        '$container.HostConfig.DeviceCgroupRules == null',
        '$container.HostConfig.ReadonlyRootfs == true',
        '$container.HostConfig.Privileged == false',
        '$container.HostConfig.CapDrop == ["ALL"]',
        '$container.HostConfig.SecurityOpt == ["no-new-privileges:true"]',
        '$container.HostConfig.DeviceRequests[0].Driver == ""',
        '$container.HostConfig.DeviceRequests[0].Count == 0',
        '$container.HostConfig.DeviceRequests[0].DeviceIDs == [$gpu_uuid]',
        '($container.NetworkSettings.Networks // {}) | keys) == ["none"]',
        '($container.Config.Env | environment_map) == $expected_environment',
        'Propagation:"rprivate"',
        'validate_container_contract "$runtime_receipts/container-inspect-pre.json" created 0 false',
        'docker inspect "$container_name" >"$runtime_receipts/container-inspect-pre.json"',
        'docker cp "$container_name:/opt/rustinfer/bin/rustinfer" "$container_binary_copy"',
        'test "$(sha256_file "$container_binary_copy")" = "$expected_release_binary_sha256"',
        'runtime_closure_receipt="$runtime_receipts/release-runtime-closure.tsv"',
        '"$container_name:/opt/rustinfer-soak/release-runtime-closure.tsv"',
        'test "$(stat -c \'%a\' "$runtime_closure_receipt")" = 444',
        '$2 == "NOT_FOUND"',
        '$1 != "libcuda.so.1" || $3 != "-" || $4 != "-"',
        'unresolved != 1',
        'sort -u "$runtime_closure_receipt" | cmp --silent - "$runtime_closure_receipt"',
        'release_runtime_closure_sha256=$(sha256_file "$runtime_closure_receipt")',
        "require_gpu_idle immediate-pre-start",
        'docker wait "$container_name" >"$output_dir/container-exit-code.txt"',
        'docker inspect "$container_name" >"$runtime_receipts/container-inspect-post.json"',
        'docker cp "$container_name:/evidence/." "$container_evidence_export"',
        'validate_container_contract "$runtime_receipts/container-inspect-post.json" exited 0 true',
        '["Id","Name","Image","Path","Args","Created","Config","HostConfig","Mounts"]',
        'run_json_sha256=$(sha256_file "$container_evidence_export/run/run.json")',
        'events_jsonl_sha256=$(sha256_file "$container_evidence_export/run/events.jsonl")',
        "rustinfer.reliability-soak-launcher-receipt.v3",
        'host:{hostname:$hostname,gpu_name:$gpu_name,gpu_uuid:$gpu_uuid,compute_capability:$compute_capability,memory_total_mib:$memory_total_mib,driver_version:$driver_version}',
        'source:{git_revision:$git_revision,source_archive_sha256:$source_archive_sha256,release_binary_sha256:$release_binary_sha256,model_tree_sha256:$model_tree_sha256,manifest_sha256:$manifest_sha256,correctness_golden_sha256:$correctness_golden_sha256,native_correctness_report_sha256:$native_correctness_report_sha256}',
        'evidence:{run_json_sha256:$run_json_sha256,events_jsonl_sha256:$events_jsonl_sha256,release_runtime_closure_sha256:$release_runtime_closure_sha256}',
        'images:{release_image_id:$release_image_id,test_layer_image_id:$test_layer_image_id}',
        'container:{id:$container_id,name:$container_name,exit_code:$container_exit_code}',
        'keys == ["container","evidence","host","images","schema_version","source"]',
        '(.host | keys) == ["compute_capability","driver_version","gpu_name","gpu_uuid","hostname","memory_total_mib"]',
        '(.source | keys) == ["correctness_golden_sha256","git_revision","manifest_sha256","model_tree_sha256","native_correctness_report_sha256","release_binary_sha256","source_archive_sha256"]',
        '(.evidence | keys) == ["events_jsonl_sha256","release_runtime_closure_sha256","run_json_sha256"]',
        '(.images | keys) == ["release_image_id","test_layer_image_id"]',
        '(.container | keys) == ["exit_code","id","name"]',
        '(.host.memory_total_mib | type) == "number"',
        '(.container.exit_code | type) == "number"',
        'expected_runtime_receipts=$(printf',
        "container-inspect-post.json",
        "container-inspect-pre.json",
        "host-gpu.csv",
        "launcher-receipt.json",
        "release-image-inspect.json",
        "release-runtime-closure.tsv",
        "test-layer-image-inspect.json",
        'test "$(<"$runtime_receipts/host-gpu.csv")" = "$gpu_rows"',
        '>launcher-SHA256SUMS',
        'rustinfer.remote-release-soak.completed.v1 >"$output_dir/completed"',
    )
    for marker in required_markers:
        if marker not in code:
            _fail(f"remote soak runner lacks required marker: {marker}")
    for marker, expected_count in (
        ("metadata = os.fstat(descriptor)", 1),
        ("stat.S_IMODE(metadata.st_mode) != 0o600", 2),
        ("fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)", 1),
        ("os.O_NOFOLLOW", 3),
        ("os.O_NONBLOCK", 2),
        ("os.O_EXCL", 1),
    ):
        if code.count(marker) != expected_count:
            _fail(
                f"remote soak lock boundary must contain {expected_count} exact "
                f"occurrences of: {marker}"
            )

    inventory_start = code.find("expected_runtime_receipts=$(printf '%s\\n'")
    inventory_end = code.find("| sort)", inventory_start)
    if inventory_start < 0 or inventory_end < 0:
        _fail("remote soak runner lacks the closed runtime receipt inventory")
    inventory_block = code[inventory_start : inventory_end + len("| sort)")]
    inventory_names = tuple(
        re.findall(
            r"(?m)^\s{4}([a-z0-9.-]+)(?: \\| \| sort\))?$",
            inventory_block,
        )
    )
    expected_inventory = (
        "container-inspect-post.json",
        "container-inspect-pre.json",
        "host-gpu.csv",
        "launcher-receipt.json",
        "release-image-inspect.json",
        "release-runtime-closure.tsv",
        "test-layer-image-inspect.json",
    )
    if inventory_names != expected_inventory:
        _fail(f"remote soak runner runtime receipt inventory is open: {inventory_names}")

    checksum_end = code.find(">launcher-SHA256SUMS")
    checksum_start = code.rfind("sha256sum \\", 0, checksum_end)
    if checksum_start < 0 or checksum_end < 0:
        _fail("remote soak runner lacks its final launcher checksum")
    checksum_block = code[checksum_start:checksum_end]
    for receipt_name in expected_inventory:
        marker = f"runtime-receipts/{receipt_name}"
        if checksum_block.count(marker) != 1:
            _fail(f"remote soak checksum must bind exactly one {marker}")

    for marker in (
        "require_gpu_idle initial-preflight",
        "require_gpu_idle immediate-pre-start",
        "DOCKER_BUILDKIT=1 docker build",
        'docker start "$container_name" >/dev/null',
        'docker wait "$container_name" >"$output_dir/container-exit-code.txt"',
        'validate_container_contract "$runtime_receipts/container-inspect-pre.json" created 0 false',
        'validate_container_contract "$runtime_receipts/container-inspect-post.json" exited 0 true',
        'rustinfer.remote-release-soak.completed.v1 >"$output_dir/completed"',
    ):
        _require_once(code, marker, "remote soak runner")

    _require_in_order(
        code,
        (
            "fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "require_gpu_idle initial-preflight",
            'mkdir -m 0700 "$output_dir"',
            'snapshot_regular_file "$source_archive_input" "$source_archive"',
            'cp --recursive --no-preserve=mode,ownership,timestamps -- "$model_dir_input/." "$model_snapshot/"',
            'archive_revision=$(git get-tar-commit-id <"$source_archive")',
            'tar --extract',
            'docker image inspect "$release_image_id" >"$runtime_receipts/release-image-inspect.json"',
            'release image has an unsafe or unexpected inherited build environment',
            'cmp --silent "$build_context_pre_manifest" "$build_context_immediate_manifest"',
            "DOCKER_BUILDKIT=1 docker build",
            'container_id=$(docker create',
            'validate_container_contract "$runtime_receipts/container-inspect-pre.json" created 0 false',
            'docker cp "$container_name:/opt/rustinfer/bin/rustinfer" "$container_binary_copy"',
            '"$container_name:/opt/rustinfer-soak/release-runtime-closure.tsv"',
            'release_runtime_closure_sha256=$(sha256_file "$runtime_closure_receipt")',
            'verify_input_snapshots immediate-pre-start',
            "require_gpu_idle immediate-pre-start",
            'docker start "$container_name" >/dev/null',
            'docker wait "$container_name" >"$output_dir/container-exit-code.txt"',
            'docker inspect "$container_name" >"$runtime_receipts/container-inspect-post.json"',
            'validate_container_contract "$runtime_receipts/container-inspect-post.json" exited 0 true',
            'verify_input_snapshots post',
            'run_json_sha256=$(sha256_file "$container_evidence_export/run/run.json")',
            'events_jsonl_sha256=$(sha256_file "$container_evidence_export/run/events.jsonl")',
            '>"$runtime_receipts/launcher-receipt.json"',
            "actual_runtime_receipts=",
            ">launcher-SHA256SUMS",
            'rustinfer.remote-release-soak.completed.v1 >"$output_dir/completed"',
        ),
        "remote soak runner",
    )

    network_options = re.findall(r"(?<![\w-])--network(?:=|\s+)([^\s\\]+)", code)
    if network_options != ["none"]:
        _fail(f"remote soak runner network option is not exactly none: {network_options}")
    pid_options = re.findall(r"(?<![\w-])--pid(?:=|\s+)([^\s\\]+)", code)
    if pid_options != ["host"]:
        _fail(f"remote soak runner PID option is not exactly host: {pid_options}")
    user_options = re.findall(r"(?<![\w-])--user(?:=|\s+)([^\s\\]+)", code)
    if user_options != ["65532:65532"]:
        _fail(f"remote soak runner USER option is not exact: {user_options}")

    for pattern in (
        r"(?<![\w-])--privileged(?:\s|$)",
        r"\bdocker\s+(?:container\s+)?rm\b",
        r"(?<![\w-])--rm(?:\s|$)",
        r"\bdocker\s+network\s+connect\b",
        r"source=\$\{model_dir\},destination=/model",
        r'DeviceRequests\[0\]\.Driver\s*==\s*"nvidia"',
        r'DeviceRequests\[0\]\.Count\s*==\s*-1',
        r'exec\s+9>>',
        r'flock\s+-n\s+9',
        r'os\.set_inheritable\s*\(',
        r'RUSTINFER_SOAK_GPU_LOCK_FD',
    ):
        if re.search(pattern, code):
            _fail(f"remote soak runner contains forbidden operation: {pattern}")


def verify_release_soak_driver(path: Path = DRIVER) -> None:
    contents = path.read_text(encoding="utf-8")
    if hashlib.sha256(contents.encode("utf-8")).hexdigest() != EXPECTED_RELEASE_DRIVER_SHA256:
        _fail("release soak driver differs from the reviewed exact source digest")
    code = _shell_code(contents)
    if not contents.startswith("#!/usr/bin/bash\n"):
        _fail("release soak driver must use the absolute reviewed Bash shebang")
    required_markers = (
        "export PATH=/usr/bin:/bin",
        "export HOME=/nonexistent CURL_HOME=/nonexistent LC_ALL=C TZ=UTC",
        "curl --disable",
        "golden_profile=$(jq -er '.golden.request_profile",
        "golden_preflight_complete=0",
        'generated=$(probe_hash "$golden_profile")',
        'if [ "$generated" != "$golden_generated_sha256" ]; then',
        'stage:"golden-preflight"',
        '[ "$action" = normal ] && [ "$profile" = "$golden_profile" ]',
        "semantic_mismatch=1",
        'stage:"golden-request"',
        'jq -cS --arg profile "$profile" --argjson request_stream "$request_stream"',
        '.stream = $request_stream',
        '--max-time 0.05 -o "$output"',
        '--no-buffer --max-time 300 --limit-rate 1K',
        '| head -c 1024 >"$output"',
        'curl_code=${pipeline_codes[0]}',
        'head_code=${pipeline_codes[1]}',
        '[ "$curl_code" -eq 28 ]',
        '[ "$curl_code" -eq 23 ] && [ "$head_code" -eq 0 ]',
        '[ "$response_bytes" -eq 1024 ]',
        'request_profile:$request_profile,client_action:$client_action,request_stream:$request_stream,curl_exit_code:$curl_exit_code',
        'request_body_sha256:$request_body_sha256,response_body_sha256:$response_body_sha256,response_bytes:$response_bytes',
        "soak_parent_pid=$BASHPID",
        'started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)',
        'run_stamp=${started_at_utc//-/}',
        'run_stamp=${run_stamp//:/}',
        'run_id="soak-${run_stamp}-${RUSTINFER_SOURCE_REVISION:0:12}"',
        '--arg started_at_utc "$started_at_utc"',
        "trap handle_sampler_failure USR1",
        'kill -USR1 "$soak_parent_pid"',
        "trap 'exit 0' TERM INT",
        'trap \'sampler_status=$?; if [ "$sampler_status" -ne 0 ]; then kill -USR1 "$soak_parent_pid" 2>/dev/null || true; fi\' EXIT',
        "stop_sampler_planned()",
        'kill -TERM "$sampler_pid"',
        'wait "$sampler_pid" || status=$?',
        'message "foreign GPU compute PID is present: pid=$gpu_pid"',
        "require_post_shutdown_gpu_idle()",
        "--query-compute-apps=pid --format=csv,noheader,nounits",
        'stage:"post-shutdown-nvidia-smi"',
        "trap - EXIT USR1",
    )
    for marker in required_markers:
        if marker not in code:
            _fail(f"release soak driver lacks required marker: {marker}")

    curl_commands = re.findall(r"(?m)(?:until\s+|\$\()(curl\s+[^\n]+)", code)
    if not curl_commands or any(not command.startswith("curl --disable ") for command in curl_commands):
        _fail("every release soak curl invocation must disable user configuration first")

    if code.count('kill -USR1 "$soak_parent_pid"') != 1:
        _fail("release soak sampler must propagate every nonzero sampler exit")
    if code.count('pipeline_codes=("${PIPESTATUS[@]}")') != 2:
        _fail("disconnect must capture both Bash pipeline statuses immediately")
    for branch in re.findall(
        r"(?ms)(?:then|else)\n\s+(pipeline_codes=\(\"\$\{PIPESTATUS\[@\]\}\"\))",
        code,
    ):
        if branch != 'pipeline_codes=("${PIPESTATUS[@]}")':
            _fail("disconnect pipeline status capture is not immediate")
    _require_in_order(
        code,
        (
            '| head -c 1024 >"$output"',
            'pipeline_codes=("${PIPESTATUS[@]}")',
            'curl_code=${pipeline_codes[0]}',
            'head_code=${pipeline_codes[1]}',
            '[ "$curl_code" -eq 23 ] && [ "$head_code" -eq 0 ]',
        ),
        "release soak disconnect transport",
    )
    if code.count('launch_target "$mode" "$id"') != 2:
        _fail("release soak driver must bind initial and restart launches")
    _require_in_order(
        code,
        (
            'generated=$(probe_hash "$golden_profile")',
            "golden_preflight_complete=1",
            "sampler_loop()",
            "stop_sampler_planned()",
            "run_request()",
            "semantic_mismatch=1",
            'stage:"golden-request"',
            "require_post_shutdown_gpu_idle()",
            "require_post_shutdown_gpu_idle",
            'append_event \'{"kind":"run_end"',
        ),
        "release soak driver",
    )

    failure_literals = re.findall(
        r'\{(?:kind:"failure"|"kind":"failure")[^}]*\}', code
    )
    if not failure_literals:
        _fail("release soak driver lacks closed failure evidence")
    for literal in failure_literals:
        if literal.startswith('{kind:"failure"'):
            keys = re.findall(r'(?:\{|,)([a-z_]+):', literal)
        else:
            keys = re.findall(r'(?:\{|,)"([a-z_]+)":', literal)
        if keys != ["kind", "scenario_id", "stage", "message"]:
            _fail(f"release soak failure payload is open: {keys}")


def main() -> int:
    try:
        verify_soak_dockerfile()
        verify_remote_soak_runner()
        verify_release_soak_driver()
    except (OSError, ReleaseContractError) as error:
        print(f"reliability soak test-layer verification failed: {error}", file=os.sys.stderr)
        return 1
    print("reliability soak test-layer contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
