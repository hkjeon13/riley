#!/usr/bin/env python3
"""CPU-only adversarial tests for the reliability-soak execution boundary."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import verify_soak_test_layer as soak_contract
from release_common import ReleaseContractError
from verify_soak_test_layer import (
    DOCKERFILE,
    DRIVER,
    RUNNER,
    verify_release_soak_driver,
    verify_remote_soak_runner,
    verify_soak_dockerfile,
)


class ReliabilitySoakDockerfileTests(unittest.TestCase):
    def test_reviewed_dockerfile_passes(self) -> None:
        verify_soak_dockerfile()

    def test_base_packages_payload_and_identity_are_closed(self) -> None:
        original = DOCKERFILE.read_text(encoding="utf-8")
        mutations = {
            "mutable-syntax-frontend": "# syntax=docker/dockerfile:latest\n" + original,
            "default-base": original.replace(
                "ARG RILEY_RELEASE_IMAGE_REF\n",
                "ARG RILEY_RELEASE_IMAGE_REF=ubuntu:22.04\n",
                1,
            ),
            "fixed-untrusted-base": original.replace(
                "FROM ${RILEY_RELEASE_IMAGE_REF} AS reliability-soak-test-layer",
                "FROM ubuntu:22.04 AS reliability-soak-test-layer",
                1,
            ),
            "python-package": original.replace("        jq \\\n", "        jq \\\n        python3 \\\n", 1),
            "missing-gawk": original.replace("        gawk \\\n", "", 1),
            "package-upgrade-enabled": original.replace(" --no-upgrade", "", 1),
            "closure-self-comparison": original.replace(
                'cmp --silent "$closure_before" "$closure_after"',
                'cmp --silent "$closure_after" "$closure_after"',
                1,
            ),
            "closure-not-captured-before-install": original.replace(
                'capture_runtime_closure "$closure_before";',
                ': >"$closure_before";',
                1,
            ),
            "closure-target-bytes-unhashed": original.replace(
                'dependency_sha256=$(sha256sum "$dependency_target");',
                'dependency_sha256=$(printf %064d 0);',
                1,
            ),
            "closure-unresolved-dependency-omitted": original.replace(
                "printf '%s\\tNOT_FOUND\\t-\\t-\\n'",
                "printf '%s\\tIGNORED\\t-\\t-\\n'",
                1,
            ),
            "closure-unreviewed-unresolved-dependency-allowed": original.replace(
                'test "$dependency_name" = libcuda.so.1',
                "true",
                1,
            ),
            "closure-unresolved-count-unchecked": original.replace(
                'test "$unresolved_count" -eq 1;',
                "true;",
                1,
            ),
            "closure-address-retained": original.replace(
                '"$dependency_sha256" >>"$closure_unsorted";',
                '"$dependency_sha256" "$_" >>"$closure_unsorted";',
                1,
            ),
            "closure-receipt-not-kept": original.replace(
                'chmod 0444 "$closure_before";',
                'rm -f "$closure_before";',
                1,
            ),
            "binary-substitution": original.replace(
                "COPY ci/run_release_soak.sh",
                "COPY target/release/riley /opt/riley/bin/riley\n"
                "COPY ci/run_release_soak.sh",
                1,
            ),
            "missing-release-label": original.replace(
                'org.riley.reliability-soak.release-image-id="${RILEY_RELEASE_IMAGE_ID}" \\\n      ',
                "",
                1,
            ),
            "root-runtime": original.replace("USER 65532:65532", "USER 0:0", 1),
            "production-entrypoint": original.replace(
                '["/opt/riley-soak/ci/run_release_soak.sh"]',
                '["/opt/riley/bin/riley"]',
                1,
            ),
            "reordered-environment": original.replace(
                "ENV LC_ALL=C\nENV TZ=UTC",
                "ENV TZ=UTC\nENV LC_ALL=C",
                1,
            ),
            "duplicate-final-user": original.replace(
                "USER 65532:65532\nENTRYPOINT",
                "USER 65532:65532\nUSER 65532:65532\nENTRYPOINT",
                1,
            ),
            "duplicate-final-entrypoint": original.replace(
                "CMD []",
                'ENTRYPOINT ["/opt/riley-soak/ci/run_release_soak.sh"]\nCMD []',
                1,
            ),
            "duplicate-final-cmd": original.replace("CMD []", "CMD []\nCMD []", 1),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(contents, original)
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "Dockerfile"
                    candidate.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ReleaseContractError):
                        verify_soak_dockerfile(candidate)

    def test_runtime_closure_guards_do_not_rely_only_on_the_source_digest(self) -> None:
        original = DOCKERFILE.read_text(encoding="utf-8")
        mutations = {
            "upgrade": original.replace(" --no-upgrade", "", 1),
            "self-comparison": original.replace(
                'cmp --silent "$closure_before" "$closure_after"',
                'cmp --silent "$closure_after" "$closure_after"',
                1,
            ),
            "missing-pre-capture": original.replace(
                'capture_runtime_closure "$closure_before";',
                ': >"$closure_before";',
                1,
            ),
            "unresolved-omitted": original.replace(
                "printf '%s\\tNOT_FOUND\\t-\\t-\\n'",
                "printf '%s\\tIGNORED\\t-\\t-\\n'",
                1,
            ),
            "unreviewed-unresolved-allowed": original.replace(
                'test "$dependency_name" = libcuda.so.1',
                "true",
                1,
            ),
            "unresolved-count-unchecked": original.replace(
                'test "$unresolved_count" -eq 1;',
                "true;",
                1,
            ),
            "target-bytes-unhashed": original.replace(
                'dependency_sha256=$(sha256sum "$dependency_target");',
                'dependency_sha256=$(printf %064d 0);',
                1,
            ),
            "ASLR-address-retained": original.replace(
                '"$dependency_sha256" >>"$closure_unsorted";',
                '"$dependency_sha256" "$_" >>"$closure_unsorted";',
                1,
            ),
            "receipt-deleted": original.replace(
                'chmod 0444 "$closure_before";',
                'rm -f "$closure_before";',
                1,
            ),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                normalized = (
                    "\n".join(soak_contract._instructions(contents)) + "\n"
                ).encode("utf-8")
                candidate_digest = hashlib.sha256(normalized).hexdigest()
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "Dockerfile"
                    candidate.write_text(contents, encoding="utf-8")
                    with mock.patch.object(
                        soak_contract,
                        "EXPECTED_NORMALIZED_INSTRUCTION_SHA256",
                        candidate_digest,
                    ):
                        with self.assertRaises(ReleaseContractError):
                            verify_soak_dockerfile(candidate)


class RemoteReliabilitySoakRunnerTests(unittest.TestCase):
    def test_reviewed_runner_passes(self) -> None:
        verify_remote_soak_runner()

    def test_host_tool_inventory_and_absolute_dispatch_are_closed(self) -> None:
        original = RUNNER.read_text(encoding="utf-8")
        mutations = {
            "path-dispatched-jq": original.replace("jq -e ", "command jq -e ", 1),
            "bootstrap-python-loads-site": original.replace(
                "/usr/bin/python3.10 -I -S -c '",
                "/usr/bin/python3.10 -c '",
                1,
            ),
            "snapshot-python-loads-site": original.replace(
                '"$PYTHON_BIN" -I -S - "$1" "$2"',
                '"$PYTHON_BIN" - "$1" "$2"',
                1,
            ),
            "unreviewed-path-dispatched-command": (
                original + "\nsed -n '1p' /etc/hosts\n"
            ),
            "unlocked-jq-wrapper": original.replace(
                " jq mawk mkdir nvidia-smi od sha256sum",
                " mawk mkdir nvidia-smi od sha256sum",
                1,
            ),
            "find-exec-path-lookup": original.replace(
                '-exec "$CHMOD_BIN" 0444', "-exec chmod 0444", 1
            ),
            "closure-receipt-not-from-created-container": original.replace(
                '"$container_name:/opt/riley-soak/release-runtime-closure.tsv"',
                '"$test_image_tag:/opt/riley-soak/release-runtime-closure.tsv"',
                1,
            ),
            "closure-receipt-not-canonical": original.replace(
                'sort -u "$runtime_closure_receipt" | cmp --silent - "$runtime_closure_receipt"',
                "true",
                1,
            ),
            "closure-unresolved-row-not-closed": original.replace(
                'if ($1 != "libcuda.so.1" || $3 != "-" || $4 != "-") exit 1',
                'if ($1 != "libcuda.so.1") exit 1',
                1,
            ),
            "closure-unresolved-set-not-exact": original.replace(
                "unresolved != 1",
                "unresolved < 1",
                1,
            ),
            "closure-receipt-not-hashed": original.replace(
                'release_runtime_closure_sha256=$(sha256_file "$runtime_closure_receipt")',
                "release_runtime_closure_sha256=$(printf %064d 0)",
                1,
            ),
            "missing-root-owner-check": original.replace(
                'test "$("$STAT_BIN" -c \'%u\' -- "$tool_path")" = 0',
                "true",
                1,
            ),
            "missing-group-world-mode-check": original.replace(
                "(( (8#$tool_mode & 8#022) == 0 ))", "true", 1
            ),
            "missing-tool-digest-check": original.replace(
                'test "$("$SHA256SUM_BIN" "$tool_path" | "$MAWK_BIN" '
                '\'{print $1}\')" = "$tool_sha256"',
                "true",
                1,
            ),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(contents, original)
                candidate_sha256 = hashlib.sha256(contents.encode("utf-8")).hexdigest()
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "runner.sh"
                    candidate.write_text(contents, encoding="utf-8")
                    with mock.patch.object(
                        soak_contract,
                        "EXPECTED_REMOTE_RUNNER_SHA256",
                        candidate_sha256,
                    ):
                        with self.assertRaises(ReleaseContractError):
                            verify_remote_soak_runner(candidate)

    def test_namespace_network_and_persistence_are_closed(self) -> None:
        original = RUNNER.read_text(encoding="utf-8")
        mutations = {
            "env-shebang": original.replace("#!/usr/bin/bash", "#!/usr/bin/env bash", 1),
            "symlink-awk-tool": original.replace(
                "mawk|/usr/bin/mawk|dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e",
                "mawk|/usr/bin/awk|dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e",
                1,
            ),
            "missing-reviewed-tool-nonlink": original.replace(
                'test -f "$tool_path" && test ! -L "$tool_path" && test -x "$tool_path"',
                'test -f "$tool_path" && test -x "$tool_path"',
                1,
            ),
            "unreviewed-absolute-tool": original + "\n: /usr/bin/true\n",
            "host-awk-not-mawk": original.replace(
                "mawk '{$1=$1;print}'", "awk '{$1=$1;print}'", 1
            ),
            "bridge-network": original.replace("--network none", "--network bridge", 1),
            "late-network-connect": original.replace(
                'docker start "$container_name" >/dev/null',
                'docker network connect bridge "$container_name"\n'
                'docker start "$container_name" >/dev/null',
                1,
            ),
            "private-pid": original.replace("--pid host", "--pid private", 1),
            "privileged": original.replace("--read-only", "--privileged --read-only", 1),
            "missing-shared-lock-nofollow": original.replace("os.O_NOFOLLOW", "0", 1),
            "missing-shared-lock-nonblock": original.replace("os.O_NONBLOCK", "0", 1),
            "missing-shared-lock-fstat": original.replace(
                "metadata = os.fstat(descriptor)", "metadata = os.stat(lock_path)", 1
            ),
            "relaxed-shared-lock-mode": original.replace(
                "stat.S_IMODE(metadata.st_mode) != 0o600",
                "metadata.st_mode & 0o022",
                1,
            ),
            "shared-lock-inherited-by-launcher": original.replace(
                "os.close(descriptor)",
                "os.set_inheritable(descriptor, True)",
                1,
            ),
            "forged-supervisor-environment": original.replace(
                'test "$PPID" = "$RILEY_SOAK_SUPERVISOR_PID"',
                "true",
                1,
            ),
            "missing-kernel-flock-proof": original.replace(
                'if re.search(lock_pattern, fdinfo, re.MULTILINE) is None:',
                "if False:",
                1,
            ),
            "missing-parent-death-signal": original.replace(
                "libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)",
                "0",
                1,
            ),
            "missing-pre-start-idle-check": original.replace(
                "require_gpu_idle immediate-pre-start",
                "true",
                1,
            ),
            "late-input-snapshot-recheck": original.replace(
                "verify_input_snapshots immediate-pre-start\n",
                "true\n",
                1,
            ).replace(
                'docker start "$container_name" >/dev/null',
                'docker start "$container_name" >/dev/null\nverify_input_snapshots immediate-pre-start',
                1,
            ),
            "mutable-image": original.replace(
                'test "$resolved_release_image_id" = "$release_image_id"',
                "true",
                1,
            ),
            "host-user-override": original.replace(
                "--user 65532:65532",
                '--user "$(id -u):$(id -g)"',
                1,
            ),
            "missing-tmpfs": original.replace(
                "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=67108864 \\\n",
                "",
                1,
            ),
            "top-level-evidence-mount": original.replace(
                "source=${container_evidence},destination=/evidence",
                "source=${output_dir},destination=/evidence",
                1,
            ),
            "self-authorized-golden": original.replace(
                "--expected-correctness-golden-sha256",
                "--untrusted-correctness-golden-sha256",
            ),
            "legacy-builder": original.replace(
                "DOCKER_BUILDKIT=1 docker build",
                "DOCKER_BUILDKIT=0 docker build",
                1,
            ),
            "late-build-context-recheck": original.replace(
                'cmp --silent "$build_context_pre_manifest" "$build_context_immediate_manifest"',
                "true",
                1,
            ).replace(
                'post_build_base_image_id=',
                'cmp --silent "$build_context_pre_manifest" "$build_context_immediate_manifest"\n'
                'post_build_base_image_id=',
                1,
            ),
            "checkout-dockerfile": original.replace(
                '--file "$build_context/ci/release/ReliabilitySoak.Dockerfile"',
                '--file "$repository_root/ci/release/ReliabilitySoak.Dockerfile"',
                1,
            ),
            "checkout-build-context": original.replace(
                '"$build_context" 2>&1 | tee "$output_dir/test-layer-build.log"',
                '. 2>&1 | tee "$output_dir/test-layer-build.log"',
                1,
            ),
            "compressed-source-archive": original.replace(
                'echo "source archive must be an uncompressed tar stream"',
                'echo "compressed source archives are accepted"',
                1,
            ),
            "missing-ustar-validation": original.replace(
                'test "$archive_ustar_magic" = 757374617200',
                "true",
                1,
            ),
            "digest-free-base-tag": original.replace(
                'base_image_tag="riley-soak-release-base:${resolved_revision}-${release_image_id#sha256:}"',
                'base_image_tag="riley-soak-release-base:${resolved_revision}"',
                1,
            ),
            "unverified-base-after-build": original.replace(
                'test "$post_build_base_image_id" = "$release_image_id"',
                "true",
                1,
            ),
            "unbound-rootfs": original.replace(
                '$test_layer[0:($release | length)] == $release',
                "true",
                1,
            ),
            "writable-model": original.replace(
                "source=${model_snapshot},destination=/model,readonly",
                "source=${model_snapshot},destination=/model",
                1,
            ),
            "original-model-mounted": original.replace(
                "source=${model_snapshot},destination=/model,readonly",
                "source=${model_dir},destination=/model,readonly",
                1,
            ),
            "missing-model-post-hash": original.replace(
                "verify_input_snapshots post",
                "true",
                1,
            ),
            "source-snapshot-bypass": original.replace(
                'snapshot_regular_file "$source_archive_input" "$source_archive"',
                'source_archive="$source_archive_input"',
                1,
            ),
            "source-snapshot-not-create-only": original.replace(
                "os.O_EXCL", "0", 1
            ),
            "relaxed-image-command": original.replace(
                'and .[0].Config.Cmd == []',
                "and true",
                1,
            ),
            "relaxed-oom-state": original.replace(
                'and $container.State.OOMKilled == false',
                "and true",
                1,
            ),
            "relaxed-network-inventory": original.replace(
                'and (($container.NetworkSettings.Networks // {}) | keys) == ["none"]',
                "and true",
                1,
            ),
            "relaxed-security": original.replace(
                '$container.HostConfig.SecurityOpt == ["no-new-privileges:true"]',
                "true",
                1,
            ),
            "permissive-device-driver": original.replace(
                '$container.HostConfig.DeviceRequests[0].Driver == ""',
                '($container.HostConfig.DeviceRequests[0].Driver == "" or '
                '$container.HostConfig.DeviceRequests[0].Driver == "nvidia")',
                1,
            ),
            "permissive-device-count": original.replace(
                '$container.HostConfig.DeviceRequests[0].Count == 0',
                '($container.HostConfig.DeviceRequests[0].Count == 0 or '
                '$container.HostConfig.DeviceRequests[0].Count == -1)',
                1,
            ),
            "missing-healthcheck-contract": original.replace(
                '$container.Config.Healthcheck == {Test:["NONE"]}', "true", 1
            ),
            "missing-container-path": original.replace(
                '$container.Path == "/opt/riley-soak/ci/run_release_soak.sh"',
                "true",
                1,
            ),
            "missing-container-args": original.replace(
                "$container.Args == []", "true", 1
            ),
            "missing-host-devices": original.replace(
                "$container.HostConfig.Devices == []", "true", 1
            ),
            "relaxed-ipc-mode": original.replace(
                '$container.HostConfig.IpcMode == "private"', "true", 1
            ),
            "missing-mount-propagation": original.replace(
                'Propagation:"rprivate"', 'Propagation:"shared"'
            ),
            "relaxed-runtime-duration": original.replace(">= 26100", ">= 0", 1),
            "missing-release-env-preflight": original.replace(
                '$name == "LD_PRELOAD"', '$name == "IGNORED_PRELOAD"', 1
            ),
            "unsafe-release-home": original.replace(
                '$name == "HOME"', '$name == "IGNORED_HOME"', 1
            ),
            "unsafe-release-xdg-config": original.replace(
                '$name == "XDG_CONFIG_HOME"', '$name == "IGNORED_XDG"', 1
            ),
            "dead-permissive-device-branch": original
            + '\n: \'$container.HostConfig.DeviceRequests[0].Driver == "nvidia"\'\n',
            "dead-marker-cannot-authorize-live-bypass": original.replace(
                '$container.HostConfig.DeviceRequests[0].Driver == ""',
                "true",
                1,
            )
            + '\n: \'$container.HostConfig.DeviceRequests[0].Driver == ""\'\n',
            "unbound-container-environment": original.replace(
                'and ($container.Config.Env | environment_map) == $expected_environment',
                "and true",
                1,
            ),
            "unverified-container-binary": original.replace(
                'test "$(sha256_file "$container_binary_copy")" = "$expected_release_binary_sha256"',
                "true",
                1,
            ),
            "open-launcher-receipt": original.replace(
                'images:{release_image_id:$release_image_id,test_layer_image_id:$test_layer_image_id}',
                'images:{release_image_id:$release_image_id,test_layer_image_id:$test_layer_image_id,tag:$test_image_tag}',
                1,
            ),
            "missing-runtime-receipt": original.replace(
                "    host-gpu.csv \\\n",
                "",
                1,
            ),
            "completed-before-checksum": original.replace(
                'printf \'%s\\n\' riley.remote-release-soak.completed.v1 >"$output_dir/completed"',
                "true",
                1,
            ).replace(
                "(\n    cd \"$output_dir\"",
                'printf \'%s\\n\' riley.remote-release-soak.completed.v1 >"$output_dir/completed"\n(\n    cd "$output_dir"',
                1,
            ),
            "ephemeral-container": original.replace(
                "docker wait",
                "docker container rm\ndocker wait",
                1,
            ),
            "discarded-export": original.replace(
                'docker cp "$container_name:/evidence/." "$container_evidence_export"',
                "true",
                1,
            ),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(contents, original)
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "runner.sh"
                    candidate.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ReleaseContractError):
                        verify_remote_soak_runner(candidate)


class ReliabilitySoakDriverTests(unittest.TestCase):
    def test_reviewed_driver_passes(self) -> None:
        verify_release_soak_driver()

    def test_disconnect_pipeline_status_capture_is_exact(self) -> None:
        script = r'''
set -uo pipefail
if "$BASH" -c "printf '%02048d' 0; exit 23" | head -c 1024 >"$1"; then
    pipeline_codes=("${PIPESTATUS[@]}")
else
    pipeline_codes=("${PIPESTATUS[@]}")
fi
response_bytes=$(wc -c <"$1")
response_bytes=$((response_bytes))
printf '%s %s %s\n' "${pipeline_codes[0]}" "${pipeline_codes[1]}" "$response_bytes"
'''
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "limited-response"
            completed = subprocess.run(
                ["/bin/bash", "-c", script, "bash", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.stdout, "23 0 1024\n")

    def test_golden_sampler_and_gpu_failures_are_closed(self) -> None:
        original = DRIVER.read_text(encoding="utf-8")
        mutations = {
            "env-shebang": original.replace("#!/usr/bin/bash", "#!/usr/bin/env bash", 1),
            "open-path": original.replace(
                "export PATH=/usr/bin:/bin", "export PATH=/tmp:/usr/bin:/bin", 1
            ),
            "unsafe-home": original.replace(
                "HOME=/nonexistent", "HOME=/tmp", 1
            ),
            "unsafe-curl-home": original.replace(
                "CURL_HOME=/nonexistent", "CURL_HOME=/tmp", 1
            ),
            "curl-config-enabled": original.replace("curl --disable", "curl", 1),
            "cancel-is-streaming": original.replace(
                "normal|invalid|overload|cancel) request_stream=false",
                "normal|invalid|overload) request_stream=false\n        cancel) request_stream=true",
                1,
            ),
            "disconnect-is-timeout": original.replace(
                '[ "$curl_code" -eq 23 ] && [ "$head_code" -eq 0 ]',
                '[ "$curl_code" -eq 28 ] && [ "$head_code" -eq 0 ]',
                1,
            ),
            "disconnect-does-not-limit-bytes": original.replace(
                '| head -c 1024 >"$output"', '| /bin/cat >"$output"', 1
            ),
            "late-pipeline-status-capture": original.replace(
                'pipeline_codes=("${PIPESTATUS[@]}")',
                'true\n            pipeline_codes=("${PIPESTATUS[@]}")',
                1,
            ),
            "unbound-request-body": original.replace(
                'request_body_sha256=$(printf \'%s\' "$body" | sha256sum',
                'request_body_sha256=$(printf \'%s\' "$profile" | sha256sum',
                1,
            ),
            "missing-first-golden-probe": original.replace(
                'generated=$(probe_hash "$golden_profile")',
                'generated="$golden_generated_sha256"',
                1,
            ),
            "missing-first-golden-compare": original.replace(
                '[ "$generated" != "$golden_generated_sha256" ]',
                "false",
                1,
            ),
            "golden-request-not-normal-only": original.replace(
                '[ "$action" = normal ] && [ "$profile" = "$golden_profile" ]',
                '[ "$profile" = "$golden_profile" ]',
                1,
            ),
            "missing-golden-request-failure": original.replace(
                'stage:"golden-request"',
                'stage:"request"',
                1,
            ),
            "sampler-error-not-propagated": original.replace(
                'kill -USR1 "$soak_parent_pid"',
                "true",
                1,
            ),
            "planned-stop-is-an-error": original.replace(
                "trap 'exit 0' TERM INT",
                "trap 'exit 1' TERM INT",
                1,
            ),
            "foreign-gpu-pid-accepted": original.replace(
                '--arg message "foreign GPU compute PID is present: pid=$gpu_pid"',
                '--arg message "foreign GPU compute PID ignored: pid=$gpu_pid"',
                1,
            ),
            "missing-final-gpu-idle-check": original.replace(
                "require_post_shutdown_gpu_idle\nfinal_metrics=",
                "true\nfinal_metrics=",
                1,
            ),
            "open-failure-payload": original.replace(
                '{kind:"failure",scenario_id:$scenario_id,stage:"nvidia-smi",message:$message}',
                '{kind:"failure",scenario_id:$scenario_id,stage:"nvidia-smi",message:$message,foreign_pid:$gpu_pid}',
                1,
            ),
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(contents, original)
                with tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "driver.sh"
                    candidate.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ReleaseContractError):
                        verify_release_soak_driver(candidate)


if __name__ == "__main__":
    unittest.main()
