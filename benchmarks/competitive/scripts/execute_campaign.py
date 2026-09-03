#!/usr/bin/env python3
"""Consume an immutable C01 plan through an injected process transport.

This is deliberately an adapter library, not a remote benchmark launcher.
Callers provide a transport that implements ``InvocationExecutor``; production
SSH/container/GPU transport is outside C01 source work.  The adapter accepts
no command, matrix, threshold, or environment override.  It only reads the
hashed plan and its fully materialized lane files, then appends one terminal
raw row per planned invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import check_campaign
from competitive_common import (
    ContractError,
    canonical_json_bytes,
    load_json,
    require_campaign_artifact_path,
    sha256_bytes,
    sha256_file,
    validate_lane,
    verify_campaign_lane_binding,
)
from materialize_lane import verify_campaign_lane_binding_value
from raw_journal import AppendOnlyRawJournal


SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[2]


class TransientStartError(RuntimeError):
    """A transport may retry this only before a measured process exists."""


@dataclass(frozen=True)
class ProcessCompletion:
    """A terminal process result and the engine-produced observation payload.

    ``observation`` is intentionally limited to output evidence.  It cannot
    override plan identity, command, lane, workload, or preflight bindings.
    """

    returncode: int
    recorded_at_utc: str
    observation: Mapping[str, Any] | None
    stdout: bytes = b""
    stderr: bytes = b""


class ProcessHandle(Protocol):
    """The small lifecycle surface required from a remote/local transport."""

    @property
    def environment(self) -> Mapping[str, Any]: ...

    def wait(self, timeout_seconds: float) -> ProcessCompletion | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class InvocationContext:
    """Immutable transport input derived only from the immutable plan."""

    campaign_id: str
    campaign_plan_sha256: str
    invocation_id: str
    execution_id: str
    cell_id: str
    run_index: int
    order: str
    position: str
    role: str
    lane_id: str
    command_argv: tuple[str, ...]
    workload_sha256: str
    workload: Mapping[str, Any]


class InvocationExecutor(Protocol):
    """Injectable transport boundary; no network implementation is shipped."""

    def start(self, context: InvocationContext) -> ProcessHandle: ...


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{label} must stay inside repository root") from error
    return path


def _tokenizer_identity_sha256(model_identity: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "tokenizer_revision": model_identity["tokenizer_revision"],
                "tokenizer_files_sha256": model_identity["tokenizer_files_sha256"],
                "tokenizer_aggregate_sha256": model_identity.get("tokenizer_aggregate_sha256"),
            }
        )
    )


def _weight_sha256(model_identity: Mapping[str, Any]) -> str:
    for field in ("weights_sha256", "model_weights_sha256"):
        if field in model_identity:
            return str(model_identity[field])
    raise ContractError("plan request manifest lacks a model weight hash")


def _command_sha256(lane: Mapping[str, Any]) -> str:
    command = lane["command"]
    assert isinstance(command, Mapping)
    return sha256_bytes(canonical_json_bytes(command["argv"]))


def _load_executable_lanes(
    *,
    root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    lanes: dict[str, Mapping[str, Any]] = {}
    manifest_model = plan["request_manifest"]["model_identity"]
    assert isinstance(manifest_model, Mapping)
    expected_model_hash = sha256_bytes(canonical_json_bytes(manifest_model))
    expected_tokenizer_hash = _tokenizer_identity_sha256(manifest_model)
    for role, expected_role in (("riley", "candidate"), ("competitor", "baseline")):
        receipt = plan["lanes"][role]
        assert isinstance(receipt, Mapping)
        path = _relative_path(root, receipt["path"], f"plan.lanes.{role}.path")
        if sha256_file(path) != receipt["sha256"]:
            raise ContractError(f"plan.lanes.{role} hash drifted before execution")
        lane = validate_lane(load_json(path), str(path))
        verify_campaign_lane_binding(root, path, lane)
        verify_campaign_lane_binding_value(root=root, lane=lane)
        if lane["lane_id"] != receipt["lane_id"] or lane.get("role") != expected_role:
            raise ContractError(f"plan.lanes.{role} identity drifts from materialized lane")
        if lane["availability"] != "available" or lane["command"]["status"] != "available":
            raise ContractError(f"plan.lanes.{role} is not executable")
        materialization = lane.get("materialization")
        assert isinstance(materialization, Mapping)
        if materialization["campaign_id"] != plan["campaign_id"]:
            raise ContractError(f"plan.lanes.{role} materialization campaign differs from plan")
        if materialization["source_git_revision"] != plan["source"]["git_revision"]:
            raise ContractError(f"plan.lanes.{role} materialization source differs from plan")
        command = lane["command"]
        assert isinstance(command, Mapping)
        argv = command["argv"]
        if command.get("required_placeholders") != [] or any("{" in str(item) or "}" in str(item) for item in argv):
            raise ContractError(f"plan.lanes.{role} command is not fully materialized")
        artifact_receipts = lane["artifact_receipts"]
        assert isinstance(artifact_receipts, Mapping)
        if artifact_receipts["model_identity_sha256"] != expected_model_hash:
            raise ContractError(f"plan.lanes.{role} model identity does not match request manifest")
        if artifact_receipts["tokenizer_identity_sha256"] != expected_tokenizer_hash:
            raise ContractError(f"plan.lanes.{role} tokenizer identity does not match request manifest")
        lanes[role] = lane
    return lanes


def _validate_environment(
    *,
    environment: Mapping[str, Any],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    lane: Mapping[str, Any],
) -> None:
    if not isinstance(environment, Mapping):
        raise ContractError("executor process environment must be a mapping")
    required = list(contract["required_environment_keys"])
    if set(environment) != set(required):
        missing = sorted(set(required) - set(environment))
        extra = sorted(set(environment) - set(required))
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ContractError("executor environment must exactly match C01 required keys: " + "; ".join(details))
    for key in required:
        value = environment[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or value == "":
            raise ContractError(f"executor environment.{key} must be a non-empty scalar")
    model = plan["request_manifest"]["model_identity"]
    assert isinstance(model, Mapping)
    engine = lane["engine"]
    receipts = lane["artifact_receipts"]
    assert isinstance(engine, Mapping) and isinstance(receipts, Mapping)
    expected = {
        "git_commit": plan["source"]["git_revision"],
        "source_archive_sha256": receipts["source_or_wheel_sha256"],
        "executable_sha256": receipts["executable_sha256"],
        "dependency_lock_sha256": receipts["dependency_lock_sha256"],
        "lane_command_sha256": _command_sha256(lane),
        "engine_version": engine["version"],
        "engine_revision": engine["revision"],
        "engine_options_sha256": receipts["runtime_options_sha256"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "model_weights_sha256": _weight_sha256(model),
        "tokenizer_revision": model["tokenizer_revision"],
        "tokenizer_files_sha256": model.get("tokenizer_aggregate_sha256")
        or sha256_bytes(canonical_json_bytes(model["tokenizer_files_sha256"])),
    }
    for key, expected_value in expected.items():
        if environment[key] != expected_value:
            raise ContractError(f"executor environment.{key} drifts from plan/materialized lane")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _failure_observation(reason: str) -> dict[str, Any]:
    return {"status": "failure", "failure_reason": reason, "metrics": None, "requests": []}


def _normalise_observation(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("executor completion observation must be a JSON object")
    expected = {"status", "failure_reason", "metrics", "requests"}
    if set(value) != expected:
        raise ContractError("executor completion observation must not override adapter identity fields")
    if value["status"] not in {"success", "failure"}:
        raise ContractError("executor completion observation.status must be success or failure")
    if value["status"] == "success" and value["failure_reason"] is not None:
        raise ContractError("executor success observation must not contain failure_reason")
    if value["status"] == "failure" and (not isinstance(value["failure_reason"], str) or not value["failure_reason"]):
        raise ContractError("executor failure observation requires a failure_reason")
    if value["status"] == "success":
        # Keep malformed engine output out of the journal as a pretend
        # successful measurement.  The checker remains the sole authority on
        # plan/workload/token comparability; this is only structural raw-row
        # validation so a broken transport becomes one terminal failure arm.
        check_campaign._validate_metrics(value["metrics"], "executor completion.metrics")
        if not isinstance(value["requests"], list) or not value["requests"]:
            raise ContractError("executor success completion requires a non-empty requests array")
    elif value["metrics"] is not None:
        raise ContractError("executor failure completion.metrics must be null")
    if not isinstance(value["requests"], list):
        raise ContractError("executor completion.requests must be a JSON array")
    for index, request in enumerate(value["requests"]):
        check_campaign._validate_request(request, f"executor completion.requests[{index}]")
    return value


def _cleanup_timed_out_process(process: ProcessHandle, *, grace_seconds: float) -> None:
    """Terminate, then kill if needed; a surviving process is fail-closed."""

    try:
        process.terminate()
    except Exception as error:  # transport-specific error, followed by kill attempt
        terminate_error: Exception | None = error
    else:
        terminate_error = None
    try:
        completed = process.wait(grace_seconds)
    except Exception as error:
        completed = None
        wait_error: Exception | None = error
    else:
        wait_error = None
    if completed is not None:
        return
    try:
        process.kill()
        completed = process.wait(grace_seconds)
    except Exception as error:
        raise ContractError(f"timed-out process cleanup failed: {error}") from error
    if completed is None:
        details = []
        if terminate_error is not None:
            details.append(f"terminate failed: {terminate_error}")
        if wait_error is not None:
            details.append(f"grace wait failed: {wait_error}")
        suffix = "; ".join(details)
        raise ContractError("timed-out process survived terminate/kill" + (f" ({suffix})" if suffix else ""))


def _run_process(
    *,
    executor: InvocationExecutor,
    context: InvocationContext,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
    max_start_attempts: int,
    now_utc: Callable[[], str],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """Return output-only observation, immutable environment, and timestamp.

    Retrying is allowed only when ``start`` failed before returning a process.
    Once a process exists, timeout/nonzero exit becomes its single terminal raw
    observation; retrying it would launder a failed measured arm.
    """

    process: ProcessHandle | None = None
    for attempt in range(1, max_start_attempts + 1):
        try:
            process = executor.start(context)
            break
        except TransientStartError:
            if attempt == max_start_attempts:
                raise ContractError("executor exhausted pre-start retries") from None
    if process is None:  # defensive guard for malformed executor implementations
        raise ContractError("executor did not return a process")

    # A start failure has no process/environment receipt and therefore cannot
    # safely become a raw row.  We leave the journal incomplete; the checker
    # then emits incomparable rather than accepting a fabricated failure.
    # Environment receipt acquisition is itself inside the lifecycle guard:
    # a transport exception must not leave an already-started remote process
    # running merely because there is no trustworthy raw evidence to append.
    environment: Mapping[str, Any] | None = None
    observation: Mapping[str, Any] | None = None
    recorded_at_utc = now_utc()
    try:
        try:
            environment_value = process.environment
            if not isinstance(environment_value, Mapping):
                raise ContractError("executor process environment must be a mapping")
            environment = environment_value
        except Exception as error:
            try:
                _cleanup_timed_out_process(process, grace_seconds=cleanup_grace_seconds)
            except ContractError as cleanup_error:
                raise ContractError(
                    "executor process environment receipt failed and cleanup could not be confirmed: "
                    f"{cleanup_error}"
                ) from cleanup_error
            raise ContractError(
                f"executor process environment receipt failed before a raw row could be written: {error}"
            ) from error
        try:
            completion = process.wait(timeout_seconds)
        except Exception as error:
            _cleanup_timed_out_process(process, grace_seconds=cleanup_grace_seconds)
            observation = _failure_observation(f"process wait failed: {error}")
        else:
            if completion is None:
                _cleanup_timed_out_process(process, grace_seconds=cleanup_grace_seconds)
                observation = _failure_observation(f"process timeout after {timeout_seconds:g} seconds")
            else:
                candidate_timestamp = getattr(completion, "recorded_at_utc", None)
                if isinstance(candidate_timestamp, str) and candidate_timestamp:
                    recorded_at_utc = candidate_timestamp
                try:
                    if not isinstance(completion.returncode, int) or isinstance(completion.returncode, bool):
                        raise ContractError("executor completion returncode must be an integer")
                    if not isinstance(completion.stdout, bytes) or not isinstance(completion.stderr, bytes):
                        raise ContractError("executor completion stdout/stderr must be bytes")
                    if len(completion.stdout) > 1024 * 1024 or len(completion.stderr) > 1024 * 1024:
                        observation = _failure_observation(
                            "process stdout/stderr exceeded 1048576-byte capture limit"
                        )
                    elif completion.returncode != 0:
                        observation = _failure_observation(f"process exited with status {completion.returncode}")
                    else:
                        observation = _normalise_observation(completion.observation)
                except (AttributeError, ContractError, TypeError) as error:
                    # A process exists and supplied an environment receipt, so this is
                    # safe to record as a failed terminal arm rather than silently
                    # dropping it from the campaign.
                    observation = _failure_observation(f"adapter rejected process completion: {error}")
    finally:
        try:
            process.close()
        except Exception as error:
            # Closing is part of the fresh-process contract.  Once an
            # environment is known, a close failure is valid negative evidence
            # and must not be hidden by a successful engine observation.
            observation = _failure_observation(f"process close failed: {error}")
            recorded_at_utc = now_utc()
    assert environment is not None
    assert observation is not None
    return observation, environment, recorded_at_utc


def _raw_row(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    invocation: Mapping[str, Any],
    workload_receipt: Mapping[str, Any],
    environment: Mapping[str, Any],
    observation: Mapping[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    preflight = plan["preflight"]
    assert isinstance(preflight, Mapping)
    if preflight["status"] != "passed":
        raise ContractError("execution adapter requires a passed preflight receipt")
    cell_entry = next(
        entry for entry in plan["cells"] if entry["cell"]["cell_id"] == invocation["cell_id"]
    )
    workload = workload_receipt["value"]
    assert isinstance(workload, Mapping)
    return {
        "schema_version": "riley.competitive.raw.v1",
        "campaign_id": plan["campaign_id"],
        "campaign_plan_sha256": plan_sha256,
        "invocation_id": invocation["invocation_id"],
        "lane_id": invocation["lane_id"],
        "role": invocation["role"],
        "execution_id": invocation["execution_id"],
        "cell_id": invocation["cell_id"],
        "run_index": invocation["run_index"],
        "order": invocation["order"],
        "position": invocation["position"],
        "measurement_mode": cell_entry["cell"]["measurement_mode"],
        "request_manifest_sha256": plan["request_manifest"]["sha256"],
        "workload_sha256": workload_receipt["sha256"],
        "workload": check_campaign.workload_execution_receipt(workload),
        "recorded_at_utc": recorded_at_utc,
        "source": plan["source"],
        "environment": dict(environment),
        "phase": "measured",
        "status": observation["status"],
        "failure_reason": observation["failure_reason"],
        "metrics": observation["metrics"],
        "requests": observation["requests"],
        "preflight_receipt_sha256": preflight["sha256"],
    }


def execute_plan(
    *,
    plan_path: Path,
    raw_path: Path,
    executor: InvocationExecutor,
    root: Path = REPOSITORY_ROOT,
    timeout_seconds: float = 60.0,
    cleanup_grace_seconds: float = 5.0,
    max_start_attempts: int = 1,
    now_utc: Callable[[], str] = _now_utc,
) -> dict[str, Any]:
    """Run the exact remaining plan schedule and return the existing checker report.

    The function never shells out, opens SSH, or chooses an engine.  A caller
    must deliberately inject a transport.  Existing valid journal rows are
    resumed in order; collisions and any plan/asset drift are rejected.  A
    failure before a process/environment receipt exists, or a timeout whose
    cleanup cannot be confirmed, leaves an incomplete journal intentionally;
    the checker will fail closed as incomparable rather than fabricate raw
    evidence.
    """

    root = root.resolve()
    # Generated plans and raw evidence must live in the one ignored campaign
    # workspace.  Without this guard, an adapter could measure an arm and
    # only later discover that its own untracked output made the source claim
    # dirty/incomparable.
    plan_path = require_campaign_artifact_path(root, plan_path, "execution plan path")
    raw_path = require_campaign_artifact_path(root, raw_path, "adapter raw output path")
    if timeout_seconds <= 0.0 or cleanup_grace_seconds <= 0.0:
        raise ContractError("execution adapter timeouts must be positive")
    if not isinstance(max_start_attempts, int) or isinstance(max_start_attempts, bool) or max_start_attempts < 1:
        raise ContractError("execution adapter max_start_attempts must be an integer >= 1")
    plan_sha256 = sha256_file(plan_path)
    plan = check_campaign.validate_plan(load_json(plan_path), root=root)
    readiness_reasons = check_campaign.rederive_readiness(plan, root=root)
    if readiness_reasons:
        raise ContractError("campaign is not execution-ready: " + "; ".join(sorted(readiness_reasons)))
    lanes = _load_executable_lanes(root=root, plan=plan)
    contract_path = _relative_path(root, plan["contract"]["path"], "plan.contract.path")
    contract = check_campaign.validate_contract(load_json(contract_path), str(contract_path))
    workloads = {str(item["cell_id"]): item for item in plan["workloads"]}
    invocations = sorted(plan["invocations"], key=lambda item: int(item["sequence"]))
    expected_invocation_ids = [str(item["invocation_id"]) for item in invocations]
    journal = AppendOnlyRawJournal(
        path=raw_path,
        plan_sha256=plan_sha256,
        expected_invocation_ids=expected_invocation_ids,
    )
    # The lease starts before selecting the next invocation and outlives the
    # terminal append.  Otherwise two adapters can both read an empty journal
    # and execute the same expensive arm before the loser discovers an append
    # collision.
    with journal.execution_lease():
        completed_count = len(journal._read_rows())  # validated read; private by design to avoid a second parse.
        for invocation in invocations[completed_count:]:
            role = str(invocation["role"])
            lane = lanes[role]
            workload_receipt = workloads[str(invocation["cell_id"])]
            command = lane["command"]
            assert isinstance(command, Mapping)
            context = InvocationContext(
                campaign_id=str(plan["campaign_id"]),
                campaign_plan_sha256=plan_sha256,
                invocation_id=str(invocation["invocation_id"]),
                execution_id=str(invocation["execution_id"]),
                cell_id=str(invocation["cell_id"]),
                run_index=int(invocation["run_index"]),
                order=str(invocation["order"]),
                position=str(invocation["position"]),
                role=role,
                lane_id=str(invocation["lane_id"]),
                command_argv=tuple(str(item) for item in command["argv"]),
                workload_sha256=str(workload_receipt["sha256"]),
                workload=_freeze_json(workload_receipt["value"]),
            )
            observation, environment, recorded_at_utc = _run_process(
                executor=executor,
                context=context,
                timeout_seconds=timeout_seconds,
                cleanup_grace_seconds=cleanup_grace_seconds,
                max_start_attempts=max_start_attempts,
                now_utc=now_utc,
            )
            _validate_environment(
                environment=environment,
                plan=plan,
                contract=contract,
                lane=lane,
            )
            journal.append(
                _raw_row(
                    plan=plan,
                    plan_sha256=plan_sha256,
                    invocation=invocation,
                    workload_receipt=workload_receipt,
                    environment=environment,
                    observation=observation,
                    recorded_at_utc=recorded_at_utc,
                )
            )
        return check_campaign.check_campaign(plan_path=plan_path, raw_paths=[raw_path], root=root)
