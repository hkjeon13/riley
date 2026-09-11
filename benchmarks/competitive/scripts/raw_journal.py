"""Append-only, hash-chained raw JSONL records for the C01 execution adapter.

The checker already validates the semantic content of every raw row.  This
module additionally makes adapter-produced rows tamper-evident in their
physical append order.  It is intentionally not a general evidence store:
one journal corresponds to one immutable execution plan and one ordered list
of invocations.
"""

from __future__ import annotations

import fcntl
import os
import stat
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping, Sequence

from competitive_common import (
    RAW_SCHEMA_VERSION,
    ContractError,
    canonical_json_bytes,
    load_jsonl,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
)


ADAPTER_RECEIPT_SCHEMA_VERSION = "riley.competitive.adapter-receipt.v1"
JOURNAL_FIELDS = (
    "adapter_sequence",
    "adapter_previous_receipt_sha256",
    "adapter_receipt_sha256",
)


# ``flock`` coordinates independently started adapters.  A process-local
# lock is also required because some platforms treat two separately opened
# descriptors in the same process differently from two processes.  The
# adapter must never send the same arm to a GPU twice merely because two
# callers raced before the first terminal raw row was appended.
_PROCESS_LEASE_GUARD = threading.Lock()
_PROCESS_LEASES: dict[Path, threading.Lock] = {}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in JOURNAL_FIELDS}


def adapter_receipt_sha256(
    row: Mapping[str, Any],
    *,
    sequence: int,
    previous_receipt_sha256: str | None,
) -> str:
    """Hash the plan identity, append position, predecessor, and raw payload."""

    if row.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ContractError("adapter journal row must use the C01 raw schema")
    plan_sha = row.get("campaign_plan_sha256")
    if not isinstance(plan_sha, str):
        raise ContractError("adapter journal row is missing campaign_plan_sha256")
    validate_sha256(plan_sha, "adapter journal campaign_plan_sha256")
    invocation_id = row.get("invocation_id")
    validate_identifier(invocation_id, "adapter journal invocation_id")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": ADAPTER_RECEIPT_SCHEMA_VERSION,
                "campaign_plan_sha256": plan_sha,
                "invocation_id": invocation_id,
                "sequence": sequence,
                "previous_receipt_sha256": previous_receipt_sha256,
                "raw_payload_sha256": sha256_bytes(canonical_json_bytes(_payload(row))),
            }
        )
    )


def validate_append_only_chain(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_invocation_ids: Sequence[str] | None = None,
) -> None:
    """Reject mixed, reordered, truncated, or collision-prone adapter rows."""

    if not rows:
        return
    field_presence = [{field for field in JOURNAL_FIELDS if field in row} for row in rows]
    any_journal = any(presence for presence in field_presence)
    if not any_journal:
        return
    if any(presence != set(JOURNAL_FIELDS) for presence in field_presence):
        raise ContractError("adapter journal fields must be present together on every raw row")

    previous: str | None = None
    seen_invocations: set[str] = set()
    for expected_sequence, row in enumerate(rows, start=1):
        label = f"adapter journal row {expected_sequence}"
        sequence = _integer(row["adapter_sequence"], f"{label}.adapter_sequence", minimum=1)
        if sequence != expected_sequence:
            raise ContractError(f"{label} has out-of-order adapter_sequence {sequence}")
        predecessor = row["adapter_previous_receipt_sha256"]
        if predecessor is not None:
            validate_sha256(predecessor, f"{label}.adapter_previous_receipt_sha256")
        if predecessor != previous:
            raise ContractError(f"{label} previous receipt does not match append order")
        receipt = row["adapter_receipt_sha256"]
        validate_sha256(receipt, f"{label}.adapter_receipt_sha256")
        expected_receipt = adapter_receipt_sha256(
            row,
            sequence=sequence,
            previous_receipt_sha256=previous,
        )
        if receipt != expected_receipt:
            raise ContractError(f"{label} receipt does not match raw payload")
        invocation_id = str(row.get("invocation_id"))
        if invocation_id in seen_invocations:
            raise ContractError(f"{label} repeats invocation {invocation_id!r}")
        seen_invocations.add(invocation_id)
        if expected_invocation_ids is not None:
            if expected_sequence > len(expected_invocation_ids):
                raise ContractError("adapter journal has more rows than the immutable plan")
            if invocation_id != expected_invocation_ids[expected_sequence - 1]:
                raise ContractError(f"{label} invocation order drifts from immutable plan")
        previous = receipt


class AppendOnlyRawJournal:
    """A single-plan JSONL journal with create-or-append, never-overwrite I/O."""

    def __init__(
        self,
        *,
        path: Path,
        plan_sha256: str,
        expected_invocation_ids: Sequence[str],
    ) -> None:
        validate_sha256(plan_sha256, "adapter journal plan_sha256")
        if not expected_invocation_ids:
            raise ContractError("adapter journal requires at least one planned invocation")
        for invocation_id in expected_invocation_ids:
            validate_identifier(invocation_id, "adapter journal expected invocation")
        if len(expected_invocation_ids) != len(set(expected_invocation_ids)):
            raise ContractError("adapter journal expected invocations must be unique")
        if path.is_symlink():
            raise ContractError(f"adapter journal must not be a symbolic link: {path}")
        if path.exists():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"adapter journal must be a regular file: {path}")
        elif not path.parent.is_dir():
            raise ContractError(f"adapter journal parent does not exist: {path.parent}")
        self.path = path
        self.plan_sha256 = plan_sha256
        self.expected_invocation_ids = tuple(expected_invocation_ids)

    def execution_lease(self) -> "ExecutionJournalLease":
        """Return a non-blocking campaign-wide lease for this journal.

        The lease is intentionally distinct from the short append lock: it
        begins before an invocation is selected and remains held through its
        process lifecycle and terminal append.  A second adapter therefore
        fails before ``start`` rather than measuring the same planned arm.
        """

        return ExecutionJournalLease(self.path)

    def _read_rows(self) -> list[Mapping[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size == 0:
            return []
        rows = [row for _path, _line, row in load_jsonl([self.path])]
        for index, row in enumerate(rows, start=1):
            if row.get("campaign_plan_sha256") != self.plan_sha256:
                raise ContractError(f"adapter journal row {index} belongs to a different execution plan")
        validate_append_only_chain(rows, expected_invocation_ids=self.expected_invocation_ids)
        return rows

    def append(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Append exactly the next plan invocation, fsyncing the complete row."""

        row = dict(_mapping(value, "adapter raw row"))
        if any(field in row for field in JOURNAL_FIELDS):
            raise ContractError("adapter raw row must not predeclare journal receipt fields")
        if row.get("campaign_plan_sha256") != self.plan_sha256:
            raise ContractError("adapter raw row plan hash differs from journal")
        if self.path.is_symlink():
            raise ContractError(f"adapter journal must not be a symbolic link: {self.path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        # The pre-open lstat above is useful for diagnostics, but cannot by
        # itself protect the check/open gap.  Linux and macOS both expose
        # O_NOFOLLOW; keep the fallback for stdlib portability and still
        # verify the opened descriptor below.
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o644)
        except OSError as error:
            raise ContractError(f"cannot safely open adapter raw journal {self.path}: {error}") from error
        try:
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError(f"adapter journal must be a regular file: {self.path}")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    rows = self._read_rows()
                    sequence = len(rows) + 1
                    if sequence > len(self.expected_invocation_ids):
                        raise ContractError("adapter journal already contains every planned invocation")
                    invocation_id = row.get("invocation_id")
                    if invocation_id != self.expected_invocation_ids[sequence - 1]:
                        raise ContractError("adapter raw row is out of immutable plan order")
                    previous = rows[-1]["adapter_receipt_sha256"] if rows else None
                    row["adapter_sequence"] = sequence
                    row["adapter_previous_receipt_sha256"] = previous
                    row["adapter_receipt_sha256"] = adapter_receipt_sha256(
                        row,
                        sequence=sequence,
                        previous_receipt_sha256=previous,
                    )
                    encoded = canonical_json_bytes(row)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            raise ContractError(f"cannot append adapter raw journal {self.path}: {error}") from error
        return row


class ExecutionJournalLease(AbstractContextManager["ExecutionJournalLease"]):
    """Exclusive, non-blocking execution lease next to one raw journal."""

    def __init__(self, journal_path: Path) -> None:
        self.journal_path = journal_path
        self.path = journal_path.with_name(journal_path.name + ".lock")
        self._process_lock: threading.Lock | None = None
        self._handle: Any | None = None

    def __enter__(self) -> "ExecutionJournalLease":
        with _PROCESS_LEASE_GUARD:
            process_lock = _PROCESS_LEASES.setdefault(self.path, threading.Lock())
        if not process_lock.acquire(blocking=False):
            raise ContractError(f"another execution adapter already holds journal lease: {self.journal_path}")
        self._process_lock = process_lock
        try:
            if self.path.is_symlink():
                raise ContractError(f"adapter execution lease must not be a symbolic link: {self.path}")
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, flags, 0o644)
            except OSError as error:
                raise ContractError(
                    f"cannot safely open adapter execution lease {self.path}: {error}"
                ) from error
            handle = os.fdopen(descriptor, "a+b", closefd=True)
            try:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError(f"adapter execution lease must be a regular file: {self.path}")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ContractError(
                        f"another execution adapter already holds journal lease: {self.journal_path}"
                    ) from error
            except Exception:
                handle.close()
                raise
            self._handle = handle
            return self
        except Exception:
            self._release_process_lock()
            raise

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            if self._handle is not None:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self._handle.close()
                    self._handle = None
        finally:
            self._release_process_lock()

    def _release_process_lock(self) -> None:
        if self._process_lock is not None:
            self._process_lock.release()
            self._process_lock = None
