"""Select whole tool records when the caller has declared an exact-key lookup."""

from collections.abc import Mapping, Sequence


def project_exact_records(
    records: Sequence[Mapping], *, key: str, requested_ids: Sequence[str] | None
) -> Sequence[Mapping]:
    """Return the full result for undeclared, missing or ambiguous keys.

    The caller supplies authorized tool output and the IDs from its typed lookup.
    Preserve all fields on selected records and their source order.
    """
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not isinstance(requested_ids, Sequence)
        or isinstance(requested_ids, (str, bytes))
        or not requested_ids
    ):
        return records
    if not all(isinstance(item, str) and item for item in requested_ids):
        return records
    wanted = set(requested_ids)
    if len(wanted) != len(requested_ids):
        return records
    selected = []
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            return records
        value = record.get(key)
        if not isinstance(value, str):
            return records
        if value in wanted:
            if value in seen:
                return records
            selected.append(record)
            seen.add(value)
    return selected if seen == wanted else records
