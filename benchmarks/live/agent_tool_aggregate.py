"""Evaluate exact typed aggregate queries on authorized tool records."""

from collections.abc import Mapping, Sequence


def project_unique_integer_max(
    records: Sequence[Mapping], *, key: str
) -> Sequence[Mapping]:
    """Keep the complete winning record for a declared unique integer maximum."""
    if not key:
        return records
    winner = None
    largest = None
    tied = False
    for record in records:
        if not isinstance(record, Mapping):
            return records
        value = record.get(key)
        if type(value) is not int:
            return records
        if largest is None or value > largest:
            largest, winner, tied = value, record, False
        elif value == largest:
            tied = True
    if winner is None or tied:
        return records
    return [winner]
