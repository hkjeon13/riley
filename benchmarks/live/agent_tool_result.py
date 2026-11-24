"""Build an exact JSON answer for explicitly declared tool result contracts."""

from collections.abc import Mapping, Sequence

from agent_tool_aggregate import project_unique_integer_max
from agent_tool_projection import project_exact_records


def exact_tool_result(
    records: Sequence[Mapping],
    *,
    operation: str,
    key: str,
    fields: Mapping[str, type],
    requested_values: Sequence[str] | None = None,
) -> dict | None:
    """Return None when the input cannot satisfy the caller's exact result contract.

    unique_matches returns a keyed object in the caller's requested order.
    unique_integer_max returns one object. Only explicitly declared fields from
    the winning rows can appear in the result. The caller controls authorization.
    """
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not isinstance(key, str)
        or not key
        or not isinstance(fields, Mapping)
        or not fields
    ):
        return None
    if not all(
        isinstance(name, str) and name and typ in (str, int, bool)
        for name, typ in fields.items()
    ):
        return None
    if operation == "unique_matches" and requested_values:
        selected = project_exact_records(
            records, key=key, requested_ids=requested_values
        )
    elif operation == "unique_integer_max" and requested_values is None:
        selected = project_unique_integer_max(records, key=key)
    else:
        return None
    if selected is records or any(
        type(row.get(name)) is not typ
        for row in selected
        for name, typ in fields.items()
    ):
        return None
    if operation == "unique_integer_max":
        return {name: selected[0][name] for name in fields}
    by_value = {row[key]: {name: row[name] for name in fields} for row in selected}
    return {value: by_value[value] for value in requested_values}
