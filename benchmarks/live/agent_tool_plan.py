"""Resolve a caller-owned named function step with one possible argument object."""


def resolve_named_step(tools, choice):
    """Return (name, arguments) for explicit application dispatch, otherwise None.

    The caller controls authorization and executes the tool. This helper handles
    closed objects with zero properties or required single-valued string fields.
    """
    if not isinstance(choice, dict) or set(choice) != {"type", "function"}:
        return None
    selected = choice["function"]
    if (
        choice["type"] != "function"
        or not isinstance(selected, dict)
        or set(selected) != {"name"}
    ):
        return None
    name = selected["name"]
    if not isinstance(name, str) or not name or not isinstance(tools, list):
        return None
    functions = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return None
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            return None
        if fn["name"] == name:
            functions.append(fn)
    if len(functions) != 1:
        return None
    schema = functions[0].get("parameters")
    if not isinstance(schema, dict) or set(schema) != {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }:
        return None
    fields, required = schema["properties"], schema["required"]
    if (
        schema["type"] != "object"
        or schema["additionalProperties"] is not False
        or not isinstance(fields, dict)
    ):
        return None
    if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
        return None
    if len(required) != len(fields) or set(required) != fields.keys():
        return None
    args = {}
    for key, field in fields.items():
        if (
            not isinstance(field, dict)
            or set(field) != {"type", "enum"}
            or field["type"] != "string"
        ):
            return None
        values = field["enum"]
        if (
            not isinstance(values, list)
            or len(values) != 1
            or not isinstance(values[0], str)
        ):
            return None
        args[key] = values[0]
    return name, args
