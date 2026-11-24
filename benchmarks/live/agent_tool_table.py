"""Encode a rectangular JSON record list as column names and complete rows."""


def encode_record_table(records):
    """Return the original value when its records do not share one ordered schema."""
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        return records
    columns = list(records[0])
    if not columns or not all(isinstance(key, str) for key in columns):
        return records
    rows = []
    for record in records:
        if not isinstance(record, dict) or list(record) != columns:
            return records
        rows.append(list(record.values()))
    return {"columns": columns, "rows": rows}
