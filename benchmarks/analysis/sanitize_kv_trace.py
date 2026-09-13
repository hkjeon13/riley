"""Export only numeric CUDA activity and public API names, never Nsight env data."""
import argparse
import hashlib
import re
import sqlite3
from pathlib import Path


def sanitize(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as original:
        copies = list(original.execute("select start,end,deviceId,contextId,streamId,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
        runtime = list(original.execute("select start,end,correlationId,nameId,returnValue from CUPTI_ACTIVITY_KIND_RUNTIME"))
        for row in copies + runtime:
            assert all(value is None or isinstance(value, int) for value in row), "non-numeric activity field"
        names = []
        for identifier in sorted({row[3] for row in runtime}):
            name, = original.execute("select value from StringIds where id=?", (identifier,)).fetchone()
            # Nsight's runtime table also contains public driver entry points
            # such as cuInit and cuGetProcAddress_v2.
            assert re.fullmatch(r"cu[A-Za-z0-9_]{1,160}", name), "non-public CUDA API name"
            names.append((identifier, name))
        expected_labels = {1: "Host-to-Device", 2: "Device-to-Host", 8: "Device-to-Device"}
        labels = []
        for kind in sorted({row[7] for row in copies}):
            label, = original.execute("select label from ENUM_CUDA_MEMCPY_OPER where id=?", (kind,)).fetchone()
            assert expected_labels.get(kind) == label, "unexpected copy direction"
            labels.append((kind, label))
    with sqlite3.connect(destination) as curated:
        curated.executescript("""
            create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,deviceId integer,contextId integer,streamId integer,correlationId integer,bytes integer,copyKind integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,correlationId integer,nameId integer,returnValue integer);
            create table StringIds(id integer primary key,value text);
            create table ENUM_CUDA_MEMCPY_OPER(id integer primary key,label text);
            create table RILEY_TRACE_REDACTION(version integer,source_sha256 text,policy text);
        """)
        curated.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?,?,?,?)", copies)
        curated.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)", runtime)
        curated.executemany("insert into StringIds values(?,?)", names)
        curated.executemany("insert into ENUM_CUDA_MEMCPY_OPER values(?,?)", labels)
        curated.execute("insert into RILEY_TRACE_REDACTION values(1,?,?)", (
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "Numeric CUDA activity and public API names only; no environment, process metadata or original report"))
    print(f"curated trace: {len(copies)} copy activities, {len(runtime)} runtime activities; environment excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize(args.source, args.destination)
