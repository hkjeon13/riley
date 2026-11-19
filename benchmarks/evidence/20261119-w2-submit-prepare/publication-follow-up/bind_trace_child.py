"""Bind shutdown sampling to the known profiler child without accepting other PIDs."""

from pathlib import Path


def rewrite(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path.name} does not match the reviewed insertion point")
    updated = text.replace(old, new)
    compile(updated, str(path), "exec")
    path.write_text(updated)


if __name__ == "__main__":
    rewrite(
        Path("tools/profile_workload.py"),
        '(out / "server.pid").write_text(str(server.pid) + "\\n")',
        'from server_owner import record_child\n'
        '            record_child(out, server)\n'
        '            (out / "server.pid").write_text(str(server.pid) + "\\n")',
    )
    rewrite(
        Path("tools/run_profile_retry.py"),
        "if not owned and not observed_exit(pid, server_binding):",
        'from server_owner import is_owned\n'
        '            if (not owned and not observed_exit(pid, server_binding)\n'
        '                    and not is_owned(out / "server-owner.json", pid, start_ticks)):' ,
    )
