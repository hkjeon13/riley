"""Recognize an NVML sample from the bound process during Linux exit."""
from pathlib import Path
import sys

source=Path(sys.argv[1])
old=source.read_text()
needle='    return len(fields) > 19 and fields[0] in ("Z", "X", "x") and fields[19] == binding[1]\n'
replacement='''    if len(fields) <= 19 or fields[19] != binding[1]:
        return False
    if fields[0] in ("Z", "X", "x"):
        return True
    process = proc_root / str(pid)
    try:
        # Linux can clear the command line before publishing a terminal
        # process state. The bound PID and start time must still match.
        if (process / "cmdline").read_bytes():
            return False
        after = (process / "stat").read_text().rpartition(")")[2].split()
        return len(after) > 19 and after[19] == binding[1]
    except FileNotFoundError:
        return True
'''
assert old.count(needle)==1
new=old.replace(needle,replacement,1)
compile(new,str(source),"exec")
source.write_text(new)
