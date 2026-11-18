"""Compare status and smaps accounting while touching 2 MiB of private pages on Linux."""

import json
import mmap
import os
from pathlib import Path


SIZE = 2 * 1024**2
STRIDE = 64 * 1024
STATUS = Path("/proc/self/status")
SMAPS = Path("/proc/self/smaps_rollup")


def value(path, field):
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key == field:
            return int(rest.split()[0])
    raise ValueError(f"missing {field}")


def sample(touched):
    return {"touched_kib": touched // 1024, "status_rss_anon_kib": value(STATUS, "RssAnon"),
            "rollup_anonymous_kib": value(SMAPS, "Anonymous")}


def main():
    page = os.sysconf("SC_PAGE_SIZE")
    if SIZE % STRIDE or STRIDE % page:
        raise ValueError("mapping and stride must be whole pages")
    with mmap.mmap(-1, SIZE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as buffer:
        buffer.madvise(mmap.MADV_NOHUGEPAGE)
        sample(0)
        rows = [sample(0)]
        for end in range(STRIDE, SIZE + STRIDE, STRIDE):
            for offset in range(end - STRIDE, end, page):
                buffer[offset] = 1
            rows.append(sample(end))
    print(json.dumps({"kernel_release": os.uname().release, "page_kib": page // 1024,
                      "anonymous_fault_stride_kib": STRIDE // 1024, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
