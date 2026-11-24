"""Compare generated token IDs for the existing long-input holdout at C1 and C8."""

import asyncio
import os
from pathlib import Path
import random
import sys
import time

from holdout_load import one
from load import prompt
from profile_workload import save


async def capture(output, address):
    host, port = address.split(":")
    rng = random.Random(703)
    bodies = [{"model": "m", "temperature": 0, "max_tokens": 64, "stream": True,
               "stream_options": {"include_usage": True}, "return_token_ids": True,
               "messages": [{"role": "user", "content": prompt(rng, words)}]}
              for words in (30, 41, 1400, 1600, 2400, 2600, 2900, 3200)]

    async def issue(body):
        row = {}
        await asyncio.wait_for(one(host, int(port), body, time.perf_counter(), row,
                                   capture_hashes=True, capture_token_ids=True), timeout=90)
        if row.get("error") is not False:
            raise ValueError("holdout token capture failed")
        return {key: row[key] for key in ("input_sha256", "output_sha256", "prompt_ids_sha256", "token_ids_sha256",
                                         "prompt_tokens", "completion_tokens", "finish")}

    c1 = [await issue(body) for body in bodies]
    c8 = await asyncio.gather(*(issue(body) for body in bodies))
    save(output, {"c1": c1, "c8": c8})


if __name__ == "__main__":
    os.sched_setaffinity(0, {int(os.environ["W3_CLIENT_CPU"])})
    asyncio.run(capture(Path(sys.argv[1]), sys.argv[2]))
