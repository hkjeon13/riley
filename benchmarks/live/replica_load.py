"""Sample Riley admission counters away from the workflow submission path."""

import concurrent.futures
import http.client
import json
import threading
import time
from functools import partial
from urllib.parse import urlsplit


def read_load(endpoint, on_result=None):
    address, bearer = endpoint
    url = urlsplit(address if "://" in address else "http://" + address)
    transport = {
        "http": http.client.HTTPConnection,
        "https": http.client.HTTPSConnection,
    }
    connection = transport[url.scheme](url.hostname, url.port, timeout=0.25)
    path = url.path.rstrip("/").removesuffix("/v1") + "/readyz"
    started = time.monotonic()
    status = None
    outcome = "invalid"
    try:
        headers = {"Authorization": "Bearer " + bearer} if bearer else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        status = response.status
        if status not in (200, 503):
            outcome = "http_error"
            return None
        raw = response.read(65537)
        if len(raw) > 65536:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        if any(type(value.get(k)) is not bool for k in ("ready", "accepting")):
            return None
        if any(
            type(value.get(k)) is not int or value[k] < 0
            for k in ("active_requests", "waiting_requests")
        ):
            return None
        outcome = "sample"
        return started, {
            "available": response.status == 200
            and value["ready"]
            and value["accepting"],
            "active": value["active_requests"],
            "waiting": value["waiting_requests"],
        }
    except TimeoutError:
        outcome = "timeout"
        return None
    except (OSError, http.client.HTTPException):
        outcome = "transport_error"
        return None
    except ValueError:
        return None
    finally:
        connection.close()
        if on_result is not None:
            on_result(started, time.monotonic(), status, outcome)


class ReplicaLoad:
    def __init__(self, endpoints, on_poll=None):
        self.endpoints = endpoints
        self.on_poll = on_poll
        self.samples = [None] * len(endpoints)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, name="replica-readiness")

    def refresh(self, index):
        observe = partial(self.on_poll, index) if self.on_poll else None
        sample = read_load(self.endpoints[index], observe)
        if sample is not None:
            with self.lock:
                self.samples[index] = sample

    def poll(self):
        with concurrent.futures.ThreadPoolExecutor(min(8, len(self.endpoints))) as pool:
            while not self.stop.is_set():
                list(pool.map(self.refresh, range(len(self.endpoints))))
                self.stop.wait(0.1)

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            return [
                dict(s[1], age_ms=(now - s[0]) * 1000)
                if s and now - s[0] <= 0.5
                else None
                for s in self.samples
            ]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
