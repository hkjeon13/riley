"""Join admission timestamps to completed worker and CUDA execution intervals."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def ordered(spans):
    if any(not 0 <= span.start < span.end for span in spans) or any(
        left.end > right.start for left, right in zip(spans, spans[1:])
    ):
        raise ValueError("worker or execution spans are invalid or overlap")


def parse(log):
    admission, workers, execution = {}, [], []
    snapshots = []
    for line in log.splitlines():
        label, _, payload = line.partition(" ")
        if label == "W3-ADMISSION-TRACE":
            snapshots.append(tuple(map(int, payload.split())))
        elif label == "W3-ADMISSION":
            request, start, queue, receive, ack = map(int, payload.split())
            if request in admission or not 0 <= start <= queue <= receive <= ack:
                raise ValueError(f"duplicate or inconsistent admission for request {request}")
            admission[request] = (start, queue, receive, ack)
        elif label == "W3-WORKER-STEP":
            workers.append(Span(*map(int, payload.split())))
        elif label == "W3-EXECUTION":
            start, end, rows, decode = map(int, payload.split())
            if not 0 <= decode <= rows or rows == 0:
                raise ValueError("invalid CUDA iteration shape")
            execution.append((Span(start, end), rows, decode))
    if snapshots != [(len(admission) + len(workers) + len(execution), 0)] or not admission or not workers:
        raise ValueError("worker trace is incomplete or dropped data")
    ordered(workers)
    ordered([span for span, _, _ in execution])
    worker_starts = [span.start for span in workers]
    cuda_by_worker = {}
    for span, rows, decode in execution:
        idx = bisect_right(worker_starts, span.start) - 1
        if idx < 0 or span.end > workers[idx].end or idx in cuda_by_worker:
            raise ValueError("CUDA span is not uniquely enclosed by its worker step")
        cuda_by_worker[idx] = (span, rows, decode)
    return admission, workers, worker_starts, cuda_by_worker


def overlap(span, start, end):
    return max(0, min(span.end, end) - max(span.start, start))


def c32_execution_spans(log, capture, holdout_count):
    admissions, _, _, executions = parse(log)
    warmup = [int(request.removeprefix("cmpl-"), 16) for request in capture["warmup_response_ids"]]
    measured = [int(row["response_id"].removeprefix("cmpl-"), 16) for row in capture["measured"]]
    requests = sorted(warmup + measured)
    if requests != list(range(requests[0], requests[-1] + 1)) or max(warmup) >= min(measured):
        raise ValueError("C32 request IDs are not a contiguous phase after warmup")
    holdout = sorted(request for request in admissions if request > requests[-1])
    if holdout != list(range(requests[-1] + 1, requests[-1] + 1 + holdout_count)):
        raise ValueError("C32 capture lacks the contiguous post-replay holdout")
    start = min(admissions[request][0] for request in warmup)
    measurement_start = min(admissions[request][0] for request in measured)
    end = min(admissions[request][0] for request in holdout)
    if not start < measurement_start < end:
        raise ValueError("C32 phase timestamps are inconsistent")
    selected = [(span, rows, decode) for span, rows, decode in executions.values()
                if span.start >= start and span.end <= end]
    if not selected or selected[0][0].start >= measurement_start:
        raise ValueError("C32 CUDA execution trace lacks warmup")
    return selected, measurement_start


def reconcile(log, observations, *, expect_cuda=True):
    admissions, workers, starts, cuda = parse(log)
    if expect_cuda and not cuda:
        raise ValueError("no CUDA execution spans in the measured process")
    result = []
    for row in observations:
        request = int(row["response_id"].removeprefix("cmpl-"), 16)
        submit, sent, received, ack = admissions[request]
        queue = received - sent
        eligible = range(max(0, bisect_right(starts, sent) - 1), bisect_left(starts, received))
        worker_ns = sum(overlap(workers[idx], sent, received) for idx in eligible)
        execution_ns = sum(overlap(cuda[idx][0], sent, received) for idx in eligible if idx in cuda)
        incumbent = next((idx for idx in eligible if workers[idx].start <= sent < workers[idx].end), None)
        incumbent_ns = overlap(workers[incumbent], sent, received) if incumbent is not None else 0
        if not 0 <= execution_ns <= worker_ns <= queue or not 0 <= incumbent_ns <= worker_ns:
            raise ValueError(f"nested queue intervals are inconsistent for request {request}")
        expected = row["backend_admission_ms"]
        unmeasured = expected - (ack - submit) / 1e6
        if unmeasured < -0.001:
            raise ValueError("engine timing does not fit inside service admission")
        execution = cuda.get(incumbent)
        shape = None if execution is None else execution[1:]
        result.append({
            "request_index": row["request_index"],
            "response_id": row["response_id"],
            "admission_ms": expected,
            "before_enqueue_ms": (sent - submit) / 1e6,
            "command_queue_ms": queue / 1e6,
            "worker_step_remaining_ms": incumbent_ns / 1e6,
            "later_worker_steps_ms": (worker_ns - incumbent_ns) / 1e6,
            "cuda_span_overlap_ms": execution_ns / 1e6,
            "queue_outside_steps_ms": (queue - worker_ns) / 1e6,
            "dispatch_to_ack_ms": (ack - received) / 1e6,
            "admission_remainder_ms": unmeasured,
            "incumbent_shape": shape,
            "incumbent_cuda_start_offset_ms": None if execution is None else (sent - execution[0].start) / 1e6,
            "incumbent_cuda_duration_ms": None if execution is None else (execution[0].end - execution[0].start) / 1e6,
        })
    return result
