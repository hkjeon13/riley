"""Summarize checked runs and the effect of moving the later cohort."""

REFERENCE_SCHEDULE_SHA256 = "a368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0"


def compact_run(run):
    mismatches = sum(map(len, run["stock_output_mismatches"].values()))
    requested = sum(run["cells"][cell]["requests"] for cell in REQUEST_COUNTS)
    return {**{key: run[key] for key in (
        "short_ttft_ms", "short_queue_ms", "short_gpu_steps_over_steady",
        "fidelity", "receipt_sha256", "workload_sha256", "shutdown_sha256",
    )}, "short": run["cells"]["short"], "rag": run["cells"]["rag"],
        "stock_output_matched": requested - mismatches, "requests": requested}


def order_of_variants(stage):
    events = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
              if line.startswith(("original START ", "late12 START "))]
    order = {gpu: [entry[0] for entry in events if entry[2].endswith(f"-gpu{gpu}")] for gpu in ORDERS}
    require(order == {"6": ["original"] * 4 + ["late12"] * 4,
                      "7": ["late12"] * 4 + ["original"] * 4}, "schedule order changed")
    return order


def check_write_shift(before, after, mode, gpu, planned_ms):
    differences = {}
    for limit in (0, 8):
        name = f"private-short-d{limit}-{mode}-gpu{gpu}"
        arms = [read(root / name / "workload.json")["short"]["rows"] for root in (before, after)]
        origins = [min(row["sent_at"] for row in rows) for rows in arms]
        shifts = [1000 * ((b["sent_at"] - origins[1]) - (a["sent_at"] - origins[0]))
                  for a, b in zip(*arms, strict=True)]
        by_cohort = {}
        for cohort, values, target in (("first_32", shifts[:32], 0), ("later_64", shifts[32:], planned_ms)):
            error = p([abs(value - target) for value in values], .95)
            require(error <= 5, f"observed shift differs from the schedule for {name}: {cohort}")
            by_cohort[cohort] = {"target_ms": target, "p50_ms": p(values, .5), "error_p95_ms": error}
        differences[str(limit)] = by_cohort
    return differences


def phase_effect(stage, verified):
    before, after = (stage.with_name(f"{stage.name}-{v}") for v in DELAYS_NS)
    differences = {}
    for gpu in ORDERS:
        differences[gpu] = {}
        for mode in ("control", "admission"):
            before_pairs = verified["original"]["paired_ordinary_phases"][gpu][mode]["later_64"]
            after_pairs = verified["late12"]["paired_ordinary_phases"][gpu][mode]["later_64"]
            changes = {field: after_pairs[field] - before_pairs[field] for field in before_pairs}
            differences[gpu][mode] = {
                "late_minus_original_candidate_effect_ms": changes,
                "observed_relative_write_shift_ms": check_write_shift(
                    before, after, mode, gpu, DELAYS_NS["late12"] / 1_000_000
                ),
            }
    return differences


def wide_step_timing(folder, report):
    timings = {}
    for gpu in ORDERS:
        name = f"private-short-d8-admission-gpu{gpu}"
        arm = folder / name
        rows = read(arm / "workload.json")["short"]["rows"]
        request_ids = [int(row["response_id"].removeprefix("cmpl-"), 16) for row in rows]
        windows = {request: {} for request in request_ids}
        spans = []
        for line in (arm / "server.log").read_text().splitlines():
            if line.startswith("W3-REQUEST-PHASE "):
                _, request, phase, stamp = line.split()
                request = int(request)
                if request in windows:
                    windows[request][phase] = int(stamp)
            elif line.startswith("W3-ENGINE-SPAN "):
                spans.append(tuple(map(int, line.split()[1:])))
        ordered = [windows[request] for request in request_ids]
        first = min(window["submit"] for window in ordered)
        first_later_send = min(window["send"] for window in ordered[32:])
        # The fixed C32 replay is the final request group in each container.
        wide = [(a, b, n, d) for a, b, n, d in spans if a >= first and n > 256]
        count = report["runs"][name]["short_gpu_steps_over_steady"]
        require(len(wide) == count, f"trace and committed wide-step counters differ: {name}")
        steps = []
        for start, end, total, decode in wide:
            overlaps = [max(start, item["send"]) < min(end, item["worker"]) for item in ordered]
            steps.append({
                "total_rows": total,
                "decode_rows": decode,
                "start_after_first_submit_ms": (start - first) / 1_000_000,
                "end_after_first_submit_ms": (end - first) / 1_000_000,
                "early_requests_waiting": sum(overlaps[:32]),
                "later_requests_waiting": sum(overlaps[32:]),
            })
        timings[gpu] = {
            "first_later_send_after_first_submit_ms": (first_later_send - first) / 1_000_000,
            "last_wide_end_to_first_later_send_ms": (first_later_send - wide[-1][1]) / 1_000_000 if wide else None,
            "steps": steps,
        }
    return timings


def arrival_phase_report(stage):
    reference = stage.parent / "w3-c32-short-burst-admission-gpu-20261123.TA5Cmib0/schedules.json"
    require(digest(reference) == REFERENCE_SCHEDULE_SHA256, "the original schedule changed")
    originals = read(reference)
    variants = {}
    for variant, delay in DELAYS_NS.items():
        folder = stage.with_name(f"{stage.name}-{variant}")
        report = verify(folder, shifted_schedules(originals, delay))
        report["queue_boundaries"] = queue_boundary_report(
            folder, report["result_sha256"]
        )["paired_trace_queue_boundaries"]
        report["wide_step_timing"] = wide_step_timing(folder, report)
        report["runs"] = {name: compact_run(run) for name, run in report["runs"].items()}
        variants[variant] = report
    baseline, delayed = variants.values()
    for key in ("build", "image_id", "source_revision", "reservations", "stock_workload_sha256"):
        require(baseline[key] == delayed[key], f"{key} differs across schedules")
    holder_after = {}
    for gpu in ORDERS:
        holder_after.update(checked_reservation(stage / f"gpu{gpu}-after.json", gpu))
    require(holder_after == baseline["reservations"], "the original reservation was not restored")
    return {
        "status": "verified",
        "remote_evidence": str(stage),
        "reference_schedule_sha256": REFERENCE_SCHEDULE_SHA256,
        "dispatch_sha256": digest(stage / "dispatch.sha256"),
        "variant_results_sha256": digest(stage / "variant-results.sha256"),
        "schedule_order": order_of_variants(stage),
        "holder_after": holder_after,
        "variants": variants,
        "effect_of_arrival_shift": phase_effect(stage, variants),
    }


if __name__ == "__main__":
    print(json.dumps(arrival_phase_report(Path(sys.argv[1])), indent=2, allow_nan=False))
