"""Check that linked reports and published claims match their evidence."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("improvement_evidence", ROOT / "ci/check_improvement_evidence.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DocumentedProfileTests(unittest.TestCase):
    def test_varlen_defaults_match_the_profile_resolver(self):
        profile = "native-bf16-varlen-v1"
        configuration = ROOT / "docs/configuration.md"
        rows = checker.markdown_table(configuration, ("Option", f"`{profile}` default"))
        documented = {key.strip("`"): value.strip("`") for key, (value,) in rows.items()}
        self.assertEqual(documented.keys(), {
            "--native-attention-mode", "--max-sequence-tokens", "--batch-token-budget",
            "--prefill-chunk-tokens", "--event-publication",
        })

        source = ROOT / "crates/riley-server/src/inference_profile.rs"
        program = f"""
#[path = {json.dumps(str(source), ensure_ascii=False)}]
mod inference_profile;

fn main() {{
    let expanded = inference_profile::expand(std::env::args_os().skip(1)).unwrap();
    for argument in expanded {{
        println!("{{}}", argument.to_str().unwrap());
    }}
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "resolve-profile"
            subprocess.run(["rustc", "--edition=2024", "-o", str(binary), "-"], input=program,
                           text=True, check=True, timeout=30, cwd=ROOT)
            command = [str(binary), "serve", "--inference-profile", profile]
            verb, *options = subprocess.check_output(command, text=True, timeout=5).splitlines()
        self.assertEqual(verb, "serve")
        resolved = dict(zip(options[::2], options[1::2], strict=True))
        self.assertEqual(documented, {flag: resolved.get(flag) for flag in documented})


class ReadmeEvidenceTests(unittest.TestCase):
    def test_readme_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            for filename in ("Cargo.toml", "docs/model-coverage-a100-perf-table.md"):
                (root / filename).write_bytes((ROOT / filename).read_bytes())
            path = root / "README.md"
            original = (ROOT / path.name).read_text(encoding="utf-8")
            path.write_text(original, encoding="utf-8")
            self.assertEqual(checker.check_readme(root), 24)
            for pattern, replacement, message in (
                (r"lower in \d+ of", "lower in 0 of", "summary differs for Time per output token"),
                (r"(higher in all \d+ cells — )\d+\.\d+×", r"\g<1>0.00×", "summary differs for Throughput"),
                (r"> Status: \S+", "> Status: invalid", "Status version must match Cargo"),
            ):
                with self.subTest(message=message):
                    changed, count = re.subn(pattern, replacement, original, count=1)
                    self.assertEqual(count, 1)
                    path.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        checker.check_readme(root)


class DistributionServingTests(unittest.TestCase):
    def test_rehashed_report_and_published_coverage_drift_are_rejected(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        candidate = manifest["current_distribution_candidate"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            records = candidate["serving_reports"]
            for filename in ("docs/model-support.md", *(entry["path"] for entry in records.values())):
                destination = root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / filename).read_bytes())
            self.assertEqual(checker.check_distribution_serving(root, candidate), 6)

            support = root / "docs/model-support.md"
            original = support.read_text(encoding="utf-8")
            support.write_text(original.replace("peak 31 active", "peak 32 active"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage differs for Qwen2.5-32B-Instruct"):
                checker.check_distribution_serving(root, candidate)
            support.write_text(original, encoding="utf-8")

            record = records["tp2"]
            path = root / record["path"]
            report = json.loads(path.read_bytes())
            report["riley_binary_sha256"] = "0" * 64
            raw = json.dumps(report).encode("utf-8")
            path.write_bytes(raw)
            record["sha256"] = sha256(raw).hexdigest()
            self.assertEqual(checker.check_evidence(root, records), 5)
            with self.assertRaisesRegex(ValueError, "tp2 belongs to a different image or binary"):
                checker.check_distribution_serving(root, candidate)


class PackageFollowUpTests(unittest.TestCase):
    def test_prefix_metrics_uses_transport_pairs_for_c1_and_exact_matches_for_concurrency(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        candidate = manifest["distribution_with_prefix_metrics"]
        report_path = ROOT / candidate["evidence"]["a100_serving"]["path"]
        serving = json.loads(report_path.read_bytes())
        checker.check_prefix_metrics_serving(serving)

        suites = serving["accuracy"]["suites"]["structured_holdout"]
        suites["modes"]["c8"]["exact_matches"] -= 1
        with self.assertRaisesRegex(ValueError, "lacks exact long and structured output evidence"):
            checker.check_prefix_metrics_serving(serving)

    def test_gpu_receipt_is_bound_to_the_reconstructed_package(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        candidate = manifest["distribution_after_terminal_fix"]
        checker.check_package_follow_up(ROOT, candidate)

        entry = candidate["evidence"]["a100_serving"]
        changed = json.loads((ROOT / entry["path"]).read_bytes())
        changed["image_id"] = "sha256:" + "0" * 64
        raw = json.dumps(changed).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a100-serving.json"
            path.write_bytes(raw)
            entry.update(path=str(path), sha256=sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError, "GPU serving is not bound to the A100 package"):
                checker.check_package_follow_up(ROOT, candidate)

    def test_soak_keeps_a_measured_rss_failure_in_the_release_gate(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        entry = manifest["distribution_after_terminal_fix"]["evidence"]["a100_serving"]
        serving = json.loads((ROOT / entry["path"]).read_bytes())
        cache_off = {"failed_checks": [], "status": "passed",
                     "rss": {"baseline": {"rss_kib": 50}, "drain": {"rss_kib": 50}, "rss_growth_kib": 0},
                     "p95_change_percent": {"ttft_ms": 1.0, "e2e_ms": -0.5}}
        cache_on = deepcopy(cache_off)
        cache_on.update(status="failed", failed_checks=["rss_growth_second_half_at_most_zero"])
        cache_on["rss"].update(drain={"rss_kib": 51}, rss_growth_kib=1)
        soaks = {key: serving[key] for key in ("source_revision", "image_id", "checkpoint_sha256")}
        soaks.update(binary_sha256=serving["riley_binary_sha256"], duration_seconds_per_mode=1800,
                     requests_per_second_per_mode=8, release_gate="failed",
                     modes={"cache-off": cache_off, "cache-on": cache_on})
        checker.check_package_soaks(soaks, serving)

        soaks["release_gate"] = "passed"
        with self.assertRaisesRegex(ValueError, "gate differs from the two cache outcomes"):
            checker.check_package_soaks(soaks, serving)
        cache_on.update(status="passed", failed_checks=[])
        with self.assertRaisesRegex(ValueError, "cache-on soak verdict differs"):
            checker.check_package_soaks(soaks, serving)

    def test_allocator_soaks_match_the_attested_runtime_settings(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        candidate = manifest["distribution_after_terminal_fix"]
        entry = candidate["allocator_follow_up"]["evidence"]["soaks"]
        changed = json.loads((ROOT / entry["path"]).read_bytes())
        changed["allocator_settings"] = "dirty_decay_ms:1000"
        raw = json.dumps(changed).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soaks.json"
            path.write_bytes(raw)
            entry.update(path=str(path), sha256=sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError, "allocator soaks used different package evidence or settings"):
                checker.check_package_follow_up(ROOT, candidate)

    def test_rag_tail_change_matches_both_image_orders(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        entry = manifest["distribution_after_terminal_fix"]["allocator_follow_up"]["evidence"]["rag"]
        report = json.loads((ROOT / entry["path"]).read_bytes())
        parent = report["arms"]["parent-first"]["image_id"]
        variant = report["arms"]["packaged-first"]["image_id"]
        allocator = report["arms"]["packaged-first"]["server"]["allocator"]
        args = (report, parent, variant, allocator["library_sha256"], allocator["malloc_conf"])
        checker.check_latest_allocator_rag(*args)

        report["cells"]["rag-c8-seed2"]["change_percent_by_order"]["second"]["ttft_p95_ms"] = 0
        with self.assertRaisesRegex(ValueError, "seed2 RAG TTFT change differs"):
            checker.check_latest_allocator_rag(*args)

    def test_fixed_arrival_rag_binds_capture_and_write_precision(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        records = manifest["distribution_after_terminal_fix"]["allocator_follow_up"]["evidence"]
        closed = json.loads((ROOT / records["rag"]["path"]).read_bytes())
        replay = json.loads((ROOT / records["rag_replay"]["path"]).read_bytes())
        checksum = records["rag"]["sha256"]
        checker.check_latest_allocator_rag_replay(replay, closed, checksum)

        schedule = replay["schedules"]["parent-first"]
        precision = "all_p95_relative_write_errors_at_most_5ms"
        schedule[precision] = not schedule[precision]
        with self.assertRaisesRegex(ValueError, "write precision differs"):
            checker.check_latest_allocator_rag_replay(replay, closed, checksum)
        schedule[precision] = not schedule[precision]
        schedule["source_sha256"] = replay["schedules"]["packaged-first"]["source_sha256"]
        with self.assertRaisesRegex(ValueError, "missing a matched schedule"):
            checker.check_latest_allocator_rag_replay(replay, closed, checksum)


class AllocatorTrialTests(unittest.TestCase):
    def setUp(self):
        manifest = json.loads((ROOT / checker.MANIFEST).read_bytes())
        self.trial = manifest["current_allocator_trial"]
        self.distribution = manifest["current_distribution_candidate"]

    def test_linked_reports_pass(self):
        self.assertEqual(checker.check_allocator_trial(ROOT, self.trial, self.distribution), 8)

    def test_rehashed_evidence_from_another_arm_or_run_fails(self):
        def change_prior_observation(report):
            report["result"]["matched_prior_report_sha256"] = "0" * 64
            encoded = (json.dumps(report["result"], indent=2, allow_nan=False) + "\n").encode("utf-8")
            report["execution"]["remote_verified_json_sha256"] = sha256(encoded).hexdigest()

        for kind, replace, message in (
            ("accuracy", lambda report: report.update(combined_report_sha256="0" * 64), "combined GPU report"),
            ("performance", lambda report: report["arms"]["packaged-first"].update(image_id="sha256:" + "0" * 64),
             "different image"),
            ("rag_confirmation", lambda report: report.update(prior_closed_loop_summary_sha256="0" * 64),
             "different prior performance report"),
            ("rag_confirmation", lambda report: report["cells"]["rag-c8-seed2"].update(aligned_samples_per_arm=64),
             "three matched 256-request cells"),
            ("metrics_observer_warmed", change_prior_observation, "does not match the original run"),
        ):
            with self.subTest(report=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                trial = deepcopy(self.trial)
                for role, entry in trial["evidence"].items():
                    raw = (ROOT / entry["path"]).read_bytes()
                    destination = root / entry["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if role == kind:
                        report = json.loads(raw)
                        replace(report)
                        raw = json.dumps(report).encode("utf-8")
                        entry["sha256"] = sha256(raw).hexdigest()
                    destination.write_bytes(raw)
                self.assertEqual(checker.check_evidence(root, trial), 8)
                with self.assertRaisesRegex(ValueError, message):
                    checker.check_allocator_trial(root, trial, self.distribution)
