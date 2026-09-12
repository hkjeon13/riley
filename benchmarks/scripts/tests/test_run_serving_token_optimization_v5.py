"""CPU controller contract tests; no GPU, remote execution or desktop mutation."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_serving_token_optimization_v5 as runner


def plan_fixture():
    return {"schema_version": runner.SCHEMA,
            "workload": {"id": "tokens-new-c1-c2", "purpose": "initial-measurement", "pairs": 5,
                         "retained_requests_per_process": 1000, "warmups_per_worker_per_transport": 5,
                         "arrival_policy": "closed-loop-refill"},
            "settings": [{"id": "c1", "offered_concurrency": 1, "vllm_token_budget": 128},
                         {"id": "c2", "offered_concurrency": 2, "vllm_token_budget": 256}],
            "comparisons": [{"id": "api-vllm", "left": "api", "right": "vllm"}],
            "lanes": {"api": {"kind": "riley", "port": 19341}, "vllm": {"kind": "vllm", "port": 19342}},
            "startup_timeout_seconds": 120, "request_timeout_seconds": 30, "cooldown_timeout_seconds": 120}


def lane_result(name):
    distribution = {"median": 2, "p95": 3, "p99": 4, "samples": 1000, "max": 5}
    metrics = {key: dict(distribution) for key in ("token_ttft_ms", "token_tpot_ms", "token_itl_ms", "e2e_ms", "first_text_ms")}
    metrics["successful_output_tokens_per_wall_second"] = 100 if name == "api" else 125
    return {"completed": True, "summary": metrics}


class ControllerTests(unittest.TestCase):

    def test_grouped_client_historical_documents_pinned_without_relative_reinterpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            historical=root/'historical.json'
            runner.write(historical, {'files':[{'path':'config.json','bytes':704,'sha256':'a'*64}]})
            raw=root/'raw.jsonl';raw.write_text('recorded raw')
            log=root/'cpu.log';log.write_text('26 passed')
            proof=root/'proof.json'
            runner.write(proof, {'schema_version':'riley.grouped-token-client-qualification.v1',
                'inputs':{'plan':runner.evidence(historical)},
                'raw_files':{'retained':runner.evidence(raw)}, 'tests':{'log':runner.evidence(log)}})
            inventory={};runner.freeze_receipt(runner.evidence(proof),inventory,validated_grouped_receipt=runner.evidence(proof))
            self.assertEqual(set(inventory),{str(proof),str(historical),str(raw),str(log)})
            historical.write_text('changed historical bytes')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                runner.freeze_receipt(runner.evidence(proof),{},validated_grouped_receipt=runner.evidence(proof))


    def test_grouped_measurement_and_existing_api_clients_bound_separately(self):
        from types import SimpleNamespace
        plan={'token_client_validation':{'path':'proof','sha256':'proofhash'},
              'token_client_validator':{'path':'validator','sha256':'validatorhash'},
              'reference':{'binding':{'path':'binding','sha256':'bindinghash'},'parent_c1_plan':{'path':'parent','sha256':'parenthash'}}}
        expected={'schema_version':'riley.grouped-token-client-qualification.v1','completed':True,
                  'client_v1':runner.evidence(runner.qualified_tokens.__file__),
                  'client_v2':runner.evidence(runner.tokens.__file__),
                  'inputs':{'binding':plan['reference']['binding'],'reference_parent':plan['reference']['parent_c1_plan']}}
        self.assertNotEqual(expected['client_v1'],expected['client_v2'])
        validator=SimpleNamespace(validate_completion=Mock(return_value=expected))
        with patch.object(runner,'check_ref',side_effect=lambda ref,pins:ref['path']), \
                patch.object(runner,'load_module',return_value=validator):
            self.assertEqual(runner.validate_measurement_client(plan,{}),expected)
            for key in ('client_v1','client_v2'):
                bad=copy.deepcopy(expected);bad[key]={'path':'wrong','sha256':'wrong'}
                validator.validate_completion.return_value=bad
                with self.assertRaisesRegex(ValueError,'client differs'):
                    runner.validate_measurement_client(plan,{})


    def test_initial_settings_and_screening_are_distinct(self):
        plan = plan_fixture()
        runner.validate_workload(plan)
        for mutate in (lambda p: p["workload"].update(pairs=1),
                       lambda p: p["workload"].update(retained_requests_per_process=256),
                       lambda p: p["settings"][1].update(offered_concurrency=4, vllm_token_budget=512),
                       lambda p: p["settings"][1].update(vllm_token_budget=128),
                       lambda p: p["workload"].update(warmups_per_worker_per_transport=0),
                       lambda p: p.update(schema_version="riley.serving-concurrency-plan.v1")):
            bad = copy.deepcopy(plan); mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_workload(bad)
        plan["workload"].update(purpose="screening", pairs=1, retained_requests_per_process=256)
        runner.validate_workload(plan)

    def test_runtime_is_explicit_and_private_first(self):
        runtime = {"child_only_overrides": {"PATH": "/private/bin:/usr/bin", "LD_LIBRARY_PATH": "/private/gl:/private/compute",
                    "__GLX_VENDOR_LIBRARY_NAME": "nvidia"}}
        plan = {"base_environment": {"HOME": "/tmp/explicit", "PATH": "/usr/bin"}}
        lane = {"env": {"LD_LIBRARY_PATH": "/private/gl:/private/compute:/cuda/lib", "CUDA_VISIBLE_DEVICES": "0"}}
        with patch.dict("os.environ", {"UNRELATED_SENTINEL": "secret", "LD_PRELOAD": "bad"}):
            result = runner.lane_environment(plan, lane, runtime)
        self.assertNotIn("UNRELATED_SENTINEL", result)
        self.assertNotIn("LD_PRELOAD", result)
        for env in ({"LD_LIBRARY_PATH": "/host/lib:/private/compute"}, {"VLLM_BATCH_INVARIANT": "1"}, {"RILEY_PROFILE": "1"}, {"PYTHONPATH": "/other"}):
            with self.assertRaises(ValueError):
                runner.lane_environment(plan, {"env": env}, runtime)

    def test_evidence_checks_nested_log_refs_and_exact_plan_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "log"
            path.write_text("one")
            ref = runner.evidence(path)
            runner.check_refs({"checks": [{"name": "test", "passed": True, **ref}]})
            path.write_text("two")
            with self.assertRaises(ValueError):
                runner.check_refs({"checks": [{"name": "test", "passed": True, **ref}]})
            plan_path = Path(temporary) / "plan.json"; plan_path.write_text('{"lanes": {}}')
            plan = runner.read(plan_path); prepared = {"plan": runner.evidence(plan_path)}
            plan_path.write_text('{ "lanes": {} }')
            with patch.object(runner, "validate_pins"):
                with self.assertRaisesRegex(ValueError, "exact plan bytes"):
                    runner.unchanged(plan_path, plan, prepared)

    def test_prepare_only_never_stops_or_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary) / "plan.json", Path(temporary) / "out"
            path.write_text("{}")
            prepared = {"measurement_started": False}
            with patch.object(runner, "prepare", return_value=({}, prepared, None, None, None, None)), \
                    patch.object(runner, "measure") as measure, patch.object(runner, "session_action") as session, \
                    patch.object(runner.subprocess, "Popen") as popen, patch.object(runner, "gpu_snapshot") as gpu:
                runner.main(["--plan", str(path), "--output", str(output)])
            measure.assert_not_called(); session.assert_not_called(); popen.assert_not_called(); gpu.assert_not_called()
            self.assertEqual(runner.read(output / "preparation.json"), prepared)
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.main(["--plan", str(path), "--output", str(output)])

    def test_pair_requires_both_complete_and_direction_is_explicit(self):
        comparison = plan_fixture()["comparisons"][0]
        result = runner.pair_result(comparison, 1, ["api", "vllm"], {name: lane_result(name) for name in ("api", "vllm")})
        self.assertEqual(result["ratios"]["throughput_right_over_left"], 1.25)
        with self.assertRaisesRegex(ValueError, "incomplete pair"):
            runner.pair_result(comparison, 1, ["api", "vllm"], {"api": lane_result("api")})

    def test_every_phase_including_warmups_requires_strict_exact_overlap(self):
        row = {"status": "success", "mode": "strict", "protocol_valid": True, "transport_complete": True, "reference_match": True}
        accounting = {"completed": True, "requested": 2, "succeeded": 2, "failed": 0, "observed_max_request_in_flight": 2}
        runner.phase_check([row, row], accounting, 2, 2)
        for key, value in (("mode", "observe"), ("reference_match", False), ("transport_complete", False)):
            with self.assertRaises(ValueError):
                runner.phase_check([{**row, key: value}, row], accounting, 2, 2)
        with self.assertRaises(ValueError):
            runner.phase_check([row, row], {**accounting, "observed_max_request_in_flight": 1}, 2, 2)

    def campaign(self, temporary, *, lane_failure=False, stop_failure=False, restore_failure=False):
        output = Path(temporary) / "output"; output.mkdir()
        session_root = Path(temporary) / "round13"; session_root.mkdir()
        snapshot = session_root / "session.json"; snapshot.write_text("[]")
        session = Mock(ROOT=session_root, SNAPSHOT=snapshot)
        runtime, sessions = {"runtime": "test"}, [1, 2, 3]
        session.validate_runtime.return_value = runtime
        session.bound_runtime.return_value = runtime
        session.check.return_value = sessions
        plan = plan_fixture()
        plan["session"] = {"root": str(session_root)}
        runner.write(session_root / "deadline.json", time.time() + 1800)
        prepared = {"runtime": runtime, "sessions": sessions, "plan": {"path": "unused", "sha256": "unused"}}
        calls, actions = [], []
        def action(plan, name, directory):
            actions.append(name)
            if name == "stop" and stop_failure:
                raise RuntimeError("partial stop")
            if name == "restore" and restore_failure:
                raise RuntimeError("restore pending")
        def lane(plan, setting, name, directory, *args):
            calls.append((setting["id"], name))
            if lane_failure:
                directory.mkdir()
                runner.write(directory / "retained-accounting.json", {"completed": False, "failed": 1})
                raise RuntimeError("exact token mismatch")
            return lane_result(name)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "unchanged"))
            stack.enter_context(patch.object(runner.shared, "check_port"))
            stack.enter_context(patch.object(runner, "session_action", side_effect=action))
            stack.enter_context(patch.object(runner, "run_lane", side_effect=lane))
            stack.enter_context(patch.object(runner, "verify_restoration", return_value={"path": "restore", "sha256": "ok"}))
            if lane_failure or stop_failure or restore_failure:
                with self.assertRaisesRegex(ValueError, "measurement incomplete"):
                    runner.measure(Path("unused"), output, plan, prepared, session, None, None, None)
            else:
                runner.measure(Path("unused"), output, plan, prepared, session, None, None, None)
        return output, calls, actions

    def test_five_alternating_pairs_per_setting_then_restoration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, calls, actions = self.campaign(temporary)
            expected = ["api", "vllm", "vllm", "api", "api", "vllm", "vllm", "api", "api", "vllm"]
            self.assertEqual(calls, [(c, lane) for c in ("c1", "c2") for lane in expected])
            self.assertEqual(actions, ["stop", "restore"])
            result = runner.read(output / "completion.json")
            self.assertEqual(len(result["pairs"]), 10)
            self.assertFalse(result["p99_stability_qualified"])
            self.assertFalse(result["performance_claim"])

    def test_failed_lane_preserves_accounting_and_restores_without_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, calls, actions = self.campaign(temporary, lane_failure=True)
            self.assertEqual(actions, ["stop", "restore"])
            self.assertEqual(len(calls), 1)
            self.assertFalse((output / "completion.json").exists())
            self.assertEqual(len(list(output.rglob("retained-accounting.json"))), 1)
            self.assertTrue(runner.read(output / "finalization.json")["failure"])

    def test_partial_stop_and_failed_restore_never_promote(self):
        for kwargs in ({"stop_failure": True}, {"lane_failure": True, "restore_failure": True}):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as temporary:
                output, _, actions = self.campaign(temporary, **kwargs)
                self.assertEqual(actions, ["stop", "restore"])
                self.assertFalse((output / "completion.json").exists())

    def test_restoration_uses_actual_new_pid_schema_birth_and_original_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            originals = [{"pid": i, "port": 9800+i, "argv": ["blender", str(i)], "cwd": "/tmp", "env": {"DISPLAY": ":0"}} for i in range(1, 4)]
            resumed = [{"original_pid": i, "new_pid": 10+i, "port": 9800+i, "start": str(100+i), "tag": str(i)} for i in range(1, 4)]
            receipt = {"alive_and_listening": True, "commands_and_gui_environment_match": True,
                       "all_relaunched_processes_have_pinned_vendor_maps": True, "processes": resumed}
            runner.write(root / "verified.json", receipt)
            session = Mock(ROOT=root)
            runtime = {"identity": "runtime"}
            session.bound_runtime.return_value = runtime
            session.live.return_value = session.listening.return_value = True
            session.identity.side_effect = lambda pid: {**originals[pid-11], "pid": pid, "start": str(100+pid-10)}
            session.same_command.side_effect = lambda current, old: all(current[k] == old[k] for k in ("argv", "cwd", "env"))
            session.process_tag.side_effect = lambda pid: str(pid-10)
            session.verify_private_maps.return_value = {"ready": True}
            prepared = {"runtime": runtime, "sessions": originals}
            runner.verify_restoration(session, prepared)
            session.process_tag.return_value = "reused"
            session.process_tag.side_effect = None
            with self.assertRaisesRegex(ValueError, "tag"):
                runner.verify_restoration(session, prepared)


    def test_alias_content_and_nested_receipt_closure_are_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target, alias = root / "blob", root / "model.safetensors"
            target.write_bytes(b"model")
            alias.symlink_to(target)
            log = root / "proof.log"; log.write_text("passed")
            inner = root / "inner.json"
            runner.write(inner, {"checks": [{"name": "gpu", **runner.evidence(log)}], "model_files": {str(alias): runner.shared.digest(alias)}})
            outer = root / "outer.json"; runner.write(outer, {"model_tests": runner.evidence(inner)})
            inventory = {}
            runner.freeze_receipt(runner.evidence(outer), inventory)
            self.assertEqual(set(inventory), {str(outer), str(inner), str(log), str(alias)})
            contract = root / "contract.md"; contract.write_text("contract")
            pins = {str(alias): runner.shared.digest(alias), str(contract): runner.shared.digest(contract)}
            for module in (runner, runner.tokens, runner.qualified_tokens, runner.legacy, runner.shared):
                pins[str(Path(module.__file__).resolve())] = runner.shared.digest(module.__file__)
            plan = {"immutable_files": pins, "contract": runner.evidence(contract), "lanes": {}}
            runner.validate_pins(plan)
            plan_path = root / "plan.json"; runner.write(plan_path, plan)
            prepared = {"plan": runner.evidence(plan_path), "transitive_evidence": inventory,
                        "resolved_targets": {str(alias): str(target)}}
            runner.unchanged(plan_path, plan, prepared)
            equal_target = root / "equal-blob"; equal_target.write_bytes(b"model")
            alias.unlink(); alias.symlink_to(equal_target)
            with self.assertRaisesRegex(ValueError, "symlink target"):
                runner.unchanged(plan_path, plan, prepared)
            alias.unlink(); alias.symlink_to(target)
            log.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "transitive qualification"):
                runner.unchanged(plan_path, plan, prepared)

    def test_termination_handlers_restore_prior_handlers(self):
        saved = {}
        def install(signum, handler):
            saved[signum] = handler
            return "previous-" + str(signum)
        with patch.object(runner.signal, "signal", side_effect=install) as install_mock:
            with runner.termination_handlers():
                with self.assertRaises(runner.MeasurementInterrupted):
                    saved[runner.signal.SIGTERM](runner.signal.SIGTERM, None)
                saved[runner.signal.SIGTERM](runner.signal.SIGTERM, None)  # Repeated signal cannot interrupt cleanup.
            self.assertEqual(install_mock.call_count, 4)
            self.assertEqual(saved[runner.signal.SIGTERM], "previous-" + str(runner.signal.SIGTERM))


    def test_termination_is_not_swallowed_by_readiness_io_retry(self):
        installed = {}
        def install(signum, handler):
            installed[signum] = handler
            return None
        with patch.object(runner.signal, "signal", side_effect=install), runner.termination_handlers():
            connection = Mock()
            connection.request.side_effect = lambda *args: installed[runner.signal.SIGTERM](runner.signal.SIGTERM, None)
            process = Mock()
            process.poll.return_value = None
            with patch.object(runner.shared.http.client, "HTTPConnection", return_value=connection):
                with self.assertRaises(runner.MeasurementInterrupted):
                    runner.shared.wait_ready(process, 19341, 1)
            connection.close.assert_called_once()


    def test_lane_runs_both_concurrent_warmups_and_retained_then_cleans_up(self):
        for fail_warmup in (False, True):
            with self.subTest(fail_warmup=fail_warmup), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                directory = Path(temporary) / "lane"
                plan = plan_fixture()
                plan["lanes"]["api"].update(argv=["/qualified/riley", "serve"], cwd="/qualified/source", env={})
                plan["base_environment"] = {"HOME": "/tmp/declared", "PATH": "/usr/bin"}
                runtime = {"child_only_overrides": {"PATH": "/private/bin:/usr/bin", "LD_LIBRARY_PATH": "/private/lib"}}
                reference = runner.tokens.TokenReference("g04-smol", tuple(range(128)), tuple(range(32)), "text")
                process = Mock(pid=424242, returncode=0)
                process.poll.return_value = None
                stack.enter_context(patch.object(runner.shared, "check_port"))
                stack.enter_context(patch.object(runner, "cooldown", return_value={"condition": "idle"}))
                stack.enter_context(patch.object(runner.shared, "wait_ready"))
                stack.enter_context(patch.object(runner, "compute_maps", return_value={"compute_loaded": True}))
                stack.enter_context(patch.object(runner.subprocess, "Popen", return_value=process))
                stack.enter_context(patch.object(runner.os, "getsid", return_value=424242))
                stack.enter_context(patch.object(Path, "iterdir", return_value=iter(())))
                cleanup = stack.enter_context(patch.object(runner.shared, "stop_owned_process"))
                phases = []
                def phase(client, one, *, concurrency, count, phase, per_worker):
                    phases.append((phase, concurrency, count, per_worker))
                    success = not fail_warmup
                    row = {"status": "success" if success else "failed", "mode": "strict", "protocol_valid": success,
                           "transport_complete": success, "reference_match": success}
                    accounting = {"completed": success, "requested": count, "succeeded": count if success else 0,
                                  "failed": 0 if success else count, "observed_max_request_in_flight": concurrency}
                    return [row.copy() for _ in range(count)], accounting
                stack.enter_context(patch.object(runner.tokens, "run_phase", side_effect=phase))
                stack.enter_context(patch.object(runner.tokens, "summarize_phase", return_value=lane_result("api")["summary"]))
                args = (plan, plan["settings"][1], "api", directory, Mock(), runtime, reference, {"prompt": "Hello"}, {}, Mock())
                if fail_warmup:
                    with self.assertRaisesRegex(ValueError, "lane incomplete"):
                        runner.run_lane(*args)
                    self.assertEqual(phases, [("warmup-nonstream", 2, 10, True)])
                    self.assertFalse((directory / "summary.json").exists())
                    self.assertTrue((directory / "warmup-nonstream.jsonl").exists())
                else:
                    runner.run_lane(*args)
                    self.assertEqual(phases, [("warmup-nonstream", 2, 10, True), ("warmup-stream", 2, 10, True), ("retained", 2, 1000, False)])
                    self.assertTrue((directory / "summary.json").exists())
                cleanup.assert_called_once_with(process)
                self.assertTrue(runner.read(directory / "process-exit.json")["cleanup"]["cleanup_verified"])


    def test_native_libcuda_alias_nested_receipt_is_valid_but_explicit_ref_is_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target, alias = root / "libcuda.so.580.173.02", root / "libcuda.so.1"
            target.write_bytes(b"pinned-native-driver")
            alias.symlink_to(target.name)
            receipt = {"path": str(alias), "sha256": runner.shared.digest(alias)}
            runner.check_refs({"runtime": {"libcuda": receipt}})
            runner.check_refs({"name": "native", **receipt})
            with self.assertRaisesRegex(ValueError, "canonical"):
                runner.check_ref(receipt)
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "nested artifact changed"):
                runner.check_refs({"runtime": {"libcuda": receipt}})


    def test_exact_historical_local_receipt_is_metadata_only(self):
        actual = SCRIPTS.parent / "results/20260912-serving-optimization/http-token-source-after-v2.json"
        self.assertEqual(runner.shared.digest(actual), runner.LOCAL_SOURCE_RECEIPT["sha256"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt = root / "local-source.json"; receipt.write_bytes(actual.read_bytes())
            parent = root / "qualification.json"
            runner.write(parent, {"local_source_receipt": runner.evidence(receipt)})
            inventory = {}
            runner.freeze_receipt(runner.evidence(parent), inventory)
            self.assertEqual(set(inventory), {str(parent), str(receipt)})
            # The exception is pinned to exact bytes, never all local receipts.
            receipt.write_bytes(actual.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "transitive artifact"):
                runner.freeze_receipt(runner.evidence(receipt), {})

    def test_module_loader_registers_validator_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validator.py"
            path.write_text("import sys\nassert sys.modules[__name__] is __import__(__name__)\nvalue=42\n")
            self.assertEqual(runner.load_module(path, "test_registered_token_validator").value, 42)


if __name__ == "__main__":
    unittest.main()


class ArrayReceiptRegression(unittest.TestCase):
    def test_array_receipt_still_freezes_nested_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'leaf.txt'
            leaf.write_text('nested evidence')
            receipt = root / 'array.json'
            receipt.write_text(json.dumps([runner.evidence(leaf)]))
            inventory = {}
            runner.freeze_receipt(runner.evidence(receipt), inventory)
            self.assertEqual(inventory[str(leaf.resolve())], runner.evidence(leaf)['sha256'])
            leaf.write_text('changed')
            with self.assertRaises(ValueError):
                runner.freeze_receipt(runner.evidence(receipt), {})
