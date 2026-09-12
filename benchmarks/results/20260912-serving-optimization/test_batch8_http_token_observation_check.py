#!/usr/bin/env python3
"""CPU-only child adaptation contracts, including archived receipt reconstruction."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batch8_http_token_observation_check as child
import http_token_observation_check_v3 as baseline


class ChildContracts(unittest.TestCase):
    def test_sampler_changes_only_source_and_binary_paths(self):
        expected = inspect.getsource(baseline.run_sampler).replace("root/'http-token-target/release/riley'", "root/'batch8-target/release/riley'")
        expected = expected.replace("root/'http-token-source'", "root/'batch8-source'")
        self.assertEqual(inspect.getsource(child.run_sampler), expected)
        self.assertNotIn('--c02-', expected)
        self.assertEqual(child.PROOFS, baseline.PROOFS)
        self.assertEqual(child.check_specs(), baseline.check_specs())
        self.assertEqual(child.STOP_TEXT, ", I'm")

    def test_isolated_contract_leaves_baseline_identity_and_files_unchanged(self):
        self.assertIsNot(child.contract, baseline)
        self.assertEqual(baseline.EXPECTED_COMMIT, 'a179617070526068b66ba5627ba82a7151da8c64')
        self.assertEqual(baseline.SCHEMA, 'riley.http-token-observation-correctness.v3')
        self.assertEqual(child.contract.EXPECTED_COMMIT, child.EXPECTED_COMMIT)
        self.assertEqual(child.shared.sha(HERE/'http_token_observation_check_v3.py'), child.PARENT_SHA)
        self.assertTrue(child.dependency_evidence())

    def test_child_model_adapter_rejects_wrong_path_and_delegates_actual_contract(self):
        root, base = Path('/tmp/child'), Path('/tmp/base')
        build = {'source_commit': child.EXPECTED_COMMIT,
                 'binaries': {str(root/'batch8-target/release/riley'): child.EXPECTED_BINARY}}
        with patch.object(child.qualifier, 'validate_model', return_value={'server_lib_tests': {'mock': True}}) as validate:
            result = child.validate_child_full_gpu_receipt(build, root/'batch8-build.json', root/'batch8-model-tests.json', base/'native-binding.json')
            self.assertEqual(result, {'server_unit_tests': {'mock': True}})
            validate.assert_called_once_with(root, base, build)
            with self.assertRaisesRegex(ValueError, 'path differs'):
                child.validate_child_full_gpu_receipt(build, root/'batch8-build.json', root/'http-token-gpu-tests.json', base/'native-binding.json')

    def test_archived_actual_qualification_schema_and_inherited_http_pins(self):
        raw = HERE/'raw'; root = Path('/tmp/riley-opt-260912'); base = Path('/tmp/riley-g04-vllm-profile-260911')
        build = child.read(raw/'batch8-build.json'); model = child.read(raw/'batch8-model-tests.json')
        fusion = child.read(raw/'batch8-fusion-probe/receipt.json')
        qualified = child.read(raw/'batch8-qualification.json')
        binding = child.read(HERE.parent/'20260911-g04-vllm-profile/native-binding.json')
        original_read, original_sha = child.read, child.shared.sha
        def mapped(path):
            path = Path(path)
            if str(path).startswith(str(root/'batch8-source')+'/'):
                return HERE.parents[2]/path.relative_to(root/'batch8-source')
            if str(path).startswith(str(root)+'/'):
                target = raw/path.relative_to(root)
                return target if target.exists() else HERE/path.relative_to(root)
            return path
        def read(path):
            return copy.deepcopy(qualified) if Path(path) == root/'batch8-qualification.json' else original_read(mapped(path))
        def evidence(path):
            return {'path': str(path), 'sha256': original_sha(mapped(path))}
        with patch.object(child, 'read', side_effect=read), patch.object(child, 'evidence', side_effect=evidence), \
             patch.object(child.shared, 'sha', side_effect=lambda path: original_sha(mapped(path))), \
             patch.object(child.qualifier, 'validate_build', return_value=build), \
             patch.object(child.qualifier, 'validate_model', return_value=model), \
             patch.object(child.qualifier, 'reference', return_value=(binding, model['references'])), \
             patch.object(child.qualifier, 'validate_fusion', return_value=fusion), \
             patch.object(child.contract, 'check_artifacts'):
            observed_build, prereqs = child.validate_prerequisites(root, base)
            self.assertEqual(observed_build, build)
            self.assertEqual(prereqs['inherited_http_files'], qualified['source_lineage']['inherited_http_files'])
            qualified['source_lineage']['inherited_http_files']['crates/riley-server/src/openai.rs'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'reconstructed evidence'):
                child.validate_prerequisites(root, base)


def load_tests(loader, tests, pattern):
    # Run the unchanged V3 sampler success/failure lifecycle against this child.
    # It uses real CPU loopback HTTP, mocked owned server PIDs, and no GPU APIs.
    spec = importlib.util.spec_from_file_location('_batch8_v3_lifecycle_tests', HERE/'test_http_token_observation_check_v3.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.subject = child
    tests.addTests(loader.loadTestsFromTestCase(module.SamplerLifecycle))
    return tests


if __name__ == '__main__': unittest.main()
