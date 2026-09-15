from __future__ import annotations

from copy import deepcopy
import json
import sys
import tempfile
import unittest
from pathlib import Path


NEXT_STAGE = Path(__file__).resolve().parents[1]
if str(NEXT_STAGE) not in sys.path:
    sys.path.insert(0, str(NEXT_STAGE))

import validate_stage_contract as validator


MODELS_PATH = NEXT_STAGE / "model-descriptors-v1.json"
NUMERICAL_PATH = NEXT_STAGE / "numerical-profiles-v1.json"


class StageContractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
        self.numerical = json.loads(NUMERICAL_PATH.read_text(encoding="utf-8"))

    def _validate(self, models: object, numerical: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models_path = root / "models.json"
            numerical_path = root / "numerical.json"
            models_path.write_text(json.dumps(models), encoding="utf-8")
            numerical_path.write_text(json.dumps(numerical), encoding="utf-8")
            validator.validate_paths(models_path, numerical_path)

    def _assert_invalid(self, models: object, numerical: object, message: str) -> None:
        with self.assertRaisesRegex(validator.ContractError, message):
            self._validate(models, numerical)

    def test_checked_in_documents_validate(self) -> None:
        self._validate(self.models, self.numerical)

    def test_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models_path = root / "models.json"
            numerical_path = root / "numerical.json"
            models_path.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")
            numerical_path.write_text(json.dumps(self.numerical), encoding="utf-8")
            with self.assertRaisesRegex(validator.ContractError, "duplicate JSON key"):
                validator.validate_paths(models_path, numerical_path)

    def test_rejects_unpinned_source_revision(self) -> None:
        models = deepcopy(self.models)
        models["models"][2]["source"]["revision"] = "0" * 40
        self._assert_invalid(models, self.numerical, "pinned source model and revision")

    def test_rejects_incorrect_architecture_or_tensor_shape(self) -> None:
        models = deepcopy(self.models)
        models["models"][1]["architecture"]["rope"]["theta"] = 100000
        self._assert_invalid(models, self.numerical, "pinned architecture")

    def test_rejects_wrong_bf16_kv_payload(self) -> None:
        models = deepcopy(self.models)
        models["models"][2]["memory_scenarios"][2]["kv_payload_bytes"] += 2
        self._assert_invalid(models, self.numerical, "BF16 K/V payload formula")

    def test_rejects_memory_budget_that_spends_the_reserve(self) -> None:
        models = deepcopy(self.models)
        models["models"][0]["memory_scenarios"][0]["remaining_non_model_budget_bytes"] += 1
        self._assert_invalid(models, self.numerical, "20GB limit and 1GB reserve")

    def test_rejects_qwen_d128_as_supported(self) -> None:
        models = deepcopy(self.models)
        models["models"][2]["serving_support"]["status"] = "supported"
        self._assert_invalid(models, self.numerical, "pinned implementation support state")

    def test_rejects_native_profile_with_implicit_execution_semantics(self) -> None:
        numerical = deepcopy(self.numerical)
        numerical["profiles"][1]["execution"]["rope_placement"] = "q-and-k-before-attention"
        self._assert_invalid(self.models, numerical, "arithmetic-placement policy explicitly pending")

    def test_rejects_native_profile_false_qualification(self) -> None:
        numerical = deepcopy(self.numerical)
        numerical["profiles"][1]["performance_eligibility"] = "qualified"
        self._assert_invalid(self.models, numerical, "unqualified native BF16 R0 profile")

    def test_rejects_profile_linkage_mismatch(self) -> None:
        numerical = deepcopy(self.numerical)
        numerical["profiles"][1]["applies_to"] = [
            "smollm2-135m-strict-regression",
            "smollm2-1.7b-instruct-connection",
        ]
        self._assert_invalid(self.models, numerical, "unqualified native BF16 R0 profile")


if __name__ == "__main__":
    unittest.main()
