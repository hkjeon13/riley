"""Immutable PR 01 reference-lane contract."""

from __future__ import annotations

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
MODEL_CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
TOKENIZER_ARTIFACT_FILENAMES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
TOKENIZER_FILES_SHA256 = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
TOKENIZER_SHA256 = "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db"
DTYPE = "bf16"
MAX_CONTEXT_TOKENS = 8192
# On the pinned RTX 4090/BF16/eager stack, Hugging Face itself can choose a
# different greedy token at output index 17 when comparing a full-prefix
# no-cache forward with a one-token KV-cache forward.  The first 16 decode
# steps are the predeclared exact golden window; longer performance runs use
# the cache-on benchmark lane and are not reference-fixture parity evidence.
GOLDEN_GREEDY_MAX_NEW_TOKENS = 16

PRIMARY_GPU_NAME = "NVIDIA GeForce RTX 4090"
PRIMARY_GPU_COMPUTE_CAPABILITY = "8.9"
PRIMARY_GPU_COUNT = 1
PRIMARY_GPU_MEMORY_MIB = 24_564
PRIMARY_ENVIRONMENT_ID = "rtx4090-ubuntu22-driver580-v1"
PRIMARY_NVIDIA_DRIVER_VERSION = "580.173.02"
PRIMARY_DRIVER_CUDA_API_VERSION = "13.0"
PRIMARY_CPU_MODEL = "Intel Core i7-13700K"
PRIMARY_CPU_PHYSICAL_CORES = 16
PRIMARY_CPU_LOGICAL_THREADS = 24
PRIMARY_CPU_GOVERNOR = "powersave"
PRIMARY_CPU_GOVERNOR_POLICY_COUNT = 24
PRIMARY_RAM_BYTES = 67_185_598_464
PRIMARY_OS_ID = "ubuntu"
PRIMARY_OS_VERSION_ID = "22.04"
PRIMARY_KERNEL_RELEASE = "6.8.0-138-generic"
PRIMARY_MACHINE = "x86_64"

PYTHON_VERSION = "3.13.15"
PYTHON_VERSION_RANGE = ">=3.13,<3.14"
PYTHON_EXECUTABLE_SHA256 = "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866"
PYTHON_PLATFORM_SYSTEM = "linux"
PYTHON_PLATFORM_MACHINE = "x86_64"
PYTHON_VERSION_FILE_SHA256 = "861b3dd8083d28f336ef70f6755bc399538ddad627b1d095820ca34cb953cf14"
TORCH_VERSION = "2.13.0"
TRANSFORMERS_VERSION = "5.15.1"
NVIDIA_ML_PY_VERSION = "13.610.43"
PSUTIL_VERSION = "7.2.2"
SAFETENSORS_VERSION = "0.8.0"

FIXTURE_SCHEMA_VERSION = "1.0.0"
PROMPT_CONTRACT_VERSION = "1.0.0"
GENERATOR_NAME = "riley-python-reference"
GENERATOR_VERSION = "0.1.0"

RNG_ALGORITHM = "riley.philox4x32-10.v1"
SEMANTIC_CLASS = "reference"
RUNTIME_DEPENDENCY_CLASS = "python-reference"
ATTENTION_BACKEND = "eager"

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
