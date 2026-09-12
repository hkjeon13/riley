"""Local parser regression tests; never compile CUDA or execute GPU work."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "batch6_projection_probe", Path(__file__).with_name("batch6_projection_probe.py")
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProductionFlagsTests(unittest.TestCase):
    def test_documented_cmake_cache_entries_are_line_local(self):
        with tempfile.TemporaryDirectory(prefix="riley-cmake-cache-test-") as temporary:
            root = Path(temporary).resolve()
            source = root / "batch6-source"
            target = root / "batch6-target"
            cmake = target / "release/build/riley-cuda-fixture/out/cuda-native-build"
            directory = cmake / "CMakeFiles/riley_cuda_native.dir"
            directory.mkdir(parents=True)
            nvcc = root / "cuda/bin/nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("fixture compiler; never executed")
            runtime = root / "cuda/libcudart.so"
            runtime.write_text("fixture runtime; never loaded")
            build = root / "batch6-build.json"
            build.write_text(json.dumps({"build_environment": {"CARGO_TARGET_DIR": str(target)}}))
            manifest = {"source_build": {"path": str(build)}, "source_root": str(source)}
            # Real CMake caches use comments and blank lines before each key.
            cache = (
                "# This is the CMakeCache file.\n\n"
                "//Path to a program.\n"
                f"CMAKE_CUDA_COMPILER:FILEPATH={nvcc}\n\n"
                "//Choose the type of build, options are: None Debug Release\n"
                "CMAKE_BUILD_TYPE:STRING=Release\n\n"
                "//Source directory with the top level CMakeLists.txt file\n"
                f"CMAKE_HOME_DIRECTORY:INTERNAL={source}/kernels\n\n"
            )
            (cmake / "CMakeCache.txt").write_text(cache)
            (cmake / "riley-cuda-cudart-Release.path").write_text(str(runtime) + "\n")
            (directory / "includes_CUDA.rsp").write_text(f'-I"{source}/kernels/include"\n')
            (directory / "flags.make").write_text(
                "CUDA_DEFINES = \n"
                "CUDA_INCLUDES = --options-file CMakeFiles/riley_cuda_native.dir/includes_CUDA.rsp\n"
                'CUDA_FLAGS = -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" '
                "-Xcompiler=-fPIC --objdir-as-tempdir -Xcompiler=-fno-exceptions\n"
            )
            obj = "CMakeFiles/riley_cuda_native.dir/src/graph_numerics_precise.cu.o"
            (directory / "build.make").write_text(
                f"\t{nvcc} -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) "
                f"-MD -MT {obj} -MF {obj}.d -x cu -c {source}/{PROBE.PRECISE} -o {obj}\n"
            )
            options = PROBE.production_flags(manifest, nvcc)
            self.assertEqual(options["original_compile_cwd"], str(cmake))
            self.assertIn("-I" + str(source / "kernels/include"), options["flags"])
            self.assertEqual(options["runtime_library_link"], str(runtime))
            # Windows line endings also stay within a single cache entry.
            (cmake / "CMakeCache.txt").write_bytes(cache.replace("\n", "\r\n").encode())
            self.assertEqual(PROBE.production_flags(manifest, nvcc)["flags"], options["flags"])
            (cmake / "CMakeCache.txt").write_text(cache.replace("STRING=Release", "STRING=Debug"))
            with self.assertRaisesRegex(ValueError, "native build was not Release"):
                PROBE.production_flags(manifest, nvcc)


if __name__ == "__main__":
    unittest.main()
