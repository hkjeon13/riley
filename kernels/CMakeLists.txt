cmake_minimum_required(VERSION 3.22)

project(rustinfer_cuda_native VERSION 0.1.0 LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)

option(RUSTINFER_CUDA_ENABLE_TEST_FAULT_INJECTION
    "Build destructive test-only CUDA memory fault injection hooks" OFF)
option(RUSTINFER_CUDA_ENABLE_NVML_PROBE
    "Build the development-only in-process NVML environment probe" OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES must contain explicit AOT compile targets")
endif()

message(STATUS
    "rustinfer CUDA AOT architectures: ${CMAKE_CUDA_ARCHITECTURES} (compile targets only)")

add_library(rustinfer_cuda_native STATIC
    src/attention_reference.cu
    src/attention_online.cu
    src/batch_primitives.cu
    src/decode_attention.cu
    src/fixed37_gemm.cu
    src/fixed37_attention.cu
    src/fixed37_primitives.cu
    src/gemm.cu
    src/host_runtime.cu
    src/memory.cu
    src/primitives.cu
    src/smoke_fill.cu
    src/version.cu
)

# This target compiles the installed header as C11 and pins every public
# struct layout independently of the CUDA implementation's C++ assertions.
add_library(rustinfer_cuda_abi_checks OBJECT
    tests/abi_layout.c
)
target_include_directories(rustinfer_cuda_abi_checks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(rustinfer_cuda_abi_checks PRIVATE c_std_11)
add_dependencies(rustinfer_cuda_native rustinfer_cuda_abi_checks)

target_include_directories(rustinfer_cuda_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(rustinfer_cuda_native PUBLIC cuda_std_17)

if(RUSTINFER_CUDA_ENABLE_TEST_FAULT_INJECTION)
    target_compile_definitions(rustinfer_cuda_native PRIVATE
        RUSTINFER_CUDA_ENABLE_TEST_FAULT_INJECTION=1)
endif()

if(RUSTINFER_CUDA_ENABLE_NVML_PROBE)
    if(NOT TARGET CUDA::nvml)
        message(FATAL_ERROR
            "The selected CUDA toolkit does not provide the shared CUDA::nvml target")
    endif()
    target_compile_definitions(rustinfer_cuda_native PRIVATE
        RUSTINFER_CUDA_ENABLE_NVML_PROBE=1)
    target_compile_definitions(rustinfer_cuda_abi_checks PRIVATE
        RUSTINFER_CUDA_ENABLE_NVML_PROBE=1)
endif()

# Static archives do not carry transitive linker directives into Cargo. Keep
# the CMake dependencies accurate and emit the toolkit-selected linker files
# for build.rs to validate and forward explicitly.
target_link_libraries(rustinfer_cuda_native PRIVATE
    CUDA::cublasLt
    CUDA::cudart
    CUDA::cuda_driver
)
if(RUSTINFER_CUDA_ENABLE_NVML_PROBE)
    target_link_libraries(rustinfer_cuda_native PRIVATE CUDA::nvml)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # nvcc otherwise names generated host intermediates with a process-derived
    # tmpxft prefix. Those STT_FILE names survive in the final ELF .strtab and
    # make identical clean builds differ even when code and build IDs match.
    # Keeping intermediates beside the stable object output gives cudafe files
    # deterministic object-derived names without stripping release symbols.
    target_compile_options(rustinfer_cuda_native PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--objdir-as-tempdir>
    )
    if(WIN32)
        target_compile_options(rustinfer_cuda_native PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHs-c->
        )
    else()
        target_compile_options(rustinfer_cuda_native PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-exceptions>
        )
    endif()
endif()

file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/rustinfer-cuda-cudart-$<CONFIG>.path"
    CONTENT "$<TARGET_FILE:CUDA::cudart>\n"
)
file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/rustinfer-cuda-driver-$<CONFIG>.path"
    CONTENT "$<TARGET_FILE:CUDA::cuda_driver>\n"
)
file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/rustinfer-cuda-cublaslt-$<CONFIG>.path"
    CONTENT "$<TARGET_FILE:CUDA::cublasLt>\n"
)
if(RUSTINFER_CUDA_ENABLE_NVML_PROBE)
    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/rustinfer-cuda-nvml-$<CONFIG>.path"
        CONTENT "$<TARGET_FILE:CUDA::nvml>\n"
    )
endif()

set_target_properties(rustinfer_cuda_native PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS rustinfer_cuda_native
    ARCHIVE DESTINATION lib
)
install(FILES include/rustinfer_cuda.h
    DESTINATION include
)
