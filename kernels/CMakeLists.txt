cmake_minimum_required(VERSION 3.22)

project(rustinfer_cuda_native VERSION 0.1.0 LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES must contain explicit AOT compile targets")
endif()

message(STATUS
    "rustinfer CUDA AOT architectures: ${CMAKE_CUDA_ARCHITECTURES} (compile targets only)")

add_library(rustinfer_cuda_native STATIC
    src/attention_reference.cu
    src/attention_online.cu
    src/decode_attention.cu
    src/gemm.cu
    src/host_runtime.cu
    src/memory.cu
    src/primitives.cu
    src/smoke_fill.cu
    src/version.cu
)

# This target compiles the installed header as C11 and pins every public PR 10
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

# Static archives do not carry transitive linker directives into Cargo. Keep
# the CMake dependencies accurate and emit the toolkit-selected linker files
# for build.rs to validate and forward explicitly.
target_link_libraries(rustinfer_cuda_native PRIVATE
    CUDA::cublasLt
    CUDA::cudart
    CUDA::cuda_driver
)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
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
