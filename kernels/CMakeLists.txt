cmake_minimum_required(VERSION 3.22)

project(rustinfer_cuda_native VERSION 0.1.0 LANGUAGES CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES must contain explicit AOT compile targets")
endif()

message(STATUS
    "rustinfer CUDA AOT architectures: ${CMAKE_CUDA_ARCHITECTURES} (compile targets only)")

add_library(rustinfer_cuda_native STATIC src/version.cu)

target_include_directories(rustinfer_cuda_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(rustinfer_cuda_native PUBLIC cuda_std_17)

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
