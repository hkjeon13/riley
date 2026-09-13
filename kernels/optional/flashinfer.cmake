# Opt-in compile support only; this does not select a serving numerical profile.
set(RILEY_FLASHINFER_DATA "" CACHE PATH "Pinned FlashInfer 0.6.16.post3 data directory")
if(RILEY_FLASHINFER_DATA)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/verify_flashinfer.py" "${RILEY_FLASHINFER_DATA}"
        RESULT_VARIABLE flashinfer_verify_result OUTPUT_VARIABLE flashinfer_digest
        ERROR_VARIABLE flashinfer_verify_error OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT flashinfer_verify_result EQUAL 0)
        message(FATAL_ERROR "FlashInfer dependency verification failed: ${flashinfer_verify_error}")
    endif()
    file(GLOB_RECURSE flashinfer_headers CONFIGURE_DEPENDS
        "${RILEY_FLASHINFER_DATA}/include/*" "${RILEY_FLASHINFER_DATA}/cccl/libcudacxx/include/*"
        "${RILEY_FLASHINFER_DATA}/cccl/cub/*" "${RILEY_FLASHINFER_DATA}/cccl/thrust/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${flashinfer_headers} "${CMAKE_CURRENT_LIST_DIR}/verify_flashinfer.py")
    # FlashInfer uses host exceptions internally. Keep them in this object only;
    # the adapter catches them at its noexcept boundary.
    add_library(riley_flashinfer_object OBJECT "${CMAKE_CURRENT_LIST_DIR}/flashinfer_decode.cu")
    target_include_directories(riley_flashinfer_object PRIVATE
        "${RILEY_FLASHINFER_DATA}/include" "${RILEY_FLASHINFER_DATA}/cccl/libcudacxx/include"
        "${RILEY_FLASHINFER_DATA}/cccl/cub" "${RILEY_FLASHINFER_DATA}/cccl/thrust")
    target_compile_features(riley_flashinfer_object PRIVATE cuda_std_17)
    set_target_properties(riley_flashinfer_object PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(riley_flashinfer_object PRIVATE --objdir-as-tempdir)
    endif()
    target_sources(riley_cuda_native PRIVATE $<TARGET_OBJECTS:riley_flashinfer_object>)
    target_compile_definitions(riley_cuda_native PRIVATE RILEY_CUDA_ENABLE_FLASHINFER=1)
    message(STATUS "FlashInfer compile support enabled; dependency SHA256=${flashinfer_digest}; serving selection unchanged")
endif()
