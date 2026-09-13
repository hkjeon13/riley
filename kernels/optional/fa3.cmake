# Independent SM90a objects: never add SM90a to the ordinary SM89 kernel targets.
set(RILEY_FA3_SOURCE "" CACHE PATH "Checkout containing pinned FA3 and CUTLASS git objects")
if(RILEY_FA3_SOURCE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(fa3_prepare "${CMAKE_CURRENT_LIST_DIR}/../../benchmarks/analysis/build_fa3_native_adapter.py")
    set(fa3_base_prepare "${CMAKE_CURRENT_LIST_DIR}/../../benchmarks/analysis/build_fa3_native_probe.py")
    set(fa3_overlay "${CMAKE_CURRENT_BINARY_DIR}/fa3-native-overlay-v1")
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${fa3_prepare}" --prepare-only
        "${RILEY_FA3_SOURCE}" "${fa3_overlay}"
        RESULT_VARIABLE fa3_result OUTPUT_VARIABLE fa3_receipt ERROR_VARIABLE fa3_error)
    if(NOT fa3_result EQUAL 0)
        message(FATAL_ERROR "FA3 pinned export verification failed: ${fa3_error}")
    endif()
    file(GLOB_RECURSE fa3_inputs CONFIGURE_DEPENDS "${fa3_overlay}/fa3/*" "${fa3_overlay}/cutlass/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${fa3_inputs} "${fa3_prepare}" "${fa3_base_prepare}" "${fa3_overlay}/overlay.json")
    add_library(riley_fa3_object OBJECT
        "${CMAKE_CURRENT_LIST_DIR}/fa3_adapter.cu"
        "${fa3_overlay}/fa3/hopper/flash_prepare_scheduler.cu")
    target_include_directories(riley_fa3_object PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}" "${fa3_overlay}/fa3/hopper"
        "${fa3_overlay}/cutlass/include" "${fa3_overlay}/cutlass/tools/util/include")
    target_compile_features(riley_fa3_object PRIVATE cuda_std_17)
    target_compile_options(riley_fa3_object PRIVATE
        --expt-relaxed-constexpr --expt-extended-lambda --objdir-as-tempdir)
    set_target_properties(riley_fa3_object PROPERTIES CUDA_ARCHITECTURES "90a-real"
        CUDA_EXTENSIONS OFF CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
    target_sources(riley_cuda_native PRIVATE $<TARGET_OBJECTS:riley_fa3_object>)
    target_sources(riley_cuda_native PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fa3_owner.cu")
    target_sources(riley_cuda_native PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fa3_model_metadata.cu")
    message(STATUS "FA3 native support enabled (SM90a only); serving selection unchanged")
endif()
