project(template)
cmake_minimum_required(VERSION 3.22)

include(FindCUDAToolkit)

enable_language(CUDA)

add_subdirectory(third_party/gtest)
add_subdirectory(third_party/gbenchmark)

add_compile_options(-W -Wall -fconcepts)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_FLAGS "--extended-lambda --generate-line-info --expt-relaxed-constexpr")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "--ptxas-options=-v")

add_library(to_bench
  src/baseline.cu
  src/opti1.cu
  src/opti2.cu
  src/cuda_tools/host_shared_ptr.cu)

include_directories(src)

add_executable(bench bench/main.cc)
set_property(TARGET bench PROPERTY CXX_STANDARD 17)
add_executable(test bench/test.cc)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

target_link_libraries(bench to_bench gtest benchmark::benchmark)
target_link_libraries(test to_bench gtest benchmark::benchmark)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()
