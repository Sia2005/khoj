cmake_minimum_required(VERSION 3.24)
project(khoj VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

option(KHOJ_BUILD_TESTS "Build the Catch2 test suite" ON)
option(KHOJ_BUILD_PYTHON "Build the pybind11 extension module" ON)

include(FetchContent)

add_library(khoj_core STATIC
    src/hnsw_index.cpp
)
add_library(khoj::core ALIAS khoj_core)

target_include_directories(khoj_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(khoj_core PUBLIC cxx_std_17)

target_compile_options(khoj_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(KHOJ_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON NEW)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.1.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(khoj_bindings python/bindings.cpp)
    set_target_properties(khoj_bindings PROPERTIES OUTPUT_NAME khoj)
    target_link_libraries(khoj_bindings PRIVATE khoj::core)
endif()

if(KHOJ_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.9.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(khoj_tests tests/test_hnsw_index.cpp)
    target_link_libraries(khoj_tests PRIVATE khoj::core Catch2::Catch2WithMain)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(khoj_tests)
endif()
