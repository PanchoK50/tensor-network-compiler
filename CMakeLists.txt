## CMakeLists.txt adapted from the MLIR example project "Standalone"


cmake_minimum_required(VERSION 3.20.0)
project(tensor_network_dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#TODO: Remove this, or change to corresponding paths for your system.
# This asumes that the llvm-project was cloned and built in the same directory as this project.
set(MLIR_DIR "../llvm-project/build/lib/cmake/mlir")
set(LLVM_DIR "../llvm-project/build/lib/cmake/llvm")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
else()
    # Build via external projects mechanism
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(TENSORNETWORK_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TENSORNETWORK_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TENSORNETWORK_SOURCE_DIR}/include)
include_directories(${TENSORNETWORK_SOURCE_DIR}/lib/mlir})
include_directories(${TENSORNETWORK_BINARY_DIR}/include)

# add_library(tensor_network_include_dirs INTERFACE)
# target_include_directories(
#     ${LLVM_INCLUDE_DIRS}
#     ${MLIR_INCLUDE_DIRS}
#     ${TENSORNETWORK_SOURCE_DIR}/include
#     ${TENSORNETWORK_BINARY_DIR}/include
# )


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)