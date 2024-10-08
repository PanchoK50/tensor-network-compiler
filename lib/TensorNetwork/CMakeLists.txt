# Set CMP0148 policy to NEW for modern Python finding modules
cmake_policy(SET CMP0148 NEW)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0 -fno-inline -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-4 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Ensure RTTI and exceptions are enabled globallyset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

# Manually specify the include directory for pybind11
include_directories(/home/francisco/.local/lib/python3.10/site-packages/pybind11/include)

# Find pybind11 package
find_package(pybind11 REQUIRED CONFIG PATHS /home/francisco/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

# Include the Python3 development headers
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

# Add MLIRTensorNetwork library
add_mlir_dialect_library(MLIRTensorNetwork
    TensorNetworkTypes.cpp
    TensorNetworkDialect.cpp
    TensorNetworkOps.cpp
    TensorNetworkPasses.cpp
    TensorNetworkDialectBindings.cpp
    ../mlir/NaiveLowering.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TensorNetwork

    DEPENDS
    MLIRTensorNetworkOpsIncGen
    MLIRTensorNetworkPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRInferTypeOpInterface
    pybind11::pybind11
)

# Ensure RTTI and exceptions are enabled for the MLIRTensorNetwork target
# target_compile_options(MLIRTensorNetwork PRIVATE -frtti -fexceptions)

# Add the pybind11 module
pybind11_add_module(tensor_network_ext TensorNetworkDialectBindings.cpp)
target_include_directories(tensor_network_ext 
    PRIVATE 
    ${LLVM_INCLUDE_DIRS} 
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../mlir
)
target_link_libraries(tensor_network_ext PRIVATE 
    LLVMCore
    MLIRAffineDialect
    MLIRAnalysis
    MLIRArithToLLVM
    MLIRAffineToStandard
    MLIRBufferizationPipelines
    MLIRBytecodeOpInterface
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRFunctionInterfaces
    MLIRIndexToLLVM
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgToStandard
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRParser
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorNetwork
    MLIRTensorToLinalg
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorToLLVMPass
    MLIRExecutionEngine
    MLIRMathToLLVM
    MLIRAsyncDialect
    MLIRAsyncToLLVM
    MLIRAsyncTransforms
    MLIRControlFlowTransforms
    MLIRGPUTransforms
    MLIRConvertToLLVMPass
    MLIRUBToLLVM
    MLIRComplexToLLVM
    MLIRFuncAllExtensions


    pybind11::module
)

# target_compile_options(MLIRTensorNetwork PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
# target_compile_options(tensor_network_ext PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)

target_compile_options(MLIRTensorNetwork PRIVATE -O3 -march=native -DNDEBUG)
target_compile_options(tensor_network_ext PRIVATE -O3 -march=native -DNDEBUG)
