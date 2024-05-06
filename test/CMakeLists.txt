add_executable(test_tensor_network test.cpp)

#Print llvm and mlir include directories
message(STATUS "TEST CMAKE:")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

target_include_directories(test_tensor_network PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../TensorNetwork)

llvm_update_compile_flags(test_tensor_network)
llvm_map_components_to_libnames(llvm_libs all)
target_link_libraries(test_tensor_network
 PRIVATE
 ${llvm_libs}
 MLIRIR
 MLIRParser
 MLIRSupport
 LLVMCore
 MLIRFuncDialect
 MLIRBytecodeOpInterface
 MLIRSideEffectInterfaces
 MLIRFunctionInterfaces
 MLIRTransforms
)

target_sources(test_tensor_network PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TensorNetwork/TensorNetworkDialect.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TensorNetwork/TensorNetworkOps.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TensorNetwork/TensorNetworkTypes.cpp)

set_target_properties(test_tensor_network PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(generate_mlir_ir)

