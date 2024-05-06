add_executable(generate_mlir_ir generate.cpp)

target_include_directories(generate_mlir_ir PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../TensorNetwork)

llvm_update_compile_flags(generate_mlir_ir)
llvm_map_components_to_libnames(llvm_libs all)
target_link_libraries(generate_mlir_ir
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

target_sources(generate_mlir_ir PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/TensorNetwork/TensorNetworkDialect.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/TensorNetwork/TensorNetworkOps.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/TensorNetwork/TensorNetworkTypes.cpp)

set_target_properties(generate_mlir_ir PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


