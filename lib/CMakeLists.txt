add_subdirectory(TensorNetwork)

# add_llvm_library(TensorNetwork
#   TensorNetwork/TensorNetworkDialect.cpp
#   TensorNetwork/TensorNetworkOps.cpp
#   TensorNetwork/TensorNetworkTypes.cpp
#   TensorNetwork/TensorNetworkPasses.cpp

# )

# add_llvm_library(MLIRTensorNetwork
#   TensorNetwork/TensorNetworkDialect.cpp
#   TensorNetwork/TensorNetworkOps.cpp
#   TensorNetwork/TensorNetworkTypes.cpp
#   TensorNetwork/TensorNetworkPasses.cpp


#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/include/TensorNetwork
# )

# add_dependencies(
#   MLIRTensorNetwork
#   MLIRTensorNetworkOpsIncGen
#   MLIRTensorNetworkPassesIncGen
# )

# target_link_libraries(TensorNetwork MLIRIR)
